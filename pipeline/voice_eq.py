"""Adaptive tone correction for bad microphones.

voice_eq(audio, sr, mode) -> audio   (float32, shape (channels, samples), sample aligned)

Stages (all offline, zero-phase, deterministic, no latency):
  0. hum        : mains-hum comb (50/60 Hz + harmonics) found in the stationary spectrum and
                  removed with narrow zero-phase notches.
  1. analysis   : speech-activity detection, noise floor, long-term average speech spectrum
                  (LTAS) in 1/3-octave bands, noise-subtracted.
  2. tone curve : smooth target-matching EQ (dead-band, +-9 dB, SNR-aware) + local resonance cut
                  + adaptive low-cut, applied as one linear-phase FIR.
  3. bandwidth  : conservative extension for telephone-like (band-limited) input.
  4. de-esser   : level-adaptive split-band sibilance control (only the hot outliers).
  5. leveler    : slow, speech-gated gain rider (never lifts the noise floor).
"""
import numpy as np
from scipy.signal import (stft, oaconvolve, firwin, iirnotch, tf2sos, sosfiltfilt, butter)
from scipy.ndimage import uniform_filter1d

P = dict(
    nfft=4096, hop=1024,
    stat="db",                    # 'db' (mean of band dB) or 'pow' (power mean) LTAS statistic
    strength=0.85, ramp=3.0, res_strength=0.7, res_dead=2.5, smooth_bands=2,
    bwe=True, bwe_fine=False, bwe_overlap=0.0, bwe_kappa=0.5, bwe_on_lo=15.0, bwe_on_hi=30.0, bwe_fmax=8000.0,
    bwe_edge_max=5200.0, bwe_tilt=0.8, bwe_low=-0.35, bwe_gate=True, bwe_dg_lo=-20.0, bwe_dg_hi=9.0,
    hpf_min=2.0, hpf_max=100.0, hpf_hz=None,
    hum=True, hum_prom_db=10.0, hum_bw=3.0, hum_kmax=12,
    deess=True, deess_lo=4500.0, deess_hi=9500.0, deess_thr=-4.0, deess_ratio=2.5, deess_max=6.0,
    level=True, level_alpha=0.4, level_up=3.0, level_down=6.0, level_dead=3.0, level_win=0.4,
)

_CENT = 1000.0 * 2.0 ** (np.arange(-12, 14) / 3.0)          # 62 Hz .. 20 kHz, 26 bands
_LO = _CENT / 2 ** (1 / 6)
_HI = _CENT * 2 ** (1 / 6)
_LOGC = np.log2(_CENT)


def _lim(kf, kv):
    return np.interp(_LOGC, np.log2(kf), kv)


# Target long-term speech spectrum, dB per 1/3-octave band relative to the mean of the
# 500 Hz - 2.5 kHz bands (power-mean statistic; knots in log frequency).
_TK_F = np.array([50, 80, 125, 200, 315, 500, 800, 1250, 2000, 3150, 5000, 8000, 12000, 20000], float)
_TK_G = np.array([-24, -12, -4, 2.0, 4.0, 3.0, 0.5, -2.5, -5.0, -5.5, -7.5, -12.0, -20.0, -40.0], float)
# offset converting the power-mean target into the mean-of-dB target (HF is far more variable)
_OF_F = np.array([100, 250, 500, 800, 1250, 2000, 3150, 4000, 5000, 6300, 8000], float)
_OF_G = np.array([3.0, 2.5, 1.5, 0.0, 0.5, 1.5, 1.5, -1.0, -4.0, -6.0, -6.0], float)

_MAXB = _lim([62, 125, 200, 315, 500, 800, 5000, 6400, 8000, 10000, 16000], [0, 0, 3, 5, 6, 8, 9, 6, 3, 2, 0])
_MAXC = _lim([62, 125, 200, 315, 500, 800, 6400, 8000, 16000], [12, 9, 9, 9, 9, 9, 9, 9, 9])
_DEAD = _lim([62, 125, 250, 400, 4000, 5000, 6300, 8000, 10000], [5.0, 5.0, 4.0, 3.0, 3.0, 4.0, 6.0, 8.0, 9.0])


def _target(stat):
    T = np.interp(_LOGC, np.log2(_TK_F), _TK_G)
    if stat == "db":
        T = T + np.interp(_LOGC, np.log2(_OF_F), _OF_G)
    return T


def _band_matrix(x, sr):
    n, h = P["nfft"], P["hop"]
    f, _, S = stft(x, sr, nperseg=n, noverlap=n - h, boundary=None, padded=False)
    pw = np.abs(S) ** 2
    B = np.stack([pw[(f >= l) & (f < u)].sum(0) for l, u in zip(_LO, _HI)])
    return f, B


# ----------------------------------------------------------------------------- analysis
def analyse(x, sr):
    f, B = _band_matrix(x, sr)
    sel = (_CENT >= 150) & (_CENT <= 4000)
    e = 10 * np.log10(B[sel].sum(0) + 1e-20)
    nz = np.percentile(e, 10)
    sp = np.percentile(e, 95)
    act = e > max(sp - 25.0, nz + 9.0)
    if act.sum() < 5:
        act = e > np.percentile(e, 30)
    if not act.any():                                       # (near-)constant level
        act = np.ones_like(act)
    quiet = e <= np.percentile(e, 15)
    Pa = B[:, act].mean(1)
    Pn = B[:, quiet].mean(1) if (~act).sum() > 3 else np.zeros_like(Pa)
    Bs = np.maximum(B[:, act] - Pn[:, None], 0.1 * B[:, act])
    Ls = (10 * np.log10(Bs + 1e-20)).mean(1)               # noise-subtracted mean-of-dB LTAS
    return dict(B=B, e=e, actmask=act, Pa=Pa, Pn=Pn, Ls=Ls, act=float(act.mean()), nz=nz, sp=sp)


def choose_hpf(an, sr):
    """Low-cut corner (Hz): always remove DC / infrasound; raise it only where the pauses
    show real rumble (low-frequency energy that is not speech)."""
    if P["hpf_hz"]:
        return float(P["hpf_hz"])
    Pn, Pa = an["Pn"], an["Pa"]
    lf = (_CENT >= 40) & (_CENT <= 100)
    core = (_CENT >= 500) & (_CENT <= 2500)
    core_p = max(Pa[core].mean() - Pn[core].mean(), 1e-20)
    rum = 10 * np.log10((Pn[lf].sum() + 1e-20) / core_p)          # dB re speech core band
    fc = P["hpf_min"] + np.clip((rum + 38.0) * 3.5, 0.0, P["hpf_max"] - P["hpf_min"])
    return float(fc), float(rum)


def _smooth_bands(v, k):
    if k <= 0:
        return v
    w = np.concatenate([np.arange(1, k + 2), np.arange(k, 0, -1)]).astype(float)
    w /= w.sum()
    pad = np.pad(v, k, mode="edge")
    return np.convolve(pad, w, mode="valid")


def _firm(d, t1, t2, strength):
    """Firm thresholding: no correction below t1, smooth ramp to (almost) full correction at t2."""
    a = np.abs(d)
    r = np.clip((a - t1) / np.maximum(t2 - t1, 1e-6), 0.0, 1.0)
    return np.sign(d) * a * (r * r * (3.0 - 2.0 * r)) * strength


def gain_curve(an, sr):
    stat = P["stat"]
    Pa, Pn = an["Pa"], an["Pn"]
    Ps = np.maximum(Pa - Pn, Pa * 0.01) + 1e-20           # power-mean speech LTAS
    core = (_CENT >= 500) & (_CENT <= 2500)
    if stat == "db":
        Lraw = an["Ls"]
        ref = Lraw[core].mean()
    else:
        Lraw = 10 * np.log10(Ps)
        ref = 10 * np.log10(Ps[core].mean())
    L = Lraw - ref
    T = _target(stat)
    d = T - L                                              # >0: speech is below target
    r = (_CENT >= 200) & (_CENT <= 5000)
    d = d - np.median(d[r])
    snr = 10 * np.log10(Ps / (Pn + 1e-20))
    ds = _smooth_bands(np.clip(d, -30.0, _MAXB + 6.0), int(P["smooth_bands"]))   # broad component
    gA = _firm(ds, _DEAD, _DEAD + P["ramp"], P["strength"])
    rr = d - ds                                            # local structure (resonances / dips)
    mid = (_CENT >= 500) & (_CENT <= 5000)
    gB = np.where(mid, _firm(rr, P["res_dead"], P["res_dead"] + 2.5, P["res_strength"]), 0.0)
    gB = np.clip(gB, -4.0, 2.0)
    g = gA + gB
    boost_ok = (d <= _MAXB + 9.0) & (snr > 6.0)
    g = np.where(g > 0, np.where(boost_ok, np.minimum(g, np.maximum(snr - 3.0, 0.0)), 0.0), g)
    g = np.clip(g, -_MAXC, _MAXB)
    g[_CENT > 0.45 * sr] = 0.0
    info = dict(L=L, T=T, d=d, snr=snr, ref=ref, Ps=Ps, gA=gA, gB=gB)
    return g, info


# ----------------------------------------------------------------------------- filters
def _fir_from_response(H, nfft):
    """Symmetric (zero-phase), odd-length (nfft+1) FIR from a real response on rfftfreq(nfft)."""
    h = np.fft.fftshift(np.fft.irfft(H, nfft))              # delta at index nfft/2
    h = np.concatenate([h, h[:1]])                          # make it symmetric about nfft/2
    return h * np.hanning(nfft + 1)


def _zero_phase_fir(g_db, sr, f_hp, nfft=16384):
    f = np.fft.rfftfreq(nfft, 1 / sr)
    gl = np.interp(np.log2(np.maximum(f, 20.0)), _LOGC, g_db)
    H = 10 ** (gl / 20)
    if f_hp:
        H = H / np.sqrt(1 + (f_hp / np.maximum(f, 1.0)) ** 8)
    return _fir_from_response(H, nfft)


def _lin_gain(q, f):
    return np.interp(np.log2(np.maximum(f, 20.0)), _LOGC, q)


def _zero_phase_fir_lin(q, sr, nfft=16384):
    f = np.fft.rfftfreq(nfft, 1 / sr)
    return _fir_from_response(_lin_gain(q, f), nfft)


# ----------------------------------------------------------------------------- bandwidth extension
def _ssb_shift(x, sr, f1, f2, shift):
    """Translate the band [f1, f2] of x up by `shift` Hz (analytic single sideband), same length."""
    ntap = 2047
    p = firwin(ntap, (f2 - f1) / 2.0, fs=sr, window=("kaiser", 8.0))
    n = np.arange(ntap) - ntap // 2
    ha = p * np.exp(2j * np.pi * ((f1 + f2) / 2.0) * n / sr)
    z = oaconvolve(x, ha, mode="same")
    t = np.arange(len(x))
    return 2.0 * np.real(z * np.exp(2j * np.pi * shift * t / sr))


def _plan_bwe(an, info, g, sr):
    Pobs = info["Ps"] * 10 ** (g / 10)
    ref_p = 10 * np.log10(info["Ps"][(_CENT >= 500) & (_CENT <= 2500)].mean())
    PT = 10 ** ((np.interp(_LOGC, np.log2(_TK_F), _TK_G) + ref_p) / 10)
    deficit = 10 * np.log10(PT / Pobs)
    fmax = min(P["bwe_fmax"], 0.45 * sr)
    cand = (_CENT >= 2500) & (_CENT <= fmax * 1.02)
    idx = np.flatnonzero(cand)

    def first_block(mask):
        for j in idx:
            if np.all(mask[j:][cand[j:]]):
                return j
        return None

    pe = first_block(info["d"] >= P["bwe_on_lo"])        # where the usable band ends (pre-EQ)
    if pe is None or _CENT[pe] > P["bwe_edge_max"]:
        return None
    edge = first_block(deficit >= P["bwe_on_lo"])        # what the EQ could not repair
    if edge is None:
        return None
    a = np.clip((deficit - P["bwe_on_lo"]) / (P["bwe_on_hi"] - P["bwe_on_lo"]), 0, 1)
    a = np.where(cand & (np.arange(len(_CENT)) >= edge), a, 0.0)
    S = P["bwe_kappa"] * np.maximum(PT - Pobs, 0.0) * a
    return dict(S=S, deficit=deficit, edge=edge, f_edge=_LO[min(pe, edge)])


def _fit_fill_filter(hf, sr, S, actmask, iters=3):
    n, h = P["nfft"], P["hop"]
    f, _, Z = stft(hf, sr, nperseg=n, noverlap=n - h, boundary=None, padded=False)
    pw = np.abs(Z) ** 2
    m = actmask[: pw.shape[1]] if actmask.shape[0] >= pw.shape[1] else np.ones(pw.shape[1], bool)
    Pbin = pw[:, m].mean(1)
    bidx = [(f >= l) & (f < u) for l, u in zip(_LO, _HI)]
    on = S > 0
    Ph = np.array([Pbin[b].sum() for b in bidx]) + 1e-20
    q = np.zeros(len(_CENT))
    q[on] = np.sqrt(S[on] / Ph[on])
    for _ in range(iters):
        H = _lin_gain(q, f)
        real = np.array([(Pbin[b] * H[b] ** 2).sum() for b in bidx]) + 1e-20
        q = np.where(on, q * np.clip(np.sqrt(np.where(on, S, 1) / real), 0.25, 4.0), 0.0)
    return np.minimum(q, 10 ** (30 / 20))


def _bwe_envelope(src, sr, an, f1, f2):
    """Per-sample gain: speech presence (Wiener-like) x spectral-tilt / low-band dependent level."""
    n = len(src)
    nper, hop = 1024, 256
    fr, _, Z = stft(src, sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    pw = np.abs(Z) ** 2
    fm = (f1 + f2) / 2
    A = pw[(fr >= f1) & (fr < fm)].sum(0) + 1e-20
    Bq = pw[(fr >= fm) & (fr < f2)].sum(0) + 1e-20
    low = pw[(fr >= 300) & (fr < min(1500.0, f1))].sum(0) + 1e-20
    tot = A + Bq
    tcent = np.arange(Z.shape[1]) * hop + nper // 2
    Pn_src = an["Pn"][(_CENT >= f1) & (_CENT < f2)].sum() * (nper / P["nfft"])
    w = np.clip(1.0 - 1.5 * Pn_src / tot, 0.0, 1.0) if P["bwe_gate"] else np.ones_like(tot)
    tilt = 10 * np.log10(Bq / A)
    lowmid = 10 * np.log10(low / tot)
    act = w > 0.5
    if act.sum() > 5:
        wt = tot[act]
        tref = np.average(tilt[act], weights=wt)
        lref = np.average(lowmid[act], weights=wt)
        dg = P["bwe_tilt"] * (tilt - tref) + P["bwe_low"] * (lowmid - lref)
        dg = np.clip(dg, P["bwe_dg_lo"], P["bwe_dg_hi"])
    else:
        dg = np.zeros_like(tilt)
    gt = w * 10 ** (dg / 20)
    gt = np.convolve(np.pad(gt, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return np.interp(np.arange(n), tcent, gt)


def _bwe(out, sr, an, plan):
    S = plan["S"]
    f2 = plan["f_edge"]
    W = float(np.clip(f2 - 1200.0, 800.0, 2000.0))
    f1 = f2 - W
    top = min(P["bwe_fmax"], 0.45 * sr)
    K = int(min(4, np.ceil((top - f2) / W)))
    patches = [W * (k + 1) for k in range(K)]
    for c in range(out.shape[0]):
        src = out[c]
        hf = np.zeros_like(src)
        for s in patches:
            hf += _ssb_shift(src, sr, f1 - P["bwe_overlap"], f2 + P["bwe_overlap"], s)
        hf *= _bwe_envelope(src, sr, an, f1, f2)
        if P["bwe_fine"]:
            fir = _fit_fill_fine(hf, sr, S, an["actmask"])
        else:
            fir = _zero_phase_fir_lin(_fit_fill_filter(hf, sr, S, an["actmask"]), sr)
        out[c] = src + oaconvolve(hf, fir, mode="same")
    return out


def _fit_fill_fine(hf, sr, S, actmask, nfft=16384):
    """Fine-resolution fill filter. The band targets S (power per 1/3-octave band) are turned
    into a smooth per-bin power density (log-linear in frequency), and the filter flattens the
    long-term spectrum of the patched signal onto it. Unlike per-band gains this cannot leave
    holes at patch junctions or steps at band edges, and it follows the smooth target."""
    n, h = P["nfft"], P["hop"]
    f, _, Z = stft(hf, sr, nperseg=n, noverlap=n - h, boundary=None, padded=False)
    pw = np.abs(Z) ** 2
    m = actmask[: pw.shape[1]] if actmask.shape[0] >= pw.shape[1] else np.ones(pw.shape[1], bool)
    Pbin = pw[:, m].mean(1) + 1e-30
    on = np.flatnonzero(S > 0)
    nb = np.array([((f >= l) & (f < u)).sum() for l, u in zip(_LO, _HI)], float)
    D = S[on] / np.maximum(nb[on], 1.0)
    lf = np.log2(np.maximum(f, 1.0))
    T = np.exp(np.interp(lf, _LOGC[on], np.log(D)))
    lo = _LO[on[0]]
    T = T * np.clip((f - lo / 2 ** (1 / 6)) / (lo - lo / 2 ** (1 / 6)), 0.0, 1.0) * (f <= _HI[on[-1]])
    k = max(int(round(100.0 / (f[1] - f[0]))) | 1, 3)
    Ps = uniform_filter1d(Pbin, k, mode="nearest")
    H = np.minimum(np.sqrt(T / Ps), 10 ** (24 / 20))
    f2 = np.fft.rfftfreq(nfft, 1 / sr)
    return _fir_from_response(np.interp(f2, f, H), nfft)


# ----------------------------------------------------------------------------- mains hum
def detect_hum(x, sr, mode="voice"):
    """Find a mains-hum comb (50/60 Hz + harmonics) in the stationary part of the spectrum.
    A low percentile over time per FFT bin (minimum-statistics style) keeps speech from hiding
    or faking lines. Returns [(frequency_hz, prominence_db), ...] or []."""
    n = 16384
    if len(x) < 2 * n:
        return []
    f, _, S = stft(x, sr, nperseg=n, noverlap=n * 3 // 4, boundary=None, padded=False)
    N = np.percentile(np.abs(S) ** 2, 20, axis=1) + 1e-20
    df = f[1] - f[0]
    thr = P["hum_prom_db"] + (10.0 if mode == "music" else 0.0)
    best, best_score = [], 0.0
    for f0 in (50.0, 60.0):
        lines, score = [], 0.0
        for k in range(1, int(P["hum_kmax"]) + 1):
            fk = f0 * k
            if fk > 1000.0 or fk > 0.4 * sr:
                break
            c = int(round(fk / df))
            cand = np.flatnonzero(np.abs(f - fk) <= 1.5)       # mains drift is far below 1 Hz
            if len(cand) == 0:
                continue
            j = int(cand[np.argmax(N[cand])])
            ref = np.r_[N[max(j - 40, 0):max(j - 6, 0)], N[j + 7:j + 41]]
            if len(ref) < 10:
                continue
            prom = 10 * np.log10(N[j] / np.median(ref))
            if prom >= 0.7 * thr:
                a, b, c2 = np.log(N[j - 1]), np.log(N[j]), np.log(N[j + 1])
                den = a - 2 * b + c2
                d = 0.5 * (a - c2) / den if den != 0 else 0.0
                fl = float(f[j] + np.clip(d, -1, 1) * df)
                if abs(fl - fk) <= 1.2:                         # a line must sit ON the mains comb
                    lines.append((fl, float(prom), k))
                    score += prom
        # a real mains comb has a clearly prominent fundamental or 2nd harmonic (a lone line at
        # 150 Hz can be a monotone synthetic voice, so higher orders never qualify alone)
        if any(p >= thr and k <= 2 for _, p, k in lines) and score > best_score:
            best, best_score = lines, score
    return best


def _apply_notches(audio, sr, lines):
    out = audio
    for fl, _, _ in lines:
        b, a = iirnotch(fl, max(fl / P["hum_bw"], 5.0), fs=sr)
        out = sosfiltfilt(tf2sos(b, a), out, axis=-1)
    return out


# ----------------------------------------------------------------------------- de-esser
def _env_db(s, sr, win_s):
    w = max(int(sr * win_s) | 1, 3)
    return 10 * np.log10(uniform_filter1d(s * s, w, mode="nearest") + 1e-14)


def deess_gain(mono, sr):
    """Per-sample sibilance gain reduction in dB (>= 0) from a mono mix.
    The trigger level is relative to the clip's own speech body level (P95 of the 150 Hz-4.5 kHz
    envelope), so it adapts to recording level; only speech-active frames can trigger it."""
    hi = min(P["deess_hi"], 0.45 * sr)
    s = sosfiltfilt(butter(2, [P["deess_lo"], hi], "bp", fs=sr, output="sos"), mono)
    body = sosfiltfilt(butter(2, [150.0, P["deess_lo"]], "bp", fs=sr, output="sos"), mono)
    e_s = _env_db(s, sr, 0.005)
    e_b = _env_db(body, sr, 0.020)
    ref = np.percentile(e_b, 95)
    active = np.clip((e_b - (ref - 30.0)) / 10.0, 0.0, 1.0)
    over = np.maximum(e_s - (ref + P["deess_thr"]), 0.0)
    gr = np.minimum((1.0 - 1.0 / P["deess_ratio"]) * over, P["deess_max"]) * active
    w = max(int(sr * 0.004) | 1, 3)
    return uniform_filter1d(uniform_filter1d(gr, w, mode="nearest"), w, mode="nearest")


def deess(out, sr):
    mono = out.mean(0)
    gr = deess_gain(mono, sr)
    if gr.max() < 0.05:
        return out, 0.0
    hi = min(P["deess_hi"], 0.45 * sr)
    sos = butter(2, [P["deess_lo"], hi], "bp", fs=sr, output="sos")
    gl = 10 ** (-gr / 20.0)
    res = np.empty_like(out)
    for c in range(out.shape[0]):
        s = sosfiltfilt(sos, out[c])
        res[c] = out[c] - s + s * gl                 # only the sibilance band is attenuated
    return res, float(gr.max())


# ----------------------------------------------------------------------------- leveler
def level_gain(mono, sr):
    """Per-sample leveler gain in dB from a mono mix: slow speech-gated rider that evens out
    phrase-to-phrase level swings (mic distance, head turns) without touching syllable
    dynamics. Computed from speech-active loudness only, held through pauses, smoothed over
    ~0.5 s, and boosts are capped by local SNR so the noise floor is never lifted."""
    y = sosfiltfilt(butter(2, [100.0, 5000.0], "bp", fs=sr, output="sos"), mono)
    hop = int(sr * 0.02)
    nfr = len(y) // hop
    if nfr < 40:
        return None
    fr = (y[: nfr * hop] ** 2).reshape(nfr, hop).mean(1) + 1e-14
    inst = 10 * np.log10(fr)
    k = max(int(round(P["level_win"] / 0.02)) | 1, 3)
    lst = 10 * np.log10(uniform_filter1d(fr, k, mode="nearest"))      # short-term level, dB
    nz = np.percentile(inst, 10)
    sp = np.percentile(lst, 95)
    act = lst > max(sp - 25.0, nz + 12.0)
    if act.sum() < 20:
        return None
    tgt = np.percentile(lst[act], 60)
    d = tgt - lst
    d = np.sign(d) * np.maximum(np.abs(d) - P["level_dead"], 0.0) * P["level_alpha"]
    up_cap = np.clip((lst - nz - 18.0) / 3.0, 0.0, P["level_up"])
    g = np.clip(d, -P["level_down"], up_cap)
    idx = np.arange(nfr)
    g = np.interp(idx, idx[act], g[act])                  # hold nearest active value in pauses
    g = np.where(act, g, np.minimum(g, 0.0))              # ... but never boost a pause (noise floor)
    ks = max(int(round(0.5 / 0.02)) | 1, 3)
    g = uniform_filter1d(uniform_filter1d(g, ks, mode="nearest"), ks, mode="nearest")
    return np.interp(np.arange(len(mono)), (idx + 0.5) * hop, g)


# ----------------------------------------------------------------------------- driver
def _process(audio, sr, mode="voice", level=True, bwe=True):
    audio = np.nan_to_num(np.asarray(audio, np.float32).astype(np.float64))
    if audio.ndim == 1:
        audio = audio[None, :]
    mono = audio.mean(0)
    plan = {}
    music = (mode == "music")
    # too short / silent / constant: nothing reliable to analyse -> pass through untouched
    if audio.shape[-1] < 4 * P["nfft"] or np.std(mono) < 1e-6:
        plan["skipped"] = True
        return audio.astype(np.float32), plan
    # 0. mains hum: narrow zero-phase notches, before analysis so it neither drives the
    #    low-cut up nor gets treated as speech
    if P["hum"]:
        lines = detect_hum(mono, sr, mode)
        plan["hum"] = lines
        if lines:
            audio = _apply_notches(audio, sr, lines)
            mono = audio.mean(0)
    # 1. tone curve + adaptive low-cut (one zero-phase FIR)
    an = analyse(mono, sr)
    g, info = gain_curve(an, sr)
    fc = choose_hpf(an, sr)
    fc = fc[0] if isinstance(fc, tuple) else fc
    if music:
        # the target curve describes SPEECH; music gets only rumble/DC removal (+ hum notches)
        g = np.zeros_like(g)
        fc = min(fc, 40.0)
    h = _zero_phase_fir(g, sr, fc)
    out = np.stack([oaconvolve(ch, h, mode="same") for ch in audio])
    plan.update(g=g, info=info, an=an, hpf=fc)
    # 2. conservative bandwidth extension for band-limited input (speech only)
    if P["bwe"] and bwe and not music:
        bp = _plan_bwe(an, info, g, sr)
        plan["bwe"] = bp
        if bp is not None and bp["S"].max() > 0:
            out = _bwe(out, sr, an, bp)
    if mode == "voice":
        # 3. level-adaptive de-esser (split-band, outliers only)
        if P["deess"]:
            out, plan["deess_max_db"] = deess(out, sr)
        # 4. slow speech-gated leveler; one gain curve shared by all channels
        if P["level"] and level:
            gl = level_gain(out.mean(0), sr)
            if gl is not None:
                out = out * 10 ** (gl / 20.0)
                plan["level_max_db"] = float(np.abs(gl).max())
    out = np.nan_to_num(out)
    pk = float(np.abs(out).max())
    if pk > 0.98:                       # one static trim instead of letting the host hard-clip
        out = out * (0.98 / pk)
        plan["trim_db"] = float(20 * np.log10(0.98 / pk))
    return out.astype(np.float32), plan


def voice_eq(audio, sr, mode="voice"):
    """Adaptive tone correction. audio: float32 (channels, samples) -> same shape."""
    return _process(audio, sr, mode)[0]


def voice_eq_with_plan(audio, sr, mode="voice", level=True, bwe=True):
    """Like voice_eq, but also returns the plan dict (what was detected / applied) for reporting.
    level=False skips the slow leveler (used when the pipeline levels elsewhere).
    bwe=False disables bandwidth extension (used when the recording has no steady noise floor to judge it by)."""
    return _process(audio, sr, mode, level, bwe)
