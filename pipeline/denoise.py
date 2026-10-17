"""Noise reduction (+ hum removal, + automatic de-reverb) for the repair pipeline.

    denoise(audio, sr, mode="voice", strength=0.8, dereverb=True) -> float32 (channels, samples)

Signal flow (all zero-delay / sample aligned, deterministic):

  1. mains hum / buzz  : detect a stable 50/60 Hz harmonic comb in the quietest segments, remove it with a
                         zero-phase notch comb (only when a comb is actually present).
  2. de-reverb         : WPE (pipeline/dereverb.py), scaled by a decay-rate room detector -> dry audio untouched.
  3. noise gains       : two independent estimators of a per-bin Wiener-type gain on a 2048-pt STFT
                           a. classical: bias-calibrated minimum-statistics noise PSD (+ two-pass noise-only
                              re-estimation, stationarity test and an 8 s percentile cap), two-step
                              decision-directed a-priori SNR, local-SNR mask against musical noise;
                           b. RNNoise (neural, optional): its time-domain output is turned into a gain mask
                              |RNN(x)|/|x| on the same grid.
                         fused as a geometric mean, floored at -25 dB (never gates to digital silence).
                         Without RNNoise (or in "music" mode) the classical gains are used alone.
  4. presence factor   : the gains are raised to  strength * presence  where presence falls from 1 to 0 as the
                         recording's speech-to-floor dynamic range grows from 28 to 42 dB: noise that is already
                         >40 dB below speech is inaudible, and a recording that is clean is left alone.

How the noise floor is found when there is NO speech-free gap: the minimum-statistics tracker only needs
the *spectral valleys* between syllables/harmonics (not whole frames); its bias is calibrated on synthetic
white noise at start-up; the 8 s per-bin 10th-percentile cap bounds it from above; RNNoise needs no gap at all
(it classifies speech vs. noise from the signal itself).  When nothing can be separated, the conservative
direction is taken: the estimate is too high, so less (never more) is removed.

music mode: no RNNoise (speech model), no WPE; the noise floor is the slow per-bin percentile floor with a
low percentile across frequency (valleys between partials), Wiener gain floored at -12 dB without the local
mask, and nothing at all is done unless the music has >= 8 dB of frame dynamic range (otherwise a noise
floor cannot be told from sustained notes and the file is returned untouched - hum removal is gated the same
way because 50/55/60 Hz bass notes look like a mains comb).

strength 0 returns the input unchanged; 0..1 scales the attenuation in dB (G ** strength).
Requirements: numpy, scipy.  Optional: `pip install --no-deps pyrnnoise` (ships rnnoise.dll; located via
importlib without importing pyrnnoise's heavy dependencies; override with env RNNOISE_LIB=<path>).
"""
import ctypes
import functools
import glob
import importlib.util
import os
import threading

import numpy as np
from scipy import ndimage, signal

from ._dsp import istft, stft
from .dfn import dfn_available, dfn_enhance
from .dereverb import dereverb as _dereverb

EPS = 1e-20
_FLOOR_DB = -25.0
RNN_WEIGHT = 0.5                 # weight of the RNNoise gain in the geometric-mean fusion (voice mode)

# ----------------------------------------------------------------------------- classical estimator
_BASE = dict(
    sm_t=7, sm_f=3, min_win_s=0.8, avg_win_s=0.5, thr=3.0, iters=2, hold=5,
    dd=0.95, xi_min_db=_FLOOR_DB, gmin_db=_FLOOR_DB,
    r_hi=0.9, r_lo=0.55, cap_win_s=8.0, cap_m=3.0,
    mt=3, mf=3, r0=2.0, r1=5.0,
)


def _sm(P, mt, mf):
    return np.maximum(ndimage.uniform_filter(P, size=(mt, mf), mode="nearest"), 0.0)


def _pct_floor(S, blk, q=10.0):
    """Per-bin low percentile of S over blocks of `blk` frames (50% overlap), interpolated in time."""
    T, K = S.shape
    if T <= blk:
        return np.broadcast_to(np.percentile(S, q, axis=0), S.shape).copy()
    starts = list(range(0, T - blk + 1, max(blk // 2, 1)))
    if starts[-1] != T - blk:
        starts.append(T - blk)
    cen = np.array([s + blk / 2.0 for s in starts])
    vals = np.stack([np.percentile(S[s:s + blk], q, axis=0) for s in starts])
    t = np.arange(T)
    idx = np.clip(np.searchsorted(cen, t) - 1, 0, len(cen) - 2)
    w = np.clip((t - cen[idx]) / (cen[idx + 1] - cen[idx]), 0, 1)[:, None]
    return vals[idx] * (1 - w) + vals[idx + 1] * w


def _pass2(P, S, lam, thr, iters, hold, ldur, lam0):
    """Re-estimate the noise PSD as the running mean of power over cells classified noise-only."""
    for _ in range(iters):
        sp = S > thr * lam
        if hold > 1:
            sp = ndimage.maximum_filter1d(sp, size=hold, axis=0, mode="nearest")
        sp = ndimage.maximum_filter1d(sp, size=3, axis=1, mode="nearest")
        m = (~sp).astype(np.float64)
        num = np.maximum(ndimage.uniform_filter1d(P * m, ldur, axis=0, mode="nearest"), 0.0)
        den = np.maximum(ndimage.uniform_filter1d(m, ldur, axis=0, mode="nearest"), 0.0)
        est = num / np.maximum(den, 1e-9)
        w = np.clip(den / 0.15, 0, 1)              # confidence: fraction of noise-only cells
        lam = w * est + (1 - w) * lam0
        lam = ndimage.uniform_filter1d(lam, 3, axis=1, mode="nearest")
        lam = np.minimum(np.maximum(lam, 0.0), S)  # noise cannot exceed the (smoothed) observation
    return lam


@functools.lru_cache(maxsize=16)
def _calib(n_fft, hop, mt, mf, W, thr, iters, hold, ldur):
    """Bias of the raw min-tracker / of the two-pass estimate / of the percentile floor on white noise."""
    rng = np.random.default_rng(1234)
    n = hop * (W * 3 + 60)
    x = rng.standard_normal(n)
    P = np.abs(stft(x, n_fft, hop)) ** 2
    true = np.mean(P)
    S = _sm(P, mt, mf)
    smin = ndimage.minimum_filter1d(S, size=W, axis=0, mode="nearest")
    b_min = true / np.mean(smin[W:-W])
    lam = _pass2(P, S, b_min * smin, thr, iters, hold, ldur, b_min * smin)
    c = true / np.mean(lam[W:-W])
    blk = max(int(round(8.0 * 48000 / hop)), 10)
    b10 = true / np.mean(_pct_floor(S, min(blk, len(S) // 2))[W:-W])
    sm = ndimage.uniform_filter1d(S, W, axis=0, mode="nearest")
    rr = (sm / np.maximum(smin, 1e-30))[W:-W]
    r_q = float(np.percentile(rr, 1.0))
    return float(b_min), float(c), float(b10), r_q


def _noise_psd(P, hop, sr, cfg):
    mt, mf = cfg["sm_t"], cfg["sm_f"]
    W = max(int(round(cfg["min_win_s"] * sr / hop)), 5)
    ldur = max(int(round(cfg["avg_win_s"] * sr / hop)), 5)
    S = _sm(P, mt, mf)
    smin = ndimage.minimum_filter1d(S, size=W, axis=0, mode="nearest")
    b_min, c, b10, r_q = _calib(cfg["n_fft"], hop, mt, mf, W, cfg["thr"], cfg["iters"], cfg["hold"], ldur)
    if cfg.get("floor_only"):
        # music: sustained notes look stationary to a min-tracker, so noise is taken to be the *broadband
        # floor*: slow per-bin percentile over ~8 s, then a low percentile across frequency (+-7 bins), i.e.
        # the valleys between partials.  Narrow tonal peaks can never be classified as noise this way.
        blk = max(int(round(cfg["cap_win_s"] * sr / hop)), 10)
        fl = b10 * _pct_floor(S, blk, 10.0)
        return cfg["floor_m"] * ndimage.percentile_filter(fl, cfg["floor_q"], size=(1, 15), mode="nearest")
    lam0 = b_min * smin
    lam = _pass2(P, S, lam0, cfg["thr"], cfg["iters"], cfg["hold"], ldur, lam0) * c
    # stationarity test: where the local level is far above its own running minimum the "noise" estimate is
    # really speech valleys -> do not trust it (this is what keeps clean / reverberant speech intact)
    sm = ndimage.uniform_filter1d(S, W, axis=0, mode="nearest")
    r = sm / np.maximum(smin, 1e-30)
    hi, lo = cfg["r_hi"] * r_q, cfg["r_lo"] * r_q
    w = np.clip((r - lo) / (hi - lo), 0.0, 1.0)
    w = np.where(smin > 0, w, 1.0)               # digital silence: nothing to protect
    lam = lam * w
    # slow upper bound: a few x the 10th-percentile level of each bin over ~8 s blocks
    blk = max(int(round(cfg["cap_win_s"] * sr / hop)), 10)
    lam = np.minimum(lam, cfg["cap_m"] * b10 * _pct_floor(S, blk, 10.0))
    return lam


def _dd(gam, xi_ml, a, xi_min, gmin):
    """Decision-directed a-priori SNR with Wiener gain."""
    T, K = gam.shape
    G = np.empty_like(gam)
    gprev = np.ones(K)
    gamprev = np.ones(K)
    for t in range(T):
        xi = np.maximum(a * (gprev ** 2) * gamprev + (1 - a) * xi_ml[t], xi_min)
        g = np.maximum(xi / (1 + xi), gmin)
        G[t] = g
        gprev = g
        gamprev = gam[t]
    return G


def _classical_gains(P, lam, cfg):
    gam = P / np.maximum(lam, EPS)
    xi_ml = np.maximum(gam - 1.0, 0.0)
    xi_min = 10 ** (cfg["xi_min_db"] / 10)
    gmin = 10 ** (cfg["gmin_db"] / 20)
    G1 = _dd(gam, xi_ml, cfg["dd"], xi_min, gmin)
    xi2 = np.maximum(G1 ** 2 * gam, xi_min)                 # two-step: refine with the first gain
    G = np.maximum(xi2 / (1 + xi2), gmin)
    # local-SNR mask: cells whose smoothed SNR is near 0 dB are pushed to the floor in one coherent patch,
    # which removes the isolated "musical noise" islands a per-bin Wiener gain leaves behind
    zl = ndimage.uniform_filter(gam, size=(cfg["mt"], cfg["mf"]), mode="nearest")
    zdb = 10 * np.log10(np.maximum(zl, 1e-6))
    m = np.clip((zdb - cfg["r0"]) / (cfg["r1"] - cfg["r0"]), 0, 1)
    m = ndimage.uniform_filter(m, size=(3, 3), mode="nearest")
    if not cfg.get("mask", True):
        return G
    return np.exp(m * np.log(np.maximum(G, gmin)) + (1 - m) * np.log(gmin))


def _make_cfg(sr, n_fft, music):
    hop = n_fft // 4
    r = hop * 48000.0 / (sr * 256.0)             # time-resolution scale relative to the tuning point (256 @ 48k)
    c = dict(_BASE)
    c["n_fft"] = n_fft
    c["sm_t"] = max(1, int(round(c["sm_t"] / r)) | 1)
    c["hold"] = max(1, int(round(c["hold"] / r)))
    c["dd"] = c["dd"] ** r
    if music:
        c["xi_min_db"] = c["gmin_db"] = -12.0
        c["mask"] = False
        c["floor_only"] = True
        c["floor_m"] = 1.0
        c["floor_q"] = 20
    return c, hop


# ----------------------------------------------------------------------------------- RNNoise (optional)
_rnn_lib = None          # None = not tried, False = unavailable, CDLL otherwise


_rnn_lock = threading.Lock()


def _load_rnnoise():
    global _rnn_lib
    if _rnn_lib is not None:
        return _rnn_lib
    with _rnn_lock:                      # concurrent first jobs must not see a half-loaded "unavailable"
        if _rnn_lib is None:
            _rnn_lib = _find_rnnoise() or False
    return _rnn_lib


def _find_rnnoise():
    found = None
    cands = []
    env = os.environ.get("RNNOISE_LIB")
    if env:
        cands.append(env)
    try:
        spec = importlib.util.find_spec("pyrnnoise")        # locates the wheel's folder without importing it
        for d in (spec.submodule_search_locations or []) if spec else []:
            for pat in ("*rnnoise*.dll", "*rnnoise*.so*", "*rnnoise*.dylib"):
                cands += sorted(glob.glob(os.path.join(d, pat)))
    except Exception:
        pass
    for path in cands:
        try:
            lib = ctypes.CDLL(path)
            lib.rnnoise_create.restype = ctypes.c_void_p
            lib.rnnoise_create.argtypes = [ctypes.c_void_p]
            lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
            lib.rnnoise_destroy.restype = None
            lib.rnnoise_process_frame.restype = ctypes.c_float
            lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                                  ctypes.POINTER(ctypes.c_float)]
            # smoke test: a short run must produce finite output
            st = lib.rnnoise_create(None)
            if not st:
                continue
            a = np.zeros(480, np.float32)
            b = np.empty(480, np.float32)
            lib.rnnoise_process_frame(st, b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                      a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            lib.rnnoise_destroy(st)
            if np.all(np.isfinite(b)):
                found = lib
                break
        except Exception:
            continue
    return found


_RNN_LAG = 960           # algorithmic delay (samples @48k) of the bundled RNNoise build, measured


def rnnoise_available():
    return bool(_load_rnnoise())


def _rnnoise_channel(x):
    """x float64 (n,) @ 48 kHz -> RNNoise-enhanced signal, sample aligned, level independent."""
    lib = _load_rnnoise()
    n = len(x)
    # RNNoise expects int16-range speech at ordinary level: normalise the loud-speech level, undo afterwards
    fr = int(0.05 * 48000)
    k = n // fr
    g = 1.0
    if k >= 4:
        rms = np.sqrt(np.mean(x[:k * fr].reshape(k, fr) ** 2, axis=1)) + 1e-9
        L = 20 * np.log10(np.percentile(rms, 95))
        g = 10 ** (float(np.clip(-24.0 - L, -20.0, 40.0)) / 20)
    pad = (-n) % 480 + _RNN_LAG
    xp = np.concatenate([x * (g * 32768.0), np.zeros(pad)]).astype(np.float32)
    out = np.empty_like(xp)
    fin = np.empty(480, np.float32)
    fout = np.empty(480, np.float32)
    pin = fin.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pout = fout.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    st = lib.rnnoise_create(None)
    try:
        for i in range(0, len(xp), 480):
            fin[:] = xp[i:i + 480]
            lib.rnnoise_process_frame(st, pout, pin)
            out[i:i + 480] = fout
    finally:
        lib.rnnoise_destroy(st)
    return out[_RNN_LAG:_RNN_LAG + n].astype(np.float64) / (g * 32768.0)


# ------------------------------------------------------------------------------------------- hum
def _quiet_psd(x, sr, nper):
    hop = nper // 2
    n = len(x)
    if n < nper * 2:
        return None, None
    nseg = (n - nper) // hop + 1
    idx = np.arange(nseg)[:, None] * hop + np.arange(nper)[None, :]
    seg = x[idx] * np.hanning(nper)
    pw = np.mean(seg ** 2, axis=1)
    k = max(3, int(round(0.3 * nseg)))
    sel = np.argsort(pw)[:k]
    P = np.mean(np.abs(np.fft.rfft(seg[sel], axis=1)) ** 2, axis=0)
    return np.fft.rfftfreq(nper, 1.0 / sr), P


def detect_hum(x, sr, mode="voice", max_hz=1500.0):
    """(f0, harmonic numbers) of a stable 50/60 Hz mains comb found in the quietest 30% of the file."""
    nper = 1 << int(np.ceil(np.log2(0.7 * sr)))
    f, P = _quiet_psd(x, sr, nper)
    if f is None:
        return None, []
    df = f[1] - f[0]
    thr_db = 9.0 if mode == "voice" else 14.0
    best = (0.0, None, [])
    w = int(round(1.5 / df)) + 1
    for f0n in (50.0, 60.0):
        hs, score = [], 0.0
        for h in range(1, int(max_hz // f0n) + 1):
            fc = h * f0n
            i0 = int(round(fc / df))
            lo = np.r_[int(round((fc - 25) / df)):int(round((fc - 8) / df))]
            hi = np.r_[int(round((fc + 8) / df)):int(round((fc + 25) / df))]
            if lo[0] < 0 or hi[-1] >= len(P) or i0 - w < 1 or i0 + w + 1 >= len(P):
                continue
            pk_i = i0 - w + int(np.argmax(P[i0 - w:i0 + w + 1]))
            pk = P[pk_i]
            floor = np.median(np.r_[P[lo], P[hi]])
            prom = 10 * np.log10((pk + 1e-30) / (floor + 1e-30))
            fl = np.r_[P[int(round((fc - 6) / df)):int(round((fc - 3) / df))],
                       P[int(round((fc + 3) / df)):int(round((fc + 6) / df))]]
            sharp = 10 * np.log10((pk + 1e-30) / (np.median(fl) + 1e-30))
            if prom >= thr_db and sharp >= thr_db * 0.7:
                a, b, c = np.log(P[pk_i - 1] + 1e-30), np.log(pk + 1e-30), np.log(P[pk_i + 1] + 1e-30)
                d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)
                hs.append((h, (pk_i + np.clip(d, -0.5, 0.5)) * df, prom))
                score += prom
        seeded = (mode == "voice" and len(hs) == 1 and hs[0][0] <= 2 and hs[0][2] >= 12.0)   # one strong low line
        if (len(hs) >= 2 or seeded) and score > best[0]:
            best = (score, f0n, hs)
    if best[1] is None:
        return None, []
    hn = np.array([h for h, _, _ in best[2]], float)
    fr = np.array([q for _, q, _ in best[2]], float)
    wt = np.array([p for _, _, p in best[2]], float)
    f0 = float(np.sum(wt * hn * fr) / np.sum(wt * hn * hn))
    keep = [int(h) for h, q in zip(hn, fr) if abs(q - h * f0) < 1.0]
    if len(keep) < 1:
        return None, []

    # The comb is confirmed.  Walk it from the bottom and accept weaker harmonics too (lower prominence bar:
    # they must land exactly on h*f0, which an unrelated partial or speech harmonic will not).  Stop after two
    # consecutive misses, and require the comb to start at the fundamental, 2nd or 3rd harmonic - a "comb" that only
    # shows up at, say, harmonics 11 and 22 is musical partials, not mains hum.
    def line(h):
        fc = h * f0
        i0 = int(round(fc / df))
        lo = np.r_[int(round((fc - 25) / df)):int(round((fc - 8) / df))]
        hi = np.r_[int(round((fc + 8) / df)):int(round((fc + 25) / df))]
        if lo[0] < 0 or hi[-1] >= len(P) or i0 - w < 1 or i0 + w + 1 >= len(P):
            return None
        pk_i = i0 - w + int(np.argmax(P[i0 - w:i0 + w + 1]))
        prom = 10 * np.log10((P[pk_i] + 1e-30) / (np.median(np.r_[P[lo], P[hi]]) + 1e-30))
        return prom, pk_i * df

    lenient = 5.5 if mode == "voice" else 10.0
    full, misses = [], 0
    for h in range(1, int(max_hz // f0) + 1):
        r = line(h)
        if r is not None and r[0] >= lenient and abs(r[1] - h * f0) < 1.0:
            full.append(h)
            misses = 0
        else:
            misses += 1
            if full and misses >= 2:
                break
            if not full and h >= 3:
                break
    comb = sorted(set(keep) | set(full))
    if len(comb) < 2 or comb[0] > 3:
        return None, []
    return f0, comb


def _notch_comb(x, sr, f0, harmonics, bw_hz=3.0):
    sos = []
    for h in harmonics:
        fc = h * f0
        if fc >= 0.45 * sr:
            continue
        b, a = signal.iirnotch(fc, Q=max(fc / bw_hz, 4.0), fs=sr)
        sos.append(signal.tf2sos(b, a))
    if not sos:
        return x
    return signal.sosfiltfilt(np.vstack(sos), x, axis=-1)


# ------------------------------------------------------------------------------------ public entry
def _frame_dyn(mono, sr):
    """Dynamic range (dB) between loud and quiet 50 ms frames: p90 - p10 of the frame RMS."""
    n = int(0.05 * sr)
    k = len(mono) // n
    if k < 8:
        return None
    rms = 20 * np.log10(np.sqrt(np.mean(mono[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-9)
    return float(np.percentile(rms, 90) - np.percentile(rms, 10))


def _presence(mono, sr, hi=42.0, lo=28.0):
    """Speech: 1 when the floor is audible, 0 when it is >= `hi` dB under the speech (nothing to remove)."""
    dyn = _frame_dyn(mono, sr)
    return 1.0 if dyn is None else float(np.clip((hi - dyn) / (hi - lo), 0.0, 1.0))


def _music_presence(mono, sr, lo=8.0, hi=16.0):
    """Music: a noise floor can only be told from sustained notes when the music actually gets quiet.
    No measurable dynamic range -> 0 (leave the file alone); >= `hi` dB -> 1."""
    dyn = _frame_dyn(mono, sr)
    return 0.0 if dyn is None else float(np.clip((dyn - lo) / (hi - lo), 0.0, 1.0))


MAX_SPEECH_DROP_DB = 3.0     # removing noise must not lower the loud (speech) frames by more than this


def _do_no_harm(before, after, sr):
    """Sanity check on the gain mask.  Real noise removal leaves the speech level almost untouched; when
    the loud frames drop by much more than MAX_SPEECH_DROP_DB the mask has eaten the wanted voice
    (typically a competing talker / babble it cannot tell apart), so blend part of the original back."""
    n = int(0.05 * sr)
    m0, m1 = before.mean(axis=0), after.mean(axis=0)
    k = len(m0) // n
    if k < 8:
        return after
    f = lambda m: np.percentile(np.sqrt(np.mean(m[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-12, 90)
    drop = 20 * np.log10(f(m0) / f(m1))
    if drop <= MAX_SPEECH_DROP_DB:
        return after
    w = float(np.clip(1.0 - (drop - MAX_SPEECH_DROP_DB) / 5.0, 0.15, 1.0))
    return before + w * (after - before)


def denoise(audio, sr, mode="voice", strength=0.8, dereverb=True):
    """Reduce background noise / hum / room reverb.  strength 0..1 (0 = untouched).

    audio: float array (channels, samples) (a 1-D array is accepted).  Returns float32, same shape.
    dereverb: apply the automatic (room-detector scaled) WPE stage; pass False if the caller runs
    pipeline.dereverb.dereverb itself.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return audio
    a = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    squeeze = a.ndim == 1
    if squeeze:
        a = a[None]
    n = a.shape[1]
    if n < int(0.5 * sr) or not np.any(a):
        return audio
    music = mode != "voice"
    mono = a.mean(axis=0)

    pres = _music_presence(mono, sr) if music else _presence(mono, sr)

    # 1. hum  (music: only when the music gets quiet enough to see a floor - sustained bass notes at 50/55/
    #    60 Hz are otherwise indistinguishable from a mains comb)
    f0, hs = detect_hum(mono, sr, "music" if music else "voice") if (not music or pres > 0.5) else (None, [])
    if f0 is not None:
        notched = np.stack([_notch_comb(c, sr, f0, hs) for c in a])
        k = min(1.0, strength / 0.8)
        a = a + k * (notched - a)

    # 2. de-reverb (voice only)
    if dereverb and not music:
        a = _dereverb(a, sr, strength=min(1.0, strength / 0.8), auto=True).astype(np.float64)

    # 3. gains
    expo = strength * pres
    neural = None
    if expo >= 1e-3 and not music and sr == 48000 and dfn_available():
        try:
            neural = dfn_enhance(a, sr).astype(np.float64)
        except Exception:
            neural = None                                        # fall back to the built-in estimator
    if expo < 1e-3:
        out = a                                                  # nothing audible / measurable to remove
    elif neural is not None:
        k = min(1.0, expo / 0.8)                                 # same strength / audibility scaling as the mask path
        out = _do_no_harm(a, a + k * (neural - a), sr)
    else:
        n_fft = 2048 if sr >= 32000 and n >= 3 * sr else 1024
        cfg, hop = _make_cfg(sr, n_fft, music)
        Y = np.stack([stft(c, n_fft, hop) for c in a])
        P = np.mean(np.abs(Y) ** 2, axis=0)
        lam = _noise_psd(P, hop, sr, cfg)
        Gc = _classical_gains(P, lam, cfg)
        use_rnn = (not music) and sr == 48000 and rnnoise_available()
        gfl = 10 ** (_FLOOR_DB / 20)
        outs = []
        for i in range(a.shape[0]):
            G = Gc
            if use_rnn:
                try:
                    R = stft(_rnnoise_channel(a[i]), n_fft, hop)
                    Gr = np.clip(np.abs(R) / (np.abs(Y[i]) + 1e-12), gfl, 1.0)
                    G = Gc ** (1.0 - RNN_WEIGHT) * Gr ** RNN_WEIGHT
                except Exception:
                    G = Gc
            if expo < 1.0:
                G = G ** expo
            outs.append(istft(Y[i] * G, n, n_fft, hop))
        out = np.stack(outs)
        out = _do_no_harm(a, out, sr)

    out = np.nan_to_num(out)
    out = out[0] if squeeze else out
    return out.astype(np.float32)
