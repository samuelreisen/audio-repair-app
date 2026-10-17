"""Loudness mastering: gated-LUFS normalisation + transparent true-peak limiting.

Public API (unchanged):
    master(audio, sr, mode="voice") -> float32 array, same shape as `audio`
    measure_lufs(audio, sr)         -> float (integrated, gated LUFS; -inf if silent)

Why this replaces the pedalboard version
----------------------------------------
pedalboard.Limiter adds ~+5 dB make-up gain, so loudness landed at -14.9 LUFS
instead of -16 and peaks reached -0.5 dBTP.  Everything here is our own, measured
end-to-end, and nothing is trusted: the *final* signal is re-measured.

Signal flow (all float64 internally, offline, zero latency / sample-aligned)
----------------------------------------------------------------------------
1. Sanitise: NaN/inf -> 0, remove DC (it only wastes headroom).
2. Leveler (voice only, <= 2 dB): slow, gated upward gain on quiet words.  Never
   lifts gaps/room tone: it fades out near the estimated noise floor, for very
   quiet passages (breaths), and switches itself off if speech-to-floor is poor.
3. Loudness loop: gain -> peak control -> re-measure gated LUFS (secant / bracket,
   <= 8 passes) so the *post-limiter* file lands on target (-16 voice, -14 music).
4. Peak control = soft clipper -> look-ahead limiter:
   * clipper: 4x oversampled, smooth (tanh, C2) knee.  It takes the sharp
     glottal/transient spikes that need most of the peak reduction and cost almost
     no loudness; its products are harmonic-ish, masked by the peak itself.
   * limiter: offline non-causal look-ahead; reduction (dB) = max-filter + Hann
     ramp (provably >= required reduction everywhere, ramps start before the
     peak); release = exponentially decaying peak hold in two stages (fast, and a
     slow one with capped depth so a lone click can't duck the speech after it).
     Stereo is linked.  Because the clipper already did the heavy lifting the
     limiter normally moves <= 2 dB, which is why it cannot pump.
5. True-peak guarantee: result is oversampled 4x (bench/metrics method) and
   trimmed if a hair over the ceiling.  Ceiling is -1.0 dBTP (we aim -1.03).
6. Guard: if reaching the target would cost more than ~3 LU of loudness to
   clipping+limiting (or >6 dB 99th-percentile gain reduction) we stop short of
   the target instead of crushing the material; `info["depth_limited"]` says so.
"""
import numpy as np
import pyloudnorm as pyln
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly, oaconvolve, butter, sosfiltfilt

TARGET_LUFS = {"voice": -16.0, "music": -14.0}
CEILING_DBTP = -1.0           # what we promise (true peak, 4x oversampled)
_CEIL_MARGIN_DB = 0.03        # aim slightly below so PCM16 rounding can't cross it
MAX_BOOST_DB = 42.0
MAX_CUT_DB = 40.0
TOL_LU = 0.05                 # loudness-loop tolerance
MAX_GR_P99_DB = 6.0           # never crush more than this (99th pct of limiter GR)
MAX_LOSS_LU = {"voice": 3.0, "music": 4.5}   # loudness the peak control may eat

DEFAULT_CFG = {
    "look_ms": 3.0,           # limiter attack ramp (look-ahead)
    "rel_fast_ms": 40.0,      # fast release time constant (dB domain)
    "rel_slow_ms": 160.0,     # slow release time constant
    "slow_cap_db": 4.0,       # max depth the slow stage may hold
    "clip_head_db": 1.5,      # clipper asymptote above the ceiling (0 = clipper off)
    "clip_knee_db": 2.0,      # clipper knee below the ceiling
    "level_db": 2.0,          # leveler max boost (0 = off); voice only
    "max_iter": 8,
}
MODE_CFG = {
    "voice": {},
    "music": {"level_db": 0.0, "clip_head_db": 2.5, "clip_knee_db": 3.0,
              "look_ms": 4.0, "rel_fast_ms": 60.0, "rel_slow_ms": 250.0},
}


# --------------------------------------------------------------------------- utils
def _as_2d(audio):
    a = np.asarray(audio)
    if a.ndim == 1:
        return a[None, :], True
    return a, False


def _clean(audio):
    x = np.array(audio, dtype=np.float64, copy=True)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _kweight(x, sr):
    """BS.1770 K-weighting using pyloudnorm's own filter design (x: (C, N))."""
    m = pyln.Meter(sr)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        y = x[c]
        for f in m._filters.values():
            y = f.apply_filter(y)
        out[c] = y
    return out


def _block_ms(xk, sr, block=0.4, overlap=0.75):
    """Mean-square per gating block, summed over channels (BS.1770 weights = 1)."""
    n = xk.shape[1]
    bl = int(round(block * sr))
    hop = max(1, int(round(block * (1 - overlap) * sr)))
    if n < bl:
        return np.array([np.sum(xk ** 2) / max(n, 1)]) if n else np.array([0.0])
    e = np.sum(xk ** 2, axis=0)
    cs = np.concatenate([[0.0], np.cumsum(e)])
    starts = np.arange(0, n - bl + 1, hop)
    return (cs[starts + bl] - cs[starts]) / bl


def _gated_loudness_from_blocks(z):
    with np.errstate(divide="ignore"):
        l = -0.691 + 10 * np.log10(z)
    ok = l >= -70.0
    if not ok.any():
        return -np.inf
    gamma_r = -0.691 + 10 * np.log10(np.mean(z[ok])) - 10.0
    ok2 = ok & (l > gamma_r)
    if not ok2.any():
        return -np.inf
    return float(-0.691 + 10 * np.log10(np.mean(z[ok2])))


def _lufs_2d(x, sr):
    """Integrated gated loudness of float64 (C, N). -inf if silent.

    pyloudnorm for material >= 0.4 s (identical to the benchmark's meter).
    Shorter clips, which pyloudnorm rejects, get the same K-weighted measure over
    the whole clip as one block (K-weighted RMS), no gating.
    Material whose every block is under the -70 LUFS absolute gate is measured
    with the relative gate only, so very quiet recordings can still be brought up.
    """
    n = x.shape[1]
    if n == 0 or not np.any(x):
        return -np.inf
    if x.shape[0] > 5:
        x = x[:5]
    if n >= int(np.ceil(0.4 * sr)):
        try:
            v = float(pyln.Meter(sr).integrated_loudness(x.T))
            if np.isfinite(v):
                return v
        except Exception:
            pass
    z = _block_ms(_kweight(x, sr), sr)
    v = _gated_loudness_from_blocks(z)
    if np.isfinite(v):
        return v
    zz = z[z > 0]
    if zz.size == 0:
        return -np.inf
    gamma_r = 10 * np.log10(np.mean(zz)) - 10.0
    keep = zz[10 * np.log10(zz) > gamma_r]
    return float(-0.691 + 10 * np.log10(np.mean(keep if keep.size else zz)))


def measure_lufs(audio, sr):
    """Integrated (gated) loudness in LUFS of a (channels, samples) array."""
    a, _ = _as_2d(audio)
    return _lufs_2d(_clean(a), sr)


# ----------------------------------------------------------------------- peaks
_CHUNK = 1 << 19
_PAD = 256


def _chunks(n):
    s = 0
    while s < n:
        e = min(n, s + _CHUNK)
        yield s, e, max(0, s - _PAD), min(n, e + _PAD)
        s = e


def _true_peak_env(x):
    """Per-sample true-peak magnitude: max over the 4 oversampled phases that follow
    each sample, max over channels.  Same interpolator as bench.metrics."""
    c, n = x.shape
    out = np.zeros(n)
    for ch in range(c):
        for s, e, a, b in _chunks(n):
            up = np.abs(resample_poly(x[ch, a:b], 4, 1))
            up = up[: 4 * (b - a)].reshape(-1, 4).max(axis=1)
            out[s:e] = np.maximum(out[s:e], up[s - a: s - a + (e - s)])
    return out


def true_peak_dbfs(x):
    """max |4x oversampled| in dBFS over all channels of (C, N)."""
    pk = 0.0
    for ch in range(x.shape[0]):
        for s, e, a, b in _chunks(x.shape[1]):
            up = resample_poly(x[ch, a:b], 4, 1)[4 * (s - a): 4 * (s - a) + 4 * (e - s)]
            pk = max(pk, float(np.max(np.abs(up))))
    return 20 * np.log10(pk + 1e-12)


# --------------------------------------------------------------------- clipper
def _softclip_os(x, ceil_lin, head_db, knee_db):
    """Oversampled (4x), stereo-linked soft-knee clipper.

    Identity below T = ceil * 10^(-knee/20); above it the curve bends smoothly
    (tanh, C2-continuous) toward A = ceil * 10^(head/20).  The shaping gain is
    taken from the loudest channel and applied to all, so the stereo image is
    kept.  It runs at 4x so its harmonics never alias; the decimation filter
    removes what is above Nyquist."""
    if head_db <= 0:
        return x
    T = ceil_lin * 10 ** (-knee_db / 20)
    A = ceil_lin * 10 ** (head_db / 20)
    w = A - T
    c, n = x.shape
    out = np.empty_like(x)
    for s, e, a, b in _chunks(n):
        up = [resample_poly(x[ch, a:b], 4, 1) for ch in range(c)]
        mag = np.abs(up[0]) if c == 1 else np.max(np.abs(np.stack(up)), axis=0)
        idx = np.nonzero(mag > T)[0]
        if idx.size:
            m = mag[idx]
            g = (T + w * np.tanh((m - T) / w)) / m
            for u in up:
                u[idx] *= g
        for ch in range(c):
            out[ch, s:e] = resample_poly(up[ch], 1, 4)[s - a: s - a + (e - s)]
    return out


# --------------------------------------------------------------------- limiter
def _decay_hold(R, tau):
    """out[n] = max(R[n], exp(-1/tau) * out[n-1]) for R >= 0, vectorised.

    max_k R[k] a^(n-k) = exp(n log a + cummax(log R[k] - k log a))."""
    n = R.size
    if n == 0 or tau <= 0:
        return R.copy()
    idx = np.arange(n, dtype=np.float64)
    with np.errstate(divide="ignore"):
        u = np.log(np.maximum(R, 0.0))
    c = np.maximum.accumulate(u + idx / tau)
    out = np.exp(c - idx / tau)
    out[out < 1e-7] = 0.0
    return np.maximum(out, R)


def _hann(w):
    k = np.hanning(w + 2)[1:-1]
    return k / k.sum()


def _attack(need_db, half):
    """R1 >= need everywhere; ramps in over `half` samples before each peak."""
    if half < 1:
        return need_db.copy()
    w = 2 * half + 1
    m = maximum_filter1d(need_db, size=w, mode="nearest")
    mp = np.pad(m, half, mode="edge")
    return oaconvolve(mp, _hann(w), mode="valid")[: need_db.size]


def gain_reduction_db(need_db, sr, cfg):
    """Required reduction (dB, >=0, per sample) -> smooth limiter reduction (dB)."""
    half = max(1, int(round(cfg["look_ms"] * 1e-3 * sr / 2)))
    r1 = np.maximum(_attack(need_db, half), need_db)
    fast = _decay_hold(r1, cfg["rel_fast_ms"] * 1e-3 * sr)
    slow = _decay_hold(np.minimum(r1, cfg["slow_cap_db"]), cfg["rel_slow_ms"] * 1e-3 * sr)
    return np.maximum(fast, slow)


def _reduction_blocked(p_db, ceil_db, sr, cfg, block=1 << 21):
    """gain_reduction_db over a long envelope in blocks (bounded memory).

    Each block carries 2.5 s of left context (release state: e^-15 of the slowest
    time constant) and 0.1 s of right context (look-ahead)."""
    n = p_db.size
    if n <= block + (block >> 1):
        return gain_reduction_db(np.maximum(p_db - ceil_db, 0.0), sr, cfg)
    left, right = int(2.5 * sr), int(0.1 * sr)
    R = np.empty(n)
    s = 0
    while s < n:
        e = min(n, s + block)
        a, b = max(0, s - left), min(n, e + right)
        r = gain_reduction_db(np.maximum(p_db[a:b] - ceil_db, 0.0), sr, cfg)
        R[s:e] = r[s - a: s - a + (e - s)]
        s = e
    return R


# --------------------------------------------------------------------- leveler
def _leveler_gain_db(x, sr, max_boost_db):
    """Slow, gated upward leveler.  Returns a dB gain curve (>= 0) or None.

    D = how far the 400 ms loudness is below the integrated loudness.  Gain is
    0.25*D after a soft knee (a 1.33:1 upward ratio), capped at `max_boost_db`,
    fades out for D > 14..24 dB (breaths / pauses are not 'quiet words') and
    fades out within 8..18 dB of the noise floor so gaps are never lifted.
    Disabled outright when the recording's speech-to-floor ratio is < 18 dB."""
    if max_boost_db <= 0 or x.shape[1] < int(2.0 * sr):
        return None
    L = _lufs_2d(x, sr)
    if not np.isfinite(L):
        return None
    e = np.mean(_kweight(x, sr) ** 2, axis=0)
    with np.errstate(divide="ignore"):
        S = -0.691 + 10 * np.log10(uniform_filter1d(e, int(0.4 * sr), mode="nearest") + 1e-12)
        F = -0.691 + 10 * np.log10(uniform_filter1d(e, int(0.05 * sr), mode="nearest") + 1e-12)
    floor = np.percentile(F[:: int(0.05 * sr)], 10)
    if L - floor < 18.0:
        return None
    D = np.clip(L - S, 0.0, None)
    knee = np.where(D < 4.0, D * D / 8.0, D - 2.0)
    fade_hi = 1.0 - np.clip((D - 14.0) / 10.0, 0.0, 1.0)
    gate = np.clip((S - floor - 8.0) / 10.0, 0.0, 1.0)
    shape = (fade_hi * fade_hi * (3 - 2 * fade_hi)) * (gate * gate * (3 - 2 * gate))
    g = np.minimum(0.25 * knee, max_boost_db) * shape
    return uniform_filter1d(g, int(0.15 * sr), mode="nearest")


# ---------------------------------------------------------------- core routine
def _master_core(x, sr, mode, cfg, info):
    """x: float64 (C, N), already sanitised. Returns float64 (C, N)."""
    target = TARGET_LUFS.get(mode, TARGET_LUFS["voice"])
    max_loss = MAX_LOSS_LU.get(mode, MAX_LOSS_LU["voice"])
    ceil_db = CEILING_DBTP - _CEIL_MARGIN_DB
    ceil_lin = 10 ** (ceil_db / 20)

    mu = x.mean(axis=1, keepdims=True)
    if np.max(np.abs(mu)) > 1e-4:
        # DC / sub-audio drift wastes headroom.  A zero-phase 3 Hz high-pass removes it; subtracting the
        # mean would instead put a constant offset into any digitally silent stretch of the file.
        if x.shape[1] > 8 * sr // 10:
            x = sosfiltfilt(butter(2, 3.0, "hp", fs=sr, output="sos"), x, axis=1)
        else:
            x = x - mu

    L0 = _lufs_2d(x, sr)
    info["lufs_in"] = L0
    if np.max(np.abs(x)) < 1e-9 or not np.isfinite(L0):
        info["note"] = "silent"          # nothing meaningful to normalise
        return x

    if mode == "voice" and cfg["level_db"] > 0:
        lg = _leveler_gain_db(x, sr, cfg["level_db"])
        if lg is not None:
            x = x * 10 ** (lg / 20)[None, :]
            info["leveler_max_db"] = float(lg.max())
            L0 = _lufs_2d(x, sr)

    use_clip = cfg["clip_head_db"] > 0
    if not use_clip:
        p_db0 = 20 * np.log10(np.maximum(_true_peak_env(x), 1e-12))

    def run(G):
        if use_clip:
            xg = _softclip_os(x * 10 ** (G / 20), ceil_lin, cfg["clip_head_db"], cfg["clip_knee_db"])
            pdb = 20 * np.log10(np.maximum(_true_peak_env(xg), 1e-12))
        else:
            xg = x * 10 ** (G / 20)
            pdb = p_db0 + G
        R = _reduction_blocked(pdb, ceil_db, sr, cfg)
        return xg * 10 ** (-R / 20)[None, :], R

    G = float(np.clip(target - L0, -MAX_CUT_DB, MAX_BOOST_DB))
    lo = hi = None                        # bracket on G: lo feasible & under, hi over/infeasible
    best, hist = None, []
    for _ in range(cfg["max_iter"]):
        y, R = run(G)
        Ly = _lufs_2d(y, sr)
        p99 = float(np.percentile(R, 99)) if R.size else 0.0
        loss = (L0 + G) - Ly
        err = target - Ly
        feasible = (loss <= max_loss) and (p99 <= MAX_GR_P99_DB)
        hist.append((G, Ly))
        if feasible and (best is None or abs(err) < best[0]):
            best = (abs(err), G, y, R, Ly, p99, loss)
        if feasible and abs(err) <= TOL_LU:
            break
        if feasible and err > 0:          # still under target: raise G
            lo = G if lo is None else max(lo, G)
        else:                             # over target or too deep: lower G
            hi = G if hi is None else min(hi, G)
            if not feasible:
                info["depth_limited"] = True
        # next G: secant on measured loudness, kept inside the bracket
        if len(hist) >= 2 and abs(hist[-1][0] - hist[-2][0]) > 1e-6:
            slope = (hist[-1][1] - hist[-2][1]) / (hist[-1][0] - hist[-2][0])
            slope = float(np.clip(slope, 0.25, 1.0))
        else:
            slope = 1.0
        nxt = G + err / slope
        if not feasible:                  # step back by at least the excess depth
            excess = max(loss - max_loss, p99 - MAX_GR_P99_DB, 0.0)
            nxt = min(nxt, G - max(0.15, 0.8 * excess))
        if lo is not None and hi is not None:
            if not (lo < nxt < hi):
                nxt = 0.5 * (lo + hi)
            if hi - lo < 0.02:
                break
        G = float(np.clip(nxt, -MAX_CUT_DB, MAX_BOOST_DB))
    if best is None:                      # never feasible: fall back to the shallowest tried
        G = min(h[0] for h in hist) - 1.0
        y, R = run(G)
        Ly = _lufs_2d(y, sr)
        best = (abs(target - Ly), G, y, R, Ly, float(np.percentile(R, 99)), (L0 + G) - Ly)
    _, G, y, R, Ly, p99, loss = best

    # true-peak verification / trim (exact: TP scales linearly with gain)
    tp = true_peak_dbfs(y)
    trim = 0.0
    for _ in range(4):
        if tp <= ceil_db + 1e-4:
            break
        d = tp - ceil_db + 0.005
        y = y * 10 ** (-d / 20)
        trim += d
        tp = true_peak_dbfs(y)
    info.update(gain_db=G, lufs_out=_lufs_2d(y, sr), tp=tp, trim_db=trim, loss_lu=loss,
                gr_max=float(R.max()) if R.size else 0.0, gr_p99=p99, iters=len(hist))
    if info.get("keep"):
        info["R"] = R
    return y


def _master_impl(audio, sr, mode="voice", cfg=None, info=None):
    info = info if info is not None else {}
    c = {**DEFAULT_CFG, **MODE_CFG.get(mode, {}), **(cfg or {})}
    a, was_1d = _as_2d(audio)
    if a.size == 0:
        return np.asarray(audio, dtype=np.float32)
    y = _master_core(_clean(a), sr, mode, c, info)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y[0] if was_1d else y


def master(audio, sr, mode="voice"):
    """Normalise loudness to target LUFS (gated) and limit true peaks to -1 dBTP."""
    return _master_impl(audio, sr, mode)


def master_with_info(audio, sr, mode="voice"):
    """Like master(), but also returns an info dict (loudness in/out, gain, true peak, gain reduction...)."""
    info = {}
    return _master_impl(audio, sr, mode, info=info), info
