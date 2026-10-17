"""Clipping repair (declipping).

Public API (unchanged)::

    declip(audio, threshold=0.985) -> audio      # (channels, samples) float32

What it does
------------
1. **Detect** clipping from the data itself, per channel and per polarity.  No absolute
   level is assumed: the ceiling may be 1.0, 0.9, 0.5 (file normalised afterwards), the
   positive and negative ceilings may differ, and several ceilings may coexist.  A ceiling
   is a *spike* in the amplitude histogram (many samples sharing exactly one value, which
   is also the local maximum around them).  If no flat top exists, a soft-saturation test
   looks for a top of the amplitude distribution that is flat instead of decaying.
2. **Reconstruct** the missing peaks with a consistent, time-varying autoregressive
   (Janssen / EM) interpolation: speech is well modelled as AR(48) at 48 kHz; the unknown
   samples are the minimum-prediction-error completion subject to the clipping
   constraint (the true sample is on the far side of the clip level).  The AR model is
   re-estimated from the repaired signal (3 EM passes).  The normal equations are banded
   and are solved with a sparse LU, so cost is linear in the number of clipped samples.
3. **Headroom**: the repaired peaks exceed full scale, so the signal is scaled down by one
   common gain (only when needed) so that its true peak (4x oversampled) is at most
   ``OUTPUT_PEAK`` = -1 dBTP.  Downstream gain/limiter stages then never hard-clip the
   repair again.

Strict no-op: if nothing is detected the *same array object* is returned.
"""
import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly
from scipy.signal.windows import hann

OUTPUT_PEAK = 10 ** (-1.0 / 20)   # repaired signal is scaled (only if needed) so its TRUE peak
                                  # (4x oversampled) is at most -1 dBTP, the master stage's ceiling
AR_ORDER = 48               # AR order at 48 kHz (~ fs/1000 for speech)
AR_WIN = 2048               # analysis window for the AR model (samples)
AR_HOP = 256                # AR model update interval (samples)
EM_ITERS = 3                # AR re-estimation passes
WNC = 1e-8                  # white-noise correction for the AR estimate (-80 dB)
MAX_OVERSHOOT = 8.0         # exact plateaus: repaired |x| <= this many times the clip level
SMEAR_OVERSHOOT = 6.0       # smeared plateaus
SOFT_OVERSHOOT = 4.0        # soft saturation
CHUNK_UNKNOWNS = 12000      # unknown samples solved per block (bounds memory, not quality)

# detection
MIN_PLATEAU = 4             # samples sharing the ceiling value
SPIKE_FLOOR = 0.50          # ceilings are looked for above this fraction of the peak (-6 dB)
MAX_LEVELS = 4              # more repeated levels than this = tonal/periodic, not clipping
LOCAL_WIN = 513             # plateau samples must be the local maximum within this window
PLATEAU_LOCAL_TOL = 1.15    # ... up to ringing overshoot of this factor
MAX_PLATEAU_RUN = 1024      # longer constant runs are holds/DC, not clipped peaks
MIN_FLAT_FRACTION = 0.2     # share of ceiling samples that must sit in runs of >= 2 (flat top)


# --------------------------------------------------------------------------- detection
def _plateau_levels(v):
    """Tier 1 - exact flat tops.  v: positive part of one polarity.

    A ceiling is a value shared by several samples *and* standing out of the histogram
    of its neighbouring distinct values *and* being about the largest value around its
    samples.  Returns a list of levels (floats).
    """
    mx = float(v.max()) if v.size else 0.0
    if mx <= 0.0:
        return []
    a = v[v >= SPIKE_FLOOR * mx]
    if a.size < MIN_PLATEAU:
        return []
    u, cnt = np.unique(a, return_counts=True)
    cand = np.flatnonzero(cnt >= MIN_PLATEAU)
    if cand.size == 0:
        return []
    n = u.size
    cs = np.concatenate(([0], np.cumsum(cnt)))
    lo = np.maximum(cand - 3, 0)
    hi = np.minimum(cand + 4, n)
    exp = (cs[hi] - cs[lo] - cnt[cand]) / np.maximum(hi - lo - 1, 1)
    c = cnt[cand]
    ok = (c - exp >= 4.0 * np.sqrt(exp + 1.0) + 2.0) & (c >= 3.0 * np.maximum(exp, 1.0))
    cand = cand[ok]
    if cand.size == 0:
        return []
    lm = maximum_filter1d(v, size=LOCAL_WIN, mode="nearest")
    levels = []
    for i in cand[np.argsort(-cnt[cand])][:32]:
        lev = float(u[i])
        at = np.flatnonzero(v == lev)
        # overshoot ringing (resampled / band-limited files) may exceed the ceiling a bit
        if np.mean(lm[at] <= lev * PLATEAU_LOCAL_TOL) < 0.8:
            continue
        # a clipped top is *flat*: its samples come in runs of >= 2.  Isolated repeats of one
        # value (an exactly periodic synthetic tone, a limiter's repeated peaks) are not clipping.
        d = np.diff(at) == 1
        flat = np.zeros(at.size, bool)
        flat[1:] |= d
        flat[:-1] |= d
        if flat.mean() >= MIN_FLAT_FRACTION:
            levels.append(lev)
    return levels if len(levels) <= MAX_LEVELS else []


def _bump_threshold(v, n_total):
    """Tier 2 - smeared flat tops (lossy codec, resampling, filtering).

    Natural amplitude distributions decay monotonically in the upper range; a clipped
    signal shows a *bump* (mode) near the ceiling after the plateau has been smeared.
    Returns the amplitude where the bump starts (valley below the mode) or None.
    """
    mx = float(v.max()) if v.size else 0.0
    if mx <= 0.0:
        return None
    nb = 120
    h, _ = np.histogram(v[v > 0], bins=nb, range=(0.0, mx))
    h = h.astype(np.float64)
    hp = np.concatenate(([h[0]], h, [h[-1]]))
    hs = 0.25 * hp[:-2] + 0.5 * hp[1:-1] + 0.25 * hp[2:]
    lo_b, hi0 = int(0.40 * nb), int(0.55 * nb)
    b_pk = hi0 + int(np.argmax(hs[hi0:]))
    b_v = lo_b + int(np.argmin(hs[lo_b:b_pk + 1]))
    if b_pk - b_v < 3 or hs[b_pk] < 2.0 * hs[b_v] + 4.0:
        return None
    mass = float(np.maximum(h[b_v:] - hs[b_v], 0.0).sum())
    if mass < max(150.0, 4e-4 * n_total) or h[b_v:].sum() > 0.15 * n_total:
        return None
    return b_v * mx / nb


def _flat_top_threshold(v, n_total):
    """Tier 3 - soft saturation: the top of a (peaky, speech-like) amplitude distribution is
    flat instead of decaying.  Returns the amplitude above which samples are suspect."""
    mx = float(v.max()) if v.size else 0.0
    if mx <= 0.0:
        return None
    nb = 100
    h, _ = np.histogram(v[v > 0], bins=nb, range=(0.0, mx))
    h = h.astype(np.float64)
    top, ref, mid = h[-6:-1].mean(), h[-30:-25].mean(), h[10:20].mean()
    if ref <= 0 or top < 0.5 * ref or top > 0.5 * mid:
        return None
    if h[-30:].sum() < max(150.0, 0.003 * n_total) or h[50:].sum() > 0.15 * n_total:
        return None
    return 0.5 * mx


def detect_clipping(x):
    """Find clipped / saturated samples of the 1-D signal x.

    Returns (indices, signs, cap_mag, kind): ``indices`` sorted positions of samples whose
    true magnitude is >= the observed one, ``signs`` their polarity, ``cap_mag`` a
    plausibility ceiling for the repaired magnitude, ``kind`` None / 'hard' / 'smeared' /
    'soft'.
    """
    x = np.asarray(x, dtype=np.float64)
    empty = (np.empty(0, np.int64), np.empty(0), np.empty(0), None)
    if x.size < 64 or not np.isfinite(x).all():
        return empty
    n = x.size
    mask = np.zeros(n, bool)
    cap = np.zeros(n)
    kind = None
    for v in (np.maximum(x, 0.0), np.maximum(-x, 0.0)):
        mx = float(v.max())
        if mx <= 0.0:
            continue
        levels = _plateau_levels(v)
        exact = np.zeros(n, bool)
        for lev in levels:
            exact |= v == lev
        if levels:
            run_ok = _short_runs(exact, MAX_PLATEAU_RUN)
            mask |= run_ok
            cap[run_ok] = np.maximum(cap[run_ok], MAX_OVERSHOOT * mx)
            kind = "hard"
        th = _bump_threshold(np.where(exact, 0.0, v), n)
        if th is not None:
            sel = v >= th
            mask |= sel
            cap[sel] = np.maximum(cap[sel], SMEAR_OVERSHOOT * mx)
            kind = kind or "smeared"
        elif not levels:
            th = _flat_top_threshold(v, n)
            if th is not None:
                sel = v >= th
                mask |= sel
                cap[sel] = np.maximum(cap[sel], SOFT_OVERSHOOT * mx)
                kind = kind or "soft"
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return empty
    return idx, np.sign(x[idx]), cap[idx], kind


def _short_runs(mask, max_len):
    """Keep only runs of True shorter than/equal to max_len (long holds are not clipping)."""
    if not mask.any():
        return mask
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    bad = (ends - starts) > max_len
    if not bad.any():
        return mask
    out = mask.copy()
    for s, e in zip(starts[bad], ends[bad]):
        out[s:e] = False
    return out


# ------------------------------------------------------------------ AR model estimation
def _levinson(r, p):
    """Batched Levinson-Durbin.  r: (F, p+1) autocorrelations -> (a (F,p+1), err (F,))."""
    F = r.shape[0]
    a = np.zeros((F, p + 1))
    a[:, 0] = 1.0
    e = r[:, 0].copy()
    for m in range(1, p + 1):
        acc = np.einsum("fk,fk->f", a[:, :m], r[:, m:0:-1])
        k = -acc / e
        a[:, :m + 1] += k[:, None] * a[:, m::-1].copy()
        e = e * (1.0 - k * k)
    return a, e


def _frame_ar(y, p, W, hop, wnc, win):
    """AR coefficients of hop-spaced frames (frame f is centred on f*hop + hop/2)."""
    N = y.size
    nF = (N + hop - 1) // hop
    pad = (W - hop) // 2
    xp = np.concatenate((np.zeros(pad), y, np.zeros(W + hop)))
    fr = np.lib.stride_tricks.sliding_window_view(xp, W)[::hop][:nF] * win
    S = sfft.rfft(fr, 2 * W, axis=1)
    r = sfft.irfft(S.real ** 2 + S.imag ** 2, axis=1)[:, :p + 1]
    r[:, 0] = r[:, 0] * (1.0 + wnc) + 1e-18
    a, e = _levinson(r, p)
    return a, np.maximum(e, 1e-30 * np.maximum(r[:, 0], 1e-30)) / (win ** 2).sum()


def _solve_unknowns(y, U, a_f, w_f, hop, p):
    """Minimise sum_n w_n (a_n * y)_n^2 over y[U]; returns the minimiser (M,)."""
    N = y.size
    M = U.size
    uidx = np.full(N, -1, dtype=np.int32)
    uidx[U] = np.arange(M, dtype=np.int32)
    cnt = np.zeros(N + p + 2, dtype=np.int32)
    np.add.at(cnt, U, 1)
    np.add.at(cnt, U + p + 1, -1)
    rows = np.flatnonzero(np.cumsum(cnt)[:N] > 0)
    rows = rows[rows >= p]
    f = np.minimum(rows // hop, a_f.shape[0] - 1)
    C = a_f[f] * np.sqrt(w_f[f])[:, None]
    pos = rows[:, None] - np.arange(p + 1)[None, :]
    ui = uidx[pos]
    yk = y.copy()
    yk[U] = 0.0
    e0 = np.einsum("rj,rj->r", C, yk[pos])
    R = rows.size
    valid = ui >= 0
    rr = np.broadcast_to(np.arange(R, dtype=np.int32)[:, None], (R, p + 1))
    Cu = sp.csr_matrix((C[valid], (rr[valid], ui[valid])), shape=(R, M))
    H = (Cu.T @ Cu).tocsc()
    b = -(Cu.T @ e0)
    H = H + sp.diags(1e-9 * H.diagonal() + 1e-30)
    lu = spl.splu(H, permc_spec="NATURAL", diag_pivot_thresh=0.0,
                  options=dict(SymmetricMode=True))
    return lu.solve(b)


def _initial_fill(x, mask):
    """Rough interpolation across clipped samples (cubic spline through the known ones)."""
    known = np.flatnonzero(~mask)
    y = x.copy()
    if known.size < 8:
        return y
    cs = CubicSpline(known, x[known], bc_type="natural")
    u = np.flatnonzero(mask)
    y[u] = cs(u)
    return y


def _repair_block(seg, Uc, sgn, lo_mag, cap):
    """EM loop on one block.  seg: block samples (observed); Uc: unknown positions in it."""
    p, W, hop = AR_ORDER, AR_WIN, AR_HOP
    win = hann(W, sym=False)
    mask = np.zeros(seg.size, bool)
    mask[Uc] = True
    y = _initial_fill(seg, mask)
    y[Uc] = sgn * np.clip(sgn * y[Uc], lo_mag, cap)
    for _ in range(EM_ITERS):
        a_f, e_f = _frame_ar(y, p, W, hop, WNC, win)
        xu = _solve_unknowns(y, Uc, a_f, 1.0 / e_f, hop, p)
        y[Uc] = sgn * np.clip(sgn * xu, lo_mag, cap)
    return y[Uc]


def _repair_channel(x, U, sgn, cap):
    """Return the repaired channel (float64) given unknown indices U (sorted), their signs
    and a per-sample plausibility ceiling for the repaired magnitude."""
    N = x.size
    lo_mag = np.abs(x[U])                     # the true magnitude is at least the observed one
    cap = np.maximum(cap, lo_mag)
    y = x.copy()
    M = U.size
    ctx = AR_WIN
    s = 0
    while s < M:
        e = min(M, s + CHUNK_UNKNOWNS)
        if e < M:                              # cut at the widest gap of the last quarter
            w0 = e - CHUNK_UNKNOWNS // 4
            g = np.diff(U[w0:e + 1])
            e = w0 + int(np.argmax(g)) + 1
        a0 = max(0, U[s] - ctx)
        b0 = min(N, U[e - 1] + 1 + ctx)
        seg = y[a0:b0]
        out = _repair_block(seg, U[s:e] - a0, sgn[s:e], lo_mag[s:e], cap[s:e])
        y[U[s:e]] = out
        s = e
    return y


def _true_peak(work, block=1 << 18, pad=64):
    """Largest absolute value of the 4x oversampled signal (inter-sample peak), computed in
    blocks so memory stays bounded for long files.  work: (channels, samples) float64."""
    peak = float(np.max(np.abs(work)))
    n = work.shape[1]
    for c in range(work.shape[0]):
        for s in range(0, n, block):
            a, b = max(0, s - pad), min(n, s + block + pad)
            up = resample_poly(work[c, a:b], 4, 1)
            lo, hi = (s - a) * 4, (min(n, s + block) - a) * 4
            peak = max(peak, float(np.max(np.abs(up[lo:hi]))))
    return peak


# ------------------------------------------------------------------------- public API
def declip(audio, threshold=0.985):
    """Reconstruct clipped peaks.

    ``audio`` is (channels, samples) float (a 1-D array is accepted too).  ``threshold``
    is kept for API compatibility: detection no longer depends on an absolute level.
    The result is float32, sample-aligned with the input, and is scaled by one common
    gain (only if the repaired peaks would exceed ``OUTPUT_PEAK``).  If no clipping is
    found the input object is returned unchanged.
    """
    arr = np.asarray(audio)
    one_d = arr.ndim == 1
    a2 = arr[None, :] if one_d else arr
    if a2.size == 0 or not np.isfinite(a2).all():
        return audio
    work = None
    for ci in range(a2.shape[0]):
        x = a2[ci].astype(np.float64)
        U, sgn, cap, kind = detect_clipping(x)
        if kind is None or U.size == 0:
            continue
        if work is None:
            work = a2.astype(np.float64)
        work[ci] = _repair_channel(x, U, sgn, cap)
    if work is None:
        return audio
    peak = _true_peak(work)
    if peak > OUTPUT_PEAK:
        work *= OUTPUT_PEAK / peak
    out = work.astype(np.float32)
    return out[0] if one_d else out
