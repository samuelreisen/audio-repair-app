"""Click / pop removal (de-click).

Real phone and voice-memo recordings are often peppered with short, loud impulses: mouth clicks, mic bumps,
crackle, digital glitches.  They are 0.5-8 ms long and can sit 20-55 dB above the voice around them.  Besides
being unpleasant they ruin loudness mastering (the peak is far above the speech, so the file cannot be brought
up to level without crushing it).

Detection
    An event is a short run of samples that is more than ``RATIO`` times the RMS of its own surroundings
    (30 ms either side, excluding a 3 ms guard).  Speech peaks are at most ~5-8x their local RMS, and a speech
    onset after a pause is surrounded by speech on the loud side, so neither is mistaken for a click.  Events
    wider than ``MAX_MS`` are left alone (a thump or a real transient, not a click).
Repair
    Each gap is filled by autoregressive (LPC) prediction forward from the left context and backward from the
    right context, cross-faded.  Order 32 at 48 kHz models voiced speech well over a few milliseconds.

``declick(audio, sr) -> (audio, n_events)``; audio is (channels, samples); returns the very same array object
and 0 when nothing was found.
"""
import numpy as np
from scipy.linalg import solve_toeplitz

RATIO = 8.0           # event peak vs RMS of its surroundings (18 dB); 6.5 starts false-alarming on clean speech
MIN_ABS = 0.02        # ignore anything below -34 dBFS (nothing audible to remove)
GUARD_MS = 3.0        # surroundings exclude +-this around a candidate
WIN_MS = 30.0         # surroundings window each side
MAX_MS = 8.0          # longest gap we will repair
PAD_MS = 0.6          # widen each detected region by this on both sides
MERGE_MS = 2.0        # events closer than this are one gap
ORDER = 32
GAMMA = 0.985          # LPC bandwidth expansion
TAIL_X = 3.0          # a gap is grown outward while samples stay above this multiple of the surroundings
CTX_MS = 20.0         # context used to fit the AR model, each side


def _lpc(seg, order):
    seg = seg * np.hanning(len(seg))
    r = np.correlate(seg, seg, "full")[len(seg) - 1:len(seg) + order]
    if r[0] <= 1e-12:
        return None
    r[0] *= 1.0 + 1e-6
    try:
        return solve_toeplitz(r[:order], r[1:order + 1])
    except Exception:
        return None


def _predict(ctx, a, n):
    """Extend ctx by n samples with AR coefficients a (ctx long enough)."""
    p = len(a)
    buf = np.concatenate([ctx[-p:], np.zeros(n)])
    for i in range(n):
        buf[p + i] = np.dot(a, buf[i:p + i][::-1])
    return buf[p:]


def _repair_gap(x, s, e, sr):
    n = e - s
    ctx = int(CTX_MS * 1e-3 * sr)
    if s < ORDER + 8 or e + ORDER + 8 > len(x):
        x[s:e] = np.linspace(x[max(s - 1, 0)], x[min(e, len(x) - 1)], n)       # edge of file: plain interpolation
        return
    left, right = x[max(0, s - ctx):s], x[e:e + ctx]
    a = None
    segs = [q for q in (left, right) if len(q) > 2 * ORDER]
    if segs:
        al = [_lpc(q, ORDER) for q in segs]
        al = [q for q in al if q is not None]
        if al:
            a = np.mean(al, axis=0) * GAMMA ** np.arange(1, ORDER + 1)    # bandwidth expansion: damps the extrapolation
    if a is None:
        x[s:e] = np.linspace(x[s - 1], x[e], n)
        return
    fwd = _predict(left, a, n)
    bwd = _predict(right[::-1], a, n)[::-1]
    w = (np.arange(n) + 0.5) / n
    rep = (1 - w) * fwd + w * bwd
    near = int(5e-3 * sr)
    lim = 1.5 * max(np.abs(left[-near:]).max(), np.abs(right[:near]).max(), 1e-4)
    if not np.all(np.isfinite(rep)) or np.abs(rep).max() > 3.0 * lim:
        rep = np.linspace(x[s - 1], x[e], n)                                # runaway prediction: plain interpolation
    x[s:e] = np.clip(rep, -lim, lim)


def find_clicks(x, sr):
    """List of (start, end) sample ranges of click events in 1-D x."""
    n = len(x)
    guard, win = int(GUARD_MS * 1e-3 * sr), int(WIN_MS * 1e-3 * sr)
    if n < 4 * win:
        return []
    a = np.abs(x)
    cand = np.flatnonzero(a > MIN_ABS)
    if cand.size == 0:
        return []
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    i = cand[(cand > win) & (cand < n - win)]
    if i.size == 0:
        return []
    left = (cs[i - guard] - cs[i - win]) / (win - guard)
    right = (cs[i + win] - cs[i + guard]) / (win - guard)
    sur = np.sqrt(np.maximum(0.5 * (left + right), 1e-12))
    hit = i[a[i] > RATIO * sur]
    if hit.size == 0:
        return []
    merge = int(MERGE_MS * 1e-3 * sr)
    pad = int(PAD_MS * 1e-3 * sr)
    cut = np.flatnonzero(np.diff(hit) > merge) + 1
    events = []
    for grp in np.split(hit, cut):
        s, e = int(grp[0]) - pad, int(grp[-1]) + pad + 1
        if events and s - events[-1][1] < merge:
            events[-1] = (events[-1][0], e)
        else:
            events.append((s, e))
    maxn = int(MAX_MS * 1e-3 * sr)
    out = []
    for s, e in events:
        mid = (s + e) // 2
        g = (e - s) // 2 + guard
        lo, hi = max(mid - g - win, 0), min(mid + g + win, n)
        ctx = np.concatenate([x[lo:max(mid - g, lo)], x[min(mid + g, hi):hi]])
        sur = np.sqrt(np.mean(ctx ** 2)) + 1e-9 if ctx.size else 1e-9
        while e - s < maxn and s > 1 and a[s - 1] > TAIL_X * sur:       # ring-down before / after the main spike
            s -= 1
        while e - s < maxn and e < n - 1 and a[e] > TAIL_X * sur:
            e += 1
        out.append((max(s, 1), min(e, n - 1)))
    return [(s, e) for s, e in out if e - s <= maxn]


def declick(audio, sr):
    x = np.asarray(audio)
    one_d = x.ndim == 1
    a = x[None, :] if one_d else x
    out = None
    total = 0
    for c in range(a.shape[0]):
        ch = a[c].astype(np.float64)
        ev = find_clicks(ch, sr)
        if not ev:
            continue
        if out is None:
            out = a.astype(np.float64, copy=True)
        for s, e in ev:
            _repair_gap(out[c], s, e, sr)
        total += len(ev)
    if out is None:
        return audio, 0
    out = np.nan_to_num(out).astype(np.float32)
    return (out[0] if one_d else out), total
