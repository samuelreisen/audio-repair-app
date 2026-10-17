"""Bang / thump suppression ("tame").

Voice memos and phone recordings often contain a few knocks, bumps, coughs or table hits that are 30-50 dB
louder than the voice and last 20-200 ms.  Too long for the click remover, too loud to leave alone: they set the
peak level of the whole file, so loudness mastering can only raise the voice by a few dB before it slams into them.

A short-term level (20 ms RMS) is compared with the speaker's own normal level around it (the 90th percentile of
the level within +-4 s).  Anything more than ``HEADROOM_DB`` above that is turned down to that ceiling with a smooth,
look-ahead gain curve.  Real speech peaks are only ~11 dB above their local 90th percentile, so ordinary
speech, plosives and laughter are not touched.

``tame(audio, sr) -> (audio, n_events, max_reduction_db)``; the very same array object and (0, 0.0) when nothing
exceeds the ceiling.
"""
import numpy as np
from scipy.ndimage import minimum_filter1d, percentile_filter, uniform_filter1d

FRAME_MS = 10.0        # level resolution
WIN_MS = 20.0          # RMS window
REF_S = 4.0            # speaker's normal level: percentile of the level within +- this
REF_PCT = 90
HEADROOM_DB = 14.0     # allowed excess over the local normal level
MIN_EVENT_DB = 3.0     # ignore smaller excursions
MIN_ABS_DBFS = -45.0   # never act on material this quiet in absolute terms
MAX_CUT_DB = 30.0


def _levels(mono, hop, win):
    n = len(mono)
    k = max(1, n // hop)
    cs = np.concatenate([[0.0], np.cumsum(mono * mono)])
    c = np.arange(k) * hop + hop // 2
    lo, hi = np.clip(c - win // 2, 0, n), np.clip(c + win // 2, 0, n)
    ms = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return 10 * np.log10(ms + 1e-12), c


def tame(audio, sr):
    x = np.asarray(audio)
    one_d = x.ndim == 1
    a = x[None, :] if one_d else x
    n = a.shape[1]
    if n < int(2 * sr):
        return audio, 0, 0.0
    mono = a.mean(axis=0).astype(np.float64)
    hop, win = int(FRAME_MS * 1e-3 * sr), int(WIN_MS * 1e-3 * sr)
    lvl, centers = _levels(mono, hop, win)
    size = int(2 * REF_S * 1000 / FRAME_MS) | 1
    ref = percentile_filter(lvl, REF_PCT, size=min(size, len(lvl) | 1), mode="nearest")
    excess = lvl - (ref + HEADROOM_DB)
    excess[lvl < MIN_ABS_DBFS] = 0.0
    excess = np.where(excess > MIN_EVENT_DB, excess, 0.0)
    if not excess.any():
        return audio, 0, 0.0
    g = -np.minimum(excess, MAX_CUT_DB)                           # dB, per frame
    g = minimum_filter1d(g, size=7, mode="nearest")                # +-30 ms look-ahead / hold
    g = uniform_filter1d(g, size=5, mode="nearest")                # smooth attack and release
    g = np.minimum(g, 0.0)
    gain = 10 ** (np.interp(np.arange(n), centers, g) / 20.0)
    out = (a * gain[None, :]).astype(np.float32)
    ev = int(np.sum(np.diff(np.r_[0, (g < -MIN_EVENT_DB).astype(int)]) == 1))
    return (out[0] if one_d else out), ev, float(-g.min())
