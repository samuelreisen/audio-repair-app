"""Single-channel dereverberation: weighted prediction error (WPE) in the STFT domain.

WPE predicts the *late* reverberation of every frequency bin from that bin's own past frames (taps that
start `delay` frames back, so the direct sound and early reflections are never predicted) and subtracts it.
It is blind, linear and parameter-light, so unlike spectral-subtraction dereverbs it does not smear or
gate speech.  A decay-rate based room detector scales the amount applied so that already-dry audio is
left (almost) untouched.

    dereverb(audio, sr, strength=0.8, auto=True) -> audio   # (channels, samples) float32, same shape
"""
import numpy as np
from scipy import ndimage

from ._dsp import istft, stft


# ------------------------------------------------------------------------------------------ WPE
def _wpe_channel(x, sr, strength=1.0, delay=2, taps=20, iters=3, nbr=5):
    n_fft = 1024 if sr >= 32000 else 512
    hop = n_fft // 2
    X = stft(x, n_fft, hop)                       # (T, K)
    T, K = X.shape
    Y = X.T.copy()                                # (K, T)
    out = Y.copy()
    ridge = np.eye(taps)[None]
    chunk = int(np.clip(8e7 / (T * taps * 16.0), 2, 128))      # bins per batch: bounds memory on long files
    for k0 in range(0, K, chunk):
        k1 = min(k0 + chunk, K)
        Yc = Y[k0:k1]                             # (F, T)
        F = Yc.shape[0]
        Yt = np.zeros((F, T, taps), dtype=Yc.dtype)      # delayed tap matrix
        for j in range(taps):
            s = delay + j
            if s < T:
                Yt[:, s:, j] = Yc[:, :T - s]
        Yh = Yt.conj().transpose(0, 2, 1)         # (F, taps, T)
        Xh = Yc
        for _ in range(iters):
            lam = ndimage.uniform_filter(np.abs(Xh) ** 2, size=(1, 2 * nbr + 1), mode="nearest")
            lam = np.maximum(lam, 1e-10 * lam.max() + 1e-30)
            w = 1.0 / lam                         # (F, T) inverse source-power weights
            R = np.matmul(Yh * w[:, None, :], Yt)                       # (F, taps, taps)
            r = np.matmul(Yh * w[:, None, :], Yc[:, :, None])[:, :, 0]  # (F, taps)
            tr = np.trace(R, axis1=1, axis2=2).real[:, None, None]
            R = R + ridge * (1e-6 * tr / taps + 1e-30)
            g = np.linalg.solve(R, r[:, :, None])[:, :, 0]
            Xh = Yc - np.einsum("ftk,fk->ft", Yt, g)
        out[k0:k1] = Xh
    if strength < 1.0:
        out = out * strength + Y * (1.0 - strength)
    return istft(out.T, len(x), n_fft, hop)


# ------------------------------------------------------------------------------ room detection
def reverb_amount(x, sr, band=(150.0, 4000.0)):
    """0..1: how reverberant the recording is, from the decay rate of energy tails after speech.

    Dry speech decays naturally in ~0.2 s (-60 dB extrapolated); a reverberant room makes many clean
    straight-line decays that are 0.3..0.6 s.  Noisy recordings hide most decays under the floor, so
    the score is also weighted by how many decays could actually be observed (do no harm when unsure).
    """
    x = np.asarray(x, np.float64)
    n_fft, hop = 1024, 256
    if sr < 32000:
        n_fft, hop = 512, 128
    if len(x) < sr * 1.5:
        return 0.0
    X = stft(x, n_fft, hop)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = (f >= band[0]) & (f <= band[1])
    e = 10 * np.log10(np.sum(np.abs(X[:, m]) ** 2, axis=1) + 1e-20)
    e = ndimage.uniform_filter1d(e, 3, mode="nearest")
    fps = sr / hop
    L = max(int(round(0.1 * fps)), 6)
    T = len(e)
    if T < 4 * L:
        return 0.0
    floor = np.percentile(e, 5)
    seg = np.lib.stride_tricks.sliding_window_view(e, L)              # (T-L+1, L)
    t = np.arange(L) - (L - 1) / 2.0
    mean = seg.mean(axis=1)
    sl = (seg - mean[:, None]) @ t / np.dot(t, t)
    fit = mean[:, None] + sl[:, None] * t[None, :]
    ss_res = np.sum((seg - fit) ** 2, axis=1)
    ss_tot = np.sum((seg - mean[:, None]) ** 2, axis=1) + 1e-9
    r2 = 1 - ss_res / ss_tot
    drop = seg[:, 0] - seg[:, -1]
    ok = (seg[:, 0] > floor + 12) & (sl < 0) & (drop >= 8.0) & (r2 >= 0.85) & (seg[:, -1] > floor + 6)
    cnt = int(ok.sum())
    if cnt == 0:
        return 0.0
    rt = -60.0 / (sl[ok] * fps)
    rt_med = float(np.median(rt))
    conf = np.clip((cnt / float(len(seg)) - 0.03) / 0.04, 0.0, 1.0)
    score = np.clip((rt_med - 0.24) / 0.12, 0.0, 1.0)
    return float(conf * score)


def dereverb(audio, sr, strength=0.8, auto=True):
    """WPE dereverberation.  strength 0..1 (0 = exact pass-through).  With auto=True the amount is also
    scaled by the detected room reverberation, so dry/clean recordings are left essentially untouched."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return audio
    a = np.asarray(audio, np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[None]
    amt = strength
    if auto:
        amt *= reverb_amount(a.mean(axis=0), sr)
    if amt < 0.02 or a.shape[1] < sr * 0.5:
        return audio
    iters, taps = (3, 20) if a.shape[1] < sr * 90 else (2, 16)      # keep very long files tractable
    out = np.stack([_wpe_channel(c, sr, amt, iters=iters, taps=taps) for c in a])
    out = out[0] if squeeze else out
    return out.astype(np.float32)
