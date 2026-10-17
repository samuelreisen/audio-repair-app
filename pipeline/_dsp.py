"""Small DSP helpers shared by denoise.py and dereverb.py (zero-delay STFT)."""
import numpy as np


def _win(n_fft):
    # sqrt-Hann: used for analysis AND synthesis -> perfect reconstruction at hop = n_fft/4 or n_fft/2
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft))


def stft(x, n_fft, hop):
    """x (n,) float -> (frames, n_fft//2+1) complex.  Reflect padded so frame 0 starts before sample 0
    and istft() returns exactly n samples, sample-aligned with the input."""
    pad = n_fft - hop
    n = len(x)
    extra = (-(n + 2 * pad - n_fft)) % hop
    xp = np.pad(x, (pad, pad + extra), mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]
    return np.fft.rfft(fr * _win(n_fft), axis=1)


def istft(X, n, n_fft, hop):
    T = X.shape[0]
    pad = n_fft - hop
    fr = np.fft.irfft(X, n_fft, axis=1) * _win(n_fft)
    out = np.zeros((T - 1) * hop + n_fft)
    for r in range(n_fft // hop):
        out[r * hop: r * hop + T * hop] += fr[:, r * hop:(r + 1) * hop].reshape(-1)
    out *= hop / (n_fft * 0.5)          # sum of squared sqrt-Hann windows = (n_fft/hop) * 0.5
    return out[pad: pad + n]
