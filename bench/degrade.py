"""Deterministic degradations applied to clean speech, to emulate bad recordings."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve


def _rng(seed):
    return np.random.default_rng(seed)


def _pink(n, rng):
    # Voss-style via FFT shaping: 1/f power
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec)); f[0] = 1
    spec /= np.sqrt(f)
    p = np.fft.irfft(spec, n)
    return p / np.std(p)


def _scale_to_snr(sig, noise, snr_db):
    ps = np.mean(sig ** 2)
    pn = np.mean(noise ** 2)
    return noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def room_ir(sr, rt60=0.5, seed=1):
    rng = _rng(seed)
    n = int(sr * rt60)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 0
    ir = ir / np.sqrt(np.sum(ir ** 2)) * 0.6  # wet energy ratio
    ir[0] = 1.0                                # direct path, aligned
    return ir


def hard_clip(x, gain_db=12.0, limit=1.0):
    return np.clip(x * 10 ** (gain_db / 20), -limit, limit)


def add_noise(x, sr, snr_db=8.0, hum=True, seed=2):
    rng = _rng(seed)
    n = len(x)
    noise = 0.6 * _pink(n, rng) + 0.4 * rng.standard_normal(n)
    out = x + _scale_to_snr(x, noise, snr_db)
    if hum:
        t = np.arange(n) / sr
        h = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3))
        out = out + _scale_to_snr(x, h, snr_db + 12)
    return out


def bad_mic(x, sr, rt60=0.45):
    sos_hp = butter(4, 220, "hp", fs=sr, output="sos")
    sos_lp = butter(4, 4200, "lp", fs=sr, output="sos")
    y = sosfiltfilt(sos_lp, sosfiltfilt(sos_hp, x))
    # nasal resonance ~1.1 kHz
    sos_bp = butter(2, [900, 1400], "bp", fs=sr, output="sos")
    y = y + 0.8 * sosfiltfilt(sos_bp, y)
    if rt60:
        y = fftconvolve(y, room_ir(sr, rt60), mode="full")[: len(x)]
    return y


def lowlevel(x, peak_dbfs=-30.0):
    return x / (np.max(np.abs(x)) + 1e-12) * 10 ** (peak_dbfs / 20)


CASES = {
    "clean":    lambda x, sr: x.copy(),   # do-no-harm check: must not make clean audio worse
    "clipped": lambda x, sr: hard_clip(x / np.max(np.abs(x)) * 0.9, 12.0),
    "noisy":    lambda x, sr: add_noise(x, sr, 6.0),
    "badmic":   lambda x, sr: bad_mic(x, sr),
    "combined": lambda x, sr: hard_clip(
        add_noise(bad_mic(x, sr, 0.35), sr, 10.0) / 1.0 * 4, 6.0) * 0.9,
    "quiet":    lambda x, sr: lowlevel(add_noise(x, sr, 14.0, hum=False), -34.0),
}


def remove_pauses(x, sr, fade=0.008):
    """Cut every pause out of speech (continuous talking, no gaps with a visible noise floor)."""
    n = int(0.01 * sr); k = len(x) // n
    rms = 20 * np.log10(np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-9)
    keep = np.repeat(rms > (np.percentile(rms, 95) - 32), n)
    xs = x[:len(keep)]; segs = []; i = 0
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j]:
                j += 1
            segs.append(xs[i:j]); i = j
        else:
            i += 1
    f = int(fade * sr); out = segs[0]
    for s_ in segs[1:]:
        if len(s_) <= 2 * f or len(out) <= 2 * f:
            out = np.concatenate([out, s_]); continue
        w = np.linspace(0, 1, f)
        out = np.concatenate([out[:-f], out[-f:] * (1 - w) + s_[:f] * w, s_[f:]])
    return out


CASES["nopause"] = lambda x, sr: remove_pauses(x, sr)
