"""Objective audio-quality metrics. `ref` is the clean original, `deg` the processed signal.
All inputs are mono float arrays at the same sample rate and sample-aligned."""
import numpy as np
import pyloudnorm as pyln
from scipy.signal import stft, resample_poly
from pystoi import stoi


def lufs(x, sr):
    try:
        return float(pyln.Meter(sr).integrated_loudness(x[:, None]))
    except Exception:
        return float("nan")


def match_level(x, ref, sr):
    lx, lr = lufs(x, sr), lufs(ref, sr)
    if not (np.isfinite(lx) and np.isfinite(lr)):
        return x
    return x * 10 ** ((lr - lx) / 20)


def si_sdr(ref, est, eps=1e-12):
    ref = ref - ref.mean(); est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def lsd_db(ref, est, sr, lo=100, hi=8000):
    """Mean log-spectral distance (dB) over speech-active frames, band-limited."""
    f, _, R = stft(ref, sr, nperseg=1024, noverlap=768)
    _, _, E = stft(est, sr, nperseg=1024, noverlap=768)
    m = (f >= lo) & (f <= hi)
    Rm, Em = np.abs(R[m]) + 1e-8, np.abs(E[m]) + 1e-8
    act = Rm.max(axis=0) > np.percentile(Rm.max(axis=0), 40)
    d = 20 * np.log10(Rm[:, act] / Em[:, act])
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def noise_floor_db(x, sr, win=0.05):
    """Level (dBFS) of the quietest 10% of frames, and speech-to-floor ratio (dB)."""
    n = int(sr * win)
    k = len(x) // n
    rms = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    return float(np.percentile(db, 10)), float(np.percentile(db, 90) - np.percentile(db, 10))


def clip_fraction(x, thr=0.985):
    p = np.max(np.abs(x))
    return float(np.mean(np.abs(x) >= thr * p)) if p > 0 else 0.0


def true_peak_dbfs(x, sr):
    up = resample_poly(x, 4, 1)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def brightness(x, sr):
    """Share of spectral energy above 4 kHz (a proxy for 'air'/clarity)."""
    f, _, S = stft(x, sr, nperseg=1024)
    p = np.abs(S) ** 2
    return float(p[f >= 4000].sum() / (p.sum() + 1e-12))


def report(ref, out, sr):
    a = match_level(out, ref, sr)
    nf, dr = noise_floor_db(out, sr)
    return {
        "STOI": round(float(stoi(ref, a, sr, extended=False)), 3),
        "SI-SDR": round(si_sdr(ref, a), 1),
        "LSD": round(lsd_db(ref, a, sr), 1),
        "LUFS": round(lufs(out, sr), 1),
        "TP": round(true_peak_dbfs(out, sr), 1),
        "floor": round(nf, 1),
        "SNRdyn": round(dr, 1),
        "clip%": round(100 * clip_fraction(out), 3),
        "air": round(brightness(a, sr), 3),
    }
