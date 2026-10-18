"""Tone changes applied AFTER the repair, so the repair itself never has to run again.

Four simple controls, in dB (each -8 .. +8):
    warmth      low shelf at 200 Hz     body and weight of the voice
    clarity     bell at 3 kHz           presence, how forward and intelligible it sounds
    brightness  high shelf at 6 kHz     the overall brightness
    air         high shelf at 12 kHz    the top-end sparkle

The filters are the standard RBJ "Audio EQ Cookbook" biquads, with the same formulas the browser's Web Audio filters use
(shelf slope 1, bell Q 0.8). The page previews the tone live with those browser filters; saving applies these ones to the
stored repair result and then re-matches the loudness (-16 LUFS voice / -14 LUFS music, true peak -1 dBTP), because a
brighter or warmer voice is also a louder or quieter one. Nothing earlier in the repair is touched.
"""
import math

import numpy as np
from scipy.signal import sosfilt

# name -> (kind, centre/corner frequency in Hz, Q for the bell)
BANDS = {
    "warmth": ("lowshelf", 200.0, None),
    "clarity": ("peaking", 3000.0, 0.8),
    "brightness": ("highshelf", 6000.0, None),
    "air": ("highshelf", 12000.0, None),
}
LIMIT_DB = 8.0


def clean_tone(tone):
    """{band: dB} with unknown names dropped, junk turned to 0 and values limited to +-8 dB."""
    out = {}
    for k in BANDS:
        try:
            v = float((tone or {}).get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        if not math.isfinite(v):
            v = 0.0
        out[k] = max(-LIMIT_DB, min(LIMIT_DB, v))
    return out


def is_flat(tone, eps=0.05):
    return all(abs(v) < eps for v in clean_tone(tone).values())


def biquad(kind, f0, gain_db, sr, q=None):
    """One RBJ biquad as an SOS row [b0, b1, b2, 1, a1, a2]."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * min(f0, 0.49 * sr) / sr
    cw, sw = math.cos(w0), math.sin(w0)
    if kind == "peaking":
        alpha = sw / (2.0 * q)
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    else:
        alpha = sw / 2.0 * math.sqrt(2.0)                    # shelf slope S = 1
        sq = 2.0 * math.sqrt(A) * alpha
        if kind == "lowshelf":
            b0 = A * ((A + 1) - (A - 1) * cw + sq)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sq)
            a0 = (A + 1) + (A - 1) * cw + sq
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sq
        elif kind == "highshelf":
            b0 = A * ((A + 1) + (A - 1) * cw + sq)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sq)
            a0 = (A + 1) - (A - 1) * cw + sq
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sq
        else:
            raise ValueError("unknown filter " + kind)
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def sos_for(tone, sr):
    t = clean_tone(tone)
    rows = [biquad(kind, f0, t[name], sr, q) for name, (kind, f0, q) in BANDS.items() if abs(t[name]) >= 0.05]
    return np.asarray(rows, dtype=np.float64) if rows else None


def response_db(tone, sr, freqs):
    """Magnitude response (dB) of the tone curve at the given frequencies (used by tests and the level estimate)."""
    sos = sos_for(tone, sr)
    f = np.asarray(freqs, dtype=np.float64)
    if sos is None:
        return np.zeros_like(f)
    z = np.exp(-1j * 2 * np.pi * f / sr)
    h = np.ones_like(z)
    for b0, b1, b2, _, a1, a2 in sos:
        h *= (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-12))


def apply_tone(audio, sr, tone):
    """audio: float array (channels, samples) or (samples,). Returns the filtered copy, same shape and length."""
    sos = sos_for(tone, sr)
    x = np.asarray(audio, dtype=np.float64)
    if sos is None:
        return np.asarray(audio, dtype=np.float32).copy()
    return sosfilt(sos, x, axis=-1).astype(np.float32)
