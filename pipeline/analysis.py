"""Cheap, deterministic input analysis for the repair pipeline.

    analyse(audio, sr, mode)            -> dict of measurements (see below)
    describe(an)                        -> list[str], what was detected, in plain words
    auto_denoise_strength(an, mode)     -> float, noise-reduction strength chosen from the analysis

Everything here is a measurement on the *input*; no audio is changed.  The numbers are used by
pipeline.__init__ to (a) skip stages that would do nothing, (b) pick per-stage strengths when the
caller did not force them and (c) write the human-readable report.  Cost: well under 1 s for 10 s.

Measurements (mono mix unless stated):
  duration_s, channels, sr
  peak_dbfs                   sample peak
  lufs                        integrated, gated loudness (BS.1770)
  clip_pct, clip_kind         share of samples that are clipped / saturated, and kind
                              (None / 'hard' / 'smeared' / 'soft'), via declip.detect_clipping
  floor_dbfs, speech_dbfs     10th / 90th percentile of 50 ms frame RMS
  dyn_db                      speech_dbfs - floor_dbfs  ("speech-to-floor" range)
  presence                    0..1, how audible the noise floor is (same rule the denoiser uses)
  hum_hz                      mains hum fundamental (50/60) or None
  reverb                      0..1 reverberance (decay-rate room detector, same as the de-reverb stage)
  rolloff_hz                  frequency under which 99 % of the speech energy lies (bandwidth)
  air                         share of speech energy above 4 kHz
  tilt_db_oct                 slope of the long-term speech spectrum 500 Hz .. 4 kHz (dB / octave)
  dc                          mean value of the signal (DC offset)
"""
import numpy as np
from scipy import signal

from .declip import detect_clipping
from .dereverb import reverb_amount
from .denoise import _presence, detect_hum
from .master import measure_lufs

NO_PAUSE_MAX_DYN_DB = 25.0  # loud-to-quiet range (dB) below which an unsteady floor means the talker never pauses
FLOOR_STEADY_MAX_DB = 3.0  # std of the quietest quarter of frames; above it there is no steady noise floor
BAND_LIMITED_HZ = 4200.0  # 99 % of the speech energy below this: telephone / bad-mic bandwidth.
                          # (tilt_db_oct is measured for information only - the tone stage reports its own
                          # corrections; a slope threshold cannot be calibrated on three synthetic voices.)


def _frame_rms_db(mono, sr, win=0.05):
    n = int(win * sr)
    k = len(mono) // n
    if k < 4:
        return None
    rms = np.sqrt(np.mean(mono[:k * n].reshape(k, n) ** 2, axis=1)) + 1e-9
    return 20 * np.log10(rms)


def _speech_spectrum(mono, sr):
    """Long-term power spectrum of the loud frames: (freqs, power)."""
    nper = 2048
    if len(mono) < 4 * nper:
        return None, None
    f, _, S = signal.stft(mono, sr, nperseg=nper, noverlap=nper // 2, boundary=None, padded=False)
    P = np.abs(S) ** 2
    e = 10 * np.log10(P.sum(axis=0) + 1e-20)
    act = e >= np.percentile(e, 95) - 20.0
    if act.sum() < 3:
        act = e >= np.percentile(e, 50)
    return f, P[:, act].mean(axis=1)


def analyse(audio, sr, mode="voice"):
    a = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    if a.ndim == 1:
        a = a[None]
    ch, n = a.shape
    mono = a.mean(axis=0)
    an = dict(channels=int(ch), sr=int(sr), duration_s=float(n / sr), mode=mode,
              peak_dbfs=float(20 * np.log10(np.max(np.abs(a)) + 1e-12)) if n else -120.0,
              dc=float(mono.mean()) if n else 0.0)
    an["silent"] = bool(n == 0 or not np.any(a))
    if an["silent"] or n < int(0.5 * sr):
        an.update(lufs=float("-inf"), clip_pct=0.0, clip_kind=None, floor_dbfs=-120.0, speech_dbfs=-120.0,
                  dyn_db=0.0, presence=0.0, hum_hz=None, reverb=0.0, rolloff_hz=None, air=None,
                  tilt_db_oct=None, too_short=True, floor_spread_db=0.0, floor_steady=True, no_pauses=False)
        return an
    an["too_short"] = False

    an["lufs"] = float(measure_lufs(a, sr))

    # clipping: worst channel
    pct, kind = 0.0, None
    for c in range(ch):
        idx, _, _, k = detect_clipping(a[c])
        if k is not None and idx.size:
            p = 100.0 * idx.size / n
            if p >= pct:
                pct, kind = p, k
    an["clip_pct"], an["clip_kind"] = float(pct), kind

    # noise floor / speech level
    db = _frame_rms_db(mono, sr)
    if db is None:
        an.update(floor_dbfs=an["peak_dbfs"], speech_dbfs=an["peak_dbfs"], dyn_db=0.0,
                  floor_spread_db=0.0, floor_steady=True, no_pauses=False)
    else:
        fl, sp = float(np.percentile(db, 10)), float(np.percentile(db, 90))
        an.update(floor_dbfs=max(fl, -120.0), speech_dbfs=sp, dyn_db=float(min(sp - max(fl, -120.0), 120.0)))
        # Is the "floor" a real, steady noise floor?  Stationary noise makes the quietest frames cluster within
        # ~2 dB of each other; when a talker never pauses, the quietest frames are just soft phonemes and vary
        # by 3.5 dB or more (bench: noisy 0.7-1.9, pauseless speech 3.6-4.4, pauseless speech + real noise 2.0-2.7).
        # Digital silence (<= -100 dBFS) is "no noise", not a misread floor.
        low = np.sort(db)[:max(4, len(db) // 4)]
        an["floor_spread_db"] = float(np.std(low))
        an["floor_steady"] = bool(fl <= -100.0 or an["floor_spread_db"] <= FLOOR_STEADY_MAX_DB)
        # "Never pauses" = no steady floor AND a small loud-to-quiet range.  A reverberant or band-limited recording
        # that does have pauses also has an unsteady floor (decaying tails) but a large range (bench badmic: 60 dB).
        an["no_pauses"] = bool(not an["floor_steady"] and an["dyn_db"] < NO_PAUSE_MAX_DYN_DB)
    an["presence"] = float(_presence(mono, sr)) if mode == "voice" else None

    # hum
    try:
        f0, hs = detect_hum(mono, sr, "voice" if mode == "voice" else "music")
    except Exception:
        f0, hs = None, []
    an["hum_hz"] = float(f0) if f0 else None

    # reverberance
    try:
        an["reverb"] = float(reverb_amount(mono, sr)) if mode == "voice" else 0.0
    except Exception:
        an["reverb"] = 0.0

    # bandwidth / tilt of the speech spectrum
    f, P = _speech_spectrum(mono, sr)
    if f is None:
        an.update(rolloff_hz=None, air=None, tilt_db_oct=None)
    else:
        m = f >= 80.0
        cs = np.cumsum(P[m])
        fm = f[m]
        an["rolloff_hz"] = float(fm[min(np.searchsorted(cs, 0.99 * cs[-1]), len(fm) - 1)])
        an["air"] = float(P[f >= 4000.0].sum() / (P.sum() + 1e-20))
        # 1/3-octave band levels between 500 Hz and 4 kHz -> least-squares slope in dB / octave
        cent = 500.0 * 2.0 ** (np.arange(0, 9) / 3.0)
        lv = []
        for c0 in cent:
            sel = (f >= c0 / 2 ** (1 / 6)) & (f < c0 * 2 ** (1 / 6))
            lv.append(10 * np.log10(P[sel].sum() + 1e-20))
        x = np.log2(cent)
        an["tilt_db_oct"] = float(np.polyfit(x, np.array(lv), 1)[0])
    return an


def auto_denoise_strength(an, mode="voice"):
    """Noise-reduction strength from the analysis.

    The denoiser already fades itself out as the noise gets inaudible (presence gate) and is
    scaled by the detected reverberance, so this only has to choose how hard to push when
    there is something to remove.  0.8 is the measured sweet spot on the bench (SI-SDR is flat
    between 0.7 and 1.0 while a higher value costs more speech detail)."""
    if mode != "voice":
        return 0.7
    return 0.8


def describe(an):
    """Plain-language findings about the input."""
    out = []
    if an.get("silent"):
        return ["The file is silent - nothing to repair."]
    if an.get("too_short"):
        return ["The clip is shorter than half a second - only loudness mastering is applied."]
    out.append("Loudness %.1f LUFS, peak %.1f dBFS." % (an["lufs"], an["peak_dbfs"]))
    if an["clip_kind"]:
        kind = {"hard": "hard clipping", "smeared": "clipping (smeared by resampling/coding)",
                "soft": "soft saturation"}[an["clip_kind"]]
        out.append("Clipping: %s on %.1f%% of samples." % (kind, an["clip_pct"]))
    else:
        out.append("No clipping detected.")
    if an["mode"] == "voice":
        d = an["dyn_db"]
        if d < 20:
            lvl = "heavy"
        elif d < 28:
            lvl = "clearly audible"
        elif d < 42:
            lvl = "mild"
        else:
            lvl = "inaudible"
        if an["floor_dbfs"] <= -100.0:
            out.append("Background noise: none (the pauses are digitally silent).")
        elif an.get("no_pauses"):
            out.append("Background noise: cannot be measured (the speaker never pauses), so it is left alone.")
        else:
            out.append("Background noise: %s (speech is %.0f dB above the floor, floor %.0f dBFS)."
                       % (lvl, d, an["floor_dbfs"]))
    else:
        out.append("Noise floor %.0f dBFS, dynamic range %.0f dB." % (an["floor_dbfs"], an["dyn_db"]))
    if an["hum_hz"]:
        out.append("Mains hum at %.0f Hz." % an["hum_hz"])
    if an["mode"] == "voice":
        r = an["reverb"]
        if an.get("no_pauses"):
            out.append("Room reverb: not judged (no pauses to listen to the room decay in).")
        else:
            out.append("Room reverb: %s." % ("strong" if r > 0.6 else "moderate" if r > 0.25
                                             else "slight" if r > 0.05 else "none detected"))
    if an.get("rolloff_hz"):
        if an["rolloff_hz"] < BAND_LIMITED_HZ:
            out.append("Band-limited recording (almost nothing above %.1f kHz)." % (an["rolloff_hz"] / 1000.0))
    return out
