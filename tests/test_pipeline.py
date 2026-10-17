"""Behavioural tests for the repair pipeline (~30 s).

Every test measures an effect (waveform error, SNR, loudness, true peak, hum level, report contents).
Module-level tests use small synthetic speech-like signals; the "does it help real speech" tests use the
bench's clean speech clips and its degradations (bench/degrade.py) and are skipped if those are absent.
"""
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from pipeline import process_array, process_with_report, run  # noqa: E402
from pipeline.analysis import analyse  # noqa: E402
from pipeline.declip import declip  # noqa: E402
from pipeline.denoise import denoise  # noqa: E402
from pipeline.master import master, measure_lufs  # noqa: E402
from pipeline.voice_eq import voice_eq  # noqa: E402

SR = 48000


# ----------------------------------------------------------------------------------- helpers
def speechlike(seconds=4.0, sr=SR, seed=0, peak=0.5):
    """Voiced 'syllables' (gliding f0, formant-like harmonic weights) separated by pauses; float32 (1, n)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = np.zeros(n)
    t = 0.2
    while t < seconds - 0.4:
        dur = rng.uniform(0.18, 0.32)
        i0, m = int(t * sr), int(dur * sr)
        f0 = rng.uniform(100, 180) * np.linspace(1.0, rng.uniform(0.85, 1.15), m)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        form = rng.uniform(500, 2500)
        seg = np.zeros(m)
        for h in range(1, 30):
            fh = f0[0] * h
            w = np.exp(-((fh - form) / 700.0) ** 2) + 0.4 / h
            seg += w * np.sin(h * ph + rng.uniform(0, 6.28))
        x[i0:i0 + m] += seg * np.hanning(m) ** 0.7
        t += dur + rng.uniform(0.12, 0.25)
    x = x / np.max(np.abs(x)) * peak
    return x.astype(np.float32)[None, :]


def real_clip(name="david_news", seconds=6.0):
    """A bench clean-speech clip at 48 kHz, peak 0.7, float32 (1, n); skips the test if it is not there."""
    p = ROOT / "bench" / "clean" / f"{name}.wav"
    if not p.exists():
        pytest.skip("bench clean clips not available")
    x, sr = sf.read(p, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = gcd(sr, SR)
    x = resample_poly(x, SR // g, sr // g)
    x = x / np.max(np.abs(x)) * 0.7
    return x[: int(seconds * SR)].astype(np.float32)[None, :]


def si_sdr(ref, est):
    ref = np.asarray(ref, np.float64).ravel()
    est = np.asarray(est, np.float64).ravel()
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    tgt = a * ref
    return 10 * np.log10((np.sum(tgt ** 2) + 1e-20) / (np.sum((est - tgt) ** 2) + 1e-20))


def true_peak_db(x):
    return 20 * np.log10(np.max(np.abs(resample_poly(np.asarray(x, np.float64), 4, 1, axis=-1))) + 1e-12)


def crest(x):
    x = np.asarray(x, np.float64)
    return np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))


def noise_like(n, seed=1):
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    p = np.fft.irfft(X / np.sqrt(f), n)
    p = p / np.std(p)
    return 0.6 * p + 0.4 * rng.standard_normal(n)


def add_noise(x, snr_db, seed=1):
    n = noise_like(x.shape[1], seed)
    n = n / np.std(n) * np.sqrt(np.mean(x ** 2) / 10 ** (snr_db / 10))
    return (x + n[None, :]).astype(np.float32)


def band_db(x, lo, hi):
    f, p = welch(np.asarray(x, np.float64).ravel(), SR, nperseg=16384)
    return 10 * np.log10(p[(f >= lo) & (f <= hi)].sum() + 1e-20)


# ----------------------------------------------------------------------------------- declip
def test_declip_restores_peaks_and_reduces_error():
    orig = speechlike(3.0, peak=0.9)
    clipped = np.clip(orig * 3.0, -0.9, 0.9)                 # ~10 dB of hard clipping
    assert np.mean(np.abs(clipped) >= 0.899) > 0.01          # really clipped (>1% of the samples)
    fixed = declip(clipped)
    before, after = si_sdr(orig, clipped), si_sdr(orig, fixed)
    assert after > before + 8.0, (before, after)             # waveform error falls by > 8 dB
    assert crest(fixed) > 1.3 * crest(clipped)               # flat tops rebuilt into peaks
    assert true_peak_db(fixed) <= -0.99                      # headroom: later stages never hard-clip it again
    assert fixed.shape == clipped.shape


def test_declip_repairs_clipped_real_speech():
    from degrade import hard_clip
    ref = real_clip("david_sibil")
    clipped = np.clip(hard_clip(ref[0] / np.max(np.abs(ref)) * 0.9, 12.0), -1, 1).astype(np.float32)[None]
    fixed = declip(clipped)
    assert si_sdr(ref, fixed) > si_sdr(ref, clipped) + 8.0


def test_declip_leaves_unclipped_audio_alone():
    x = add_noise(speechlike(2.0, peak=0.3), 25.0)
    assert declip(x) is x


# ----------------------------------------------------------------------------------- denoise
def test_denoise_improves_snr_on_tone_like_speech_in_noise(monkeypatch):
    monkeypatch.setenv("AR_DFN", "0")      # synthetic tones are not speech: test the built-in estimator
    clean = speechlike(4.0, peak=0.5, seed=3)
    noisy = add_noise(clean, 5.0)
    out = denoise(noisy, SR, "voice", 0.8)
    assert out.shape == noisy.shape and out.dtype == np.float32
    gain = si_sdr(clean, out) - si_sdr(clean, noisy)
    assert gain > 3.0, gain
    quiet = np.zeros(clean.shape[1], bool)
    quiet[: int(0.15 * SR)] = True                           # leading pause: noise only
    assert 10 * np.log10(np.mean(out[0, quiet] ** 2) + 1e-20) < 10 * np.log10(np.mean(noisy[0, quiet] ** 2)) - 6


def test_neural_denoise_on_real_speech_if_available():
    from pipeline.dfn import dfn_available
    if not dfn_available():
        import pytest
        pytest.skip("DeepFilterNet environment (.venv311) not installed")
    from degrade import add_noise as bench_noise
    ref = real_clip("david_news")
    noisy = bench_noise(ref[0].astype(np.float64), SR, 6.0).astype(np.float32)[None]
    out = denoise(noisy, SR, "voice", 0.8)
    assert out.shape == noisy.shape and np.isfinite(out).all()
    assert si_sdr(ref, out) > si_sdr(ref, noisy) + 4.0


def test_denoise_improves_real_speech_in_noise():
    from degrade import add_noise as bench_noise
    ref = real_clip("zira_story")
    noisy = bench_noise(ref[0].astype(np.float64), SR, 6.0).astype(np.float32)[None]
    out = denoise(noisy, SR, "voice", 0.8)
    assert si_sdr(ref, out) > si_sdr(ref, noisy) + 4.0


def test_denoise_strength_zero_is_identity_and_clean_audio_untouched():
    clean = speechlike(3.0, seed=4)
    noisy = add_noise(clean, 6.0)
    assert denoise(noisy, SR, "voice", 0.0) is noisy
    out = denoise(clean, SR, "voice", 0.8)               # no noise: must not damage the speech
    assert si_sdr(clean, out) > 25.0


def test_denoise_removes_mains_hum():
    clean = speechlike(4.0, seed=5, peak=0.4)
    t = np.arange(clean.shape[1]) / SR
    hum = 0.03 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3))
    x = (clean + hum[None, :]).astype(np.float32)
    out = denoise(x, SR, "voice", 0.8)
    assert band_db(out, 48, 52) < band_db(x, 48, 52) - 15.0


# ----------------------------------------------------------------------------------- tone
def test_voice_eq_removes_hum_and_leaves_real_speech_balanced():
    clean = real_clip("david_news", 8.0)
    t = np.arange(clean.shape[1]) / SR
    hum = 0.01 * np.sin(2 * np.pi * 60 * t) + 0.006 * np.sin(2 * np.pi * 120 * t)
    x = (clean + hum[None, :]).astype(np.float32)
    out = voice_eq(x, SR, "voice")
    assert band_db(out, 58, 62) < band_db(x, 58, 62) - 12.0
    assert abs(band_db(out, 300, 3000) - band_db(clean, 300, 3000)) < 1.5     # speech band level kept
    # a balanced recording is left essentially as it is (dead-bands): waveform nearly unchanged
    same = voice_eq(clean, SR, "voice")
    assert si_sdr(clean, same) > 15.0


def test_voice_eq_fixes_dull_recording():
    from scipy.signal import butter, sosfiltfilt
    clean = speechlike(4.0, seed=7, peak=0.4)
    dull = sosfiltfilt(butter(4, 2200, "lp", fs=SR, output="sos"), clean, axis=-1).astype(np.float32)
    out = voice_eq(dull, SR, "voice")
    assert band_db(out, 2500, 5000) > band_db(dull, 2500, 5000) + 3.0     # presence band lifted


# ----------------------------------------------------------------------------------- master
@pytest.mark.parametrize("amp,mode,target", [(0.05, "voice", -16.0), (0.6, "voice", -16.0), (0.2, "music", -14.0)])
def test_master_hits_target_and_ceiling(amp, mode, target):
    x = speechlike(4.0, seed=8, peak=amp)
    out = master(x, SR, mode)
    assert abs(measure_lufs(out, SR) - target) < 0.5
    assert true_peak_db(out) <= -1.0 + 1e-3
    assert out.shape == x.shape


def test_master_never_overshoots_on_hot_input():
    x = np.clip(speechlike(3.0, seed=9, peak=0.9) * 1.6, -1, 1)
    out = master(x, SR, "voice")
    assert true_peak_db(out) <= -1.0 + 1e-3
    assert np.max(np.abs(out)) <= 1.0


# ----------------------------------------------------------------------------------- analysis
def test_analysis_detects_clipping_noise_and_hum():
    clean = speechlike(4.0, seed=10, peak=0.5)
    a = analyse(clean, SR, "voice")
    assert a["clip_kind"] is None and a["hum_hz"] is None
    c = analyse(np.clip(clean * 8.0, -1, 1), SR, "voice")
    assert c["clip_kind"] is not None and c["clip_pct"] > 1.0
    nz = analyse(add_noise(clean, 5.0), SR, "voice")
    assert nz["dyn_db"] < a["dyn_db"] and nz["presence"] > 0.5
    t = np.arange(clean.shape[1]) / SR
    hum = (0.02 * np.sin(2 * np.pi * 50 * t) + 0.015 * np.sin(2 * np.pi * 100 * t))[None, :].astype(np.float32)
    h = analyse(clean * 0.5 + hum, SR, "voice")
    assert h["hum_hz"] is not None and abs(h["hum_hz"] - 50.0) < 1.0


# ----------------------------------------------------------------------------------- pipeline
def test_pipeline_does_minimal_work_on_clean_speech():
    clean = real_clip("zira_story", 8.0)
    out, rep, an = process_with_report(clean, SR, "voice")
    text = "\n".join(rep)
    assert "Skipped: clipping repair" in text            # nothing clipped -> declip not run
    assert "Skipped: noise reduction" in text            # noise inaudible -> denoise not run
    assert "Applied: mastered to" in text
    assert out.shape == clean.shape
    assert abs(measure_lufs(out, SR) + 16.0) < 0.5
    assert true_peak_db(out) <= -1.0 + 1e-3
    # waveform essentially kept (only level / peak control): compare above 30 Hz, because the TTS clips carry a
    # few percent of sub-audio DC wander that the pipeline's DC / rumble filters rightly remove
    from scipy.signal import butter, sosfiltfilt
    hp = butter(4, 30.0, "hp", fs=SR, output="sos")
    assert si_sdr(sosfiltfilt(hp, clean, axis=-1), sosfiltfilt(hp, out, axis=-1)) > 15.0


@pytest.mark.parametrize("case", ["combined", "noisy", "clipped"])
def test_pipeline_improves_degraded_real_speech(case):
    from degrade import CASES
    ref = real_clip("david_news", 8.0)
    deg = np.clip(CASES[case](ref[0].astype(np.float64), SR), -1, 1).astype(np.float32)[None]
    rep = []
    out = process_array(deg, SR, "voice", {"report": rep})
    assert out.shape == deg.shape and out.dtype == np.float32
    assert si_sdr(ref, out) > si_sdr(ref, deg) + 2.0     # closer to the clean original
    assert abs(measure_lufs(out, SR) + 16.0) < 0.5
    assert true_peak_db(out) <= -1.0 + 1e-3
    text = "\n".join(rep)
    if case in ("combined", "clipped"):
        assert "Applied: clipping repair" in text
    if case in ("combined", "noisy"):
        assert "Applied: noise reduction" in text


def test_pipeline_options_are_respected():
    x = np.clip(add_noise(speechlike(3.0, seed=13, peak=0.9) * 2.0, 10.0), -1, 1)
    off = process_array(x, SR, "voice", {"declip": False, "denoise": False, "eq": False, "master": False})
    assert np.array_equal(off, np.clip(x, -1, 1).astype(np.float32))        # everything off = untouched
    only_master = process_array(x, SR, "voice", {"declip": False, "denoise": False, "eq": False})
    assert abs(measure_lufs(only_master, SR) + 16.0) < 0.5
    forced = []
    process_array(x, SR, "voice", {"denoise_strength": 0.5, "report": forced, "eq": False, "master": False,
                                   "declip": False})
    assert any("strength 50%" in s and "auto" not in s for s in forced)
    with pytest.raises(ValueError):
        process_array(x, SR, "voice", {"order": ("declip", "bogus")})


def test_run_end_to_end_file(tmp_path):
    clean = real_clip("david_sibil", 5.0)
    x = np.clip(add_noise(clean * 1.4, 12.0), -1, 1)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x.T.astype(np.float32), SR)
    rep = []
    run(str(src), str(dst), "voice", {"denoise_strength": 0.7, "report": rep})
    y, sr = sf.read(dst, dtype="float32")
    assert sr == SR and len(y) == x.shape[1]            # same length, same rate
    assert np.max(np.abs(y)) <= 1.0
    assert abs(measure_lufs(y[None, :], SR) + 16.0) < 1.0   # PCM16 rounding included
    assert rep, "the report must be filled"


def test_stereo_is_kept_aligned_and_balanced():
    a = speechlike(3.0, seed=15, peak=0.4)
    st = np.concatenate([a, 0.5 * a], axis=0)           # right channel is 6 dB lower
    out = process_array(st, SR, "voice", {"denoise": False})
    assert out.shape == st.shape
    ratio = np.sqrt(np.mean(out[1] ** 2) / np.mean(out[0] ** 2))
    assert 0.4 < ratio < 0.6                            # image preserved (about 0.5)


# ----------------------------------------------------------------- click removal and bump control (real-recording findings)
def _clicks_on(ref, seed=7, per_s=1.5):
    rng = np.random.default_rng(seed)
    y = ref.copy()
    for p in np.sort(rng.integers(SR, ref.shape[-1] - SR, int(ref.shape[-1] / SR * per_s))):
        w = int(rng.integers(24, 200))
        t = np.arange(w)
        y[0, p:p + w] += rng.uniform(0.6, 1.4) * rng.choice([-1, 1]) * np.exp(-t / (w / 3)) * np.cos(2 * np.pi * rng.uniform(0.05, 0.3) * t)
    return y


def test_declick_repairs_clicks_and_leaves_clean_speech_alone():
    from pipeline.declick import declick
    ref = real_clip("david_news")
    same, n = declick(ref, SR)
    assert n == 0 and same is ref                            # no false alarms on clean speech
    y = _clicks_on(ref)
    out, n = declick(y.astype(np.float32), SR)
    assert n >= 5
    assert si_sdr(ref, out) > si_sdr(ref, y) + 3.0
    assert np.abs(out).max() < 1.3                           # no runaway AR extrapolation


def test_tame_turns_down_bangs_but_not_speech():
    from pipeline.tame import tame
    ref = real_clip("zira_story")
    same, n, cut = tame(ref, SR)
    assert n == 0 and same is ref                            # ordinary speech and plosives are not touched
    y = ref.copy()
    rng = np.random.default_rng(1)
    sp = float(np.sqrt(np.mean(ref ** 2)))
    for p in (int(1.5 * SR), int(3.5 * SR)):
        w = int(0.08 * SR)
        y[0, p:p + w] += sp * 100 * np.exp(-np.arange(w) / (w / 3)) * rng.standard_normal(w) * 0.5
    out, n, cut = tame(y.astype(np.float32), SR)
    assert n == 2 and cut > 10
    assert si_sdr(ref, out) > si_sdr(ref, y) + 3.0


def test_pipeline_reports_click_and_bump_stages():
    ref = real_clip("david_news")
    y = _clicks_on(ref).astype(np.float32)
    rep = []
    process_array(y, SR, "voice", {"report": rep, "denoise": False})
    assert any("click removal" in line and "repaired" in line for line in rep)
