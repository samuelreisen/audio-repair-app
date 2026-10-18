"""Tone changes after the repair (pipeline/tone_after.py) and the engine tone job (engine.py)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import tone_after  # noqa: E402

SR = 48000


def wait(engine, jid, limit=600):
    for _ in range(limit):
        s = engine.status(jid)
        if s["state"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def band_db(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-30)


def sine(f, sr=SR, secs=1.0, amp=0.1):
    return (amp * np.sin(2 * np.pi * f * np.arange(int(sr * secs)) / sr)).astype(np.float32)


# ---------- clean_tone / is_flat ----------

def test_clean_tone_keeps_only_the_four_bands():
    out = tone_after.clean_tone({"warmth": 3, "bogus": 5, "volume": 2})
    assert set(out) == {"warmth", "clarity", "brightness", "air"}
    assert out["warmth"] == 3.0
    assert out["clarity"] == 0.0 and out["brightness"] == 0.0 and out["air"] == 0.0


def test_clean_tone_junk_becomes_zero():
    out = tone_after.clean_tone({"warmth": "loud", "clarity": None, "brightness": float("nan"), "air": float("inf")})
    assert out == {"warmth": 0.0, "clarity": 0.0, "brightness": 0.0, "air": 0.0}
    assert tone_after.clean_tone(None) == {"warmth": 0.0, "clarity": 0.0, "brightness": 0.0, "air": 0.0}


def test_clean_tone_clamps_to_plus_minus_8():
    out = tone_after.clean_tone({"air": 50, "warmth": -50, "clarity": 8.0, "brightness": -8.0})
    assert out == {"warmth": -8.0, "clarity": 8.0, "brightness": -8.0, "air": 8.0}


def test_is_flat():
    assert tone_after.is_flat({}) is True
    assert tone_after.is_flat({"brightness": 3}) is False
    assert tone_after.is_flat({"brightness": 0.01}) is True


# ---------- apply_tone ----------

def test_flat_tone_returns_equal_copy():
    x = (0.3 * np.random.default_rng(0).standard_normal(4800)).astype(np.float32)
    out = tone_after.apply_tone(x, SR, {})
    assert out is not x
    assert out.dtype == np.float32 and out.shape == x.shape
    assert np.array_equal(out, x)
    x2 = np.stack([x, -x])
    out2 = tone_after.apply_tone(x2, SR, {"warmth": 0.01})
    assert out2 is not x2 and out2.shape == x2.shape and np.array_equal(out2, x2)


def test_apply_tone_keeps_shape_and_dtype_1d_and_2d():
    x = (0.2 * np.random.default_rng(2).standard_normal(SR)).astype(np.float32)
    for inp in (x, np.stack([x, x * 0.5])):
        out = tone_after.apply_tone(inp, SR, {"warmth": 4, "clarity": -2, "brightness": 3, "air": 5})
        assert out.dtype == np.float32
        assert out.shape == inp.shape


# ---------- response_db ----------

def test_brightness_response():
    r = tone_after.response_db({"brightness": 6}, SR, [200, 12000])
    assert abs(r[1] - 6.0) <= 1.5
    assert abs(r[0]) <= 0.5


def test_warmth_response():
    r = tone_after.response_db({"warmth": 6}, SR, [50, 8000])
    assert abs(r[0] - 6.0) <= 1.5
    assert abs(r[1]) <= 0.5


def test_clarity_response():
    r = tone_after.response_db({"clarity": 6}, SR, [3000, 100, 10000])
    assert abs(r[0] - 6.0) <= 0.3
    assert abs(r[1]) <= 1.0
    assert abs(r[2]) <= 1.0


def test_negative_gain_flips_sign_and_flat_is_zero():
    neg = tone_after.response_db({"brightness": -6}, SR, [12000])
    pos = tone_after.response_db({"brightness": 6}, SR, [12000])
    assert neg[0] < 0 < pos[0]
    assert np.allclose(tone_after.response_db({}, SR, [100, 1000, 10000]), 0.0)


# ---------- apply_tone really filters ----------

def test_brightness_filters_noise():
    x = (0.1 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32)
    y = tone_after.apply_tone(x, SR, {"brightness": 6})
    hi_delta = band_db(y.astype(np.float64), SR, 6000, SR / 2) - band_db(x.astype(np.float64), SR, 6000, SR / 2)
    lo_delta = band_db(y.astype(np.float64), SR, 0, 1000) - band_db(x.astype(np.float64), SR, 0, 1000)
    assert 3.0 <= hi_delta <= 9.0, hi_delta
    assert abs(lo_delta) < 1.0, lo_delta


@pytest.mark.parametrize("f", [1000.0, 8000.0])
def test_measured_gain_matches_response(f):
    tone = {"warmth": 3, "clarity": -4, "brightness": 6, "air": 2}
    x = sine(f, secs=1.0)
    y = tone_after.apply_tone(x, SR, tone)
    half = len(x) // 2
    measured = 20 * np.log10(np.sqrt(np.mean(y[half:].astype(np.float64) ** 2))
                             / np.sqrt(np.mean(x[half:].astype(np.float64) ** 2)))
    expected = tone_after.response_db(tone, SR, [f])[0]
    assert abs(measured - expected) <= 0.5, (f, measured, expected)


# ---------- engine tone job ----------

@pytest.fixture
def voice_wav(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(3 * SR) / SR
    x = 0.07 * np.sin(2 * np.pi * 150 * t) + 0.07 * np.sin(2 * np.pi * 2000 * t) + 0.07 * np.sin(2 * np.pi * 5000 * t)
    x = (x + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    p = tmp_path / "voice.wav"
    sf.write(p, x, SR)
    return p


@pytest.fixture
def engine_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_REPAIR_DATA_DIR", str(tmp_path / "data"))
    import engine
    return engine


def test_tone_job_on_repaired_file(tmp_path, engine_env, voice_wav):
    engine = engine_env
    from pipeline.master import measure_lufs

    jid = engine.start(str(voice_wav))
    assert wait(engine, jid)["state"] == "done"

    tid = engine.start(str(voice_wav), base_job=jid, tone={"brightness": 5, "warmth": -3})
    s = wait(engine, tid)
    assert s["state"] == "done", s
    assert any(line.startswith("Applied: tone changed after the repair") for line in s["report"])
    assert any(line.startswith("Applied: loudness matched") for line in s["report"])

    out_p = engine.output_path(tid)
    assert out_p is not None and Path(out_p[0]).exists()
    assert out_p[1].endswith("_fixed.wav")
    base_p = engine.output_path(jid)[0]

    toned, sr_t = sf.read(out_p[0], dtype="float64", always_2d=True)
    plain, sr_p = sf.read(base_p, dtype="float64", always_2d=True)
    assert sr_t == sr_p
    assert toned.shape == plain.shape                       # same number of samples
    toned, plain = toned[:, 0], plain[:, 0]

    def hi_share(v):
        return band_db(v, sr_t, 4000, sr_t / 2) - band_db(v, sr_t, 0, sr_t / 2)
    assert hi_share(toned) - hi_share(plain) >= 2.0, (hi_share(toned), hi_share(plain))

    lufs = measure_lufs(toned[None, :], sr_t)
    assert abs(lufs - (-16.0)) <= 1.0, lufs


def test_tone_job_unknown_base_raises(engine_env, voice_wav):
    from pipeline.io import AudioError
    with pytest.raises(AudioError):
        engine_env.start(str(voice_wav), base_job="does-not-exist", tone={"brightness": 3})


def test_mp3_repair_then_wav_tone(engine_env, voice_wav):
    engine = engine_env
    if not engine.mp3_supported():
        pytest.skip("MP3 export not available here")
    jid = engine.start(str(voice_wav), fmt="mp3")
    assert wait(engine, jid)["state"] == "done"
    tid = engine.start(str(voice_wav), base_job=jid, fmt="wav", tone={"air": 4})
    s = wait(engine, tid)
    assert s["state"] == "done", s
    assert engine.output_path(tid) is not None


def test_tone_job_does_not_rerun_repair(engine_env, voice_wav, monkeypatch):
    engine = engine_env
    jid = engine.start(str(voice_wav))
    assert wait(engine, jid)["state"] == "done"

    def no_repair(*a, **k):
        raise AssertionError("the repair must not run again for a tone change")
    monkeypatch.setattr(engine, "run", no_repair)
    tid = engine.start(str(voice_wav), base_job=jid, tone={"clarity": 2})
    s = wait(engine, tid)
    assert s["state"] == "done", s
