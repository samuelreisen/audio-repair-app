"""Advanced error report (engine.py) and the Studio voice settings-file fix (pipeline/studio_core.py)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "in.wav"
    sf.write(p, (0.1 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)).astype(np.float32), 48000)
    return p


def wait(engine, jid):
    for _ in range(100):
        s = engine.status(jid)
        if s["state"] != "running":
            return s
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_failed_repair_reports_details(tmp_path, monkeypatch, wav):
    monkeypatch.setenv("AUDIO_REPAIR_DATA_DIR", str(tmp_path / "data"))
    import engine

    def boom(src, out, mode, opts, progress):
        progress(0.3, "Rebuilding the voice (AI)")
        opts["report"].append("Applied: click removal")
        raise NotImplementedError("cannot instantiate 'PosixPath' on your system")
    monkeypatch.setattr(engine, "run", boom)
    s = wait(engine, engine.start(str(wav), studio=True))
    assert s["state"] == "error"
    assert "NotImplementedError" in s["message"] and "Rebuilding the voice (AI)" in s["message"]
    d = s["detail"]
    for needle in ("Audio Repair error report", "Version:", "in.wav", "studio=True", "Stopped at: Rebuilding the voice (AI)",
                   "Applied: click removal", "NotImplementedError: cannot instantiate", "Traceback"):
        assert needle in d, needle
    assert str(tmp_path) not in d.split("Traceback")[0]          # the header holds the file name, never its folder
    log = Path(s["log"])
    assert log.is_file() and "NotImplementedError" in log.read_text(encoding="utf-8")


def test_normal_errors_have_no_details(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_REPAIR_DATA_DIR", str(tmp_path / "data"))
    import engine
    from pipeline.io import AudioError
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    s = wait(engine, engine.start(str(bad)))
    assert s["state"] == "error" and "detail" not in s            # a plain AudioError already says what is wrong


def test_log_is_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_REPAIR_DATA_DIR", str(tmp_path / "data"))
    import engine
    for _ in range(40):
        engine._log_error("x" * 10000)
    assert Path(engine._log_path()).stat().st_size < 260000


LINUX_YAML = """fg_dir: !!python/object/apply:pathlib.PosixPath
- data
- fg
bg_dir: !!python/object/apply:pathlib.PosixPath
- data
- bg
rir_dir: !!python/object/apply:pathlib.PosixPath
- data
- rir
load_fg_only: false
wav_rate: 44100
"""


def test_posix_paths_in_model_settings_are_made_plain():
    from pipeline.studio_core import _plain_paths
    for text in (LINUX_YAML, LINUX_YAML.replace("\n", "\r\n")):
        out = _plain_paths(text)
        assert "PosixPath" not in out and "pathlib" not in out
        assert "'data/fg'" in out and "'data/rir'" in out
        assert "load_fg_only: false" in out and "wav_rate: 44100" in out
    plain = "fg_dir: data/fg\nwav_rate: 44100\n"
    assert _plain_paths(plain) == plain                           # settings that are already plain are left alone


def test_settings_file_loads_through_the_enhancer_class(tmp_path):
    pytest.importorskip("resemble_enhance")
    from pipeline import studio_core
    from resemble_enhance.enhancer.hparams import HParams
    studio_core._patch_hparams_loader()
    f = tmp_path / "hparams.yaml"
    f.write_text(LINUX_YAML, encoding="utf-8")
    assert HParams.from_yaml(f).wav_rate == 44100                 # raised NotImplementedError on Windows before


STUB_CHECK = """
import sys
sys.modules["deepspeed"] = None                      # as if deepspeed were not installed
sys.path.insert(0, sys.argv[1])
from pipeline import _deepspeed_stub
assert _deepspeed_stub.install() is True
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.utils import clip_grad_norm_
from resemble_enhance.enhancer import enhancer           # raised ModuleNotFoundError: deepspeed in the packaged app
from resemble_enhance.denoiser.inference import load_denoiser
try:
    deepspeed.DeepSpeedConfig({})
except RuntimeError as e:
    assert "training is not supported" in str(e)
else:
    raise SystemExit("training should be refused")
print("stand-in ok")
"""


def test_deepspeed_stand_in_lets_studio_voice_import(tmp_path):
    pytest.importorskip("resemble_enhance")
    import subprocess
    out = subprocess.run([sys.executable, "-c", STUB_CHECK, str(ROOT)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "stand-in ok" in out.stdout, out.stderr[-1500:]


def test_a_real_deepspeed_is_left_alone(monkeypatch):
    import types
    from pipeline import _deepspeed_stub
    fake = types.ModuleType("deepspeed")
    monkeypatch.setitem(sys.modules, "deepspeed", fake)
    assert _deepspeed_stub.install() is False and sys.modules["deepspeed"] is fake
