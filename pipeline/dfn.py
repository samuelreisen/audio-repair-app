"""Optional DeepFilterNet3 neural enhancement.

PyTorch has no Python 3.14 build yet, so DeepFilterNet lives in a separate Python 3.11 environment
(``.venv311`` in the project folder, or the interpreter named by the DFN_PYTHON environment variable) and is
reached through a subprocess.  If that environment is missing, ``dfn_available()`` is False and the app
silently uses its built-in denoiser instead.  Set AR_DFN=0 to force the built-in path (used for A/B benchmarks).
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = Path(__file__).with_name("dfn_worker.py")


def _python():
    env = os.environ.get("DFN_PYTHON")
    cand = Path(env) if env else _ROOT / ".venv311" / "Scripts" / "python.exe"
    if not cand.exists():
        cand = _ROOT / ".venv311" / "bin" / "python"
    return cand if cand.exists() else None


def dfn_available():
    from . import ai
    return os.environ.get("AR_DFN", "1") != "0" and (ai.inprocess() or _python() is not None)


def dfn_enhance(audio, sr):
    """audio: float (channels, samples) at 48 kHz -> enhanced float32 array, same shape.  Raises on failure."""
    if sr != 48000:
        raise ValueError("DeepFilterNet path expects 48 kHz")
    from . import ai
    if ai.inprocess():                                         # packaged app: model runs right here
        return ai.dfn_enhance_inproc(audio)
    py = _python()
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "i.wav"), os.path.join(td, "o.wav")
        sf.write(i, np.asarray(audio, np.float32).T, sr, subtype="FLOAT")
        r = subprocess.run([str(py), str(_WORKER), i, o], capture_output=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError("DeepFilterNet worker failed: " + r.stderr.decode(errors="replace")[-300:])
        y, _ = sf.read(o, dtype="float32", always_2d=True)
    return y.T
