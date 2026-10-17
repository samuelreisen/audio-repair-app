"""Optional "Studio voice" stage: generative voice restoration (Resemble Enhance + a DeepFilterNet blend).

It runs in the separate Python 3.11 environment (``.venv311``, or the interpreter named by STUDIO_PYTHON / DFN_PYTHON)
because PyTorch has no Python 3.14 build yet, and it needs the Resemble Enhance model weights (about 713 MB) in
``%LOCALAPPDATA%/AudioRepair/models`` (or the folder named by AR_MODELS).  If either is missing, ``studio_available()``
is False and the app carries on with its normal pipeline.  Set AR_STUDIO=0 to switch it off.

It is slow on a CPU (several times the length of the recording) and only suits spoken voice.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parents[1]
_WORKER = Path(__file__).with_name("studio_worker.py")


def _python():
    env = os.environ.get("STUDIO_PYTHON") or os.environ.get("DFN_PYTHON")
    cand = Path(env) if env else _ROOT / ".venv311" / "Scripts" / "python.exe"
    if not cand.exists():
        cand = _ROOT / ".venv311" / "bin" / "python"
    return cand if cand.exists() else None


def models_dir():
    """Shipped inside the packaged app, else the developer download folder (or AR_MODELS)."""
    from . import ai
    return ai.models_root() / "enhancer_stage2"


def _has_resemble():
    import importlib.util
    try:
        return importlib.util.find_spec("resemble_enhance") is not None
    except Exception:
        return False


def studio_available():
    from . import ai
    if os.environ.get("AR_STUDIO", "1") == "0":
        return False
    if not (ai.inprocess() and _has_resemble()) and _python() is None:
        return False
    m = models_dir()
    w = m / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    return (m / "hparams.yaml").exists() and w.exists() and w.stat().st_size >= 713_000_000   # not a partial download


def studio_enhance(audio, sr, progress=None, nfe=32, blend=0.25):
    """audio: float (channels, samples). Returns float32 (channels, samples) at the same rate and length.

    The restored voice is mono, so a stereo recording comes back as two identical channels.
    progress(fraction 0..1) is called as the model works through the recording."""
    a = np.asarray(audio, np.float32)
    n_in = a.shape[1]
    a = _level_for_model(a)
    from . import ai
    if ai.inprocess() and _has_resemble():                     # packaged app: model runs right here
        from .studio_core import restore
        y, ysr = restore(a.mean(axis=0), sr, models_dir(), nfe=nfe, blend=blend,
                         progress=(lambda d, t: progress(d / max(t, 1))) if progress else None)
        return _fit(y, ysr, sr, n_in, a.shape[0])
    py = _python()
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "in.wav"), os.path.join(td, "out.wav")
        sf.write(i, a.mean(axis=0), sr, subtype="FLOAT")
        env = {**os.environ, "AR_MODELS": str(models_dir().parent)}
        p = subprocess.Popen([str(py), str(_WORKER), i, o, str(nfe), "0.4", "0.5", str(blend)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for line in p.stdout:
            if line.startswith("PROGRESS") and progress:
                _, done, total = line.split()
                progress(int(done) / max(int(total), 1))
        err = p.stderr.read()
        if p.wait() != 0 or not os.path.exists(o):
            raise RuntimeError("Studio voice worker failed: " + err[-400:])
        y, ysr = sf.read(o, dtype="float32")
    return _fit(y, ysr, sr, n_in, a.shape[0])


def _level_for_model(a):
    """The model was trained on normally levelled speech and does much worse on quiet input (for example after
    clipping repair, which leaves ~10-15 dB of headroom). Bring the speech to about -20 dBFS RMS, peaks below 0.95.
    Mastering sets the final level afterwards, so this gain is not undone."""
    m = a.mean(axis=0)
    n = 2400
    k = len(m) // n
    if k >= 4:
        fr = np.sqrt(np.mean(m[:k * n].reshape(k, n) ** 2, axis=1))
        speech = float(np.sqrt(np.mean(np.square(fr[fr >= np.percentile(fr, 50)])))) if fr.any() else 0.0
    else:
        speech = float(np.sqrt(np.mean(m ** 2)))
    peak = float(np.abs(a).max())
    if speech <= 1e-6 or peak <= 0:
        return a
    g = min(10 ** (-20 / 20) / speech, 0.95 / peak)
    return (a * g).astype(np.float32)


def _fit(y, ysr, sr, n_in, channels):
    """Back to the input rate and length; mono restoration copied to every channel."""
    if ysr != sr:
        from math import gcd
        g = gcd(ysr, sr)
        y = resample_poly(y, sr // g, ysr // g).astype(np.float32)
    y = np.pad(y, (0, max(0, n_in - len(y))))[:n_in]
    return np.tile(y[None, :], (channels, 1)).astype(np.float32)
