"""Where the AI models live and whether they can run inside this process.

The installed desktop app is built with PyTorch, DeepFilterNet and Resemble Enhance included and their weights
shipped in a ``models`` folder next to the code, so everything works offline from the first launch. When the app runs
from the source folder on a Python without PyTorch, the models are reached through the separate ``.venv311``
environment instead (see dfn.py and studio.py).
"""
import os
import sys
import threading
from pathlib import Path

_lock = threading.Lock()
_inproc = None


def bundled_models():
    """The models folder shipped inside the packaged app, or None when running from source."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = Path(base) / "models"
        if p.is_dir():
            return p
    return None


def user_models():
    """Where a source / developer setup keeps downloaded weights (or AR_MODELS)."""
    return Path(os.environ.get("AR_MODELS") or (Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
                                                / "AudioRepair" / "models"))


def models_root():
    return bundled_models() or user_models()


def _shim_torchaudio():
    """DeepFilterNet 0.5.6 imports torchaudio.backend.common, which newer torchaudio moved; give it an alias."""
    try:
        import torchaudio.backend.common  # noqa: F401
    except Exception:
        try:
            import types
            import torchaudio
            mod = types.ModuleType("torchaudio.backend.common")
            mod.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
            sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
            sys.modules["torchaudio.backend.common"] = mod
        except Exception:
            pass


def inprocess():
    """True when PyTorch and the model code can be imported here (the packaged app, or a Python 3.11 source run)."""
    global _inproc
    if _inproc is None:
        with _lock:
            if _inproc is None:
                try:
                    import importlib.util
                    ok = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchaudio", "df"))
                    if ok:
                        import torch  # noqa: F401
                        _shim_torchaudio()
                    _inproc = bool(ok)
                except Exception:
                    _inproc = False
    return _inproc


# ------------------------------------------------------------------ DeepFilterNet3, loaded once per process
_dfn = None
_dfn_lock = threading.Lock()


def dfn_model():
    global _dfn
    with _dfn_lock:
        if _dfn is None:
            _shim_torchaudio()
            from df.enhance import init_df
            d = models_root() / "DeepFilterNet3"
            if (d / "config.ini").exists():
                model, state, _ = init_df(model_base_dir=str(d), log_level="ERROR")
            else:
                model, state, _ = init_df(log_level="ERROR")       # developer setup: DeepFilterNet's own cache
            _dfn = (model, state)
    return _dfn


def dfn_enhance_inproc(audio):
    """audio float (channels, samples) at 48 kHz -> enhanced float32, same shape."""
    import numpy as np
    import torch
    from df.enhance import enhance
    model, state = dfn_model()
    a = np.asarray(audio, np.float32)
    n = a.shape[1]
    outs = []
    with _dfn_lock, torch.no_grad():
        for ch in a:
            e = enhance(model, state, torch.from_numpy(np.ascontiguousarray(ch)).unsqueeze(0)).squeeze(0).numpy()
            outs.append(np.pad(e, (0, max(0, n - len(e))))[:n])
    return np.stack(outs).astype(np.float32)
