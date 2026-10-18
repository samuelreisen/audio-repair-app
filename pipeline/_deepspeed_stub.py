"""A stand-in for the `deepspeed` library, so Studio voice (Resemble Enhance) can be imported without it.

Resemble Enhance imports DeepSpeed at start-up because it also contains its training code. Studio voice only runs the
finished model, which never touches DeepSpeed, but the import itself must succeed. The real library is large, needs a
compiler to install and does not run on Windows, so the app leaves it out (Resemble Enhance is installed `--no-deps`)
and registers this stand-in instead. Anything that would actually train raises a clear error.

Without it the packaged app stopped Studio voice with "ModuleNotFoundError: No module named 'deepspeed'" (the
GitHub-built 1.1.0 installers shipped that way; a local environment had a hand-made copy of this file, which hid it).
"""
import sys
import types

_MSG = "deepspeed is not installed: training is not supported in Audio Repair (only the finished model is used)"


def _unsupported(*args, **kwargs):
    raise RuntimeError(_MSG)


class DeepSpeedConfig:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(_MSG)


class DeepSpeedEngine:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(_MSG)


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__file__ = "<audio-repair stand-in for deepspeed>"
    return m


def install():
    """Register the stand-in unless a real deepspeed can be imported. Returns True when the stand-in is in use."""
    if sys.modules.get("deepspeed") is not None:
        return getattr(sys.modules["deepspeed"], "__audio_repair_stub__", False)
    try:
        import deepspeed  # noqa: F401
        return False
    except Exception:
        pass
    root = _module("deepspeed", __version__="0.0.0-stand-in", __audio_repair_stub__=True, DeepSpeedConfig=DeepSpeedConfig,
                   init_distributed=_unsupported, __path__=[])
    accel = _module("deepspeed.accelerator", get_accelerator=_unsupported)
    runtime = _module("deepspeed.runtime", __path__=[])
    engine = _module("deepspeed.runtime.engine", DeepSpeedEngine=DeepSpeedEngine)
    utils = _module("deepspeed.runtime.utils", clip_grad_norm_=_unsupported)
    root.accelerator, root.runtime = accel, runtime
    runtime.engine, runtime.utils = engine, utils
    for m in (root, accel, runtime, engine, utils):
        sys.modules[m.__name__] = m
    return True
