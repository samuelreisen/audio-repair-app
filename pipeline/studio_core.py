"""Studio voice restoration (Resemble Enhance + a DeepFilterNet blend), callable in-process or from studio_worker.py.

The approach of CrispVoice (https://github.com/Francium-Tech/CrispVoice, MIT):
  1. Resemble Enhance (https://github.com/resemble-ai/resemble-enhance, MIT): a denoiser plus a generative enhancer
     that re-synthesises the voice at 44.1 kHz, restoring body and clarity a poor microphone never captured.
  2. A phase-preserving DeepFilterNet-denoised copy of the original is blended back in (default 25 %): re-synthesis
     alone sounds subtly synthetic; the blend restores natural transients, breaths and room decay.
"""
import gc
import os
import threading

import numpy as np

try:
    from . import ai
except ImportError:                      # run as a plain script from pipeline/ (studio_worker.py)
    import ai

_lock = threading.Lock()                 # one restoration at a time: the model is large


def _to_rate(x, a, b):
    from torchaudio.functional import resample
    return x if a == b else resample(x, orig_freq=a, new_freq=b, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                     resampling_method="sinc_interp_kaiser", beta=14.769656459379492)


def restore(mono, sr, run_dir, nfe=32, lambd=0.4, tau=0.5, blend=0.25, threads=None, progress=None):
    """mono: float32 1-D array at sr.  Returns (restored float32 1-D array, its sample rate 44100).

    progress(done, total) is called after every 10-second chunk."""
    import torch
    from resemble_enhance.enhancer.inference import load_enhancer
    from resemble_enhance.inference import inference_chunk, merge_chunks, remove_weight_norm_recursively

    with _lock:
        torch.set_num_threads(threads or max(1, (os.cpu_count() or 4) // 2))
        x = torch.from_numpy(np.ascontiguousarray(np.asarray(mono, np.float32)))

        # --- 1. generative restoration, chunked to bound memory
        enhancer = load_enhancer(run_dir, "cpu")
        enhancer.configurate_(nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)
        remove_weight_norm_recursively(enhancer)
        hp_rate = enhancer.hp.wav_rate
        dwav = _to_rate(x, sr, hp_rate)
        chunk_len = int(hp_rate * 10.0)
        hop = chunk_len - int(hp_rate * 1.0)
        starts = list(range(0, dwav.shape[-1], hop))
        chunks = []
        with torch.no_grad():
            for i, s in enumerate(starts, 1):
                chunks.append(inference_chunk(enhancer, dwav[s:s + chunk_len], hp_rate, "cpu"))
                gc.collect()
                if progress:
                    progress(i, len(starts) + 1)
        hwav = merge_chunks(chunks, chunk_len, hop, sr=hp_rate, length=dwav.shape[-1]).cpu()
        del enhancer, chunks
        gc.collect()

        # --- 2. blend in the phase-preserving DeepFilterNet denoise of the original
        if blend > 0:
            try:
                from df.enhance import enhance
                model, state = ai.dfn_model()
                x48 = _to_rate(x, sr, state.sr())
                with torch.no_grad():
                    e = enhance(model, state, (x48 * 0.5).unsqueeze(0)).squeeze(0) * 2.0   # 6 dB headroom
                e = _to_rate(e, state.sr(), hp_rate)
                m = min(len(hwav), len(e))
                hwav = (1 - blend) * hwav[:m] + blend * e[:m]
            except Exception as exc:                                                   # a refinement, never fatal
                print("note: blend skipped (%s)" % exc, flush=True)
        if progress:
            progress(len(starts) + 1, len(starts) + 1)

        peak = float(hwav.abs().max())          # the model can emit peaks above full scale
        if peak > 0.98:
            hwav = hwav * (0.98 / peak)
        return hwav.numpy().astype(np.float32), hp_rate
