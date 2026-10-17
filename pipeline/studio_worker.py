"""Studio voice: generative voice restoration. Runs inside the Python 3.11 environment (.venv311), which has PyTorch.

    python studio_worker.py in.wav out.wav [nfe] [lambd] [tau] [blend] [threads]

Pipeline (the approach of CrispVoice, https://github.com/Francium-Tech/CrispVoice, MIT):
  1. Resemble Enhance (https://github.com/resemble-ai/resemble-enhance, MIT): a denoiser plus a generative enhancer
     that re-synthesises the voice at 44.1 kHz, restoring body and clarity a poor microphone never captured.
  2. A phase-preserving DeepFilterNet-denoised copy of the original is blended back in (default 25 %): re-synthesis
     alone sounds subtly synthetic; the blend restores natural transients, breaths and room decay.
Writes a mono 44.1 kHz float WAV, the same length as the input in time. Prints "PROGRESS i total" lines.
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

inp, outp = sys.argv[1], sys.argv[2]
nfe = int(sys.argv[3]) if len(sys.argv) > 3 else 32
lambd = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
tau = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
blend = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
threads = int(sys.argv[7]) if len(sys.argv) > 7 else max(1, (os.cpu_count() or 4) // 2)
torch.set_num_threads(threads)

run_dir = Path(os.environ.get("AR_MODELS") or Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local/share")
               / "AudioRepair" / "models") / "enhancer_stage2"

from torchaudio.functional import resample  # noqa: E402


def to_rate(x, a, b):
    return x if a == b else resample(x, orig_freq=a, new_freq=b, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                     resampling_method="sinc_interp_kaiser", beta=14.769656459379492)


audio, sr = sf.read(inp, dtype="float32", always_2d=True)
mono = torch.from_numpy(np.ascontiguousarray(audio.mean(axis=1)))
n_in = len(mono)

# --- 1. generative restoration (Resemble Enhance), chunked to bound memory
from resemble_enhance.enhancer.inference import load_enhancer  # noqa: E402
from resemble_enhance.inference import inference_chunk, merge_chunks, remove_weight_norm_recursively  # noqa: E402

enhancer = load_enhancer(run_dir, "cpu")
enhancer.configurate_(nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)
remove_weight_norm_recursively(enhancer)
hp_rate = enhancer.hp.wav_rate
dwav = to_rate(mono, sr, hp_rate)
chunk_len = int(hp_rate * 10.0)
overlap = int(hp_rate * 1.0)
hop = chunk_len - overlap
starts = list(range(0, dwav.shape[-1], hop))
chunks = []
t0 = time.time()
with torch.no_grad():
    for i, s in enumerate(starts, 1):
        chunks.append(inference_chunk(enhancer, dwav[s:s + chunk_len], hp_rate, "cpu"))
        gc.collect()
        print("PROGRESS %d %d" % (i, len(starts) + 1), flush=True)
hwav = merge_chunks(chunks, chunk_len, hop, sr=hp_rate, length=dwav.shape[-1]).cpu()
del enhancer, chunks
gc.collect()

# --- 2. blend in the phase-preserving DeepFilterNet denoise of the original
if blend > 0:
    try:
        from df.enhance import enhance, init_df
        model, state, _ = init_df(log_level="ERROR")
        x48 = to_rate(mono, sr, state.sr())
        e = enhance(model, state, (x48 * 0.5).unsqueeze(0)).squeeze(0) * 2.0      # 6 dB headroom, as CrispVoice does
        e = to_rate(e, state.sr(), hp_rate)
        m = min(len(hwav), len(e))
        hwav = (1 - blend) * hwav[:m] + blend * e[:m]
    except Exception as exc:                                                       # blend is a refinement, never fatal
        print("note: blend skipped (%s)" % exc, flush=True)
print("PROGRESS %d %d" % (len(starts) + 1, len(starts) + 1), flush=True)

# the model can emit peaks above full scale; normalise before anything encodes them
peak = float(hwav.abs().max())
if peak > 0.98:
    hwav = hwav * (0.98 / peak)
sf.write(outp, hwav.numpy().astype(np.float32), hp_rate, subtype="FLOAT")
