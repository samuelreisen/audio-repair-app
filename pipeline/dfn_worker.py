"""Runs inside the Python 3.11 environment (.venv311) that has PyTorch + DeepFilterNet installed.

    python dfn_worker.py in.wav out.wav

Enhances every channel of a 48 kHz WAV with DeepFilterNet3 (noise suppression, some de-reverb)
and writes a same-length WAV.  The main app (Python 3.14, which PyTorch does not support yet) calls this
as a subprocess; see pipeline/dfn.py.
"""
import sys
import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance, init_df

x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
assert sr == 48000, "dfn_worker expects 48 kHz"
model, state, _ = init_df(log_level="ERROR")
n = x.shape[0]
outs = []
for c in range(x.shape[1]):
    e = enhance(model, state, torch.from_numpy(np.ascontiguousarray(x[:, c])).unsqueeze(0))
    e = e.squeeze(0).numpy()
    outs.append(np.pad(e, (0, max(0, n - len(e))))[:n])
sf.write(sys.argv[2], np.stack(outs, 1).astype(np.float32), sr, subtype="FLOAT")
