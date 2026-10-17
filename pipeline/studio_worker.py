"""Studio voice as a separate process, for running from the source folder on a Python without PyTorch.
It runs inside the Python 3.11 environment (.venv311). The packaged app calls studio_core.restore() directly instead.

    python studio_worker.py in.wav out.wav [nfe] [lambd] [tau] [blend] [threads]

Writes a mono 44.1 kHz float WAV, the same length as the input in time. Prints "PROGRESS i total" lines.
"""
import os
import sys

import numpy as np
import soundfile as sf

import ai
from studio_core import restore

inp, outp = sys.argv[1], sys.argv[2]
nfe = int(sys.argv[3]) if len(sys.argv) > 3 else 32
lambd = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
tau = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
blend = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
threads = int(sys.argv[7]) if len(sys.argv) > 7 else None
ai._shim_torchaudio()

audio, sr = sf.read(inp, dtype="float32", always_2d=True)
y, rate = restore(audio.mean(axis=1), sr, ai.models_root() / "enhancer_stage2", nfe, lambd, tau, blend, threads,
                  lambda d, t: print("PROGRESS %d %d" % (d, t), flush=True))
sf.write(outp, y, rate, subtype="FLOAT")
