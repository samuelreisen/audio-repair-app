"""Compare the normal pipeline with Studio voice (AI) on degraded speech, scored against the clean original.

    python bench/studio_test.py [seconds] [cases]       e.g.  python bench/studio_test.py 12 badmic,noisy
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from degrade import CASES            # noqa: E402
from metrics import report           # noqa: E402
from run_bench import load_clean, SR  # noqa: E402
import pipeline                       # noqa: E402
from pipeline.studio import studio_available  # noqa: E402

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
cases = (sys.argv[2] if len(sys.argv) > 2 else "badmic,noisy,combined").split(",")
print("studio available:", studio_available())
clip = sorted((ROOT / "bench" / "clean").glob("*.wav"))[0]
ref = load_clean(clip)[: int(seconds * SR)]
keys = ("STOI", "SI-SDR", "LSD", "LUFS", "TP", "air")
print(f"{'case':10s}{'variant':10s}" + "".join(f"{k:>9}" for k in keys) + f"{'sec':>7}")
for case in cases:
    deg = np.clip(CASES[case](ref, SR), -1, 1)
    r = report(ref, deg, SR)
    print(f"{case:10s}{'input':10s}" + "".join(f"{r[k]:>9}" for k in keys))
    for name, opts in (("normal", {}), ("studio", {"studio": True})):
        t = time.time()
        out = pipeline.process_array(deg[None].astype(np.float32), SR, "voice", opts)[0].astype(np.float64)
        dt = time.time() - t
        n = min(len(ref), len(out))
        r = report(ref[:n], out[:n], SR)
        print(f"{case:10s}{name:10s}" + "".join(f"{r[k]:>9}" for k in keys) + f"{dt:>7.0f}")
