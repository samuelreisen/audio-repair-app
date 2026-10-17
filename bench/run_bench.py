"""Run the repair pipeline over degraded speech and print objective metrics.

    python bench/run_bench.py [--mode voice] [--cases clipped,noisy,...] [--save]
Compare the 'in' row (degraded, untouched) against 'out' (repaired); 'ref' is the clean
original for context. Higher STOI / SI-SDR is better, lower LSD is better.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from degrade import CASES            # noqa: E402
from metrics import report            # noqa: E402
import pipeline                       # noqa: E402

SR = 48000


def load_clean(p):
    x, sr = sf.read(p, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    from math import gcd
    g = gcd(sr, SR)
    x = resample_poly(x, SR // g, sr // g)
    return x / np.max(np.abs(x)) * 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="voice")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--clips", default="")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--opts", default="")
    a = ap.parse_args()

    opts = {}
    for kv in filter(None, a.opts.split(",")):
        k, v = kv.split("=")
        opts[k] = (v.lower() == "true") if v.lower() in ("true", "false") else float(v)

    clips = sorted((ROOT / "bench" / "clean").glob("*.wav"))
    if a.clips:
        clips = [c for c in clips if c.stem in a.clips.split(",")]
    out_dir = ROOT / "bench" / "out"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for clip in clips:
        ref = load_clean(clip)
        for case in a.cases.split(","):
            deg = CASES[case](ref, SR)
            deg = np.clip(deg, -1, 1)
            t0 = time.time()
            out = pipeline.process_array(deg[None, :].astype(np.float32), SR, a.mode, opts)[0]
            dt = time.time() - t0
            out = out.astype(np.float64)
            if len(out) != len(ref):
                out = np.pad(out, (0, max(0, len(ref) - len(out))))[: len(ref)]
            for tag, sig in (("ref", ref), ("in", deg), ("out", out)):
                r = report(ref, sig, SR)
                rows.append((clip.stem, case, tag, r, dt if tag == "out" else None))
            if a.save:
                sf.write(out_dir / f"{clip.stem}_{case}_in.wav", deg, SR, subtype="PCM_16")
                sf.write(out_dir / f"{clip.stem}_{case}_out.wav", out, SR, subtype="PCM_16")

    keys = list(rows[0][3])
    print(f"{'clip':<12}{'case':<10}{'':<5}" + "".join(f"{k:>9}" for k in keys) + f"{'sec':>7}")
    for clip, case, tag, r, dt in rows:
        print(f"{clip:<12}{case:<10}{tag:<5}" + "".join(f"{r[k]:>9}" for k in keys)
              + (f"{dt:>7.1f}" if dt else ""))

    print("\nMEAN improvement (out vs in), per case:")
    for case in a.cases.split(","):
        ins = [r for c, k, t, r, _ in rows if k == case and t == "in"]
        outs = [r for c, k, t, r, _ in rows if k == case and t == "out"]
        d = {k: round(np.mean([o[k] for o in outs]) - np.mean([i[k] for i in ins]), 2) for k in keys}
        print(f"  {case:<10}", d)


if __name__ == "__main__":
    main()
