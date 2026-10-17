"""Ablation / tuning harness around the project bench (same degradations, same metrics, same alignment),
running clips x cases in parallel processes.

    .venv\\Scripts\\python.exe -I bench/ablate.py --name tag [--order declip,denoise,eq,master]
        [--opts declip=false,denoise_strength=0.6,...] [--eqp hpf_min=25,level=false,...]
        [--mcfg clip_head_db=0.5,T_voice=-16.3,...] [--cases ...] [--clips ...] [--mode voice] [--procs 6]
        [--save DIR]

--order : stage order (declip, dereverb, denoise, level, eq, master; a stage may be repeated).
--eqp   : overrides for pipeline.voice_eq.P;  --mcfg : overrides for pipeline.master.DEFAULT_CFG
          (T_voice / T_music set the loudness targets).
Writes bench/out/<tag>.json (per clip/case metrics in / out / ref plus the pipeline's report) and prints the
mean table.  Metrics are exactly those of run_bench.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SR = 48000
CLIPS = ("david_news", "david_sibil", "zira_story")
CASES = ("clean", "clipped", "noisy", "badmic", "combined", "quiet")


def _parse_kv(s):
    d = {}
    for kv in filter(None, (s or "").split(",")):
        k, v = kv.split("=")
        vl = v.lower()
        if vl in ("true", "false"):
            d[k] = vl == "true"
        elif vl == "none":
            d[k] = None
        else:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def work(task):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    clip, case, spec = task
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    from degrade import CASES as DC
    from metrics import report
    import pipeline
    VE = sys.modules["pipeline.voice_eq"]
    MA = sys.modules["pipeline.master"]
    for k, v in spec["eqp"].items():
        VE.P[k] = v
    for k, v in spec["mcfg"].items():
        if k == "T_voice":
            MA.TARGET_LUFS["voice"] = v
        elif k == "T_music":
            MA.TARGET_LUFS["music"] = v
        else:
            MA.DEFAULT_CFG[k] = v
    x, sr = sf.read(HERE / "clean" / f"{clip}.wav", dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = gcd(sr, SR)
    x = resample_poly(x, SR // g, sr // g)
    ref = x / np.max(np.abs(x)) * 0.7
    deg = np.clip(DC[case](ref, SR), -1, 1)
    opts = dict(spec["opts"])
    if spec["order"]:
        opts["order"] = tuple(spec["order"])
    rep = []
    opts["report"] = rep
    t0 = time.time()
    out = pipeline.process_array(deg[None, :].astype(np.float32), SR, spec["mode"], opts)[0]
    dt = time.time() - t0
    out = out.astype(np.float64)
    if len(out) != len(ref):
        out = np.pad(out, (0, max(0, len(ref) - len(out))))[: len(ref)]
    res = {"clip": clip, "case": case, "sec": dt, "in": report(ref, deg, SR), "out": report(ref, out, SR),
           "ref": report(ref, ref, SR), "report": rep}
    if spec.get("save"):
        d = Path(spec["save"])
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{clip}_{case}_in.wav", deg, SR, subtype="PCM_16")
        sf.write(d / f"{clip}_{case}_out.wav", out, SR, subtype="PCM_16")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--order", default="")
    ap.add_argument("--opts", default="")
    ap.add_argument("--eqp", default="")
    ap.add_argument("--mcfg", default="")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--mode", default="voice")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    spec = dict(order=[s for s in a.order.split(",") if s], opts=_parse_kv(a.opts), eqp=_parse_kv(a.eqp),
                mcfg=_parse_kv(a.mcfg), mode=a.mode, save=a.save)
    tasks = [(c, k, spec) for c in a.clips.split(",") for k in a.cases.split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        results = list(ex.map(work, tasks))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"{a.name}.json").write_text(json.dumps(results, indent=1))
    keys = ["STOI", "SI-SDR", "LSD", "LUFS", "TP", "floor", "SNRdyn", "clip%", "air"]
    import numpy as np
    print(f"== {a.name}  order={spec['order'] or 'default'} opts={spec['opts']} eqp={spec['eqp']} "
          f"mcfg={spec['mcfg']}  ({time.time() - t0:.0f}s)")
    print(f"{'case':<10}{'':<5}" + "".join(f"{k:>9}" for k in keys) + f"{'sec':>7}")
    for case in a.cases.split(","):
        rs = [r for r in results if r["case"] == case]
        for tag in ("in", "out"):
            row = {k: float(np.mean([r[tag][k] for r in rs])) for k in keys}
            print(f"{case:<10}{tag:<5}"
                  + "".join(f"{row[k]:>9.3f}" if k in ("STOI", "air") else f"{row[k]:>9.1f}" for k in keys)
                  + (f"{np.mean([r['sec'] for r in rs]):>7.1f}" if tag == "out" else ""))


if __name__ == "__main__":
    main()
