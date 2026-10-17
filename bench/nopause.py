import sys; from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"bench"))
from run_bench import load_clean, SR
from metrics import report
import pipeline

def remove_pauses(x, sr, thr_db=-40, fade=0.008):
    n = int(0.01*sr); k = len(x)//n
    rms = 20*np.log10(np.sqrt(np.mean(x[:k*n].reshape(k, n)**2, axis=1))+1e-9)
    keep = np.repeat(rms > rms.max()+thr_db+25 - 25 + 0, n)  # frames within 40 dB of peak
    keep = np.repeat(rms > (np.percentile(rms, 95) - 32), n)
    segs = []; i = 0; xs = x[:len(keep)]
    while i < len(keep):
        if keep[i]:
            j = i
            while j < len(keep) and keep[j]: j += 1
            segs.append(xs[i:j]); i = j
        else: i += 1
    f = int(fade*sr); out = segs[0]
    for s in segs[1:]:
        if len(s) <= 2*f or len(out) <= 2*f: out = np.concatenate([out, s]); continue
        w = np.linspace(0, 1, f); out = np.concatenate([out[:-f], out[-f:]*(1-w)+s[:f]*w, s[f:]])
    return out

for clip in sorted((ROOT/"bench/clean").glob("*.wav")):
    ref = remove_pauses(load_clean(clip), SR)
    rep = []
    out = pipeline.process_array(ref[None].astype(np.float32), SR, "voice", {"report": rep})[0].astype(np.float64)
    n = min(len(ref), len(out)); ref, out = ref[:n], out[:n]
    r = report(ref, ref, SR); o = report(ref, out, SR)
    print(f"{clip.stem:12s} dur {n/SR:4.1f}s  STOI {o['STOI']}  SI-SDR {o['SI-SDR']}  LSD {o['LSD']}  LUFS {o['LUFS']}")
    print("    " + " | ".join(l for l in rep if l.startswith(("Applied", "Detected: Background", "Detected: Room"))))
