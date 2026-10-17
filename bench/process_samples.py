"""Run the full pipeline on every real recording in samples/ and write fixed audio + before/after spectrograms."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
from pipeline import process_with_report
from pipeline.io import load_audio, save_audio
from metrics import lufs, true_peak_dbfs, noise_floor_db, clip_fraction

SR = 48000
src, out = ROOT / "samples", ROOT / "samples" / "out"
out.mkdir(exist_ok=True)
exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".mp4", ".wma", ".webm"}
files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]
if not files:
    print("No audio files in samples/ yet."); sys.exit(0)
mode = sys.argv[1] if len(sys.argv) > 1 else "voice"
for p in files:
    x, sr = load_audio(str(p))
    y, rep, an = process_with_report(x, sr, mode)
    save_audio(str(out / f"{p.stem}_fixed.wav"), y, sr)
    mx, my = x.mean(0).astype(np.float64), y.mean(0).astype(np.float64)
    print(f"\n=== {p.name}  ({len(mx)/sr:.1f}s) ===")
    for tag, m in (("before", mx), ("after ", my)):
        fl, dr = noise_floor_db(m, sr)
        print(f"  {tag}: {lufs(m, sr):6.1f} LUFS  peak {true_peak_dbfs(m, sr):5.1f} dBTP  quiet-floor {fl:6.1f} dBFS  clipped {100*clip_fraction(m):.2f}%")
    print("  " + "\n  ".join(rep))
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for a, m, t in ((ax[0], mx, "before"), (ax[1], my, "after")):
        f, tt, S = spectrogram(m, sr, nperseg=1024, noverlap=768)
        a.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-12), vmin=-110, vmax=-20, shading="auto"); a.set_ylim(0, 12)
        a.set_ylabel("kHz"); a.set_title(t)
    fig.tight_layout(); fig.savefig(out / f"{p.stem}_compare.png", dpi=90); plt.close(fig)
print("\nWrote results to", out)
