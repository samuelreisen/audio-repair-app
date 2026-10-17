"""Edge-case suite: the pipeline must never crash, return NaN, change length, or exceed [-1, 1]."""
import sys, time, tempfile, os
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pipeline
from pipeline import run
from pipeline.io import load_audio

rng = np.random.default_rng(0)
def speechish(sr, sec, amp=0.5):
    t = np.arange(int(sr*sec))/sr
    env = np.maximum(0, np.sin(2*np.pi*2.2*t))**2
    v = sum(np.sin(2*np.pi*140*h*t)/h for h in range(1, 7))
    return (amp*v*env/2).astype(np.float32)
real, rsr = sf.read(ROOT/"bench/clean/david_news.wav", dtype="float32")
cases = {
 "silence 3s 48k":      (np.zeros(48000*3, np.float32), 48000),
 "0.1s tone":           (speechish(48000, .1), 48000),
 "1 sample":            (np.array([0.3], np.float32), 48000),
 "1s speechish":        (speechish(48000, 1), 48000),
 "8k speech":           (real[: rsr*3], 8000),
 "16k speech":          (real[: rsr*3], 16000),
 "44.1k speech":        (real[: rsr*3], 44100),
 "96k speech":          (real[: rsr*3], 96000),
 "DC offset":           (real[: rsr*3] + 0.2, rsr),
 "full-scale square":   (np.sign(np.sin(2*np.pi*200*np.arange(48000*2)/48000)).astype(np.float32), 48000),
 "pure 440 tone":       ((0.5*np.sin(2*np.pi*440*np.arange(48000*3)/48000)).astype(np.float32), 48000),
 "chirp":               (np.sin(2*np.pi*(100*np.arange(48000*3)/48000 + 2000*(np.arange(48000*3)/48000)**2/3)).astype(np.float32)*0.5, 48000),
 "-70 dBFS quiet":      (real[: rsr*3]*10**(-70/20)/np.abs(real[: rsr*3]).max(), rsr),
 "white noise only":    ((0.3*rng.standard_normal(48000*3)).astype(np.float32), 48000),
 "all clipped":         (np.clip(real[: rsr*3]*100, -1, 1), rsr),
 "loud mastered noise": (np.clip(0.9*rng.standard_normal(48000*3), -1, 1).astype(np.float32), 48000),
}
stereo = np.stack([real[: rsr*3], real[: rsr*3]*0.5], 1)
cases["stereo"] = (stereo, rsr)
cases["6-channel"] = (np.tile(real[: rsr*2, None], (1, 6)), rsr)
nan = real[: rsr*3].copy(); nan[1000:1010] = np.nan; cases["NaN samples"] = (nan, rsr)

bad = 0
for name, (x, sr) in cases.items():
    with tempfile.TemporaryDirectory() as td:
        i, o = os.path.join(td, "i.wav"), os.path.join(td, "o.wav")
        try:
            sf.write(i, np.nan_to_num(x) if name == "NaN samples" and False else x, sr, subtype="FLOAT")
            t0 = time.time(); run(i, o, "voice", {}); dt = time.time()-t0
            y, so = sf.read(o, always_2d=True)
            a, _ = load_audio(i)
            ok = np.isfinite(y).all() and y.shape[0] == a.shape[1] and np.abs(y).max() <= 1.0 and so == 48000
            msg = f"len {y.shape[0]}/{a.shape[1]} ch {y.shape[1]} peak {np.abs(y).max():.2f} {dt:.1f}s"
        except Exception as e:
            ok, msg = False, f"EXC {type(e).__name__}: {str(e)[:100]}"
        bad += not ok
        print(("PASS " if ok else "FAIL ") + f"{name:22s} {msg}")
print("failures:", bad)
