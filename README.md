# Audio Repair

Local web app that fixes clipping, background noise, room reverb, poor mic tone and loudness.
Everything runs on the CPU; a 10 s clip takes roughly 5-10 s.

## Start
1. Install Python: `winget install Python.Python.3.11`
2. Install ffmpeg (for MP3/M4A): `winget install ffmpeg`
3. Double-click `run.bat`. First run installs packages, then opens http://127.0.0.1:8000.

Optional but recommended (neural speech-noise model used by the denoiser, ~13 MB):
`.venv\Scripts\python.exe -m pip install --no-deps pyrnnoise`. Without it the classical estimator is used.

## Pipeline
`analyse -> declip -> de-reverb + denoise -> tone -> master`  (`pipeline/__init__.py`)

| Stage | Module | What it does |
|---|---|---|
| analyse | `analysis.py` | Measures clipping, noise floor / speech-to-noise range, mains hum, reverberance, bandwidth, loudness. Decides which stages to skip and (unless forced) how strong to run them. |
| declip | `declip.py` | Data-driven clipping detection (flat tops, smeared tops, soft saturation), time-varying AR / EM reconstruction of the missing peaks, headroom to -1 dBTP. No-op (same array object) when nothing is clipped. |
| denoise | `denoise.py`, `dereverb.py` | Mains-hum notch comb -> WPE de-reverb (scaled by the detected reverberance) -> classical minimum-statistics Wiener gain fused with an RNNoise mask, faded out when the noise is already inaudible. |
| tone | `voice_eq.py` | Adaptive tone correction toward a speech target (dead-banded, SNR-aware), resonance cut, adaptive low-cut, conservative bandwidth extension for band-limited input, level-adaptive de-esser, slow speech-gated leveler. |
| master | `master.py` | Gated-LUFS loudness (-16 voice / -14 music) measured *after* the limiter, 4x-oversampled soft clipper + look-ahead limiter, true peak <= -1 dBTP. Stops short of the target rather than crushing dynamics. |

All stages are deterministic, sample-aligned (no delay, no length change) and CPU-only.

### Automatic behaviour and the report
* No clipping detected -> declip is skipped. Noise more than ~40 dB under the speech and no hum / reverb ->
  denoise is skipped. Stages you switch off in the UI stay off.
* "Auto" noise-reduction strength (the default; untick it in the UI to use the slider). Over the API simply omit
  `denoise_strength`.
* Every run produces a short report of what was detected and done (shown under the result in the UI, also
  `GET /status/<id>` -> `report`). From Python:

```python
from pipeline import process_with_report
audio, report, analysis = process_with_report(audio, sr, "voice")   # audio: float32 (channels, samples)
print("\n".join(report))
```
or pass `opts={"report": []}` / `opts={"analysis": {}}` to `process_array` / `run` and read the list back.

Options (`process_array(audio, sr, mode, opts, progress)`): `declip`, `denoise`, `eq`, `master` (bool),
`denoise_strength` (0..1, absent = auto), `auto` (False = never skip stages), `report` (list), `analysis` (dict),
`order` (advanced: tuple of stage names).

### Stage order (verified by ablation)
Mean over 3 clips x the 5 degraded bench cases (noisy, badmic, combined, clipped, quiet), voice mode;
input row = untouched degraded signal:

| Order | STOI | SI-SDR dB | LSD dB | max TP dBTP | clean STOI / LSD |
|---|---|---|---|---|---|
| (input) | 0.912 | 5.8 | 17.7 | - | 1.000 / 0.0 |
| **declip, denoise(+dereverb), eq, master** | **0.945** | **9.0** | **10.7** | **-1.0** | **0.995 / 2.5** |
| eq before denoise | 0.945 | 8.8 | 10.4 | -1.0 | 0.995 / 2.5 |
| de-reverb as its own stage after denoise | 0.942 | 8.9 | 10.7 | -1.0 | 0.995 / 2.5 |
| de-reverb as its own stage before denoise | 0.944 | 9.0 | 10.7 | -1.0 | 0.995 / 2.5 |
| leveler before denoise | 0.945 | 9.0 | 10.7 | -1.0 | 0.995 / 2.5 |
| leveler as a stage between denoise and eq | 0.945 | 9.0 | 10.7 | -1.0 | 0.995 / 2.5 |
| master -> denoise -> eq -> master | 0.943 | 8.8 | 11.0 | -1.0 | 0.993 / 2.9 |
| denoise before declip | 0.932 | 8.6 | 10.9 | -1.0 | 0.995 / 2.5 |
| eq after master | 0.945 | 9.0 | 11.0 | **+0.1** (ceiling broken) | 0.995 / 2.5 |

Conclusions: declip must come first ('combined' STOI 0.864 -> 0.801 when denoise runs first); mastering must be
last (anything after it breaks the true-peak ceiling and the loudness); leveling before/after denoise and the
position of de-reverb are within noise, so the signal-flow-logical order (denoise before any gain riding, so its
stationary-noise assumption holds) is kept; tone before denoise lowers LSD by 0.3 dB but costs 0.2 dB SI-SDR and
boosts noise that the denoiser then has to remove.

## Benchmark
`bench/` degrades 3 clean speech clips (Windows SAPI voices) and scores the result against the clean original
(STOI, SI-SDR, log-spectral distance, loudness, true peak, noise floor, "air" = share of energy above 4 kHz).
`bench/ablate.py` runs the same bench in parallel with a custom stage order / stage parameters (see its docstring).

    .venv\Scripts\python.exe -I bench/run_bench.py [--mode voice|music] [--cases clipped,noisy,...] [--opts declip=false,...]

Final voice-mode numbers, mean of the 3 clips, input (degraded, untouched) -> output:

| case | STOI | SI-SDR dB | LSD dB | LUFS | TP dBTP | floor dBFS | SNRdyn dB | air |
|---|---|---|---|---|---|---|---|---|
| clean | 1.000 -> 0.995 | 155 -> 16.4 | 0.0 -> 2.5 | -22.2 -> -16.0 | -3.1 -> -1.0 | -180 -> -69 | 162 -> 57 | 0.050 -> 0.057 |
| clipped | 0.931 -> 0.993 | 10.5 -> 16.3 | 9.9 -> 3.2 | -9.7 -> -16.0 | +0.6 -> -1.0 | -180 -> -69 | 174 -> 57 | 0.074 -> 0.057 |
| noisy | 0.945 -> 0.964 | 5.9 -> 11.4 | 21.4 -> 12.6 | -21.8 -> -16.0 | -2.6 -> -1.0 | -29.5 -> -44.1 | 11.8 -> 31.7 | 0.093 -> 0.060 |
| badmic | 0.900 -> 0.917 | 1.0 -> 2.3 | 19.0 -> 12.8 | -21.9 -> -16.0 | -0.5 -> -1.0 | -80.3 -> -70.2 | 62.0 -> 58.2 | 0.006 -> 0.025 |
| combined | 0.796 -> 0.864 | -2.7 -> 1.2 | 21.9 -> 14.6 | -7.8 -> -16.0 | +0.4 -> -1.0 | -16.2 -> -45.4 | 12.0 -> 33.4 | 0.060 -> 0.022 |
| quiet | 0.989 -> 0.987 | 14.2 -> 13.8 | 16.4 -> 10.1 | -53.0 -> -16.0 | -34.0 -> -1.0 | -68.8 -> -50.5 | 19.6 -> 38.0 | 0.059 -> 0.064 |

(The naive original pipeline made STOI and SI-SDR *worse* in every case, left noise nearly untouched, raised "air" to
0.13-0.22, missed -16 LUFS and overshot -1 dBTP.)  `floor` is in absolute dBFS, so it also contains the loudness gain
(for example +38 dB in "quiet"); `SNRdyn` is the level-independent figure.

Known limits shown by the table, with the reasons:
* **clean** LSD rises by ~2.5 dB: bringing peaky speech (peak-to-loudness ratio ~20 dB) to -16 LUFS / -1 dBTP needs
  ~5 dB of peak reduction. A plain hard clip at the same loudness gives 3.4 / 3.0 / 1.4 dB on the three clips, a
  limiter-only chain 1.3 / 1.2 / 0.5 dB but with STOI 0.978 / 0.985 / 0.998 (and -16.3 LUFS on the first clip).
  STOI stays >= 0.98.
* **quiet** is a marginal case: denoise alone gives +2.5 dB SI-SDR / +0.002 STOI, but the +38 dB gain and ~4 dB
  of peak control cost 2.2 dB / 0.004, net -0.4 dB / -0.002.
* **badmic** has no noise at all (just a band-limited, reverberant mic), so there is no floor to lower; the tone
  stage lifts the missing low and high ends, which also lifts the (inaudible) reverb tails by a few dB.
* The reference clips contain ~2-4 % sub-10 Hz DC wander (a property of the SAPI voices). Any DC / rumble filter
  removes it, and SI-SDR counts that as error, so absolute SI-SDR values are capped around 15-20 dB here.

## Tests
`.venv\Scripts\python.exe -I -m pytest -q`   (about 30 s; behavioural tests: declip improves waveform error,
denoise improves SNR and removes hum, tone stage, master hits target and ceiling, analysis, end-to-end, stereo).

## Do-no-harm checks (added after adversarial review)
* **Continuous speech (no pauses).** With no pauses there is no noise floor to measure: quiet phonemes look like
  one. The analysis tells them apart (a real floor is steady within ~2 dB; soft phonemes vary by 3.5 dB or more, and
  the loud-to-quiet range is under 25 dB). When the speaker never pauses, noise reduction, de-reverb and bandwidth
  extension are skipped and the report says why. Measured on the clips with every pause cut out: LSD 11.7 / 5.0 / 5.9 dB
  down to 3.1 / 2.9 / 1.6 dB, STOI 0.993-0.999 (`bench/nopause.py`).
* **Denoiser guard.** If the gain mask lowers the loud (speech) frames by more than 3 dB it has eaten the wanted
  voice (typically a competing talker), so part of the original is blended back.
* **Bandwidth extension** only fires for strong spectral deficits (a genuine band-limited recording).
* **Hum.** One strong low line seeds the comb and weaker harmonics landing exactly on it are added; unrelated
  partials in music are no longer notched. 0 false positives on the clean clips.
* **Edge cases** (`bench/robust.py`, 19 cases: silence, 1 sample, 8-96 kHz, 6 channels, NaN, DC, full-scale square,
  -70 dBFS, pure tones, white noise, fully clipped...): no crashes, no NaN, length preserved, nothing above full scale.

## Honest limits
All numbers come from synthetic speech (Windows voices) with synthetic damage. They catch regressions and show the
direction, but they do not prove how it sounds on your real recordings, which is the real test.
A bad mic or room echo is only modestly improved. No heavy neural enhancer is used: DeepFilterNet has no
Python 3.14 build, so noise removal is RNNoise plus a classical estimator.

## Neural noise removal (DeepFilterNet3, optional, recommended)
Run `setup_neural.bat` once (about 130 MB: PyTorch CPU plus DeepFilterNet; needs Python 3.11 because PyTorch has
no Python 3.14 build). It creates `.venv311`; the app then runs DeepFilterNet3 through a subprocess
(`pipeline/dfn.py`, `pipeline/dfn_worker.py`) for voice recordings and falls back to the built-in estimator when it is
missing. Set `AR_DFN=0` to compare against the built-in path. Cost: about 10-20 s extra per file (model load).
Measured on the bench, mean of 3 clips, with vs without it: noisy LSD -8.8 -> -11.1 dB, speech-to-noise range
+20 -> +25 dB; clipped+noisy+bad mic+echo STOI +0.07 -> +0.09 and noise far lower (+21 -> +40 dB range).

## What real recordings taught us (voice memos, `samples/`)
The first two real recordings (AAC voice memos, 109 s and 160 s) exposed what the synthetic bench could never show:
they are full of short loud **clicks** (0.5-8 ms, up to 55 dB above the voice, peaks over full scale) and a few
**bangs** (20-200 ms knocks 30-50 dB above the voice). Those peaks stopped loudness mastering from raising the
voice (one file stayed at -25 LUFS). Two stages now run first:
* `pipeline/declick.py`: finds spikes more than 8x the RMS of their own surroundings (no false alarms on clean
  speech; 84% of injected clicks found, +5.5 dB SI-SDR) and repairs each gap by two-sided LPC prediction, with
  damping, a clamp and an interpolation fallback.
* `pipeline/tame.py`: turns any 20 ms level more than 14 dB above the speaker's own loud level (90th percentile within
  +-4 s) down to that ceiling, with look-ahead. Speech peaks are only ~11 dB over that level, so speech is untouched.

Result on the two recordings: both now land at -16.0 LUFS / -1.0 dBTP (josh6 was stuck at -25.1), with 58 + 6 and
40 + 6 clicks / bangs repaired. M4A / MP4 / AAC input is decoded with PyAV, so ffmpeg is not required.
Try your own with `bench\process_samples.py` (outputs in `samples\out\`).

## Desktop apps (.exe, .dmg, Linux)
`desktop.py` is the desktop entry point: the same interface in its own native window (WebView2 on Windows, the system
web view on macOS, Qt WebEngine on Linux), talking to the Python engine directly through pywebview's bridge. **No
server and no network port**: files are opened and saved with the system dialogs and processed where they are. The page
(`static/index.html`) detects whether it runs in the app window or in a browser and uses the matching route.
If the native window cannot be created, or has not loaded after 30 seconds, the app restarts itself in browser mode
(`launcher.py`: a small local server plus your browser) instead of freezing; startup problems are written to
`audio_repair.log` in the temp folder. `build.py` packages it with PyInstaller (it builds in the temp folder, because
OneDrive and similar sync clients lock the build files).

Tested on the packaged Windows app (`AUDIO_REPAIR_SELFTEST=<file>` makes the app drive its own page and write a JSON
result): window opens, file selected, repair runs, both players load audio supplied by Python, Save works, the app's
process tree has **zero listening TCP ports**, and a real 109 s MP4 voice memo is decoded and repaired. The forced
fallback to browser mode was also tested. Not tested: the macOS and Linux native windows.

* **Windows (built and tested here):** `python build.py` makes `dist\Audio Repair\Audio Repair.exe` (235 MB folder,
  about 105 MB zipped as `release\Audio-Repair-Windows.zip`). Keep the whole folder together; double-click the exe.
  Windows SmartScreen may warn because the app is not code-signed: choose More info, then Run anyway.
* **macOS (.dmg) and Linux (AppImage and .tar.gz):** these cannot be built on Windows. `.github/workflows/build.yml`
  builds all three on GitHub's own Windows, macOS and Linux machines. Run it from the repository's Actions tab, or push
  a tag such as `v1.0` to attach the files to a Release. The macOS app is unsigned: right-click the app, choose Open the
  first time (or `xattr -dr com.apple.quarantine "Audio Repair.app"`). The native-window version builds successfully on all three platforms on GitHub (run 39039245519, commit dd8c928). Each download was checked to be well-formed (zip, dmg disk-image trailer, AppImage ELF header, gzip magic). Only the Windows app has been launched; the macOS and Linux apps have never been run on a Mac or Linux machine.
* **What the packaged app includes:** the full repair pipeline with RNNoise and M4A/MP4 decoding. The optional
  DeepFilterNet neural denoiser needs a separate Python 3.11 environment and is not bundled; the app falls back to the
  built-in denoiser automatically.
