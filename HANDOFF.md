<!-- handoff-mark: 3 -->
# Audio Repair – project handoff

> **MAINTENANCE RULE: everything that is done to this project is saved in this file.** Add a dated entry at the top of
> the **Change log (section 12)**, newest first, in the same format as the others (**Asked / Changed / Verified /
> Not done**), and fix any earlier section the change makes untrue, so the document stays correct rather than just
> appended to. Read this file first in every session.
>
> **The update system carries this file.** `python tools/make_update.py` writes the update's entry into this file,
> stamps the first line `<!-- handoff-mark: N -->` and packs the file into the `.srar` update. Installing the update
> replaces the user's older HANDOFF.md (in the install folder and the app's data folder) with this one; the old copy is
> kept in `handoff-history/`. An older document never replaces a newer one. Do not remove the `handoff-mark` line.
>
> **Privacy rule for this file:** it travels with the app to anyone who has it. Never write personal details into it:
> no names, e-mail addresses, account names, folder locations on anyone's computer, or anything about the owner's own
> recordings.

Written for an AI (or a person) picking this project up cold: what the app is, how it is built, the parts that are easy
to break, what was verified and how, what is unfinished, and what the owner cares about. Where a claim comes from a
measurement it says so; where something was never tested it says that too.

---------------------------------------------------------------------------------------------------------------

## 1. What it is

**Audio Repair** is a desktop app that takes a bad recording (mostly speech: voice memos, a poor microphone, a noisy or
echoey room, clipping, clicks, uneven level) and fixes it automatically: clicks and bumps removed, clipped peaks
rebuilt, noise, hum and room echo reduced, tone corrected, and the result mastered to a consistent loudness
(-16 LUFS for voice, -14 LUFS for music, true peak at most -1 dBTP). It shows Before / After players and a plain report
of what it did, and saves the repaired file (WAV or MP3).

- Python engine (numpy / scipy DSP written for this project, plus two AI models), with a single-page interface
  (`static/index.html`) shown in a native window by **pywebview**. **No server, no network port, works offline.**
- Windows (installer, per-user) and Mac (Apple silicon `.dmg`). Linux was built once and then dropped at the owner's
  request.
- Updates are `.srar` files ("marks", Mk II, Mk III, ...), modelled on the owner's other app, SoundField 3D (section 9).

## 2. The owner and what they care about (read before changing anything)

- Not a programmer. They judge by ear and by whether it behaves like a finished app. Short, blunt feedback
  ("its horrible", "the app keeps saying this" with a screenshot): find the real cause.
- **They want a proper app, never scripts or servers.** "i cant keep running these scripts i want an app not these
  servers". Delivery = an installer they double-click, with Start-menu / desktop shortcuts, like SoundField 3D.
- **Quality bar: "high end audio clarity".** Their reaction to the current pipeline on real recordings: "sounds
  amazing". Do not regress it: run the bench and the tests (section 10) before shipping any audio change.
- **Finish the whole job in one go.** "i cant keep having to tell you to carry on ... do your thing". Work
  autonomously to the end, verify, then report what was and was not verified.
- **Neat, easily identifiable names.** Installers are `Audio Repair - Windows Installer.exe` and
  `Audio Repair - Mac Installer.dmg`; updates are `Audio Repair - Mk <N> update.srar`.
- **The GitHub repositories stay private.** Never change their visibility, never share links unless asked.
- **Never commit or upload the owner's own recordings** (`samples/` is git-ignored except `samples/README.txt`).
- Wants cheaper models used for routine work: they asked for Haiku sub-agents for running tests and mechanical tasks.
- Animations must play in both scroll directions (they noticed reveal animations only playing once).

## 3. The conversation that built it (what was asked, in order, and what was done)

Paraphrased from the owner's own words; personal details removed.

1. **"build me an app so that i can upload bad quality audio files that might be clipping, might be recorded through a
   bad mic, in need of mastering, in need of background reduction – this app must fix all this."** First version: a
   local web app (FastAPI + a plain HTML page) with declip, denoise, voice EQ and mastering stages.
2. **"the animations only play once, they must play both ways"**: scroll reveals now reverse on exit and replay.
3. **"are the repair tools actually working properly and nicely? i want this to be high end audio clarity" / "run it
   and show me the results"**: a benchmark was built (`bench/`: three clean synthetic voices, degraded six ways, scored
   with STOI, SI-SDR, log-spectral distance, LUFS, true peak). It showed the first naive pipeline made things *worse*, so
   every stage was rewritten (section 5) and the stage order was settled by ablation.
4. **"you have to finish everything in one session ... bypass permissions are on, do your thing"**: worked through to
   completion. Then the interface was redesigned (Apple-style inset grouped lists, system colours, light/dark,
   reduced-motion and contrast support) with a frontend design skill.
5. The owner supplied **two real voice memos** (private; never commit them). They revealed what synthetic tests never
   showed: dozens of short loud clicks and a few loud bangs, which stopped mastering from reaching the target. Two new
   stages, **declick** and **tame**, run first now. Owner: **"sounds amazing."**
6. **"how well does it work if i run it straight out of index.html"**: it cannot work from the bare file (it needs the
   engine); the page now detects that and says to open the app instead.
7. **"can you make it an actual app like an exe and a dmg and whatever linux uses" / "yes, build them on github"**:
   PyInstaller build script and a GitHub Actions workflow for Windows, macOS and Linux.
8. **"what model does this app use"**: DeepFilterNet3 (noise and echo) plus RNNoise and classical DSP; later Resemble
   Enhance ("Studio voice").
9. **"how possible is it to get this app to not run through a server and instead an actual app" / "id like a proper
   app"**: the native window (`desktop.py`, pywebview, JavaScript bridge, no server, no port).
10. **"i just tried with another file thats distorted and its horrible" / "can you implement this repo
    (CrispVoice)"**: the CrispVoice approach was implemented as the optional **Studio voice (AI)** switch (Resemble
    Enhance re-synthesis plus a 25 % DeepFilterNet blend of the original). On the synthetic bench it scores below the
    normal pipeline, so it is off by default and, when on, it replaces declip / denoise / tone (section 6).
11. **"how do i run the exe?" / "i want it to be like that SoundField app"**: proper packaging: a Windows installer
    (Inno Setup, per user, shortcuts, uninstaller), a drag-to-Applications `.dmg`, an app icon, an install guide.
12. **"the app keeps saying this" (screenshot of "The app is not running" inside the app)**: the banner was meant only
    for the bare page file but showed while the window's bridge was still starting. Fixed: the app injects
    `window.__AR_NATIVE = true` into the page before its scripts run.
13. **"the ai models are being weird, ensure they are downloaded with the install"**: both models' weights are now
    shipped inside the app (no download at first use, works offline). Checksums are verified at build time.
14. **"please create sub agents to help you, use haiku"**: routine work (test runs, file checks) went to Haiku agents.
15. **"lose the linux install and clean the unused files, sort everything, make it all neat, rename everything to a
    usable easily identifiable name"**: Linux dropped from the build and the docs; build leftovers, logs, caches and
    old builds removed; installers renamed; README gained a project layout table.
16. **"can you push this to github under a new repo"**: pushed to a new private repository, `audio-repair-app` (the
    older private `audio-repair` repository still exists).
17. **"similar to what we did with soundfield, add update functionality, make the file extension SRAR"**: the `.srar`
    update system (section 9). App version 1.1.0, built-in mark Mk I.
18. **"add a handoff file to the github and install folder ... it must update with the app ... purely to let any Claude
    know how this app is made, similar to soundfield"**: this file. It ships in the repository, next to the installed
    program, in the `.dmg`, and inside every update, and the app keeps the user's copies up to date.

(A separate request, to make a Logic Pro plug-in, was handled in a different folder and session. It is not part of
this app; do not commit that folder from here.)

## 4. Repository map

| Path | What it is |
|---|---|
| `desktop.py` | The app: native window, `Api` class the page calls (`window.pywebview.api.*`), drop handling, opening files the app was launched with, self-test |
| `app_updates.py` | Updates (`.srar` marks) and the handoff document: format, install, switch, startup loading |
| `engine.py` | In-process job manager: `start(path, mode, fmt, ...)`, `status`, `output_path`, `save_copy`, `capabilities()` |
| `pipeline/` | The repair stages (section 5), `io.py` (decode via PyAV / soundfile, encode), the AI glue (`ai.py`, `dfn.py`, `studio.py`, `studio_core.py`) |
| `static/index.html` | The whole interface (HTML + CSS + JS, no build step); works in the app window and, for development, in a browser |
| `mark.json` | Which mark is built into this source (`{"mark": 1, "name": "Mk I", ...}`) |
| `tools/make_update.py` | Makes the next `.srar` update (section 9) |
| `pipeline/tone_after.py` | The four tone controls applied after the repair (section 5, "Tone after the repair") |
| `bench/ui_harness.py` | Opens the page in a browser with a mock engine, for looking at and driving the interface (section 7) |
| `tools/clean_hparams.py` | Makes the Studio model's settings file readable on Windows (section 6); run by the GitHub build |
| `updates/history.json` | Every mark made; `updates/out/` holds the made `.srar` files (git-ignored) |
| `build.py` | PyInstaller build (builds in the temp folder, copies to `dist/`) |
| `installer.iss` | Windows installer (Inno Setup 6): per user, shortcuts, uninstaller, `.srar` file type, HANDOFF.md next to the program |
| `.github/workflows/build.yml` | Builds both installers on GitHub (Actions tab, "Build desktop apps") |
| `assets/` | App icon (`icon.png`, `icon.ico`) |
| `app.py`, `launcher.py` | Browser-mode fallback (local FastAPI server + browser), used only if the native window cannot start |
| `run.bat`, `setup_neural.bat` | Developer setup for running from source on Windows (`.venv` and `.venv311`) |
| `tests/`, `bench/` | Behavioural tests (pytest) and the quality benchmark |
| `INSTALL.md`, `README.md` | The user's install guide; the developer README (pipeline details and numbers) |

## 5. The audio pipeline

`analyse -> declick -> tame -> declip -> denoise (hum notch, WPE de-reverb, DeepFilterNet3 / RNNoise / classical) ->
tone -> master`, in `pipeline/__init__.py`. All stages are deterministic, sample-aligned (no delay, no length change)
and run on the CPU. Each stage is skipped when the analysis finds nothing to fix, and the report says why.

- **analyse** (`analysis.py`): clipping, noise floor / speech-to-noise range, mains hum, reverberance, bandwidth,
  loudness; decides which stages run and how strongly. It tells a real noise floor (steady within ~2 dB) from soft
  phonemes, so continuous speech without pauses is not "denoised" into mush.
- **declick** (`declick.py`): spikes more than 8x the RMS of their surroundings, repaired by two-sided LPC prediction.
- **tame** (`tame.py`): 20 ms levels more than 14 dB above the speaker's own loud level are turned down to that
  ceiling (speech peaks are only ~11 dB over it, so speech is untouched).
- **declip** (`declip.py`): flat / smeared tops and soft saturation detected; peaks rebuilt with time-varying AR / EM.
- **denoise** (`denoise.py`, `dereverb.py`, `dfn.py`): hum comb, WPE de-reverb scaled by the measured reverberance,
  then DeepFilterNet3 for voice (fallbacks: RNNoise mask fused with a minimum-statistics Wiener gain). A guard blends
  the original back if the mask eats the voice.
- **tone** (`voice_eq.py`): adaptive correction toward a speech target, resonance cut, low-cut, conservative
  bandwidth extension, de-esser, slow speech-gated leveller.
- **master** (`master.py`): gated LUFS measured after the limiter, 4x-oversampled soft clip + look-ahead limiter.

Stage order was chosen by ablation: declip must be first (denoise first drops STOI 0.864 -> 0.801 on the combined
case), mastering must be last (anything after it breaks the true-peak ceiling). Bench numbers (synthetic voices, mean of
3 clips, input -> output): combined damage STOI 0.796 -> 0.864, clipped 0.931 -> 0.993, noisy SI-SDR 5.9 -> 11.4 dB;
all cases land at -16.0 LUFS / -1.0 dBTP. Full tables in README.md. The two real voice memos: 58 + 6 and 40 + 6
clicks / bangs repaired, both now at -16.0 LUFS (one was stuck at -25 LUFS before declick and tame).

### Tone after the repair (added in Mk III)
`pipeline/tone_after.py`: four RBJ biquads, each -8..+8 dB: **warmth** (low shelf 200 Hz), **clarity** (bell 3 kHz,
Q 0.8), **brightness** (high shelf 6 kHz), **air** (high shelf 12 kHz). They are applied to the *stored result of the
repair*, never to the original, so changing the tone never re-runs the repair (a 160 s file re-tones in a few seconds;
a Studio-voice repair is not repeated). How it works end to end:
- A normal repair now always writes its full-quality WAV first (`job["base"]`; the MP3, if asked for, is encoded from
  it), and the job keeps that file while any tone job still uses it (`_still_used` in `engine.py`).
- `engine.start(path, ..., base_job=<finished job id>, tone={...})` starts a **tone job**: load the base WAV at its own
  sample rate, `apply_tone`, then `master_with_info` to match the loudness again (-16 LUFS voice / -14 music, true peak
  -1 dBTP; a brighter voice is also a louder one), then save as WAV or MP3. Its report lines are "Applied: tone changed
  after the repair (...)" and "Applied: loudness matched to ...". Tone is always relative to the base, so repeating it is
  idempotent. A tone job is an ordinary job (poll `status`, then `save`), which is why `desktop.py`'s `Api` needed no
  change: `Api.start` passes any options straight to `engine.start`, and an update can only change `engine.py`,
  `pipeline/` and `static/`.
- The page previews the same four curves live with Web Audio `BiquadFilterNode`s (same formulas: shelf slope 1, bell Q
  0.8) with a small gain compensation estimated over the speech range; the saved file is matched exactly by the
  mastering stage. Measured in the packaged app (13 s clip, brightness +6, warmth -2, air +3): same length, -16.0 LUFS
  before and after, the share of energy above 5 kHz up 3.6 dB.

## 6. AI models

- **DeepFilterNet3** (DeepFilterNet 0.5.6, MIT/Apache): noise and echo removal for voice, always used when
  available. Needs PyTorch 2.1.2, which supports Python up to 3.11, hence `.venv311` and Python 3.11 builds.
  0.5.6 imports `torchaudio.backend.common`, which newer torchaudio moved: `pipeline/ai.py` adds an alias (shim).
- **Studio voice** = Resemble Enhance (MIT, the CrispVoice approach): `pipeline/studio_core.py`, chunked 10 s with 1 s
  overlap, nfe 32, lambd 0.4, tau 0.5, then 25 % DeepFilterNet blend, peak-normalised to 0.98. Installed with
  `--no-deps` (its training extras, deepspeed and gradio, are not needed). Off by default; slow (102 s for a 13 s clip
  in the packaged app). When on, it **replaces** declip, denoise and tone (only declick, tame, studio, master run):
  declip before it dropped STOI 0.66 -> 0.54, and the full chain gave 0.35; replacing them gives 0.65. The normal
  pipeline scores 0.864 on the same case, which is why Studio voice is opt-in.
- Weights ship inside the app (`<app>/_internal/models/DeepFilterNet3/`, `.../enhancer_stage2/`, 689 MB).
  `ai.models_root()` = bundled folder, else `AR_MODELS`, else the user's local data folder. The workflow downloads them
  at build time and checks the Studio weights' SHA-256
  (`f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6`, 713,176,232 bytes).
- **The model's `hparams.yaml` was written on Linux** and stores folders as `!!python/object/apply:pathlib.PosixPath`
  followed by their parts. Windows cannot build a PosixPath, so Studio voice failed with `NotImplementedError: cannot
  instantiate 'PosixPath' on your system` (the 1.1.0 installers shipped this way; found 2026-10-18 on a real
  recording). Three guards now: `tools/clean_hparams.py` rewrites the file in the GitHub build, the build then loads
  it with `HParams.load` and fails if it cannot, and `studio_core._patch_hparams_loader` makes the app read either
  form (so an update fixes already-installed apps). Lesson: **test Studio voice on the installer GitHub built, not
  only on a local build**: the locally staged copy of this file had been cleaned by hand, which hid the bug.
- Running from source on Python 3.14 (`.venv`, no PyTorch), the models are reached through `.venv311` in a subprocess
  (`dfn_worker.py`, `studio_worker.py`). The packaged app runs them in-process (`ai.inprocess()`).

## 7. The desktop window (traps that cost real time)

- **Pass the page as `html=`, never `url=`**: given a file location, pywebview starts its own local HTTP server (it
  did, on port 26354). The app is meant to open zero ports; verified with the process tree's listening sockets.
- The page talks to Python through `window.pywebview.api`. It runs in two modes: native (bridge) or HTTP (browser
  fallback / development). `window.__AR_NATIVE = true` is injected before the page's scripts so it knows it is in the
  app even before the bridge is ready (the bridge can take over 1.5 s; that was the "app is not running" bug).
- A watchdog: if the window has not loaded within 30 s, the app restarts itself in browser mode (`launcher.py`).
- Dropping a file: the web page cannot see a dropped file's path, so `desktop.py` listens to the DOM drop event
  through pywebview and reads `pywebviewFullPath`. A dropped `.srar` is offered as an update.
- Windowed exes have `sys.stdout = None`: anything that logs (uvicorn) must get a devnull stream and
  `log_config=None`.
- Resemble Enhance's model loading calls `pandas.to_markdown` (tabulate) for a log table, which fails in a frozen app
  (`InvalidVersion: 'unknown'`): `studio_core.py` replaces `Enhancer.summarize` with a no-op, and the build copies the
  tabulate metadata.
- Do not give the `Api` class a public attribute that holds the window (`window`): pywebview enumerates the exposed
  object's public attributes and recurses into the window. It is `_window`.
- Force-killing `msedgewebview2` processes during tests can leave the next launch hanging at "creating window"; wait
  for them to exit. Never kill web-view processes you did not start (other apps use them).

### Error reports (added in Mk II)
A failed repair no longer only says "Something went wrong". `engine._error_detail` builds a plain-text report (time,
version / mark, system, file name and size, mode and settings, the stage that was running and how far it got, what had
already been done, the error type and message, the last 6000 characters of the traceback; never the recording or its
folder). The status call returns it as `detail` (and `log`), the page shows it under **Show details**, with **Copy
details** and **Open error log**, and it is appended to `errors.log` in the data folder (capped at about 200 KB).
Plain `AudioError`s (unsupported file, no such file) have no report: their message already says what is wrong. The
short message names the error type and stage, e.g. "(NotImplementedError, at: Rebuilding the voice (AI))". When a user
reports a failure, ask for the copied details first.

### The result view: scrubbing, spectrum, tone (added in Mk III)
All in `static/index.html` (search for "Players: scrubbing"). Both players are always at the same position:
- **Scrubbing.** Each picture is a `.scrub` box (`role="slider"`, arrow keys 1 s / Shift 5 s, Home / End) with pointer
  capture: press and drag seeks **both** players (`seekAll`), a playhead line and "m:ss / m:ss" labels follow (a
  `requestAnimationFrame` loop while anything plays). Playing one player pauses the other, and pausing or seeking one
  copies its position to the other, so Before / After can be switched while listening.
- **Picture** switch: **Waveform** (peak bars) or **Spectrum** (a spectrogram computed in the page: 1024-point FFT,
  0-12 kHz, 72 dB range, with a kHz scale). The browser decodes each file once (`decodeClip`); nothing is sent anywhere.
- **Tone** (native app only, `#tonebox`): presets Flat / Warmer / Brighter / Clearer and four sliders. The After
  player is routed through `MediaElementSource -> 4 filters -> gain`, created on first play (`ensureGraph`). In Spectrum
  view the After picture is re-rendered through the same filters in an `OfflineAudioContext` (debounced 220 ms), so the
  brightness change is visible. **Save** with tone flat saves the stored result; otherwise it starts a tone job
  (`start(path, {mode, fmt, base_job, tone})`, see section 5), shows its progress in the Save box, then calls
  `save(<tone job id>)`. The chosen format (WAV / MP3) at save time is used.
- Looking at the page without the app: `python bench/ui_harness.py before.wav after.wav` serves it with a mock engine
  (records `start` / `save` calls in `window.__calls`) so it can be opened and driven in any browser.

## 8. Building and packaging

- `python build.py` with the Python 3.11 environment that has PyTorch, DeepFilterNet and Resemble Enhance, and
  `AUDIO_REPAIR_MODELS` pointing at a folder with `DeepFilterNet3/` and `enhancer_stage2/`. Without PyTorch it builds a
  smaller app without the AI models (useful for quick tests of the window and updates).
- It **builds in the temp folder**, then copies to `dist/`: a sync client (OneDrive, Dropbox) locks the thousands of
  build files and PyInstaller cannot clean up. Paths given to PyInstaller are absolute (relative ones broke when
  `--specpath` moved).
- Needs `--collect-all webview` (data files and the WebView2 DLLs; `--collect-submodules` froze at window
  creation), `--collect-all clr_loader pythonnet` on Windows, `--collect-all df libdf resemble_enhance`,
  `--collect-submodules torchaudio`, `--copy-metadata tabulate`. Excludes torchvision and the worker scripts.
- Pin `numpy<2` (PyTorch 2.1.2 is built against NumPy 1). On GitHub: Python 3.11, `constraints.txt`.
- **GitHub workflow** (`build.yml`, manual "Run workflow", or a `v*` tag to attach files to a Release): Windows
  (Inno Setup via ISCC, installed with choco if missing) and macOS arm64 (`hdiutil` dmg with an Applications link and
  HANDOFF.md). A run takes about 25 minutes. Download sizes: Windows installer about 820 MB, Mac dmg about 880 MB;
  installed app about 1.5 GB (the models and PyTorch).
- Windows installer: per user (`{localappdata}\Programs\Audio Repair`, no admin), AppId
  `{6F1C3E2A-8B4D-4E7A-9C51-2D7A0B3E9F14}` (keep it: it is how upgrades replace the old install), Start-menu and
  optional desktop shortcut, uninstaller, `.srar` association (`AudioRepair.Update`).
- Mac: `--argv-emulation` (a double-clicked `.srar` arrives as an argument), Info.plist gets the `.srar` document
  type, then the bundle is re-signed ad hoc. Unsigned: users right-click, Open the first time.
- Committing from the owner's machine: git has no global identity there; earlier commits pass it on the command line.

## 9. Updates (.srar "marks")

Modelled on SoundField 3D's `.jasr` marks. An update is one file, `Audio Repair - Mk <N> update.srar`:
`"SRARMARK"`, a 4-byte big-endian header length, a JSON header (format `audio-repair-mark`, formatVersion 1, mark,
name, title, notes, created, minApp, payload size and SHA-256, and each file's path, size and SHA-256), then gzip of the
files. `app_updates.parse` checks all of it and gives a plain-English reason for any problem.

- **What it carries:** a complete copy of `static/`, `engine.py`, `pipeline/*.py`, `mark.json` and `HANDOFF.md`.
  Only those paths are accepted. **What it cannot change:** `desktop.py`, `app_updates.py`, the bundled libraries and
  the AI models; a change there needs a new installer (and a higher `APP_VERSION`, which marks name as `minApp`).
  A mark's code may only import libraries the app already contains.
- **Making one:** `python tools/make_update.py --title "..." --notes "..." [--entry entry.md]` (section 12 entry,
  handoff stamp, `mark.json`, pack, self-check, `updates/history.json`, file in `updates/out/`).
- **Installing:** double-click the file, drop it on the window, or Updates > Install an update > Choose. The app shows
  the mark's name, title and notes; Install unpacks it into the data folder (`%LOCALAPPDATA%\AudioRepair\updates\mkN`
  on Windows, `~/Library/Application Support/AudioRepair/updates/mkN` on a Mac), switches to it, and offers Restart.
  The installed program is never modified (apart from its HANDOFF.md copy on Windows). The newest five marks are kept;
  Updates lists the others with **Use** to switch back (including the built-in mark).
- **Running a mark:** `app_updates.boot()` runs first in `desktop.main()`. It puts a finder at the front of
  `sys.meta_path` that loads `engine` and `pipeline.*` from the active mark's folder ahead of the frozen copies, and
  reads the page from the mark. If the mark's engine fails to import, it switches back to the built-in copy and shows
  a notice under Updates. From source (development) marks are ignored unless `AUDIO_REPAIR_USE_MARKS=1`.
- **The handoff document:** built into the app, copied next to the program on Windows and into the data folder,
  carried by every mark. At startup and on install the newest copy (by its `handoff-mark` line) replaces older ones;
  replaced copies go to `handoff-history/` in the data folder.
- Not signed: anyone could make a `.srar`. Only install updates you made or trust.

## 10. How to test (and what counts as verified)

- `python -m pytest -q` in `.venv` and in `.venv311` (25 behavioural tests plus the update tests in
  `tests/test_updates.py`). Bench: `python -I bench/run_bench.py`; edge cases `bench/robust.py` (19 cases).
- **Self-test of a built app:** set `AUDIO_REPAIR_SELFTEST=<audio file>` (and `AUDIO_REPAIR_SELFTEST_OUT=<json>`) and
  start the exe. The app drives its own page like a user (selects the file, clicks Repair, waits, checks the players,
  saves) and writes JSON: the mark in use, the engine file loaded, the banner state, the report lines, the duration,
  saved bytes. `AUDIO_REPAIR_SELFTEST_STUDIO=1` turns Studio voice on; `AUDIO_REPAIR_SELFTEST_UPDATE=<.srar>` offers an
  update, clicks Install, and records the result and the handoff copies. `AUDIO_REPAIR_DATA_DIR` points the data
  folder somewhere else (always use a scratch folder for update tests). Breadcrumbs: `audio_repair_trace.log` and
  `audio_repair_stderr.log` in the temp folder.
- Tests added over time: `tests/test_updates.py` (update files, relative to the built-in mark, so it survives every
  release), `tests/test_errors.py` (error report, Studio settings fix, deepspeed stand-in), `tests/test_tone_after.py`
  (filter accuracy against the RBJ curves, the tone job: same length, loudness matched, brighter, repair not re-run).
- Run test builds with a scratch data folder; do not leave test marks installed in a real user's data folder.

## 11. Open items and known limits

- **The Mac app has been built on GitHub but never opened on a Mac.** The Mac `.srar` double-click (argv emulation
  plus Info.plist) is likewise untested. Intel Macs and Windows on ARM are not built.
- Studio voice scores below the normal pipeline on the synthetic bench; judge it on real recordings.
- The bench uses synthetic voices with synthetic damage: it shows direction, not how real recordings sound.
- Updates are not signed. A future version could sign marks (the public key built into the app).
- A second copy of the app opens if one is already running when a `.srar` is double-clicked (no single-instance
  handling); the update still installs from the new window.

## 12. Change log (newest first; add an entry for everything you do; see the rule at the top)

Format: **date – title.** Then Asked / Changed / Verified / Not done.

### 2026-10-18 – Mk III "Scrub the spectrum and change the tone after repairing" – update file
- **Asked:** be able to scrub on the audio spectrum, and change the brightness and tone of the voice after the
  processing without reprocessing the audio.
- **Changed:**
  - **Scrubbing.** Drag or click on the Before or After picture to jump anywhere, with a moving playhead and time
    labels; both players stay at the same position and playing one pauses the other, so Before / After can be switched
    while listening. Keyboard: arrows (Shift = 5 s), Home, End. (HANDOFF section 7, "The result view".)
  - **Spectrum view.** A Picture switch (Waveform / Spectrum): a spectrogram with a kHz scale, scrubbable like the
    waveform; the After picture follows the tone settings.
  - **Tone after the repair.** `pipeline/tone_after.py` (warmth 200 Hz low shelf, clarity 3 kHz bell, brightness 6 kHz
    high shelf, air 12 kHz high shelf, each -8..+8 dB) and `engine.start(..., base_job, tone)`: a repair now keeps its
    full-quality WAV, and a tone job applies the curves to it and re-matches the loudness; the repair stages never run
    again. The page previews it live (Web Audio filters, presets Flat / Warmer / Brighter / Clearer) and Save applies
    it. Section 5, "Tone after the repair".
  - `tests/test_tone_after.py` (17 tests); `tests/test_updates.py` now works relative to the built-in mark;
    `bench/ui_harness.py` (look at / drive the page with a mock engine); INSTALL.md steps 5-7.
- **Verified:** the page was driven in a browser against a mock engine: Choose file, Repair, result view; dragging on
  the After picture at 72 % moved both players to 9.23 s and both playheads / labels together; Spectrum view drew the
  Before (noisy) and After (clean) spectrograms with a kHz scale; the Brighter preset raised the After spectrum's
  8-12 kHz band by 3.3 dB and changed Save to "Save with your tone changes"; playing After created the filter graph
  with the slider values, switching to Before kept the position and stopped After; Save with tone made the page call
  `start({base_job, tone})` then `save(<tone job>)`, Save with tone flat called `save(<repair job>)` directly. In the
  real installed Windows app (trial Mk III installed into it): a repair, then a tone job (warmth -2, brightness +6, air
  +3): same length (615,217 samples), -16.0 LUFS before and after, the share of energy above 5 kHz up 3.6 dB. Tests
  60/60 (Python 3.11), 58 passed and 2 skipped (Python 3.14, no AI libraries).
- **Not done:** the tone sliders and the Save-with-tone flow were driven against a mock engine in a browser, not by
  clicking in the installed app's own window (the app's self-test cannot drive them); scrubbing / spectrum / tone were
  not tried on the Mac; the live preview's level compensation is an estimate (the saved file's loudness is exact);
  the tone controls are not shown in the browser fallback mode.
- **Update file:** `Audio Repair - Mk III update.srar`, needs app 1.1.0 or newer. Installing it replaces the user's older HANDOFF.md with this one.

### 2026-10-18 – Mk II "Studio voice fix and error details" – update file
- **Asked:** a user's Studio voice repair of a real 160 s voice memo ended with "This file could not be repaired /
  Something went wrong while processing this file." They asked for it to be fixed in Mk II, and for Mk II to include an
  advanced error message.
- **Changed:**
  - **Cause 1 (Windows could not read the model's settings file).** The `hparams.yaml` shipped with the Studio model
    stores folders as `!!python/object/apply:pathlib.PosixPath`, which Windows cannot build
    (`NotImplementedError: cannot instantiate 'PosixPath' on your system`). `studio_core._patch_hparams_loader` now
    reads that form (and plain paths) safely; `tools/clean_hparams.py` cleans the file in the GitHub build, which then
    loads it with `HParams.load` and fails the build if it cannot.
  - **Cause 2 (found by the new error report, behind cause 1).** `resemble_enhance` imports `deepspeed` (its training
    code) at start-up, and the GitHub-built app has no such package: `ModuleNotFoundError: No module named
    'deepspeed'`. A local environment had a hand-made fake, which hid it. `pipeline/_deepspeed_stub.py` now registers a
    stand-in when the real package is missing (training raises a clear error); `studio_core.restore` installs it.
  - **Advanced error message.** `engine._error_detail` / `_log_error`; the page's **Show details**, **Copy details**
    and **Open error log**; `errors.log` in the data folder (HANDOFF section 7, "Error reports"). The short message now
    names the error type and the stage, e.g. "(NotImplementedError, at: Rebuilding the voice (AI))".
  - Tests: `tests/test_errors.py` (error report, log cap, settings-file fix, the stand-in with deepspeed hidden).
- **Verified:** on the real, GitHub-built Windows installer (1.1.0): the failure was reproduced with the error report
  showing cause 1, then cause 2 after fixing cause 1; with Mk II installed into that app, Studio voice on a 13 s test
  clip completed (rebuilt, mastered to -16.0 LUFS, saved, no error). A full Studio run of the 160 s voice memo from
  source was past the old failure point and progressing when this was written. Normal mode on the real 160 s and 109 s
  voice memos was fine on the installed app (33 s and 46 s). Tests 43/43 in the Python 3.11 environment.
- **Not done:** the Mac app was not run; Studio voice on a Mac was never tested (the Mac build reads the same
  settings file, where PosixPath is valid, so cause 1 does not apply there, but cause 2 (deepspeed) would have). New
  installers built after this carry the cleaned settings file; the stand-in is in the app either way.
- **Update file:** `Audio Repair - Mk II update.srar`, needs app 1.1.0 or newer. Installing it replaces the user's older HANDOFF.md with this one.

### 2026-10-17 – Updates (.srar marks) and this handoff document; app 1.1.0 with Mk I built in
- **Asked:** update functionality like SoundField, with the file extension SRAR; a handoff file in GitHub and the
  install folder that updates with the app and tells any Claude how the app is made, without personal details.
- **Changed:** `app_updates.py` (format, install, switch, startup loading with fallback, handoff syncing),
  `tools/make_update.py`, `mark.json`, `updates/history.json`; `desktop.py` (Updates API, opening files the app was
  launched with, `.srar` drops, restart, self-test of updates); Updates section in `static/index.html`; installer
  registers `.srar` and installs HANDOFF.md; Mac build registers `.srar`; workflow ships HANDOFF.md; version 1.1.0.
- **Verified:** tests 36/36 in both Python environments (format damage, only-app-files rule, install / switch / keep five,
  older mark installs without switching, handoff newer-wins and history, boot loads the mark's code, a broken mark falls
  back, `make_update.py` dry run). Packaged Windows app (test build without the AI models), scratch data folder: Mk II
  installed through the page (offer shown, Install clicked), restart ran Mk II's page and `engine.py` from the data
  folder; a deliberately broken Mk III was installed, then switched off at the next start with a notice under
  Updates; a non-update `.srar` was refused with a plain message; HANDOFF.md copies updated and the old ones kept in
  `handoff-history/`. Four cold launches in a row all opened the window.
- **Also changed (window hang):** once, a cold launch never fired the page's "loaded" event (the bridge was ready, the
  watchdog fell back to browser mode). `Api.window` is now `Api._window`: pywebview walks the public attributes of the
  object it exposes, and the window object made it recurse ("maximum recursion depth exceeded" in the log). Restarts
  wait for the old copy to exit (`AUDIO_REPAIR_WAIT_PID`), and the watchdog tries the native window once more before
  falling back to the browser. The earlier hang was seen once and not reproduced after the change (4 of 4 cold
  launches fine), so treat the cause as likely, not proven.
- **Not done:** Mac double-click of `.srar` untested (no Mac); updates are not signed.

### 2026-10-17 – Tidy-up, Linux dropped, new repository
- **Asked:** lose Linux, clean unused files, neat and identifiable names; push to a new repository.
- **Changed:** Linux removed from the workflow and docs; installers renamed; build leftovers, logs, caches, old builds
  and test outputs removed; README layout table; INSTALL.md sizes and timings corrected; pushed to the new private
  repository `audio-repair-app`.
- **Verified:** the Windows installer installed silently per user (Start menu, desktop shortcut, uninstall entry); the
  installed app repaired a clipped, noisy clip in 12 s with DeepFilterNet3 (noise floor -33 -> -67 dBFS, -16.0 LUFS,
  -1.0 dBTP), both players loaded, save worked; tests 25/25; a GitHub run built both renamed installers.
- **Not done:** Mac app never launched.

### 2026-10-17 – AI models inside the app; "app is not running" banner fixed; Studio voice in the packaged app
- **Asked:** the banner showed inside the app; "ensure the AI models are downloaded with the install".
- **Changed:** `window.__AR_NATIVE` injection; weights bundled; in-process DeepFilterNet3 / Resemble Enhance
  (`ai.py`, `studio_core.py`); Studio replaces declip / denoise / tone; `Enhancer.summarize` no-op.
- **Verified:** packaged Windows app self-test, normal mode and Studio mode (102 s for 12.8 s, -16 LUFS).
- **Not done:** Studio voice remains below the normal pipeline on the bench.

### 2026-10-17 – Native desktop app, installers, Studio voice (CrispVoice approach)
- **Asked:** an actual app (exe, dmg, Linux) built on GitHub; no server; "like SoundField"; implement CrispVoice.
- **Changed:** `desktop.py` (pywebview, `html=`, bridge, drop, watchdog, fallback), `build.py`, `installer.iss`,
  `build.yml`, icon, INSTALL.md, Studio voice.
- **Verified:** zero listening ports in the app's process tree; window, repair, players, save; a 109 s MP4 voice memo
  decoded and repaired; all three platforms built on GitHub.
- **Not done:** Mac and Linux apps never launched.

### 2026-10-14 to 2026-10-17 – The repair pipeline, benchmark, real recordings, interface
- **Asked:** fix clipping, bad mic, mastering, background noise, at "high end audio clarity"; animations both ways.
- **Changed:** every stage written and tuned against the bench; declick and tame added after the real voice memos;
  analysis-driven skipping and the report; HIG-style interface with two-way reveal animations.
- **Verified:** bench tables in README.md; 19 edge cases without crashes; the owner listened: "sounds amazing".
- **Not done:** —
