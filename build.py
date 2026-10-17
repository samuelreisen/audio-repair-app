"""Build the desktop app with PyInstaller (works on Windows, macOS and Linux; run it on the platform you want).

    python build.py

Output: dist/Audio Repair/  (Windows / Linux folder with the program)  or  dist/Audio Repair.app  (macOS).
The GitHub workflow (.github/workflows/build.yml) runs this on all three systems and wraps the result
into an installer-style file for each (zip with .exe, .dmg, AppImage / tar.gz).
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "Audio Repair"
VERSION = "1.0.0"


def find_rnnoise():
    import importlib.util
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or not spec.submodule_search_locations:
        return None
    for d in spec.submodule_search_locations:
        for pat in ("*rnnoise*.dll", "*rnnoise*.so*", "*rnnoise*.dylib"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    return None


def main():
    sep = os.pathsep
    # Build outside the project folder: if it lives in OneDrive / Dropbox, the sync client locks the build's
    # thousands of files and PyInstaller cannot clean up after itself. The finished app is copied into dist/.
    scratch = Path(os.environ.get("AUDIO_REPAIR_BUILD_DIR") or Path(tempfile.gettempdir()) / "audio_repair_build")
    shutil.rmtree(scratch, ignore_errors=True)
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", NAME, "--windowed",
            "--distpath", str(scratch / "dist"), "--workpath", str(scratch / "work"), "--specpath", str(scratch),
            "--add-data", f"{ROOT / 'static'}{sep}static",
            "--collect-all", "av",
            "--collect-submodules", "uvicorn",
            "--collect-submodules", "pipeline",
            "--collect-all", "webview",                  # the native window, including its WebView2 helper DLLs
            "--hidden-import", "engine", "--hidden-import", "launcher",
            "--hidden-import", "multipart", "--hidden-import", "python_multipart",
            "--exclude-module", "pytest", "--exclude-module", "IPython", "--exclude-module", "torchvision",
            "--exclude-module", "pipeline.studio_worker", "--exclude-module", "pipeline.dfn_worker",
            "--exclude-module", "pystoi", "--exclude-module", "noisereduce", "--exclude-module", "pedalboard"]
    # AI models (DeepFilterNet3, Studio voice): included when this Python has PyTorch, with the model weights from
    # AUDIO_REPAIR_MODELS (a folder holding DeepFilterNet3/ and enhancer_stage2/) shipped inside the app.
    import importlib.util
    ai = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchaudio", "df", "resemble_enhance"))
    models = os.environ.get("AUDIO_REPAIR_MODELS")
    if ai:
        if not models or not (Path(models) / "DeepFilterNet3" / "config.ini").exists()                 or not (Path(models) / "enhancer_stage2" / "hparams.yaml").exists():
            sys.exit("AI build needs AUDIO_REPAIR_MODELS pointing at a folder with DeepFilterNet3/ and enhancer_stage2/")
        args += ["--collect-all", "df", "--collect-all", "libdf", "--collect-all", "resemble_enhance",
                 "--collect-submodules", "torchaudio", "--hidden-import", "pipeline.ai", "--hidden-import",
                 "pipeline.studio_core", "--add-data", f"{models}{sep}models"]
        print("AI build: PyTorch + DeepFilterNet3 + Studio voice, models from", models)
    else:
        args += ["--exclude-module", "torch", "--exclude-module", "torchaudio", "--exclude-module", "df",
                 "--exclude-module", "resemble_enhance", "--exclude-module", "matplotlib"]
        print("note: no PyTorch in this Python: building without the AI models (built-in denoiser only)")
    if sys.platform == "win32":                         # Windows window = WebView2 through pythonnet
        args += ["--collect-all", "clr_loader", "--collect-all", "pythonnet", "--icon", str(ROOT / "assets" / "icon.ico")]
    elif sys.platform == "darwin":
        args += ["--icon", str(ROOT / "assets" / "icon.png"), "--osx-bundle-identifier", "com.audiorepair.app"]
    onefile = os.environ.get("AUDIO_REPAIR_ONEFILE") == "1"   # one self-contained .exe ("portable"), slower to start
    if onefile:
        args.append("--onefile")
    lib = find_rnnoise()
    if lib:
        args += ["--add-binary", f"{lib}{sep}rnnoise"]
        print("bundling RNNoise:", lib)
    else:
        print("note: pyrnnoise not installed, the app will use its built-in denoiser only")
    args.append(str(ROOT / "desktop.py"))
    print(" ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)
    (ROOT / "dist").mkdir(exist_ok=True)
    if onefile:
        exe = NAME + (".exe" if sys.platform == "win32" else "")
        out = ROOT / "dist" / ("Audio Repair Portable" + (".exe" if sys.platform == "win32" else ""))
        shutil.copyfile(scratch / "dist" / exe, out)
    else:
        built = scratch / "dist" / (NAME + (".app" if sys.platform == "darwin" else ""))
        out = ROOT / "dist" / built.name
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(built, out, symlinks=True, dirs_exist_ok=True)   # tolerate files a sync client still holds
    print("built:", out, "exists:", out.exists())


if __name__ == "__main__":
    main()
