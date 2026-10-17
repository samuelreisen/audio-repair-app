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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "Audio Repair"


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
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", NAME, "--windowed",
            "--add-data", f"static{sep}static",
            "--collect-all", "av",
            "--collect-submodules", "uvicorn",
            "--collect-submodules", "pipeline",
            "--hidden-import", "multipart", "--hidden-import", "python_multipart",
            "--exclude-module", "matplotlib", "--exclude-module", "pytest", "--exclude-module", "IPython",
            "--exclude-module", "torch", "--exclude-module", "pystoi", "--exclude-module", "noisereduce",
            "--exclude-module", "pedalboard"]
    lib = find_rnnoise()
    if lib:
        args += ["--add-binary", f"{lib}{sep}rnnoise"]
        print("bundling RNNoise:", lib)
    else:
        print("note: pyrnnoise not installed, the app will use its built-in denoiser only")
    args.append("launcher.py")
    print(" ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)
    out = ROOT / "dist" / (NAME + (".app" if sys.platform == "darwin" else ""))
    print("built:", out, "exists:", out.exists())


if __name__ == "__main__":
    main()
