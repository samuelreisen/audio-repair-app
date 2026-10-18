"""Make the Studio voice model's settings file readable on Windows.

    python tools/clean_hparams.py models/enhancer_stage2/hparams.yaml

The file was written on Linux and stores folders as "!!python/object/apply:pathlib.PosixPath" followed by their parts.
Windows cannot build a PosixPath, so Studio voice failed there with "NotImplementedError: cannot instantiate
'PosixPath' on your system". The folders are only training settings, so they are written as plain text instead.
The GitHub build runs this on the downloaded file; the app itself also copes with the original file
(pipeline/studio_core.py), so an older installer's copy works after an update.
"""
import re
import sys
from pathlib import Path

PATH_TAG = re.compile(r"!!python/object/apply:pathlib\.\w*Path[ \t]*\r?\n((?:[ \t]*-[^\r\n]*\r?\n?)+)")


def plain_paths(text):
    def one(m):
        parts = [ln.strip()[1:].strip() for ln in m.group(1).splitlines() if ln.strip()]
        return "'" + "/".join(parts) + "'" + chr(10)
    return PATH_TAG.sub(one, text)


def main(path):
    f = Path(path)
    text = plain_paths(f.read_text(encoding="utf-8"))
    if "pathlib" in text:
        sys.exit("%s still has Python path tags" % f)
    f.write_text(text, encoding="utf-8")
    print("cleaned", f)
    print(text[:160])


if __name__ == "__main__":
    main(sys.argv[1])
