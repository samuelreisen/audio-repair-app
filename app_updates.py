"""Updates ("marks") for the installed Audio Repair app, delivered as .srar files.

An update is one file, e.g. "Audio Repair - Mk II update.srar", made with tools/make_update.py. It carries a complete
copy of the parts of the app that change between releases: the window's page (static/), the audio engine (engine.py)
and the repair stages (pipeline/). Installing it never touches the installed program: the files go to the user's data
folder, and from then on the app runs that newest mark instead of the copy built into it. Older marks stay installed
(the newest five) so the user can switch back, and the built-in copy is always there to fall back to.

What an update cannot change: the window itself (desktop.py, this file) and the libraries and AI models built into the
app. A mark that needs a newer app says so (minApp), and a mark whose code does not load is switched off at startup.

File format (.srar)
    bytes 0-7  "SRARMARK"
    u32 BE     header length L
    L bytes    UTF-8 JSON header: format, formatVersion, mark, name, title, notes, created, minApp,
               payloadSize, payloadSha256, files: [{path, size, sha256}]
    rest       gzip of every file's bytes, in the order listed

Only this module and desktop.py read it. tools/make_update.py builds it with pack().
"""
import gzip
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

APP_VERSION = "1.1.0"                       # this app's own version (the installer); marks name the oldest they need
EXT = ".srar"
MAGIC = b"SRARMARK"
FORMAT = "audio-repair-mark"
KEEP = 5                                    # installed marks kept for switching back
MAX_BYTES = 200 * 1024 * 1024
SOURCE_ROOT = Path(__file__).resolve().parent

# the files a mark carries, relative and "/"-separated (the same paths as in the project folder)
ALLOWED = re.compile(r"^(HANDOFF\.md|mark\.json|engine\.py|pipeline/[A-Za-z0-9_]+\.py|static/[A-Za-z0-9_][A-Za-z0-9_.-]*)$")
REQUIRED = ("HANDOFF.md", "mark.json", "engine.py", "pipeline/__init__.py", "static/index.html")
HANDOFF = "HANDOFF.md"
SKIP = {"pipeline/studio_worker.py", "pipeline/dfn_worker.py"}     # helper scripts for the source setup only
MANAGED = ("engine", "pipeline")                                   # the Python modules a mark replaces


class UpdateError(Exception):
    """A plain-English reason the update cannot be used."""


# ---------------------------------------------------------------- names and folders
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
         (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def mark_name(n):
    s = ""
    for v, r in ROMAN:
        while n >= v:
            s += r
            n -= v
    return "Mk " + s


def file_name(n):
    return "Audio Repair - %s update%s" % (mark_name(n), EXT)


def bundled_root():
    """Where the built-in copy lives: inside the packaged app, or the project folder when run from source."""
    return Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))


def data_dir():
    if os.environ.get("AUDIO_REPAIR_DATA_DIR"):
        return Path(os.environ["AUDIO_REPAIR_DATA_DIR"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "AudioRepair"


def updates_dir():
    return data_dir() / "updates"


def _read_json(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_atomic(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".%s.tmp" % uuid.uuid4().hex[:8])
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _info(d):
    n = int(d.get("mark") or 1)
    return {"mark": n, "name": d.get("name") or mark_name(n), "title": d.get("title") or "",
            "notes": d.get("notes") or "", "created": d.get("created") or ""}


def cmp_version(a, b):
    pa = [int(x) if x.isdigit() else 0 for x in re.split(r"[.+-]", str(a))[:3]]
    pb = [int(x) if x.isdigit() else 0 for x in re.split(r"[.+-]", str(b))[:3]]
    pa += [0] * (3 - len(pa))
    pb += [0] * (3 - len(pb))
    return (pa > pb) - (pa < pb)


# ---------------------------------------------------------------- the file format
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def safe_path(p):
    return isinstance(p, str) and len(p) < 200 and bool(ALLOWED.match(p)) and ".." not in p


def pack(meta, files):
    """files: {"pipeline/x.py": bytes, ...}. Returns the .srar bytes."""
    paths = sorted(files)
    for p in paths:
        if not safe_path(p):
            raise UpdateError("An update cannot contain %s." % p)
    payload = gzip.compress(b"".join(files[p] for p in paths), compresslevel=9, mtime=0)
    header = {"format": FORMAT, "formatVersion": 1, "mark": meta["mark"], "name": mark_name(meta["mark"]),
              "title": meta.get("title", ""), "notes": meta.get("notes", ""),
              "created": meta.get("created") or time.strftime("%Y-%m-%d"), "minApp": meta.get("minApp", APP_VERSION),
              "payloadSize": len(payload), "payloadSha256": _sha(payload),
              "files": [{"path": p, "size": len(files[p]), "sha256": _sha(files[p])} for p in paths]}
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return MAGIC + len(hb).to_bytes(4, "big") + hb + payload


def parse(buf, app_version=APP_VERSION):
    """Check a .srar file and unpack it in memory: (header, {path: bytes}). Raises UpdateError at the first problem."""
    if not isinstance(buf, (bytes, bytearray)) or len(buf) < 12 or buf[:8] != MAGIC:
        raise UpdateError("This is not an Audio Repair update file.")
    n = int.from_bytes(buf[8:12], "big")
    if n < 2 or 12 + n > len(buf):
        raise UpdateError("The update file is damaged (its header is cut short).")
    try:
        h = json.loads(bytes(buf[12:12 + n]).decode("utf-8"))
    except ValueError:
        raise UpdateError("The update file is damaged (its header cannot be read).")
    if not isinstance(h, dict) or h.get("format") != FORMAT:
        raise UpdateError("This is not an Audio Repair update file.")
    if h.get("formatVersion") != 1:
        raise UpdateError("This update uses a newer format than this app understands. Install the latest Audio Repair first.")
    if not isinstance(h.get("mark"), int) or isinstance(h.get("mark"), bool) or h["mark"] < 1:
        raise UpdateError("The update file is damaged (it does not say which mark it is).")
    if h.get("minApp") and cmp_version(app_version, h["minApp"]) < 0:
        raise UpdateError("%s needs Audio Repair %s or newer, and this app is %s. Install the newer app first."
                          % (mark_name(h["mark"]), h["minApp"], app_version))
    payload = bytes(buf[12 + n:])
    if len(payload) != h.get("payloadSize"):
        raise UpdateError("The update file is incomplete or damaged. Copy it again.")
    if _sha(payload) != str(h.get("payloadSha256", "")).lower():
        raise UpdateError("The update file is damaged (its checksum does not match). Copy it again.")
    try:
        data = gzip.decompress(payload)
    except (OSError, EOFError, ValueError):
        raise UpdateError("The update file is damaged (its data cannot be unpacked).")
    listed = h.get("files")
    if not isinstance(listed, list) or not listed:
        raise UpdateError("The update file is damaged (it lists no files).")
    files, off = {}, 0
    for f in listed:
        p = f.get("path") if isinstance(f, dict) else None
        if not safe_path(p):
            raise UpdateError("The update file contains a file it may not: %r." % (p,))
        if p.lower() in (k.lower() for k in files):
            raise UpdateError("The update file lists %s twice." % p)
        size = f.get("size")
        if not isinstance(size, int) or size < 0 or off + size > len(data):
            raise UpdateError("The update file is damaged (%s is cut short)." % p)
        b = data[off:off + size]
        off += size
        if _sha(b) != str(f.get("sha256", "")).lower():
            raise UpdateError("The update file is damaged (%s does not match its checksum)." % p)
        files[p] = b
    if off != len(data):
        raise UpdateError("The update file is damaged (it has unexpected data at the end).")
    missing = [p for p in REQUIRED if p not in files]
    if missing:
        raise UpdateError("The update file is incomplete (%s is missing)." % ", ".join(missing))
    return h, files


def project_files(root=SOURCE_ROOT):
    """Every file a mark made from the project folder carries: {relative path: bytes}."""
    root = Path(root)
    out = {"mark.json": (root / "mark.json").read_bytes(), "engine.py": (root / "engine.py").read_bytes(),
           HANDOFF: (root / HANDOFF).read_bytes()}
    for p in sorted((root / "pipeline").glob("*.py")):
        rel = "pipeline/" + p.name
        if rel not in SKIP:
            out[rel] = p.read_bytes()
    for p in sorted((root / "static").iterdir()):
        if p.is_file() and not p.name.startswith("."):
            out["static/" + p.name] = p.read_bytes()
    return out


# ---------------------------------------------------------------- the handoff document
# HANDOFF.md says how this app is made, for any Claude (or person) picking it up. It is built into the app, copied next
# to the installed program and into the data folder, and carried by every mark. Its first line
# <!-- handoff-mark: N --> says which mark's document it is: a copy is only ever replaced by a NEWER document, and the
# copy it replaces is kept in handoff-history/ in the data folder.
def handoff_mark(b):
    m = re.search(rb"<!--\s*handoff-mark:\s*(\d+)\s*-->", bytes(b)[:4000])
    return int(m.group(1)) if m else 0


def stamp_handoff(text, n):
    body = re.sub(r"(?m)^[ \t]*<!--\s*handoff-mark:\s*\d+\s*-->[ \t]*\r?\n", "", text.lstrip("﻿"))
    return "<!-- handoff-mark: %d -->\n" % n + body


def handoff_targets():
    """Where the user's copies live: the data folder, and on Windows the install folder next to the program
    (a per-user install, so it is writable; a Mac app bundle is never modified)."""
    out = [data_dir() / HANDOFF]
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        out.append(Path(sys.executable).resolve().parent / HANDOFF)
    return out


def sync_handoff(buf, targets=None):
    """Bring each copy up to date with buf when buf is newer. Returns the files that were replaced."""
    incoming, done = handoff_mark(buf), []
    for t in targets or handoff_targets():
        try:
            cur = t.read_bytes() if t.is_file() else None
            if cur == buf or (cur is not None and handoff_mark(cur) >= incoming):
                continue
            if cur is not None:
                hist = data_dir() / "handoff-history"
                hist.mkdir(parents=True, exist_ok=True)
                (hist / ("HANDOFF-%s-was-mark%d.md" % (time.strftime("%Y%m%d-%H%M%S"), handoff_mark(cur)))).write_bytes(cur)
            t.parent.mkdir(parents=True, exist_ok=True)
            tmp = t.with_name(t.name + ".tmp")
            tmp.write_bytes(buf)
            os.replace(tmp, t)
            done.append(str(t))
        except OSError:
            pass                              # never fatal: the app and the update work without it
    return done


def sync_handoffs_now():
    """At startup: the newest document available (built in, or from an installed mark) goes to every copy."""
    best = None
    for p in [bundled_root() / HANDOFF] + [_mark_dir(m["mark"]) / HANDOFF for m in installed()]:
        try:
            b = p.read_bytes()
        except OSError:
            continue
        if best is None or handoff_mark(b) > handoff_mark(best):
            best = b
    return sync_handoff(best) if best else []


# ---------------------------------------------------------------- what is installed
def bundled():
    return _info(_read_json(bundled_root() / "mark.json", {"mark": 1}))


def _mark_dir(n):
    return updates_dir() / ("mk%d" % n)


def _has(n):
    return (_mark_dir(n) / "static" / "index.html").is_file() and (updates_dir() / ("mk%d.json" % n)).is_file()


def installed():
    out = []
    d = updates_dir()
    if d.is_dir():
        for p in d.glob("mk*.json"):
            m = re.match(r"^mk(\d+)\.json$", p.name)
            if m and _has(int(m.group(1))):
                info = _info(dict(_read_json(p, {}), mark=int(m.group(1))))
                out.append(info)
    return sorted(out, key=lambda i: -i["mark"])


def _state():
    s = _read_json(updates_dir() / "state.json", {})
    return s if isinstance(s, dict) else {}


def active():
    """The installed mark being used, or 0 for the built-in copy."""
    n = _state().get("active")
    return n if isinstance(n, int) and n > bundled()["mark"] and _has(n) else 0


def active_root():
    n = active()
    return _mark_dir(n) if n else bundled_root()


def web_file(*parts):
    """A file of the page (static/...) from the mark in use."""
    return active_root().joinpath("static", *parts)


def status():
    n, b = active(), bundled()
    running = _info(dict(_read_json(updates_dir() / ("mk%d.json" % n), {}), mark=n)) if n else b
    return {"app_version": APP_VERSION, "running": running, "bundled": b, "active": n,
            "installed": installed(), "notice": _state().get("notice") or ""}


def inspect(path):
    """Read an update file without installing it: what it is and whether it can be installed."""
    p = Path(path)
    try:
        if not p.is_file():
            raise UpdateError("The update file cannot be found.")
        if p.stat().st_size > MAX_BYTES:
            raise UpdateError("This file is far too large to be an Audio Repair update.")
        h, files = parse(p.read_bytes())
    except UpdateError as e:
        return {"ok": False, "error": str(e), "file": p.name}
    except OSError as e:
        return {"ok": False, "error": "The update file cannot be read (%s)." % e.strerror, "file": p.name}
    info = _info(h)
    b = bundled()
    highest = max([b["mark"], active()] + [m["mark"] for m in installed()])
    info.update(ok=True, file=p.name, path=str(p), files=len(files), minApp=h.get("minApp", ""))
    if info["mark"] <= b["mark"]:
        info.update(ok=False, error="%s is already part of this app, which is %s." % (info["name"], b["name"]))
    elif _has(info["mark"]):
        info["reinstall"] = True
    elif info["mark"] < highest:
        info["older"] = True                  # allowed: it is installed alongside, and not switched to
    return info


def install(path):
    """Install an update file and switch to it (from the next start). Returns its info."""
    info = inspect(path)
    if not info.get("ok"):
        raise UpdateError(info.get("error") or "This update cannot be installed.")
    h, files = parse(Path(path).read_bytes())
    n = h["mark"]
    root = updates_dir()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / (".staging-" + uuid.uuid4().hex[:12])
    try:
        for rel, b in files.items():
            fp = staging.joinpath(*rel.split("/"))
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b)
        shutil.rmtree(_mark_dir(n), ignore_errors=True)
        os.replace(staging, _mark_dir(n))
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise UpdateError("The update could not be saved (%s)." % e.strerror)
    meta = _info(h)
    meta.update(minApp=h.get("minApp", ""), installed=time.strftime("%Y-%m-%d %H:%M"), files=len(files))
    _write_atomic(root / ("mk%d.json" % n), json.dumps(meta, indent=1, ensure_ascii=False))
    switch = not info.get("older")
    if switch:
        _write_atomic(root / "state.json", json.dumps({"active": n}))
    for old in installed()[KEEP:]:            # keep the newest few, never the one in use
        if old["mark"] not in (n, active()):
            shutil.rmtree(_mark_dir(old["mark"]), ignore_errors=True)
            (root / ("mk%d.json" % old["mark"])).unlink(missing_ok=True)
    meta["switched"] = switch
    meta["handoff"] = sync_handoff(files[HANDOFF])
    return meta


def switch_to(n):
    """Use mark n from the next start (0 or the built-in mark = the copy built into the app)."""
    b = bundled()
    if n in (0, b["mark"]):
        n = 0
    elif not _has(n):
        raise UpdateError("%s is not installed." % mark_name(n))
    _write_atomic(updates_dir() / "state.json", json.dumps({"active": n}))
    return status()


def cleanup():
    d = updates_dir()
    if d.is_dir():
        for p in d.glob(".staging-*"):           # an install interrupted by a crash
            shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------- running a mark's code
class _MarkFinder(importlib.abc.MetaPathFinder):
    """Loads engine and pipeline.* from the mark in use, ahead of the copies built into the app."""

    def __init__(self, root):
        self.root = str(root)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top not in MANAGED:
            return None
        where = [self.root] if fullname == top else [os.path.join(self.root, *fullname.split(".")[:-1])]
        return importlib.machinery.PathFinder.find_spec(fullname, where)


_finder = None


def _forget_modules():
    for name in list(sys.modules):
        if name.split(".", 1)[0] in MANAGED:
            del sys.modules[name]


def boot():
    """Call once at startup, before engine or pipeline is imported. Uses the active mark's code if it loads,
    otherwise switches back to the built-in copy and leaves a notice for the Updates section."""
    global _finder
    cleanup()
    if getattr(sys, "frozen", False):
        sync_handoffs_now()
    # Running from the project folder (development), the project's own files are the latest: marks are only used by
    # the installed app, unless AUDIO_REPAIR_USE_MARKS=1 (for testing).
    if not getattr(sys, "frozen", False) and os.environ.get("AUDIO_REPAIR_USE_MARKS") != "1":
        return 0
    n = active()
    if not n:
        return 0
    _finder = _MarkFinder(_mark_dir(n))
    sys.meta_path.insert(0, _finder)
    _forget_modules()
    try:
        import engine  # noqa: F401  (loads the whole pipeline)
        loaded = Path(sys.modules["engine"].__file__).resolve()
        if _mark_dir(n).resolve() not in loaded.parents:
            raise ImportError("the mark's engine was not used (%s)" % loaded)
        return n
    except Exception as e:                    # the mark's code does not run on this app: fall back to the built-in copy
        sys.meta_path.remove(_finder)
        _finder = None
        _forget_modules()
        try:
            _write_atomic(updates_dir() / "state.json", json.dumps({
                "active": 0, "notice": "%s could not start on this app (%s), so the built-in %s is used. Install a newer "
                                       "update, or the newest Audio Repair app." % (mark_name(n), e, bundled()["name"])}))
        except OSError:
            pass
        return 0
