"""Updates (.srar marks) and the handoff document: app_updates.py and tools/make_update.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app_updates as up  # noqa: E402


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_REPAIR_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def make(tmp_path, n, change=None, name=None):
    """A .srar for mark n made from the project, with optional edits {path: bytes}."""
    files = up.project_files(ROOT)
    files["mark.json"] = json.dumps({"mark": n, "title": "Test %d" % n}).encode()
    files[up.HANDOFF] = up.stamp_handoff(files[up.HANDOFF].decode("utf-8"), n).encode()
    files.update(change or {})
    p = tmp_path / (name or up.file_name(n))
    p.write_bytes(up.pack({"mark": n, "title": "Test %d" % n, "notes": "notes %d" % n}, files))
    return p


def test_names():
    assert up.mark_name(2) == "Mk II" and up.mark_name(14) == "Mk XIV"
    assert up.file_name(3) == "Audio Repair - Mk III update.srar"


def test_round_trip_and_damage(tmp_path):
    buf = make(tmp_path, 2).read_bytes()
    h, files = up.parse(buf)
    assert h["mark"] == 2 and "pipeline/__init__.py" in files and "HANDOFF.md" in files
    for bad, why in [(b"PK" + buf[2:], "not an Audio Repair update"),
                     (buf[:-10], "incomplete or damaged"),
                     (buf[:-1] + bytes([buf[-1] ^ 1]), "checksum")]:
        with pytest.raises(up.UpdateError, match=why):
            up.parse(bad)
    with pytest.raises(up.UpdateError, match="needs Audio Repair 9.0.0"):
        up.parse(up.pack({"mark": 2, "minApp": "9.0.0"}, up.parse(buf)[1]))


def test_only_app_files_allowed():
    for p in ("desktop.py", "../engine.py", "pipeline/sub/x.py", "static/.hidden", "C:/x.py", "app_updates.py"):
        assert not up.safe_path(p), p
    with pytest.raises(up.UpdateError):
        up.pack({"mark": 2}, {"desktop.py": b"x"})


def test_install_switch_and_keep(tmp_path, data):
    assert up.status()["running"]["mark"] == 1 and up.active() == 0
    info = up.install(make(tmp_path, 2))
    assert info["switched"] and up.active() == 2 and up.web_file("index.html").is_file()
    assert "mk2" in str(up.active_root())
    assert up.inspect(make(tmp_path, 2))["reinstall"]
    up.install(make(tmp_path, 3))
    older = up.inspect(make(tmp_path, 2, name="again.srar"))
    assert older["ok"] and older["reinstall"]
    assert up.active() == 3
    up.switch_to(2)
    assert up.active() == 2
    up.switch_to(1)                              # the built-in mark
    assert up.active() == 0 and up.status()["running"]["name"] == "Mk I"
    for n in range(4, 10):
        up.install(make(tmp_path, n))
    assert [m["mark"] for m in up.installed()] == [9, 8, 7, 6, 5]
    assert not up.inspect(make(tmp_path, 1, name="one.srar"))["ok"]       # already built in


def test_older_mark_installs_without_switching(tmp_path, data):
    up.install(make(tmp_path, 3))
    info = up.install(make(tmp_path, 2))
    assert not info["switched"] and up.active() == 3


def test_handoff_newer_wins(tmp_path, data):
    t = tmp_path / "copy" / "HANDOFF.md"
    assert up.sync_handoff(b"<!-- handoff-mark: 2 -->\nnew", [t]) == [str(t)]
    assert up.sync_handoff(b"<!-- handoff-mark: 1 -->\nold", [t]) == []          # never older over newer
    assert t.read_bytes().endswith(b"new")
    up.sync_handoff(b"<!-- handoff-mark: 3 -->\nnewer", [t])
    kept = list((data / "handoff-history").glob("*was-mark2.md"))
    assert kept and kept[0].read_bytes().endswith(b"new")
    assert up.stamp_handoff("<!-- handoff-mark: 4 -->\nbody", 5) == "<!-- handoff-mark: 5 -->\nbody"


def test_install_updates_handoff(tmp_path, data):
    up.install(make(tmp_path, 2))
    assert up.handoff_mark((data / "HANDOFF.md").read_bytes()) == 2


def test_project_handoff_is_stamped_and_has_changelog():
    text = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    assert up.handoff_mark(text.encode()) >= 1
    assert "\nFormat: **date – title.**" in text


BOOT = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
import app_updates
n = app_updates.boot()
import engine, pipeline.declip
print(json.dumps({"n": n, "engine": engine.__file__, "declip": sys.modules["pipeline.declip"].__file__,
                  "marker": getattr(engine, "MARK_TEST", None), "notice": app_updates.status()["notice"]}))
"""


def boot(tmp_path):
    env = dict(os.environ, AUDIO_REPAIR_USE_MARKS="1")
    out = subprocess.run([sys.executable, "-c", BOOT, str(ROOT)], env=env, capture_output=True, text=True,
                         cwd=str(tmp_path), timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_boot_runs_the_marks_code(tmp_path, data):
    engine = (ROOT / "engine.py").read_bytes() + b"\nMARK_TEST = 'from mk2'\n"
    up.install(make(tmp_path, 2, {"engine.py": engine}))
    r = boot(tmp_path)
    assert r["n"] == 2 and r["marker"] == "from mk2"
    assert "mk2" in r["engine"] and "mk2" in r["declip"]


def test_broken_mark_falls_back(tmp_path, data):
    up.install(make(tmp_path, 2, {"engine.py": b"raise RuntimeError('broken on purpose')\n"}))
    r = boot(tmp_path)
    assert r["n"] == 0 and "mk2" not in r["engine"] and "could not start" in r["notice"]
    assert up.active() == 0


def test_make_update_dry_run():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "make_update.py"), "--title", "Test", "--notes",
                          "one\ntwo", "--dry-run"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "Mk II" in out.stdout and "dry run" in out.stdout
