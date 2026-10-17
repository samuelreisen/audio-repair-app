"""Audio Repair as a native desktop app: its own window, no server, no network port.

The page (static/index.html) talks to this module directly through pywebview's JavaScript bridge. If the native
window cannot be created (missing system web view, no display), the app falls back to the browser version
(``launcher.py``: a small local server plus your browser) instead of failing.

Updates: app_updates.boot() runs first, so the page, engine and pipeline come from the newest installed .srar update
("mark") when there is one. Opening a .srar file with the app (double-click), dropping it on the window or choosing it
under Updates offers to install it.
"""
import base64
import glob
import json
import os
import sys
from pathlib import Path


def _bundled(*parts):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def _setup_environment():
    if os.environ.get("AUDIO_REPAIR_SELFTEST"):          # self-test: keep the window library's own log
        os.environ.setdefault("PYWEBVIEW_LOG", "debug")
        logf = open(os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair_stderr.log"), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = logf
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if getattr(sys, "frozen", False) and not os.environ.get("RNNOISE_LIB"):
        found = glob.glob(_bundled("rnnoise", "*rnnoise*"))
        if found:
            os.environ["RNNOISE_LIB"] = found[0]


UPDATE_TYPES = ("Audio Repair update (*.srar)", "All files (*.*)")
AUDIO_TYPES = ("Audio files (*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac;*.mp4;*.opus;*.wma)", "All files (*.*)")
MAX_PREVIEW_BYTES = 80 * 1024 * 1024        # larger files are repaired but not previewed in the page
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg",
        ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".aac": "audio/aac", ".wma": "audio/x-ms-wma"}


def _trace(msg):
    """Breadcrumbs for diagnosing the packaged app (only when the self-test is on)."""
    if os.environ.get("AUDIO_REPAIR_SELFTEST"):
        import time
        with open(os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair_trace.log"), "a", encoding="utf-8") as f:
            f.write("%.1f %s" % (time.time() % 1000, msg) + chr(10))


class Api:
    """Methods the page can call as window.pywebview.api.<name>(...)."""

    def __init__(self):
        self._window = None
        self._paths = {}

    # --- environment
    def capabilities(self):
        import engine
        return engine.capabilities()

    # --- choosing a recording
    def pick_file(self):
        import webview
        res = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=AUDIO_TYPES)
        if not res:
            return None
        return self.use_path(res[0] if isinstance(res, (list, tuple)) else res)

    def use_path(self, path):
        p = Path(path)
        if not p.is_file():
            return None
        return {"path": str(p), "name": p.name, "size": p.stat().st_size}

    # --- running a repair
    def start(self, path, options):
        import engine
        from pipeline.io import AudioError
        try:
            return {"id": engine.start(path, **(options or {}))}
        except AudioError as e:
            return {"error": str(e)}

    def status(self, job_id):
        import engine
        return engine.status(job_id)

    # --- results
    def read_original(self, path):
        return self._read(path)

    def read_result(self, job_id):
        import engine
        found = engine.output_path(job_id)
        return self._read(found[0]) if found else None

    def _read(self, path):
        p = Path(path)
        if not p.is_file() or p.stat().st_size > MAX_PREVIEW_BYTES:
            return None
        return {"mime": MIME.get(p.suffix.lower(), "application/octet-stream"),
                "data": base64.b64encode(p.read_bytes()).decode("ascii")}

    def save(self, job_id):
        import engine
        import webview
        found = engine.output_path(job_id)
        if not found:
            return {"error": "The repaired file is no longer available. Repair the recording again."}
        src, name = found
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name)
        if not res:
            return {"cancelled": True}
        dest = res if isinstance(res, str) else res[0]
        engine.save_copy(job_id, dest)
        return {"saved": dest}

    # --- updates (.srar)
    def updates_status(self):
        import app_updates
        return app_updates.status()

    def pick_update(self):
        import webview
        res = self._window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=UPDATE_TYPES)
        if not res:
            return None
        return self.inspect_update(res[0] if isinstance(res, (list, tuple)) else res)

    def inspect_update(self, path):
        import app_updates
        return app_updates.inspect(path)

    def install_update(self, path):
        import app_updates
        try:
            return app_updates.install(path)
        except app_updates.UpdateError as e:
            return {"error": str(e)}

    def switch_version(self, mark):
        import app_updates
        try:
            return app_updates.switch_to(int(mark))
        except (app_updates.UpdateError, ValueError) as e:
            return {"error": str(e)}

    def restart(self):
        """Start a fresh copy of the app (which loads the chosen version) and close this one."""
        import subprocess
        cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("AUDIO_REPAIR_SELFTEST")}
        env["AUDIO_REPAIR_WAIT_PID"] = str(os.getpid())   # the new window opens once this one has gone
        subprocess.Popen(cmd, env=env, close_fds=True)
        self._window.destroy()

    def reveal(self, path):
        import subprocess
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except OSError:
            pass


LOAD_TIMEOUT_S = 30


def _wait_for_exit(pid, timeout=20.0):
    """Wait until process pid has ended (the copy of the app that restarted us), so two windows never start up
    against the same web view data at once."""
    import time
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000, False, pid)            # SYNCHRONIZE
        if h:
            k.WaitForSingleObject(h, int(timeout * 1000))
            k.CloseHandle(h)
    else:
        end = time.time() + timeout
        while time.time() < end:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.2)
    time.sleep(1.0)                                          # its web view helpers close just after it


def _watchdog(loaded):
    """If the native window never finishes loading (a stuck system web view), start a fresh copy of the app once;
    if that one is stuck too, start the browser version instead of leaving the user with a frozen window."""
    import subprocess
    timeout = float(os.environ.get("AUDIO_REPAIR_LOAD_TIMEOUT") or LOAD_TIMEOUT_S)
    if loaded.wait(timeout):
        return
    retry = not os.environ.get("AUDIO_REPAIR_NATIVE_RETRY")
    try:
        with open(os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair.log"), "a", encoding="utf-8") as f:
            f.write("native window did not load within %s s; %s" % (timeout, "trying once more" if retry else
                                                                     "relaunching in browser mode") + chr(10))
    except OSError:
        pass
    env = dict(os.environ, AUDIO_REPAIR_WAIT_PID=str(os.getpid()))
    if retry:
        env["AUDIO_REPAIR_NATIVE_RETRY"] = "1"
    else:
        env["AUDIO_REPAIR_BROWSER"] = "1"
        env.pop("AUDIO_REPAIR_LOAD_TIMEOUT", None)
    cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
    subprocess.Popen(cmd, env=env)
    os._exit(0)


def _opened_with():
    """A file the app was opened with (double-clicking a .srar update or an audio file, "Open with")."""
    for a in sys.argv[1:]:
        if not a.startswith("-") and os.path.isfile(a):
            return os.path.abspath(a)
    return None


def _offer_file(api, path):
    """Show a file in the page: an update is offered for installing, anything else is the recording to repair."""
    if path.lower().endswith(".srar"):
        api._window.evaluate_js("window.__updateFile(%s)" % json.dumps(api.inspect_update(path)))
    else:
        api._window.evaluate_js("window.__dropped(%s)" % json.dumps(api.use_path(path)))


def main():
    if os.environ.get("AUDIO_REPAIR_WAIT_PID"):
        _wait_for_exit(os.environ.pop("AUDIO_REPAIR_WAIT_PID"))
    _setup_environment()
    import app_updates
    app_updates.boot()                                   # before anything imports engine / pipeline
    if os.environ.get("AUDIO_REPAIR_BROWSER"):           # started by the watchdog: no native window
        import launcher
        return launcher.main()
    try:
        import webview
    except Exception:
        return _fallback("pywebview is not available")
    import threading
    loaded = threading.Event()
    threading.Thread(target=_watchdog, args=(loaded,), daemon=True).start()
    api = Api()
    # Hand the window the page itself rather than a file location: given a path, pywebview starts its own little
    # local web server, and this app is meant to run with no server and no network port at all.
    page = app_updates.web_file("index.html").read_text(encoding="utf-8")
    page = page.replace("<head>", "<head><script>window.__AR_NATIVE = true;</script>", 1)   # "you are in the app"
    try:
        api._window = webview.create_window("Audio Repair", html=page, js_api=api, width=560, height=900,
                                           min_size=(380, 560), text_select=False)
        _trace("creating window")
        def on_loaded():
            _trace("page loaded")
            loaded.set()
            threading.Thread(target=_enable_drop, args=(api,), daemon=True).start()
            opened = _opened_with()
            if opened:
                _offer_file(api, opened)
        api._window.events.loaded += on_loaded
        if os.environ.get("AUDIO_REPAIR_SELFTEST"):          # automated check of the packaged app
            webview.start(_selftest, (api,))
        else:
            webview.start()
    except Exception as e:                       # no usable system web view
        return _fallback(str(e))


def _selftest(api):
    """Drive our own page like a user would and write what happened to AUDIO_REPAIR_SELFTEST_OUT."""
    import time
    import engine
    out = os.environ.get("AUDIO_REPAIR_SELFTEST_OUT") or os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair_selftest.json")
    sample = os.environ["AUDIO_REPAIR_SELFTEST"]
    r = {}
    js = api._window.evaluate_js
    try:
        _trace("selftest thread started")
        time.sleep(4)
        _trace("evaluating page state")
        try:                                           # why can (or can't) the bundled decoder open this file?
            from pipeline.io import _av_decode
            a, sr_ = _av_decode(sample)
            r["av_probe"] = "ok: %s channels, %d Hz, %.1f s" % (a.shape[0], sr_, a.shape[1] / sr_)
        except Exception as e:
            import traceback
            r["av_probe"] = "FAILED: " + traceback.format_exc()[-600:]
        import app_updates
        r["running_mark"] = app_updates.status()["running"]["name"]
        r["engine_file"] = sys.modules["engine"].__file__ if "engine" in sys.modules else "(not loaded yet)"
        r["page_version"] = js("document.getElementById('ver').textContent")
        r["updates_notice"] = app_updates.status()["notice"]
        upd = os.environ.get("AUDIO_REPAIR_SELFTEST_UPDATE")
        if upd:                                        # offer the update like a double-click would, then click Install
            _offer_file(api, upd)
            time.sleep(1)
            r["update_offer"] = js("document.getElementById('upd-title').textContent")
            js("document.getElementById('upd-install').click()")
            for _ in range(30):
                time.sleep(1)
                if js("document.getElementById('upd-msg').textContent"):
                    break
            r["update_result"] = js("document.getElementById('upd-msg').textContent")
            r["handoff_copies"] = [str(t) for t in app_updates.handoff_targets() if t.is_file()]
            r["handoff_marks"] = [app_updates.handoff_mark(t.read_bytes()) for t in app_updates.handoff_targets() if t.is_file()]
        r["native_detected"] = js("native")
        _trace("native = %r" % r["native_detected"])
        r["offline_banner_hidden"] = js("document.getElementById('offline').hidden")
        js("window.__dropped(%s)" % json.dumps(api.use_path(sample)))
        r["studio_switch_enabled"] = js("!document.getElementById('studio').disabled")
        if os.environ.get("AUDIO_REPAIR_SELFTEST_STUDIO") == "1":
            js("document.getElementById('studio').checked = true")
        r["file_shown"] = js("document.getElementById('fname').textContent")
        _trace("file dropped, clicking Repair")
        js("document.getElementById('go').click()")
        t0 = time.time()
        for _ in range(400):
            time.sleep(1)
            if js("!document.getElementById('result').hidden") or js("document.getElementById('err').textContent"):
                break
        _trace("job finished or errored")
        r["seconds"] = round(time.time() - t0)
        r["error"] = js("document.getElementById('err').textContent")
        time.sleep(2)
        r["applied"] = js("[...document.querySelectorAll('#report li')].map(l=>l.textContent).filter(t=>t.startsWith('Applied'))")
        r["players_loaded"] = js("document.getElementById('orig').src.startsWith('blob:') && document.getElementById('fixed').src.startsWith('blob:')")
        r["fixed_duration_s"] = js("document.getElementById('fixed').duration")
        dest = os.path.join(os.path.dirname(out), "audio_repair_selftest_saved.wav")
        engine.save_copy(js("jobId"), dest)
        r["saved_bytes"] = os.path.getsize(dest)
    except Exception as e:
        r["exception"] = repr(e)
    finally:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        api._window.destroy()


def _enable_drop(api):
    """Dropping a file on the window: the web page cannot see the file's path, the app can."""
    try:
        from webview.dom import DOMEventHandler

        def on_drop(e):
            files = (e.get("dataTransfer") or {}).get("files") or []
            path = files[0].get("pywebviewFullPath") if files else None
            if path:
                _offer_file(api, path)

        api._window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
    except Exception:
        pass                                   # choosing a file with the button still works


def _fallback(reason):
    import traceback
    with open(os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair.log"), "a", encoding="utf-8") as f:
        f.write("native window unavailable (%s); using the browser version\n" % reason)
        f.write(traceback.format_exc() + "\n")
    import launcher
    launcher.main()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
