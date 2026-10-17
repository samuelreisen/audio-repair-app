"""Audio Repair as a native desktop app: its own window, no server, no network port.

The page (static/index.html) talks to this module directly through pywebview's JavaScript bridge. If the native
window cannot be created (missing system web view, no display), the app falls back to the browser version
(``launcher.py``: a small local server plus your browser) instead of failing.
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
        self.window = None
        self._paths = {}

    # --- environment
    def capabilities(self):
        import engine
        return engine.capabilities()

    # --- choosing a recording
    def pick_file(self):
        import webview
        res = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=AUDIO_TYPES)
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
        res = self.window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name)
        if not res:
            return {"cancelled": True}
        dest = res if isinstance(res, str) else res[0]
        engine.save_copy(job_id, dest)
        return {"saved": dest}

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


def _watchdog(loaded):
    """If the native window never finishes loading (a stuck system web view), start the browser version instead
    of leaving the user with a frozen window."""
    import subprocess
    timeout = float(os.environ.get("AUDIO_REPAIR_LOAD_TIMEOUT") or LOAD_TIMEOUT_S)
    if loaded.wait(timeout):
        return
    try:
        with open(os.path.join(os.environ.get("TEMP") or "/tmp", "audio_repair.log"), "a", encoding="utf-8") as f:
            f.write("native window did not load within %s s; relaunching in browser mode" % timeout + chr(10))
    except OSError:
        pass
    env = dict(os.environ, AUDIO_REPAIR_BROWSER="1")
    env.pop("AUDIO_REPAIR_LOAD_TIMEOUT", None)
    cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
    subprocess.Popen(cmd, env=env)
    os._exit(0)


def main():
    _setup_environment()
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
    page = Path(_bundled("static", "index.html")).read_text(encoding="utf-8")
    try:
        api.window = webview.create_window("Audio Repair", html=page, js_api=api, width=560, height=900,
                                           min_size=(380, 560), text_select=False)
        _trace("creating window")
        def on_loaded():
            _trace("page loaded")
            loaded.set()
            threading.Thread(target=_enable_drop, args=(api,), daemon=True).start()
        api.window.events.loaded += on_loaded
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
    js = api.window.evaluate_js
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
        r["native_detected"] = js("native")
        _trace("native = %r" % r["native_detected"])
        r["offline_banner_hidden"] = js("document.getElementById('offline').hidden")
        js("window.__dropped(%s)" % json.dumps(api.use_path(sample)))
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
        api.window.destroy()


def _enable_drop(api):
    """Dropping a file on the window: the web page cannot see the file's path, the app can."""
    try:
        from webview.dom import DOMEventHandler

        def on_drop(e):
            files = (e.get("dataTransfer") or {}).get("files") or []
            path = files[0].get("pywebviewFullPath") if files else None
            info = api.use_path(path) if path else None
            api.window.evaluate_js("window.__dropped(%s)" % json.dumps(info))

        api.window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
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
