"""Browser-mode fallback for Audio Repair (desktop.py switches to it if the native window cannot start).

Starts the local server on a free port, opens the browser, and shows a small window with Open and Quit buttons so
there is always a visible way to close the app. Falls back to a console loop if no display is available.
"""
import glob
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser


def _bundled(*parts):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def _setup_environment():
    # A windowed (no console) program has no stdout / stderr; give libraries something harmless to write to.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    # The packaged app ships the RNNoise library next to the code; point the denoiser at it.
    if getattr(sys, "frozen", False) and not os.environ.get("RNNOISE_LIB"):
        found = glob.glob(_bundled("rnnoise", "*rnnoise*"))
        if found:
            os.environ["RNNOISE_LIB"] = found[0]
    # Make sure a stray proxy setting can never break talking to ourselves.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")


def _free_port(preferred=8000):
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError("no free port")


LOG = os.path.join(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp", "audio_repair.log")
_error = []


def _fail(where):
    import traceback
    msg = traceback.format_exc()
    _error.append("%s failed:" % where + chr(10) + msg)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(_error[-1] + chr(10))
    except OSError:
        pass


def main():
    _setup_environment()
    port = _free_port()
    url = "http://127.0.0.1:%d" % port
    server = None
    try:
        import uvicorn
        from app import app
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning"))

        def serve():
            try:
                server.run()
            except BaseException:
                _fail("server")
        threading.Thread(target=serve, daemon=True).start()
        for _ in range(150):                  # wait until it answers
            if server.started or _error:
                break
            time.sleep(0.1)
    except BaseException:
        _fail("startup")
    if server is not None and server.started and not _error:
        webbrowser.open(url)

    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:                         # no display: just keep serving
        print("Audio Repair is running at", url, "(Ctrl+C to quit)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if server is not None:
                server.should_exit = True
        return

    root.title("Audio Repair")
    root.geometry("340x150" if not _error else "520x260")
    root.resizable(False, False)
    if _error:
        tk.Label(root, text="Audio Repair could not start", font=("Segoe UI", 13, "bold"), fg="#b00020").pack(pady=(14, 4))
        box = tk.Text(root, height=9, width=64, wrap="word")
        box.insert("1.0", _error[-1][-900:] + chr(10) + "(log: %s)" % LOG)
        box.pack(padx=8)
        tk.Button(root, text="Close", width=10, command=root.destroy).pack(pady=8)
        root.mainloop()
        return
    tk.Label(root, text="Audio Repair is running", font=("Segoe UI", 13, "bold")).pack(pady=(18, 2))
    tk.Label(root, text=url, fg="#555").pack()
    row = tk.Frame(root)
    row.pack(pady=18)
    tk.Button(row, text="Open in browser", width=15, command=lambda: webbrowser.open(url)).pack(side="left", padx=6)

    def quit_app():
        if server is not None:
            server.should_exit = True
        root.destroy()

    tk.Button(row, text="Quit", width=10, command=quit_app).pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()
    if server is not None:
        server.should_exit = True


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
