"""Look at the app's page in an ordinary browser, with a pretend engine behind it (no app needed).

    python bench/ui_harness.py before.wav after.wav [port]

Makes a temporary folder holding static/index.html with a small mock of the app's engine (window.pywebview.api) and the
two audio files, and serves it on http://127.0.0.1:<port>/index.html (default 8765). Open it in a browser (or the Claude
desktop app's built-in browser): click Choose file, then Repair audio, and the result view appears with the files you
gave as Before and After. The mock records the calls the page makes in window.__calls (start / save), which is how the
scrubbing, spectrum, tone and save-with-tone behaviour was checked. It does not run any audio processing.
"""
import http.server
import shutil
import socketserver
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MOCK = r"""
window.__AR_NATIVE = true;
window.__calls = [];
const b64 = async url => { const buf = new Uint8Array(await (await fetch(url)).arrayBuffer()); let s = ''; for (let i = 0; i < buf.length; i += 8192) s += String.fromCharCode.apply(null, buf.subarray(i, i + 8192)); return btoa(s); };
const mockApi = {
  capabilities: async () => ({ mp3: true, studio: true }),
  updates_status: async () => ({ app_version: '1.1.0', running: { mark: 1, name: 'Mk I' }, bundled: { mark: 1, name: 'Mk I', title: '' }, active: 0, installed: [], notice: '' }),
  pick_file: async () => ({ path: 'mock/voice memo.wav', name: 'voice memo.wav', size: 1 }),
  use_path: async p => ({ path: p, name: 'voice memo.wav', size: 1 }),
  start: async (path, opts) => { window.__calls.push(['start', opts]); return { id: opts.base_job ? 'tone1' : 'job1' }; },
  status: async id => ({ state: 'done', progress: 1, message: 'Done', report: ['Applied: (mock) repair'] }),
  read_original: async () => ({ mime: 'audio/wav', data: await b64('before.wav') }),
  read_result: async () => ({ mime: 'audio/wav', data: await b64('after.wav') }),
  save: async id => { window.__calls.push(['save', id]); return { saved: 'mock/saved.wav' }; },
  reveal: async () => { }, restart: async () => { },
  pick_update: async () => null, inspect_update: async () => null, install_update: async () => ({}), switch_version: async () => ({}),
};
window.pywebview = { api: mockApi };
"""


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765
    tmp = Path(tempfile.mkdtemp(prefix="audio_repair_ui_"))
    shutil.copyfile(before, tmp / "before.wav")
    shutil.copyfile(after, tmp / "after.wav")
    (tmp / "mock.js").write_text(MOCK, encoding="utf-8")
    page = (ROOT / "static" / "index.html").read_text(encoding="utf-8").replace("<head>", '<head><script src="mock.js"></script>', 1)
    (tmp / "index.html").write_text(page, encoding="utf-8")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp), **k)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Quiet) as srv:
        print("serving http://127.0.0.1:%d/index.html  (Ctrl+C to stop)" % port)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
