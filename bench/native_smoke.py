"""Open the real desktop window, drive the page by script, and check the whole round trip works with no server."""
import json, os, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import webview
import desktop

sample = str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "bench/out/david_news_combined_in.wav"))
api = desktop.Api()
api.window = webview.create_window("smoke", html=(ROOT / "static" / "index.html").read_text(encoding="utf-8"), js_api=api, width=560, height=900)
result = {}

def js(code):
    return api.window.evaluate_js(code)

def drive():
    try:
        time.sleep(4)
        result["native_detected"] = js("native")
        result["offline_banner_hidden"] = js("document.getElementById('offline').hidden")
        result["mp3_disabled"] = js("document.querySelector('input[name=fmt][value=mp3]').disabled")
        info = api.use_path(sample)
        js("window.__dropped(%s)" % json.dumps(info))
        result["file_shown"] = js("document.getElementById('fname').textContent")
        result["go_enabled"] = js("!document.getElementById('go').disabled")
        js("document.getElementById('go').click()")
        t0 = time.time()
        for _ in range(300):
            time.sleep(1)
            done = js("!document.getElementById('result').hidden")
            err = js("document.getElementById('err').textContent")
            if done or err:
                break
        result["seconds"] = round(time.time() - t0)
        result["error"] = js("document.getElementById('err').textContent")
        time.sleep(2)
        result["report"] = js("[...document.querySelectorAll('#report li')].map(l=>l.textContent).filter(t=>t.startsWith('Applied')).slice(0,3)")
        result["players"] = js("[document.getElementById('orig').src.startsWith('blob:'), document.getElementById('fixed').src.startsWith('blob:'), document.getElementById('fixed').hidden]")
        result["fixed_duration_s"] = js("document.getElementById('fixed').duration")
        result["save_button_text"] = js("document.getElementById('dl').textContent")
        import engine
        jid = js("jobId")
        dest = str(ROOT / "bench" / "demo" / "native_saved.wav")
        engine.save_copy(jid, dest)
        result["saved_bytes"] = os.path.getsize(dest)
    except Exception as e:
        result["exception"] = repr(e)
    finally:
        print(json.dumps(result, indent=1))
        api.window.destroy()

webview.start(drive)
