"""In-process job manager used by the native desktop app (no server, no network port).

The browser/server version (app.py) keeps its own copy of this logic; both call the same ``pipeline.run``.
"""
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from pipeline import run
from pipeline import tone_after
from pipeline.io import AudioError, load_audio, mp3_supported, save_audio
from pipeline.master import master_with_info
from pipeline.studio import studio_available

WORK = Path(tempfile.gettempdir()) / "audio_repair_work"
WORK.mkdir(exist_ok=True)
MAX_JOBS = 20
JOB_TTL_S = 3600

_jobs = {}
_lock = threading.Lock()


def capabilities():
    return {"mp3": mp3_supported(), "studio": studio_available()}


def _forget(job_id):
    j = _jobs.pop(job_id, None)
    if j:
        for key in ("out", "base"):
            if j.get(key) and not _still_used(j[key]):
                Path(j[key]).unlink(missing_ok=True)


def _still_used(path):
    """The stored repair result is shared by the tone-adjusted copies made from it: keep it while any job uses it."""
    return any(path in (j.get("out"), j.get("base")) for j in _jobs.values())


def _sweep():
    now = time.time()
    with _lock:
        done = sorted((k for k, j in _jobs.items() if j["state"] != "running"), key=lambda k: _jobs[k]["t"])
        for i, k in enumerate(done):
            if now - _jobs[k]["t"] > JOB_TTL_S or len(done) - i > MAX_JOBS:
                _forget(k)
    live = {Path(j[k]).name for j in _jobs.values() for k in ("out", "base") if j.get(k)}
    for p in WORK.iterdir():
        try:
            if p.name not in live and now - p.stat().st_mtime > JOB_TTL_S:
                p.unlink()
        except OSError:
            pass


def _mark_name():
    try:
        import app_updates
        return "%s (app %s)" % (app_updates.status()["running"]["name"], app_updates.APP_VERSION)
    except Exception:
        return "unknown"


def _log_path():
    try:
        import app_updates
        return app_updates.data_dir() / "errors.log"
    except Exception:
        return WORK / "errors.log"


def _error_detail(job_id, exc):
    """Everything needed to understand a failed repair, as plain text: shown under "Show details" and kept in
    errors.log. Contains the file's name and size and the settings, never the recording itself."""
    import platform
    import sys
    import traceback
    j = _jobs.get(job_id, {})
    src = Path(j.get("src", ""))
    try:
        size = "%.1f MB" % (src.stat().st_size / 1048576)
    except OSError:
        size = "unknown size"
    opts = {k: v for k, v in (j.get("opts") or {}).items() if k != "report"}
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    lines = [
        "Audio Repair error report",
        "Time:       " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "Version:    " + _mark_name(),
        "System:     %s, Python %s" % (platform.platform(), sys.version.split()[0]),
        "File:       %s (%s)" % (src.name, size),
        "Mode:       %s, output %s" % (j.get("mode", "?"), j.get("fmt", "?")),
        "Settings:   " + ", ".join("%s=%s" % kv for kv in sorted(opts.items())),
        "Stopped at: %s (%d%% done)" % (j.get("message", "?"), round(100 * j.get("progress", 0))),
        "Already done: " + ("; ".join(j.get("report") or []) or "nothing"),
        "Error:      %s: %s" % (type(exc).__name__, exc),
        "",
        tb.strip()[-6000:],
    ]
    return "\n".join(lines)


def _log_error(text):
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        old = path.read_text(encoding="utf-8", errors="replace")[-200000:] if path.exists() else ""
        path.write_text(old + ("\n" if old else "") + "=" * 78 + "\n" + text + "\n", encoding="utf-8")
        return str(path)
    except OSError:
        return ""


def _worker(job_id, src, out, mode, opts):
    def progress(p, msg):
        _jobs[job_id].update(progress=p, message=msg)
    try:
        base = _jobs[job_id]["base"]
        run(str(src), base, mode, opts, progress)
        if base != str(out):                                   # MP3 asked for: encode from the full-quality result
            import soundfile as sf
            data, sr = sf.read(base, dtype="float32", always_2d=True)
            save_audio(str(out), data.T, sr)
        _jobs[job_id].update(state="done", progress=1.0, message="Done", t=time.time())
    except AudioError as e:
        _jobs[job_id].update(state="error", message=str(e), t=time.time())
    except MemoryError as e:
        detail = _error_detail(job_id, e)
        _jobs[job_id].update(state="error", t=time.time(), detail=detail, log=_log_error(detail),
                             message="This recording is too long to process in memory. Try a shorter file.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        detail = _error_detail(job_id, e)
        where = _jobs[job_id].get("message", "")
        _jobs[job_id].update(state="error", t=time.time(), detail=detail, log=_log_error(detail),
                             message="Something went wrong while processing this file (%s%s)."
                                     % (type(e).__name__, ", at: " + where if where else ""))


def _tone_worker(job_id, base, out, mode, tone):
    """Apply the tone settings to a finished repair and re-match its loudness. The repair itself is not run again."""
    def progress(p, msg):
        _jobs[job_id].update(progress=p, message=msg)
    try:
        import soundfile as sf
        progress(0.1, "Loading the repaired audio")
        data, sr = sf.read(base, dtype="float32", always_2d=True)
        x = data.T
        progress(0.3, "Applying your tone changes")
        x = tone_after.apply_tone(x, sr, tone)
        progress(0.55, "Matching the loudness")
        x, info = master_with_info(x, sr, mode)
        progress(0.9, "Saving")
        save_audio(str(out), np.clip(x, -1.0, 1.0).astype(np.float32), sr)
        t = tone_after.clean_tone(tone)
        said = ", ".join("%s %+.1f dB" % (k, v) for k, v in t.items() if abs(v) >= 0.05)
        _jobs[job_id]["report"].append("Applied: tone changed after the repair (%s)" % said)
        _jobs[job_id]["report"].append("Applied: loudness matched to %.1f LUFS with true peak %.1f dBTP"
                                       % (info.get("lufs_out", float("nan")), info.get("tp", float("nan"))))
        _jobs[job_id].update(state="done", progress=1.0, message="Done", t=time.time())
    except AudioError as e:
        _jobs[job_id].update(state="error", message=str(e), t=time.time())
    except Exception as e:
        import traceback
        traceback.print_exc()
        detail = _error_detail(job_id, e)
        _jobs[job_id].update(state="error", t=time.time(), detail=detail, log=_log_error(detail),
                             message="The tone change could not be applied (%s)." % type(e).__name__)


def start(path, mode="voice", fmt="wav", declick=True, tame=True, declip=True, denoise=True, eq=True,
          master=True, denoise_strength=None, studio=False, base_job=None, tone=None):
    src = Path(path)
    if not src.is_file():
        raise AudioError("That file could not be found.")
    if mode not in ("voice", "music") or fmt not in ("wav", "mp3"):
        raise AudioError("Unsupported settings.")
    if fmt == "mp3" and not mp3_supported():
        raise AudioError("MP3 export is not available on this computer. Choose WAV.")
    _sweep()
    job_id = uuid.uuid4().hex
    out = WORK / ("%s_out.%s" % (job_id, fmt))
    if base_job is not None:                      # a tone change on an earlier, finished repair
        b = _jobs.get(base_job)
        if not b or b["state"] != "done" or not b.get("base") or not Path(b["base"]).exists():
            raise AudioError("The repaired audio is no longer available. Repair the recording again.")
        _jobs[job_id] = {"state": "running", "progress": 0.0, "message": "Queued", "t": time.time(), "out": str(out),
                         "base": b["base"], "report": [], "name": "%s_fixed.%s" % (src.stem[:80] or "audio", fmt),
                         "src": str(src), "mode": b.get("mode", mode), "fmt": fmt, "opts": {"tone": tone_after.clean_tone(tone)}}
        threading.Thread(target=_tone_worker, args=(job_id, b["base"], out, b.get("mode", mode), tone), daemon=True).start()
        return job_id
    base = out if fmt == "wav" else WORK / ("%s_base.wav" % job_id)
    report = []
    _jobs[job_id] = {"state": "running", "progress": 0.0, "message": "Queued", "t": time.time(), "out": str(out),
                     "base": str(base), "report": report, "name": "%s_fixed.%s" % (src.stem[:80] or "audio", fmt), "src": str(src),
                     "mode": mode, "fmt": fmt}
    opts = dict(declick=bool(declick), tame=bool(tame), declip=bool(declip), denoise=bool(denoise), eq=bool(eq),
                master=bool(master), studio=bool(studio), report=report)
    if denoise_strength is not None:
        opts["denoise_strength"] = max(0.0, min(1.0, float(denoise_strength)))
    _jobs[job_id]["opts"] = opts
    threading.Thread(target=_worker, args=(job_id, src, out, mode, opts), daemon=True).start()
    return job_id


def status(job_id):
    j = _jobs.get(job_id)
    if not j:
        return {"state": "error", "progress": 0.0, "message": "This job is no longer available.", "report": []}
    out = {"state": j["state"], "progress": j["progress"], "message": j["message"], "report": list(j["report"])}
    if j.get("detail"):
        out.update(detail=j["detail"], log=j.get("log", ""))
    return out


def output_path(job_id):
    j = _jobs.get(job_id)
    if not j or j["state"] != "done" or not Path(j["out"]).exists():
        return None
    return j["out"], j["name"]


def save_copy(job_id, dest):
    found = output_path(job_id)
    if not found:
        raise AudioError("The repaired file is no longer available. Repair the recording again.")
    shutil.copyfile(found[0], dest)
    return str(dest)
