import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import run
from pipeline.io import AudioError, mp3_supported

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))     # packaged app: bundled files live here
# Recordings are private: keep them in the system temp folder, not in this (possibly cloud-synced) project folder.
WORK = Path(tempfile.gettempdir()) / "audio_repair_work"
WORK.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 200
MAX_JOBS = 20            # finished jobs kept in memory / on disk
JOB_TTL_S = 3600         # ... for at most an hour
_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".aif", ".aiff", ".wma", ".opus", ".webm", ".mp4"}

app = FastAPI(title="Audio Repair")
jobs = {}
_lock = threading.Lock()


def _forget(job_id):
    j = jobs.pop(job_id, None)
    for key in ("in", "out"):
        if j and j.get(key):
            Path(j[key]).unlink(missing_ok=True)


def _sweep():
    """Drop old / surplus finished jobs and any orphaned files."""
    now = time.time()
    with _lock:
        done = sorted((k for k, j in jobs.items() if j["state"] != "running"), key=lambda k: jobs[k]["t"])
        for k in done:
            if now - jobs[k]["t"] > JOB_TTL_S or len(done) - done.index(k) > MAX_JOBS:
                _forget(k)
    live = {Path(j[x]).name for j in jobs.values() for x in ("in", "out") if j.get(x)}
    for p in WORK.iterdir():
        try:
            if p.name not in live and now - p.stat().st_mtime > JOB_TTL_S:
                p.unlink()
        except OSError:
            pass


def _worker(job_id, in_path, out_path, mode, opts):
    def progress(p, msg):
        jobs[job_id].update(progress=p, message=msg)
    try:
        run(str(in_path), str(out_path), mode, opts, progress)
        jobs[job_id].update(state="done", progress=1.0, message="Done", t=time.time())
    except AudioError as e:
        jobs[job_id].update(state="error", message=str(e), t=time.time())
    except MemoryError:
        jobs[job_id].update(state="error", t=time.time(),
                            message="This recording is too long to process in memory. Try a shorter file.")
    except Exception:
        jobs[job_id].update(state="error", t=time.time(),
                            message="Something went wrong while processing this file.")
        import traceback
        traceback.print_exc()
    finally:
        Path(in_path).unlink(missing_ok=True)       # the original is no longer needed


@app.get("/capabilities")
def capabilities():
    return {"mp3": mp3_supported(), "max_upload_mb": MAX_UPLOAD_MB}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    mode: str = Form("voice"),
    fmt: str = Form("wav"),
    declick: bool = Form(True),
    tame: bool = Form(True),
    declip: bool = Form(True),
    denoise: bool = Form(True),
    # omitted / empty = chosen automatically from the analysis of the recording
    denoise_strength: Optional[float] = Form(None, ge=0.0, le=1.0),
    eq: bool = Form(True),
    master: bool = Form(True),
):
    if mode not in ("voice", "music"):
        raise HTTPException(400, "Mode must be voice or music.")
    if fmt not in ("wav", "mp3"):
        raise HTTPException(400, "Output format must be WAV or MP3.")
    if fmt == "mp3" and not mp3_supported():
        raise HTTPException(400, "MP3 export is not available on this computer. Choose WAV.")
    _sweep()
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"That file is larger than {MAX_UPLOAD_MB} MB.")
    stem = re.sub(r"[^\w.\- ]+", "_", Path(file.filename or "audio").stem)[:80] or "audio"
    ext = Path(file.filename or "").suffix.lower()
    ext = ext if ext in _ALLOWED_EXT else ".wav"       # extension is only a hint; content is sniffed
    job_id = uuid.uuid4().hex
    in_path = WORK / f"{job_id}_in{ext}"
    out_path = WORK / f"{job_id}_out.{fmt}"
    in_path.write_bytes(data)
    report = []   # the pipeline appends "Detected: ..." / "Applied: ..." / "Skipped: ..." lines as it works
    jobs[job_id] = {"state": "running", "progress": 0.0, "message": "Queued", "t": time.time(),
                    "in": str(in_path), "out": str(out_path), "report": report,
                    "name": f"{stem}_fixed.{fmt}"}
    opts = dict(declick=declick, tame=tame, declip=declip, denoise=denoise, eq=eq, master=master, report=report)
    if denoise_strength is not None:
        opts["denoise_strength"] = denoise_strength
    threading.Thread(target=_worker, args=(job_id, in_path, out_path, mode, opts),
                     daemon=True).start()
    return {"id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "This job is no longer available (the app may have been restarted).")
    out = {k: j[k] for k in ("state", "progress", "message")}
    out["report"] = list(j.get("report", []))
    return out


@app.get("/download/{job_id}")
def download(job_id: str):
    j = jobs.get(job_id)
    if not j or j["state"] != "done" or not Path(j["out"]).exists():
        raise HTTPException(404, "Not ready")
    return FileResponse(j["out"], filename=j["name"])


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
