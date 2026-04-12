"""
FastAPI server exposing the elephant-rumble cleaning pipeline plus the
Interspecies Identification & Labeling Factory.

Endpoints:
    GET   /api/health          - liveness probe
    POST  /api/clean           - wav upload -> cleaned wav/zip + DB registration
    GET   /api/registry        - full elephant + clip database dump
    PUT   /api/rename_elephant - update an elephant's nickname
    GET   /api/clip/{clip_id}  - stream a specific murmur clip wav

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import shutil
import sys

# Load .env from project root
from pathlib import Path as _P
_env_path = _P(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from id_engine import (  # noqa: E402
    get_clip_audio_path,
    get_registry,
    register_cleaning_result,
    rename_elephant,
)
from inference import TARGET_SR, process_audio  # noqa: E402
from llm_engine import analyze_clip_with_llm  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


app = FastAPI(title="ElephantVoices Audio Cleaner", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)


class RenameRequest(BaseModel):
    elephant_id: str
    nickname: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/clean")
async def clean_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a noisy .wav, run the cleaning pipeline, splice murmurs,
    register everything in the database, and return a JSON payload with
    caller profiles + clip metadata + download URLs.
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload must be a .wav file.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="elephant_"))
    input_path = tmp_dir / f"input_{uuid.uuid4().hex}.wav"
    output_dir = tmp_dir / "outputs"

    try:
        with input_path.open("wb") as dst:
            shutil.copyfileobj(file.file, dst)
        output_paths = process_audio(
            str(input_path), str(output_dir),
            original_filename=os.path.basename(file.filename),
        )
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        await file.close()

    if not output_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="No cleaned output was produced.")

    # Read back the cleaned waveforms for splicing and DB registration
    caller_waveforms: List[np.ndarray] = []
    for p in output_paths:
        wav_data, sr = sf.read(p, dtype="float32")
        caller_waveforms.append(wav_data)

    parent_name = os.path.basename(file.filename)
    clip_records = register_cleaning_result(
        parent_wav_name=parent_name,
        caller_waveforms=caller_waveforms,
        sr=TARGET_SR,
    )

    # Build per-caller profile for the frontend
    callers = {}
    for rec in clip_records:
        eid = rec["elephant_id"]
        if eid not in callers:
            callers[eid] = {"elephant_id": eid, "clips": []}
        callers[eid]["clips"].append(rec)

    # Also build the zip/wav download if the frontend still wants a direct file
    download_url: str = ""
    if len(output_paths) == 1:
        download_url = f"/api/download/{os.path.basename(output_paths[0])}"
    else:
        zip_name = f"cleaned_{uuid.uuid4().hex}.zip"
        zip_path = tmp_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src in output_paths:
                zf.write(src, arcname=os.path.basename(src))
        download_url = f"/api/download/{zip_name}"

    # Copy outputs to a persistent location so download URLs work after
    # the temp dir is cleaned up.  The murmur splices already live in
    # backend/clips/ via id_engine.
    persistent_dir = Path(__file__).resolve().parent / "outputs_persist"
    persistent_dir.mkdir(exist_ok=True)
    for src in output_paths:
        dst = persistent_dir / os.path.basename(src)
        shutil.copy2(src, dst)
    if len(output_paths) > 1:
        zip_src = tmp_dir / zip_name
        shutil.copy2(zip_src, persistent_dir / zip_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    registry = get_registry()
    return JSONResponse({
        "parent_wav": parent_name,
        "detected_callers": len(callers),
        "callers": list(callers.values()),
        "clip_count": len(clip_records),
        "download_url": download_url,
        "elephants": registry["elephants"],
    })


@app.get("/api/registry")
def registry_endpoint() -> JSONResponse:
    """Fetch all labeled elephants and their clips."""
    return JSONResponse(get_registry())


@app.put("/api/rename_elephant")
def rename_endpoint(body: RenameRequest) -> JSONResponse:
    """Update an elephant's user-facing nickname."""
    if not body.elephant_id or not body.nickname:
        raise HTTPException(status_code=400, detail="Both elephant_id and nickname are required.")
    ok = rename_elephant(body.elephant_id, body.nickname)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Elephant '{body.elephant_id}' not found.")
    return JSONResponse({"status": "ok", "elephant_id": body.elephant_id, "nickname": body.nickname})


@app.get("/api/clip/{clip_id}")
def clip_audio_endpoint(clip_id: int) -> FileResponse:
    """Stream a specific murmur clip wav file."""
    path = get_clip_audio_path(clip_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found.")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/analyze_clip/{clip_id}")
def analyze_clip_endpoint(clip_id: int) -> JSONResponse:
    """Run acoustic feature extraction + Claude LLM analysis on a murmur clip."""
    clip_path = get_clip_audio_path(clip_id)
    if clip_path is None:
        raise HTTPException(status_code=404, detail=f"Clip {clip_id} not found.")
    result = analyze_clip_with_llm(str(clip_path))
    return JSONResponse({"clip_id": clip_id, **result})


@app.get("/api/download/{filename}")
def download_endpoint(filename: str) -> FileResponse:
    """Serve a cleaned output file or zip from the persistent store."""
    persist = Path(__file__).resolve().parent / "outputs_persist"
    target = (persist / filename).resolve()
    if not str(target).startswith(str(persist.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    media = "application/zip" if filename.endswith(".zip") else "audio/wav"
    return FileResponse(target, media_type=media, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
