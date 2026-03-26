"""
Meeting Diarizer — FastAPI service.
Combines faster-whisper transcription with pyannote speaker diarization.
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .transcriber import Transcriber
from .diarizer import Diarizer
from .enrollment import EnrollmentStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR      = Path(os.environ.get("DATA_DIR", "/data"))
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

_transcriber: Transcriber    = None
_diarizer:    Diarizer       = None
_store:       EnrollmentStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber, _diarizer, _store

    # Point HuggingFace cache at our data volume
    os.environ["HF_HOME"] = str(DATA_DIR / "models")

    _store       = EnrollmentStore(DATA_DIR / "enrollments")
    _transcriber = Transcriber(model_size=WHISPER_MODEL)
    _diarizer    = Diarizer(hf_token=HF_TOKEN, enrollment_store=_store)

    log.info("Meeting Diarizer ready on port 10301.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="Meeting Diarizer", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), threshold: float = Form(0.75)):
    """
    Transcribe an audio file with speaker diarization.
    Returns a list of speaker-labeled segments.
    Optional: threshold (float, default 0.75) — speaker identification confidence cutoff.
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        log.info("Transcribe request — threshold=%.2f", threshold)
        words    = _transcriber.transcribe(tmp_path)
        segments = _diarizer.diarize(tmp_path, words, threshold=threshold)
        return JSONResponse({"segments": segments})
    except Exception as e:
        log.exception("Transcription/diarization failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/enroll")
async def enroll(name: str = Form(...), audio: UploadFile = File(...)):
    """
    Enroll a speaker by name with a reference audio sample.
    Provide several minutes of clean audio for best accuracy.
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        _diarizer.enroll_speaker(name, tmp_path)
        return {"status": "enrolled", "name": name}
    except Exception as e:
        log.exception("Enrollment failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/speakers")
async def list_speakers():
    """List all enrolled speakers."""
    return {"speakers": _store.list_speakers()}


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    """Remove an enrolled speaker."""
    _store.delete_speaker(name)
    return {"status": "deleted", "name": name}
