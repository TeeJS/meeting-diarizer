"""
Meeting Diarizer — FastAPI service.
Combines faster-whisper transcription with pyannote speaker diarization.
"""

import os
from pathlib import Path

# Must run BEFORE importing transcriber/diarizer below, which transitively
# import huggingface_hub and lock in its cache location. setdefault lets an
# explicit container-level HF_HOME still win.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "models"))

import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

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

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

_transcriber: Transcriber    = None
_diarizer:    Diarizer       = None
_store:       EnrollmentStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber, _diarizer, _store

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
async def transcribe(
    audio: UploadFile = File(...),
    threshold: float = Form(0.35),
    attendees: Optional[str] = Form(None),
):
    """
    Transcribe an audio file with speaker diarization.
    Returns a list of speaker-labeled segments.

    Optional form fields:
      - threshold (float, default 0.35) — speaker identification confidence cutoff.
      - attendees (str)                 — comma-separated list of enrolled speaker
                                          names known to be in the meeting. Enrolled
                                          speakers NOT in this list have 0.15
                                          subtracted from their similarity score,
                                          biasing the diarizer toward in-meeting
                                          candidates. Names must match enrolled
                                          names exactly (no fuzzy matching).
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    attendee_list = None
    if attendees:
        attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]

    try:
        log.info("Transcribe request — threshold=%.2f, attendees=%s",
                 threshold, attendee_list if attendee_list else "(none)")
        words    = _transcriber.transcribe(tmp_path)
        segments = _diarizer.diarize(tmp_path, words, threshold=threshold,
                                     attendees=attendee_list)
        return JSONResponse({"segments": segments})
    except Exception as e:
        log.exception("Transcription/diarization failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    threshold: float = Form(0.35),
    attendees: Optional[str] = Form(None),
):
    """
    Diagnostic: run diarization + enrolled-speaker identification only — no
    transcription. Returns each detected speaker cluster with its match (if
    any) and the full similarity score breakdown against every enrolled
    speaker, for verifying an enrollment or tuning threshold/attendees.

    Optional form fields: same as /transcribe — threshold, attendees.
    """
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    attendee_list = None
    if attendees:
        attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]

    try:
        log.info("Identify request — threshold=%.2f, attendees=%s",
                 threshold, attendee_list if attendee_list else "(none)")
        clusters = _diarizer.identify_speakers(tmp_path, threshold=threshold,
                                               attendees=attendee_list)
        return JSONResponse({"clusters": clusters})
    except Exception as e:
        log.exception("Identification failed")
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
