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
    threshold: float = Form(0.45),
    attendees: Optional[str] = Form(None),
):
    """
    Transcribe an audio file with speaker diarization.

    Returns {"speaker_report": {...}, "segments": [...]}. The report carries
    one entry per detected speaker cluster — how much it spoke, what it
    matched, the margin over the runner-up, and the full similarity breakdown
    — plus a threshold sweep showing what every candidate threshold would have
    produced for this recording. Similarity scores do not depend on the
    threshold, so the sweep lets a threshold be tuned from past runs without
    re-processing any audio.

    Optional form fields:
      - threshold (float, default 0.45) — speaker identification confidence cutoff.
                                          Measured genuine matches bottom out
                                          near 0.51 and strangers top out near
                                          0.36, so 0.45 sits in the gap. Every
                                          response carries a threshold_sweep
                                          for retuning this from real runs.
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
        words = _transcriber.transcribe(tmp_path)
        segments, report = _diarizer.diarize(tmp_path, words, threshold=threshold,
                                             attendees=attendee_list)
        # speaker_report first so it reads at the top of a saved JSON file
        return JSONResponse({"speaker_report": report, "segments": segments})
    except Exception as e:
        log.exception("Transcription/diarization failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    threshold: float = Form(0.45),
    attendees: Optional[str] = Form(None),
):
    """
    Diagnostic: run diarization + enrolled-speaker identification only — no
    transcription. Returns the same speaker_report that /transcribe embeds,
    for verifying an enrollment or tuning threshold/attendees without paying
    for transcription.

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
        report = _diarizer.identify_speakers(tmp_path, threshold=threshold,
                                             attendees=attendee_list)
        return JSONResponse(report)
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
