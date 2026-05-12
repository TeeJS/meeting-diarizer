"""
Meeting Diarizer — FastAPI service.

Diarizes uploaded audio with pyannote and transcribes each speaker turn
via a remote wyoming-faster-whisper instance. The pyannote models are
lazy-loaded on first request and released after IDLE_RELEASE_SECONDS of
inactivity, so the service holds ~0 GB VRAM when idle.

Environment
-----------
DATA_DIR              Internal data path. Default: /data
HF_TOKEN              HuggingFace token for downloading pyannote models.

WYOMING_HOST          Default Wyoming host. Default: 192.168.1.25
WYOMING_PORT          Default Wyoming port. Default: 10300
WYOMING_LANGUAGE      Language code passed to Wyoming. Default: en

WHISPER_MODELS        Optional comma-separated map of named backends, used
                      for A/B testing different Whisper model sizes.
                      Two accepted forms:
                          medium:10300,large:10302   (host inherits WYOMING_HOST)
                          medium=192.168.1.25:10300,large=192.168.1.30:10302
WHISPER_DEFAULT       Name of the default entry in WHISPER_MODELS (used when
                      a request omits the `model` form field). If unset, the
                      single-backend default (WYOMING_HOST/WYOMING_PORT) wins.

IDLE_RELEASE_SECONDS  Seconds of inactivity after which the pyannote models
                      are released from GPU. Default: 300 (5 min).

API
---
POST /transcribe   multipart: audio (file), threshold (float), model (str optional)
POST /enroll       multipart: name (str), audio (file)
GET  /speakers
DELETE /speakers/{name}
GET  /health
"""

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .diarizer import Diarizer
from .enrollment import EnrollmentStore
from .wyoming_client import WyomingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time; restart the container to change)
# ---------------------------------------------------------------------------

DATA_DIR             = Path(os.environ.get("DATA_DIR", "/data"))
HF_TOKEN             = os.environ.get("HF_TOKEN", "")

WYOMING_HOST         = os.environ.get("WYOMING_HOST", "192.168.1.25")
WYOMING_PORT         = int(os.environ.get("WYOMING_PORT", "10300"))
WYOMING_LANGUAGE     = os.environ.get("WYOMING_LANGUAGE", "en")

IDLE_RELEASE_SECONDS = int(os.environ.get("IDLE_RELEASE_SECONDS", "300"))


def _parse_whisper_models(raw: str) -> Dict[str, Tuple[str, int]]:
    """Parse the WHISPER_MODELS env var into {name: (host, port)}.

    Accepted entry forms (comma-separated):
        name:port              -> host inherits WYOMING_HOST
        name=host:port         -> explicit host
    """
    out: Dict[str, Tuple[str, int]] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "=" in entry:
                name, hostport = entry.split("=", 1)
                host, _, port_str = hostport.partition(":")
                if not host or not port_str:
                    raise ValueError("expected host:port after '='")
            else:
                name, _, port_str = entry.partition(":")
                if not port_str:
                    raise ValueError("expected 'name:port'")
                host = WYOMING_HOST
            out[name.strip()] = (host.strip(), int(port_str))
        except ValueError as e:
            log.warning("Ignoring malformed WHISPER_MODELS entry %r: %s", entry, e)
    return out


WHISPER_MODELS  = _parse_whisper_models(os.environ.get("WHISPER_MODELS", ""))
WHISPER_DEFAULT = os.environ.get("WHISPER_DEFAULT", "").strip()


def _resolve_backend(model: Optional[str]) -> Tuple[str, int]:
    """Decide which (host, port) to use for this request.

    - model None/empty + WHISPER_DEFAULT set + present in map: use that.
    - model None/empty otherwise: use WYOMING_HOST/WYOMING_PORT.
    - model named: look up in WHISPER_MODELS, raise ValueError on miss.
    """
    if not model:
        if WHISPER_DEFAULT and WHISPER_DEFAULT in WHISPER_MODELS:
            return WHISPER_MODELS[WHISPER_DEFAULT]
        return (WYOMING_HOST, WYOMING_PORT)
    if model not in WHISPER_MODELS:
        known = sorted(WHISPER_MODELS) or "(none configured — set WHISPER_MODELS)"
        raise ValueError(f"unknown model {model!r}; configured: {known}")
    return WHISPER_MODELS[model]


# ---------------------------------------------------------------------------
# Lazy-loaded diarizer with idle release
# ---------------------------------------------------------------------------

_store:         Optional[EnrollmentStore] = None
_diarizer:      Optional[Diarizer]        = None
_diarizer_lock: Optional[asyncio.Lock]    = None  # created in lifespan
_last_used:     float                     = 0.0
_release_task:  Optional[asyncio.Task]    = None


def _ensure_diarizer_locked() -> Diarizer:
    """Construct the diarizer on first use. Caller MUST hold _diarizer_lock.

    Schedules the background idle-release task if not already running.
    """
    global _diarizer, _release_task, _last_used
    if _diarizer is None:
        _diarizer = Diarizer(hf_token=HF_TOKEN, enrollment_store=_store)
    _last_used = time.monotonic()
    if _release_task is None or _release_task.done():
        _release_task = asyncio.create_task(_release_when_idle())
    return _diarizer


async def _release_when_idle() -> None:
    """Background task: release the diarizer once it has been idle long enough.

    The task exits after releasing. A new task is spawned by the next
    request via `_ensure_diarizer_locked`.
    """
    global _diarizer
    while True:
        idle_for = time.monotonic() - _last_used
        wait     = IDLE_RELEASE_SECONDS - idle_for
        if wait > 0:
            # Sleep until the deadline (+1s margin so the deadline check
            # below is reliably past the threshold).
            await asyncio.sleep(wait + 1.0)
            continue

        # We're past the idle threshold — try to release. Re-check under
        # the lock in case a new request landed while we were waiting.
        assert _diarizer_lock is not None
        async with _diarizer_lock:
            if (
                _diarizer is not None
                and time.monotonic() - _last_used >= IDLE_RELEASE_SECONDS
            ):
                log.info(
                    "Idle for %ds — releasing pyannote models.",
                    IDLE_RELEASE_SECONDS,
                )
                _diarizer.release()
                _diarizer = None
                return  # task done; next request spawns a new one
        # Otherwise the timer was bumped; loop and re-compute the wait.


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _diarizer_lock

    # Point HuggingFace cache at our data volume so model downloads
    # persist across container restarts.
    os.environ["HF_HOME"] = str(DATA_DIR / "models")

    _store         = EnrollmentStore(DATA_DIR / "enrollments")
    _diarizer_lock = asyncio.Lock()

    log.info(
        "Meeting Diarizer ready on port 10301. "
        "Default Wyoming backend: %s:%d. Named backends: %s. "
        "Idle release after %ds.",
        WYOMING_HOST, WYOMING_PORT,
        WHISPER_MODELS or "(none)",
        IDLE_RELEASE_SECONDS,
    )
    yield
    log.info("Shutting down.")


app = FastAPI(title="Meeting Diarizer", version="2.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _save_upload_to_tempfile(audio: UploadFile) -> str:
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        return tmp.name


@app.post("/transcribe")
async def transcribe(
    audio:     UploadFile     = File(...),
    threshold: float          = Form(0.75),
    model:     Optional[str]  = Form(None),
):
    """Transcribe an audio file with speaker diarization.

    Form fields:
        audio      — the audio file (required).
        threshold  — speaker-identification cosine cutoff. Default 0.75.
        model      — name of a backend in WHISPER_MODELS, e.g. "medium"
                     or "large". Omit to use the default backend.

    Returns: {"segments": [{"speaker", "start", "end", "text"}, ...]}.
    """
    try:
        host, port = _resolve_backend(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tmp_path = await _save_upload_to_tempfile(audio)

    try:
        log.info(
            "Transcribe request — model=%s wyoming=%s:%d threshold=%.2f",
            model or "(default)", host, port, threshold,
        )
        wy = WyomingClient(host=host, port=port, language=WYOMING_LANGUAGE)

        assert _diarizer_lock is not None
        async with _diarizer_lock:
            d = _ensure_diarizer_locked()
            # Load pyannote models on the main thread. Doing this from a
            # worker thread segfaults under our pyannote 4.x setup; the
            # block here is unavoidable but short after the first call
            # (subsequent calls are no-ops while models stay resident).
            d.ensure_loaded()
            loop = asyncio.get_running_loop()
            # diarize() does GPU inference AND blocking Wyoming network
            # I/O. Offload to a worker thread so the event loop stays
            # responsive; inference (unlike loading) is thread-safe here.
            segments = await loop.run_in_executor(
                None, d.diarize, tmp_path, wy, threshold,
            )
            # Bump idle timestamp after the work completes too — long
            # transcribes shouldn't count their own runtime as "idle".
            global _last_used
            _last_used = time.monotonic()

        return JSONResponse({"segments": segments})
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Transcription/diarization failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/enroll")
async def enroll(name: str = Form(...), audio: UploadFile = File(...)):
    """Enroll a speaker by name with a reference audio sample.

    Provide several minutes of clean audio for best accuracy.
    """
    tmp_path = await _save_upload_to_tempfile(audio)

    try:
        assert _diarizer_lock is not None
        async with _diarizer_lock:
            d = _ensure_diarizer_locked()
            # See transcribe() — model load must be on the main thread.
            d.ensure_loaded()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, d.enroll_speaker, name, tmp_path)
            global _last_used
            _last_used = time.monotonic()
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
