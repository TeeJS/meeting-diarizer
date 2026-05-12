"""
Pyannote speaker diarization + enrolled speaker identification.

Pipeline shape (post-Wyoming pivot)
-----------------------------------
We no longer transcribe the whole audio first and then align words to
speaker turns. Wyoming returns text only — no word-level timestamps —
so the old "transcribe → align words" approach has nothing to align on.

Instead this module DRIVES the transcription, in this order:

    1. Load audio once (16 kHz mono float32).
    2. Run pyannote → speaker turns with (start, end, pyannote_label).
    3. Identify each pyannote label against enrolled embeddings.
    4. For each turn, crop the waveform, hand the int16 PCM to the
       Wyoming client, and use the returned text as that turn's text.
    5. Return [{speaker, start, end, text}, ...] — same shape as before
       so existing clients keep working.

Lazy load + release
-------------------
The pyannote pipeline and embedding model together hold ~1 GB of VRAM.
We don't want them resident 24/7 for a service that runs 4-6x/week.
Construction is deferred until the first call (`_ensure_loaded`), and
`release()` drops both models and asks CUDA to reclaim the memory.
"""

import gc
import logging
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

from .enrollment import EnrollmentStore
from .wyoming_client import WyomingClient

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75
EMBEDDING_MODEL      = "pyannote/wespeaker-voxceleb-resnet34-LM"
_LABELS              = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Pyannote models are trained at 16 kHz; loading at this rate avoids a
# separate resample step.
PYANNOTE_SR = 16000

# Turns shorter than this are skipped entirely — Whisper hallucinates on
# sub-200ms audio and the Wyoming round-trip isn't worth it.
MIN_TURN_SECONDS = 0.2

# How much of each speaker's audio to use when computing their average
# embedding for identification. The original code capped at 10 segments;
# that stays.
MAX_EMBED_SEGMENTS = 10


def _load_audio(path: str) -> dict:
    """Load an audio file as a pyannote-compatible waveform dict.

    Returns: {"waveform": torch.Tensor(1, N) float32, "sample_rate": 16000}.
    Mixes multi-channel to mono and resamples to 16 kHz if needed.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)          # (channels, samples)
    if waveform.shape[0] > 1:                    # mix down to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != PYANNOTE_SR:
        orig_len = waveform.shape[1]
        new_len  = int(orig_len * PYANNOTE_SR / sr)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len,
            mode="linear", align_corners=False,
        ).squeeze(0)
    return {"waveform": waveform, "sample_rate": PYANNOTE_SR}


def _crop_audio(audio: dict, start: float, end: float) -> dict:
    """Crop a waveform dict to the given time range (seconds)."""
    sr = audio["sample_rate"]
    s  = int(start * sr)
    e  = int(end * sr)
    return {"waveform": audio["waveform"][:, s:e], "sample_rate": sr}


def _waveform_to_int16_pcm(audio: dict, start: float, end: float) -> np.ndarray:
    """Crop the audio to [start, end] and convert to a 1-D int16 PCM array.

    The Wyoming client wants signed-16-bit-little-endian PCM as a 1-D
    numpy array. Our internal audio is float32 [-1.0, 1.0] in a 2-D
    (channels, samples) tensor, so we squeeze + scale + clamp + cast.
    """
    sr = audio["sample_rate"]
    s  = int(start * sr)
    e  = int(end * sr)
    chunk = audio["waveform"][:, s:e]
    if chunk.shape[0] > 1:                       # mono safety
        chunk = chunk.mean(dim=0, keepdim=True)
    samples = chunk.squeeze(0).cpu()             # 1-D float32 on CPU
    # 32767 maps full-scale +1.0 to int16 max; clamp guards against
    # rounding overshoot at the edges.
    pcm = (samples * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
    return pcm.numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _default_label(pyannote_label: str, index_map: dict) -> str:
    idx = index_map.get(pyannote_label, 0)
    return f"Speaker {_LABELS[idx]}" if idx < len(_LABELS) else pyannote_label


class Diarizer:
    """Speaker diarization + identification, driving Wyoming transcription.

    Models are loaded lazily on first use and can be released via
    `release()` to free VRAM when the service is idle.
    """

    def __init__(self, hf_token: str, enrollment_store: EnrollmentStore):
        self._hf_token  = hf_token
        self._store     = enrollment_store
        self._pipeline: Optional[Pipeline]  = None
        self._inference: Optional[Inference] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Construct pyannote models on first use. Idempotent.

        Important: call this from the asyncio main thread, NOT from a
        thread-pool executor. Pyannote 4.x's loading path (HuggingFace
        download + model init) segfaults intermittently when called
        from a worker thread under our environment. Inference itself
        (Pipeline.__call__, Inference.__call__) is fine on a worker.
        """
        if self._pipeline is not None and self._inference is not None:
            return

        log.info("Loading pyannote speaker-diarization-3.1 ...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self._hf_token,
        )
        self._pipeline.to(torch.device("cuda"))

        log.info("Loading speaker embedding model: %s", EMBEDDING_MODEL)
        emb_model = Model.from_pretrained(EMBEDDING_MODEL, token=self._hf_token)
        emb_model = emb_model.to(torch.device("cuda"))
        self._inference = Inference(emb_model, window="whole")

        log.info("Diarizer ready (pyannote models on GPU).")

    def release(self) -> None:
        """Drop model references and ask CUDA to reclaim their memory.

        Safe to call when nothing is loaded. After release(), the next
        call to enroll/diarize will lazily reload the models.
        """
        if self._pipeline is None and self._inference is None:
            return
        log.info("Releasing pyannote models from GPU.")
        self._pipeline  = None
        self._inference = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_speaker(self, name: str, audio_path: str) -> None:
        """Extract and store a speaker embedding from a reference audio file.

        Caller is responsible for having invoked `ensure_loaded()` first
        (on the asyncio main thread, not a worker).
        """
        audio     = _load_audio(audio_path)
        embedding = self._inference(audio)
        self._store.save(name, np.array(embedding))
        log.info("Enrolled speaker: %s", name)

    # ------------------------------------------------------------------
    # Diarize + transcribe
    # ------------------------------------------------------------------

    def diarize(
        self,
        audio_path:     str,
        wyoming_client: WyomingClient,
        threshold:      float = SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """Diarize an audio file and transcribe each speaker turn.

        Caller is responsible for having invoked `ensure_loaded()` first
        (on the asyncio main thread, not a worker).

        Returns segments shaped like:
            [{"speaker": str, "start": float, "end": float, "text": str}, ...]

        Empty / sub-MIN_TURN_SECONDS turns are dropped. Turns where
        Wyoming returns empty text are also dropped (silence / non-speech).
        """
        audio  = _load_audio(audio_path)
        result = self._pipeline(audio)

        # pyannote 3.3+ wraps the annotation in a DiarizeOutput;
        # older versions return Annotation directly.
        annotation = (
            result.speaker_diarization
            if hasattr(result, "speaker_diarization") else result
        )
        timeline = [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
        if not timeline:
            log.info("Pyannote found no speaker turns in audio.")
            return []

        unique_spks = sorted(set(t[2] for t in timeline))
        index_map   = {spk: i for i, spk in enumerate(unique_spks)}

        # Default labels (Speaker A, Speaker B, ...) — overridden below
        # if a pyannote label matches an enrolled speaker.
        label_map = {spk: _default_label(spk, index_map) for spk in unique_spks}

        if self._store.list_speakers():
            self._apply_enrolled_labels(audio, timeline, unique_spks, label_map, threshold)

        # Per-turn transcription via Wyoming.
        segments: List[Dict] = []
        log.info("Transcribing %d turn(s) via Wyoming ...", len(timeline))
        for start, end, pyannote_label in timeline:
            duration = end - start
            if duration < MIN_TURN_SECONDS:
                continue
            try:
                pcm  = _waveform_to_int16_pcm(audio, start, end)
                text = wyoming_client.transcribe(pcm)
            except Exception:
                # One bad turn shouldn't kill the whole meeting. Log and
                # move on — the gap will show up as missing text in the
                # output, which is more useful than a 500.
                log.exception(
                    "Wyoming transcription failed for turn [%.2f-%.2f] (%s)",
                    start, end, pyannote_label,
                )
                continue
            if not text:
                continue
            segments.append({
                "speaker": label_map.get(pyannote_label, pyannote_label),
                "start":   round(start, 2),
                "end":     round(end, 2),
                "text":    text,
            })

        log.info("Produced %d non-empty transcribed segment(s).", len(segments))
        return segments

    # ------------------------------------------------------------------
    # Speaker identification (unchanged from pre-pivot behavior)
    # ------------------------------------------------------------------

    def _apply_enrolled_labels(
        self,
        audio:       dict,
        timeline:    List[tuple],
        unique_spks: List[str],
        label_map:   Dict[str, str],
        threshold:   float,
    ) -> None:
        """Match each pyannote label to an enrolled speaker, when possible.

        Mutates `label_map` in place: replaces "Speaker A" with the
        enrolled name where the embedding cosine similarity clears
        `threshold`. Anything that doesn't match keeps its default label.
        """
        for pyannote_label in unique_spks:
            speaker_segs = [
                Segment(s, e) for s, e, spk in timeline
                if spk == pyannote_label
            ]
            embeddings = []
            for seg in speaker_segs[:MAX_EMBED_SEGMENTS]:
                try:
                    if seg.end - seg.start < 0.5:
                        # Too short to embed reliably; skip.
                        continue
                    cropped = _crop_audio(audio, seg.start, seg.end)
                    emb     = np.array(self._inference(cropped))
                    if np.any(np.isnan(emb)):
                        log.warning(
                            "Skipping %.2fs segment [%.2f-%.2f]: NaN embedding",
                            seg.end - seg.start, seg.start, seg.end,
                        )
                        continue
                    embeddings.append(emb)
                except Exception:
                    # Embedding a single segment failed — keep going,
                    # other segments may still produce a usable average.
                    continue

            if not embeddings:
                continue

            avg_emb = np.mean(embeddings, axis=0)
            name    = self._identify(avg_emb, threshold=threshold)
            if name:
                label_map[pyannote_label] = name
                log.info("Identified %s as: %s", pyannote_label, name)

    def _identify(
        self, embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD,
    ) -> Optional[str]:
        """Best-match enrolled speaker for an embedding, or None below threshold."""
        if np.any(np.isnan(embedding)):
            log.warning("Skipping segment: NaN embedding (segment too short or degenerate)")
            return None
        best_name, best_score = None, -1.0
        scores = {}
        for name, enrolled in self._store.all_embeddings().items():
            score = _cosine_similarity(embedding, enrolled)
            scores[name] = round(score, 4)
            if score > best_score:
                best_name, best_score = name, score
        log.info(
            "Speaker similarity scores (threshold=%.2f): %s", threshold,
            ", ".join(f"{n}={s}" for n, s in sorted(scores.items(), key=lambda x: -x[1])),
        )
        if best_score >= threshold:
            log.info("  -> Matched: %s (%.4f)", best_name, best_score)
            return best_name
        log.info("  -> No match (best was %s at %.4f)", best_name, best_score)
        return None
