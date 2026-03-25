"""
Pyannote speaker diarization + enrolled speaker identification.
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

from .enrollment import EnrollmentStore

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75
EMBEDDING_MODEL      = "pyannote/wespeaker-voxceleb-resnet34-LM"
_LABELS              = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _default_label(pyannote_label: str, index_map: dict) -> str:
    idx = index_map.get(pyannote_label, 0)
    return f"Speaker {_LABELS[idx]}" if idx < len(_LABELS) else pyannote_label


def _words_to_segments(words: List[Dict], label_map: Dict[str, str]) -> List[Dict]:
    """Group consecutive same-speaker words into text segments."""
    if not words:
        return []

    segments = []
    cur_spk   = words[0].get("speaker", "Unknown")
    cur_words = [words[0]["word"]]
    cur_start = words[0]["start"]
    cur_end   = words[0]["end"]

    for w in words[1:]:
        spk = w.get("speaker", "Unknown")
        if spk == cur_spk:
            cur_words.append(w["word"])
            cur_end = w["end"]
        else:
            segments.append({
                "speaker": label_map.get(cur_spk, cur_spk),
                "start":   round(cur_start, 2),
                "end":     round(cur_end, 2),
                "text":    "".join(cur_words).strip(),
            })
            cur_spk   = spk
            cur_words = [w["word"]]
            cur_start = w["start"]
            cur_end   = w["end"]

    segments.append({
        "speaker": label_map.get(cur_spk, cur_spk),
        "start":   round(cur_start, 2),
        "end":     round(cur_end, 2),
        "text":    "".join(cur_words).strip(),
    })
    return [s for s in segments if s["text"]]


class Diarizer:
    def __init__(self, hf_token: str, enrollment_store: EnrollmentStore):
        log.info("Loading pyannote speaker-diarization-3.1 ...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        self._pipeline.to(torch.device("cuda"))

        log.info("Loading speaker embedding model: %s", EMBEDDING_MODEL)
        emb_model = Model.from_pretrained(EMBEDDING_MODEL, token=hf_token)
        self._inference = Inference(emb_model, window="whole")

        self._store = enrollment_store
        log.info("Diarizer ready.")

    def enroll_speaker(self, name: str, audio_path: str):
        """Extract and store a speaker embedding from a reference audio file."""
        embedding = self._inference(audio_path)
        self._store.save(name, np.array(embedding))
        log.info("Enrolled speaker: %s", name)

    def _identify(self, embedding: np.ndarray) -> Optional[str]:
        """Compare embedding to enrolled speakers. Returns name or None."""
        best_name, best_score = None, -1.0
        for name, enrolled in self._store.all_embeddings().items():
            score = _cosine_similarity(embedding, enrolled)
            if score > best_score:
                best_name, best_score = name, score
        return best_name if best_score >= SIMILARITY_THRESHOLD else None

    def diarize(self, audio_path: str, words: List[Dict]) -> List[Dict]:
        """
        Run diarization, align with word timestamps, identify speakers,
        and return grouped segments.
        """
        annotation  = self._pipeline(audio_path)
        timeline    = [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
        unique_spks = sorted(set(t[2] for t in timeline))
        index_map   = {spk: i for i, spk in enumerate(unique_spks)}

        # Assign each word to a speaker by midpoint
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            w["speaker"] = "UNKNOWN"
            for start, end, spk in timeline:
                if start <= mid <= end:
                    w["speaker"] = spk
                    break

        # Build label map: start with default labels, then try enrolled speakers
        label_map = {spk: _default_label(spk, index_map) for spk in unique_spks}

        if self._store.list_speakers():
            for pyannote_label in unique_spks:
                speaker_segs = [
                    Segment(s, e) for s, e, spk in timeline
                    if spk == pyannote_label
                ]
                embeddings = []
                for seg in speaker_segs[:10]:  # cap for speed
                    try:
                        emb = self._inference({"audio": audio_path}, excerpt=seg)
                        embeddings.append(np.array(emb))
                    except Exception:
                        continue

                if embeddings:
                    avg_emb = np.mean(embeddings, axis=0)
                    name    = self._identify(avg_emb)
                    if name:
                        label_map[pyannote_label] = name
                        log.info("Identified %s as: %s", pyannote_label, name)

        return _words_to_segments(words, label_map)
