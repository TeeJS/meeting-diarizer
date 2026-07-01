"""
Pyannote speaker diarization + enrolled speaker identification.
"""

import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import List, Dict, Optional

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

from .enrollment import EnrollmentStore

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.35
ATTENDEE_OFFSET      = 0.15  # subtracted from similarity scores of non-attendees
EMBEDDING_MODEL      = "pyannote/wespeaker-voxceleb-resnet34-LM"
_LABELS              = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


PYANNOTE_SR = 16000  # sample rate pyannote models expect


def _load_audio(path: str) -> dict:
    """Load audio file as a pyannote-compatible waveform dict using soundfile only."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)          # (channels, samples)
    if waveform.shape[0] > 1:                    # mix down to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != PYANNOTE_SR:                        # resample if needed
        orig_len  = waveform.shape[1]
        new_len   = int(orig_len * PYANNOTE_SR / sr)
        waveform  = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
    return {"waveform": waveform, "sample_rate": PYANNOTE_SR}


def _crop_audio(audio: dict, start: float, end: float) -> dict:
    """Crop a waveform dict to the given time range (seconds)."""
    sr = audio["sample_rate"]
    s  = int(start * sr)
    e  = int(end   * sr)
    return {"waveform": audio["waveform"][:, s:e], "sample_rate": sr}


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
        emb_model = emb_model.to(torch.device("cuda"))
        self._inference = Inference(emb_model, window="whole")

        self._store = enrollment_store
        log.info("Diarizer ready.")

    def enroll_speaker(self, name: str, audio_path: str):
        """Extract and store a speaker embedding from a reference audio file."""
        audio     = _load_audio(audio_path)
        embedding = self._inference(audio)
        self._store.save(name, np.array(embedding))
        log.info("Enrolled speaker: %s", name)

    def _identify(self, embedding: np.ndarray, threshold: float = SIMILARITY_THRESHOLD,
                  attendees: Optional[set] = None, return_scores: bool = False):
        """Compare embedding to enrolled speakers. Returns name or None.

        If `attendees` is provided, enrolled speakers NOT in that set have
        ATTENDEE_OFFSET subtracted from their similarity score before ranking.
        Attendees in the set keep their raw scores.

        If `return_scores` is True, returns (name_or_None, scores) where
        scores is a list of per-enrolled-speaker dicts sorted best-first —
        used by the /identify diagnostic endpoint.
        """
        if np.any(np.isnan(embedding)):
            log.warning("Skipping segment: NaN embedding (segment too short or degenerate)")
            return (None, []) if return_scores else None

        scored = []
        best_name, best_score = None, -1.0
        for name, enrolled in self._store.all_embeddings().items():
            raw = _cosine_similarity(embedding, enrolled)
            is_attendee = attendees is None or name in attendees
            score = raw if is_attendee else raw - ATTENDEE_OFFSET
            scored.append({
                "name":     name,
                "raw":      round(raw, 4),
                "score":    round(score, 4),
                "attendee": is_attendee if attendees is not None else None,
            })
            if score > best_score:
                best_name, best_score = name, score
        scored.sort(key=lambda s: s["score"], reverse=True)

        legend = " (*=attendee)" if attendees is not None else ""
        log.info("Speaker similarity scores (threshold=%.2f%s): %s", threshold, legend,
                 ", ".join(f"{s['name']}={s['score']}{'*' if s['attendee'] else ''}" for s in scored))

        matched = best_score >= threshold
        if matched:
            log.info("  → Matched: %s (%.4f)", best_name, best_score)
        else:
            log.info("  → No match (best was %s at %.4f)", best_name, best_score)

        result_name = best_name if matched else None
        return (result_name, scored) if return_scores else result_name

    def _resolve_attendees(self, attendees: Optional[List[str]]) -> Optional[set]:
        """Validate attendees against the enrolled set; log mismatches so name-format
        drift (e.g. "Schmitz, TJ" vs "T.J. Schmitz") is visible instead of silent."""
        if not attendees:
            return None
        enrolled  = set(self._store.list_speakers())
        requested = set(attendees)
        attendees_set = requested & enrolled
        unmatched = requested - enrolled
        if unmatched:
            log.warning("Attendees not in enrolled set (no offset benefit, no penalty either): %s",
                        sorted(unmatched))
        log.info("Attendees recognized for offset (n=%d): %s", len(attendees_set), sorted(attendees_set))
        return attendees_set

    def identify_speakers(self, audio_path: str, threshold: float = SIMILARITY_THRESHOLD,
                           attendees: Optional[List[str]] = None) -> List[Dict]:
        """
        Run diarization + enrolled-speaker identification only — no transcription.
        Mirrors the clustering and embedding-extraction logic in diarize(), for
        diagnosing enrollment/threshold issues against a sample clip.

        Returns one entry per detected speaker cluster with its match (if any)
        and the full similarity score breakdown against every enrolled speaker.
        """
        attendees_set = self._resolve_attendees(attendees)

        audio      = _load_audio(audio_path)
        result     = self._pipeline(audio)
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        timeline   = [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
        unique_spks = sorted(set(t[2] for t in timeline))
        index_map   = {spk: i for i, spk in enumerate(unique_spks)}

        clusters = []
        for pyannote_label in unique_spks:
            speaker_segs = [Segment(s, e) for s, e, spk in timeline if spk == pyannote_label]
            duration = sum(seg.end - seg.start for seg in speaker_segs)

            embeddings = []
            for seg in speaker_segs[:10]:  # cap for speed, matches diarize()
                try:
                    if seg.end - seg.start < 0.5:
                        continue
                    cropped = _crop_audio(audio, seg.start, seg.end)
                    emb     = np.array(self._inference(cropped))
                    if np.any(np.isnan(emb)):
                        continue
                    embeddings.append(emb)
                except Exception:
                    continue

            entry = {
                "cluster":       _default_label(pyannote_label, index_map),
                "segment_count": len(speaker_segs),
                "duration_sec":  round(duration, 2),
                "matched":       None,
                "scores":        [],
            }
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                name, scores = self._identify(avg_emb, threshold=threshold,
                                              attendees=attendees_set, return_scores=True)
                entry["matched"] = name
                entry["scores"]  = scores
            clusters.append(entry)

        return clusters

    def diarize(self, audio_path: str, words: List[Dict], threshold: float = SIMILARITY_THRESHOLD,
                attendees: Optional[List[str]] = None) -> List[Dict]:
        """
        Run diarization, align with word timestamps, identify speakers,
        and return grouped segments.

        If `attendees` is provided, enrolled speakers not in the list have
        ATTENDEE_OFFSET subtracted from their score during identification.
        """
        attendees_set = self._resolve_attendees(attendees)
        audio      = _load_audio(audio_path)
        result     = self._pipeline(audio)
        # pyannote 3.3+ returns DiarizeOutput; older versions return Annotation directly
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        timeline   = [
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
                        if seg.end - seg.start < 0.5:  # skip segments too short to embed reliably
                            continue
                        cropped = _crop_audio(audio, seg.start, seg.end)
                        emb     = np.array(self._inference(cropped))
                        if np.any(np.isnan(emb)):
                            log.warning("Skipping %.2fs segment [%.2f-%.2f]: NaN embedding",
                                        seg.end - seg.start, seg.start, seg.end)
                            continue
                        embeddings.append(emb)
                    except Exception:
                        continue

                if embeddings:
                    avg_emb = np.mean(embeddings, axis=0)
                    name    = self._identify(avg_emb, threshold=threshold, attendees=attendees_set)
                    if name:
                        label_map[pyannote_label] = name
                        log.info("Identified %s as: %s", pyannote_label, name)

        return _words_to_segments(words, label_map)
