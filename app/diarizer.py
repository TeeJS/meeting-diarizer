"""
Pyannote speaker diarization + enrolled speaker identification.
"""

import logging
import re
import subprocess
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment

from .enrollment import EnrollmentStore

log = logging.getLogger(__name__)

# Genuine matches measured across five runs bottomed out at 0.506; confirmed
# strangers topped out at 0.311 and blended/cross-speaker clusters at 0.362.
# 0.45 sits in the gap. At the previous 0.35 an unenrolled attendee scored
# 0.362 against the nearest enrolled voice and was published under their name.
SIMILARITY_THRESHOLD = 0.45
ATTENDEE_OFFSET      = 0.15  # subtracted from similarity scores of non-attendees
EMBEDDING_MODEL      = "pyannote/wespeaker-voxceleb-resnet34-LM"
_LABELS              = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Segment selection for cluster embeddings. These are chosen longest-first --
# see _cluster_embedding() for why the order matters more than the count.
#
# Raised from 20 to 30. The cap binds hardest on people who speak in short
# turns: one speaker with 222s of speech across turns averaging 3.9s yielded
# only 77.7s of usable audio for a re-enrollment, because 20 spans was all the
# report would surface. It also means the cluster embedding itself averages
# over more of the speaker. Cost is 50% more embedding inferences per cluster,
# which is the smaller half of the work next to diarization itself.
MAX_EMBED_SEGMENTS = 30
MIN_SEGMENT_SEC    = 0.5    # shorter than this does not embed reliably

# Reported in every speaker report so a threshold can be tuned from real runs
# without re-processing audio (similarity scores do not depend on it).
THRESHOLD_SWEEP    = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

AMBIGUOUS_MARGIN   = 0.05   # best-vs-second gap below which a match is a coin flip

# An unidentified speaker holding at least this share of the meeting is worth
# enrolling -- they are almost certainly a colleague who will recur.
ENROLL_CANDIDATE_PCT = 5.0


PYANNOTE_SR = 16000  # sample rate pyannote models expect


def _load_audio(path: str) -> dict:
    """Decode to 16 kHz mono with ffmpeg, as a pyannote-compatible dict.

    This used to read the file with soundfile and resample with
    torch.nn.functional.interpolate(mode="linear"), which quietly destroyed
    long recordings. interpolate computes its source coordinates in the
    tensor's dtype, and float32 represents consecutive integers exactly only up
    to 2**24 = 16,777,216. A 42-minute meeting at 48 kHz is 121 million
    samples, so coordinates beyond roughly the first six minutes were off by
    1-8 samples, varying with position -- broadband distortion that worsened
    the further into the file it went. Measured against a float64 reference on
    a real meeting: 20.4 dB SNR over the first ten minutes, falling to 6.3 dB
    over the last. Speaker embeddings did not survive it. The same audio range
    scored 0.5616 for the correct speaker with exact coordinates and 0.1399
    with float32 ones.

    ffmpeg resamples properly, in one step, with an anti-aliasing filter, and
    is already how transcriber.py prepares audio -- which is why transcription
    quality was never affected while identification was.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-ac", "1", "-ar", str(PYANNOTE_SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    # copy() because frombuffer is read-only and torch needs a writable array
    samples  = np.frombuffer(out, dtype=np.float32).copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)   # (1, samples)
    return {"waveform": waveform, "sample_rate": PYANNOTE_SR}


def _crop_audio(audio: dict, start: float, end: float) -> dict:
    """Crop a waveform dict to the given time range (seconds)."""
    sr = audio["sample_rate"]
    s  = int(start * sr)
    e  = int(end   * sr)
    return {"waveform": audio["waveform"][:, s:e], "sample_rate": sr}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def normalize_name(name: str) -> str:
    """Fold a display name to a comparison key.

    Teams writes "Evenson, Matthew" while Outlook writes "Matthew Evenson" for
    the same person, and both forms appear throughout real calendar data. Exact
    string matching therefore fails depending on which system an attendee list
    was copied from -- and a failed match is worse than passing no attendees at
    all, because the speaker is then treated as absent and takes the
    ATTENDEE_OFFSET penalty against their own voice.

    Word order, punctuation, spacing and case are folded, so "Schmitz, TJ",
    "Schmitz, T.J." and "T.J. Schmitz" all agree. Diminutives are left alone:
    "Matt" and "Matthew" are deliberately NOT treated as equal, since guessing
    there risks merging two real people. Unmatched names are logged so the
    drift stays visible.
    """
    n = (name or "").strip()
    if "," in n:
        last, _, first = n.partition(",")
        n = f"{first.strip()} {last.strip()}"
    # drop rather than substitute, so "T.J." and "TJ" collapse the same way
    return re.sub(r"[^0-9a-z]", "", n.lower())


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


def _threshold_sweep(clusters: List[Dict]) -> List[Dict]:
    """What each candidate threshold would have produced for this recording.

    Similarity scores are computed independently of the threshold -- it only
    decides whether `matched` gets populated -- so a single run contains the
    data to evaluate every threshold retrospectively.

    `collisions` counts matched clusters beyond the number of distinct names --
    two clusters claiming the same person, which means one speaker was split in
    two. A good threshold names most of the speech with no collisions.
    """
    total = sum(c["duration_sec"] for c in clusters) or 1.0
    sweep = []
    for t in THRESHOLD_SWEEP:
        hits  = [c for c in clusters if c["scores"] and c["scores"][0]["score"] >= t]
        named = sum(c["duration_sec"] for c in hits)
        names = [c["scores"][0]["name"] for c in hits]
        sweep.append({
            "threshold":        t,
            "clusters_matched": len(hits),
            "distinct_names":   len(set(names)),
            "collisions":       len(names) - len(set(names)),
            "speech_named_pct": round(named / total * 100, 1),
        })
    return sweep


def _summarise_speakers(clusters: List[Dict]) -> List[Dict]:
    """Roll clusters up into one entry per speaker.

    Usually one cluster per person, but pyannote can split someone across two,
    in which case both carry the same matched name and are merged here -- and
    the sweep's collision count flags that it happened.
    """
    total = sum(c["duration_sec"] for c in clusters) or 1.0
    agg: Dict[str, Dict] = {}
    for c in clusters:
        a = agg.setdefault(c["cluster"], {
            "label":        c["cluster"],
            "identified":   bool(c["matched"]),
            "duration_sec": 0.0,
            "clusters":     0,
            "best_score":   None,
            "nearest":      None,
        })
        a["duration_sec"] += c["duration_sec"]
        a["clusters"]     += 1
        if c["scores"]:
            top = c["scores"][0]
            if a["best_score"] is None or top["score"] > a["best_score"]:
                a["best_score"] = top["score"]
                a["nearest"]    = top["name"]

    out = []
    for a in agg.values():
        a["duration_sec"] = round(a["duration_sec"], 2)
        a["speech_pct"]   = round(a["duration_sec"] / total * 100, 1)
        out.append(a)
    return sorted(out, key=lambda a: -a["duration_sec"])


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

    def _cluster_embedding(self, audio: dict, segs: List[Segment]) -> Tuple:
        """Compute one averaged embedding representing a speaker cluster.

        Segments are selected LONGEST-FIRST rather than in the order pyannote
        emitted them. Chronological order put the opening moments of the
        meeting first -- greetings, "yeah", "mhm" -- which embed poorly, and it
        penalised the people who spoke most, since their leading segments were
        the least representative slice of their speech.

        Each segment embedding is L2-normalised before averaging so that a
        single loud or long segment cannot dominate the mean direction.

        Returns (embedding_or_None, used) where `used` lists the (start, end)
        spans that actually contributed. Those spans are reported so a speaker's
        best audio can be cut back out of the recording and re-enrolled --
        enrolling from the setup a person is actually recorded on is worth far
        more than enrolling from a clean reference (one speaker went from 0.5739
        to 0.9755 on exactly that change).
        """
        usable = [s for s in segs if (s.end - s.start) >= MIN_SEGMENT_SEC]
        usable.sort(key=lambda s: s.end - s.start, reverse=True)

        embeddings, used = [], []
        for seg in usable[:MAX_EMBED_SEGMENTS]:
            try:
                cropped = _crop_audio(audio, seg.start, seg.end)
                emb     = np.array(self._inference(cropped))
                if np.any(np.isnan(emb)):
                    log.warning("Skipping %.2fs segment [%.2f-%.2f]: NaN embedding",
                                seg.end - seg.start, seg.start, seg.end)
                    continue
                norm = float(np.linalg.norm(emb))
                if norm < 1e-8:
                    continue
                embeddings.append(emb / norm)
                used.append((seg.start, seg.end))
            except Exception:
                continue

        if not embeddings:
            return None, []

        avg = np.mean(embeddings, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg, used

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
        """Map requested attendee names onto enrolled speakers.

        Matching is done on normalize_name(), so "Evenson, Matthew" from a Teams
        transcript and "Matthew Evenson" from an Outlook invite both resolve to
        the same enrollment. Names that still fail to match are logged, since a
        silent miss costs that speaker the attendee offset against their own
        voice.
        """
        if not attendees:
            return None

        by_norm = {normalize_name(e): e for e in self._store.list_speakers()}
        attendees_set, unmatched = set(), []
        for requested in attendees:
            enrolled = by_norm.get(normalize_name(requested))
            if enrolled:
                attendees_set.add(enrolled)
            else:
                unmatched.append(requested)

        if unmatched:
            log.warning("Attendees not in enrolled set (no offset benefit, no penalty "
                        "either) — check for a directory rename: %s", sorted(unmatched))
        log.info("Attendees recognized for offset (n=%d): %s",
                 len(attendees_set), sorted(attendees_set))
        return attendees_set

    def _analyze_clusters(self, audio: dict, timeline: List[tuple], threshold: float,
                          attendees_set: Optional[set]) -> List[Dict]:
        """Embed and identify every speaker cluster in the timeline.

        Returns one entry per cluster, ordered to match sorted(unique pyannote
        labels) so the caller can zip them back together.
        """
        unique_spks = sorted(set(t[2] for t in timeline))
        index_map   = {spk: i for i, spk in enumerate(unique_spks)}

        have_enrollments = bool(self._store.list_speakers())
        if not have_enrollments:
            log.warning("No enrolled speakers — clusters will keep generic labels.")

        clusters = []
        for pyannote_label in unique_spks:
            segs = [Segment(s, e) for s, e, spk in timeline if spk == pyannote_label]
            duration = sum(seg.end - seg.start for seg in segs)

            entry = {
                "cluster":       _default_label(pyannote_label, index_map),
                "segment_count": len(segs),
                "duration_sec":  round(duration, 2),
                "segments_used": 0,
                "seconds_used":  0.0,
                "embed_segments": [],
                "matched":       None,
                "margin":        None,
                "ambiguous":     False,
                "scores":        [],
            }

            if have_enrollments:
                emb, used = self._cluster_embedding(audio, segs)
                sec_used = sum(e - s for s, e in used)
                entry["segments_used"]  = len(used)
                entry["seconds_used"]   = round(sec_used, 2)
                entry["embed_segments"] = [[round(s, 2), round(e, 2)] for s, e in used]

                if emb is None:
                    log.warning("%s: no segments >= %.1fs (of %d) — cannot identify",
                                entry["cluster"], MIN_SEGMENT_SEC, len(segs))
                else:
                    log.info("%s: embedding from %d segment(s), %.1fs of speech "
                             "(cluster total %.1fs)",
                             entry["cluster"], len(used), sec_used, duration)
                    name, scores = self._identify(emb, threshold=threshold,
                                                  attendees=attendees_set,
                                                  return_scores=True)
                    entry["matched"] = name
                    entry["scores"]  = scores
                    if len(scores) > 1:
                        margin = round(scores[0]["score"] - scores[1]["score"], 4)
                        entry["margin"]    = margin
                        entry["ambiguous"] = margin < AMBIGUOUS_MARGIN

            clusters.append(entry)

        return clusters

    def _run_pipeline(self, audio: dict) -> List[tuple]:
        """Run diarization and flatten it to (start, end, label) tuples."""
        result = self._pipeline(audio)
        # pyannote 3.3+ returns DiarizeOutput; older versions return Annotation directly
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        return [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]

    def _diarize(self, audio: dict, threshold: float,
                 attendees_set: Optional[set]) -> Tuple[List[tuple], List[Dict], Dict]:
        """Diarize the whole recording in one pass and label every cluster.

        An earlier version split long recordings into ~10 minute windows and
        stitched the speakers back together. That was compensating for the
        float32 resampling bug in _load_audio, not for anything in pyannote:
        with the audio path fixed, a single pass scores higher on every speaker
        and, on a 42-minute meeting, correctly isolated a sixth participant
        that windowing had silently folded into someone else. One pass also
        gives each speaker a single embedding drawn from their best segments
        across the entire meeting rather than the best within each window.

        Returns (timeline, clusters, label_map), where label_map takes a
        pyannote label to its display name.
        """
        timeline = self._run_pipeline(audio)
        if not timeline:
            log.warning("No speech detected.")
            return [], [], {}

        clusters = self._analyze_clusters(audio, timeline, threshold, attendees_set)
        spks     = sorted(set(t[2] for t in timeline))

        # Identified clusters take the speaker's name; the rest are lettered in
        # order of appearance, so "Speaker A" is always the first unknown voice
        # rather than whichever cluster pyannote happened to emit first.
        label_map, unnamed = {}, 0
        for spk, entry in zip(spks, clusters):
            if entry["matched"]:
                entry["cluster"] = entry["matched"]
            else:
                entry["cluster"] = (f"Speaker {_LABELS[unnamed]}"
                                    if unnamed < len(_LABELS) else f"Speaker {unnamed + 1}")
                unnamed += 1
            label_map[spk] = entry["cluster"]

        return timeline, clusters, label_map

    def _build_report(self, clusters: List[Dict], threshold: float,
                      attendees_set: Optional[set]) -> Dict:
        """Wrap the clusters with the rolled-up speaker view, the threshold
        sweep, and anyone worth enrolling."""
        total    = sum(c["duration_sec"] for c in clusters)
        speakers = _summarise_speakers(clusters)
        named    = sum(s["duration_sec"] for s in speakers if s["identified"])

        candidates = [
            {
                "label":        s["label"],
                "duration_sec": s["duration_sec"],
                "speech_pct":   s["speech_pct"],
                "nearest":      s["nearest"],
                "best_score":   s["best_score"],
            }
            for s in speakers
            if not s["identified"] and s["speech_pct"] >= ENROLL_CANDIDATE_PCT
        ]
        if candidates:
            log.warning("Unidentified speakers worth enrolling: %s",
                        ", ".join(f"{c['label']} ({c['speech_pct']}% of speech)"
                                  for c in candidates))

        return {
            "threshold_used":        threshold,
            "attendees_applied":     sorted(attendees_set) if attendees_set else None,
            "speaker_count":         len(speakers),
            "total_speech_sec":      round(total, 2),
            "speech_named_pct":      round(named / total * 100, 1) if total else 0.0,
            "speakers":              speakers,
            "enrollment_candidates": candidates,
            "clusters":              clusters,
            "threshold_sweep":       _threshold_sweep(clusters),
        }

    def identify_speakers(self, audio_path: str, threshold: float = SIMILARITY_THRESHOLD,
                           attendees: Optional[List[str]] = None) -> Dict:
        """
        Run diarization + enrolled-speaker identification only — no transcription.
        For diagnosing enrollment/threshold issues against a sample clip.

        Returns the same speaker report that diarize() produces.
        """
        attendees_set  = self._resolve_attendees(attendees)
        audio          = _load_audio(audio_path)
        _, clusters, _ = self._diarize(audio, threshold, attendees_set)
        return self._build_report(clusters, threshold, attendees_set)

    def diarize(self, audio_path: str, words: List[Dict], threshold: float = SIMILARITY_THRESHOLD,
                attendees: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
        """
        Run diarization, align with word timestamps, identify speakers,
        and return (segments, speaker_report).

        If `attendees` is provided, enrolled speakers not in the list have
        ATTENDEE_OFFSET subtracted from their score during identification.
        """
        attendees_set = self._resolve_attendees(attendees)
        audio         = _load_audio(audio_path)
        timeline, clusters, label_map = self._diarize(audio, threshold, attendees_set)

        # Assign each word to a speaker by the midpoint of its timing
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            w["speaker"] = "Unknown"
            for start, end, spk in timeline:
                if start <= mid <= end:
                    w["speaker"] = spk
                    break

        report = self._build_report(clusters, threshold, attendees_set)
        return _words_to_segments(words, label_map), report
