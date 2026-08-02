"""
Pyannote speaker diarization + enrolled speaker identification.
"""

import logging
import numpy as np
import soundfile as sf
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
MAX_EMBED_SEGMENTS = 20
MIN_SEGMENT_SEC    = 0.5    # shorter than this does not embed reliably

# Diarization window. pyannote's clustering degrades over long recordings: on a
# 42-minute meeting it merged two speakers into one 640s cluster whose blended
# embedding matched nobody (0.25, margin 0.01), while the same voices in
# 10-minute windows scored 0.51-0.62. See _diarize_windowed().
CHUNK_SEC          = 630.0

# Cosine similarity at which two unidentified clusters from different windows
# are treated as the same person. Both sides come from the same recording, so
# this is an easier comparison than matching against an enrollment.
UNKNOWN_MERGE_SIM  = 0.55

# Reported in every speaker report so a threshold can be tuned from real runs
# without re-processing audio (similarity scores do not depend on it).
THRESHOLD_SWEEP    = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

AMBIGUOUS_MARGIN   = 0.05   # best-vs-second gap below which a match is a coin flip

# An unidentified speaker holding at least this share of the meeting is worth
# enrolling -- they are almost certainly a colleague who will recur.
ENROLL_CANDIDATE_PCT = 5.0


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


def _threshold_sweep(clusters: List[Dict]) -> List[Dict]:
    """What each candidate threshold would have produced for this recording.

    Similarity scores are computed independently of the threshold -- it only
    decides whether `matched` gets populated -- so a single run contains the
    data to evaluate every threshold retrospectively.

    `collisions` counts, within a single window, matched clusters beyond the
    number of distinct names -- i.e. two clusters in the same window claiming
    the same person, which means one speaker was split in two. The same name
    recurring across windows is expected and is not a collision.
    """
    total = sum(c["duration_sec"] for c in clusters) or 1.0
    sweep = []
    for t in THRESHOLD_SWEEP:
        hits  = [c for c in clusters if c["scores"] and c["scores"][0]["score"] >= t]
        named = sum(c["duration_sec"] for c in hits)

        by_window = {}
        for c in hits:
            by_window.setdefault(c.get("window", 0), []).append(c["scores"][0]["name"])
        collisions = sum(len(v) - len(set(v)) for v in by_window.values())

        sweep.append({
            "threshold":        t,
            "clusters_matched": len(hits),
            "distinct_names":   len({c["scores"][0]["name"] for c in hits}),
            "collisions":       collisions,
            "speech_named_pct": round(named / total * 100, 1),
        })
    return sweep


def _summarise_speakers(clusters: List[Dict]) -> List[Dict]:
    """Roll per-window clusters up into one entry per speaker.

    Windowed diarization emits the same person once per window they spoke in;
    this is the view worth reading.
    """
    total = sum(c["duration_sec"] for c in clusters) or 1.0
    agg: Dict[str, Dict] = {}
    for c in clusters:
        a = agg.setdefault(c["cluster"], {
            "label":        c["cluster"],
            "identified":   bool(c["matched"]),
            "duration_sec": 0.0,
            "windows":      0,
            "best_score":   None,
            "nearest":      None,
        })
        a["duration_sec"] += c["duration_sec"]
        a["windows"]      += 1
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

        Returns (embedding_or_None, segments_used, seconds_used).
        """
        usable = [s for s in segs if (s.end - s.start) >= MIN_SEGMENT_SEC]
        usable.sort(key=lambda s: s.end - s.start, reverse=True)

        embeddings, used_sec = [], 0.0
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
                used_sec += seg.end - seg.start
            except Exception:
                continue

        if not embeddings:
            return None, 0, 0.0

        avg = np.mean(embeddings, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        return avg, len(embeddings), used_sec

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

    def _analyze_clusters(self, audio: dict, timeline: List[tuple], threshold: float,
                          attendees_set: Optional[set]) -> Tuple[List[Dict], Dict]:
        """Embed and identify every speaker cluster in one window's timeline.

        Returns (clusters, embeddings). `clusters` is ordered to match
        sorted(unique pyannote labels); `embeddings` maps a pyannote label to
        its cluster embedding, used to stitch unidentified speakers across
        windows.
        """
        unique_spks = sorted(set(t[2] for t in timeline))
        index_map   = {spk: i for i, spk in enumerate(unique_spks)}

        have_enrollments = bool(self._store.list_speakers())
        if not have_enrollments:
            log.warning("No enrolled speakers — clusters will keep generic labels.")

        clusters, embeddings = [], {}
        for pyannote_label in unique_spks:
            segs = [Segment(s, e) for s, e, spk in timeline if spk == pyannote_label]
            duration = sum(seg.end - seg.start for seg in segs)

            entry = {
                "cluster":       _default_label(pyannote_label, index_map),
                "segment_count": len(segs),
                "duration_sec":  round(duration, 2),
                "segments_used": 0,
                "seconds_used":  0.0,
                "matched":       None,
                "margin":        None,
                "ambiguous":     False,
                "scores":        [],
            }
            embeddings[pyannote_label] = None

            if have_enrollments:
                emb, n_used, sec_used = self._cluster_embedding(audio, segs)
                entry["segments_used"] = n_used
                entry["seconds_used"]  = round(sec_used, 2)
                embeddings[pyannote_label] = emb

                if emb is None:
                    log.warning("%s: no segments >= %.1fs (of %d) — cannot identify",
                                entry["cluster"], MIN_SEGMENT_SEC, len(segs))
                else:
                    log.info("%s: embedding from %d segment(s), %.1fs of speech "
                             "(cluster total %.1fs)",
                             entry["cluster"], n_used, sec_used, duration)
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

        return clusters, embeddings

    def _assign_unknown(self, groups: List[Dict], emb: Optional[np.ndarray]) -> str:
        """Give an unidentified cluster a label that is stable across windows.

        Enrolled speakers stitch for free -- the same name in two windows is the
        same person by construction. Everyone else has to be matched window to
        window by embedding similarity, which is an easier comparison than
        matching against an enrollment: both sides come from this recording, so
        room and channel conditions are identical.
        """
        if emb is None:
            return "Unknown"

        best, best_sim = None, -1.0
        for g in groups:
            sim = _cosine_similarity(emb, g["mean"])
            if sim > best_sim:
                best, best_sim = g, sim

        if best is not None and best_sim >= UNKNOWN_MERGE_SIM:
            best["embs"].append(emb)
            mean = np.mean(best["embs"], axis=0)
            best["mean"] = mean / (np.linalg.norm(mean) + 1e-8)
            log.info("Unnamed cluster merged into %s (sim=%.4f)", best["label"], best_sim)
            return best["label"]

        label = (f"Speaker {_LABELS[len(groups)]}" if len(groups) < len(_LABELS)
                 else f"Speaker {len(groups) + 1}")
        groups.append({"label": label, "embs": [emb], "mean": emb})
        log.info("New unnamed speaker %s (closest existing group %.4f)",
                 label, best_sim if best is not None else -1.0)
        return label

    def _run_pipeline(self, audio: dict) -> List[tuple]:
        """Run diarization and flatten it to (start, end, label) tuples."""
        result = self._pipeline(audio)
        # pyannote 3.3+ returns DiarizeOutput; older versions return Annotation directly
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        return [
            (turn.start, turn.end, spk)
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]

    def _diarize_windowed(self, audio: dict, threshold: float,
                          attendees_set: Optional[set]) -> Tuple[List[tuple], List[Dict]]:
        """Diarize in windows and stitch the speakers back together.

        pyannote's clustering degrades over long recordings -- it merges
        speakers into blended clusters whose embeddings then match nobody.
        Diarizing ~10 minutes at a time keeps clustering clean; identities are
        recombined afterwards, by name for enrolled speakers and by embedding
        similarity for the rest.

        Recordings shorter than one window take a single pass, so this is a
        no-op for short clips.

        Returns (timeline, clusters) with timeline in absolute seconds and
        labels already resolved to their final display names.
        """
        sr    = audio["sample_rate"]
        total = audio["waveform"].shape[1] / sr
        n     = max(1, int(np.ceil(total / CHUNK_SEC)))
        width = total / n
        log.info("Diarizing %.0fs in %d window(s) of %.0fs", total, n, width)

        timeline: List[tuple] = []
        clusters: List[Dict]  = []
        unknown:  List[Dict]  = []

        for i in range(n):
            w_start = i * width
            w_end   = min((i + 1) * width, total)
            chunk   = _crop_audio(audio, w_start, w_end)

            local = self._run_pipeline(chunk)
            if not local:
                log.info("Window %d (%.0f-%.0fs): no speech detected", i, w_start, w_end)
                continue

            entries, embeddings = self._analyze_clusters(chunk, local, threshold, attendees_set)
            spks = sorted(set(t[2] for t in local))

            label_map = {}
            for spk, entry in zip(spks, entries):
                if entry["matched"]:
                    label_map[spk] = entry["matched"]
                else:
                    label_map[spk] = self._assign_unknown(unknown, embeddings.get(spk))
                entry["window"]  = i
                entry["cluster"] = label_map[spk]
                clusters.append(entry)

            for st, en, spk in local:
                timeline.append((st + w_start, en + w_start, label_map[spk]))

        timeline.sort(key=lambda t: t[0])
        return timeline, clusters

    def _build_report(self, clusters: List[Dict], threshold: float,
                      attendees_set: Optional[set], window_count: int) -> Dict:
        """Wrap the per-window clusters with the rolled-up speaker view, the
        threshold sweep, and anyone worth enrolling."""
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
            "windows":               window_count,
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
        attendees_set     = self._resolve_attendees(attendees)
        audio             = _load_audio(audio_path)
        _, clusters       = self._diarize_windowed(audio, threshold, attendees_set)
        windows           = (max(c["window"] for c in clusters) + 1) if clusters else 0
        return self._build_report(clusters, threshold, attendees_set, windows)

    def diarize(self, audio_path: str, words: List[Dict], threshold: float = SIMILARITY_THRESHOLD,
                attendees: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
        """
        Run diarization, align with word timestamps, identify speakers,
        and return (segments, speaker_report).

        If `attendees` is provided, enrolled speakers not in the list have
        ATTENDEE_OFFSET subtracted from their score during identification.
        """
        attendees_set      = self._resolve_attendees(attendees)
        audio              = _load_audio(audio_path)
        timeline, clusters = self._diarize_windowed(audio, threshold, attendees_set)
        windows            = (max(c["window"] for c in clusters) + 1) if clusters else 0

        # Assign each word to a speaker by midpoint. Labels in the timeline are
        # already final display names, so the map below is an identity.
        for w in words:
            mid = (w["start"] + w["end"]) / 2
            w["speaker"] = "Unknown"
            for start, end, label in timeline:
                if start <= mid <= end:
                    w["speaker"] = label
                    break

        identity = {label: label for _, _, label in timeline}
        report   = self._build_report(clusters, threshold, attendees_set, windows)
        return _words_to_segments(words, identity), report
