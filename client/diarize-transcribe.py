#!/usr/bin/env python3
"""
Drop-in replacement for wyoming-transcribe.py using the meeting-diarizer service.
Outputs speaker-labeled transcript to stdout, errors to stderr.

Usage: python3 diarize-transcribe.py <audio_file> [host] [port] [--threshold 0.35]
                                     [--report <path>]

The speaker detection report is always written to stderr; --report also saves
it to a file.
"""

import os
import sys
import urllib.request
import urllib.error

DIARIZER_HOST      = os.environ.get("DIARIZER_HOST", "192.168.1.25")
DIARIZER_PORT      = int(os.environ.get("DIARIZER_PORT", "10301"))
# Matches the service default. This was 0.75, which sits above the score of a
# correct match on clean audio -- every genuine identification failed.
DEFAULT_THRESHOLD  = float(os.environ.get("DIARIZER_THRESHOLD", "0.35"))


def format_report(report: dict) -> str:
    """Render the speaker_report as readable text."""
    if not report:
        return "(no speaker report returned)"

    out = []
    out.append("=" * 78)
    out.append(f"SPEAKER DETECTION REPORT   threshold={report.get('threshold_used')}"
               f"   clusters={report.get('cluster_count')}"
               f"   named={report.get('speech_named_pct')}% of speech")
    if report.get("attendees_applied"):
        out.append(f"attendees applied: {', '.join(report['attendees_applied'])}")
    if report.get("collisions"):
        out.append(f"WARNING: {report['collisions']} cluster(s) share a name with "
                   f"another cluster -- likely one speaker split in two")
    out.append("=" * 78)

    total = report.get("total_speech_sec") or 1.0
    for c in report.get("clusters", []):
        scores = c.get("scores", [])
        share  = (c["duration_sec"] / total * 100) if total else 0
        out.append("")
        out.append(f"{c['cluster']}   segments={c['segment_count']}   "
                   f"duration={c['duration_sec']:.1f}s ({share:.0f}% of speech)")
        out.append(f"   embedded from {c['segments_used']} segment(s), "
                   f"{c['seconds_used']:.1f}s")
        out.append(f"   matched : {c['matched'] or '-- NO MATCH --'}")
        if scores:
            out.append(f"   best    : {scores[0]['name']:<18} {scores[0]['score']:.4f}")
        if len(scores) > 1:
            out.append(f"   second  : {scores[1]['name']:<18} {scores[1]['score']:.4f}")
            flag = "   <-- AMBIGUOUS" if c.get("ambiguous") else ""
            out.append(f"   MARGIN  : {c['margin']:.4f}{flag}")
            rest = ", ".join(f"{s['name'].split()[0]}={s['score']:.3f}"
                             for s in scores[2:7])
            if rest:
                out.append(f"   next    : {rest}")

    out.append("")
    out.append("=" * 78)
    out.append("THRESHOLD SWEEP  (what each cutoff would have produced here)")
    out.append(f"{'thresh':>8} {'matched':>9} {'names':>7} {'collide':>8} {'named %':>9}")
    for s in report.get("threshold_sweep", []):
        out.append(f"{s['threshold']:>8.2f} {s['clusters_matched']:>9} "
                   f"{s['distinct_names']:>7} {s['collisions']:>8} "
                   f"{s['speech_named_pct']:>8.1f}%")
    out.append("=" * 78)
    out.append(f"total attributed speech: {report.get('total_speech_sec')}s")
    return "\n".join(out)


def transcribe(audio_path: str, host: str, port: int,
               threshold: float = DEFAULT_THRESHOLD) -> tuple:
    url = f"http://{host}:{port}/transcribe"

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    filename   = os.path.basename(audio_path)
    boundary   = "----MeetingDiarizerBoundary"
    threshold_bytes = str(threshold).encode()
    body_parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        audio_data,
        f"\r\n--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="threshold"\r\n\r\n'.encode(),
        threshold_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    print(f"Sending {audio_path} to {host}:{port} (threshold={threshold}) ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            import json
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error connecting to diarizer at {host}:{port}: {e}", file=sys.stderr)
        sys.exit(1)

    report   = result.get("speaker_report", {})
    segments = result.get("segments", [])

    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text    = seg.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n\n".join(lines), report


def main():
    if len(sys.argv) < 2:
        print("Usage: diarize-transcribe.py <audio_file> [host] [port] [--threshold 0.65]", file=sys.stderr)
        sys.exit(1)

    args        = sys.argv[1:]
    audio_file  = args[0]
    host        = DIARIZER_HOST
    port        = DIARIZER_PORT
    threshold   = DEFAULT_THRESHOLD
    report_path = None

    # Parse optional positional host/port and --threshold / --report flags
    i = 1
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--report" and i + 1 < len(args):
            report_path = args[i + 1]
            i += 2
        elif i == 1 and not args[i].startswith("--"):
            host = args[i]
            i += 1
        elif i == 2 and not args[i].startswith("--"):
            port = int(args[i])
            i += 1
        else:
            i += 1

    if not os.path.exists(audio_file):
        print(f"Error: file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    transcript, report = transcribe(audio_file, host, port, threshold=threshold)

    rendered = format_report(report)
    print(rendered, file=sys.stderr)
    if report_path:
        with open(report_path, "w", encoding="utf8") as f:
            f.write(rendered + "\n")
        print(f"Speaker report written to {report_path}", file=sys.stderr)

    print(transcript)


if __name__ == "__main__":
    main()
