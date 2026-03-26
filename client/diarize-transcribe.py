#!/usr/bin/env python3
"""
Drop-in replacement for wyoming-transcribe.py using the meeting-diarizer service.
Outputs speaker-labeled transcript to stdout, errors to stderr.

Usage: python3 diarize-transcribe.py <audio_file> [host] [port] [--threshold 0.65]
"""

import os
import sys
import urllib.request
import urllib.error

DIARIZER_HOST      = os.environ.get("DIARIZER_HOST", "192.168.1.25")
DIARIZER_PORT      = int(os.environ.get("DIARIZER_PORT", "10301"))
DEFAULT_THRESHOLD  = float(os.environ.get("DIARIZER_THRESHOLD", "0.75"))


def transcribe(audio_path: str, host: str, port: int, threshold: float = DEFAULT_THRESHOLD) -> str:
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

    segments = result.get("segments", [])
    if not segments:
        return ""

    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text    = seg.get("text", "").strip()
        if text:
            lines.append(f"{speaker}: {text}")

    return "\n\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: diarize-transcribe.py <audio_file> [host] [port] [--threshold 0.65]", file=sys.stderr)
        sys.exit(1)

    args       = sys.argv[1:]
    audio_file = args[0]
    host       = DIARIZER_HOST
    port       = DIARIZER_PORT
    threshold  = DEFAULT_THRESHOLD

    # Parse optional positional host/port and --threshold flag
    i = 1
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
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

    transcript = transcribe(audio_file, host, port, threshold=threshold)
    print(transcript)


if __name__ == "__main__":
    main()
