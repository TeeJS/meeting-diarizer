#!/usr/bin/env python3
"""
Drop-in replacement for wyoming-transcribe.py using the meeting-diarizer service.
Outputs speaker-labeled transcript to stdout, errors to stderr.

Usage: python3 diarize-transcribe.py <audio_file> [host] [port]
"""

import os
import sys
import urllib.request
import urllib.error

DIARIZER_HOST = os.environ.get("DIARIZER_HOST", "192.168.1.25")
DIARIZER_PORT = int(os.environ.get("DIARIZER_PORT", "10301"))


def transcribe(audio_path: str, host: str, port: int) -> str:
    url = f"http://{host}:{port}/transcribe"

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    filename   = os.path.basename(audio_path)
    boundary   = "----MeetingDiarizerBoundary"
    body_parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: application/octet-stream\r\n\r\n".encode(),
        audio_data,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    print(f"Sending {audio_path} to {host}:{port} ...", file=sys.stderr)
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
        print("Usage: diarize-transcribe.py <audio_file> [host] [port]", file=sys.stderr)
        sys.exit(1)

    audio_file = sys.argv[1]
    host       = sys.argv[2] if len(sys.argv) > 2 else DIARIZER_HOST
    port       = int(sys.argv[3]) if len(sys.argv) > 3 else DIARIZER_PORT

    if not os.path.exists(audio_file):
        print(f"Error: file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    transcript = transcribe(audio_file, host, port)
    print(transcript)


if __name__ == "__main__":
    main()
