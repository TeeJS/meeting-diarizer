"""
Wyoming-protocol client for wyoming-faster-whisper.

Why this exists
---------------
The diarizer used to load its own faster-whisper model into VRAM. We've
moved transcription off-box to an existing wyoming-faster-whisper
instance (default: 192.168.1.25:10300). This client speaks the Wyoming
wire format so the diarizer can ask that server to transcribe each
speaker turn.

Wyoming is NOT HTTP. It is a line-oriented JSON protocol over plain TCP,
designed for the Home Assistant voice pipeline. Each event on the wire
looks like:

    <one-line JSON header>\n
    [optional JSON data block of length `data_length` bytes]
    [optional binary payload of length `payload_length` bytes]

The header is always one line of JSON terminated with a newline. If the
header object includes a `data_length` int, the next that-many bytes are
a JSON data block. If it includes `payload_length`, the next that-many
bytes are binary (in our case: raw PCM audio).

Transcription flow used by this client
--------------------------------------
For each audio segment we want transcribed, open a TCP connection and
send these events in order:

    {"type":"transcribe","data":{"language":"en"}}
    {"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}
    {"type":"audio-chunk","data":{"rate":16000,"width":2,"channels":1},"payload_length":N}
    <N bytes of signed 16-bit little-endian mono PCM>
    ... more audio-chunk events as needed ...
    {"type":"audio-stop"}

Then read events back until a `transcript` event arrives. Two shapes
exist in the wild — handle both:

    {"type":"transcript","data":{"text":"..."}}           (inline)
    {"type":"transcript","data_length":N}\n{"text":"..."} (data block)

Connection lifetime
-------------------
We open one TCP connection per `transcribe()` call. Reusing one
connection across many calls is possible but adds state-machine
complexity, and TCP connect to a LAN host is sub-millisecond — not
worth the complexity for our 4–6x/week use case.
"""

import json
import logging
import socket
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# 16 kHz mono signed-16-bit-little-endian PCM. wyoming-faster-whisper
# resamples internally so other rates work, but 16 kHz matches Whisper's
# training rate and minimizes bytes on the wire.
SAMPLE_RATE   = 16000
SAMPLE_WIDTH  = 2
CHANNELS      = 1

# Per-chunk size for audio-chunk events. 4096 samples ~= 256 ms of audio.
# Wyoming has no protocol-level minimum chunk size; this is just polite
# streaming so the server can begin decoding before audio-stop arrives.
CHUNK_SAMPLES = 4096

# Generous socket timeout — transcribing a long speaker turn on the
# medium model on a 3060 can take several seconds; large-v3 longer.
SOCK_TIMEOUT  = 60.0


class WyomingError(RuntimeError):
    """Raised on any Wyoming protocol or network failure."""


class WyomingClient:
    """Stateless client for wyoming-faster-whisper.

    Each call to `transcribe()` opens a fresh TCP connection. Use one
    `WyomingClient` instance per (host, port) backend; instances are
    cheap to construct and safe to discard.
    """

    def __init__(self, host: str, port: int, language: str = "en"):
        self._host     = host
        self._port     = port
        self._language = language

    def transcribe(self, pcm_int16: np.ndarray) -> str:
        """Transcribe one mono audio segment.

        Parameters
        ----------
        pcm_int16 : 1-D numpy array of int16 samples at 16 kHz.

        Returns
        -------
        The transcribed text, stripped of leading/trailing whitespace.
        Returns "" if the server transcribes the segment to nothing
        (silence, sub-speech noise, etc.).

        Raises
        ------
        WyomingError on any network or protocol failure.
        ValueError  on bad input shape.
        """
        if pcm_int16.ndim != 1:
            raise ValueError(
                f"WyomingClient.transcribe expects 1-D PCM, got shape {pcm_int16.shape}"
            )
        if pcm_int16.dtype != np.int16:
            # Don't silently round float audio — make the misuse visible
            # so callers know to convert explicitly.
            log.warning(
                "WyomingClient.transcribe got dtype %s — converting to int16. "
                "Callers should pass int16 PCM directly.", pcm_int16.dtype,
            )
            pcm_int16 = pcm_int16.astype(np.int16)

        audio_bytes = pcm_int16.tobytes()
        if not audio_bytes:
            return ""

        try:
            sock = socket.create_connection(
                (self._host, self._port), timeout=SOCK_TIMEOUT,
            )
        except OSError as e:
            raise WyomingError(
                f"could not connect to Wyoming server at "
                f"{self._host}:{self._port}: {e}"
            ) from e

        try:
            sock.settimeout(SOCK_TIMEOUT)
            self._send_header(sock, {
                "type": "transcribe",
                "data": {"language": self._language},
            })
            self._send_header(sock, {
                "type": "audio-start",
                "data": {
                    "rate":     SAMPLE_RATE,
                    "width":    SAMPLE_WIDTH,
                    "channels": CHANNELS,
                },
            })

            chunk_bytes = CHUNK_SAMPLES * SAMPLE_WIDTH
            for i in range(0, len(audio_bytes), chunk_bytes):
                chunk = audio_bytes[i : i + chunk_bytes]
                self._send_header(sock, {
                    "type": "audio-chunk",
                    "data": {
                        "rate":     SAMPLE_RATE,
                        "width":    SAMPLE_WIDTH,
                        "channels": CHANNELS,
                    },
                    "payload_length": len(chunk),
                })
                sock.sendall(chunk)

            self._send_header(sock, {"type": "audio-stop"})
            return self._read_transcript(sock).strip()
        finally:
            try:
                sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Wire-format internals
    # ------------------------------------------------------------------

    @staticmethod
    def _send_header(sock: socket.socket, event: dict) -> None:
        """Serialize one event header as compact JSON + newline and send it."""
        line = json.dumps(event, separators=(",", ":")) + "\n"
        sock.sendall(line.encode("utf-8"))

    def _read_transcript(self, sock: socket.socket) -> str:
        """Read events from `sock` until a `transcript` event arrives.

        Returns the text from that event. Raises WyomingError if the
        connection closes first or if a transcript event arrives without
        text in either shape we know about.
        """
        buf = b""
        while True:
            header_line, buf = self._read_line(sock, buf)
            if header_line is None:
                raise WyomingError(
                    "Wyoming server closed connection without sending transcript"
                )

            try:
                event = json.loads(header_line)
            except json.JSONDecodeError as e:
                raise WyomingError(
                    f"malformed Wyoming header: {header_line!r}: {e}"
                ) from e

            # `data_length` (when present): JSON data block follows.
            data_block: Optional[dict] = None
            data_length = event.get("data_length")
            if isinstance(data_length, int) and data_length > 0:
                raw, buf = self._read_n(sock, buf, data_length)
                try:
                    data_block = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise WyomingError(
                        f"malformed Wyoming data block for "
                        f"{event.get('type')!r}: {e}"
                    ) from e

            # `payload_length` (when present): binary payload follows.
            # We don't expect any for transcript events, but consume
            # defensively so we stay aligned with the stream.
            payload_length = event.get("payload_length")
            if isinstance(payload_length, int) and payload_length > 0:
                _, buf = self._read_n(sock, buf, payload_length)

            if event.get("type") == "transcript":
                # Two valid shapes: text inline in the header, or in the
                # data block. The user's notes confirm both happen in the
                # wild depending on the server build.
                inline = (event.get("data") or {}).get("text")
                if isinstance(inline, str):
                    return inline
                if isinstance(data_block, dict) and isinstance(data_block.get("text"), str):
                    return data_block["text"]
                raise WyomingError(
                    f"transcript event missing 'text': "
                    f"header={event!r} data={data_block!r}"
                )

            # Anything else (e.g., `info`, `transcript-start`) we just
            # consume and keep reading until a transcript arrives.

    @staticmethod
    def _read_line(
        sock: socket.socket, buf: bytes,
    ) -> tuple[Optional[bytes], bytes]:
        """Read until \\n. Returns (line_without_newline, remaining_buf).

        Returns (None, remaining_buf) on clean EOF before any data.
        """
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                # Clean EOF. If we have a partial line buffered, surface
                # it so the caller can decide; otherwise signal EOF.
                if buf:
                    return buf, b""
                return None, b""
            buf += chunk
        i = buf.index(b"\n")
        return buf[:i], buf[i + 1 :]

    @staticmethod
    def _read_n(
        sock: socket.socket, buf: bytes, n: int,
    ) -> tuple[bytes, bytes]:
        """Read exactly `n` bytes, drawing from `buf` first then the socket."""
        while len(buf) < n:
            chunk = sock.recv(max(4096, n - len(buf)))
            if not chunk:
                raise WyomingError(
                    f"unexpected EOF: needed {n} bytes, got {len(buf)}"
                )
            buf += chunk
        return buf[:n], buf[n:]
