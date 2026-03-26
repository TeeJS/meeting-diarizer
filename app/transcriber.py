"""
Faster-whisper transcription wrapper.
Returns word-level timestamps needed for speaker alignment.
"""

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Dict

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)


def _ensure_16k_mono(path: str) -> tuple:
    """Return (path, is_temp). Convert to 16kHz mono WAV via ffmpeg if needed."""
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                if wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                    return path, False
        except Exception:
            pass

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name, True


class Transcriber:
    def __init__(self, model_size: str = "large-v3"):
        log.info("Loading faster-whisper model: %s", model_size)
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                self._model = WhisperModel(model_size, device="cuda", compute_type="auto")
                log.info("faster-whisper loaded on GPU (auto compute type).")
                return
            except Exception as e:
                log.warning("GPU init failed (%s) — falling back to CPU int8", e)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        log.warning("faster-whisper running on CPU int8 — transcription will be slow.")
        log.info("faster-whisper ready.")

    def transcribe(self, audio_path: str) -> List[Dict]:
        """
        Transcribe audio and return word-level results.
        Each item: {"word": str, "start": float, "end": float}
        """
        wav_path, is_temp = _ensure_16k_mono(audio_path)
        try:
            segments, _ = self._model.transcribe(
                wav_path,
                word_timestamps=True,
                language="en",
                vad_filter=True,
            )
            words = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        words.append({"word": w.word, "start": w.start, "end": w.end})
            return words
        finally:
            if is_temp:
                Path(wav_path).unlink(missing_ok=True)
