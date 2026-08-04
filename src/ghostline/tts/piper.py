"""Piper TTS: fast, local, offline text-to-speech.

Synthesizes text to 8 kHz mono 16-bit PCM bytes using a local ONNX model.
No API keys, no network calls, no cost. Synthesis is ~100-400ms for typical
phone utterances vs 1-2s for ElevenLabs API round-trips.

Usage:
    Set PIPER_MODEL_PATH in .env to the path of the .onnx model file.
    Models: https://huggingface.co/rhasspy/piper-voices
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from pydub import AudioSegment

__all__ = ("PiperVoiceService",)

_LOGGER = logging.getLogger(__name__)

_TARGET_FRAME_RATE: Final[int] = 8000
_TARGET_CHANNELS: Final[int] = 1
_FALLBACK_SILENCE_MS: Final[int] = 500


class PiperVoiceService:
    """Drop-in replacement for ElevenLabs VoiceService using local Piper TTS."""

    def __init__(self, model_path: str | Path) -> None:
        from piper import PiperVoice

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {self._model_path}")
        self._voice = PiperVoice.load(str(self._model_path))
        _LOGGER.info(
            "Piper TTS loaded: %s (sample_rate=%d)",
            self._model_path.name,
            self._voice.config.sample_rate,
        )

    async def synth(self, text: str, voice_id: str = "", persona: str = "professional") -> bytes:
        """Synthesize ``text`` to 8 kHz mono 16-bit PCM bytes.

        Compatible with the ElevenLabs VoiceService.synth() interface.
        voice_id and persona are accepted but ignored (Piper uses the
        loaded model's voice).
        """
        del voice_id, persona  # Piper uses the loaded model, not a voice_id
        try:
            chunks = list(self._voice.synthesize(text))
            if not chunks:
                _LOGGER.warning("Piper returned no audio chunks for: %r", text[:60])
                return self._silence_pcm()

            # Concatenate all chunks into one PCM buffer
            pcm_parts = []
            for chunk in chunks:
                pcm_parts.append(chunk.audio_int16_bytes)
            raw_pcm = b"".join(pcm_parts)

            # Convert from native sample rate to 8kHz mono via pydub
            source_rate = self._voice.config.sample_rate
            seg = AudioSegment(
                data=raw_pcm,
                sample_width=2,  # 16-bit
                frame_rate=source_rate,
                channels=1,
            )
            seg = seg.set_frame_rate(_TARGET_FRAME_RATE).set_channels(_TARGET_CHANNELS)
            return bytes(seg.raw_data)

        except Exception:
            _LOGGER.exception("Piper TTS synthesis failed")
            return self._silence_pcm()

    @staticmethod
    def _silence_pcm() -> bytes:
        """Return a short silence buffer as fallback."""
        n_samples = _TARGET_FRAME_RATE * _FALLBACK_SILENCE_MS // 1000
        return np.zeros(n_samples, dtype=np.int16).tobytes()

    async def clone(self, *args: object, **kwargs: object) -> None:
        """No-op: Piper doesn't support voice cloning."""
        _LOGGER.warning("Piper TTS does not support voice cloning; ignoring clone request")
