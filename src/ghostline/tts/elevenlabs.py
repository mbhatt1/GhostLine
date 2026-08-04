"""ElevenLabs voice-cloning and text-to-speech service.

Replaces the untyped ``VoiceService`` in legacy ``main.py``. Uses
``httpx.AsyncClient`` for both endpoints (testable with respx):

- ``POST /v1/voices/add`` → clone a voice from a WAV/M4A sample
- ``POST /v1/text-to-speech/{voice_id}`` → synthesize PCM at 8 kHz mono

Errors degrade gracefully: on synth failure we return a short silence
buffer so the call doesn't dead-air, matching legacy behaviour.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import httpx
from pydub import AudioSegment

__all__ = ("SynthesisError", "VoiceService")

_LOGGER = logging.getLogger(__name__)

# ElevenLabs HTTP API base.
_API_BASE: Final[str] = "https://api.elevenlabs.io/v1"
# HTTP status code for a successful response.
_HTTP_OK: Final[int] = 200
# Output format Twilio Media Streams expects: 8 kHz mono 16-bit PCM.
_TARGET_FRAME_RATE: Final[int] = 8000
_TARGET_CHANNELS: Final[int] = 1
# Sanity threshold: a successful synth should produce >= this many PCM bytes.
_MIN_VALID_PCM_BYTES: Final[int] = 1000
# Default TTS voice settings; playbooks may override per-stage later.
# Lower stability = more expressive/natural variation in pitch and cadence.
# Lower similarity = less constrained to the voice sample, more human.
# Higher style = more emotionally expressive.
_DEFAULT_STABILITY: Final[float] = 0.4
_DEFAULT_SIMILARITY: Final[float] = 0.7
_DEFAULT_STYLE: Final[float] = 0.5
# Fallback silence duration (ms) when synthesis fails.
_FALLBACK_SILENCE_MS: Final[int] = 500


class SynthesisError(RuntimeError):
    """Raised when ElevenLabs synthesis fails and no fallback is available."""


class VoiceService:
    """Wraps ElevenLabs clone + TTS endpoints behind a typed async API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "eleven_multilingual_v2",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._client = client  # injectable for tests; created lazily otherwise
        self._owns_client = client is None  # we own it only if we create it

    @property
    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(headers=self._headers, timeout=30.0)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we own it.

        If a caller injected a client, they retain ownership and must close
        it themselves.
        """
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def clone(self, sample_path: str, name: str = "ghostline_voice") -> str | None:
        """Upload a voice sample and return the new ElevenLabs voice_id.

        Args:
            sample_path: Path to a WAV or M4A file. M4A is transcoded to WAV
                via pydub/ffmpeg before upload.
            name: Display name for the cloned voice.

        Returns:
            The voice_id string on success, or ``None`` if the API rejected
            the upload.
        """
        wav_bytes = self._read_as_wav(sample_path)
        files = {"files": ("sample.wav", wav_bytes, "audio/wav")}
        data = {"name": name, "description": "GhostLine cloned voice"}
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{_API_BASE}/voices/add",
                files=files,
                data=data,
                headers=self._headers,
            )
        except httpx.HTTPError:
            _LOGGER.exception("ElevenLabs clone network error")
            return None
        if resp.status_code == _HTTP_OK:
            payload = resp.json()
            vid = payload.get("voice_id")
            if isinstance(vid, str):
                _LOGGER.info("Cloned voice: %s (name=%s)", vid, name)
                return vid
            _LOGGER.error("ElevenLabs clone response missing voice_id: %r", payload)
            return None
        _LOGGER.error("ElevenLabs clone failed (%d): %s", resp.status_code, resp.text[:300])
        return None

    @staticmethod
    def _read_as_wav(sample_path: str) -> bytes:
        """Read an audio file and return WAV bytes (transcoding M4A if needed)."""
        path_lower = sample_path.lower()
        if path_lower.endswith(".m4a"):
            seg = AudioSegment.from_file(sample_path, format="m4a")
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            return buf.getvalue()
        with open(sample_path, "rb") as fh:
            return fh.read()

    async def synth(self, text: str, voice_id: str, persona: str = "professional") -> bytes:
        """Synthesize ``text`` to 8 kHz mono 16-bit PCM bytes.

        On API error returns a short silence buffer so the call doesn't
        dead-air. Persona name is currently informational (ElevenLabs doesn't
        accept SSML); future work can map personas to voice_settings.
        """
        del persona  # reserved for future per-persona voice_settings mapping
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": _DEFAULT_STABILITY,
                "similarity_boost": _DEFAULT_SIMILARITY,
                "style": _DEFAULT_STYLE,
                "use_speaker_boost": False,
            },
        }
        url = f"{_API_BASE}/text-to-speech/{voice_id}"
        client = await self._get_client()
        try:
            resp = await client.post(url, json=payload, headers=self._headers)
        except httpx.HTTPError:
            _LOGGER.exception("ElevenLabs TTS network error")
            return self._silence_pcm()
        if resp.status_code != _HTTP_OK:
            _LOGGER.error("ElevenLabs TTS failed (%d): %s", resp.status_code, resp.text[:300])
            return self._silence_pcm()

        mp3 = resp.content
        if len(mp3) < _MIN_VALID_PCM_BYTES:
            _LOGGER.error("ElevenLabs returned very small payload: %d bytes", len(mp3))
            return self._silence_pcm()

        try:
            seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        except Exception:
            _LOGGER.exception("Failed to decode ElevenLabs MP3 (%d bytes)", len(mp3))
            return self._silence_pcm()

        seg = seg.normalize(headroom=0.1)
        seg = seg.set_frame_rate(_TARGET_FRAME_RATE).set_channels(_TARGET_CHANNELS)
        raw: bytes = seg.raw_data
        return raw

    @staticmethod
    def _silence_pcm(duration_ms: int = _FALLBACK_SILENCE_MS) -> bytes:
        """Return a short silence PCM buffer for fallback."""
        seg = AudioSegment.silent(duration=duration_ms)
        seg = seg.set_frame_rate(_TARGET_FRAME_RATE).set_channels(_TARGET_CHANNELS)
        raw: bytes = seg.raw_data
        return raw
