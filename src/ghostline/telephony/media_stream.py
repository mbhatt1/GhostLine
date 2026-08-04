"""Twilio Media Stream WebSocket protocol.

Parses Twilio's JSON frame protocol on the inbound side and produces
properly-formed ``media`` frames on the outbound side. Keeps the WS handler
in ``server/fastapi.py`` thin: protocol details live here.

Inbound events handled:
  - ``connected``: informational
  - ``start``: carries ``streamSid`` + ``callSid`` + custom params
  - ``media``: base64-encoded µ-law audio from the callee
  - ``stop``: end-of-stream

Outbound:
  - ``media``: base64-encoded µ-law audio chunk with ``streamSid``
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Final, Protocol

__all__ = (
    "MediaFrame",
    "MediaSink",
    "StartEvent",
    "TwilioEvent",
    "encode_media_frame",
    "parse_event",
    "send_audio_chunk",
)

_LOGGER = logging.getLogger(__name__)

# Standard Twilio Media Streams chunk size: 8 kHz µ-law * 20ms = 160 bytes.
CHUNK_SIZE: Final[int] = 160

# Real-time pacing: 160 bytes at 8 kHz = 20ms per frame.
# Sending all frames in a burst causes Twilio's playback buffer to
# desync from the agent's "speaking" state, leading to the silence
# monitor firing mid-playback and interrupting the agent's audio.
_FRAME_DURATION_SEC: Final[float] = 0.02


@dataclass(slots=True, frozen=True)
class StartEvent:
    """Parsed ``start`` event from Twilio Media Stream."""

    stream_sid: str
    call_sid: str
    caller_number: str | None
    custom_params: dict[str, str]


@dataclass(slots=True, frozen=True)
class MediaFrame:
    """A decoded inbound media frame (raw µ-law bytes)."""

    payload: bytes
    track: str | None = None


TwilioEvent = StartEvent | MediaFrame | str
"""Either a StartEvent, a MediaFrame, or a bare event-name string
(``'connected'`` / ``'stop'``)."""


class MediaSink(Protocol):
    """Anything that accepts outbound JSON strings (a FastAPI WebSocket)."""

    async def send_text(self, data: str) -> None:
        """Send a raw text frame to the underlying WebSocket."""


def parse_event(raw: str) -> TwilioEvent:
    """Parse a raw JSON Twilio Media Stream frame into a typed value.

    Returns:
        - :class:`StartEvent` for ``event=start``
        - :class:`MediaFrame` for ``event=media`` (base64-decoded payload)
        - ``str`` for ``connected`` / ``stop`` / unknown event names

    Raises:
        :class:`json.JSONDecodeError` if the frame is not valid JSON.
    """
    payload = json.loads(raw)
    event = str(payload.get("event", ""))
    if event == "start":
        start = payload.get("start", {})
        if not isinstance(start, dict):
            start = {}
        custom_params_raw = start.get("customParameters", {})
        if not isinstance(custom_params_raw, dict):
            custom_params_raw = {}
        custom_params = {str(k): str(v) for k, v in custom_params_raw.items()}
        caller = custom_params.get("from") or (
            start.get("from") if isinstance(start.get("from"), str) else None
        )
        return StartEvent(
            stream_sid=str(start.get("streamSid", "")),
            call_sid=str(start.get("callSid", "")),
            caller_number=caller,
            custom_params=custom_params,
        )
    if event == "media":
        media = payload.get("media", {})
        if not isinstance(media, dict):
            media = {}
        b64 = str(media.get("payload", ""))
        track = media.get("track")
        if not isinstance(track, str):
            track = None
        return MediaFrame(payload=base64.b64decode(b64), track=track)
    # connected / stop / unknown
    return event


def encode_media_frame(stream_sid: str, ulaw_chunk: bytes) -> str:
    """Build a Twilio Media Stream ``media`` JSON frame from µ-law bytes.

    Args:
        stream_sid: The streamSid from the ``start`` event.
        ulaw_chunk: Raw µ-law audio bytes (typically CHUNK_SIZE long).

    Returns:
        A JSON string ready to ``send_text`` to the WebSocket.
    """
    payload_b64 = base64.b64encode(ulaw_chunk).decode("ascii")
    return json.dumps(
        {"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}}
    )


async def send_audio_chunk(
    ws: MediaSink, stream_sid: str, ulaw: bytes, *, chunk_size: int = CHUNK_SIZE
) -> int:
    """Send ``ulaw`` audio to Twilio in CHUNK_SIZE-byte frames at real-time pace.

    Each 160-byte frame is 20ms of audio at 8 kHz. We sleep 20ms between
    frames so the agent is "speaking" for the full duration of the audio.
    This prevents the silence monitor from firing while Twilio is still
    playing buffered audio, which was the root cause of the agent's
    voice cutting out mid-sentence.

    Returns the number of frames sent.
    """
    frames_sent = 0
    for offset in range(0, len(ulaw), chunk_size):
        chunk = ulaw[offset : offset + chunk_size]
        frame = encode_media_frame(stream_sid, chunk)
        await ws.send_text(frame)
        frames_sent += 1
        await asyncio.sleep(_FRAME_DURATION_SEC)
    return frames_sent
