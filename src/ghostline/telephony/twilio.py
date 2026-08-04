"""Twilio outbound call placement and TwiML builders.

Pure functions / thin wrappers around the synchronous ``twilio.rest.Client``.
The CLI calls these from a sync entrypoint; the FastAPI server never touches
them in the hot path (it only consumes Media Stream frames via the WS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from xml.sax import saxutils

from twilio.rest import Client as TwilioClient

__all__ = (
    "CallPlaced",
    "TwilioService",
    "build_stream_twiml",
    "build_voice_twiml",
)

# Duration (seconds) the TwiML <Pause> keeps the call alive after the
# <Stream> ends. 600s = 10 minutes, enough for a full social-engineering call.
_PAUSE_SECONDS: Final[int] = 600
# Voice used by the <Say> preamble before <Connect><Stream>.
_SAY_VOICE: Final[str] = "alice"
_SAY_TEXT: Final[str] = ""


@dataclass(slots=True, frozen=True)
class CallPlaced:
    """Result of a successful outbound call placement."""

    sid: str
    to_number: str
    from_number: str


class TwilioService:
    """Wraps the synchronous Twilio REST client for outbound calls."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        self._client = TwilioClient(account_sid, auth_token)
        self._from_number = from_number

    def place_call(self, to_number: str, stream_ws_url: str) -> CallPlaced:
        """Place an outbound call and connect it to a Media Stream WSS URL.

        Args:
            to_number: E.164 destination phone number.
            stream_ws_url: ``wss://`` URL for Twilio's <Stream> to dial back
                into. Must already be XML-escaped-safe (we escape here).

        Returns:
            A :class:`CallPlaced` with the new call SID.
        """
        twiml = build_stream_twiml(stream_ws_url)
        call = self._client.calls.create(
            to=to_number,
            from_=self._from_number,
            twiml=twiml,
        )
        return CallPlaced(sid=call.sid, to_number=to_number, from_number=self._from_number)


def build_stream_twiml(stream_ws_url: str) -> str:
    """Build TwiML that connects the call to a Media Stream WebSocket.

    The URL is XML-escaped so query parameters with ``&`` are safe inside
    the ``<Stream url="...">`` attribute.
    """
    safe_url = saxutils.escape(stream_ws_url)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<Response>",
    ]
    if _SAY_TEXT:
        parts.append(f'  <Say voice="{_SAY_VOICE}">{_SAY_TEXT}</Say>')
    parts += [
        "  <Connect>",
        f'    <Stream url="{safe_url}"/>',
        "  </Connect>",
        f'  <Pause length="{_PAUSE_SECONDS}"/>',
        "</Response>",
    ]
    return "\n".join(parts)


def build_voice_twiml(stream_ws_url: str) -> str:
    """Alias for :func:`build_stream_twiml` used by the inbound /voice route."""
    return build_stream_twiml(stream_ws_url)
