"""Typed async Deepgram streaming client.

Maintains a long-lived WSS connection to Deepgram ``/v1/listen`` and exposes
a typed queue of transcript / utterance-end events. Implements:

- Heartbeat pings every ``heartbeat_interval`` seconds
- Auto-reconnect on socket close / error
- Internal ``asyncio.Queue`` decoupling receive loop from consumers

The legacy ``deepgram.py`` shadowed the PyPI ``deepgram`` SDK and had no
type hints; this module is the canonical replacement.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

import aiohttp

__all__ = (
    "DeepgramClient",
    "DeepgramEvent",
    "TranscriptEvent",
    "UtteranceEndEvent",
)

_LOGGER = logging.getLogger(__name__)

# Deepgram endpointing: 300ms of silence ends an utterance candidate.
# Lower = faster turn detection but more false positives.
_DEFAULT_ENDPOINTING_MS: Final[str] = "300"
# Utterance-end window: minimum 1000ms (Deepgram API requirement).
_DEFAULT_UTTERANCE_END_MS: Final[str] = "1000"
# Heartbeat cadence.
_DEFAULT_HEARTBEAT_S: Final[float] = 30.0
# Backoff between reconnect attempts.
_RECONNECT_BACKOFF_S: Final[float] = 1.0


@dataclass(slots=True, frozen=True)
class TranscriptEvent:
    """Final or interim transcript for a single Deepgram result."""

    transcript: str
    confidence: float
    speech_final: bool
    is_final: bool
    words: list[dict[str, object]]


@dataclass(slots=True, frozen=True)
class UtteranceEndEvent:
    """Deepgram signalled the end of an utterance."""

    last_word_end: float


DeepgramEvent = TranscriptEvent | UtteranceEndEvent


class DeepgramClient:
    """Async Deepgram streaming client with heartbeat and auto-reconnect.

    Lifecycle::

        client = DeepgramClient(api_key, encoding="mulaw", sample_rate=8000)
        await client.connect()
        try:
            await client.send(frame_bytes)
            async for event in client.events():
                ...
        finally:
            await client.close()
    """

    def __init__(
        self,
        api_key: str,
        *,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        channels: int = 1,
        language: str = "en-US",
        model: str = "nova-2",
        punctuate: bool = True,
        endpointing_ms: str = _DEFAULT_ENDPOINTING_MS,
        utterance_end_ms: str = _DEFAULT_UTTERANCE_END_MS,
        interim_results: bool = True,
        smart_format: bool = True,
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_S,
    ) -> None:
        self._api_key = api_key
        self._heartbeat_interval = heartbeat_interval
        self._ws_url = self._build_url(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            language=language,
            model=model,
            punctuate=punctuate,
            endpointing_ms=endpointing_ms,
            utterance_end_ms=utterance_end_ms,
            interim_results=interim_results,
            smart_format=smart_format,
        )
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[DeepgramEvent] = asyncio.Queue()
        self._closed = False

    @staticmethod
    def _build_url(
        *,
        encoding: str,
        sample_rate: int,
        channels: int,
        language: str,
        model: str,
        punctuate: bool,
        endpointing_ms: str,
        utterance_end_ms: str,
        interim_results: bool,
        smart_format: bool,
    ) -> str:
        params = {
            "encoding": encoding,
            "sample_rate": sample_rate,
            "channels": channels,
            "language": language,
            "model": model,
            "punctuate": str(punctuate).lower(),
            "endpointing": endpointing_ms,
            "utterance_end_ms": utterance_end_ms,
            "interim_results": str(interim_results).lower(),
            "smart_format": str(smart_format).lower(),
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        return f"wss://api.deepgram.com/v1/listen?{qs}"

    @property
    def url(self) -> str:
        """The WSS URL this client connects to (for logging/diagnostics)."""
        return self._ws_url

    async def connect(self) -> None:
        """Open the session + WSS and start the heartbeat and receiver tasks."""
        if self._closed:
            raise RuntimeError("DeepgramClient has been closed; create a new instance")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self._ws_url,
                headers={"Authorization": f"Token {self._api_key}"},
            )
        except Exception:
            _LOGGER.exception("Failed to connect to Deepgram WSS")
            raise
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
        if self._receive_task is None or self._receive_task.done():
            self._receive_task = asyncio.create_task(self._receiver_loop())
        _LOGGER.info("Deepgram WSS connected to %s", self._ws_url)

    async def _ensure_ws(self) -> None:
        if self._ws is None or self._ws.closed:
            _LOGGER.warning("Deepgram WSS closed; reconnecting")
            await self.connect()

    async def _heartbeat(self) -> None:
        try:
            while self._ws is not None and not self._ws.closed:
                await asyncio.sleep(self._heartbeat_interval)
                if self._ws is not None and not self._ws.closed:
                    await self._ws.ping()
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.warning("Deepgram heartbeat error", exc_info=True)

    async def send(self, chunk: bytes) -> None:
        """Send a raw audio frame (µ-law bytes) to Deepgram."""
        await self._ensure_ws()
        assert self._ws is not None
        for attempt in range(2):
            try:
                await self._ws.send_bytes(chunk)
                return
            except (ConnectionResetError, aiohttp.ClientConnectionError):
                if attempt == 0:
                    _LOGGER.warning("Deepgram send failed; reconnecting")
                    await self.connect()
                    continue
                raise

    async def send_keepalive(self) -> None:
        """Send a JSON ``KeepAlive`` message to extend the stream."""
        await self._ensure_ws()
        assert self._ws is not None
        await self._ws.send_str(json.dumps({"type": "KeepAlive"}))

    async def events(self) -> AsyncIterator[DeepgramEvent]:
        """Yield transcript / utterance-end events from the internal queue."""
        while True:
            event = await self._queue.get()
            yield event

    async def _receiver_loop(self) -> None:
        while not self._closed:
            try:
                await self._ensure_ws()
                assert self._ws is not None
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_text(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    _LOGGER.warning("Deepgram WSS closed/error; reconnecting")
                    await asyncio.sleep(_RECONNECT_BACKOFF_S)
                    await self.connect()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("Deepgram receiver loop error")
                await asyncio.sleep(_RECONNECT_BACKOFF_S)

    def _handle_text(self, data: str) -> None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            _LOGGER.warning("Deepgram sent non-JSON frame: %r", data[:200])
            return
        msg_type: str | None = payload.get("type")
        if msg_type == "UtteranceEnd":
            event = UtteranceEndEvent(last_word_end=float(payload.get("last_word_end", 0.0)))
            self._queue.put_nowait(event)
        elif msg_type == "Results" and payload.get("is_final"):
            self._handle_results(payload)
        # Other message types (e.g. Metadata) are intentionally ignored.

    def _handle_results(self, payload: dict[str, object]) -> None:
        channel = payload.get("channel")
        if not isinstance(channel, dict):
            return
        alternatives = channel.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            return
        alt = alternatives[0]
        if not isinstance(alt, dict):
            return
        transcript = str(alt.get("transcript", "")).strip()
        if not transcript:
            return
        words_raw = alt.get("words", [])
        words = words_raw if isinstance(words_raw, list) else []
        event = TranscriptEvent(
            transcript=transcript,
            confidence=float(alt.get("confidence", 1.0)),
            speech_final=bool(payload.get("speech_final", False)),
            is_final=bool(payload.get("is_final", False)),
            words=words,
        )
        self._queue.put_nowait(event)

    async def close(self) -> None:
        """Cancel background tasks, send CloseStream, and tear down the session."""
        self._closed = True
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
        if self._ws is not None and not self._ws.closed:
            with contextlib.suppress(Exception):
                await self._ws.send_str(json.dumps({"type": "CloseStream"}))
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        _LOGGER.info("Deepgram client closed")
