"""FastAPI app factory with WebSocket endpoint for Twilio Media Stream.

Provides /voice, /api/stats, and the HTML dashboard. This module wires
together every other module:

- ``/twilio`` WS: per-call handler running ``pump_in`` + ``pump_out`` +
  ``silence_monitor`` concurrently, bridging Twilio ↔ Deepgram ↔ Agent ↔
  ElevenLabs ↔ Twilio.
- ``/voice`` POST: returns TwiML pointing Twilio at the ngrok WSS URL.
- ``/api/stats`` GET: JSON of stage counts + call count.
- ``/`` GET: HTML dashboard.

The handler is intentionally thin: protocol details live in
:mod:`ghostline.telephony.media_stream`, agent logic in
:mod:`ghostline.agent.axel`, stage transitions in
:mod:`ghostline.playbook_runner`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from ghostline.agent.capabilities import ModelCapabilities
from ghostline.agent.reply_generator import ReplyGenerator
from ghostline.audio.ambient import AmbientNoise
from ghostline.persistence.repository import CallRecord, MessageRecord, Repository
from ghostline.playbook import Playbook
from ghostline.playbook_runner import PlaybookRunner, StageExit
from ghostline.server.dashboard import render_dashboard_from_repo
from ghostline.stt.deepgram import DeepgramClient, UtteranceEndEvent
from ghostline.taxonomy import STAGE_CHECKINS, STAGE_TIMINGS, SalesStage
from ghostline.telephony.media_stream import (
    MediaFrame,
    StartEvent,
    parse_event,
    send_audio_chunk,
)
from ghostline.telephony.twilio import build_voice_twiml
from ghostline.tts.elevenlabs import VoiceService

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)

# Minimum word count for an utterance to be considered "complete" enough to process.
# 1 = process single-word responses ("Yes", "Okay") immediately on speech_final
# instead of waiting 1000ms for utterance_end.
_MIN_COMPLETE_WORDS: int = 1

__all__ = ("CallSession", "ServerDeps", "create_app")


@dataclass(slots=True)
class ServerDeps:
    """All dependencies the FastAPI app needs, injected by the composition root.

    Holding them in a dataclass (rather than module-level globals) makes the
    app testable: tests construct a ServerDeps with mocks and pass to
    :func:`create_app`.
    """

    repository: Repository
    voice_service: VoiceService
    ambient: AmbientNoise
    deepgram_api_key: str
    deepgram_language: str = "en-US"
    deepgram_model: str = "nova-2"
    playbook: Playbook | None = None
    voice_id: str = ""
    ngrok_ws_url: str | None = None
    reply_generator: ReplyGenerator | None = None
    capabilities: ModelCapabilities | None = None
    target_name: str | None = None


class CallSession:
    """Per-call state. Created when a ``start`` event arrives, torn down on ``stop``."""

    def __init__(
        self,
        deps: ServerDeps,
        ws: WebSocket,
        voice_id: str,
        campaign: str,
        persona: str,
        *,
        target_name: str | None = None,
    ) -> None:
        self._deps = deps
        self._ws = ws
        self.voice_id = voice_id
        self.campaign = campaign
        self.persona = persona
        self.target_name = target_name
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.caller_number: str | None = None
        self.current_stage: SalesStage = SalesStage.RAPPORT
        self.last_activity: float = asyncio.get_event_loop().time()
        self.check_in_sent: bool = False
        self.deepgram: DeepgramClient | None = None
        self.runner: PlaybookRunner | None = None
        self._call_context = None
        self._speaking = False  # True while the agent is sending TTS audio
        self._utterance_lock = asyncio.Lock()  # serializes utterance processing
        if deps.playbook is not None:
            self.runner = PlaybookRunner(deps.playbook)
            self.current_stage = self.runner.current_stage
        # Create a fresh per-call context for the reply generator.
        if deps.reply_generator is not None:
            self._call_context = deps.reply_generator.new_context()
        # Log the active voice modality so operators can see which pipeline
        # is running (text vs realtime). The capabilities are detected once
        # at startup by the composition root.
        caps = deps.capabilities
        if caps is not None:
            _LOGGER.info(
                "CallSession voice modality: %s (native_stt=%s, native_tts=%s, "
                "needs_deepgram=%s, needs_elevenlabs=%s)",
                caps.modality.value,
                caps.native_stt,
                caps.native_tts,
                caps.needs_deepgram,
                caps.needs_elevenlabs,
            )

    async def run(self) -> None:
        """Main per-call coroutine: accept WS, run pump_in + pump_out + silence_monitor."""
        await self._ws.accept()
        try:
            await self._wait_for_start()
            await self._open_deepgram()
            await self._insert_call_record()
            # pump_in is the primary task; when it returns (stop event or
            # disconnect), we cancel the others.
            pump_in = asyncio.create_task(self._pump_in())
            pump_out = asyncio.create_task(self._pump_out())
            silence = asyncio.create_task(self._silence_monitor())
            greeting = asyncio.create_task(self._send_greeting())
            try:
                await pump_in
            finally:
                for t in (pump_out, silence, greeting):
                    t.cancel()
                for t in (pump_out, silence, greeting):
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t
        except WebSocketDisconnect:
            _LOGGER.info("WebSocket disconnected for call %s", self.call_sid)
        except Exception:
            _LOGGER.exception("Call session crashed for call %s", self.call_sid)
        finally:
            await self._teardown()

    async def _wait_for_start(self) -> None:
        """Read frames until we get the ``start`` event with streamSid + callSid."""
        while True:
            raw = await self._ws.receive_text()
            event = parse_event(raw)
            if isinstance(event, StartEvent):
                self.stream_sid = event.stream_sid
                self.call_sid = event.call_sid
                self.caller_number = event.caller_number
                _LOGGER.info(
                    "Call started: sid=%s stream=%s caller=%s",
                    self.call_sid,
                    self.stream_sid,
                    self.caller_number,
                )
                return

    async def _open_deepgram(self) -> None:
        """Open a Deepgram WSS for this call — unless the model handles STT natively.

        If the active LLM model supports native audio input (e.g. a realtime
        model), Deepgram is redundant and we skip it. The capability check
        happens once at startup; here we just read the result.
        """
        caps = self._deps.capabilities
        if caps is not None and not caps.needs_deepgram:
            _LOGGER.info(
                "Skipping Deepgram — model %s handles STT natively (modality=%s)",
                caps.model,
                caps.modality.value,
            )
            return
        self.deepgram = DeepgramClient(
            self._deps.deepgram_api_key,
            language=self._deps.deepgram_language,
            model=self._deps.deepgram_model,
        )
        await self.deepgram.connect()

    async def _insert_call_record(self) -> None:
        """Persist the call row + caller profile."""
        if self.call_sid is None:
            return
        await self._deps.repository.insert_call(
            CallRecord(
                call_sid=self.call_sid,
                voice_id=self.voice_id,
                campaign=self.campaign,
                persona=self.persona,
                phone_number=self.caller_number,
            )
        )
        if self.caller_number:
            await self._deps.repository.upsert_profile(self.caller_number, self.persona)

    async def _pump_in(self) -> None:
        """Read Twilio media frames → forward µ-law to Deepgram."""
        if self.deepgram is None:
            return
        try:
            while True:
                raw = await self._ws.receive_text()
                event = parse_event(raw)
                if isinstance(event, MediaFrame):
                    await self.deepgram.send(event.payload)
                    self.last_activity = asyncio.get_event_loop().time()
                    self.check_in_sent = False
                elif isinstance(event, str) and event == "stop":
                    _LOGGER.info("Stop event for call %s", self.call_sid)
                    return
        except WebSocketDisconnect:
            return

    async def _pump_out(self) -> None:
        """Read Deepgram events → generate agent reply → synth → send to Twilio.

        Utterances are NOT dropped while the agent is speaking — instead
        they wait for the utterance lock in _process_utterance so they
        queue up and get processed after the current reply finishes.
        """
        if self.deepgram is None:
            return
        buffer = ""
        waiting_for_utterance_end = False
        async for event in self.deepgram.events():
            if isinstance(event, UtteranceEndEvent):
                if waiting_for_utterance_end and buffer.strip():
                    await self._process_utterance(buffer.strip())
                    buffer = ""
                waiting_for_utterance_end = False
            elif event.transcript:
                if buffer and not buffer.endswith(" "):
                    buffer += " "
                buffer += event.transcript
                self.last_activity = asyncio.get_event_loop().time()
                self.check_in_sent = False
                if event.speech_final:
                    cleaned = buffer.strip()
                    words = cleaned.split()
                    is_complete = len(words) >= _MIN_COMPLETE_WORDS or any(
                        cleaned.endswith(p) for p in (".", "?", "!")
                    )
                    if cleaned and is_complete:
                        await self._process_utterance(cleaned)
                        buffer = ""
                        waiting_for_utterance_end = False
                    else:
                        waiting_for_utterance_end = True
                else:
                    waiting_for_utterance_end = True

    async def _silence_monitor(self) -> None:
        """Send stage check-in lines when the target goes quiet.

        Acquires the utterance lock so check-in audio can't overlap
        with reply audio. Skips if the agent is already speaking.
        """
        while True:
            await asyncio.sleep(1)
            if self.stream_sid is None:
                continue
            if self._speaking:
                continue
            silent = asyncio.get_event_loop().time() - self.last_activity
            threshold = self._stage_silence_threshold()
            if silent >= threshold and not self.check_in_sent:
                _LOGGER.info(
                    "Silence monitor: %.1fs >= %ds threshold, sending check-in for stage %s",
                    silent,
                    threshold,
                    self.current_stage.name,
                )
                async with self._utterance_lock:
                    self._speaking = True
                    try:
                        await self._speak(STAGE_CHECKINS[self.current_stage])
                    finally:
                        self._speaking = False
                        self.last_activity = asyncio.get_event_loop().time()
                self.check_in_sent = True

    async def _send_greeting(self) -> None:
        """Speak the first stage's custom_prompt (or a default) once stream is live.

        Acquires the utterance lock so the greeting can't overlap with
        a reply (if the target speaks during the greeting, the reply
        waits until the greeting finishes playing). Also stores the
        greeting in the DB so the conversation history includes it.
        """
        for _ in range(20):
            if self.stream_sid is not None:
                break
            await asyncio.sleep(0.1)
        if self.stream_sid is None:
            _LOGGER.warning("No stream_sid; skipping greeting for %s", self.call_sid)
            return
        greeting = self._stage_greeting()
        if self.call_sid is not None:
            await self._deps.repository.insert_message(
                MessageRecord(
                    call_sid=self.call_sid,
                    role="assistant",
                    content=greeting,
                    sales_stage=self.current_stage.name,
                )
            )
        async with self._utterance_lock:
            self._speaking = True
            try:
                await self._speak(greeting)
            finally:
                self._speaking = False
                self.last_activity = asyncio.get_event_loop().time()

    def _stage_greeting(self) -> str:
        """Return the greeting text for the current (first) stage.

        Substitutes ``{name}`` with the target's name if provided via
        the ``--target-name`` CLI arg (simulates OSINT-informed pretext).
        """
        text = "Hello! Thanks for taking my call today. How are you doing?"
        if self._deps.playbook is not None:
            config = self._deps.playbook.stage_for(self.current_stage)
            if config is not None and config.custom_prompt:
                text = config.custom_prompt.strip()
        if self.target_name:
            text = text.replace("{name}", self.target_name)
        return text

    def _stage_silence_threshold(self) -> int:
        """Return the silence threshold for the current stage."""
        if self._deps.playbook is not None:
            return self._deps.playbook.effective_silent_until(self.current_stage)
        return STAGE_TIMINGS[self.current_stage]

    async def _process_utterance(self, utterance: str) -> None:
        """Handle a complete user utterance: log, evaluate, reply, speak.

        Uses an asyncio.Lock to serialize utterance processing — if the
        agent is already generating a reply + speaking, the next utterance
        waits until the lock is released. This prevents the agent from
        being interrupted mid-speech and prevents overlapping TTS.
        """
        if self.call_sid is None:
            return
        async with self._utterance_lock:
            self._speaking = True
            try:
                _LOGGER.info("Processing utterance for %s: '%s'", self.call_sid, utterance)

                await self._deps.repository.insert_message(
                    MessageRecord(
                        call_sid=self.call_sid,
                        role="user",
                        content=utterance,
                        sales_stage=self.current_stage.name,
                    )
                )

                if self.runner is not None:
                    outcome = self.runner.evaluate(utterance)
                    self.current_stage = outcome.next_stage
                    if outcome.exit_ is not None:
                        await self._handle_exit(outcome.exit_)
                        return

                history = await self._build_history_prompt()
                reply = await self._generate_reply(utterance, history)

                await self._deps.repository.insert_message(
                    MessageRecord(
                        call_sid=self.call_sid,
                        role="assistant",
                        content=reply,
                        sales_stage=self.current_stage.name,
                    )
                )

                await self._speak(reply)
                # Brief cooldown after speaking — Twilio buffers audio so
                # the target is still hearing the reply for a moment after
                # we finish sending frames. Without this, queued utterances
                # (transcribed during playback) fire instantly and overlap.
                await asyncio.sleep(0.8)
            finally:
                self._speaking = False
                self.last_activity = asyncio.get_event_loop().time()

    async def _build_history_prompt(self) -> str:
        """Build a conversation history string from recent DB messages.

        Limited to the last 4 messages to keep the LLM prompt small
        (faster response through OpenRouter).
        """
        if self.call_sid is None:
            return ""
        messages = await self._deps.repository.recent_messages(self.call_sid, limit=4)
        if not messages:
            return ""
        lines = []
        if self.target_name:
            lines.append(
                f"Target's name is {self.target_name}. Do NOT overuse it — once per reply at most, and skip it entirely most of the time."
            )
        for msg in messages:
            speaker = "Target" if msg.role == "user" else "Agent"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)

    async def _generate_reply(self, utterance: str, history: str) -> str:
        """Generate a reply via the ReplyGenerator, with fallback."""
        gen = self._deps.reply_generator
        if gen is None:
            _LOGGER.warning("No reply generator configured; using fallback")
            return "I understand. Could you tell me more about that?"
        try:
            result = await gen.generate(
                utterance=utterance,
                current_stage=self.current_stage,
                context=self._call_context,
                history=history or None,
            )
            _LOGGER.info(
                "Reply generated for %s: '%s' (trigger=%s, sentiment=%s)",
                self.call_sid,
                result.reply,
                result.trigger,
                result.analysis.sentiment_score if result.analysis else None,
            )
            return result.reply
        except Exception:
            _LOGGER.exception("Reply generation failed; using fallback")
            return "I understand. Could you tell me more about that?"

    async def _handle_exit(self, exit_: StageExit) -> None:
        """Record final outcome and stop speaking."""
        if self.call_sid is not None:
            await self._deps.repository.update_call_outcome(
                self.call_sid,
                exit_.outcome,
                conversion_score=1.0 if exit_.outcome == "compromised" else 0.0,
            )
        _LOGGER.info("Call %s exited: %s (%s)", self.call_sid, exit_.outcome, exit_.reason)

    async def _speak(self, text: str) -> None:
        """Synthesize ``text`` and stream the audio to Twilio."""
        if self.stream_sid is None:
            return
        pcm = await self._deps.voice_service.synth(text, self.voice_id, self.persona)
        if not pcm or all(b == 0 for b in pcm[:100]):
            _LOGGER.warning("TTS returned silence/fallback for: %r", text[:60])
        ratio = self._stage_ambient_ratio()
        mixed = self._deps.ambient.mix(pcm, level=ratio)
        ulaw = self._deps.ambient.ulaw(mixed)
        n_frames = await send_audio_chunk(self._ws, self.stream_sid, ulaw)
        _LOGGER.info(
            "TTS sent: %d bytes PCM, %d frames ulaw for: %r", len(pcm), n_frames, text[:60]
        )

    def _stage_ambient_ratio(self) -> float:
        """Ambient mix weight for the current stage."""
        if self._deps.playbook is not None:
            return self._deps.playbook.effective_ambient_ratio(self.current_stage)
        return 0.1

    async def _teardown(self) -> None:
        """Close Deepgram, the WS, and record final state."""
        if self.deepgram is not None:
            try:
                await self.deepgram.close()
            except Exception:
                _LOGGER.exception("Error closing Deepgram for call %s", self.call_sid)
        with contextlib.suppress(Exception):
            await self._ws.close()


@asynccontextmanager
async def _lifespan(deps: ServerDeps) -> Any:
    """FastAPI lifespan: ensure the repository is open for the app's lifetime."""
    yield
    await deps.repository.close()


def create_app(deps: ServerDeps) -> FastAPI:
    """Build the FastAPI app with all routes wired to ``deps``.

    Usage::

        deps = ServerDeps(repository=..., voice_service=..., ...)
        app = create_app(deps)
        uvicorn.run(app, host=..., port=...)
    """
    app = FastAPI(title="GhostLine", version="0.5.0", lifespan=lambda _: _lifespan(deps))

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        """Render the HTML dashboard with live stats."""
        pb_name = deps.playbook.meta.name if deps.playbook else None
        return await render_dashboard_from_repo(deps.repository, playbook_name=pb_name)

    @app.get("/api/stats")
    async def stats() -> JSONResponse:
        """Return stage counts + call count as JSON."""
        return JSONResponse(
            {
                "call_count": await deps.repository.call_count(),
                "stage_counts": await deps.repository.stage_counts(),
                "playbook": deps.playbook.meta.name if deps.playbook else None,
            }
        )

    @app.post("/voice", response_class=PlainTextResponse)
    async def voice(request: Request) -> str:
        """Return TwiML pointing Twilio at the ngrok WSS URL.

        Accepts optional ``campaign`` and ``persona`` query params, which
        are appended to the WSS URL so the WS handler can read them.
        """
        if not deps.ngrok_ws_url:
            raise HTTPException(status_code=500, detail="Tunnel not initialized")
        ws_url = deps.ngrok_ws_url
        campaign = request.query_params.get("campaign")
        persona = request.query_params.get("persona")
        target_name = request.query_params.get("target_name")
        # Store target_name on deps so the WS handler can read it
        # (Twilio's <Stream> does NOT forward query params to the WSS connection).
        if target_name:
            deps.target_name = target_name
        params: list[str] = []
        if campaign:
            params.append(f"campaign={campaign}")
        if persona:
            params.append(f"persona={persona}")
        if params:
            ws_url += "?" + "&".join(params)
        return build_voice_twiml(ws_url)

    @app.websocket("/twilio")
    async def twilio_ws(ws: WebSocket) -> None:
        """Per-call WebSocket: pump Twilio media frames through the full pipeline."""
        campaign = ws.query_params.get("campaign", "general")
        persona = ws.query_params.get("persona", "professional")
        session = CallSession(
            deps, ws, deps.voice_id, campaign, persona, target_name=deps.target_name
        )
        await session.run()

    return app
