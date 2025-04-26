#!/usr/bin/env python3
"""
Enhanced DeepgramClient with proper speech_final and utterance_end support.
Includes automatic reconnect, heartbeat pings, and clean shutdown.
"""
import contextlib
import aiohttp
import asyncio
import logging
import json
from typing import Optional, Dict, Any


class DeepgramClient:
    def __init__(
        self,
        api_key: str,
        ssl_context=None,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        channels: int = 1,
        language: str = "en-US",
        model: str = "nova-2",  # Updated to the latest model
        punctuate: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        self.api_key = api_key
        self.ssl = ssl_context
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.channels = channels
        self.language = language
        self.model = model
        self.punctuate = punctuate
        self.heartbeat_interval = heartbeat_interval

        params = {
            "encoding": encoding,
            "sample_rate": sample_rate,
            "channels": channels,
            "language": language,
            "model": model,
            "punctuate": str(punctuate).lower(),
            "endpointing": "500",  # 500ms for endpointing
            "utterance_end_ms": "1000",  # 1 second for utterance end
            "interim_results": "true",  # Enable interim results for utterance_end to work
            "smart_format": "true"  # Format numbers, emails, etc.
        }
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        self.ws_url = f"wss://api.deepgram.com/v1/listen?{qs}"

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._transcript_queue = asyncio.Queue()

        logging.debug(f"[DeepgramClient] Initialized with URL: {self.ws_url}")

    async def connect(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=self.ssl)
            )
        try:
            self._ws = await self._session.ws_connect(
                self.ws_url,
                headers={"Authorization": f"Token {self.api_key}"}
            )
            logging.info("[DeepgramClient] WebSocket connected.")
        except Exception as e:
            logging.error(f"[DeepgramClient] WS connect error: {e}")
            raise

        # Start heartbeat pings
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat())

        # Start single receiver task
        if self._receive_task is None or self._receive_task.done():
            self._receive_task = asyncio.create_task(self._receiver_loop())

    async def _ensure_ws(self):
        if not self._ws or self._ws.closed:
            logging.warning("[DeepgramClient] WS closed, reconnecting...")
            await self.connect()

    async def _heartbeat(self):
        try:
            while self._ws and not self._ws.closed:
                await asyncio.sleep(self.heartbeat_interval)
                await self._ws.ping()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.warning(f"[DeepgramClient] Heartbeat error: {e}")

    async def send(self, chunk: bytes):
        await self._ensure_ws()
        for attempt in range(2):
            try:
                await self._ws.send_bytes(chunk)
                return
            except (ConnectionResetError, aiohttp.ClientConnectionError) as e:
                if attempt == 0:
                    logging.warning(f"[DeepgramClient] send error, reconnecting: {e}")
                    await self.connect()
                    continue
                else:
                    raise
            except Exception as e:
                logging.error(f"[DeepgramClient] Unexpected send error: {e}")
                raise

    async def _receiver_loop(self):
        while True:
            try:
                await self._ensure_ws()
                msg = await self._ws.receive()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Handle UtteranceEnd event
                    if data.get("type") == "UtteranceEnd":
                        utterance_end_event = {
                            "type": "utterance_end",
                            "channel": data.get("channel", [0, 1]),
                            "last_word_end": data.get("last_word_end")
                        }
                        await self._transcript_queue.put(utterance_end_event)
                        logging.debug(f"[DeepgramClient] Received UtteranceEnd: {utterance_end_event}")
                    
                    # Handle Results event (transcripts)
                    elif data.get("type") == "Results" and data.get("is_final"):
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            alternative = alternatives[0]
                            transcript = alternative.get("transcript", "").strip()
                            if transcript:
                                transcript_event = {
                                    "type": "transcript",
                                    "transcript": transcript,
                                    "confidence": alternative.get("confidence", 1.0),
                                    "words": alternative.get("words", []),
                                    "speech_final": data.get("speech_final", False),
                                    "is_final": data.get("is_final", False)
                                }
                                await self._transcript_queue.put(transcript_event)
                                logging.debug(f"[DeepgramClient] Received transcript: {transcript} (speech_final: {transcript_event['speech_final']})")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logging.warning("[DeepgramClient] WebSocket closed or error, reconnecting...")
                    await asyncio.sleep(1)
                    await self.connect()

            except Exception as e:
                logging.error(f"[DeepgramClient] Error in receiver_loop: {e}")
                await asyncio.sleep(1)

    async def receive_final(self):
        """Yields transcript events from the queue"""
        while True:
            event = await self._transcript_queue.get()
            yield event

    async def send_keepalive(self):
        """Send a KeepAlive message to Deepgram"""
        try:
            keepalive_msg = json.dumps({"type": "KeepAlive"})
            await self._ws.send_str(keepalive_msg)
            logging.debug("[DeepgramClient] Sent KeepAlive message")
        except Exception as e:
            logging.error(f"[DeepgramClient] Error sending KeepAlive: {e}")

    async def close(self):
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task

        if self._ws and not self._ws.closed:
            # Send CloseStream message before closing
            try:
                close_msg = json.dumps({"type": "CloseStream"})
                await self._ws.send_str(close_msg)
                logging.debug("[DeepgramClient] Sent CloseStream message")
            except:
                pass
            
            await self._ws.close()
            logging.info("[DeepgramClient] WebSocket closed.")

        if self._session and not self._session.closed:
            await self._session.close()
            logging.info("[DeepgramClient] HTTP session closed.")