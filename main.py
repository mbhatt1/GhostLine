#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import json
import base64
import logging
import io
from datetime import datetime
import ssl
from typing import Any, Dict, Optional
from starlette.websockets import WebSocketState
import aiohttp

import certifi
from aiohttp import TCPConnector, ClientSession, FormData
from aiohttp.client_exceptions import ClientResponseError
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from twilio.rest import Client as TwilioClient
from deepgram import DeepgramClient
from pydub import AudioSegment
import wave
import numpy as np
import audioop
from scipy.signal import resample_poly
import yaml
from pathlib import Path
from database import init_db
from data import PROSODY_VARIATIONS, STAGE_CHECKINS, STAGE_TIMINGS, SalesStage
from nlp import generate_sales_reply
from dotenv import load_dotenv

# ------------------------
# Logging Configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SalesAutomation")

# ------------------------
# Configuration & SSL
# ------------------------
class Config:
    def __init__(self):
        logger.info("Loading configuration from environment variables")
        load_dotenv()
        self.twilio_sid    = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token  = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from   = os.getenv("TWILIO_FROM_NUMBER")
        self.openai_key    = os.getenv("OPENAI_API_KEY")
        self.deepgram_key  = os.getenv("DEEPGRAM_API_KEY")
        self.eleven_key    = os.getenv("ELEVENLABS_API_KEY")
        self.ngrok_token   = os.getenv("NGROK_AUTHTOKEN")
        self.db_path       = os.getenv("SQLITE_DB_PATH", "sales_tracking.db")
        self.babble_noise  = os.getenv("BABBLE_NOISE_PATH", "ambient_noise.wav")
        
        # Validate environment variables
        missing_vars = []
        for var_name, var_value in [
            ("TWILIO_ACCOUNT_SID", self.twilio_sid),
            ("TWILIO_AUTH_TOKEN", self.twilio_token),
            ("TWILIO_FROM_NUMBER", self.twilio_from),
            ("OPENAI_API_KEY", self.openai_key),
            ("DEEPGRAM_API_KEY", self.deepgram_key),
            ("ELEVENLABS_API_KEY", self.eleven_key),
            ("NGROK_AUTHTOKEN", self.ngrok_token)
        ]:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
            
        # Basic validation of API keys
        if len(self.deepgram_key) < 10:
            logger.error("DEEPGRAM_API_KEY appears to be invalid (too short)")
            sys.exit(1)
        if len(self.eleven_key) < 10:
            logger.error("ELEVENLABS_API_KEY appears to be invalid (too short)")
            sys.exit(1)
            
        logger.info("Environment variables loaded successfully")
        
        # SSL context
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
        logger.info("SSL context configured with certifi CA bundle")

# ------------------------
# Ambient Noise Manager
# ------------------------
class AmbientNoise:
    BASE_RATIO = 0.1  
    def __init__(self, path: str, target_rate=8000):
        logger.info("Initializing ambient noise from %s", path)
        self.target_rate = target_rate
        self.buffer = self._load(path)
        self.idx = 0

    def _load(self, path: str) -> bytes:
        try:
            wf = wave.open(path, 'rb')
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            if wf.getnchannels() > 1:
                samples = samples.reshape(-1, wf.getnchannels()).mean(axis=1).astype(np.int16)
            if wf.getframerate() != self.target_rate:
                samples = resample_poly(samples, self.target_rate, wf.getframerate()).astype(np.int16)
            wf.close()
            logger.info("Loaded ambient noise file successfully")
            return samples.tobytes()
        except FileNotFoundError:
            samples = np.random.normal(0, 50, self.target_rate * 10).astype(np.int16)  # Reduced amplitude
            logger.warning("Ambient noise file not found; using synthetic noise")
            return samples.tobytes()

    def mix(self, pcm: bytes, level=0.3) -> bytes:  # Default level reduced to 0.3
        ratio = min(level, 1.0) * self.BASE_RATIO
        needed = len(pcm)
        out = b''
        idx = self.idx  # Use local variable to avoid race conditions
        while len(out) < needed:
            remaining = needed - len(out)
            chunk = self.buffer[idx : idx + remaining]
            if len(chunk) < remaining:  # Wrap around
                chunk += self.buffer[:remaining - len(chunk)]
                idx = remaining - len(chunk)
            else:
                idx += len(chunk)
            out += chunk
        self.idx = idx % len(self.buffer)  # Update index only once
        
        # Boost the original audio to make it clearer
        try:
            pcm_boosted = audioop.mul(pcm, 2, 1.5)
            ambient = audioop.mul(out, 2, ratio)
            mixed = audioop.add(pcm_boosted, ambient, 2)
        except audioop.error as e:
            logger.error(f"Audio mixing error: {e}")
            return pcm  # Return original audio if mixing fails
        logger.debug("Mixed audio with ambient noise (level=%s)", level)
        return mixed

    @staticmethod
    def ulaw(pcm: bytes) -> bytes:
        encoded = audioop.lin2ulaw(pcm, 2)
        logger.debug("Converted PCM to µ-law format, %d bytes", len(encoded))
        return encoded

# ------------------------
# Playbook Loader
# ------------------------
class Playbook:
    def __init__(self, path: str):
        logger.info("Loading playbook from %s", path)
        self.seq = self.load(path)

    @staticmethod
    def load(path: str) -> list[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            logger.error("Playbook file not found: %s", path)
            raise FileNotFoundError(f"Playbook {path} not found")
        data = yaml.safe_load(p.read_text())
        seq = []
        for entry in data.get("sequence", []):
            seq.append({
                "stage": SalesStage[entry["stage"]],
                **{k: entry.get(k) for k in entry if k != "stage"}
            })
        logger.info("Loaded %d playbook stages", len(seq))
        return seq

# ------------------------
# ElevenLabs Voice Cloner & TTS
# ------------------------
class VoiceService:
    def __init__(self, eleven_key: str, ssl_context: ssl.SSLContext):
        self.api_key = eleven_key
        self.ssl = ssl_context
        logger.info("VoiceService initialized with ElevenLabs API")

    async def clone(self, sample_path: str, name="sales_voice") -> Optional[str]:
        logger.info("Cloning voice from sample: %s (name=%s)", sample_path, name)
        ext = os.path.splitext(sample_path)[1].lower()
        if ext == '.m4a':
            audio_seg = AudioSegment.from_file(sample_path, format='m4a')
            buf = io.BytesIO()
            audio_seg.export(buf, format='wav')
            data = buf.getvalue()
        else:
            data = open(sample_path, 'rb').read()
        url = "https://api.elevenlabs.io/v1/voices/add"
        logger.info("ElevenLabs API URL for cloning: %s", url)
        headers = {"xi-api-key": self.api_key}
        form = FormData()
        form.add_field("name", name)
        form.add_field("description", "Professional warm voice")
        form.add_field("files", data, filename="sample.wav", content_type="audio/wav")
        async with ClientSession(connector=TCPConnector(ssl=self.ssl)) as sess:
            async with sess.post(url, headers=headers, data=form) as resp:
                if resp.status == 200:
                    vid = (await resp.json()).get("voice_id")
                    logger.info("Voice cloned successfully: %s", vid)
                    return vid
                text = await resp.text()
                logger.error("Voice cloning failed (status=%d): %s", resp.status, text)
        return None

    async def synth(self, text: str, voice_id: str, persona: str="default") -> bytes:
        logger.info("Synthesizing TTS: voice_id=%s, persona=%s", voice_id, persona)
        # ElevenLabs doesn't support SSML, just use plain text
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        logger.info("ElevenLabs TTS URL: %s", url)
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,  # Plain text, not SSML
            "model_id": "eleven_monolingual_v1", 
            "voice_settings": {
                "stability": 0.9, 
                "similarity_boost": 0.99,
                "style": 0.9,  # Add style parameter for more expression
                "use_speaker_boost": False  # Enhance speaker presence
            }
        }
        async with ClientSession(connector=TCPConnector(ssl=self.ssl)) as sess:
            try:
                async with sess.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"ElevenLabs API error {resp.status}: {error_text}")
                        # Return a simple beep instead of silence
                        seg = AudioSegment.silent(duration=100) + \
                              AudioSegment.from_raw(bytes([0x80, 0xFF] * 100), sample_width=1, frame_rate=8000, channels=1)
                        return seg.raw_data
                        
                    mp3 = await resp.read()
                    logger.info(f"Received audio data from ElevenLabs: {len(mp3)} bytes")

                    if len(mp3) < 1000:  # Check if response is too small
                        logger.error(f"ElevenLabs returned very small audio file: {len(mp3)} bytes")
                        # Create a silent audio segment as fallback
                        seg = AudioSegment.silent(duration=500)
                    else:
                        seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
                    
                    # Normalize and boost the audio
                    seg = seg.normalize(headroom=0.1)  # Normalize to near maximum volume
                    seg = seg.set_frame_rate(8000).set_channels(1)
                    pcm = seg.raw_data
                    logger.debug("Received PCM data, %d bytes", len(pcm))
                    return pcm
            except ClientResponseError as e:
                logger.error(f"ElevenLabs API error: {e}")
                # Return silence as fallback
                seg = AudioSegment.silent(duration=500)
                seg = seg.set_frame_rate(8000).set_channels(1)
                return seg.raw_data

# ------------------------
# Deepgram Handler
# ------------------------
class DeepgramHandler:
    def __init__(self, key: str, ssl: ssl.SSLContext):
        self.client = DeepgramClient(key, ssl_context=ssl)
        logger.info("DeepgramHandler initialized")

    async def connect(self):
        logger.info("Connecting to Deepgram WebSocket")
        try:
            await self.client.connect()
            logger.info("Connected to Deepgram successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def send(self, frame: bytes):
        try:
            await self.client.send(frame)
            logger.debug("Sent audio frame to Deepgram (%d bytes)", len(frame))
        except Exception as e:
            logger.error(f"Error sending to Deepgram: {e}")
            raise

    async def iterate(self):
        try:
            async for transcript in self.client.receive_final():
                if transcript:
                    logger.debug("Deepgram final transcript: %s", transcript)
                    yield transcript
        except Exception as e:
            logger.error(f"Error receiving from Deepgram: {e}")
            raise

    async def close(self):
        try:
            await self.client.close()
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Error closing Deepgram connection: {e}")

# ------------------------
# Twilio & WebSocket Handler
# ------------------------
class CallHandler:
    def __init__(self, config: Config, ambient: AmbientNoise, voice_svc: VoiceService,
                 dg: DeepgramHandler, playbook: Optional[Playbook], db_conn):
        self.cfg = config
        self.ambient = ambient
        self.voice_svc = voice_svc
        self.dg = dg
        self.playbook_seq = playbook.seq if playbook else None
        self.db = db_conn
        logger.info("CallHandler initialized with playbook stages: %s",
                    [s['stage'].name for s in (self.playbook_seq or [])])

    async def handle(self, ws: WebSocket, voice_id: str, campaign: str, persona: str):
        logger.info("New WebSocket connection: campaign=%s, persona=%s", campaign, persona)
        await ws.accept()
        self.cfg = Config()
        self.dg = DeepgramHandler(self.cfg.deepgram_key, self.cfg.ssl_context)
        await self.dg.connect()
        # Initialize variables
        call_sid = None
        stream_sid = None
        caller_number = None
        greeting_task = None  # Initialize greeting_task properly
        
        # Wait for the start event to get the necessary information
        try:
            first_msg = await ws.receive_text()  # Fix: remove asyncio.run()
            data = json.loads(first_msg)
            logger.debug(f"First message event: {data.get('event')}")
            
            # Twilio usually sends connected then start
            while data.get("event") != "start":
                first_msg = await ws.receive_text()  # Fix: remove asyncio.run()
                data = json.loads(first_msg)
                logger.debug(f"Waiting for start event, got: {data.get('event')}")
            
            if data.get("event") == "start":
                stream_sid = data.get("streamSid")
                start_data = data.get("start", {})
                call_sid = start_data.get("callSid")
                custom_parameters = start_data.get("customParameters", {})
                caller_number = custom_parameters.get("from", start_data.get("from"))
                logger.info(f"Start event received: CallSid={call_sid}, StreamSid={stream_sid}, Caller={caller_number}")
                
        except Exception as e:
            logger.error(f"Error processing start event: {e}", exc_info=True)
            call_sid = "unknown"
            
        logger.info("CallSid=%s accepted", call_sid)        
        
        # Insert call record
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO calls VALUES(?,?,?,?,?,?,?,?,?,?)",
                (call_sid, None, None, voice_id, campaign, persona, None, None, None, None)
            )
            self.db.commit()
            logger.debug("Inserted call record into DB for %s", call_sid)
        except Exception as e:
            logger.error(f"Failed to insert call record: {e}")
        
        # Insert profile
        if caller_number:
            try:
                self.db.execute(
                    "INSERT OR IGNORE INTO customer_profiles VALUES(?,?,?,?,?,?)",
                    (caller_number, None, None, None, persona, None)
                )
                self.db.commit()
                logger.debug("Inserted customer profile for %s", caller_number)
            except Exception as e:
                logger.error(f"Failed to insert customer profile: {e}")
        
        # Deepgram connection
        try:
            await self.dg.connect()
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            await ws.close(1011, "Failed to initialize speech recognition")
            return
            
        current_stage = SalesStage.RAPPORT
        last_activity = asyncio.get_event_loop().time()
        check_in_sent = False
        
        # Define nested functions properly indented within handle method
        async def send_greeting():
            nonlocal stream_sid
            if hasattr(send_greeting, 'already_sent'):
                logger.warning("send_greeting called multiple times, ignoring")
                return
            send_greeting.already_sent = True
            
            try:
                if not stream_sid:
                    logger.error("send_greeting called without stream_sid")
                    return
                    
                # Initial greeting
                greeting = "Hello! Thanks for taking my call today. How are you doing?"
                logger.info(f"Sending initial greeting: {greeting}")
                
                pcm = await self.voice_svc.synth(greeting, voice_id, persona)
                logger.info(f"Got {len(pcm)} bytes of PCM from ElevenLabs")
                
                mixed = self.ambient.mix(pcm, 0.2)  # Low ambient noise for clarity
                ulaw = self.ambient.ulaw(mixed)
                logger.info(f"Converted to {len(ulaw)} bytes of µ-law audio")
                
                # Send the audio in chunks
                chunk_size = 160  # Standard µ-law chunk size for Twilio
                chunks_sent = 0
                for i in range(0, len(ulaw), chunk_size):
                    try:
                        if ws.client_state != WebSocketState.CONNECTED:
                            logger.warning("WebSocket disconnected during greeting")
                            break
                        
                        chunk = base64.b64encode(ulaw[i:i+chunk_size]).decode('ascii')
                        json_msg = json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": chunk}
                        })
                        # Test if it's valid JSON
                        json.loads(json_msg)  
                        await ws.send_text(json_msg)
                        chunks_sent += 1
                    except Exception as e:
                        logger.error(f"Error sending greeting chunk: {e}")
                        break
                logger.info(f"Greeting sent successfully - {chunks_sent} chunks sent")
            except Exception as e:
                logger.exception(f"Error sending greeting: {e}")

        async def monitor_silence():
            nonlocal last_activity, check_in_sent, current_stage, stream_sid
            try:
                while True:
                    await asyncio.sleep(1)

                    # bail if the socket is gone or not connected
                    try:
                        if ws.client_state != WebSocketState.CONNECTED:
                            logger.info("WebSocket no longer connected – stopping monitor_silence")
                            return
                    except:
                        logger.info("WebSocket state check failed – stopping monitor_silence")
                        return

                    # don't send until Twilio has told us the streamSid
                    if stream_sid is None:
                        continue

                    silent_time = asyncio.get_event_loop().time() - last_activity
                    threshold = STAGE_TIMINGS[current_stage]

                    if silent_time >= threshold and not check_in_sent:
                        text = STAGE_CHECKINS[current_stage]
                        logger.info(f"Silence threshold hit ({silent_time:.1f}s) – sending check-in: {text}")

                        try:
                            pcm = await self.voice_svc.synth(text, voice_id, persona)
                            mixed = self.ambient.mix(pcm, 0.2)  # Lower ambient noise
                            ulaw = self.ambient.ulaw(mixed)

                            chunk_size = 160  # Standard µ-law chunk size for Twilio
                            for i in range(0, len(ulaw), chunk_size):
                                try:
                                    chunk = base64.b64encode(ulaw[i:i+chunk_size]).decode('ascii')
                                    json_msg = json.dumps({
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": chunk}
                                    })
                                    # Test if it's valid JSON
                                    json.loads(json_msg)  
                                    await ws.send_text(json_msg)
                                except Exception as e:
                                    logger.error(f"JSON serialization error: {e}")
                            logger.info("Check-in message sent successfully")
                            check_in_sent = True
                            # Reset activity timer after sending check-in
                            last_activity = asyncio.get_event_loop().time()
                        except Exception as e:
                            logger.error(f"Failed to send check-in message: {e}")
                            
            except asyncio.CancelledError:
                logger.info("monitor_silence task cancelled")
            except Exception as e:
                logger.exception(f"monitor_silence crashed: {e}")

        # Create and start tasks
        silence_task = asyncio.create_task(monitor_silence())
        greeting_task = None  # We'll start this after we have stream_sid
        
        if stream_sid:
            greeting_task = asyncio.create_task(send_greeting())
            logger.info("Greeting task started immediately after receiving stream_sid")
        else:
            logger.error("No stream_sid received in start event!")
        
        async def pump_in():
            nonlocal last_activity, check_in_sent, stream_sid, greeting_task
            try:
                logger.info("pump_in started.")
                
                while True:
                    try:
                        # For FastAPI WebSocket, we need to use different methods
                        msg = await ws.receive_text()
                        
                        data = json.loads(msg)
                        event_type = data.get("event")
                        logger.debug(f"pump_in event received: {event_type}")

                        if event_type == "connected":
                            logger.info("Twilio connected event received")
                        
                        elif event_type == "start":
                            # This shouldn't happen as we already processed it
                            logger.warning("Received duplicate start event")

                        elif event_type == "media":
                            payload = data["media"]["payload"]
                            frame = base64.b64decode(payload)
                            
                            # Send the original mu-law data directly to Deepgram
                            try:
                                await self.dg.send(frame)
                                last_activity = asyncio.get_event_loop().time()
                                check_in_sent = False
                            except Exception as e:
                                logger.error(f"Deepgram send error: {e}")
                                try:
                                    await self.dg.connect()
                                except Exception as conn_err:
                                    logger.error(f"Failed to reconnect to Deepgram: {conn_err}")

                        elif event_type == "stop":
                            logger.info("Call stop event received.")
                            break
                            
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in message")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if "closed" in str(e).lower():
                            logger.info("WebSocket connection closed")
                            break
                        continue

            except Exception as e:
                logger.exception(f"Unexpected pump_in error: {e}")
            finally:
                if silence_task and not silence_task.done():
                    silence_task.cancel()
                if greeting_task and not greeting_task.done():
                    greeting_task.cancel()
                logger.info("pump_in completed.")

        async def pump_out():
            nonlocal current_stage, stream_sid, last_activity, check_in_sent
            buffer = ""
            waiting_for_utterance_end = False
            last_speech_final_time = 0
            last_words_time = 0
            
            logger.info("pump_out started.")
            try:
                async for data in self.dg.client.receive_final():
                    current_time = asyncio.get_event_loop().time()
                    
                    # Handle transcript data
                    if isinstance(data, dict):
                        if data.get("type") == "transcript":
                            transcript = data.get("transcript", "").strip()
                            speech_final = data.get("speech_final", False)
                            
                            if transcript:
                                # Add to buffer
                                if buffer and not buffer.endswith(' '):
                                    buffer += " "
                                buffer += transcript
                                
                                # Reset silence detection
                                last_activity = current_time
                                check_in_sent = False
                                last_words_time = current_time
                                
                                logger.debug(f"Buffer updated: '{buffer}' (speech_final: {speech_final})")
                                
                                # Only process if speech_final and we have a meaningful utterance
                                if speech_final:
                                    cleaned_transcript = buffer.strip()
                                    
                                    # Check if the utterance is complete enough to process
                                    words = cleaned_transcript.split()
                                    is_complete_thought = (
                                        len(words) >= 3 or  # At least 3 words
                                        any(cleaned_transcript.endswith(p) for p in ['.', '?', '!']) or  # Has end punctuation
                                        (current_time - last_words_time) > 1.5  # Long pause
                                    )
                                    
                                    if cleaned_transcript and is_complete_thought:
                                        logger.info(f"Processing utterance from speech_final: '{cleaned_transcript}'")
                                        await process_utterance(cleaned_transcript)
                                        buffer = ""
                                        last_speech_final_time = current_time
                                        waiting_for_utterance_end = False
                                    else:
                                        # Keep buffering if not a complete thought
                                        waiting_for_utterance_end = True
                                else:
                                    # Continue building buffer if not speech_final
                                    waiting_for_utterance_end = True
                        
                        elif data.get("type") == "utterance_end":
                            # Process if we have buffer and were waiting for utterance_end
                            if waiting_for_utterance_end and buffer:
                                cleaned_transcript = buffer.strip()
                                if cleaned_transcript:
                                    logger.info(f"Processing utterance from utterance_end: '{cleaned_transcript}'")
                                    await process_utterance(cleaned_transcript)
                                    buffer = ""
                            waiting_for_utterance_end = False
                    
            except Exception as e:
                logger.exception(f"Unexpected pump_out error: {e}")
            finally:
                if silence_task and not silence_task.done():
                    silence_task.cancel()
                logger.info("pump_out completed.")

        async def process_utterance(cleaned_transcript):
            """Helper function to process a complete utterance"""
            nonlocal current_stage, stream_sid
            
            logger.info(f"Processing complete utterance: '{cleaned_transcript}'")
            
            # Log user transcription to database
            try:
                self.db.execute(
                    "INSERT INTO messages(call_sid, role, content, timestamp, sales_stage) VALUES (?, ?, ?, ?, ?)",
                    (call_sid, "user", cleaned_transcript, datetime.utcnow().isoformat(), current_stage.name)
                )
                self.db.commit()
                logger.debug("User transcription saved to DB.")
            except Exception as e:
                logger.error(f"Failed to save user transcription: {e}")
            
            # Generate NLP response
            try:
                response, next_stage = await generate_sales_reply(call_sid, cleaned_transcript, current_stage)
                logger.info(f"Generated sales reply: '{response}' | Next stage: {next_stage.name}")
            except Exception as e:
                logger.error(f"Failed to generate sales reply: {e}")
                response = "I appreciate what you're saying. Could you tell me more about that?"
                next_stage = current_stage
            
            current_stage = next_stage
            
            # Ensure stream_sid is available
            for _ in range(10):
                if stream_sid:
                    break
                logger.warning("Waiting for stream_sid before sending response...")
                await asyncio.sleep(0.5)
            
            if not stream_sid:
                logger.error("streamSid still not set, response NOT sent to Twilio.")
                return
            
            # Synthesize TTS response
            try:
                pcm = await self.voice_svc.synth(response, voice_id, persona)
                mixed = self.ambient.mix(pcm, 0.2)
                ulaw = self.ambient.ulaw(mixed)
                
                chunk_size = 160  # µ-law chunk size
                for i in range(0, len(ulaw), chunk_size):
                    chunk = base64.b64encode(ulaw[i:i+chunk_size]).decode('ascii')
                    json_msg = json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": chunk}
                    })
                    await ws.send_text(json_msg)
                
                logger.info("Synthesized audio response sent successfully.")
            except Exception as e:
                logger.error(f"Failed to send TTS response: {e}")
            
            # Log bot response to database
            try:
                self.db.execute(
                    "INSERT INTO messages(call_sid, role, content, timestamp, sales_stage) VALUES (?, ?, ?, ?, ?)",
                    (call_sid, "assistant", response, datetime.utcnow().isoformat(), current_stage.name)
                )
                self.db.commit()
            except Exception as e:
                logger.error(f"Failed to save bot response: {e}")


        try:
            # Run both pump tasks concurrently
            await asyncio.gather(pump_in(), pump_out())
        except Exception as e:
            logger.exception(f"Error in handle_ws: {e}")
        finally:
            # Ensure all tasks are properly cancelled
            if silence_task and not silence_task.done():
                silence_task.cancel()
                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass
                    
            if greeting_task and not greeting_task.done():
                greeting_task.cancel()
                try:
                    await greeting_task
                except asyncio.CancelledError:
                    pass
            
            # Close any lingering connections
            try:
                await self.dg.close()
            except:
                pass
            
            try:
                await ws.close()
            except:
                pass
            
            logger.info(f"WebSocket handler for call {call_sid} completed.")


# ------------------------
# Dashboard & API
# ------------------------
class DashboardApp:
    def __init__(self, db_conn):
        self.db = db_conn
        self.app = FastAPI()
        self._mount()
        logger.info("DashboardApp initialized")

    def _mount(self):
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            logger.info("Dashboard root accessed")
            # build HTML with stats
            return "<h1>Dashboard</h1>"

        @self.app.post("/voice", response_class=PlainTextResponse)
        async def voice(req: Request):
            ws_url = os.getenv("NGROK_WS_URL")
            if not ws_url:
                logger.error("NGROK_WS_URL not set when handling /voice request")
                raise HTTPException(500, "Tunnel not initialized")
            
            # Handle query parameters properly for WebSocket URLs
            campaign = req.query_params.get('campaign')
            persona = req.query_params.get('persona')
            
            if campaign:
                ws_url += f"?campaign={campaign}"
                if persona:
                    ws_url += f"&persona={persona}"
            elif persona:
                ws_url += f"?persona={persona}"
            
            # Escape ampersands for XML
            xml_safe_url = ws_url.replace('&', '&amp;')
            
            logger.info("Providing TwiML stream URL: %s", ws_url)
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we connect you.</Say>
  <Connect>
    <Stream url="{xml_safe_url}"/>
  </Connect>
  <Pause length="300"/>
</Response>"""
            return twiml

# ------------------------
# Main Application
# ------------------------
class SalesAutomationApp:
    def __init__(self):
        self.config = Config()
        self.ambient = AmbientNoise(self.config.babble_noise)
        self.voice_svc = VoiceService(self.config.eleven_key, self.config.ssl_context)
        self.dg = DeepgramHandler(self.config.deepgram_key, self.config.ssl_context)
        self.db_conn = init_db(self.config.db_path)
        logger.info("SalesAutomationApp initialized")
    
    def __del__(self):
        # Clean up database connection
        try:
            if hasattr(self, 'db_conn'):
                self.db_conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def run(self):
        parser = argparse.ArgumentParser(description="Advanced Sales Automation System")
        sub = parser.add_subparsers(dest='cmd', required=True)
        # clone
        c1 = sub.add_parser('clone', help="Clone a voice sample (WAV or M4A)")
        c1.add_argument('sample', help="Path to sample file")
        c1.add_argument('--name', default='sales_voice', help="Name for cloned voice")
        # serve
        s1 = sub.add_parser('serve', help="Start server with specified voice")
        s1.add_argument('--voice-id', required=True, help="ElevenLabs voice ID to use")
        s1.add_argument('--port', type=int, default=8000, help="Port for FastAPI/Ngrok tunnel")
        s1.add_argument('--playbook', help="Path to playbook YAML")
        # call
        c2 = sub.add_parser('call', help="Place a call via Twilio and stream to /twilio")
        c2.add_argument('phone', help="Phone number to dial (+E.164 format)")
        c2.add_argument('--campaign', help="Campaign name for analytics")
        c2.add_argument('--persona', help="Persona style for TTS prosody")
        args = parser.parse_args()
        logger.info("Running command: %s", args.cmd)

        if args.cmd == 'clone':
            vid = asyncio.run(self.voice_svc.clone(args.sample, args.name))
            if vid:
                logger.info("Clone complete, Voice ID: %s", vid)
                print(f"✅ Voice created. ID: {vid}")
            else:
                logger.error("Voice cloning failed")
                print("❌ Voice cloning failed.")

        elif args.cmd == 'serve':
            playbook = Playbook(args.playbook) if args.playbook else None
            logger.info("Starting server with voice ID %s on port %d", args.voice_id, args.port)
            # Ngrok tunnel
            from pyngrok import ngrok
            ngrok.set_auth_token(self.config.ngrok_token)
            public_url = ngrok.connect(args.port, "http").public_url
            ws_url = public_url.replace("http://", "wss://").replace("https://", "wss://") + "/twilio"
            os.environ["NGROK_WS_URL"] = ws_url
            logger.info("Ngrok public URL: %s", public_url)
            logger.info("WebSocket endpoint URL: %s", ws_url)

            app = FastAPI()
            handler = CallHandler(self.config, self.ambient, self.voice_svc, self.dg, playbook, self.db_conn)
            @app.websocket('/twilio')
            async def ws_endpoint(ws: WebSocket):
                params = ws.query_params
                campaign = params.get('campaign', 'general')
                persona = params.get('persona', 'default')
                await handler.handle(ws, args.voice_id, campaign, persona)

            dash = DashboardApp(self.db_conn)
            app.mount('/', dash.app)
            uvicorn.run(app, host='0.0.0.0', port=args.port)

        elif args.cmd == 'call':
            ws_url = os.getenv("NGROK_WS_URL")
            if not ws_url:
                logger.error("No NGROK_WS_URL found for call command")
                print("❌ No NGROK_WS_URL found. Start 'serve' first.")
                sys.exit(1)
            logger.info("Initiating Twilio call to %s via WebSocket %s", args.phone, ws_url)
            
            # Add parameters to the URL for campaign and persona
            param_ws_url = ws_url
            if args.campaign:
                param_ws_url += f"?campaign={args.campaign}"
                if args.persona:
                    param_ws_url += f"&persona={args.persona}"
            elif args.persona:
                param_ws_url += f"?persona={args.persona}"
            
            # Escape ampersands for XML
            xml_safe_url = param_ws_url.replace('&', '&amp;')
                
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we connect you.</Say>
  <Connect>
    <Stream url="{xml_safe_url}"/>
  </Connect>
  <Pause length="300"/>
</Response>"""
            client = TwilioClient(self.config.twilio_sid, self.config.twilio_token)
            call = client.calls.create(
                to=args.phone,
                from_=self.config.twilio_from,
                twiml=twiml
            )
            logger.info("Placed call SID %s to %s", call.sid, args.phone)
            print(f"📞 Call placed. SID: {call.sid}")

        else:
            parser.print_help()

if __name__ == '__main__':
    import keys
    SalesAutomationApp().run()