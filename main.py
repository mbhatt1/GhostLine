#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
import json
import base64
import logging
import wave
import datetime
import sqlite3
import time
import io
import random
import re
from enum import Enum
from data import PROSODY_VARIATIONS, STAGE_CHECKINS, STAGE_TIMINGS, SalesStage
from database import init_db
from typing import Dict, List, Tuple, Optional

import numpy as np
import audioop
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import openai
import aiohttp
from pydub import AudioSegment
from deepgram import Deepgram
from dotenv import load_dotenv
import keys
from nlp import generate_sales_reply

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
BABBLE_NOISE_PATH = os.getenv("BABBLE_NOISE_PATH", "ambient_noise.wav")
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY", OPENAI_API_KEY)
DB_PATH = os.getenv("SQLITE_DB_PATH", "sales_tracking.db")

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, 
            OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, NGROK_AUTHTOKEN]):
    print("❌ Missing one or more critical environment variables. Check .env or the top of this script.")
    sys.exit(1)

# Initialize external services
openai.api_key = OPENAI_API_KEY
dg_client = Deepgram(DEEPGRAM_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Ambient Noise
# ─────────────────────────────────────────────────────────────────────────────
BASE_AMBIENT_RATIO = 0.1
VAD_THRESHOLD = 300

_ambient_buf = None
_ambient_idx = 0

def load_ambient_noise():
    global _ambient_buf
    try:
        wf = wave.open(BABBLE_NOISE_PATH, 'rb')
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16)
        if wf.getnchannels() > 1:
            samples = samples.reshape(-1, wf.getnchannels()).mean(axis=1).astype(np.int16)
        if wf.getframerate() != 8000:
            samples = resample_poly(samples, 8000, wf.getframerate()).astype(np.int16)
        _ambient_buf = samples.tobytes()
        wf.close()
    except FileNotFoundError:
        # Fallback: synthetic noise
        samples = np.random.normal(0, 100, 8000 * 10).astype(np.int16)
        _ambient_buf = samples.tobytes()
        logging.warning("Created synthetic ambient noise as file not found")

def get_ambient_segment(n):
    global _ambient_idx
    needed = n * 2
    out = b''
    while len(out) < needed:
        chunk = _ambient_buf[_ambient_idx : _ambient_idx + (needed - len(out))]
        out += chunk
        _ambient_idx = (_ambient_idx + len(chunk)) % len(_ambient_buf)
    return out

def mix_with_ambient(pcm, energy_level=1.0):
    ratio = min(energy_level, 1.0) * BASE_AMBIENT_RATIO
    ambient = get_ambient_segment(len(pcm) // 2)
    scaled = audioop.mul(ambient, 2, ratio)
    return audioop.add(pcm, scaled, 2)

def add_strategic_pause(pcm, pause_ms=800):
    silence = b'\x00' * int(8000 * pause_ms / 1000 * 2)
    return pcm + silence

def pcm_to_ulaw(pcm):
    return audioop.lin2ulaw(pcm, 2)

def adjust_voice_properties(pcm, properties):
    """Simple pitch/time transformation using resample_poly."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    # time-stretch
    rate = properties.get("rate", 1.0)
    if rate != 1.0:
        orig_len = len(samples)
        target_len = int(orig_len / rate)
        samples = resample_poly(samples, target_len, orig_len).astype(np.int16)

    # pitch-shift
    pitch = properties.get("pitch", 1.0)
    if pitch != 1.0:
        # naive approach: upsample -> downsample
        up = int(len(samples) * pitch)
        samples = resample_poly(samples, up, len(samples))
        down = int(len(samples) * pitch)
        samples = resample_poly(samples, len(samples), up).astype(np.int16)

    return samples.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Voice Cloning (Handles M4A -> WAV)
# ─────────────────────────────────────────────────────────────────────────────
async def clone_voice(sample_path, name="sales_voice"):
    """
    Create a custom voice clone with specific tonality for sales.
    If sample_path is M4A, convert to WAV on the fly using pydub.
    """
    # Check file extension
    file_ext = os.path.splitext(sample_path)[1].lower()

    # Convert M4A -> WAV if needed
    if file_ext == ".m4a":
        logging.info("Detected M4A file; converting to WAV in memory.")
        audio_seg = AudioSegment.from_file(sample_path, format="m4a")
        tmp_wav = io.BytesIO()
        audio_seg.export(tmp_wav, format="wav")
        wav_data = tmp_wav.getvalue()
    else:
        with open(sample_path, "rb") as f:
            wav_data = f.read()

    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    form_data = aiohttp.FormData()
    form_data.add_field("name", name)
    form_data.add_field("description", "Professional sales voice with warm, trustworthy tonality")
    form_data.add_field("files", wav_data, filename="sample.wav", content_type="audio/wav")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        try:
            async with session.post(url, headers=headers, data=form_data) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    voice_id = data.get("voice_id")
                    logging.info(f"Voice cloned successfully. Voice ID: {voice_id}")
                    return voice_id
                else:
                    msg = await resp.text()
                    logging.error(f"Voice cloning failed. Status code={resp.status}, Resp={msg}")
                    return None
        except Exception as e:
            logging.exception(f"Exception during voice cloning: {e}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# TTS Synthesis
# ─────────────────────────────────────────────────────────────────────────────
async def tts_synthesize(text, voice_id, persona="default"):
    # Insert micro-pauses using SSML
    enhanced_text = text
    for phrase in ["but", "however", "importantly", "the key is", "interestingly", "specifically"]:
        pattern = f"\\b{phrase}\\b"
        replacement = f"{phrase}<break time='300ms'/>"
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)

    for word in ["exclusive", "limited", "guaranteed", "proven", "customized", "premium"]:
        pattern = f"\\b{word}\\b"
        replacement = f"<emphasis level='moderate'>{word}</emphasis>"
        enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)

    ssml_text = f"""
    <speak>
      <prosody rate="medium" pitch="medium">
        {enhanced_text}
      </prosody>
    </speak>
    """

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json"
    }

    voice_adjustments = PROSODY_VARIATIONS.get(persona, {"rate":1.0, "pitch":1.0})

    payload = {
        "text": ssml_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as s:
        async with s.post(url, json=payload, headers=headers) as r:
            r.raise_for_status()
            mp3 = await r.read()

    seg = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_data = seg.raw_data

    pcm_data = adjust_voice_properties(pcm_data, voice_adjustments)
    return pcm_data

# ─────────────────────────────────────────────────────────────────────────────
# Twilio WebSocket Handler (fastapi /ws -> handle conversation)
# ─────────────────────────────────────────────────────────────────────────────
async def handle_twilio(ws: WebSocket, voice_id, campaign=None, persona=None):
    # Omitted in interest of length; your original handle logic here
    # (We include it for completeness, but shortened comments.)
    
    log = logging.getLogger("twilio")
    global DB_CONN

    data = json.loads(await ws.receive_text())
    while data.get("event") != "start":
        data = json.loads(await ws.receive_text())
    
    call_params = data.get("start", {}).get("parameters", {})
    cs = call_params.get("callSid") or data["start"].get("callSid")
    phone = call_params.get("to")

    # Insert call if not present
    DB_CONN.execute("""
        INSERT OR IGNORE INTO calls VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        cs,
        datetime.datetime.utcnow().isoformat(),
        None,
        voice_id,
        campaign or "general",
        persona or "default",
        phone,
        None,
        None,
        None
    ))
    DB_CONN.commit()

    # Insert or ignore profile
    if phone:
        DB_CONN.execute("""
            INSERT OR IGNORE INTO customer_profiles 
            VALUES(?,?,?,?,?,?)
        """, (
            phone, None, None, None, persona or "default",
            datetime.datetime.utcnow().isoformat()
        ))
        DB_CONN.commit()

    # Start Deepgram streaming
    dg = await dg_client.transcription.live({
        "encoding": "mulaw",
        "sample_rate": 8000,
        "channels": 1
    })

    current_stage = SalesStage.RAPPORT
    last_activity = time.monotonic()
    check_in_triggered = False
    silence_threshold = STAGE_TIMINGS[current_stage]
    voice_persona = persona or "default"

    async def silence_monitor():
        nonlocal check_in_triggered, silence_threshold, current_stage
        while True:
            await asyncio.sleep(1)
            current_silence = time.monotonic() - last_activity
            silence_threshold = STAGE_TIMINGS[current_stage]
            if current_silence >= silence_threshold and not check_in_triggered:
                check_in_text = STAGE_CHECKINS[current_stage]
                DB_CONN.execute("""
                    INSERT INTO messages(call_sid, role, content, timestamp, sales_stage)
                    VALUES(?,?,?,?,?)
                """, (
                    cs, "assistant", check_in_text,
                    datetime.datetime.utcnow().isoformat(),
                    current_stage.name
                ))
                DB_CONN.commit()

                pcm = await tts_synthesize(check_in_text, voice_id, voice_persona)
                mixed_pcm = mix_with_ambient(pcm, 0.5)
                ulaw = pcm_to_ulaw(mixed_pcm)
                for i in range(0, len(ulaw), 160):
                    chunk = ulaw[i : i + 160]
                    await ws.send_json({
                        "event": "media",
                        "media": {"payload": base64.b64encode(chunk).decode()}
                    })
                check_in_triggered = True

    silence_task = asyncio.create_task(silence_monitor())

    async def pump_in():
        nonlocal last_activity, check_in_triggered
        try:
            async for msg in ws.iter_text():
                o = json.loads(msg)
                if o.get("event") == "media":
                    frame = base64.b64decode(o["media"]["payload"])
                    pcm = audioop.ulaw2lin(frame, 2)
                    if audioop.rms(pcm, 2) >= VAD_THRESHOLD:
                        last_activity = time.monotonic()
                        check_in_triggered = False
                    await dg.send(frame)
                elif o.get("event") == "stop":
                    DB_CONN.execute("""
                        UPDATE calls SET end_time=? WHERE call_sid=?
                    """, (datetime.datetime.utcnow().isoformat(), cs))
                    DB_CONN.commit()
                    await dg.finish()
                    break
        except Exception as e:
            log.exception(f"Error in input pump: {e}")
        finally:
            await dg.finish()

    async def pump_out():
        nonlocal current_stage, voice_persona
        try:
            async for msg in dg:
                if not msg.get("is_final"):
                    continue
                transcript = msg["channel"]["alternatives"][0]["transcript"].strip()
                if not transcript:
                    continue

                DB_CONN.execute("""
                    INSERT INTO messages(call_sid, role, content, timestamp, sales_stage)
                    VALUES(?,?,?,?,?)
                """, (
                    cs, "user", transcript,
                    datetime.datetime.utcnow().isoformat(),
                    current_stage.name
                ))
                DB_CONN.commit()

                response, new_stage = await generate_sales_reply(cs, transcript, current_stage)
                if new_stage != current_stage:
                    current_stage = new_stage
                    if current_stage in [SalesStage.URGENCY, SalesStage.CLOSE]:
                        voice_persona = "confident"
                    elif current_stage == SalesStage.OBJECTION:
                        voice_persona = "concerned"
                    elif current_stage == SalesStage.RAPPORT:
                        voice_persona = "excited"
                    else:
                        voice_persona = persona or "default"

                pcm = await tts_synthesize(response, voice_id, voice_persona)
                if "?" in response or current_stage in [SalesStage.CLOSE, SalesStage.TRIAL_CLOSE]:
                    pcm = add_strategic_pause(pcm, 600)

                importance = 0.7 if current_stage in [SalesStage.CLOSE, SalesStage.URGENCY] else 0.4
                mixed_pcm = mix_with_ambient(pcm, importance)
                ulaw = pcm_to_ulaw(mixed_pcm)
                for i in range(0, len(ulaw), 160):
                    chunk = ulaw[i : i + 160]
                    await ws.send_json({
                        "event": "media",
                        "media": {"payload": base64.b64encode(chunk).decode()}
                    })
        except Exception as e:
            log.exception(f"Error in output pump: {e}")

    try:
        await asyncio.gather(pump_in(), pump_out())
    finally:
        silence_task.cancel()

        # finalize outcome
        user_msgs = DB_CONN.execute("""
            SELECT content FROM messages WHERE call_sid=? AND role='user'
        """, (cs,)).fetchall()
        all_user_text = " ".join(m[0] for m in user_msgs)
        if all_user_text:
            pos_list = ["yes","interested","sign me up","sounds good","let's do it","start"]
            neg_list = ["no","not interested","too expensive","call back","maybe later"]
            pos_score = sum(1 for w in pos_list if w in all_user_text.lower())
            neg_score = sum(1 for w in neg_list if w in all_user_text.lower())
            outcome = "positive" if pos_score>neg_score else "negative" if neg_score>pos_score else "neutral"
            conv_score = (pos_score - neg_score) / max(1, (pos_score + neg_score)) + 0.5
            DB_CONN.execute("""
                UPDATE calls SET outcome=?, conversion_score=? WHERE call_sid=?
            """, (outcome, conv_score, cs))
            DB_CONN.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard & Stats
# ─────────────────────────────────────────────────────────────────────────────
async def get_dashboard_data():
    cur = DB_CONN.cursor()

    total_calls = cur.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    calls_by_outcome = dict(cur.execute("""
        SELECT outcome, COUNT(*) FROM calls 
        WHERE outcome IS NOT NULL GROUP BY outcome
    """).fetchall())

    objections_by_type = dict(cur.execute("""
        SELECT objection_type, COUNT(*) FROM objections GROUP BY objection_type
    """).fetchall())

    stage_transitions = cur.execute("""
        SELECT m1.sales_stage, m2.sales_stage, COUNT(*)
        FROM messages m1
        JOIN messages m2
             ON m1.call_sid = m2.call_sid
            AND m1.id < m2.id
        WHERE m1.sales_stage != m2.sales_stage 
          AND m1.role = 'assistant' 
          AND m2.role = 'assistant'
        GROUP BY m1.sales_stage, m2.sales_stage
    """).fetchall()

    campaign_stats = cur.execute("""
        SELECT campaign, AVG(conversion_score), COUNT(*)
        FROM calls
        WHERE conversion_score IS NOT NULL
        GROUP BY campaign
    """).fetchall()

    return {
        "total_calls": total_calls,
        "calls_by_outcome": calls_by_outcome,
        "objections_by_type": objections_by_type,
        "stage_transitions": stage_transitions,
        "campaign_stats": campaign_stats
    }

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
VOICE_ID = None
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/twilio")
async def tw(ws: WebSocket):
    await ws.accept()
    params = ws.query_params
    campaign = params.get("campaign","general")
    persona = params.get("persona","default")
    await handle_twilio(ws, VOICE_ID, campaign, persona)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    data = await get_dashboard_data()
    html = f"""
    <html>
    <head><title>Sales Automation Dashboard</title></head>
    <body>
      <h1>Sales Automation Dashboard</h1>
      <p>Total calls: {data['total_calls']}</p>
      <p>Calls by outcome: {data['calls_by_outcome']}</p>
      <p>Objections: {data['objections_by_type']}</p>
    </body>
    </html>
    """
    return html

@app.get("/api/stats")
async def api_stats():
    return await get_dashboard_data()

# ─────────────────────────────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import logging
import os
import sys

from flask import app
import uvicorn

from database import init_db
from pyngrok import ngrok
from twilio.rest import Client
from main import DB_PATH, NGROK_AUTHTOKEN, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, clone_voice, load_ambient_noise


def cmd_clone(args):
    """Clone a voice sample (handles m4a -> wav)."""
    loop = asyncio.get_event_loop()
    vid = loop.run_until_complete(clone_voice(args.sample, args.name))
    if vid:
        print(f"✅ Voice created. ID: {vid}")
        print(f"Use --voice-id {vid} with 'serve'")
    else:
        print("❌ Voice cloning failed.")

def cmd_serve(args):
    """Start the server with ngrok."""
    global VOICE_ID, DB_CONN
    VOICE_ID = args.voice_id

    load_ambient_noise()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    ngrok.set_auth_token(NGROK_AUTHTOKEN)

    DB_CONN = init_db(DB_PATH)

    public_url = ngrok.connect(args.port, "http").public_url
    ws_url = public_url.replace("http","ws") + "/twilio"
    os.environ["NGROK_WS_URL"] = ws_url

    print(f"Server is running on {public_url}")
    print(f"WebSocket endpoint: {ws_url}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)

def cmd_call(args):
    """Place a call using Twilio with a custom TwiML that streams to /twilio."""
    ws_url = os.getenv("NGROK_WS_URL")
    if not ws_url:
        print("❌ No NGROK_WS_URL found. Start 'serve' first.")
        sys.exit(1)

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    if args.campaign or args.persona:
        params = []
        if args.campaign:
            params.append(f"campaign={args.campaign}")
        if args.persona:
            params.append(f"persona={args.persona}")
        if params:
            ws_url = f"{ws_url}?{'&'.join(params)}"

    twiml = f"""
    <Response>
        <Connect>
            <Stream url='{ws_url}'>
                <Parameter name='callSid' value='{{{{CallSid}}}}' />
                <Parameter name='to' value='{{{{To}}}}' />
            </Stream>
        </Connect>
    </Response>
    """

    call = client.calls.create(
        to=args.phone,
        from_=TWILIO_FROM_NUMBER,
        twiml=twiml
    )
    print(f"📞 Call placed. SID: {call.sid}")
    if args.campaign:
        print(f"  Campaign: {args.campaign}")
    if args.persona:
        print(f"  Persona: {args.persona}")

def cmd_analytics(args):
    print("Analytics command not implemented in detail yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Sales Automation System")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # clone
    clone_parser = subparsers.add_parser("clone", help="Clone a voice sample (WAV or M4A)")
    clone_parser.add_argument("sample", help="Path to sample (e.g. sample.wav or sample.m4a)")
    clone_parser.add_argument("--name", default="sales_voice", help="Name for the cloned voice")
    clone_parser.set_defaults(func=cmd_clone)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start server with specified voice")
    serve_parser.add_argument("--voice-id", required=True, help="ElevenLabs voice ID")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    serve_parser.set_defaults(func=cmd_serve)

    # call
    call_parser = subparsers.add_parser("call", help="Place a call")
    call_parser.add_argument("phone", help="Phone number to dial (+1...)")
    call_parser.add_argument("--campaign", help="Campaign name")
    call_parser.add_argument("--persona", choices=list(PROSODY_VARIATIONS.keys()), help="Persona style (voice prosody)")
    call_parser.set_defaults(func=cmd_call)

    # analytics
    anal_parser = subparsers.add_parser("analytics", help="Generate analytics report (placeholder)")
    anal_parser.set_defaults(func=cmd_analytics)

    args = parser.parse_args()
    args.func(args)
