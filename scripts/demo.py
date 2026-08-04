#!/usr/bin/env python3
"""GhostLine live demo orchestrator.

Runs the full pipeline end-to-end with clean terminal output designed for
screen-sharing during a presentation:

  1. Verify environment
  2. Detect model capabilities
  3. Start the server + ngrok tunnel
  4. Place the call
  5. Stream the live pipeline trace (full, untruncated)
  6. Show the full call transcript
  7. Save transcript to a persistent file
  8. Clean up

Usage:
    uv run python scripts/demo.py [--playbook playbooks/it-test.yaml] [--to +1NNNNNNNNNN]
"""

from __future__ import annotations

import argparse
import os
import re as re_mod
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# ── Output helpers ────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"

# Persistent transcript directory
_TRANSCRIPT_DIR = Path("transcripts")


def step(num: int, title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─' * 60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  Step {num}: {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * 60}{_RESET}\n")


def ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


def label(key: str, val: str) -> None:
    print(f"  {key:20} {val}")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for plain-text file output."""
    return re_mod.sub(r"\033\[[0-9;]*m", "", text)


# ── Demo steps ─────────────────────────────────────────────────────────


def check_env() -> dict[str, str]:
    """Load .env and verify all required keys are present."""
    load_dotenv(override=True)
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",
        "NGROK_AUTHTOKEN",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        for k in missing:
            fail(f"Missing: {k}")
        sys.exit(1)
    model = os.environ.get("LITELLM_MODEL", "openrouter/openai/gpt-4o")
    model_lower = model.lower()
    if "openrouter" in model_lower:
        llm_key = "OPENROUTER_API_KEY"
    elif model_lower.startswith("groq/"):
        llm_key = "GROQ_API_KEY"
    elif model_lower.startswith("anthropic/"):
        llm_key = "ANTHROPIC_API_KEY"
    elif model_lower.startswith("gemini/"):
        llm_key = "GEMINI_API_KEY"
    else:
        llm_key = "OPENAI_API_KEY"
    if not os.environ.get(llm_key):
        fail(f"Missing LLM key: {llm_key}")
        sys.exit(1)

    ok("All environment variables present")
    label("Twilio SID:", os.environ["TWILIO_ACCOUNT_SID"][:8] + "...")
    label("Twilio from:", os.environ["TWILIO_FROM_NUMBER"])
    label("LLM model:", model)
    label("LLM key:", llm_key[:6] + "..." + os.environ[llm_key][-4:])
    return {"model": model, "llm_key": llm_key}


def detect_caps(model: str) -> None:
    """Run capability detection and print the result."""
    from ghostline.agent.capabilities import detect_capabilities

    caps = detect_capabilities(model)
    ok(f"Model: {caps.model}")
    label("Modality:", caps.modality.value)
    label("Native STT:", str(caps.native_stt))
    label("Native TTS:", str(caps.native_tts))
    label("Needs Deepgram:", str(caps.needs_deepgram))
    label("Needs ElevenLabs:", str(caps.needs_elevenlabs))
    label("Source:", caps.source)
    if caps.needs_deepgram:
        info("Pipeline: Deepgram STT → LLM → ElevenLabs TTS")
    else:
        info("Pipeline: Realtime voice (direct WSS bridge)")


def list_voices() -> str:
    """List ElevenLabs premade voices and return the first one."""
    import httpx

    key = os.environ["ELEVENLABS_API_KEY"]
    r = httpx.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key}, timeout=15)
    voices = r.json().get("voices", [])
    if not voices:
        fail("No ElevenLabs voices found")
        sys.exit(1)
    ok(f"{len(voices)} voices available. Using:")
    v = voices[0]
    label("Voice ID:", v["voice_id"])
    label("Name:", v["name"])
    for v2 in voices[1:4]:
        info(f"  alt: {v2['voice_id']} | {v2['name']}")
    return v["voice_id"]


_SERVER_LOG = "/tmp/ghostline_demo_server.log"


def start_server_and_tunnel(
    voice_id: str, playbook: str | None, port: int
) -> tuple[subprocess.Popen, str]:
    """Start the server, wait for startup, and extract the ngrok WSS URL.

    Redirects server stdout to a log file instead of a pipe to avoid
    buffer-fill deadlocks that kill the call when the pipe backs up.

    Returns:
        A tuple of (proc, ws_url).
    """
    cmd = ["uv", "run", "ghostline", "serve", "--voice-id", voice_id, "--port", str(port)]
    if playbook:
        cmd += ["--playbook", playbook]
    # Write server output to a file, NOT a pipe — pipe buffers fill up
    # and block the server process, killing the WebSocket connection.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_file = open(_SERVER_LOG, "w")  # noqa: SIM115 — closed in cleanup
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    ok(f"Starting server on port {port}...")
    # Poll the log file for the startup signal (avoids pipe blocking)
    ws_url: str | None = None
    for _ in range(300):
        time.sleep(0.1)
        if proc.poll() is not None:
            fail("Server process exited unexpectedly")
            sys.exit(1)
        try:
            content = Path(_SERVER_LOG).read_text()
        except OSError:
            continue
        m = re_mod.search(r"wss://[a-z0-9-]+\.ngrok-free\.app/twilio", content)
        if m:
            ws_url = m.group()
        if "Application startup complete" in content:
            ok("Server started")
            break
    if not ws_url:
        fail("Could not find ngrok WSS URL in startup output")
        sys.exit(1)
    ok(f"ngrok tunnel: {ws_url}")
    return proc, ws_url


def place_call(ws_url: str, to_number: str, campaign: str, target_name: str | None = None) -> str:
    """Place the outbound call via Twilio and return the call SID."""
    from urllib.parse import quote

    from twilio.rest import Client

    voice_url = ws_url.replace("wss://", "https://").replace("/twilio", "/voice")
    voice_url += f"?campaign={campaign}"
    if target_name:
        voice_url += f"&target_name={quote(target_name)}"
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    call = client.calls.create(
        to=to_number,
        from_=os.environ["TWILIO_FROM_NUMBER"],
        url=voice_url,
    )
    ok(f"Call placed: {call.sid}")
    label("From:", os.environ["TWILIO_FROM_NUMBER"])
    label("To:", to_number)
    label("Campaign:", campaign)
    return call.sid


_INTERESTING = (
    "Call started",
    "Deepgram WSS connected",
    "Processing utterance",
    "Reply generated",
    "ElevenLabs TTS",
    "Stop event",
    "Deepgram client closed",
    "CallSession voice modality",
    "Skipping Deepgram",
    "Stage transition",
    "Silence monitor",
)


def _format_log_line(line: str) -> str | None:  # noqa: PLR0911
    """Return formatted output for a pipeline log line, or None to skip."""
    if not any(kw in line for kw in _INTERESTING):
        return None
    if "Processing utterance" in line:
        # Extract full utterance between single quotes
        utt = line.split("'", 2)[1] if "'" in line else "(transcribed)"
        return f"  {_CYAN}TARGET{_RESET}  {utt}"
    if "Reply generated" in line:
        # Extract full reply between single quotes — no truncation
        parts = line.split("'")
        reply = parts[1] if len(parts) > 1 else "(reply generated)"
        return f"  {_GREEN}AGENT {_RESET}  {reply}"
    if "Call started" in line:
        return f"  {_BOLD}CALL CONNECTED{_RESET}"
    if "Stop event" in line:
        return f"\n  {_YELLOW}CALL ENDED{_RESET}"
    if "Deepgram WSS connected" in line:
        return f"  {_DIM}STT stream connected{_RESET}"
    if "Deepgram client closed" in line:
        return f"  {_DIM}STT stream closed{_RESET}"
    if "CallSession voice modality" in line:
        tag = (
            "text pipeline (STT + TTS)"
            if "needs_deepgram=True" in line
            else "realtime voice pipeline"
        )
        return f"  {_DIM}Voice mode: {tag}{_RESET}"
    if "Stage transition" in line:
        return f"  {_DIM}{line.split('Stage transition')[-1].strip()}{_RESET}"
    if "Silence monitor" in line:
        return f"  {_DIM}[silence check-in]{_RESET}"
    return None


def stream_pipeline(proc: subprocess.Popen, call_sid: str, timeout_s: int = 600) -> list[str]:
    """Stream the server log by tailing the log file.

    Reads from the file instead of a pipe to avoid buffer-fill deadlocks.
    Returns a list of formatted lines for persistent logging.
    """
    print(f"\n{_BOLD}  {'─' * 56}{_RESET}")
    print(f"{_BOLD}  Live Call{_RESET}")
    print(f"{_BOLD}  {'─' * 56}{_RESET}\n")

    log_path = Path(_SERVER_LOG)
    start = time.time()
    stop_seen = False
    offset = log_path.stat().st_size if log_path.exists() else 0
    live_lines: list[str] = []
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            break
        try:
            content = log_path.read_text()
        except OSError:
            time.sleep(0.1)
            continue
        new_content = content[offset:]
        offset = len(content)
        for line in new_content.splitlines():
            formatted = _format_log_line(line.rstrip())
            if formatted is not None:
                print(formatted)
                live_lines.append(formatted)
                if "CALL ENDED" in formatted:
                    stop_seen = True
        if stop_seen:
            break
        time.sleep(0.2)
    return live_lines


def show_transcript(call_sid: str) -> list[dict[str, str]]:
    """Query the SQLite DB for the call transcript and print it.

    Returns the list of message dicts for persistent logging.
    """
    import sqlite3

    print(f"\n{_BOLD}  {'─' * 56}{_RESET}")
    print(f"{_BOLD}  Full Transcript{_RESET}")
    print(f"{_BOLD}  {'─' * 56}{_RESET}\n")

    db = sqlite3.connect("sales_tracking.db")
    db.row_factory = sqlite3.Row
    msgs = db.execute(
        "SELECT role, content, sales_stage, timestamp FROM messages WHERE call_sid=? ORDER BY id",
        (call_sid,),
    ).fetchall()
    if not msgs:
        warn("No messages recorded")
        db.close()
        return []

    result: list[dict[str, str]] = []
    for m in msgs:
        role_label = "TARGET" if m["role"] == "user" else "AGENT "
        color = _CYAN if m["role"] == "user" else _GREEN
        stage = m["sales_stage"] or "?"
        print(f"  {color}{role_label}{_RESET}  [{stage:12}]  {m['content']}")
        result.append(
            {
                "role": m["role"],
                "stage": stage,
                "content": m["content"],
                "timestamp": m["timestamp"] or "",
            }
        )
    print()

    call = db.execute(
        "SELECT call_sid, start_time, outcome FROM calls WHERE call_sid=?",
        (call_sid,),
    ).fetchone()
    if call:
        label("Call SID:", call["call_sid"])
        label("Started:", str(call["start_time"]))
        label("Outcome:", call["outcome"] or "in progress")
    db.close()
    return result


def save_transcript(
    call_sid: str,
    to_number: str,
    playbook_path: str,
    live_lines: list[str],
    messages: list[dict[str, str]],
) -> Path:
    """Save the full transcript to a persistent text file."""
    _TRANSCRIPT_DIR.mkdir(exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{call_sid[:16]}.txt"
    out_path = _TRANSCRIPT_DIR / filename

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  GhostLine Demo Transcript")
    lines.append("=" * 60)
    lines.append(f"  Date:       {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Call SID:   {call_sid}")
    lines.append(f"  Target:     {to_number}")
    lines.append(f"  Playbook:   {playbook_path}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  LIVE CALL LOG")
    lines.append("-" * 60)
    for ln in live_lines:
        lines.append(_strip_ansi(ln))
    lines.append("")
    lines.append("-" * 60)
    lines.append("  FULL TRANSCRIPT")
    lines.append("-" * 60)
    for msg in messages:
        role_label = "TARGET" if msg["role"] == "user" else "AGENT "
        lines.append(f"  {role_label}  [{msg['stage']:12}]  {msg['content']}")
    lines.append("")
    lines.append("=" * 60)

    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def cleanup(proc: subprocess.Popen) -> None:
    """Kill the server, ngrok, and any lingering processes."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    subprocess.run(["pkill", "-f", "ngrok"], capture_output=True, timeout=5, check=False)
    subprocess.run(["pkill", "-f", "ghostline serve"], capture_output=True, timeout=5, check=False)
    ok("Server and ngrok shut down")


# ── Main ────────────────────────────────────────────────────────────────


# ── Demo presets ──────────────────────────────────────────────────────

# Each preset bundles a playbook, voice, campaign label, and description.
# Voice IDs are ElevenLabs premade voices (full 20-char IDs).
_DEMO_PRESETS: dict[str, dict[str, str]] = {
    "a": {
        "label": "IT Security Patch",
        "playbook": "playbooks/it-test.yaml",
        "campaign": "it-security",
        "voice_id": "iP95p4xoKVk53GoZ742B",  # Chris — Charming, Down-to-Earth
        "description": "Friendly IT support call → guides target to download a 'security update'",
    },
    "b": {
        "label": "Tech Recruiter",
        "playbook": "playbooks/recruiter.yaml",
        "campaign": "recruiter",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",  # George — Warm British Storyteller
        "description": "Excited recruiter → flatters target into clicking an 'application portal'",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GhostLine live demo — orchestrates the full pipeline for screen-sharing."
    )
    parser.add_argument(
        "--to",
        required=True,
        help="Target phone number (E.164, e.g. +15551234567)",
    )
    parser.add_argument(
        "--demo",
        choices=["a", "b"],
        default=None,
        help="Demo preset: a = IT security patch (male), b = recruiter (female)",
    )
    parser.add_argument(
        "--playbook",
        default=None,
        help="Path to playbook YAML (overrides --demo preset)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign label (overrides --demo preset)",
    )
    parser.add_argument(
        "--voice-id",
        default=None,
        help="ElevenLabs voice ID (overrides --demo preset)",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Target's name for personalized greeting (OSINT pretext)",
    )
    args = parser.parse_args()

    # Apply demo preset defaults (explicit flags override)
    preset = _DEMO_PRESETS.get(args.demo or "a", _DEMO_PRESETS["a"])
    if args.playbook is None:
        args.playbook = preset["playbook"]
    if args.campaign is None:
        args.campaign = preset.get("campaign", "demo")
    if args.voice_id is None:
        args.voice_id = preset.get("voice_id")

    print(f"\n{_BOLD}{'═' * 60}{_RESET}")
    print(f"{_BOLD}  GhostLine Live Demo{_RESET}")
    print(f"{_BOLD}  {preset['label']}: {preset['description']}{_RESET}")
    print(f"{_BOLD}{'═' * 60}{_RESET}")

    # Step 1: Environment
    step(1, "Environment Check")
    env = check_env()

    # Step 2: Capability Detection
    step(2, "Model Capability Detection")
    detect_caps(env["model"])

    # Step 3: Voice Selection
    step(3, "Voice Selection")
    if args.voice_id:
        voice_id = args.voice_id
        ok(f"Using voice: {voice_id}")
    else:
        voice_id = list_voices()

    # Step 4: Start Server
    step(4, "Start Server + ngrok Tunnel")
    proc, ws_url = start_server_and_tunnel(voice_id, args.playbook, args.port)
    label("Playbook:", args.playbook)

    # Step 5: Place Call
    step(5, "Place Outbound Call")
    call_sid = place_call(ws_url, args.to, args.campaign, args.target_name)

    # Step 6: Stream Pipeline (full output, no truncation)
    step(6, "Live Call")
    live_lines = stream_pipeline(proc, call_sid)

    # Step 7: Transcript
    step(7, "Full Transcript")
    messages = show_transcript(call_sid)

    # Step 8: Save persistent transcript
    step(8, "Save Transcript")
    if messages:
        out_path = save_transcript(call_sid, args.to, args.playbook, live_lines, messages)
        ok(f"Transcript saved: {out_path}")
        ok(f"Server log:       {_SERVER_LOG}")
    else:
        warn("No messages to save")

    # Cleanup
    step(9, "Cleanup")
    cleanup(proc)

    print(f"\n{_BOLD}  Demo complete.{_RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Interrupted.{_RESET}")
        subprocess.run(
            ["pkill", "-f", "ghostline serve"], capture_output=True, timeout=5, check=False
        )
        subprocess.run(["pkill", "-f", "ngrok"], capture_output=True, timeout=5, check=False)
        sys.exit(1)
