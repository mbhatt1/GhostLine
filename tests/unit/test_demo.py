"""Tests for the demo script helper functions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_mock
from scripts.demo import (
    _format_log_line,
    check_env,
    cleanup,
    detect_caps,
    label,
    list_voices,
    ok,
    place_call,
    show_transcript,
    start_server_and_tunnel,
    step,
)


class TestOutputHelpers:
    def test_ok_prints_checkmark(self, capsys: pytest.CaptureFixture[str]) -> None:
        ok("test message")
        out = capsys.readouterr().out
        assert "✓" in out
        assert "test message" in out

    def test_label_prints_key_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        label("Key:", "value")
        out = capsys.readouterr().out
        assert "Key:" in out
        assert "value" in out

    def test_step_prints_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        step(1, "Test Title")
        out = capsys.readouterr().out
        assert "Step 1" in out
        assert "Test Title" in out


class TestFormatLogLine:
    def test_processing_utterance_formats_target_arrow(self) -> None:
        result = _format_log_line("Processing utterance for CA123: 'Hello there'")
        assert result is not None
        assert "TARGET" in result
        assert "Hello there" in result

    def test_reply_generated_formats_agent_arrow(self) -> None:
        result = _format_log_line("Reply generated for CA123: 'Hi! How are you?' (trigger=x)")
        assert result is not None
        assert "AGENT" in result
        assert "Hi!" in result

    def test_call_started_formats_connected(self) -> None:
        result = _format_log_line("Call started: sid=CA123 stream=MZ456")
        assert result is not None
        assert "CALL CONNECTED" in result

    def test_stop_event_formats_ended(self) -> None:
        result = _format_log_line("Stop event for CA123")
        assert result is not None
        assert "CALL ENDED" in result

    def test_uninteresting_line_returns_none(self) -> None:
        result = _format_log_line("DEBUG: something irrelevant")
        assert result is None

    def test_deepgram_connected(self) -> None:
        result = _format_log_line("Deepgram WSS connected to wss://api.deepgram.com/...")
        assert result is not None
        assert "STT stream connected" in result

    def test_modality_text(self) -> None:
        result = _format_log_line("CallSession voice modality: text (needs_deepgram=True)")
        assert result is not None
        assert "text pipeline (STT + TTS)" in result

    def test_modality_realtime(self) -> None:
        result = _format_log_line("CallSession voice modality: realtime (needs_deepgram=False)")
        assert result is not None
        assert "realtime" in result


class TestCheckEnv:
    def test_returns_model_when_all_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        # Mock load_dotenv so it doesn't load the real .env
        mocker.patch("scripts.demo.load_dotenv")
        env = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "token123",
            "TWILIO_FROM_NUMBER": "+15551234567",
            "DEEPGRAM_API_KEY": "dg_key",
            "ELEVENLABS_API_KEY": "el_key",
            "NGROK_AUTHTOKEN": "ng_key",
            "OPENROUTER_API_KEY": "or_key",
            "LITELLM_MODEL": "openrouter/openai/gpt-4o",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        result = check_env()
        assert result["model"] == "openrouter/openai/gpt-4o"
        assert result["llm_key"] == "OPENROUTER_API_KEY"

    def test_exits_on_missing_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("scripts.demo.load_dotenv")
        for k in [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_FROM_NUMBER",
            "ELEVENLABS_API_KEY",
            "NGROK_AUTHTOKEN",
        ]:
            monkeypatch.setenv(k, "x")
        monkeypatch.setenv("LITELLM_MODEL", "openrouter/openai/gpt-4o")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or_key")
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            check_env()

    def test_exits_on_missing_llm_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        mocker.patch("scripts.demo.load_dotenv")
        for k in [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_FROM_NUMBER",
            "DEEPGRAM_API_KEY",
            "ELEVENLABS_API_KEY",
            "NGROK_AUTHTOKEN",
        ]:
            monkeypatch.setenv(k, "x" if "SID" not in k else "AC" + "x" * 32)
        monkeypatch.setenv("LITELLM_MODEL", "openrouter/openai/gpt-4o")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(SystemExit):
            check_env()


class TestDetectCaps:
    def test_detects_text_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        detect_caps("openai/gpt-4o-mini")
        out = capsys.readouterr().out
        assert "text" in out
        assert "Deepgram" in out


class TestListVoices:
    def test_returns_first_voice_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {
            "voices": [
                {"voice_id": "vid1", "name": "Voice1"},
                {"voice_id": "vid2", "name": "Voice2"},
            ]
        }
        mocker.patch("httpx.get", return_value=mock_response)
        result = list_voices()
        assert result == "vid1"

    def test_exits_on_no_voices(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"voices": []}
        mocker.patch("httpx.get", return_value=mock_response)
        with pytest.raises(SystemExit):
            list_voices()


class TestPlaceCall:
    def test_returns_call_sid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC" + "a" * 32)
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
        monkeypatch.setenv("TWILIO_FROM_NUMBER", "+15551234567")
        mock_client = mocker.patch("twilio.rest.Client")
        mock_call = mocker.MagicMock()
        mock_call.sid = "CA_test_sid"
        mock_client.return_value.calls.create.return_value = mock_call
        result = place_call("wss://tunnel.ngrok-free.app/twilio", "+15551234567", "demo")
        assert result == "CA_test_sid"
        mock_client.return_value.calls.create.assert_called_once()


class TestShowTranscript:
    def test_prints_messages_from_db(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import sqlite3

        db_path = tmp_path / "test.db"
        db = sqlite3.connect(str(db_path))
        db.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, call_sid TEXT, role TEXT, content TEXT, sales_stage TEXT, timestamp TEXT)"
        )
        db.execute("CREATE TABLE calls (call_sid TEXT PRIMARY KEY, start_time TEXT, outcome TEXT)")
        db.execute(
            "INSERT INTO messages VALUES (1, 'CA_test', 'user', 'Hello?', 'RAPPORT', '2025-01-01')"
        )
        db.execute(
            "INSERT INTO messages VALUES (2, 'CA_test', 'assistant', 'Hi there!', 'RAPPORT', '2025-01-01')"
        )
        db.execute("INSERT INTO calls VALUES ('CA_test', '2025-01-01', 'compromised')")
        db.commit()
        db.close()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            (tmp_path / "test.db").rename(tmp_path / "sales_tracking.db")
            show_transcript("CA_test")
            out = capsys.readouterr().out
            assert "Hello?" in out
            assert "Hi there!" in out
            assert "RAPPORT" in out
            assert "compromised" in out
        finally:
            os.chdir(old_cwd)

    def test_warns_on_no_messages(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import sqlite3

        db_path = tmp_path / "sales_tracking.db"
        db = sqlite3.connect(str(db_path))
        db.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, call_sid TEXT, role TEXT, content TEXT, sales_stage TEXT, timestamp TEXT)"
        )
        db.execute("CREATE TABLE calls (call_sid TEXT PRIMARY KEY, start_time TEXT, outcome TEXT)")
        db.commit()
        db.close()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            show_transcript("CA_nonexistent")
            out = capsys.readouterr().out
            assert "No messages" in out
        finally:
            os.chdir(old_cwd)


class TestCleanup:
    def test_terminates_process(self) -> None:
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        cleanup(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_kills_on_timeout(self) -> None:
        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        cleanup(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestStartServer:
    def test_builds_correct_command(self, mocker: pytest_mock.MockerFixture) -> None:
        mock_popen = mocker.patch("scripts.demo.subprocess.Popen")
        mock_proc = mock_popen.return_value
        mock_proc.poll.return_value = None
        mocker.patch("builtins.open", return_value=MagicMock())
        mock_path = mocker.patch("scripts.demo.Path")
        mock_path.return_value.read_text.return_value = (
            "ngrok tunnel: wss://abc.ngrok-free.app/twilio\nApplication startup complete\n"
        )
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.stat.return_value.st_size = 0
        proc, ws_url = start_server_and_tunnel("vid123", "playbooks/it-test.yaml", 8000)
        assert proc is mock_proc
        assert ws_url == "wss://abc.ngrok-free.app/twilio"
        cmd = mock_popen.call_args.args[0]
        assert "uv" in cmd
        assert "ghostline" in cmd
        assert "serve" in cmd
        assert "--voice-id" in cmd
        assert "vid123" in cmd
        assert "--playbook" in cmd

    def test_exits_on_missing_url(self, mocker: pytest_mock.MockerFixture) -> None:
        mock_popen = mocker.patch("scripts.demo.subprocess.Popen")
        mock_proc = mock_popen.return_value
        mock_proc.poll.return_value = None
        mocker.patch("builtins.open", return_value=MagicMock())
        mock_path = mocker.patch("scripts.demo.Path")
        mock_path.return_value.read_text.return_value = "Application startup complete\n"
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.stat.return_value.st_size = 0
        with pytest.raises(SystemExit):
            start_server_and_tunnel("vid", None, 8000)
