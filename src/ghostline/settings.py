"""Application settings loaded from environment variables (``.env``).

Replaces the legacy ``keys.py`` pattern of hardcoding secrets in source.
All secrets are validated at construction; missing required values raise a
``SettingsError`` with a clear, actionable message.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ("Settings", "clear_settings_cache", "get_settings")


class Settings(BaseSettings):
    """Runtime configuration for GhostLine.

    All secret-bearing fields are ``SecretStr``; access the underlying value
    via ``.get_secret_value()``. Never log a ``Settings`` instance directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Twilio ───────────────────────────────────────────────────────
    twilio_account_sid: Annotated[str, Field(min_length=34, max_length=34)]
    twilio_auth_token: SecretStr
    twilio_from_number: str

    # ── Deepgram STT ────────────────────────────────────────────────
    deepgram_api_key: SecretStr
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"

    # ── ElevenLabs TTS ──────────────────────────────────────────────
    elevenlabs_api_key: SecretStr
    elevenlabs_model: str = "eleven_multilingual_v2"

    # ── LLM via LiteLLM (model-agnostic) ───────────────────────────
    litellm_model: str = "openai/gpt-4o-mini"
    litellm_api_base: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # ── Tunnel ──────────────────────────────────────────────────────
    ngrok_authtoken: SecretStr

    # ── Piper TTS (local, offline — overrides ElevenLabs when set) ──
    piper_model_path: str | None = None

    # ── Optional runtime knobs ──────────────────────────────────────
    # Field names map 1:1 to env vars (no prefix): LOG_LEVEL, HOST, PORT, etc.
    sqlite_db_path: Path = Path("sales_tracking.db")
    babble_noise_path: Path = Path("ambient_noise.wav")
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("twilio_account_sid")
    @classmethod
    def _validate_sid_prefix(cls, v: str) -> str:
        if not v.startswith("AC"):
            raise ValueError("TWILIO_ACCOUNT_SID must start with 'AC'")
        return v

    @field_validator("twilio_from_number")
    @classmethod
    def _validate_e164(cls, v: str) -> str:
        if not v.startswith("+") or not v[1:].isdigit():
            raise ValueError("TWILIO_FROM_NUMBER must be E.164 format, e.g. '+15551234567'")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}; got {v!r}")
        return upper

    @property
    def ngrok_ws_url_env_var(self) -> str:
        """Name of the env var where the running server publishes its tunnel URL."""
        return "NGROK_WS_URL"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The cache is process-wide; tests should call :func:`clear_settings_cache`
    in a fixture to isolate environments.
    """
    return Settings()  # type: ignore[call-arg]


def clear_settings_cache() -> None:
    """Drop the cached :class:`Settings` so the next :func:`get_settings` re-reads env."""
    get_settings.cache_clear()
