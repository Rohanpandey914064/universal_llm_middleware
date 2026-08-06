"""
config/settings.py
──────────────────
Centralised, environment-driven configuration for universal_llm_middleware.

All values can be overridden via environment variables or a `.env` file placed
at the repository root.  Use ``pydantic-settings`` v2 for automatic parsing,
type coercion, and validation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Priority order (highest → lowest):
        1. Environment variables
        2. .env file
        3. Field defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root logger level.",
    )

    # ── Security Engine ───────────────────────────────────────────────────────
    injection_threshold: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
        description="Minimum threat score that triggers injection blocking.",
    )
    onnx_model_path: Path | None = Field(
        default=None,
        description=(
            "Path to a local ONNX model file for injection detection. "
            "If None, the engine falls back to heuristic-only mode."
        ),
    )
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model name used for NER-based PII extraction.",
    )
    canary_pattern: str = Field(
        default="[[CANARY-{token}]]",
        description=(
            "Format string used to wrap canary UUID tokens. "
            "Must contain the '{token}' placeholder."
        ),
    )

    # ── History Engine ────────────────────────────────────────────────────────
    session_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Time-to-live in seconds for inactive sessions.",
    )
    max_history_turns: int = Field(
        default=50,
        ge=1,
        description="Maximum number of conversational turns kept per session.",
    )

    # ── Compression Engine ────────────────────────────────────────────────────
    max_history_tokens: int = Field(
        default=3000,
        ge=100,
        description="Target token budget for compressed conversational history.",
    )
    drift_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity between original and compressed text. "
            "Values below this threshold trigger a drift warning."
        ),
    )

    # ── Upstream LLM Gateway ─────────────────────────────────────────────────
    upstream_llm_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the upstream LLM API.",
    )
    upstream_api_key: str | None = Field(
        default=None,
        description="API key forwarded to the upstream LLM provider.",
    )
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Fallback model name when the caller does not specify one.",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="HTTP timeout for upstream LLM requests.",
    )

    # ── FastAPI Reverse Proxy ─────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Bind host for uvicorn.")
    port: int = Field(default=8080, ge=1, le=65535, description="Bind port.")
    workers: int = Field(default=1, ge=1, description="Number of uvicorn workers.")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("canary_pattern")
    @classmethod
    def _canary_must_contain_placeholder(cls, v: str) -> str:
        if "{token}" not in v:
            raise ValueError(
                "canary_pattern must contain the '{token}' placeholder."
            )
        return v

    @field_validator("onnx_model_path", mode="before")
    @classmethod
    def _resolve_onnx_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        p = Path(v)
        if not p.exists():
            logging.getLogger(__name__).warning(
                "ONNX model path '%s' does not exist; "
                "falling back to heuristic injection detection.",
                p,
            )
            return None
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton ``Settings`` instance.

    Using ``lru_cache`` ensures the .env file is parsed only once per process,
    which is important for performance and for deterministic behaviour in tests
    (override with ``get_settings.cache_clear()`` if needed).
    """
    return Settings()
