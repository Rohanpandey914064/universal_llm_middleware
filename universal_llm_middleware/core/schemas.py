"""
core/schemas.py
───────────────
Pydantic v2 data models that define the strict, typed contracts between every
layer of the universal_llm_middleware.  No external module may bypass these
schemas when exchanging data across engine boundaries.

Design principles:
  • ``model_config = ConfigDict(frozen=True)`` on immutable transfer objects.
  • ``extra="forbid"`` on all models to detect accidental field leakage early.
  • Custom exception types carry enough structured data for upstream logging
    without exposing internal implementation details.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Valid roles recognised by the middleware and OpenAI-compatible APIs."""

    SYSTEM = "system"
    DEVELOPER = "developer"  # Anthropic/OpenAI o-series extension
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class ThreatCategory(str, Enum):
    """Classification labels returned by the Security Engine."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    PII_EXPOSURE = "pii_exposure"
    CANARY_LEAKAGE = "canary_leakage"
    CLEAN = "clean"


# ─────────────────────────────────────────────────────────────────────────────
# Core Message Model
# ─────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """
    A single conversational turn compatible with the OpenAI Chat Completions API.

    Attributes:
        role:    Speaker role (system, user, assistant, …).
        content: Message body text.  May be ``None`` for tool-call-only turns.
        name:    Optional speaker name hint used by some providers.
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content")
    @classmethod
    def _content_not_empty_string(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() == "":
            raise ValueError("content must be non-empty when provided.")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation suitable for LLM API calls."""
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content or ""}
        if self.name:
            d["name"] = self.name
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Security Engine Schemas
# ─────────────────────────────────────────────────────────────────────────────


class SecurityReport(BaseModel):
    """
    Structured output from the Security Engine's ``inspect_input`` method.

    Attributes:
        is_safe:      ``True`` when the input passes all security checks.
        threat_score: Continuous score in [0.0, 1.0]; higher → more dangerous.
        category:     Enum classification of the detected (or absent) threat.
        reason:       Human-readable justification string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_safe: bool
    threat_score: float = Field(ge=0.0, le=1.0)
    category: ThreatCategory = ThreatCategory.CLEAN
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Compression Engine Schemas
# ─────────────────────────────────────────────────────────────────────────────


class CompressionResult(BaseModel):
    """
    Output of a ``BaseCompressor.compress()`` call.

    Attributes:
        messages:          The compressed list of conversational turns.
        compression_ratio: Ratio of tokens kept (0.0 – 1.0).
                           ``1.0`` means no compression was applied.
        drift_score:       Cosine similarity between original and compressed
                           text.  ``None`` if drift validation was skipped.
    """

    model_config = ConfigDict(extra="forbid")

    messages: List[ChatMessage]
    compression_ratio: float = Field(ge=0.0, le=1.0)
    drift_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Session / History Schemas
# ─────────────────────────────────────────────────────────────────────────────


class SessionState(BaseModel):
    """
    In-memory representation of a single user session managed by the
    History Engine.

    Attributes:
        session_id:          Unique identifier for the session.
        immutable_messages:  System / developer directives — never compressed
                             or altered by the pipeline.
        history_messages:    Mutable conversational turns subject to compression
                             and PII masking.
        canary_token:        Active canary token appended to the system prompt
                             for this request.  Cleared after verification.
        pii_map:             Ephemeral mapping ``{ placeholder → original }``
                             used to restore PII-masked values in responses.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    immutable_messages: List[ChatMessage] = Field(default_factory=list)
    history_messages: List[ChatMessage] = Field(default_factory=list)
    canary_token: Optional[str] = None
    pii_map: Dict[str, str] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline I/O Schemas
# ─────────────────────────────────────────────────────────────────────────────


class PipelineRequest(BaseModel):
    """
    Input contract for ``UniversalPipeline.process_request()``.

    Attributes:
        session_id:  Caller-supplied session identifier.
        messages:    Full message list including system prompts and history.
        model:       Target LLM model name.
        temperature: Sampling temperature forwarded verbatim to the LLM.
        max_tokens:  Maximum tokens the LLM may generate.
        extra_params: Any additional provider-specific parameters.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    messages: List[ChatMessage] = Field(min_length=1)
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token consumption counters mirroring the OpenAI usage object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class PipelineResponse(BaseModel):
    """
    Output contract for ``UniversalPipeline.process_response()``.

    Attributes:
        session_id:   Echo of the originating session identifier.
        message:      The final, PII-restored assistant message.
        model:        Model name reported by the upstream provider.
        usage:        Token consumption counters.
        canary_clean: ``True`` if no canary leakage was detected in the response.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: ChatMessage
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    canary_clean: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Sanitised Payload (between pipeline and LLM call)
# ─────────────────────────────────────────────────────────────────────────────


class SanitisedPayload(BaseModel):
    """
    The fully processed message list ready to be forwarded to the upstream LLM.
    Produced by ``UniversalPipeline`` and consumed by SDK wrapper / proxy.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages: List[ChatMessage]
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    active_canary: Optional[str] = None  # kept for post-response audit


# ─────────────────────────────────────────────────────────────────────────────
# Structured Exception Types
# ─────────────────────────────────────────────────────────────────────────────


class MiddlewareBaseException(Exception):
    """Root exception for all middleware-generated errors."""

    def __init__(self, message: str, *, safe_message: str = "") -> None:
        super().__init__(message)
        # ``safe_message`` is the version safe to surface to callers without
        # leaking internal pipeline state.
        self.safe_message: str = safe_message or "An internal security check failed."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.safe_message,
        }


class ThreatDetectedException(MiddlewareBaseException):
    """
    Raised when ``SecurityEngine.inspect_input()`` classifies a prompt as
    unsafe.  The ``report`` attribute carries full diagnostic detail for
    internal audit logging only — never forward it to the client.
    """

    def __init__(self, report: SecurityReport) -> None:
        super().__init__(
            message=(
                f"Threat detected: category={report.category.value}, "
                f"score={report.threat_score:.3f}, reason={report.reason}"
            ),
            safe_message=(
                "Your request was blocked by the content safety filter. "
                "Please rephrase and try again."
            ),
        )
        self.report: SecurityReport = report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "ThreatDetectedException",
            "message": self.safe_message,
            "category": self.report.category.value,
            "threat_score": round(self.report.threat_score, 4),
        }


class CanaryLeakageException(MiddlewareBaseException):
    """
    Raised when the LLM output contains the injected canary token, indicating
    that the system prompt was leaked in the model's response.
    """

    def __init__(self, canary_token: str, *, session_id: str = "") -> None:
        super().__init__(
            message=(
                f"Canary leakage detected: token={canary_token!r}, "
                f"session={session_id!r}"
            ),
            safe_message=(
                "A system integrity violation was detected. "
                "The response has been suppressed."
            ),
        )
        self.canary_token: str = canary_token
        self.session_id: str = session_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "CanaryLeakageException",
            "message": self.safe_message,
            "session_id": self.session_id,
        }


class PIIProcessingException(MiddlewareBaseException):
    """Raised when PII masking or unmasking encounters an unrecoverable error."""


class CompressionException(MiddlewareBaseException):
    """Raised when the compression engine fails to produce a valid result."""


class SessionNotFoundException(MiddlewareBaseException):
    """Raised when a requested session_id has no associated state."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session not found: {session_id!r}",
            safe_message="Session expired or not found. Please start a new session.",
        )
        self.session_id = session_id
