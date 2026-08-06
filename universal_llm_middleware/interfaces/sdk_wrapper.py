"""
interfaces/sdk_wrapper.py
──────────────────────────
``UniversalAIWrapper`` — A transparent drop-in wrapper around native Python
LLM client objects (OpenAI, Groq, Anthropic-like structures).

Usage::

    from openai import OpenAI
    from interfaces.sdk_wrapper import UniversalAIWrapper

    client = UniversalAIWrapper(
        native_client=OpenAI(api_key="sk-…"),
        session_id="user-session-abc",
    )

    # Identical to client.chat.completions.create(...)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

The wrapper intercepts every call, routes it through the ``UniversalPipeline``,
then forwards the sanitised payload to the native client.  The response is
post-processed (canary audit + PII restoration) before being returned.

Both synchronous and asynchronous (``async def``) call paths are supported.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from core.pipeline import UniversalPipeline
from core.schemas import (
    CanaryLeakageException,
    ChatMessage,
    MessageRole,
    PipelineRequest,
    SanitisedPayload,
    ThreatDetectedException,
)
from modules.compression.base import BaseCompressor
from modules.history.base import BaseHistoryManager
from modules.security.base import BaseSecurityEngine

logger = logging.getLogger(__name__)


def _messages_to_chat(raw: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert raw dict messages (from caller) to typed ``ChatMessage`` list."""
    result = []
    for m in raw:
        try:
            result.append(
                ChatMessage(
                    role=MessageRole(m["role"]),
                    content=m.get("content"),
                    name=m.get("name"),
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed message %r: %s", m, exc)
    return result


def _payload_to_dict_messages(payload: SanitisedPayload) -> List[Dict[str, Any]]:
    """Convert sanitised payload messages back to raw dicts for native client."""
    return [m.to_dict() for m in payload.messages]


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper Namespace Mimics
# ─────────────────────────────────────────────────────────────────────────────


class _CompletionsNamespace:
    """
    Mimics ``client.chat.completions`` so callers can use the standard
    ``client.chat.completions.create(...)`` API surface.
    """

    def __init__(self, wrapper: "UniversalAIWrapper") -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        """
        Synchronous ``chat.completions.create`` interceptor.

        Accepts all standard OpenAI parameters.  Routes through the pipeline
        before forwarding to the native client.
        """
        return self._wrapper._sync_create(**kwargs)

    async def acreate(self, **kwargs: Any) -> Any:
        """
        Asynchronous ``chat.completions.acreate`` interceptor.
        """
        return await self._wrapper._async_create(**kwargs)


class _ChatNamespace:
    """Mimics ``client.chat``."""

    def __init__(self, wrapper: "UniversalAIWrapper") -> None:
        self.completions = _CompletionsNamespace(wrapper)


# ─────────────────────────────────────────────────────────────────────────────
# UniversalAIWrapper
# ─────────────────────────────────────────────────────────────────────────────


class UniversalAIWrapper:
    """
    Universal middleware wrapper for native LLM Python clients.

    Args:
        native_client:   An instantiated LLM client (OpenAI, Groq, Anthropic,
                         or any object with ``chat.completions.create``).
        session_id:      Session identifier.  All calls through this wrapper
                         instance share the same session context.
        pipeline:        Optional pre-constructed ``UniversalPipeline``.
                         Constructed with defaults when not provided.
        security_engine: Override the security engine in the pipeline.
        history_manager: Override the history manager in the pipeline.
        compressor:      Override the compression engine in the pipeline.
    """

    def __init__(
        self,
        native_client: Any,
        session_id: str,
        pipeline: Optional[UniversalPipeline] = None,
        security_engine: Optional[BaseSecurityEngine] = None,
        history_manager: Optional[BaseHistoryManager] = None,
        compressor: Optional[BaseCompressor] = None,
    ) -> None:
        self._client = native_client
        self._session_id = session_id
        self._pipeline = pipeline or UniversalPipeline(
            security_engine=security_engine,
            history_manager=history_manager,
            compressor=compressor,
        )
        self.chat = _ChatNamespace(self)
        logger.info(
            "UniversalAIWrapper initialised — session='%s' client=%s",
            session_id,
            type(native_client).__name__,
        )

    # ── Sync Path ─────────────────────────────────────────────────────────────

    def _sync_create(self, **kwargs: Any) -> Any:
        """
        Full synchronous pipeline: request processing → native LLM call →
        response post-processing.
        """
        messages = kwargs.pop("messages", [])
        model = kwargs.pop("model", "gpt-4o-mini")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", None)

        # Build typed pipeline request
        chat_messages = _messages_to_chat(messages)
        pipeline_req = PipelineRequest(
            session_id=self._session_id,
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )

        # Pre-process through pipeline
        try:
            payload = self._pipeline.process_request(pipeline_req)
        except ThreatDetectedException as exc:
            logger.error(
                "SDK wrapper: threat blocked for session '%s': %s",
                self._session_id,
                exc.safe_message,
            )
            raise

        # Forward to native client
        native_kwargs: Dict[str, Any] = {
            "model": payload.model,
            "messages": _payload_to_dict_messages(payload),
            "temperature": payload.temperature,
            **payload.extra_params,
        }
        if payload.max_tokens is not None:
            native_kwargs["max_tokens"] = payload.max_tokens

        logger.debug(
            "SDK wrapper: forwarding to native client — model=%s messages=%d",
            payload.model,
            len(payload.messages),
        )
        raw_response = self._client.chat.completions.create(**native_kwargs)

        # Post-process response
        try:
            raw_content = raw_response.choices[0].message.content or ""
            raw_model = getattr(raw_response, "model", payload.model)
            raw_usage = {}
            if hasattr(raw_response, "usage") and raw_response.usage:
                u = raw_response.usage
                raw_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0),
                }

            pipeline_response = self._pipeline.process_response(
                raw_content=raw_content,
                payload=payload,
                model=raw_model,
                usage=raw_usage,
            )
        except CanaryLeakageException as exc:
            logger.critical(
                "SDK wrapper: canary leakage for session '%s' — response suppressed.",
                self._session_id,
            )
            raise

        # Patch the native response object with the restored content
        try:
            raw_response.choices[0].message.content = pipeline_response.message.content
        except AttributeError:
            pass  # Read-only response objects — just return as-is

        return raw_response

    # ── Async Path ────────────────────────────────────────────────────────────

    async def _async_create(self, **kwargs: Any) -> Any:
        """
        Full asynchronous pipeline.  Requires the native client to support
        ``await client.chat.completions.create(...)`` or have an ``acreate``
        method.
        """
        messages = kwargs.pop("messages", [])
        model = kwargs.pop("model", "gpt-4o-mini")
        temperature = kwargs.pop("temperature", 0.7)
        max_tokens = kwargs.pop("max_tokens", None)

        chat_messages = _messages_to_chat(messages)
        pipeline_req = PipelineRequest(
            session_id=self._session_id,
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=kwargs,
        )

        try:
            payload = self._pipeline.process_request(pipeline_req)
        except ThreatDetectedException:
            raise

        native_kwargs: Dict[str, Any] = {
            "model": payload.model,
            "messages": _payload_to_dict_messages(payload),
            "temperature": payload.temperature,
            **payload.extra_params,
        }
        if payload.max_tokens is not None:
            native_kwargs["max_tokens"] = payload.max_tokens

        # Support both async and sync native clients
        create_fn = getattr(
            self._client.chat.completions,
            "acreate",
            self._client.chat.completions.create,
        )
        import inspect

        if inspect.iscoroutinefunction(create_fn):
            raw_response = await create_fn(**native_kwargs)
        else:
            raw_response = create_fn(**native_kwargs)

        try:
            raw_content = raw_response.choices[0].message.content or ""
            raw_model = getattr(raw_response, "model", payload.model)
            raw_usage = {}
            if hasattr(raw_response, "usage") and raw_response.usage:
                u = raw_response.usage
                raw_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0),
                }
            self._pipeline.process_response(
                raw_content=raw_content,
                payload=payload,
                model=raw_model,
                usage=raw_usage,
            )
        except CanaryLeakageException:
            raise

        return raw_response
