"""
interfaces/reverse_proxy.py
────────────────────────────
FastAPI OpenAI-compatible reverse proxy gateway.

Exposes:
  POST /v1/chat/completions  — OpenAI-compatible chat endpoint.
  GET  /health               — Health check for load balancer probes.
  GET  /metrics              — Basic operational metrics.

Architecture:
  Each request is routed through ``UniversalPipeline`` before being forwarded
  to the upstream LLM API via an ``httpx.AsyncClient``.  The upstream URL and
  API key are read from ``Settings``.

Error handling:
  • ``ThreatDetectedException`` → HTTP 422 (Unprocessable Entity)
  • ``CanaryLeakageException``  → HTTP 500 (Internal Server Error, no detail)
  • Upstream LLM errors        → HTTP 502 (Bad Gateway)
  • All other errors           → HTTP 500

The ``CanaryLeakageException`` is explicitly mapped to a 500 with a sanitised
message to prevent any internal canary / prompt details from leaking to clients.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from core.pipeline import UniversalPipeline
from core.schemas import (
    CanaryLeakageException,
    ChatMessage,
    MessageRole,
    PipelineRequest,
    ThreatDetectedException,
)
from modules.compression.base import BaseCompressor
from modules.history.base import BaseHistoryManager
from modules.security.base import BaseSecurityEngine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Compatible Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIMessage(BaseModel):
    """Single message in the OpenAI chat completions format."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible POST /v1/chat/completions request body.

    Accepts a ``X-Session-ID`` header or generates a UUID session per request.
    """

    model: str = Field(default="gpt-4o-mini")
    messages: List[OpenAIMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = False
    # Middleware-specific
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for stateful context management.",
    )


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response envelope."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: UsageInfo
    # Middleware metadata (non-standard, consumers may ignore)
    middleware_session_id: Optional[str] = None
    middleware_canary_clean: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Application Factory
# ─────────────────────────────────────────────────────────────────────────────

# Global pipeline and HTTP client — initialised during lifespan
_pipeline: Optional[UniversalPipeline] = None
_http_client: Optional[httpx.AsyncClient] = None

# Operational metrics (simple counters)
_metrics: Dict[str, int] = {
    "requests_total": 0,
    "requests_blocked": 0,
    "canary_leaks": 0,
    "upstream_errors": 0,
}


def _build_pipeline(
    security_engine: Optional[BaseSecurityEngine] = None,
    history_manager: Optional[BaseHistoryManager] = None,
    compressor: Optional[BaseCompressor] = None,
) -> UniversalPipeline:
    return UniversalPipeline(
        security_engine=security_engine,
        history_manager=history_manager,
        compressor=compressor,
    )


def create_app(
    security_engine: Optional[BaseSecurityEngine] = None,
    history_manager: Optional[BaseHistoryManager] = None,
    compressor: Optional[BaseCompressor] = None,
) -> FastAPI:
    """
    FastAPI application factory.

    Args:
        security_engine: Override security engine for testing.
        history_manager: Override history manager.
        compressor:      Override compressor.

    Returns:
        Configured ``FastAPI`` application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _pipeline, _http_client
        settings = get_settings()

        _pipeline = _build_pipeline(security_engine, history_manager, compressor)

        _http_client = httpx.AsyncClient(
            base_url=settings.upstream_llm_url,
            timeout=settings.request_timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.upstream_api_key or ''}",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "Reverse proxy started — upstream=%s", settings.upstream_llm_url
        )
        yield
        await _http_client.aclose()
        logger.info("Reverse proxy shutdown complete.")

    app = FastAPI(
        title="Universal LLM Middleware — Reverse Proxy",
        description=(
            "OpenAI-compatible gateway with integrated security, history "
            "management, and memory compression."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request Logging Middleware ────────────────────────────────────────────

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Endpoints
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/health", tags=["Infrastructure"])
    async def health_check() -> Dict[str, Any]:
        """Liveness / readiness probe for load balancers."""
        return {
            "status": "healthy",
            "pipeline_ready": _pipeline is not None,
            "timestamp": int(time.time()),
        }

    @app.get("/metrics", tags=["Infrastructure"])
    async def metrics() -> Dict[str, Any]:
        """Basic operational counters."""
        settings = get_settings()
        return {
            "uptime_timestamp": int(time.time()),
            "counters": dict(_metrics),
            "config": {
                "injection_threshold": settings.injection_threshold,
                "drift_threshold": settings.drift_threshold,
                "max_history_tokens": settings.max_history_tokens,
                "upstream_url": settings.upstream_llm_url,
            },
        }

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        tags=["Chat"],
        summary="OpenAI-compatible chat completions",
    )
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        OpenAI-compatible ``POST /v1/chat/completions`` endpoint.

        Optionally accepts a ``X-Session-ID`` header to maintain stateful
        context across multiple requests.
        """
        global _pipeline, _http_client, _metrics

        if _pipeline is None or _http_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline not initialised.",
            )

        _metrics["requests_total"] += 1

        # Resolve session ID
        session_id = (
            body.session_id
            or request.headers.get("X-Session-ID")
            or f"anon-{uuid.uuid4().hex[:12]}"
        )

        # Convert incoming messages
        chat_messages = []
        for m in body.messages:
            try:
                chat_messages.append(
                    ChatMessage(
                        role=MessageRole(m.role),
                        content=m.content,
                        name=m.name,
                    )
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid message role: {m.role!r}",
                ) from exc

        pipeline_req = PipelineRequest(
            session_id=session_id,
            messages=chat_messages,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        # ── Stage A: Pre-process ──────────────────────────────────────────────
        try:
            payload = _pipeline.process_request(pipeline_req)
        except ThreatDetectedException as exc:
            _metrics["requests_blocked"] += 1
            logger.warning(
                "Request blocked — session='%s' reason=%s",
                session_id,
                exc.safe_message,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.to_dict(),
            ) from exc
        except Exception as exc:
            logger.exception("Pipeline pre-processing error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "PipelineError", "message": str(exc)},
            ) from exc

        # ── Stage B: Forward to upstream LLM ──────────────────────────────────
        upstream_payload: Dict[str, Any] = {
            "model": payload.model,
            "messages": [m.to_dict() for m in payload.messages],
            "temperature": payload.temperature,
        }
        if payload.max_tokens is not None:
            upstream_payload["max_tokens"] = payload.max_tokens
        upstream_payload.update(payload.extra_params)

        try:
            upstream_resp = await _http_client.post(
                "/chat/completions",
                json=upstream_payload,
            )
            upstream_resp.raise_for_status()
            upstream_json = upstream_resp.json()
        except httpx.HTTPStatusError as exc:
            _metrics["upstream_errors"] += 1
            logger.error(
                "Upstream LLM error: %s %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": "UpstreamError",
                    "message": "Upstream LLM returned an error.",
                    "upstream_status": exc.response.status_code,
                },
            ) from exc
        except httpx.RequestError as exc:
            _metrics["upstream_errors"] += 1
            logger.error("Upstream LLM unreachable: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "error": "UpstreamUnreachable",
                    "message": "Could not connect to upstream LLM.",
                },
            ) from exc

        # ── Stage C: Post-process response ────────────────────────────────────
        raw_content = (
            upstream_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        raw_model = upstream_json.get("model", payload.model)
        raw_usage = upstream_json.get("usage", {})

        try:
            pipeline_resp = _pipeline.process_response(
                raw_content=raw_content,
                payload=payload,
                model=raw_model,
                usage=raw_usage,
            )
        except CanaryLeakageException as exc:
            _metrics["canary_leaks"] += 1
            logger.critical(
                "Canary leakage — session='%s' response suppressed.", session_id
            )
            # Return safe 500 without any internal details
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=exc.to_dict(),
            )
        except Exception as exc:
            logger.exception("Pipeline post-processing error: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "PostProcessingError", "message": str(exc)},
            ) from exc

        return ChatCompletionResponse(
            model=pipeline_resp.model,
            choices=[
                Choice(
                    message=ChoiceMessage(
                        role="assistant",
                        content=pipeline_resp.message.content,
                    )
                )
            ],
            usage=UsageInfo(
                prompt_tokens=pipeline_resp.usage.prompt_tokens,
                completion_tokens=pipeline_resp.usage.completion_tokens,
                total_tokens=pipeline_resp.usage.total_tokens,
            ),
            middleware_session_id=session_id,
            middleware_canary_clean=pipeline_resp.canary_clean,
        )

    # ── Global exception handler ──────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": "An unexpected error occurred."},
        )

    return app
