"""
core/pipeline.py
─────────────────
``UniversalPipeline`` — The central orchestrator that connects the three
independent engines into a complete, ordered processing pipeline.

Pipeline stages (per request):
  1. Load / initialise session state.
  2. Zone split: separate immutable directives from mutable history.
  3. Security — inject into dynamic content only:
     a. Inspect latest user message for injection / jailbreak.
     b. Mask PII in all mutable history messages.
  4. Compress mutable history to fit within token budget.
  5. Inject canary token into the immutable system prompt.
  6. Reconstruct: [immutable + canary] + [compressed history].
  7. Return ``SanitisedPayload`` to the caller for forwarding to LLM.

Post-response stages:
  8.  Verify canary — raise ``CanaryLeakageException`` on detection.
  9.  Unmask PII placeholders in the assistant reply.
  10. Persist updated session history.

Zero-coupling invariants enforced here:
  • Security engine methods are NEVER called on immutable messages.
  • Compression engine NEVER receives immutable messages.
  • Security engine is resolved via its ABC — no direct concrete import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from core.schemas import (
    CanaryLeakageException,
    ChatMessage,
    CompressionResult,
    MessageRole,
    PipelineRequest,
    PipelineResponse,
    SanitisedPayload,
    SecurityReport,
    SessionState,
    ThreatDetectedException,
    TokenUsage,
)
from modules.compression.base import BaseCompressor
from modules.compression.custom_engine import SlidingWindowCompressor
from modules.history.base import BaseHistoryManager
from modules.history.manager import DefaultHistoryManager
from modules.security.base import BaseSecurityEngine
from modules.security.engine import DefaultSecurityEngine

logger = logging.getLogger(__name__)


class UniversalPipeline:
    """
    Orchestrates the Security, History, and Compression engines into a single,
    ordered processing pipeline.

    All three engines are injected via constructor parameters, making the
    pipeline fully testable with mock implementations and swappable without
    any refactoring.

    Args:
        security_engine:  ``BaseSecurityEngine`` implementation.
        history_manager:  ``BaseHistoryManager`` implementation.
        compressor:       ``BaseCompressor`` implementation.
        drift_threshold:  Minimum compression drift score before warning.
    """

    def __init__(
        self,
        security_engine: Optional[BaseSecurityEngine] = None,
        history_manager: Optional[BaseHistoryManager] = None,
        compressor: Optional[BaseCompressor] = None,
        drift_threshold: Optional[float] = None,
    ) -> None:
        settings = get_settings()

        self._security: BaseSecurityEngine = security_engine or DefaultSecurityEngine()
        self._history: BaseHistoryManager = history_manager or DefaultHistoryManager()
        self._compressor: BaseCompressor = compressor or SlidingWindowCompressor()
        self._drift_threshold: float = (
            drift_threshold
            if drift_threshold is not None
            else settings.drift_threshold
        )

        logger.info(
            "UniversalPipeline initialised — "
            "security=%s history=%s compressor=%s",
            type(self._security).__name__,
            type(self._history).__name__,
            type(self._compressor).__name__,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase A: Pre-LLM Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_request(self, request: PipelineRequest) -> SanitisedPayload:
        """
        Transform a raw ``PipelineRequest`` into a sanitised, compressed payload
        ready for forwarding to the upstream LLM.

        Args:
            request: Validated incoming request from SDK wrapper or proxy.

        Returns:
            ``SanitisedPayload`` with security-cleaned, compressed messages and
            the active canary token embedded for post-response auditing.

        Raises:
            ``ThreatDetectedException``: If the latest user message fails
                                         injection detection.
            ``PIIProcessingException``:  On unrecoverable PII masking errors.
            ``CompressionException``:    On compressor failures.
        """
        session_id = request.session_id
        logger.info("Pipeline request started — session='%s'", session_id)

        # ── Stage 1: Session initialisation ──────────────────────────────────
        session = self._history.get_session(session_id)
        if session is None:
            session = SessionState(session_id=session_id)
            logger.debug("New session created for '%s'.", session_id)

        # Restore PII map into the security engine for this session
        if hasattr(self._security, "load_session_pii_map"):
            self._security.load_session_pii_map(session_id, session.pii_map)  # type: ignore[attr-defined]

        # ── Stage 2: Zone split ───────────────────────────────────────────────
        immutable_msgs, history_msgs = self._history.split_immutable_zone(
            request.messages
        )
        logger.debug(
            "Zone split: %d immutable, %d history messages.",
            len(immutable_msgs),
            len(history_msgs),
        )

        # ── Stage 3a: Injection detection (user content only) ─────────────────
        user_messages = [m for m in history_msgs if m.role == MessageRole.USER]
        if user_messages:
            latest_user = user_messages[-1]
            report = self._security.full_inspect(latest_user.content or "")
            logger.debug(
                "Injection scan — session='%s' safe=%s score=%.3f",
                session_id,
                report.is_safe,
                report.threat_score,
            )
            if not report.is_safe:
                logger.warning(
                    "THREAT DETECTED — session='%s' category=%s score=%.3f reason=%s",
                    session_id,
                    report.category.value,
                    report.threat_score,
                    report.reason,
                )
                raise ThreatDetectedException(report)

        # ── Stage 3b: PII masking (history only — NOT immutable) ──────────────
        masked_history = [
            ChatMessage(
                role=msg.role,
                content=self._security.mask_pii(msg.content or "", session_id),
                name=msg.name,
            )
            for msg in history_msgs
        ]

        # ── Stage 4: Compression ──────────────────────────────────────────────
        compressed_msgs, comp_ratio, drift_score = (
            self._compressor.compress_and_validate(
                masked_history,
                drift_threshold=self._drift_threshold,
            )
        )
        logger.debug(
            "Compression — ratio=%.3f drift=%.4f turns=%d→%d",
            comp_ratio,
            drift_score,
            len(masked_history),
            len(compressed_msgs),
        )

        # ── Stage 5: Canary injection (into immutable system prompt) ──────────
        canary_token: Optional[str] = None
        final_immutable = list(immutable_msgs)

        if final_immutable:
            # Inject canary into the first system / developer message
            first_system = final_immutable[0]
            modified_content, canary_token = self._security.inject_canary(
                first_system.content or ""
            )
            final_immutable[0] = ChatMessage(
                role=first_system.role,
                content=modified_content,
                name=first_system.name,
            )
        else:
            # No system prompt present — create a minimal one with just the canary
            canary_content, canary_token = self._security.inject_canary("")
            final_immutable = [
                ChatMessage(role=MessageRole.SYSTEM, content=canary_content)
            ]

        # ── Stage 6: Reconstruct payload ──────────────────────────────────────
        sanitised_messages = final_immutable + list(compressed_msgs)

        # ── Stage 7: Persist session state ───────────────────────────────────
        pii_map = {}
        if hasattr(self._security, "get_session_pii_map"):
            pii_map = self._security.get_session_pii_map(session_id)  # type: ignore[attr-defined]

        updated_session = SessionState(
            session_id=session_id,
            immutable_messages=immutable_msgs,
            history_messages=masked_history,
            canary_token=canary_token,
            pii_map=pii_map,
        )
        self._history.upsert_session(updated_session)

        logger.info(
            "Pipeline request complete — session='%s' total_messages=%d canary_set=%s",
            session_id,
            len(sanitised_messages),
            canary_token is not None,
        )

        return SanitisedPayload(
            session_id=session_id,
            messages=sanitised_messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_params=request.extra_params,
            active_canary=canary_token,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase B: Post-LLM Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_response(
        self,
        raw_content: str,
        payload: SanitisedPayload,
        model: str = "",
        usage: Optional[Dict[str, int]] = None,
    ) -> PipelineResponse:
        """
        Post-process the LLM's raw response: verify canary, unmask PII, and
        append the assistant turn to session history.

        Args:
            raw_content: The raw text content of the LLM's assistant message.
            payload:     The ``SanitisedPayload`` returned by ``process_request``.
            model:       Model name reported by the upstream provider.
            usage:       Token usage dict from the upstream response.

        Returns:
            ``PipelineResponse`` with cleaned, PII-restored assistant content.

        Raises:
            ``CanaryLeakageException``: If the canary token is present in the
                                        LLM output, indicating system-prompt leakage.
        """
        session_id = payload.session_id
        logger.info("Pipeline response processing — session='%s'", session_id)

        # ── Stage 8: Canary verification ──────────────────────────────────────
        canary_clean = True
        if payload.active_canary:
            leaked = self._security.verify_canary(raw_content, payload.active_canary)
            if leaked:
                logger.critical(
                    "CANARY LEAKAGE — session='%s' canary='%s'",
                    session_id,
                    payload.active_canary,
                )
                # Clear canary from session before raising (defence in depth)
                session = self._history.get_session(session_id)
                if session:
                    self._history.upsert_session(
                        session.model_copy(update={"canary_token": None})
                    )
                raise CanaryLeakageException(
                    canary_token=payload.active_canary,
                    session_id=session_id,
                )
            canary_clean = True

        # ── Stage 9: PII unmasking ────────────────────────────────────────────
        restored_content = self._security.unmask_pii(raw_content, session_id)

        # ── Stage 10: Persist assistant turn to session ───────────────────────
        try:
            self._history.add_turn(session_id, "assistant", restored_content)
        except Exception as exc:
            # Non-fatal — session may have been evicted between stages
            logger.warning(
                "Could not persist assistant turn for session '%s': %s",
                session_id,
                exc,
            )

        # Clear canary from session state after successful verification
        session = self._history.get_session(session_id)
        if session and session.canary_token:
            self._history.upsert_session(
                session.model_copy(update={"canary_token": None})
            )

        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=restored_content,
        )

        token_usage = TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0) if usage else 0,
            completion_tokens=usage.get("completion_tokens", 0) if usage else 0,
            total_tokens=usage.get("total_tokens", 0) if usage else 0,
        )

        logger.info(
            "Pipeline response complete — session='%s' canary_clean=%s",
            session_id,
            canary_clean,
        )

        return PipelineResponse(
            session_id=session_id,
            message=assistant_message,
            model=model or payload.model,
            usage=token_usage,
            canary_clean=canary_clean,
        )
