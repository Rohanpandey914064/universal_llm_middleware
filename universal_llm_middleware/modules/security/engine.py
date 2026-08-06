"""
modules/security/engine.py
───────────────────────────
Concrete ``DefaultSecurityEngine`` that composes ``InjectionGuard``,
``PIIAnonymizer``, and ``CanaryGuard`` into the ``BaseSecurityEngine`` ABC.

This is the implementation injected into ``UniversalPipeline`` by default.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from config.settings import get_settings
from core.schemas import (
    CanaryLeakageException,
    SecurityReport,
    ThreatCategory,
    ThreatDetectedException,
)
from modules.security.base import BaseSecurityEngine
from modules.security.canary_guard import CanaryGuard
from modules.security.input_guard import InjectionGuard
from modules.security.pii_anonymizer import PIIAnonymizer

logger = logging.getLogger(__name__)


class DefaultSecurityEngine(BaseSecurityEngine):
    """
    Production-ready security engine composing injection detection, PII
    anonymisation, and canary token management.

    All three subsystems operate independently — there are no shared state
    objects between them.  Per-session PII maps are forwarded to/from the
    ``SessionState`` schema by the pipeline orchestrator.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        spacy_model: Optional[str] = None,
        canary_pattern: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self._guard = InjectionGuard(
            threshold=threshold or settings.injection_threshold,
            onnx_path=settings.onnx_model_path,
        )
        self._anonymizer = PIIAnonymizer(
            spacy_model=spacy_model or settings.spacy_model,
        )
        self._canary = CanaryGuard(
            pattern=canary_pattern or settings.canary_pattern,
        )

        logger.info(
            "DefaultSecurityEngine initialised — "
            "injection_threshold=%.2f spacy_model=%s",
            self._guard.threshold,
            self._anonymizer._spacy_model,
        )

    # ── BaseSecurityEngine implementation ─────────────────────────────────────

    def inspect_input(self, prompt: str) -> Tuple[bool, float, str]:
        """Delegate to ``InjectionGuard``."""
        return self._guard.inspect(prompt)

    def mask_pii(self, text: str, session_id: str) -> str:
        """Delegate to ``PIIAnonymizer.mask``."""
        return self._anonymizer.mask(text, session_id)

    def unmask_pii(self, text: str, session_id: str) -> str:
        """Delegate to ``PIIAnonymizer.unmask``."""
        return self._anonymizer.unmask(text, session_id)

    def inject_canary(self, system_prompt: str) -> Tuple[str, str]:
        """Delegate to ``CanaryGuard.inject``."""
        return self._canary.inject(system_prompt)

    def verify_canary(self, output_text: str, canary_token: str) -> bool:
        """Delegate to ``CanaryGuard.verify``."""
        return self._canary.verify(output_text, canary_token)

    # ── Session PII map helpers (used by pipeline) ────────────────────────────

    def get_session_pii_map(self, session_id: str) -> Dict[str, str]:
        """Return a snapshot of the session's PII map for persistence."""
        return self._anonymizer.get_session_map(session_id)

    def load_session_pii_map(self, session_id: str, pii_map: Dict[str, str]) -> None:
        """Restore a previously saved PII map (called when resuming a session)."""
        self._anonymizer.load_session_map(session_id, pii_map)

    def clear_session(self, session_id: str) -> None:
        """Purge all in-memory PII data for a session."""
        self._anonymizer.clear_session(session_id)
