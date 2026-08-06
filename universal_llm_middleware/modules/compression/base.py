"""
modules/compression/base.py
────────────────────────────
Abstract Base Class for the Memory Compression Engine.

ZERO-COUPLING CONTRACT:
  This module imports ONLY from:
    • Python standard library
    • ``core.schemas``
  It MUST NOT import from:
    • ``modules.security``
    • ``modules.history``

IMMUTABILITY GUARANTEE:
  Compressors MUST only receive and operate on the mutable history segment
  of a conversation.  System / developer directives (immutable zone) are
  NEVER passed to a compressor — the pipeline enforces this invariant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from core.schemas import ChatMessage


class BaseCompressor(ABC):
    """
    Pluggable interface for context compression algorithms.

    Any custom compression strategy can be integrated into the pipeline by:
      1. Subclassing ``BaseCompressor``.
      2. Implementing the ``compress`` method.
      3. Injecting the instance into ``UniversalPipeline`` at construction time.

    No other pipeline code needs to change — this is the only contract the
    orchestrator touches.

    Thread-safety: Compressors SHOULD be stateless or internally thread-safe,
    as the same instance may be shared across concurrent pipeline executions.
    """

    @abstractmethod
    def compress(
        self,
        history_messages: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], float]:
        """
        Reduce the token footprint of a conversational history segment.

        CRITICAL INVARIANTS:
          • The compressor MUST NOT alter system or developer messages.
            (The pipeline guarantees these are not passed in, but implementations
            should be defensive and skip any message with role ``system`` or
            ``developer`` should they appear by mistake.)
          • The compressor MUST preserve message ordering.
          • The compressor MUST NOT fabricate content — it may only truncate,
            summarise, or drop existing messages.

        Args:
            history_messages: Ordered list of mutable conversational turns
                              (user + assistant messages only).

        Returns:
            A 2-tuple ``(compressed_messages, compression_ratio)`` where:
              • ``compressed_messages``  — The reduced message list.
              • ``compression_ratio``    — Fraction of original tokens retained,
                                          ∈ [0.0, 1.0].  ``1.0`` = no change.
        """

    # ── Optional Hook: Drift Validation ──────────────────────────────────────

    def validate_drift(
        self,
        original: List[ChatMessage],
        compressed: List[ChatMessage],
    ) -> float:
        """
        Optional hook for semantic drift scoring after compression.

        Default implementation returns ``1.0`` (no drift check).
        Override or inject a ``DriftValidator`` instance in concrete classes
        to enable cosine-similarity-based semantic validation.

        Args:
            original:   The uncompressed message list.
            compressed: The compressed message list.

        Returns:
            Cosine similarity score ∈ [0.0, 1.0].  ``1.0`` = identical.
        """
        return 1.0

    def compress_and_validate(
        self,
        history_messages: List[ChatMessage],
        drift_threshold: float = 0.90,
    ) -> Tuple[List[ChatMessage], float, float]:
        """
        Convenience wrapper that compresses and then validates semantic drift.

        Args:
            history_messages: Mutable history turns.
            drift_threshold:  Minimum acceptable cosine similarity.

        Returns:
            A 3-tuple ``(compressed_messages, compression_ratio, drift_score)``.

        Raises:
            ``CompressionException`` if drift is critically below threshold.
        """
        import logging

        logger = logging.getLogger(__name__)
        compressed, ratio = self.compress(history_messages)
        drift = self.validate_drift(history_messages, compressed)

        if drift < drift_threshold:
            logger.warning(
                "Semantic drift detected after compression: "
                "score=%.4f threshold=%.4f — compressed context may have "
                "lost significant semantic content.",
                drift,
                drift_threshold,
            )

        return compressed, ratio, drift
