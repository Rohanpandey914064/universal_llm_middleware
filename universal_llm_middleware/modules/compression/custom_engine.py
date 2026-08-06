"""
modules/compression/custom_engine.py
──────────────────────────────────────
Production-quality compression engine implementing ``BaseCompressor``.

Algorithm: Recency-Weighted Sliding Window

Strategy:
  1. Estimate token count per message using a simple approximation
     (``len(text.split()) * 1.33`` — approximately matches GPT-4 tokenisation
     for English prose).
  2. Preserve the most recent messages that fit within the ``max_tokens``
     budget, always keeping the last ``min_keep_turns`` turns intact.
  3. Optionally run ``DriftValidator`` to measure semantic drift.  If drift
     falls below the configured threshold, issue a structured warning.

Compressor invariants (enforced defensively):
  • System / developer messages are NEVER compressed even if passed in.
  • Message order is always preserved.
  • Content is never fabricated — only existing messages are dropped.

Pluggability:
  Replace this class by subclassing ``BaseCompressor`` and injecting your
  implementation into ``UniversalPipeline`` at construction time.  No other
  code needs to change.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from config.settings import get_settings
from core.schemas import ChatMessage, CompressionException, MessageRole
from modules.compression.base import BaseCompressor
from modules.compression.drift_validator import DriftValidator

logger = logging.getLogger(__name__)

_IMMUTABLE_ROLES = frozenset({MessageRole.SYSTEM, MessageRole.DEVELOPER})

# Conservative token approximation: words × 1.33 + 4 (message overhead)
_TOKENS_PER_WORD: float = 1.33
_OVERHEAD_PER_MSG: int = 4


def _estimate_tokens(message: ChatMessage) -> int:
    """Estimate the number of tokens in a ``ChatMessage``."""
    text = message.content or ""
    word_count = len(text.split())
    return math.ceil(word_count * _TOKENS_PER_WORD) + _OVERHEAD_PER_MSG


class SlidingWindowCompressor(BaseCompressor):
    """
    Recency-weighted sliding-window context compressor.

    Args:
        max_tokens:      Token budget for compressed output.
                         Defaults to ``Settings.max_history_tokens``.
        min_keep_turns:  Minimum number of most-recent turns to always keep,
                         regardless of token budget.  Default ``4``.
        drift_threshold: Minimum cosine similarity required.  If drift falls
                         below this, a warning is logged.  Set to ``0.0`` to
                         disable drift checking.
        validator:       Custom ``DriftValidator`` instance.  A default one
                         is created if not provided.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        min_keep_turns: int = 4,
        drift_threshold: Optional[float] = None,
        validator: Optional[DriftValidator] = None,
    ) -> None:
        settings = get_settings()
        self._max_tokens = max_tokens or settings.max_history_tokens
        self._min_keep_turns = max(1, min_keep_turns)
        self._drift_threshold = (
            drift_threshold if drift_threshold is not None else settings.drift_threshold
        )
        self._validator = validator or DriftValidator(threshold=self._drift_threshold)

        logger.info(
            "SlidingWindowCompressor initialised — "
            "max_tokens=%d min_keep_turns=%d drift_threshold=%.2f",
            self._max_tokens,
            self._min_keep_turns,
            self._drift_threshold,
        )

    # ── BaseCompressor implementation ─────────────────────────────────────────

    def compress(
        self,
        history_messages: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], float]:
        """
        Apply recency-weighted sliding-window compression.

        Returns:
            ``(compressed_messages, compression_ratio)``
        """
        if not history_messages:
            return [], 1.0

        # Defensive guard: never touch immutable messages
        safe_history = [m for m in history_messages if m.role not in _IMMUTABLE_ROLES]
        if len(safe_history) < len(history_messages):
            logger.error(
                "SlidingWindowCompressor received %d immutable message(s) — "
                "they were silently excluded.  Check pipeline zone splitting.",
                len(history_messages) - len(safe_history),
            )

        original = safe_history
        original_tokens = sum(_estimate_tokens(m) for m in original)

        if original_tokens <= self._max_tokens:
            logger.debug(
                "No compression needed — %d tokens within budget of %d.",
                original_tokens,
                self._max_tokens,
            )
            return list(original), 1.0

        compressed = self._sliding_window(original)
        compressed_tokens = sum(_estimate_tokens(m) for m in compressed)

        ratio = (
            compressed_tokens / original_tokens
            if original_tokens > 0
            else 1.0
        )
        ratio = round(min(ratio, 1.0), 4)

        logger.info(
            "Compression complete — original=%d tokens, compressed=%d tokens, "
            "ratio=%.3f, turns=%d→%d",
            original_tokens,
            compressed_tokens,
            ratio,
            len(original),
            len(compressed),
        )
        return compressed, ratio

    # ── BaseCompressor.validate_drift override ────────────────────────────────

    def validate_drift(
        self,
        original: List[ChatMessage],
        compressed: List[ChatMessage],
    ) -> float:
        score, _ = self._validator.validate(original, compressed)
        return score

    # ── Private: Sliding-Window Algorithm ─────────────────────────────────────

    def _sliding_window(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Select the most recent messages that fit within ``_max_tokens``.

        Algorithm:
          • Walk from the end (most recent) backwards.
          • Accumulate tokens until the budget is exhausted.
          • Always keep at least ``_min_keep_turns`` messages.
          • The selected window is returned in original chronological order.
        """
        budget = self._max_tokens
        selected: List[ChatMessage] = []
        accumulated = 0

        for msg in reversed(messages):
            tokens = _estimate_tokens(msg)

            fits_in_budget = (accumulated + tokens) <= budget
            must_keep = len(selected) < self._min_keep_turns

            if fits_in_budget or must_keep:
                selected.append(msg)
                accumulated += tokens

                if not fits_in_budget and must_keep:
                    logger.debug(
                        "Kept message beyond budget to honour min_keep_turns=%d "
                        "(accumulated=%d > budget=%d).",
                        self._min_keep_turns,
                        accumulated,
                        budget,
                    )
            else:
                # Budget exhausted and min_keep satisfied — stop
                break

        # Restore chronological order
        selected.reverse()
        return selected
