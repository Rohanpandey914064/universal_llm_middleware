"""
modules/compression/drift_validator.py
───────────────────────────────────────
Semantic drift validator using TF-IDF vectorisation and cosine similarity.

Purpose:
  After a compressor reduces the conversational history, ``DriftValidator``
  measures how much semantic content was lost.  A cosine similarity score
  below the configured threshold (``Settings.drift_threshold``, default 0.90)
  triggers a structured warning so the pipeline can decide whether to:
    • Accept the compressed output (log the warning, continue).
    • Fall back to a less aggressive compression pass.
    • Raise ``CompressionException`` for critical drift.

Implementation:
  • Concatenates all message contents into a single corpus string per variant
    (original / compressed).
  • Builds a TF-IDF vector over character n-grams (2–4) for robustness to
    stop-word removal and minor paraphrasing.
  • Computes cosine similarity between the two sparse vectors.
  • Falls back to ``1.0`` (no drift) if the corpus is too short to vectorise
    meaningfully (< 10 tokens).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from core.schemas import ChatMessage

logger = logging.getLogger(__name__)

# Minimum combined word count below which drift scoring is skipped.
_MIN_WORD_COUNT = 10


class DriftValidator:
    """
    Cosine-similarity-based semantic drift checker.

    Args:
        threshold: Minimum acceptable similarity score.  Defaults to ``0.90``.

    Example::

        validator = DriftValidator(threshold=0.90)
        score = validator.score(original_messages, compressed_messages)
        if score < 0.90:
            logger.warning("Drift detected: %.4f", score)
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        from config.settings import get_settings  # local import — no circular dep

        settings = get_settings()
        self.threshold: float = threshold if threshold is not None else settings.drift_threshold
        self._vectorizer: Optional[object] = None
        self._vectorizer_loaded: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        original: List[ChatMessage],
        compressed: List[ChatMessage],
    ) -> float:
        """
        Compute cosine similarity between original and compressed message sets.

        Args:
            original:   Uncompressed message list.
            compressed: Compressed message list.

        Returns:
            Cosine similarity score ∈ [0.0, 1.0].  Returns ``1.0`` if either
            corpus is too short or identical.
        """
        orig_text = self._concat(original)
        comp_text = self._concat(compressed)

        if orig_text == comp_text:
            return 1.0

        word_count = len(orig_text.split()) + len(comp_text.split())
        if word_count < _MIN_WORD_COUNT:
            logger.debug(
                "Drift check skipped — corpus too small (%d words).", word_count
            )
            return 1.0

        try:
            return self._cosine_similarity_tfidf(orig_text, comp_text)
        except Exception as exc:
            logger.warning(
                "Drift scoring failed (%s) — returning 1.0 (no drift assumed).", exc
            )
            return 1.0

    def validate(
        self,
        original: List[ChatMessage],
        compressed: List[ChatMessage],
    ) -> tuple[float, bool]:
        """
        Score and evaluate against the configured threshold.

        Returns:
            ``(score, passed)`` where ``passed`` is ``True`` if score ≥ threshold.
        """
        drift_score = self.score(original, compressed)
        passed = drift_score >= self.threshold

        if not passed:
            logger.warning(
                "Semantic drift warning — score=%.4f threshold=%.4f. "
                "Compressed context may have lost significant semantic content.",
                drift_score,
                self.threshold,
            )
        else:
            logger.debug("Drift check passed — score=%.4f", drift_score)

        return drift_score, passed

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _concat(messages: List[ChatMessage]) -> str:
        """Concatenate all message contents into a single string."""
        return " ".join(
            (msg.content or "") for msg in messages if msg.content
        ).strip()

    def _cosine_similarity_tfidf(self, text_a: str, text_b: str) -> float:
        """
        Build TF-IDF vectors for two texts and return their cosine similarity.

        Uses ``sklearn.feature_extraction.text.TfidfVectorizer`` with
        character n-gram range (2, 4) which is resilient to stop-word
        differences introduced by compression.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([text_a, text_b])
            sim: float = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
            return round(sim, 6)

        except ImportError:
            # sklearn not available — fall back to token-level Jaccard similarity
            logger.debug(
                "scikit-learn not available — falling back to Jaccard similarity."
            )
            return self._jaccard_similarity(text_a, text_b)

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Simple token-level Jaccard similarity as a fallback."""
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
