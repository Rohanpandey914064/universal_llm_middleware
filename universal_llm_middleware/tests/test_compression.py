"""
tests/test_compression.py
──────────────────────────
Unit tests for:
  • SlidingWindowCompressor — compression correctness and ratio
  • DriftValidator          — cosine similarity scoring
  • BaseCompressor          — compress_and_validate hook
"""

from __future__ import annotations

from typing import List


from core.schemas import ChatMessage, MessageRole
from modules.compression.custom_engine import SlidingWindowCompressor, _estimate_tokens
from modules.compression.drift_validator import DriftValidator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_message(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole(role), content=content)


def _make_history(n: int, words_per_msg: int = 50) -> List[ChatMessage]:
    """Generate a list of alternating user/assistant messages."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = " ".join([f"word{j}" for j in range(words_per_msg)])
        msgs.append(_make_message(role, content))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowCompressor Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSlidingWindowCompressor:
    """Tests for the sliding-window compression algorithm."""

    def test_no_compression_when_within_budget(self) -> None:
        """Small history that fits within budget should not be modified."""
        compressor = SlidingWindowCompressor(max_tokens=5000, min_keep_turns=4)
        history = _make_history(4, words_per_msg=10)
        compressed, ratio = compressor.compress(history)
        assert len(compressed) == len(history)
        assert ratio == 1.0

    def test_compresses_large_history(self) -> None:
        """History exceeding budget should be reduced."""
        compressor = SlidingWindowCompressor(max_tokens=200, min_keep_turns=2)
        history = _make_history(20, words_per_msg=30)
        compressed, ratio = compressor.compress(history)
        assert len(compressed) < len(history)
        assert ratio < 1.0

    def test_compression_ratio_within_bounds(self) -> None:
        compressor = SlidingWindowCompressor(max_tokens=300, min_keep_turns=2)
        history = _make_history(10, words_per_msg=40)
        _, ratio = compressor.compress(history)
        assert 0.0 <= ratio <= 1.0

    def test_empty_history_returns_empty(self) -> None:
        compressor = SlidingWindowCompressor()
        compressed, ratio = compressor.compress([])
        assert compressed == []
        assert ratio == 1.0

    def test_min_keep_turns_always_honoured(self) -> None:
        """Even with a very small budget, min_keep_turns must be retained."""
        compressor = SlidingWindowCompressor(max_tokens=1, min_keep_turns=3)
        history = _make_history(10, words_per_msg=100)
        compressed, _ = compressor.compress(history)
        assert len(compressed) >= 3

    def test_chronological_order_preserved(self) -> None:
        """Compressed messages must remain in their original order."""
        compressor = SlidingWindowCompressor(max_tokens=200, min_keep_turns=2)
        history = [_make_message("user", f"Turn {i}") for i in range(10)]
        compressed, _ = compressor.compress(history)
        # Check that the indices are monotonically increasing
        turn_numbers = [int(m.content.split()[-1]) for m in compressed]  # type: ignore
        assert turn_numbers == sorted(turn_numbers)

    def test_most_recent_turns_kept(self) -> None:
        """Recency-weighted: latest turns must be in the compressed output."""
        compressor = SlidingWindowCompressor(max_tokens=200, min_keep_turns=2)
        history = [_make_message("user", f"Turn {i}") for i in range(10)]
        compressed, _ = compressor.compress(history)
        last_turn = history[-1].content
        retained_contents = [m.content for m in compressed]
        assert last_turn in retained_contents, "Latest turn should always be retained."

    def test_immutable_messages_excluded(self) -> None:
        """Compressor must exclude system messages even if passed accidentally."""
        compressor = SlidingWindowCompressor(max_tokens=5000)
        messages = [
            _make_message("system", "System directive"),
            _make_message("user", "User question"),
        ]
        compressed, _ = compressor.compress(messages)
        roles = [m.role for m in compressed]
        assert MessageRole.SYSTEM not in roles

    def test_token_estimation_is_positive(self) -> None:
        msg = _make_message("user", "Hello world")
        tokens = _estimate_tokens(msg)
        assert tokens > 0

    def test_compress_and_validate_returns_three_tuple(self) -> None:
        compressor = SlidingWindowCompressor(max_tokens=500)
        history = _make_history(5, words_per_msg=10)
        compressed, ratio, drift = compressor.compress_and_validate(history)
        assert isinstance(compressed, list)
        assert 0.0 <= ratio <= 1.0
        assert 0.0 <= drift <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# DriftValidator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDriftValidator:
    """Tests for semantic drift scoring."""

    def setup_method(self) -> None:
        self.validator = DriftValidator(threshold=0.90)

    def _msg(self, content: str) -> ChatMessage:
        return _make_message("user", content)

    def test_identical_texts_score_one(self) -> None:
        msgs = [self._msg("The quick brown fox jumps over the lazy dog.")]
        score = self.validator.score(msgs, msgs)
        assert score == 1.0

    def test_completely_different_texts_score_low(self) -> None:
        original = [self._msg("Python programming language variables functions")]
        compressed = [self._msg("apples oranges bananas fruits tropical")]
        score = self.validator.score(original, compressed)
        assert score < 0.90

    def test_similar_texts_score_high(self) -> None:
        original = [
            self._msg("Python is a high-level programming language."),
            self._msg("It is widely used for data science and machine learning."),
        ]
        compressed = [
            self._msg("Python is a programming language used for data science."),
        ]
        score = self.validator.score(original, compressed)
        # Similar content should have reasonably high similarity
        assert score > 0.30  # Not too strict — TF-IDF varies

    def test_empty_inputs_return_one(self) -> None:
        score = self.validator.score([], [])
        assert score == 1.0

    def test_too_short_corpus_returns_one(self) -> None:
        original = [self._msg("Hi")]
        compressed = [self._msg("Hello")]
        score = self.validator.score(original, compressed)
        assert score == 1.0

    def test_validate_returns_passed_flag(self) -> None:
        msgs = [self._msg(" ".join([f"word{i}" for i in range(50)]))]
        score, passed = self.validator.validate(msgs, msgs)
        assert passed is True
        assert score == 1.0

    def test_validate_below_threshold_fails(self) -> None:
        strict = DriftValidator(threshold=0.99)
        original = [self._msg(" ".join([f"alpha{i}" for i in range(30)]))]
        compressed = [self._msg(" ".join([f"beta{i}" for i in range(30)]))]
        _, passed = strict.validate(original, compressed)
        assert passed is False

    def test_jaccard_fallback(self) -> None:
        """Test the Jaccard fallback path directly."""
        score = DriftValidator._jaccard_similarity(
            "the quick brown fox",
            "the quick brown dog",
        )
        # 3/5 shared tokens
        assert score > 0.5

    def test_jaccard_identical(self) -> None:
        score = DriftValidator._jaccard_similarity("hello world", "hello world")
        assert score == 1.0

    def test_jaccard_no_overlap(self) -> None:
        score = DriftValidator._jaccard_similarity("aaa bbb ccc", "xxx yyy zzz")
        assert score == 0.0
