"""
modules/security/input_guard.py
────────────────────────────────
Prompt injection & jailbreak detection engine.

Detection strategy (layered, in order):
  1. **Heuristic ruleset** — Fast regex / keyword matching against a curated
     set of known injection patterns.  Always runs.
  2. **ONNX/DeBERTa classifier** — Loads a local ONNX model (e.g. a fine-tuned
     DeBERTa-v3 for prompt-injection binary classification) when an ONNX path
     is configured.  Falls back gracefully to heuristics-only if the model
     cannot be loaded.

Threshold: Inputs scoring ≥ ``T = 0.58`` are flagged as unsafe.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Rule Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each rule is a (compiled_pattern, score_contribution, label) triple.
# Score contributions are summed and clamped to [0.0, 1.0].
_INJECTION_RULES: List[Tuple[re.Pattern[str], float, str]] = [
    # Role override attacks
    (re.compile(r"\bignore\s+(all\s+)?(previous|above|prior)\b", re.I), 0.60, "role_override"),
    (re.compile(r"\bforget\s+(all\s+)?(previous|prior|above)\b", re.I), 0.60, "role_override"),
    (re.compile(r"\bact\s+as\s+(if\s+you\s+are|a)\b", re.I), 0.40, "persona_override"),
    (re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.I), 0.40, "persona_override"),
    (re.compile(r"\byou\s+are\s+now\b.*\bmode\b", re.I), 0.45, "mode_switch"),
    # System prompt extraction
    (re.compile(r"\brepeat\s+.{0,40}(system\s+prompt|instructions?)\b", re.I), 0.70, "extraction"),
    (re.compile(r"\bprint\s+.{0,20}(system\s+prompt|above\s+text)\b", re.I), 0.70, "extraction"),
    (re.compile(r"\bwhat\s+(are|were)\s+your\s+(instructions?|prompts?)\b", re.I), 0.50, "extraction"),
    (re.compile(r"\bshow\s+me\s+your\s+(system\s+)?prompt\b", re.I), 0.65, "extraction"),
    # DAN / jailbreak patterns
    (re.compile(r"\bDAN\b", re.I), 0.80, "dan_jailbreak"),
    (re.compile(r"\bjailbreak\b", re.I), 0.75, "jailbreak_keyword"),
    (re.compile(r"\bdo\s+anything\s+now\b", re.I), 0.80, "dan_jailbreak"),
    (re.compile(r"\benable\s+developer\s+mode\b", re.I), 0.65, "mode_switch"),
    # Instruction smuggling via delimiters
    (re.compile(r"</?(system|instructions?|prompt)>", re.I), 0.60, "tag_injection"),
    (re.compile(r"\[SYSTEM\]|\[INST\]|\[\/INST\]", re.I), 0.55, "tag_injection"),
    # Continuation / override prompts
    (re.compile(r"\bfrom\s+now\s+on\b.{0,60}(respond|reply|answer)\b", re.I), 0.45, "policy_override"),
    (re.compile(r"\byour\s+new\s+(rules?|instructions?|guidelines?)\b", re.I), 0.55, "policy_override"),
    # Code execution injection
    (re.compile(r"eval\s*\(|exec\s*\(|__import__\s*\(", re.I), 0.50, "code_injection"),
    (re.compile(r"os\.system\s*\(|subprocess\.", re.I), 0.60, "code_injection"),
]

_MAX_HEURISTIC_SCORE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Model Loader (lazy, singleton per path)
# ─────────────────────────────────────────────────────────────────────────────

_onnx_session_cache: Dict[str, object] = {}


def _load_onnx_session(model_path: Path) -> Optional[object]:
    """
    Attempt to load an ONNX inference session.  Returns ``None`` on failure.

    The session object is cached keyed by path string to avoid repeated disk
    I/O between requests.
    """
    key = str(model_path)
    if key in _onnx_session_cache:
        return _onnx_session_cache[key]

    try:
        import onnxruntime as ort  # type: ignore[import]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=opts)
        _onnx_session_cache[key] = session
        logger.info("ONNX injection model loaded from '%s'.", model_path)
        return session
    except Exception as exc:
        logger.warning(
            "Failed to load ONNX model from '%s': %s — "
            "falling back to heuristic-only mode.",
            model_path,
            exc,
        )
        _onnx_session_cache[key] = None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# InjectionGuard
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InjectionGuard:
    """
    Multi-layer prompt injection and jailbreak detector.

    Args:
        threshold:   Score threshold above which a prompt is flagged as unsafe.
                     Default ``0.58`` per specification.
        onnx_path:   Optional path to a DeBERTa-v3 ONNX model for enhanced
                     classification.  When ``None``, heuristic-only mode is used.
        onnx_label_unsafe: The label string the ONNX model emits for unsafe
                           predictions.  Default ``"INJECTION"`` — adjust for
                           your specific model.
    """

    threshold: float = 0.58
    onnx_path: Optional[Path] = None
    onnx_label_unsafe: str = "INJECTION"

    # Private: populated lazily
    _onnx_session: Optional[object] = field(default=None, init=False, repr=False)
    _onnx_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.onnx_path is not None:
            self._onnx_session = _load_onnx_session(self.onnx_path)
            self._onnx_loaded = self._onnx_session is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def inspect(self, prompt: str) -> Tuple[bool, float, str]:
        """
        Analyse ``prompt`` for adversarial content.

        Returns:
            ``(is_safe, threat_score, reason)``
        """
        if not prompt or not prompt.strip():
            return True, 0.0, "Empty input."

        # Layer 1: heuristics (always runs)
        heuristic_score, heuristic_reasons = self._run_heuristics(prompt)

        # Layer 2: ONNX model (runs only if model is available)
        onnx_score: Optional[float] = None
        if self._onnx_loaded and self._onnx_session is not None:
            try:
                onnx_score = self._run_onnx(prompt)
            except Exception as exc:
                logger.warning("ONNX inference failed: %s — using heuristics only.", exc)

        # Combine scores: take the maximum of available signals
        final_score = heuristic_score
        if onnx_score is not None:
            final_score = max(heuristic_score, onnx_score)
            logger.debug(
                "Injection scores — heuristic=%.3f onnx=%.3f final=%.3f",
                heuristic_score,
                onnx_score,
                final_score,
            )

        is_safe = final_score < self.threshold
        reason = self._build_reason(is_safe, final_score, heuristic_reasons, onnx_score)

        logger.debug(
            "inspect_input result — is_safe=%s score=%.3f threshold=%.3f",
            is_safe,
            final_score,
            self.threshold,
        )
        return is_safe, round(final_score, 4), reason

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_heuristics(self, text: str) -> Tuple[float, List[str]]:
        """Apply the heuristic rule set and return (score, matched_labels)."""
        cumulative = 0.0
        matched: List[str] = []

        for pattern, weight, label in _INJECTION_RULES:
            if pattern.search(text):
                cumulative += weight
                matched.append(label)

        # Diminishing returns — cap at 1.0
        score = min(cumulative, _MAX_HEURISTIC_SCORE)
        return score, matched

    def _run_onnx(self, text: str) -> float:
        """
        Run the ONNX DeBERTa model and return a threat probability score.

        This implementation uses a simple tokenisation that works as a dummy
        for testing.  In production, replace the tokeniser call with the
        matching HuggingFace tokeniser for your ONNX model checkpoint.
        """
        try:
            import numpy as np  # type: ignore[import]

            session = self._onnx_session

            # Attempt HuggingFace fast tokeniser (optional dependency)
            try:
                from transformers import AutoTokenizer  # type: ignore[import]

                tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
                inputs = tokenizer(
                    text,
                    return_tensors="np",
                    truncation=True,
                    max_length=512,
                )
                feed = {k: v for k, v in inputs.items()}
            except ImportError:
                # Fallback: dummy single-token input so the pipeline doesn't crash
                logger.debug(
                    "transformers not installed — using dummy ONNX input; "
                    "install transformers for real DeBERTa inference."
                )
                feed = {
                    "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                    "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
                }

            # Run inference
            outputs = session.run(None, feed)  # type: ignore[union-attr]
            logits = outputs[0][0]  # shape (num_labels,)

            # Softmax → probability
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Assume label index 1 == INJECTION (binary classification)
            unsafe_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return unsafe_prob

        except Exception as exc:
            raise RuntimeError(f"ONNX inference error: {exc}") from exc

    @staticmethod
    def _build_reason(
        is_safe: bool,
        score: float,
        labels: List[str],
        onnx_score: Optional[float],
    ) -> str:
        if is_safe:
            return f"Input passed security checks (score={score:.3f})."
        parts = [f"Threat score: {score:.3f}."]
        if labels:
            parts.append(f"Heuristic signals: {', '.join(set(labels))}.")
        if onnx_score is not None:
            parts.append(f"ONNX score: {onnx_score:.3f}.")
        return " ".join(parts)
