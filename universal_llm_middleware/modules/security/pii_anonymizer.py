"""
modules/security/pii_anonymizer.py
────────────────────────────────────
Regex + spaCy NER PII anonymiser and re-identifier.

Detection layers:
  1. **Regex patterns** — High-precision rules for structured PII:
     emails, phone numbers, credit card numbers, SSNs, IPv4/IPv6 addresses,
     dates of birth, passport numbers, and generic API keys / secrets.
  2. **spaCy NER** — Entity recognition for unstructured PII:
     PERSON, ORG, GPE (locations), and NORP names.
     spaCy model is loaded lazily on first use to avoid startup cost.

Per-session isolation:
  All placeholder ↔ original mappings are stored in an ephemeral in-process
  dict keyed by ``session_id``.  Nothing is written to disk.  Mappings are
  cleared when the session is deleted via the History Engine.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regex PII Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (label, compiled_pattern)
_REGEX_PII_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )),
    ("PHONE", re.compile(
        r"(?<!\d)"
        r"(\+?1[\s\-.]?)?"
        r"(\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})"
        r"(?!\d)"
    )),
    ("CREDIT_CARD", re.compile(
        r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b"
    )),
    ("SSN", re.compile(
        r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"
    )),
    ("IPV4", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )),
    ("IPV6", re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    )),
    ("API_KEY", re.compile(
        # Generic bearer / secret tokens (≥ 20 hex/base64 chars)
        r"(?:key|token|secret|api[-_]?key|bearer)\s*[:=]\s*[A-Za-z0-9+/=_\-]{20,}",
        re.I,
    )),
    ("PASSPORT", re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    )),
    ("DATE_OF_BIRTH", re.compile(
        r"\b(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b"
    )),
]


# ─────────────────────────────────────────────────────────────────────────────
# spaCy Lazy Loader
# ─────────────────────────────────────────────────────────────────────────────

_spacy_lock = threading.Lock()
_spacy_nlp: Optional[object] = None
_spacy_loaded: bool = False
_spacy_failed: bool = False

_SPACY_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "NORP", "LOC"}


def _get_spacy_nlp(model_name: str = "en_core_web_sm") -> Optional[object]:
    """Return a cached spaCy nlp object, or ``None`` if unavailable."""
    global _spacy_nlp, _spacy_loaded, _spacy_failed

    if _spacy_loaded:
        return _spacy_nlp
    if _spacy_failed:
        return None

    with _spacy_lock:
        if _spacy_loaded or _spacy_failed:
            return _spacy_nlp

        try:
            import spacy  # type: ignore[import]

            _spacy_nlp = spacy.load(model_name)
            _spacy_loaded = True
            logger.info("spaCy model '%s' loaded successfully.", model_name)
        except Exception as exc:
            _spacy_failed = True
            logger.warning(
                "Failed to load spaCy model '%s': %s — "
                "NER-based PII detection is disabled.",
                model_name,
                exc,
            )

    return _spacy_nlp


# ─────────────────────────────────────────────────────────────────────────────
# PIIAnonymizer
# ─────────────────────────────────────────────────────────────────────────────


class PIIAnonymizer:
    """
    Stateful PII anonymiser with per-session placeholder mapping.

    Placeholder format: ``[LABEL_N]`` where ``LABEL`` is the PII type and
    ``N`` is a monotonically increasing counter per session.

    Example:
        >>> anon = PIIAnonymizer()
        >>> masked = anon.mask("Contact alice@example.com", session_id="s1")
        >>> masked
        'Contact [EMAIL_1]'
        >>> anon.unmask("[EMAIL_1] replied", session_id="s1")
        'alice@example.com replied'
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._spacy_model = spacy_model
        # session_id → { placeholder → original_value }
        self._session_maps: Dict[str, Dict[str, str]] = {}
        # session_id → { label → counter }
        self._session_counters: Dict[str, Dict[str, int]] = {}
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def mask(self, text: str, session_id: str) -> str:
        """
        Detect and replace all PII in ``text`` with typed placeholders.

        Args:
            text:       Input text.
            session_id: Session scope for placeholder storage.

        Returns:
            Anonymised text.
        """
        if not text:
            return text

        with self._lock:
            pii_map = self._ensure_session(session_id)
            result = text

            # Layer 1: regex-based structural PII
            result = self._apply_regex_masks(result, session_id, pii_map)

            # Layer 2: spaCy NER for unstructured entities
            result = self._apply_ner_masks(result, session_id, pii_map)

        logger.debug(
            "PII masking complete for session '%s': %d placeholders active.",
            session_id,
            len(pii_map),
        )
        return result

    def unmask(self, text: str, session_id: str) -> str:
        """
        Restore ``[LABEL_N]`` placeholders to their original values.

        Unknown placeholders are left unchanged.

        Args:
            text:       Text with potential placeholders.
            session_id: Session scope for the placeholder mapping.

        Returns:
            Text with known placeholders restored.
        """
        if not text:
            return text

        with self._lock:
            pii_map = self._session_maps.get(session_id, {})
            if not pii_map:
                return text

            result = text
            # Replace longest placeholders first to avoid partial matches
            for placeholder, original in sorted(pii_map.items(), key=lambda x: -len(x[0])):
                result = result.replace(placeholder, original)

        return result

    def clear_session(self, session_id: str) -> None:
        """Remove all PII mappings for the given session."""
        with self._lock:
            self._session_maps.pop(session_id, None)
            self._session_counters.pop(session_id, None)

    def get_session_map(self, session_id: str) -> Dict[str, str]:
        """Return a copy of the placeholder map for a session (for merging into SessionState)."""
        with self._lock:
            return dict(self._session_maps.get(session_id, {}))

    def load_session_map(self, session_id: str, pii_map: Dict[str, str]) -> None:
        """Restore a previously saved PII map into the session store."""
        with self._lock:
            self._session_maps[session_id] = dict(pii_map)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_session(self, session_id: str) -> Dict[str, str]:
        if session_id not in self._session_maps:
            self._session_maps[session_id] = {}
            self._session_counters[session_id] = {}
        return self._session_maps[session_id]

    def _next_placeholder(self, session_id: str, label: str) -> str:
        counters = self._session_counters.setdefault(session_id, {})
        counters[label] = counters.get(label, 0) + 1
        return f"[{label}_{counters[label]}]"

    def _register(
        self,
        session_id: str,
        pii_map: Dict[str, str],
        label: str,
        original: str,
    ) -> str:
        """Return an existing placeholder or create a new one."""
        # Idempotent: same value always gets the same placeholder
        for ph, val in pii_map.items():
            if val == original and ph.startswith(f"[{label}_"):
                return ph
        placeholder = self._next_placeholder(session_id, label)
        pii_map[placeholder] = original
        return placeholder

    def _apply_regex_masks(
        self,
        text: str,
        session_id: str,
        pii_map: Dict[str, str],
    ) -> str:
        result = text
        for label, pattern in _REGEX_PII_PATTERNS:
            def _replacer(m: re.Match, _label: str = label) -> str:  # noqa: E731
                return self._register(session_id, pii_map, _label, m.group(0))

            result = pattern.sub(_replacer, result)
        return result

    def _apply_ner_masks(
        self,
        text: str,
        session_id: str,
        pii_map: Dict[str, str],
    ) -> str:
        nlp = _get_spacy_nlp(self._spacy_model)
        if nlp is None:
            return text

        doc = nlp(text)  # type: ignore[call-arg]
        # Process in reverse order to preserve character offsets
        result = text
        for ent in reversed(doc.ents):  # type: ignore[attr-defined]
            if ent.label_ not in _SPACY_ENTITY_TYPES:
                continue
            original = ent.text
            # Skip if already masked by regex layer
            if any(original in v for v in pii_map.values()):
                continue
            placeholder = self._register(session_id, pii_map, ent.label_, original)
            result = result[: ent.start_char] + placeholder + result[ent.end_char :]

        return result
