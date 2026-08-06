"""
modules/history/base.py
────────────────────────
Abstract Base Class for the Conversational History Engine.

ZERO-COUPLING CONTRACT:
  This module imports ONLY from:
    • Python standard library
    • ``core.schemas``
  It MUST NOT import from:
    • ``modules.security``
    • ``modules.compression``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from core.schemas import ChatMessage, SessionState


class BaseHistoryManager(ABC):
    """
    Contract interface for the Conversational History Engine.

    Responsibilities:
      1. **Zone Splitting** — Partition messages into immutable directives and
         mutable conversation history.
      2. **Session State** — Maintain per-session context buffers with TTL
         eviction semantics.
      3. **Turn Management** — Append, retrieve, and clear conversational turns.
    """

    # ── Zone Splitting ────────────────────────────────────────────────────────

    @abstractmethod
    def split_immutable_zone(
        self,
        messages: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """
        Partition a message list into immutable directives and mutable history.

        Immutable messages are those with roles ``system`` or ``developer``.
        They represent authoritative runtime instructions and MUST NOT be
        modified, compressed, or PII-masked by subsequent pipeline stages.

        Args:
            messages: Full incoming message list in order.

        Returns:
            A 2-tuple ``(immutable_messages, history_messages)`` where:
              • ``immutable_messages`` — system/developer directives (ordered).
              • ``history_messages``   — all remaining conversational turns.
        """

    # ── Session State ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Retrieve the current state for a session, or ``None`` if not found.

        Args:
            session_id: Unique session identifier.

        Returns:
            ``SessionState`` if the session exists and has not expired;
            ``None`` otherwise.
        """

    @abstractmethod
    def upsert_session(self, state: SessionState) -> None:
        """
        Create or overwrite the stored state for a session.

        Args:
            state: Fully populated ``SessionState`` to persist.
        """

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """
        Remove a session and its associated state from the store.

        Args:
            session_id: Unique session identifier.
        """

    # ── Turn Management ───────────────────────────────────────────────────────

    @abstractmethod
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Append a single conversational turn to an existing session's history.

        Args:
            session_id: Target session.
            role:       Message role string (``"user"`` or ``"assistant"``).
            content:    Message body text.

        Raises:
            ``SessionNotFoundException`` if the session does not exist.
        """

    @abstractmethod
    def get_history(self, session_id: str) -> List[ChatMessage]:
        """
        Return the ordered list of mutable history turns for a session.

        Args:
            session_id: Target session.

        Returns:
            List of ``ChatMessage`` objects in chronological order.
            Returns an empty list for unknown sessions (non-raising).
        """
