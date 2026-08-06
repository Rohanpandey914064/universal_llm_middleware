"""
modules/history/manager.py
────────────────────────────
Concrete ``DefaultHistoryManager`` implementing ``BaseHistoryManager`` by
composing ``InMemorySessionStore`` and ``ZoneSplitter``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from core.schemas import (
    ChatMessage,
    MessageRole,
    SessionNotFoundException,
    SessionState,
)
from modules.history.base import BaseHistoryManager
from modules.history.session_store import InMemorySessionStore
from modules.history.zone_splitter import ZoneSplitter

logger = logging.getLogger(__name__)


class DefaultHistoryManager(BaseHistoryManager):
    """
    Default history manager using the in-memory session store and
    zone splitter implementations.

    Args:
        store:    Custom ``InMemorySessionStore`` instance.  A new one is
                  created with default settings if not provided.
        splitter: Custom ``ZoneSplitter`` instance.
    """

    def __init__(
        self,
        store: Optional[InMemorySessionStore] = None,
        splitter: Optional[ZoneSplitter] = None,
    ) -> None:
        self._store = store or InMemorySessionStore()
        self._splitter = splitter or ZoneSplitter()

    # ── Zone Splitting ────────────────────────────────────────────────────────

    def split_immutable_zone(
        self,
        messages: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        return self._splitter.split(messages)

    # ── Session State ─────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._store.get(session_id)

    def upsert_session(self, state: SessionState) -> None:
        self._store.put(state)

    def delete_session(self, session_id: str) -> None:
        self._store.delete(session_id)

    # ── Turn Management ───────────────────────────────────────────────────────

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        added = self._store.add_turn(session_id, role, content)
        if not added:
            raise SessionNotFoundException(session_id)

    def get_history(self, session_id: str) -> List[ChatMessage]:
        state = self._store.get(session_id)
        if state is None:
            return []
        return list(state.history_messages)

    # ── Convenience ───────────────────────────────────────────────────────────

    def ensure_session(self, session_id: str) -> SessionState:
        """
        Return the session state, creating an empty one if it does not exist.

        Unlike ``get_session``, this never returns ``None``.
        """
        state = self._store.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._store.put(state)
            logger.debug("New session created: '%s'", session_id)
        return state

    @property
    def store(self) -> InMemorySessionStore:
        """Expose the underlying store for metrics / testing."""
        return self._store

    @property
    def splitter(self) -> ZoneSplitter:
        """Expose the zone splitter for direct use."""
        return self._splitter
