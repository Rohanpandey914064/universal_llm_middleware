"""
modules/history/session_store.py
──────────────────────────────────
Thread-safe in-memory session state buffer with TTL-based eviction.

Design:
  • Sessions are stored in a ``dict`` protected by a ``threading.RLock``.
  • Each session records its ``last_accessed`` timestamp.
  • A background daemon thread (``_EvictionThread``) periodically scans for
    expired sessions and removes them to prevent unbounded memory growth.
  • The store is designed to be replaced with a Redis/DynamoDB backend by
    implementing the ``BaseHistoryManager`` interface.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from config.settings import get_settings
from core.schemas import ChatMessage, MessageRole, SessionState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TTL Eviction Thread
# ─────────────────────────────────────────────────────────────────────────────


class _EvictionThread(threading.Thread):
    """Daemon thread that periodically evicts expired sessions."""

    def __init__(
        self,
        store: "InMemorySessionStore",
        interval_seconds: float = 60.0,
    ) -> None:
        super().__init__(name="session-eviction", daemon=True)
        self._store = store
        self._interval = interval_seconds
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._store._evict_expired()
            except Exception as exc:
                logger.error("Session eviction error: %s", exc)

    def stop(self) -> None:
        self._stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# InMemorySessionStore
# ─────────────────────────────────────────────────────────────────────────────


class InMemorySessionStore:
    """
    Thread-safe, TTL-based in-memory session state store.

    Args:
        ttl_seconds:       Seconds before an idle session is evicted.
                           Defaults to ``Settings.session_ttl_seconds``.
        max_history_turns: Maximum mutable turns retained per session.
                           Oldest turns are dropped when the limit is exceeded.
        eviction_interval: How often (in seconds) the background eviction
                           thread runs.  Set to ``0`` to disable the thread.
    """

    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        max_history_turns: Optional[int] = None,
        eviction_interval: float = 60.0,
    ) -> None:
        settings = get_settings()
        self._ttl = ttl_seconds or settings.session_ttl_seconds
        self._max_turns = max_history_turns or settings.max_history_turns
        self._lock = threading.RLock()
        # session_id → (SessionState, last_accessed_timestamp)
        self._store: Dict[str, tuple[SessionState, float]] = {}

        if eviction_interval > 0:
            self._eviction_thread = _EvictionThread(self, eviction_interval)
            self._eviction_thread.start()
            logger.debug(
                "Session eviction thread started — interval=%ds ttl=%ds",
                int(eviction_interval),
                self._ttl,
            )
        else:
            self._eviction_thread = None  # type: ignore[assignment]

    # ── Read Operations ───────────────────────────────────────────────────────

    def get(self, session_id: str) -> Optional[SessionState]:
        """Return the ``SessionState`` for ``session_id``, refreshing its TTL."""
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            state, _ = entry
            # Refresh last-accessed timestamp
            self._store[session_id] = (state, time.monotonic())
            return state

    # ── Write Operations ──────────────────────────────────────────────────────

    def put(self, state: SessionState) -> None:
        """Create or replace the stored state for ``state.session_id``."""
        with self._lock:
            self._store[state.session_id] = (state, time.monotonic())
        logger.debug("Session '%s' upserted.", state.session_id)

    def add_turn(self, session_id: str, role: str, content: str) -> bool:
        """
        Append a turn to an existing session's mutable history.

        Returns:
            ``True`` if the turn was added; ``False`` if the session was not
            found (non-raising — caller decides how to handle).
        """
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                logger.warning(
                    "add_turn called for unknown session '%s' — ignoring.", session_id
                )
                return False

            state, _ = entry
            new_turn = ChatMessage(role=MessageRole(role), content=content)

            # Enforce max_turns limit (FIFO eviction)
            updated_history = list(state.history_messages) + [new_turn]
            if len(updated_history) > self._max_turns:
                overflow = len(updated_history) - self._max_turns
                updated_history = updated_history[overflow:]
                logger.debug(
                    "Session '%s' history trimmed by %d turn(s).",
                    session_id,
                    overflow,
                )

            updated_state = state.model_copy(update={"history_messages": updated_history})
            self._store[session_id] = (updated_state, time.monotonic())
            return True

    def delete(self, session_id: str) -> None:
        """Remove a session from the store."""
        with self._lock:
            removed = self._store.pop(session_id, None)
        if removed:
            logger.debug("Session '%s' deleted.", session_id)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def _evict_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                sid
                for sid, (_, last) in self._store.items()
                if (now - last) > self._ttl
            ]
            for sid in expired:
                del self._store[sid]
        if expired:
            logger.info("Evicted %d expired session(s): %s", len(expired), expired)

    def active_session_count(self) -> int:
        """Return the number of currently live sessions."""
        with self._lock:
            return len(self._store)

    def shutdown(self) -> None:
        """Stop the background eviction thread gracefully."""
        if self._eviction_thread:
            self._eviction_thread.stop()
            self._eviction_thread.join(timeout=5.0)
