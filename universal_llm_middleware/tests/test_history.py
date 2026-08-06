"""
tests/test_history.py
──────────────────────
Unit tests for:
  • ZoneSplitter     — message partitioning
  • InMemorySessionStore — state CRUD and TTL
  • DefaultHistoryManager — composed behaviour
"""

from __future__ import annotations

import time

import pytest

from core.schemas import ChatMessage, MessageRole, SessionNotFoundException, SessionState
from modules.history.manager import DefaultHistoryManager
from modules.history.session_store import InMemorySessionStore
from modules.history.zone_splitter import ZoneSplitter


# ─────────────────────────────────────────────────────────────────────────────
# ZoneSplitter Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestZoneSplitter:
    """Tests for zone partitioning correctness."""

    def setup_method(self) -> None:
        self.splitter = ZoneSplitter()

    def _msg(self, role: str, content: str = "test") -> ChatMessage:
        return ChatMessage(role=MessageRole(role), content=content)

    def test_system_goes_to_immutable(self) -> None:
        msgs = [self._msg("system", "Be helpful."), self._msg("user", "Hi")]
        immutable, history = self.splitter.split(msgs)
        assert len(immutable) == 1
        assert len(history) == 1
        assert immutable[0].role == MessageRole.SYSTEM

    def test_developer_goes_to_immutable(self) -> None:
        msgs = [self._msg("developer", "Strict guidelines."), self._msg("user", "Hi")]
        immutable, history = self.splitter.split(msgs)
        assert len(immutable) == 1
        assert immutable[0].role == MessageRole.DEVELOPER

    def test_user_assistant_go_to_history(self) -> None:
        msgs = [
            self._msg("user", "Hello"),
            self._msg("assistant", "Hi there!"),
        ]
        immutable, history = self.splitter.split(msgs)
        assert len(immutable) == 0
        assert len(history) == 2

    def test_mixed_messages_correctly_partitioned(self) -> None:
        msgs = [
            self._msg("system", "System directive"),
            self._msg("user", "User turn 1"),
            self._msg("assistant", "Assistant reply 1"),
            self._msg("user", "User turn 2"),
        ]
        immutable, history = self.splitter.split(msgs)
        assert len(immutable) == 1
        assert len(history) == 3

    def test_multiple_system_messages(self) -> None:
        """Multiple system messages (edge case) should all be immutable."""
        msgs = [
            self._msg("system", "Directive 1"),
            self._msg("system", "Directive 2"),
            self._msg("user", "Question"),
        ]
        immutable, history = self.splitter.split(msgs)
        assert len(immutable) == 2
        assert len(history) == 1

    def test_empty_list_returns_empty_partitions(self) -> None:
        immutable, history = self.splitter.split([])
        assert immutable == []
        assert history == []

    def test_order_preserved_in_immutable(self) -> None:
        msgs = [
            self._msg("system", "First"),
            self._msg("user", "Mid"),
            self._msg("developer", "Second"),
        ]
        immutable, _ = self.splitter.split(msgs)
        assert immutable[0].content == "First"
        assert immutable[1].content == "Second"

    def test_order_preserved_in_history(self) -> None:
        msgs = [
            self._msg("user", "A"),
            self._msg("assistant", "B"),
            self._msg("user", "C"),
        ]
        _, history = self.splitter.split(msgs)
        contents = [m.content for m in history]
        assert contents == ["A", "B", "C"]

    def test_merge_reconstructs_correct_order(self) -> None:
        immutable = [self._msg("system", "Directive")]
        history = [self._msg("user", "Q"), self._msg("assistant", "A")]
        merged = self.splitter.merge(immutable, history)
        assert merged[0].role == MessageRole.SYSTEM
        assert merged[1].role == MessageRole.USER
        assert merged[2].role == MessageRole.ASSISTANT

    def test_is_immutable_static_method(self) -> None:
        assert ZoneSplitter.is_immutable(self._msg("system")) is True
        assert ZoneSplitter.is_immutable(self._msg("developer")) is True
        assert ZoneSplitter.is_immutable(self._msg("user")) is False
        assert ZoneSplitter.is_immutable(self._msg("assistant")) is False


# ─────────────────────────────────────────────────────────────────────────────
# InMemorySessionStore Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInMemorySessionStore:
    """Tests for session store CRUD and TTL eviction."""

    def _make_store(self, ttl: int = 3600) -> InMemorySessionStore:
        return InMemorySessionStore(
            ttl_seconds=ttl,
            max_history_turns=10,
            eviction_interval=0,  # disable background thread in tests
        )

    def _make_state(self, session_id: str = "test-session") -> SessionState:
        return SessionState(session_id=session_id)

    def test_put_and_get(self) -> None:
        store = self._make_store()
        state = self._make_state("s1")
        store.put(state)
        retrieved = store.get("s1")
        assert retrieved is not None
        assert retrieved.session_id == "s1"

    def test_get_missing_returns_none(self) -> None:
        store = self._make_store()
        assert store.get("nonexistent") is None

    def test_delete_removes_session(self) -> None:
        store = self._make_store()
        state = self._make_state("s2")
        store.put(state)
        store.delete("s2")
        assert store.get("s2") is None

    def test_add_turn_appends_history(self) -> None:
        store = self._make_store()
        state = self._make_state("s3")
        store.put(state)
        store.add_turn("s3", "user", "Hello!")
        updated = store.get("s3")
        assert updated is not None
        assert len(updated.history_messages) == 1
        assert updated.history_messages[0].content == "Hello!"

    def test_add_turn_unknown_session_returns_false(self) -> None:
        store = self._make_store()
        result = store.add_turn("ghost-session", "user", "Hi")
        assert result is False

    def test_max_turns_enforced(self) -> None:
        store = InMemorySessionStore(ttl_seconds=3600, max_history_turns=3, eviction_interval=0)
        state = self._make_state("s4")
        store.put(state)
        for i in range(5):
            store.add_turn("s4", "user", f"Turn {i}")
        updated = store.get("s4")
        assert updated is not None
        assert len(updated.history_messages) == 3
        # Most recent 3 turns should be retained
        assert updated.history_messages[-1].content == "Turn 4"

    def test_ttl_eviction(self) -> None:
        store = self._make_store(ttl=1)
        state = self._make_state("s5")
        store.put(state)
        # Manually trigger eviction after TTL would expire
        # We patch monotonic to simulate passage of time
        import unittest.mock as mock

        original_time = time.monotonic()
        with mock.patch("modules.history.session_store.time") as mock_time:
            mock_time.monotonic.return_value = original_time + 10  # 10s > 1s TTL
            store._evict_expired()
        assert store.get("s5") is None

    def test_active_session_count(self) -> None:
        store = self._make_store()
        assert store.active_session_count() == 0
        store.put(self._make_state("a"))
        store.put(self._make_state("b"))
        assert store.active_session_count() == 2
        store.delete("a")
        assert store.active_session_count() == 1

    def test_upsert_overwrites_existing(self) -> None:
        store = self._make_store()
        state1 = SessionState(session_id="s6", canary_token="old-token")
        store.put(state1)
        state2 = SessionState(session_id="s6", canary_token="new-token")
        store.put(state2)
        retrieved = store.get("s6")
        assert retrieved is not None
        assert retrieved.canary_token == "new-token"


# ─────────────────────────────────────────────────────────────────────────────
# DefaultHistoryManager Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultHistoryManager:
    """Integration tests for the composed history manager."""

    def setup_method(self) -> None:
        store = InMemorySessionStore(eviction_interval=0)
        self.manager = DefaultHistoryManager(store=store)

    def _msg(self, role: str, content: str) -> ChatMessage:
        return ChatMessage(role=MessageRole(role), content=content)

    def test_ensure_session_creates_new(self) -> None:
        state = self.manager.ensure_session("brand-new")
        assert state.session_id == "brand-new"
        assert state.history_messages == []

    def test_ensure_session_returns_existing(self) -> None:
        state1 = self.manager.ensure_session("existing")
        state1_updated = state1.model_copy(update={"canary_token": "xyz"})
        self.manager.upsert_session(state1_updated)
        state2 = self.manager.ensure_session("existing")
        assert state2.canary_token == "xyz"

    def test_split_immutable_zone_delegates_correctly(self) -> None:
        msgs = [
            self._msg("system", "System"),
            self._msg("user", "User"),
        ]
        immutable, history = self.manager.split_immutable_zone(msgs)
        assert len(immutable) == 1
        assert len(history) == 1

    def test_add_turn_and_get_history(self) -> None:
        self.manager.ensure_session("h1")
        self.manager.add_turn("h1", "user", "Hello")
        self.manager.add_turn("h1", "assistant", "Hi!")
        history = self.manager.get_history("h1")
        assert len(history) == 2
        assert history[0].content == "Hello"
        assert history[1].content == "Hi!"

    def test_add_turn_missing_session_raises(self) -> None:
        with pytest.raises(SessionNotFoundException):
            self.manager.add_turn("ghost", "user", "This will fail.")

    def test_get_history_unknown_session_returns_empty(self) -> None:
        history = self.manager.get_history("never-created")
        assert history == []

    def test_delete_session(self) -> None:
        self.manager.ensure_session("d1")
        self.manager.delete_session("d1")
        assert self.manager.get_session("d1") is None

    def test_pii_map_stored_in_session(self) -> None:
        state = self.manager.ensure_session("pii-session")
        updated = state.model_copy(update={"pii_map": {"[EMAIL_1]": "user@test.com"}})
        self.manager.upsert_session(updated)
        retrieved = self.manager.get_session("pii-session")
        assert retrieved is not None
        assert retrieved.pii_map == {"[EMAIL_1]": "user@test.com"}
