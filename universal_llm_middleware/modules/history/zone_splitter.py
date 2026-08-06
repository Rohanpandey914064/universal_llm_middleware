"""
modules/history/zone_splitter.py
──────────────────────────────────
Immutable Zone vs. Mutable History partitioner.

The splitter enforces the fundamental architectural invariant:

  IMMUTABLE ZONE (never compressed, never PII-masked):
    • Messages with role ``system``   — OpenAI-style system instructions.
    • Messages with role ``developer``— Anthropic / OpenAI o-series extension.

  MUTABLE HISTORY (subject to compression and PII masking):
    • All other roles: ``user``, ``assistant``, ``tool``, ``function``.

Ordering guarantee:
  The relative order within each partition is preserved.  Immutable messages
  that appear interleaved among history turns are extracted and placed first
  (before any history message) so that the LLM always sees directives before
  conversation context.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from core.schemas import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

_IMMUTABLE_ROLES: frozenset[MessageRole] = frozenset(
    {MessageRole.SYSTEM, MessageRole.DEVELOPER}
)


class ZoneSplitter:
    """
    Partitions a message list into immutable directives and mutable history.

    This class is deliberately stateless — all methods are pure functions on
    the input list.  No session state is mutated.

    Example::

        splitter = ZoneSplitter()
        immutable, history = splitter.split([
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi!"),
        ])
        # immutable → [system message]
        # history   → [user message, assistant message]
    """

    def split(
        self,
        messages: List[ChatMessage],
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """
        Partition ``messages`` into immutable and mutable segments.

        Args:
            messages: Full incoming message list (arbitrary order).

        Returns:
            ``(immutable_messages, history_messages)`` where:
              • ``immutable_messages`` — system/developer messages in original
                                        relative order.
              • ``history_messages``   — all other messages in original
                                        relative order.
        """
        immutable: List[ChatMessage] = []
        history: List[ChatMessage] = []

        for msg in messages:
            if msg.role in _IMMUTABLE_ROLES:
                immutable.append(msg)
            else:
                history.append(msg)

        logger.debug(
            "Zone split complete — %d immutable, %d history messages.",
            len(immutable),
            len(history),
        )
        return immutable, history

    def merge(
        self,
        immutable_messages: List[ChatMessage],
        history_messages: List[ChatMessage],
    ) -> List[ChatMessage]:
        """
        Re-combine immutable and history segments into a single ordered list.

        Immutable messages are always prepended so the LLM receives directives
        before conversational context.

        Args:
            immutable_messages: System / developer messages.
            history_messages:   Mutable (possibly compressed) history turns.

        Returns:
            Combined ordered list: ``[immutable...] + [history...]``.
        """
        merged = list(immutable_messages) + list(history_messages)
        logger.debug(
            "Zone merge complete — %d total messages.", len(merged)
        )
        return merged

    @staticmethod
    def is_immutable(message: ChatMessage) -> bool:
        """Return ``True`` if ``message`` belongs to the immutable zone."""
        return message.role in _IMMUTABLE_ROLES
