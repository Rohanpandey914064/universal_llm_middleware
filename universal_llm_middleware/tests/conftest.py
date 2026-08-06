"""
tests/conftest.py
──────────────────
Shared pytest fixtures and configuration for the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so all imports resolve correctly
# when running pytest from the repo root.
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


import pytest
from modules.history.session_store import InMemorySessionStore


@pytest.fixture(autouse=True)
def clean_settings_cache() -> None:
    """Clear the Settings LRU cache before each test for isolation."""
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def session_store() -> InMemorySessionStore:
    """Return a fresh InMemorySessionStore with background eviction disabled."""
    return InMemorySessionStore(eviction_interval=0)
