"""
main.py
────────
Gateway entrypoint for the Universal LLM Middleware reverse proxy.

Launch::

    python main.py

Or with uvicorn directly::

    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Environment Configuration:
  All settings are loaded from environment variables or a ``.env`` file.
  See ``config/settings.py`` for the full list of available options.

Key variables:
  UPSTREAM_LLM_URL     — Base URL of the upstream LLM API.
  UPSTREAM_API_KEY     — Bearer token for the upstream provider.
  HOST                 — Bind host (default: 0.0.0.0).
  PORT                 — Bind port (default: 8080).
  LOG_LEVEL            — Logging level (default: INFO).
  INJECTION_THRESHOLD  — Threat score threshold (default: 0.58).
  DRIFT_THRESHOLD      — Compression drift threshold (default: 0.90).
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from config.settings import get_settings
from interfaces.reverse_proxy import create_app

# ── Logging bootstrap ─────────────────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Application instance ──────────────────────────────────────────────────────

app = create_app()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "Starting Universal LLM Middleware gateway on %s:%d",
        settings.host,
        settings.port,
    )
    logger.info("  Upstream LLM : %s", settings.upstream_llm_url)
    logger.info("  Log level    : %s", settings.log_level)
    logger.info("  Inj threshold: %.2f", settings.injection_threshold)
    logger.info("  Drift threshold: %.2f", settings.drift_threshold)

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
