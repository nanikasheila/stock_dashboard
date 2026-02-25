"""Neo4j connection management and write-mode detection.

Why: Centralizes driver lifecycle and NEO4J_MODE logic so all other
     graph modules share a single connection without duplication.
How: Lazy-init driver via ``_get_driver()``. Mode is auto-detected once
     per ``_MODE_TTL`` seconds and cached to avoid repeated connectivity probes.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection configuration (from environment)
# ---------------------------------------------------------------------------

_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

_driver = None


# ---------------------------------------------------------------------------
# Write mode (KIK-413)
# ---------------------------------------------------------------------------

_mode_cache: tuple[str, float] = ("", 0.0)
_MODE_TTL = 30.0


def _get_mode() -> str:
    """Return Neo4j write mode: 'off', 'summary', or 'full'.

    Why: Callers need to know whether to write full semantic sub-nodes,
         summary-only, or skip Neo4j entirely (KIK-413).
    How: Env var ``NEO4J_MODE`` overrides auto-detection. Otherwise,
         connectivity is checked once per ``_MODE_TTL`` seconds and cached.
    """
    global _mode_cache
    env_mode = os.environ.get("NEO4J_MODE", "").lower()
    if env_mode in ("off", "summary", "full"):
        return env_mode
    now = time.time()
    if _mode_cache[0] and (now - _mode_cache[1]) < _MODE_TTL:
        return _mode_cache[0]
    mode = "full" if is_available() else "off"
    _mode_cache = (mode, now)
    return mode


def get_mode() -> str:
    """Public accessor for current Neo4j write mode.

    Why: External callers (e.g. history_store) need to query the write
         mode without importing private symbols.
    How: Delegates to ``_get_mode()``.
    """
    return _get_mode()


def _get_driver():
    """Lazy-init Neo4j driver. Returns None if neo4j package not installed.

    Why: Driver instantiation is expensive; defer until first use.
    How: Module-level ``_driver`` is initialised on first call and reused.
         Returns None silently so callers can skip writes gracefully.
    """
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase  # type: ignore[import]
        _driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
        return _driver
    except Exception:
        return None


def is_available() -> bool:
    """Check if Neo4j is reachable.

    Why: Determines whether to activate write mode at startup.
    How: Calls ``driver.verify_connectivity()``; returns False on any error.
    """
    driver = _get_driver()
    if driver is None:
        return False
    try:
        driver.verify_connectivity()
        return True
    except Exception:
        return False


def close() -> None:
    """Close the Neo4j driver.

    Why: Allows clean shutdown so the process does not hang on open sockets.
    How: Calls ``driver.close()`` and resets the module-level ``_driver`` to None.
    """
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
