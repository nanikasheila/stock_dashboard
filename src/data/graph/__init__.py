"""Public API for the graph store package.

Why: External callers use ``from src.data.graph import merge_stock`` etc.
     This __init__.py re-exports all public names so the package surface
     matches what the original monolithic graph_store.py provided.
How: Explicitly imports every public symbol from the four sub-modules.
     Private helpers (_get_mode, _get_driver, _set_embedding, etc.) are
     also re-exported so advanced callers and tests can access them.

Sub-modules:
- ``connection`` — driver lifecycle and write-mode detection.
- ``schema``     — constraints, indexes, and embedding helper.
- ``queries``    — Cypher string constants (internal; not part of public API).
- ``repository`` — CRUD / query orchestration (public functions listed below).
"""

from __future__ import annotations

# Connection layer
from src.data.graph.connection import (
    _MODE_TTL,
    _NEO4J_PASSWORD,
    _NEO4J_URI,
    _NEO4J_USER,
    _get_driver,
    _get_mode,
    close,
    get_mode,
    is_available,
)

# Repository layer
from src.data.graph.repository import (
    _safe_id,
    _truncate,
    clear_all,
    get_held_symbols,
    get_stock_history,
    is_held,
    link_research_supersedes,
    merge_health,
    merge_market_context,
    merge_market_context_full,
    merge_note,
    merge_report,
    merge_report_full,
    merge_research,
    merge_research_full,
    merge_screen,
    merge_stock,
    merge_trade,
    merge_watchlist,
    sync_portfolio,
    tag_theme,
)

# Schema layer
from src.data.graph.schema import (
    _SCHEMA_CONSTRAINTS,
    _SCHEMA_INDEXES,
    _VECTOR_INDEXES,
    _set_embedding,
    init_schema,
)

__all__ = [
    # connection
    "get_mode",
    "is_available",
    "close",
    # schema
    "init_schema",
    # repository — stock
    "merge_stock",
    # repository — CRUD
    "merge_screen",
    "merge_report",
    "merge_trade",
    "merge_health",
    "merge_note",
    "tag_theme",
    "merge_research",
    "merge_watchlist",
    "link_research_supersedes",
    "sync_portfolio",
    "is_held",
    "get_held_symbols",
    "merge_market_context",
    "clear_all",
    # repository — full mode
    "merge_report_full",
    "merge_research_full",
    "merge_market_context_full",
    # repository — queries
    "get_stock_history",
]
