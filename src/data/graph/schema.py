"""Neo4j schema: constraints, indexes, and embedding helper.

Why: Schema initialisation and the embedding-set helper are logically
     separate from CRUD operations and need to be imported by repository.py.
How: Delegates driver access to ``connection._get_driver()``.
     ``init_schema()`` is idempotent (uses IF NOT EXISTS).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.graph.connection import _get_driver, _get_mode  # noqa: F401 (re-used by callers)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT stock_symbol IF NOT EXISTS FOR (s:Stock) REQUIRE s.symbol IS UNIQUE",
    "CREATE CONSTRAINT screen_id IF NOT EXISTS FOR (s:Screen) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT trade_id IF NOT EXISTS FOR (t:Trade) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT health_id IF NOT EXISTS FOR (h:HealthCheck) REQUIRE h.id IS UNIQUE",
    "CREATE CONSTRAINT note_id IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT research_id IF NOT EXISTS FOR (r:Research) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT watchlist_name IF NOT EXISTS FOR (w:Watchlist) REQUIRE w.name IS UNIQUE",
    "CREATE CONSTRAINT market_context_id IF NOT EXISTS FOR (m:MarketContext) REQUIRE m.id IS UNIQUE",
    # KIK-413 full-mode nodes
    "CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:News) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sentiment_id IF NOT EXISTS FOR (s:Sentiment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT catalyst_id IF NOT EXISTS FOR (c:Catalyst) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT analyst_view_id IF NOT EXISTS FOR (a:AnalystView) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT indicator_id IF NOT EXISTS FOR (i:Indicator) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT upcoming_event_id IF NOT EXISTS FOR (e:UpcomingEvent) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT sector_rotation_id IF NOT EXISTS FOR (r:SectorRotation) REQUIRE r.id IS UNIQUE",
    # KIK-414 portfolio sync
    "CREATE CONSTRAINT portfolio_name IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.name IS UNIQUE",
]

_SCHEMA_INDEXES = [
    "CREATE INDEX stock_sector IF NOT EXISTS FOR (s:Stock) ON (s.sector)",
    "CREATE INDEX screen_date IF NOT EXISTS FOR (s:Screen) ON (s.date)",
    "CREATE INDEX report_date IF NOT EXISTS FOR (r:Report) ON (r.date)",
    "CREATE INDEX trade_date IF NOT EXISTS FOR (t:Trade) ON (t.date)",
    "CREATE INDEX note_type IF NOT EXISTS FOR (n:Note) ON (n.type)",
    "CREATE INDEX research_date IF NOT EXISTS FOR (r:Research) ON (r.date)",
    "CREATE INDEX research_type IF NOT EXISTS FOR (r:Research) ON (r.research_type)",
    "CREATE INDEX market_context_date IF NOT EXISTS FOR (m:MarketContext) ON (m.date)",
    # KIK-413 full-mode indexes
    "CREATE INDEX news_date IF NOT EXISTS FOR (n:News) ON (n.date)",
    "CREATE INDEX sentiment_source IF NOT EXISTS FOR (s:Sentiment) ON (s.source)",
    "CREATE INDEX catalyst_type IF NOT EXISTS FOR (c:Catalyst) ON (c.type)",
    "CREATE INDEX indicator_date IF NOT EXISTS FOR (i:Indicator) ON (i.date)",
]

# KIK-420: Vector indexes for semantic search
_VECTOR_INDEXES = [
    "CREATE VECTOR INDEX screen_embedding IF NOT EXISTS FOR (s:Screen) ON (s.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX report_embedding IF NOT EXISTS FOR (r:Report) ON (r.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX trade_embedding IF NOT EXISTS FOR (t:Trade) ON (t.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX healthcheck_embedding IF NOT EXISTS FOR (h:HealthCheck) ON (h.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX research_embedding IF NOT EXISTS FOR (r:Research) ON (r.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX marketcontext_embedding IF NOT EXISTS FOR (m:MarketContext) ON (m.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX note_embedding IF NOT EXISTS FOR (n:Note) ON (n.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    "CREATE VECTOR INDEX watchlist_embedding IF NOT EXISTS FOR (w:Watchlist) ON (w.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
]


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_schema() -> bool:
    """Create constraints and indexes. Returns True on success.

    Why: Schema must exist before any MERGE/MATCH operations to guarantee
         uniqueness and query performance.
    How: Runs all ``_SCHEMA_CONSTRAINTS`` and ``_SCHEMA_INDEXES`` first,
         then attempts vector indexes in a separate try/except because
         older Neo4j versions do not support the VECTOR INDEX syntax.
    """
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            for stmt in _SCHEMA_CONSTRAINTS + _SCHEMA_INDEXES:
                session.run(stmt)
            # KIK-420: Vector indexes -- skip silently on unsupported Neo4j versions
            for stmt in _VECTOR_INDEXES:
                try:
                    session.run(stmt)
                except Exception:
                    pass
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Embedding helper (KIK-420)
# ---------------------------------------------------------------------------

def _set_embedding(
    session,
    label: str,
    node_id: str,
    semantic_summary: str = "",
    embedding: Optional[list[float]] = None,
) -> None:
    """Set semantic_summary and embedding on a node if provided.

    Why: Avoids duplicating the SET logic in every merge function.
    How: Builds a parameterised Cypher SET statement dynamically and runs
         it only when at least one of the two values is non-empty/non-None.
    """
    if not semantic_summary and embedding is None:
        return
    sets = []
    params: dict = {"id": node_id}
    if semantic_summary:
        sets.append("n.semantic_summary = $summary")
        params["summary"] = semantic_summary
    if embedding is not None:
        sets.append("n.embedding = $embedding")
        params["embedding"] = embedding
    if sets:
        query = f"MATCH (n:{label} {{id: $id}}) SET {', '.join(sets)}"
        session.run(query, **params)
