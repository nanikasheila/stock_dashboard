"""Neo4j CRUD operations for the investment knowledge graph.

Why: All merge/query operations live here so that callers import a single
     module rather than scattering graph writes across business logic.
How: Each function obtains the driver via ``connection._get_driver()`` and
     checks write mode via ``connection._get_mode()`` before any I/O.
     ``schema._set_embedding()`` handles optional vector attachment.
     All writes use MERGE for idempotent behaviour.
     Hard-coded Cypher strings are sourced from ``queries`` to keep this
     module focused on orchestration rather than query authorship.
"""

from __future__ import annotations

import json as _json
import logging
import re

from src.data.graph import queries as _q
from src.data.graph.connection import _get_driver, _get_mode
from src.data.graph.schema import _set_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Neo4j exception types (neo4j is an optional dependency).
# Imported once at module load so individual functions can catch them by name
# rather than relying on broad ``except Exception``.  When the package is not
# installed, the tuple is empty and all ``except _NEO4J_ERRORS`` clauses
# silently catch nothing — the outer ``except Exception`` remains as the
# safety net.
# ---------------------------------------------------------------------------
try:
    from neo4j.exceptions import (  # type: ignore[import]
        AuthError as _Neo4jAuthError,
    )
    from neo4j.exceptions import (
        ClientError as _Neo4jClientError,
    )
    from neo4j.exceptions import (
        DatabaseError as _Neo4jDatabaseError,
    )
    from neo4j.exceptions import (
        ServiceUnavailable as _Neo4jServiceUnavailable,
    )
    from neo4j.exceptions import (
        TransientError as _Neo4jTransientError,
    )

    _NEO4J_ERRORS: tuple[type[Exception], ...] = (
        _Neo4jServiceUnavailable,
        _Neo4jAuthError,
        _Neo4jClientError,
        _Neo4jTransientError,
        _Neo4jDatabaseError,
    )
except ImportError:
    # neo4j package not installed — graceful degradation; callers check mode/driver
    _NEO4J_ERRORS = ()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_id(text: str) -> str:
    """Make text safe for use in a node ID (replace non-alphanum with _).

    Why: Node IDs are constructed from user-supplied strings that may
         contain spaces, slashes, or other special characters.
    How: Replaces every non-alphanumeric character with ``_`` via regex.
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", text)


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text to max_len characters.

    Why: Neo4j property values should be bounded to avoid excessive storage
         and to stay within Cypher parameter limits.
    How: Slices the string at ``max_len``; coerces non-strings via ``str()``.
    """
    if not isinstance(text, str):
        return str(text)[:max_len] if text else ""
    return text[:max_len]


# ---------------------------------------------------------------------------
# Stock node
# ---------------------------------------------------------------------------


def merge_stock(symbol: str, name: str = "", sector: str = "", country: str = "") -> bool:
    """Create or update a Stock node.

    Why: Stock nodes are the central hub of the knowledge graph; all other
         entities reference them via relationships.
    How: MERGE on ``symbol`` (unique constraint), then creates an IN_SECTOR
         relationship if a sector is provided.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(
                _q.STOCK_MERGE,
                symbol=symbol,
                name=name,
                sector=sector,
                country=country,
            )
            if sector:
                session.run(
                    _q.STOCK_LINK_SECTOR,
                    sector=sector,
                    symbol=symbol,
                )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_stock failed for symbol=%r: %s", symbol, exc)
        return False
    except Exception as exc:
        logger.warning("merge_stock unexpected error for symbol=%r: %s", symbol, exc)
        return False


# ---------------------------------------------------------------------------
# Screen node
# ---------------------------------------------------------------------------


def merge_screen(
    screen_date: str,
    preset: str,
    region: str,
    count: int,
    symbols: list[str],
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Screen node and SURFACED relationships to stocks.

    Why: Records screening events so the graph captures which stocks were
         surfaced by which criteria on which date.
    How: MERGE on a composite ID, then creates SURFACED→Stock edges.
         Embedding is attached via ``_set_embedding`` if provided.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    screen_id = f"screen_{screen_date}_{region}_{preset}"
    try:
        with driver.session() as session:
            session.run(
                _q.SCREEN_MERGE,
                id=screen_id,
                date=screen_date,
                preset=preset,
                region=region,
                count=count,
            )
            _set_embedding(session, "Screen", screen_id, semantic_summary, embedding)
            for sym in symbols:
                session.run(
                    _q.SCREEN_LINK_STOCK,
                    screen_id=screen_id,
                    symbol=sym,
                )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_screen failed for screen_id=%r: %s", screen_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_screen unexpected error for screen_id=%r: %s", screen_id, exc)
        return False


# ---------------------------------------------------------------------------
# Report node
# ---------------------------------------------------------------------------


def merge_report(
    report_date: str,
    symbol: str,
    score: float,
    verdict: str,
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Report node and ANALYZED relationship.

    Why: Persists LLM analysis results so past verdicts can be queried
         and compared over time.
    How: MERGE on ``report_{date}_{symbol}`` ID, then creates ANALYZED→Stock.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    report_id = f"report_{report_date}_{symbol}"
    try:
        with driver.session() as session:
            session.run(
                _q.REPORT_MERGE,
                id=report_id,
                date=report_date,
                symbol=symbol,
                score=score,
                verdict=verdict,
            )
            session.run(
                _q.REPORT_LINK_STOCK,
                report_id=report_id,
                symbol=symbol,
            )
            _set_embedding(session, "Report", report_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_report failed for report_id=%r: %s", report_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_report unexpected error for report_id=%r: %s", report_id, exc)
        return False


# ---------------------------------------------------------------------------
# Trade node
# ---------------------------------------------------------------------------


def merge_trade(
    trade_date: str,
    trade_type: str,
    symbol: str,
    shares: int,
    price: float,
    currency: str,
    memo: str = "",
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Trade node and BOUGHT/SOLD relationship.

    Why: Records actual trade execution for portfolio history queries.
    How: Relationship type is chosen from ``trade_type`` ('buy' → BOUGHT,
         else SOLD). MERGE prevents duplicates on re-import.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    trade_id = f"trade_{trade_date}_{trade_type}_{symbol}"
    rel_type = "BOUGHT" if trade_type == "buy" else "SOLD"
    try:
        with driver.session() as session:
            session.run(
                _q.TRADE_MERGE,
                id=trade_id,
                date=trade_date,
                type=trade_type,
                symbol=symbol,
                shares=shares,
                price=price,
                currency=currency,
                memo=memo,
            )
            session.run(
                _q.trade_link_stock(rel_type),
                trade_id=trade_id,
                symbol=symbol,
            )
            _set_embedding(session, "Trade", trade_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_trade failed for trade_id=%r: %s", trade_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_trade unexpected error for trade_id=%r: %s", trade_id, exc)
        return False


# ---------------------------------------------------------------------------
# HealthCheck node
# ---------------------------------------------------------------------------


def merge_health(
    health_date: str,
    summary: dict,
    symbols: list[str],
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a HealthCheck node and CHECKED relationships.

    Why: Stores portfolio health snapshots so degradation trends can be
         tracked over time in the graph.
    How: MERGE on ``health_{date}`` ID, then creates CHECKED→Stock edges
         for every symbol in the batch.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    health_id = f"health_{health_date}"
    try:
        with driver.session() as session:
            session.run(
                _q.HEALTH_MERGE,
                id=health_id,
                date=health_date,
                total=summary.get("total", 0),
                healthy=summary.get("healthy", 0),
                exit_count=summary.get("exit", 0),
            )
            for sym in symbols:
                session.run(
                    _q.HEALTH_LINK_STOCK,
                    health_id=health_id,
                    symbol=sym,
                )
            _set_embedding(session, "HealthCheck", health_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_health failed for health_id=%r: %s", health_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_health unexpected error for health_id=%r: %s", health_id, exc)
        return False


# ---------------------------------------------------------------------------
# Note node
# ---------------------------------------------------------------------------


def merge_note(
    note_id: str,
    note_date: str,
    note_type: str,
    content: str,
    symbol: str | None = None,
    source: str = "",
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Note node and ABOUT relationship to a stock.

    Why: Captures free-form annotations (news, memos) linked to stocks
         for later semantic retrieval.
    How: MERGE on ``note_id`` provided by the caller; optional ABOUT→Stock
         relationship is created only when a symbol is given.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(
                _q.NOTE_MERGE,
                id=note_id,
                date=note_date,
                type=note_type,
                content=content,
                source=source,
            )
            if symbol:
                session.run(
                    _q.NOTE_LINK_STOCK,
                    note_id=note_id,
                    symbol=symbol,
                )
            _set_embedding(session, "Note", note_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_note failed for note_id=%r: %s", note_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_note unexpected error for note_id=%r: %s", note_id, exc)
        return False


# ---------------------------------------------------------------------------
# Theme tagging
# ---------------------------------------------------------------------------


def tag_theme(symbol: str, theme: str) -> bool:
    """Tag a stock with a theme.

    Why: Theme edges enable sector/macro grouping queries across the graph.
    How: MERGE ensures each (Stock)-[:HAS_THEME]->(Theme) edge is created once.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(
                _q.THEME_TAG_STOCK,
                theme=theme,
                symbol=symbol,
            )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("tag_theme failed for symbol=%r theme=%r: %s", symbol, theme, exc)
        return False
    except Exception as exc:
        logger.warning("tag_theme unexpected error for symbol=%r theme=%r: %s", symbol, theme, exc)
        return False


# ---------------------------------------------------------------------------
# Research node (KIK-398)
# ---------------------------------------------------------------------------


def merge_research(
    research_date: str,
    research_type: str,
    target: str,
    summary: str = "",
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Research node and optionally RESEARCHED relationship to Stock.

    For stock/business types, target is treated as a symbol and linked to Stock.
    For industry/market types, no Stock link is created.

    Why: Persists Grok/LLM research results so related research can be
         traversed in the graph over time.
    How: MERGE on composite ID including research_type and safe-encoded target.
         Stock link is created only for stock/business research types.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    research_id = f"research_{research_date}_{research_type}_{_safe_id(target)}"
    try:
        with driver.session() as session:
            session.run(
                _q.RESEARCH_MERGE,
                id=research_id,
                date=research_date,
                rtype=research_type,
                target=target,
                summary=summary,
            )
            if research_type in ("stock", "business"):
                session.run(
                    _q.RESEARCH_LINK_STOCK,
                    research_id=research_id,
                    symbol=target,
                )
            _set_embedding(session, "Research", research_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_research failed for research_id=%r: %s", research_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_research unexpected error for research_id=%r: %s", research_id, exc)
        return False


# ---------------------------------------------------------------------------
# Watchlist node (KIK-398)
# ---------------------------------------------------------------------------


def merge_watchlist(
    name: str,
    symbols: list[str],
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create a Watchlist node and BOOKMARKED relationships to stocks.

    Why: Watchlists organise candidate stocks before full analysis or trading.
    How: Watchlist uses ``name`` as its unique key (not ``id``), so embedding
         is set with a custom MATCH on ``name`` rather than via ``_set_embedding``.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(
                _q.WATCHLIST_MERGE,
                name=name,
            )
            # Why: Watchlist uses 'name' as key, not 'id'; handle embedding inline
            if semantic_summary or embedding is not None:
                _sets = []
                _params: dict = {"name": name}
                if semantic_summary:
                    _sets.append("w.semantic_summary = $summary")
                    _params["summary"] = semantic_summary
                if embedding is not None:
                    _sets.append("w.embedding = $embedding")
                    _params["embedding"] = embedding
                if _sets:
                    session.run(
                        f"MATCH (w:Watchlist {{name: $name}}) SET {', '.join(_sets)}",
                        **_params,
                    )
            for sym in symbols:
                session.run(
                    _q.WATCHLIST_LINK_STOCK,
                    name=name,
                    symbol=sym,
                )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_watchlist failed for name=%r: %s", name, exc)
        return False
    except Exception as exc:
        logger.warning("merge_watchlist unexpected error for name=%r: %s", name, exc)
        return False


# ---------------------------------------------------------------------------
# Research SUPERSEDES chain (KIK-398)
# ---------------------------------------------------------------------------


def link_research_supersedes(research_type: str, target: str) -> bool:
    """Link Research nodes of same type+target in date order with SUPERSEDES.

    Why: Supersedes edges let callers traverse to the most recent research
         for a given topic without filtering by date in application code.
    How: Collects all matching Research nodes ordered by date, then creates
         SUPERSEDES edges between consecutive pairs in a single Cypher query.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(
                _q.RESEARCH_SUPERSEDES_CHAIN,
                rtype=research_type,
                target=target,
            )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning(
            "link_research_supersedes failed for type=%r target=%r: %s",
            research_type,
            target,
            exc,
        )
        return False
    except Exception as exc:
        logger.warning(
            "link_research_supersedes unexpected error for type=%r target=%r: %s",
            research_type,
            target,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Portfolio sync (KIK-414)
# ---------------------------------------------------------------------------


def sync_portfolio(holdings: list[dict]) -> bool:
    """Sync portfolio CSV holdings to Neo4j HOLDS relationships.

    Creates a Portfolio anchor node and HOLDS relationships to each Stock.
    Removes HOLDS for stocks no longer in the portfolio.
    Cash positions (*.CASH) are excluded.

    Why: The graph must reflect current portfolio composition so ``is_held``
         and ``get_held_symbols`` serve live data without re-reading CSV.
    How: Upserts HOLDS with cost/share data, then deletes stale HOLDS edges
         in a single MATCH/DELETE query using the computed current symbol list.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    try:
        from src.core.common import is_cash  # Why: avoid circular import at module level

        with driver.session() as session:
            session.run(_q.PORTFOLIO_MERGE)

            current_symbols = []
            for h in holdings:
                symbol = h.get("symbol", "")
                if not symbol or is_cash(symbol):
                    continue
                current_symbols.append(symbol)
                session.run(
                    _q.PORTFOLIO_STOCK_MERGE,
                    symbol=symbol,
                )
                session.run(
                    _q.PORTFOLIO_HOLDS_UPSERT,
                    symbol=symbol,
                    shares=int(h.get("shares", 0)),
                    cost_price=float(h.get("cost_price", 0)),
                    cost_currency=h.get("cost_currency", "JPY"),
                    purchase_date=h.get("purchase_date", ""),
                )

            if current_symbols:
                session.run(
                    _q.PORTFOLIO_REMOVE_STALE_HOLDS,
                    symbols=current_symbols,
                )
            else:
                session.run(_q.PORTFOLIO_CLEAR_ALL_HOLDS)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("sync_portfolio failed (%d holdings): %s", len(holdings), exc)
        return False
    except Exception as exc:
        logger.warning("sync_portfolio unexpected error (%d holdings): %s", len(holdings), exc)
        return False


def is_held(symbol: str) -> bool:
    """Check if a symbol is currently held in the portfolio.

    Why: Enables rules-based filtering (e.g. skip screens for held stocks).
    How: Counts HOLDS edges via the 'default' Portfolio anchor node.
    """
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            result = session.run(
                _q.PORTFOLIO_IS_HELD,
                symbol=symbol,
            )
            record = result.single()
            return record["cnt"] > 0 if record else False
    except _NEO4J_ERRORS as exc:
        logger.warning("is_held failed for symbol=%r: %s", symbol, exc)
        return False
    except Exception as exc:
        logger.warning("is_held unexpected error for symbol=%r: %s", symbol, exc)
        return False


def get_held_symbols() -> list[str]:
    """Return symbols currently held in portfolio via HOLDS relationship.

    Why: Bulk retrieval of all held symbols avoids repeated ``is_held`` calls
         in batch operations.
    How: Single MATCH query traverses all HOLDS edges from 'default' Portfolio.
    """
    driver = _get_driver()
    if driver is None:
        return []
    try:
        with driver.session() as session:
            result = session.run(_q.PORTFOLIO_GET_HELD_SYMBOLS)
            return [r["symbol"] for r in result]
    except _NEO4J_ERRORS as exc:
        logger.warning("get_held_symbols failed: %s", exc)
        return []
    except Exception as exc:
        logger.warning("get_held_symbols unexpected error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# MarketContext node (KIK-399)
# ---------------------------------------------------------------------------


def merge_market_context(
    context_date: str,
    indices: list[dict],
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create/update a MarketContext node with index snapshots.

    indices is stored as a JSON string (Neo4j can't store list-of-maps).

    Why: Market context snapshots provide macro backdrop for stock analysis
         queries spanning a specific date.
    How: MERGE on ``market_context_{date}`` ID; serialises ``indices`` list
         to JSON string because Neo4j properties cannot hold list-of-maps.
    """
    if _get_mode() == "off":
        return False
    driver = _get_driver()
    if driver is None:
        return False
    context_id = f"market_context_{context_date}"
    try:
        with driver.session() as session:
            session.run(
                _q.MARKET_CONTEXT_MERGE,
                id=context_id,
                date=context_date,
                indices=_json.dumps(indices, ensure_ascii=False),
            )
            _set_embedding(session, "MarketContext", context_id, semantic_summary, embedding)
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_market_context failed for context_id=%r: %s", context_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_market_context unexpected error for context_id=%r: %s", context_id, exc)
        return False


# ---------------------------------------------------------------------------
# Clear all (KIK-398 --rebuild)
# ---------------------------------------------------------------------------


def clear_all() -> bool:
    """Delete all nodes and relationships. Used for --rebuild.

    Why: A full rebuild requires wiping the graph before re-importing.
    How: Single DETACH DELETE on all nodes; bypasses mode check because
         clear_all is an explicit administrative action.
    """
    driver = _get_driver()
    if driver is None:
        return False
    try:
        with driver.session() as session:
            session.run(_q.CLEAR_ALL)
        return True
    except _NEO4J_ERRORS as exc:
        logger.error("clear_all failed: %s", exc)
        return False
    except Exception as exc:
        logger.error("clear_all unexpected error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Full-mode CRUD (KIK-413)
# ---------------------------------------------------------------------------


def merge_report_full(
    report_date: str,
    symbol: str,
    score: float,
    verdict: str,
    price: float = 0,
    per: float = 0,
    pbr: float = 0,
    dividend_yield: float = 0,
    roe: float = 0,
    market_cap: float = 0,
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Extend an existing Report node with full valuation properties (KIK-413).

    Calls merge_report() first, then SETs additional numeric fields.
    Only runs in 'full' mode.

    Why: Summary mode stores only score/verdict; full mode also captures
         valuation metrics for richer historical comparison.
    How: Delegates to ``merge_report`` for the base node, then performs a
         separate SET for numeric fields. Falls back to summary mode gracefully.
    """
    if _get_mode() != "full":
        return merge_report(report_date, symbol, score, verdict, semantic_summary=semantic_summary, embedding=embedding)
    merge_report(report_date, symbol, score, verdict, semantic_summary=semantic_summary, embedding=embedding)
    driver = _get_driver()
    if driver is None:
        return False
    report_id = f"report_{report_date}_{symbol}"
    try:
        with driver.session() as session:
            session.run(
                _q.REPORT_SET_VALUATION,
                id=report_id,
                price=float(price or 0),
                per=float(per or 0),
                pbr=float(pbr or 0),
                div=float(dividend_yield or 0),
                roe=float(roe or 0),
                mcap=float(market_cap or 0),
            )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_report_full failed for report_id=%r: %s", report_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_report_full unexpected error for report_id=%r: %s", report_id, exc)
        return False


def merge_research_full(
    research_date: str,
    research_type: str,
    target: str,
    summary: str = "",
    grok_research: dict | None = None,
    x_sentiment: dict | None = None,
    news: list | None = None,
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create Research node with semantic sub-nodes (KIK-413).

    Expands grok_research data into News, Sentiment, Catalyst, AnalystView
    nodes connected to the Research node via relationships.
    Only creates sub-nodes in 'full' mode.

    Why: Full mode records the detailed evidence trail behind a research
         conclusion so the graph can answer "why was this stock flagged?".
    How: Delegates base node creation to ``merge_research``, then creates
         typed sub-nodes (News, Sentiment, Catalyst, AnalystView) with
         batch-limited list iteration to avoid runaway node counts.
    """
    if _get_mode() != "full":
        return merge_research(
            research_date, research_type, target, summary, semantic_summary=semantic_summary, embedding=embedding
        )
    merge_research(
        research_date, research_type, target, summary, semantic_summary=semantic_summary, embedding=embedding
    )
    driver = _get_driver()
    if driver is None:
        return False
    research_id = f"research_{research_date}_{research_type}_{_safe_id(target)}"
    try:
        with driver.session() as session:
            # --- News nodes (from grok recent_news + yahoo news) ---
            news_items: list[dict | str] = []
            if grok_research and isinstance(grok_research.get("recent_news"), list):
                for item in grok_research["recent_news"][:5]:
                    if isinstance(item, str):
                        news_items.append({"title": item, "source": "grok"})
                    elif isinstance(item, dict):
                        news_items.append({**item, "source": "grok"})
            if isinstance(news, list):
                for item in news[:5]:
                    if isinstance(item, dict):
                        news_items.append(
                            {
                                "title": item.get("title", ""),
                                "source": item.get("publisher", "yahoo"),
                                "link": item.get("link", ""),
                            }
                        )
            for i, nitem in enumerate(news_items[:10]):
                nid = f"{research_id}_news_{i}"
                title = _truncate(nitem.get("title", ""), 500)
                source = nitem.get("source", "")[:50]
                link = nitem.get("link", "")[:500]
                session.run(
                    _q.RESEARCH_NEWS_MERGE,
                    id=nid,
                    date=research_date,
                    title=title,
                    source=source,
                    link=link,
                    rid=research_id,
                )
                if research_type in ("stock", "business"):
                    session.run(
                        _q.RESEARCH_NEWS_LINK_STOCK,
                        nid=nid,
                        symbol=target,
                    )

            # --- Sentiment nodes ---
            if grok_research and isinstance(grok_research.get("x_sentiment"), dict):
                xs = grok_research["x_sentiment"]
                sid = f"{research_id}_sent_grok"
                session.run(
                    _q.RESEARCH_SENTIMENT_GROK_MERGE,
                    id=sid,
                    date=research_date,
                    score=float(xs.get("score", 0)),
                    summary=_truncate(xs.get("summary", ""), 500),
                    rid=research_id,
                )
            if isinstance(x_sentiment, dict) and x_sentiment:
                sid2 = f"{research_id}_sent_yahoo"
                pos = x_sentiment.get("positive", [])
                neg = x_sentiment.get("negative", [])
                pos_text = _truncate("; ".join(pos[:3]) if isinstance(pos, list) else str(pos), 500)
                neg_text = _truncate("; ".join(neg[:3]) if isinstance(neg, list) else str(neg), 500)
                session.run(
                    _q.RESEARCH_SENTIMENT_YAHOO_MERGE,
                    id=sid2,
                    date=research_date,
                    pos=pos_text,
                    neg=neg_text,
                    rid=research_id,
                )

            # --- Catalyst nodes ---
            if grok_research and isinstance(grok_research.get("catalysts"), dict):
                cats = grok_research["catalysts"]
                for polarity in ("positive", "negative"):
                    items = cats.get(polarity, [])
                    if isinstance(items, list):
                        for j, txt in enumerate(items[:5]):
                            cid = f"{research_id}_cat_{polarity[0]}_{j}"
                            session.run(
                                _q.RESEARCH_CATALYST_MERGE,
                                id=cid,
                                date=research_date,
                                polarity=polarity,
                                text=_truncate(str(txt), 500),
                                rid=research_id,
                            )

            # --- AnalystView nodes ---
            if grok_research and isinstance(grok_research.get("analyst_views"), list):
                for k, view_text in enumerate(grok_research["analyst_views"][:5]):
                    avid = f"{research_id}_av_{k}"
                    session.run(
                        _q.RESEARCH_ANALYST_VIEW_MERGE,
                        id=avid,
                        date=research_date,
                        text=_truncate(str(view_text), 500),
                        rid=research_id,
                    )
        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_research_full failed for research_id=%r: %s", research_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_research_full unexpected error for research_id=%r: %s", research_id, exc)
        return False


def merge_market_context_full(
    context_date: str,
    indices: list[dict],
    grok_research: dict | None = None,
    semantic_summary: str = "",
    embedding: list[float] | None = None,
) -> bool:
    """Create MarketContext with semantic sub-nodes (KIK-413).

    Expands indices into Indicator nodes, and grok_research into
    UpcomingEvent, SectorRotation, and Sentiment nodes.
    Only creates sub-nodes in 'full' mode.

    Why: Full mode captures the structured evidence for a market context
         date so macro conditions can be queried alongside stock analysis.
    How: Delegates base node to ``merge_market_context``, then creates typed
         sub-nodes capped at 20 indicators, 5 events, 3 rotations, 1 sentiment.
    """
    if _get_mode() != "full":
        return merge_market_context(context_date, indices, semantic_summary=semantic_summary, embedding=embedding)
    merge_market_context(context_date, indices, semantic_summary=semantic_summary, embedding=embedding)
    driver = _get_driver()
    if driver is None:
        return False
    context_id = f"market_context_{context_date}"
    try:
        with driver.session() as session:
            # --- Indicator nodes (from indices) ---
            for i, idx in enumerate(indices[:20]):
                iid = f"{context_id}_ind_{i}"
                session.run(
                    _q.MARKET_CONTEXT_INDICATOR_MERGE,
                    id=iid,
                    date=context_date,
                    name=str(idx.get("name", ""))[:100],
                    symbol=str(idx.get("symbol", ""))[:20],
                    price=float(idx.get("price", 0) or 0),
                    dchange=float(idx.get("daily_change", 0) or 0),
                    wchange=float(idx.get("weekly_change", 0) or 0),
                    mid=context_id,
                )

            if not grok_research:
                return True

            # --- UpcomingEvent nodes ---
            events = grok_research.get("upcoming_events", [])
            if isinstance(events, list):
                for j, ev in enumerate(events[:5]):
                    eid = f"{context_id}_event_{j}"
                    session.run(
                        _q.MARKET_CONTEXT_EVENT_MERGE,
                        id=eid,
                        date=context_date,
                        text=_truncate(str(ev), 500),
                        mid=context_id,
                    )

            # --- SectorRotation nodes ---
            rotations = grok_research.get("sector_rotation", [])
            if isinstance(rotations, list):
                for k, rot in enumerate(rotations[:3]):
                    rid = f"{context_id}_rot_{k}"
                    session.run(
                        _q.MARKET_CONTEXT_ROTATION_MERGE,
                        id=rid,
                        date=context_date,
                        text=_truncate(str(rot), 500),
                        mid=context_id,
                    )

            # --- Sentiment node (market-level) ---
            sentiment = grok_research.get("sentiment")
            if isinstance(sentiment, dict):
                sid = f"{context_id}_sent"
                session.run(
                    _q.MARKET_CONTEXT_SENTIMENT_MERGE,
                    id=sid,
                    date=context_date,
                    score=float(sentiment.get("score", 0)),
                    summary=_truncate(sentiment.get("summary", ""), 500),
                    mid=context_id,
                )

        return True
    except _NEO4J_ERRORS as exc:
        logger.warning("merge_market_context_full failed for context_id=%r: %s", context_id, exc)
        return False
    except Exception as exc:
        logger.warning("merge_market_context_full unexpected error for context_id=%r: %s", context_id, exc)
        return False


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_stock_history(symbol: str) -> dict:
    """Get all graph relationships for a stock.

    Returns dict with keys: screens, reports, trades, health_checks,
    notes, themes, researches.

    Why: Aggregates a stock's full event history from the graph for display
         in the dashboard stock-detail view.
    How: Issues one Cypher query per relationship type and collects results
         into a single dict; returns empty lists on driver absence or error.
    """
    _empty: dict = {
        "screens": [],
        "reports": [],
        "trades": [],
        "health_checks": [],
        "notes": [],
        "themes": [],
        "researches": [],
    }
    driver = _get_driver()
    if driver is None:
        return dict(_empty)
    try:
        result = dict(_empty)
        with driver.session() as session:
            records = session.run(
                _q.HISTORY_SCREENS,
                symbol=symbol,
            )
            result["screens"] = [dict(r) for r in records]

            records = session.run(
                _q.HISTORY_REPORTS,
                symbol=symbol,
            )
            result["reports"] = [dict(r) for r in records]

            records = session.run(
                _q.HISTORY_TRADES,
                symbol=symbol,
            )
            result["trades"] = [dict(r) for r in records]

            records = session.run(
                _q.HISTORY_HEALTH,
                symbol=symbol,
            )
            result["health_checks"] = [dict(r) for r in records]

            records = session.run(
                _q.HISTORY_NOTES,
                symbol=symbol,
            )
            result["notes"] = [dict(r) for r in records]

            records = session.run(
                _q.HISTORY_THEMES,
                symbol=symbol,
            )
            result["themes"] = [r["name"] for r in records]

            records = session.run(
                _q.HISTORY_RESEARCHES,
                symbol=symbol,
            )
            result["researches"] = [dict(r) for r in records]

        return result
    except _NEO4J_ERRORS as exc:
        logger.warning("get_stock_history failed for symbol=%r: %s", symbol, exc)
        return dict(_empty)
    except Exception as exc:
        logger.warning("get_stock_history unexpected error for symbol=%r: %s", symbol, exc)
        return dict(_empty)
