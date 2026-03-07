"""Cypher query strings for the investment knowledge graph.

Why: Centralises every hard-coded Cypher template that was previously
     scattered across repository.py so that query intent is visible at a
     glance without digging through execution code.
How: Pure string constants grouped by domain (Stock, Screen, Report, …).
     repository.py imports these names instead of writing literals inline.
     No driver, session, or business logic lives here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------

STOCK_MERGE = "MERGE (s:Stock {symbol: $symbol}) SET s.name = $name, s.sector = $sector, s.country = $country"

STOCK_LINK_SECTOR = (
    "MERGE (sec:Sector {name: $sector}) WITH sec MATCH (s:Stock {symbol: $symbol}) MERGE (s)-[:IN_SECTOR]->(sec)"
)

# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

SCREEN_MERGE = (
    "MERGE (sc:Screen {id: $id}) SET sc.date = $date, sc.preset = $preset, sc.region = $region, sc.count = $count"
)

SCREEN_LINK_STOCK = "MATCH (sc:Screen {id: $screen_id}) MERGE (s:Stock {symbol: $symbol}) MERGE (sc)-[:SURFACED]->(s)"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

REPORT_MERGE = (
    "MERGE (r:Report {id: $id}) SET r.date = $date, r.symbol = $symbol, r.score = $score, r.verdict = $verdict"
)

REPORT_LINK_STOCK = "MATCH (r:Report {id: $report_id}) MERGE (s:Stock {symbol: $symbol}) MERGE (r)-[:ANALYZED]->(s)"

# Full-mode only: extend an existing Report with valuation metrics.
REPORT_SET_VALUATION = (
    "MATCH (r:Report {id: $id}) "
    "SET r.price = $price, r.per = $per, r.pbr = $pbr, "
    "r.dividend_yield = $div, r.roe = $roe, r.market_cap = $mcap"
)

# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

TRADE_MERGE = (
    "MERGE (t:Trade {id: $id}) "
    "SET t.date = $date, t.type = $type, t.symbol = $symbol, "
    "t.shares = $shares, t.price = $price, t.currency = $currency, "
    "t.memo = $memo"
)


def trade_link_stock(rel_type: str) -> str:
    """Return the Cypher query to link a Trade to a Stock.

    Why: The relationship type ('BOUGHT' / 'SOLD') is determined at runtime
         by the trade direction, so the query cannot be a pure constant.
    How: Formats the rel_type into the Cypher template; double-braces are
         Cypher literal braces, not Python format tokens.

    Args:
        rel_type: Relationship type string, either ``'BOUGHT'`` or ``'SOLD'``.

    Returns:
        Cypher query string ready to pass to ``session.run()``.
    """
    return f"MATCH (t:Trade {{id: $trade_id}}) MERGE (s:Stock {{symbol: $symbol}}) MERGE (t)-[:{rel_type}]->(s)"


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

HEALTH_MERGE = (
    "MERGE (h:HealthCheck {id: $id}) "
    "SET h.date = $date, h.total = $total, "
    "h.healthy = $healthy, h.exit_count = $exit_count"
)

HEALTH_LINK_STOCK = "MATCH (h:HealthCheck {id: $health_id}) MERGE (s:Stock {symbol: $symbol}) MERGE (h)-[:CHECKED]->(s)"

# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------

NOTE_MERGE = "MERGE (n:Note {id: $id}) SET n.date = $date, n.type = $type, n.content = $content, n.source = $source"

NOTE_LINK_STOCK = "MATCH (n:Note {id: $note_id}) MERGE (s:Stock {symbol: $symbol}) MERGE (n)-[:ABOUT]->(s)"

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME_TAG_STOCK = "MERGE (t:Theme {name: $theme}) WITH t MERGE (s:Stock {symbol: $symbol}) MERGE (s)-[:HAS_THEME]->(t)"

# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

RESEARCH_MERGE = (
    "MERGE (r:Research {id: $id}) "
    "SET r.date = $date, r.research_type = $rtype, "
    "r.target = $target, r.summary = $summary"
)

RESEARCH_LINK_STOCK = (
    "MATCH (r:Research {id: $research_id}) MERGE (s:Stock {symbol: $symbol}) MERGE (r)-[:RESEARCHED]->(s)"
)

RESEARCH_SUPERSEDES_CHAIN = (
    "MATCH (r:Research {research_type: $rtype, target: $target}) "
    "WITH r ORDER BY r.date ASC "
    "WITH collect(r) AS nodes "
    "UNWIND range(0, size(nodes)-2) AS i "
    "WITH nodes[i] AS a, nodes[i+1] AS b "
    "MERGE (a)-[:SUPERSEDES]->(b)"
)

# --- Full-mode Research sub-nodes (KIK-413) ---

RESEARCH_NEWS_MERGE = (
    "MERGE (n:News {id: $id}) "
    "SET n.date = $date, n.title = $title, "
    "n.source = $source, n.link = $link "
    "WITH n "
    "MATCH (r:Research {id: $rid}) "
    "MERGE (r)-[:HAS_NEWS]->(n)"
)

RESEARCH_NEWS_LINK_STOCK = "MATCH (n:News {id: $nid}) MERGE (s:Stock {symbol: $symbol}) MERGE (n)-[:MENTIONS]->(s)"

RESEARCH_SENTIMENT_GROK_MERGE = (
    "MERGE (s:Sentiment {id: $id}) "
    "SET s.date = $date, s.source = 'grok_x', "
    "s.score = $score, s.summary = $summary "
    "WITH s "
    "MATCH (r:Research {id: $rid}) "
    "MERGE (r)-[:HAS_SENTIMENT]->(s)"
)

RESEARCH_SENTIMENT_YAHOO_MERGE = (
    "MERGE (s:Sentiment {id: $id}) "
    "SET s.date = $date, s.source = 'yahoo_x', "
    "s.positive = $pos, s.negative = $neg "
    "WITH s "
    "MATCH (r:Research {id: $rid}) "
    "MERGE (r)-[:HAS_SENTIMENT]->(s)"
)

RESEARCH_CATALYST_MERGE = (
    "MERGE (c:Catalyst {id: $id}) "
    "SET c.date = $date, c.type = $polarity, "
    "c.text = $text "
    "WITH c "
    "MATCH (r:Research {id: $rid}) "
    "MERGE (r)-[:HAS_CATALYST]->(c)"
)

RESEARCH_ANALYST_VIEW_MERGE = (
    "MERGE (a:AnalystView {id: $id}) "
    "SET a.date = $date, a.text = $text "
    "WITH a "
    "MATCH (r:Research {id: $rid}) "
    "MERGE (r)-[:HAS_ANALYST_VIEW]->(a)"
)

# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

WATCHLIST_MERGE = "MERGE (w:Watchlist {name: $name})"

WATCHLIST_LINK_STOCK = (
    "MATCH (w:Watchlist {name: $name}) MERGE (s:Stock {symbol: $symbol}) MERGE (w)-[:BOOKMARKED]->(s)"
)

# ---------------------------------------------------------------------------
# Portfolio (KIK-414)
# ---------------------------------------------------------------------------

PORTFOLIO_MERGE = "MERGE (p:Portfolio {name: 'default'})"

PORTFOLIO_STOCK_MERGE = "MERGE (s:Stock {symbol: $symbol})"

PORTFOLIO_HOLDS_UPSERT = (
    "MATCH (p:Portfolio {name: 'default'}) "
    "MATCH (s:Stock {symbol: $symbol}) "
    "MERGE (p)-[r:HOLDS]->(s) "
    "SET r.shares = $shares, r.cost_price = $cost_price, "
    "r.cost_currency = $cost_currency, "
    "r.purchase_date = $purchase_date"
)

# Remove HOLDS edges for stocks no longer in the current portfolio.
PORTFOLIO_REMOVE_STALE_HOLDS = (
    "MATCH (p:Portfolio {name: 'default'})-[r:HOLDS]->(s:Stock) WHERE NOT s.symbol IN $symbols DELETE r"
)

# Remove ALL HOLDS edges (used when current_symbols is empty).
PORTFOLIO_CLEAR_ALL_HOLDS = "MATCH (p:Portfolio {name: 'default'})-[r:HOLDS]->() DELETE r"

PORTFOLIO_IS_HELD = "MATCH (p:Portfolio {name: 'default'})-[:HOLDS]->(s:Stock {symbol: $symbol}) RETURN count(*) AS cnt"

PORTFOLIO_GET_HELD_SYMBOLS = "MATCH (p:Portfolio {name: 'default'})-[:HOLDS]->(s:Stock) RETURN s.symbol AS symbol"

# ---------------------------------------------------------------------------
# MarketContext (KIK-399 / KIK-413)
# ---------------------------------------------------------------------------

MARKET_CONTEXT_MERGE = "MERGE (m:MarketContext {id: $id}) SET m.date = $date, m.indices = $indices"

# Full-mode sub-nodes:

MARKET_CONTEXT_INDICATOR_MERGE = (
    "MERGE (ind:Indicator {id: $id}) "
    "SET ind.date = $date, ind.name = $name, "
    "ind.symbol = $symbol, ind.price = $price, "
    "ind.daily_change = $dchange, ind.weekly_change = $wchange "
    "WITH ind "
    "MATCH (m:MarketContext {id: $mid}) "
    "MERGE (m)-[:INCLUDES]->(ind)"
)

MARKET_CONTEXT_EVENT_MERGE = (
    "MERGE (e:UpcomingEvent {id: $id}) "
    "SET e.date = $date, e.text = $text "
    "WITH e "
    "MATCH (m:MarketContext {id: $mid}) "
    "MERGE (m)-[:HAS_EVENT]->(e)"
)

MARKET_CONTEXT_ROTATION_MERGE = (
    "MERGE (sr:SectorRotation {id: $id}) "
    "SET sr.date = $date, sr.text = $text "
    "WITH sr "
    "MATCH (m:MarketContext {id: $mid}) "
    "MERGE (m)-[:HAS_ROTATION]->(sr)"
)

MARKET_CONTEXT_SENTIMENT_MERGE = (
    "MERGE (s:Sentiment {id: $id}) "
    "SET s.date = $date, s.source = 'market', "
    "s.score = $score, s.summary = $summary "
    "WITH s "
    "MATCH (m:MarketContext {id: $mid}) "
    "MERGE (m)-[:HAS_SENTIMENT]->(s)"
)

# ---------------------------------------------------------------------------
# Administrative
# ---------------------------------------------------------------------------

CLEAR_ALL = "MATCH (n) DETACH DELETE n"

# ---------------------------------------------------------------------------
# get_stock_history read queries
# ---------------------------------------------------------------------------

HISTORY_SCREENS = (
    "MATCH (sc:Screen)-[:SURFACED]->(s:Stock {symbol: $symbol}) "
    "RETURN sc.date AS date, sc.preset AS preset, sc.region AS region "
    "ORDER BY sc.date DESC"
)

HISTORY_REPORTS = (
    "MATCH (r:Report)-[:ANALYZED]->(s:Stock {symbol: $symbol}) "
    "RETURN r.date AS date, r.score AS score, r.verdict AS verdict "
    "ORDER BY r.date DESC"
)

HISTORY_TRADES = (
    "MATCH (t:Trade)-[:BOUGHT|SOLD]->(s:Stock {symbol: $symbol}) "
    "RETURN t.date AS date, t.type AS type, "
    "t.shares AS shares, t.price AS price "
    "ORDER BY t.date DESC"
)

HISTORY_HEALTH = (
    "MATCH (h:HealthCheck)-[:CHECKED]->(s:Stock {symbol: $symbol}) RETURN h.date AS date ORDER BY h.date DESC"
)

HISTORY_NOTES = (
    "MATCH (n:Note)-[:ABOUT]->(s:Stock {symbol: $symbol}) "
    "RETURN n.id AS id, n.date AS date, n.type AS type, "
    "n.content AS content "
    "ORDER BY n.date DESC"
)

HISTORY_THEMES = "MATCH (s:Stock {symbol: $symbol})-[:HAS_THEME]->(t:Theme) RETURN t.name AS name"

HISTORY_RESEARCHES = (
    "MATCH (r:Research)-[:RESEARCHED]->(s:Stock {symbol: $symbol}) "
    "RETURN r.date AS date, r.research_type AS research_type, "
    "r.summary AS summary "
    "ORDER BY r.date DESC"
)
