"""Pure functions for computing trade statistics and style metrics.

All functions in this module are Streamlit-free and operate solely on
plain Python data structures (lists of dicts, dicts of floats).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

from src.core.behavior.models import (
    ConfidenceLevel,
    HoldingPeriodSummary,
    PortfolioTradeStats,
    SellRecord,
    StyleMetrics,
    TradeStats,
    WinLossSummary,
)
from src.core.common import is_cash

logger = logging.getLogger(__name__)

# Thresholds for confidence classification (based on number of sell transactions)
_MIN_SELLS_HIGH = 20
_MIN_SELLS_MEDIUM = 5

# Confidence ordering used for min/max operations
_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.INSUFFICIENT: 0,
    ConfidenceLevel.LOW: 1,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.HIGH: 3,
}


def min_confidence(a: ConfidenceLevel, b: ConfidenceLevel) -> ConfidenceLevel:
    """Return the lower of two confidence levels.

    Parameters
    ----------
    a, b : ConfidenceLevel
        The confidence levels to compare.

    Returns
    -------
    ConfidenceLevel
        The one with lower ordinal (i.e., less reliable).
    """
    return a if _CONFIDENCE_ORDER[a] <= _CONFIDENCE_ORDER[b] else b


def _classify_confidence(sell_count: int) -> ConfidenceLevel:
    """Map sell transaction count to a ConfidenceLevel."""
    if sell_count >= _MIN_SELLS_HIGH:
        return ConfidenceLevel.HIGH
    if sell_count >= _MIN_SELLS_MEDIUM:
        return ConfidenceLevel.MEDIUM
    if sell_count >= 1:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.INSUFFICIENT


def _parse_trade_date(date_str: str) -> date | None:
    """Parse a YYYY-MM-DD or YYYY/MM/DD date string, returning None on failure."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def _trade_amount_jpy(trade: dict, fx_rates: dict[str, float]) -> float:
    """Compute a trade's settlement amount in JPY.

    Priority order:
    1. settlement_jpy + settlement_usd * fx_rate (mixed settlement)
    2. settlement_jpy > 0 — use directly
    3. settlement_usd * fx_rate from trade record
    4. shares * price * trade-level fx_rate
    5. shares * price * portfolio-level fx_rates fallback

    Parameters
    ----------
    trade : dict
        Raw trade dict from history_store.
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    float
        Settlement amount in JPY (≥ 0).
    """
    sjpy = float(trade.get("settlement_jpy", 0) or 0)
    susd = float(trade.get("settlement_usd", 0) or 0)
    fx = float(trade.get("fx_rate", 0) or 0)

    if sjpy > 0 and susd > 0:
        return sjpy + susd * fx
    if sjpy > 0:
        return sjpy
    if susd > 0 and fx > 0:
        return susd * fx

    shares = float(trade.get("shares", 0) or 0)
    price = float(trade.get("price", 0) or 0)
    if fx > 0:
        return shares * price * fx

    cur = trade.get("currency", "JPY")
    rate = fx_rates.get(cur, 1.0)
    return shares * price * rate


# ---------------------------------------------------------------------------
# Internal FIFO engine
# ---------------------------------------------------------------------------


@dataclass
class _BuySummary:
    """Internal per-symbol buy data extracted during FIFO processing."""

    buy_count: int = 0
    total_buy_jpy: float = 0.0


@dataclass
class _SellEvent:
    """Internal per-sell-transaction result from FIFO lot matching."""

    symbol: str
    pnl_jpy: float
    proceeds_jpy: float
    hold_days: float | None  # avg hold days for lots consumed; None if dates missing
    sell_date: str = ""  # YYYY-MM-DD; empty string when trade date is unavailable


def _run_fifo_matching(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> tuple[dict[str, _BuySummary], list[_SellEvent]]:
    """Run FIFO lot matching across all trades.

    Processes trades in order, maintaining per-symbol lot queues.  Returns
    aggregated buy data and a list of per-sell events for downstream
    aggregation by the caller.

    Cash positions (``is_cash(symbol) == True``) are excluded.
    Stock-split transfers (type "transfer" with price ≤ 0) adjust the
    existing cost basis rather than opening a new lot.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts sorted ascending by date.
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    (buy_summaries, sell_events)
        buy_summaries : dict[str, _BuySummary]
            Per-symbol accumulated buy count and total buy amount (JPY).
        sell_events : list[_SellEvent]
            Ordered list of per-sell FIFO-matched events.
    """
    lots: dict[str, list[dict]] = defaultdict(list)
    buy_summaries: dict[str, _BuySummary] = {}
    sell_events: list[_SellEvent] = []

    for trade in trades:
        sym = trade.get("symbol", "")
        if not sym or is_cash(sym):
            continue

        tt = trade.get("trade_type", "buy")
        shares = float(trade.get("shares", 0) or 0)
        if shares <= 0:
            continue

        trade_date = _parse_trade_date(trade.get("date", ""))

        if tt in ("buy", "transfer"):
            cost_jpy = _trade_amount_jpy(trade, fx_rates)
            if sym not in buy_summaries:
                buy_summaries[sym] = _BuySummary()
            buy_summaries[sym].buy_count += 1
            buy_summaries[sym].total_buy_jpy += cost_jpy

            cost_per_share = cost_jpy / shares if shares > 0 else 0.0

            # Stock split: transfer with price=0 → redistribute existing cost basis
            price = float(trade.get("price", 0) or 0)
            if tt == "transfer" and price <= 0 and lots[sym]:
                existing_shares = sum(lot["shares"] for lot in lots[sym])
                if existing_shares > 0:
                    split_ratio = (existing_shares + shares) / existing_shares
                    for lot in lots[sym]:
                        lot["cost_jpy_per_share"] /= split_ratio
                        lot["shares"] *= split_ratio
            else:
                lots[sym].append(
                    {
                        "shares": shares,
                        "cost_jpy_per_share": cost_per_share,
                        "date": trade_date,
                    }
                )

        elif tt == "sell":
            proceeds_jpy = _trade_amount_jpy(trade, fx_rates)
            proceeds_per_share = proceeds_jpy / shares if shares > 0 else 0.0

            remaining = shares
            sell_pnl = 0.0
            sell_hold_days: list[float] = []

            while remaining > 0.5 and lots[sym]:
                lot = lots[sym][0]
                take = min(remaining, lot["shares"])
                pnl = take * (proceeds_per_share - lot["cost_jpy_per_share"])
                sell_pnl += pnl

                if lot["date"] is not None and trade_date is not None:
                    hold_d = (trade_date - lot["date"]).days
                    sell_hold_days.append(float(max(hold_d, 0)))

                lot["shares"] -= take
                remaining -= take
                if lot["shares"] < 0.5:
                    lots[sym].pop(0)

            avg_hold: float | None = None
            if sell_hold_days:
                avg_hold = sum(sell_hold_days) / len(sell_hold_days)

            sell_events.append(
                _SellEvent(
                    symbol=sym,
                    pnl_jpy=sell_pnl,
                    proceeds_jpy=proceeds_jpy,
                    hold_days=avg_hold,
                    sell_date=trade_date.isoformat() if trade_date is not None else "",
                )
            )

    return buy_summaries, sell_events


# ---------------------------------------------------------------------------
# Public API: per-symbol and portfolio trade statistics
# ---------------------------------------------------------------------------


def compute_trade_stats_by_symbol(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> dict[str, TradeStats]:
    """Compute per-symbol trade statistics from raw trade records.

    Uses FIFO matching to compute realized PnL and average holding periods.
    Cash positions (``is_cash(symbol) == True``) are excluded.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts as returned by ``_build_holdings_timeline``
        (i.e., sorted ascending by date).
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    dict[str, TradeStats]
        Mapping of symbol → TradeStats.  Empty dict if no non-cash trades exist.
    """
    buy_summaries, sell_events = _run_fifo_matching(trades, fx_rates)

    sell_counts: dict[str, int] = defaultdict(int)
    total_sell_jpy: dict[str, float] = defaultdict(float)
    win_counts: dict[str, int] = defaultdict(int)
    loss_counts: dict[str, int] = defaultdict(int)
    realized_pnl: dict[str, float] = defaultdict(float)
    hold_days_per_sell: dict[str, list[float]] = defaultdict(list)
    sell_records_per_symbol: dict[str, list[SellRecord]] = defaultdict(list)

    for e in sell_events:
        sym = e.symbol
        sell_counts[sym] += 1
        total_sell_jpy[sym] += e.proceeds_jpy
        realized_pnl[sym] += e.pnl_jpy
        if e.pnl_jpy >= 0:
            win_counts[sym] += 1
        else:
            loss_counts[sym] += 1
        if e.hold_days is not None:
            hold_days_per_sell[sym].append(e.hold_days)
        sell_records_per_symbol[sym].append(
            SellRecord(
                symbol=sym,
                sell_date=e.sell_date,
                pnl_jpy=e.pnl_jpy,
                holding_days=int(e.hold_days) if e.hold_days is not None else 0,
            )
        )

    all_syms = set(buy_summaries) | set(sell_counts)
    result: dict[str, TradeStats] = {}

    for sym in all_syms:
        sc = sell_counts[sym]
        avg_hold: float | None = None
        hold_list = hold_days_per_sell[sym]
        if hold_list:
            avg_hold = sum(hold_list) / len(hold_list)

        bs = buy_summaries.get(sym)
        buy_count = bs.buy_count if bs else 0
        total_buy = bs.total_buy_jpy if bs else 0.0

        result[sym] = TradeStats(
            symbol=sym,
            buy_count=buy_count,
            sell_count=sc,
            total_buy_jpy=round(total_buy, 0),
            total_sell_jpy=round(total_sell_jpy[sym], 0),
            realized_pnl_jpy=round(realized_pnl[sym], 0),
            win_count=win_counts[sym],
            loss_count=loss_counts[sym],
            avg_hold_days=round(avg_hold, 1) if avg_hold is not None else None,
            confidence=_classify_confidence(sc),
            sell_records=sell_records_per_symbol[sym],
        )

    return result


def compute_portfolio_trade_stats(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> PortfolioTradeStats:
    """Compute aggregated trade statistics for the full portfolio.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts as returned by ``_build_holdings_timeline``
        (date-sorted ascending).
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    PortfolioTradeStats
        Aggregated statistics with per-symbol breakdown.
        Returns ``PortfolioTradeStats.empty()`` if no trades are provided.
    """
    if not trades:
        return PortfolioTradeStats.empty()

    by_symbol = compute_trade_stats_by_symbol(trades, fx_rates)
    if not by_symbol:
        return PortfolioTradeStats.empty()

    total_buy = sum(s.buy_count for s in by_symbol.values())
    total_sell = sum(s.sell_count for s in by_symbol.values())
    total_pnl = sum(s.realized_pnl_jpy for s in by_symbol.values())
    total_wins = sum(s.win_count for s in by_symbol.values())
    total_losses = sum(s.loss_count for s in by_symbol.values())

    # Portfolio-wide avg hold days — sell-count-weighted
    hold_parts: list[tuple[float, int]] = [
        (s.avg_hold_days, s.sell_count) for s in by_symbol.values() if s.avg_hold_days is not None and s.sell_count > 0
    ]
    avg_hold: float | None = None
    if hold_parts:
        weighted_sum = sum(h * c for h, c in hold_parts)
        total_weight = sum(c for _, c in hold_parts)
        if total_weight > 0:
            avg_hold = round(weighted_sum / total_weight, 1)

    portfolio_confidence = _classify_confidence(total_sell)

    all_sell_recs: list[SellRecord] = [record for s in by_symbol.values() for record in s.sell_records]

    return PortfolioTradeStats(
        symbols_traded=sorted(by_symbol.keys()),
        total_buy_count=total_buy,
        total_sell_count=total_sell,
        total_realized_pnl_jpy=round(total_pnl, 0),
        overall_win_count=total_wins,
        overall_loss_count=total_losses,
        avg_hold_days=avg_hold,
        by_symbol=by_symbol,
        confidence=portfolio_confidence,
        all_sell_records=all_sell_recs,
    )


# ---------------------------------------------------------------------------
# Public API: holding-period distribution
# ---------------------------------------------------------------------------


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted list.

    Parameters
    ----------
    sorted_data : list[float]
        Non-empty list sorted in ascending order.
    p : float
        Percentile to compute (0 ≤ p ≤ 100).

    Returns
    -------
    float
        Interpolated p-th percentile value.
    """
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (k - lo) * (sorted_data[hi] - sorted_data[lo])


def compute_holding_period_summary(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> HoldingPeriodSummary:
    """Compute holding-period distribution and summary statistics.

    Derives per-sell holding periods from FIFO lot matching, then builds
    descriptive statistics and bucket counts (short / medium / long term).

    The ``short_term_ratio`` field can directly feed trading-style profiling:
    a value near 1.0 indicates a predominantly short-term trader.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts (date-sorted ascending).
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    HoldingPeriodSummary
        Distribution statistics for holding periods.
        Returns ``HoldingPeriodSummary.empty()`` when no trades are provided.
    """
    _, sell_events = _run_fifo_matching(trades, fx_rates)
    total_closed = len(sell_events)

    if total_closed == 0:
        return HoldingPeriodSummary.empty()

    hold_days = [e.hold_days for e in sell_events if e.hold_days is not None]
    total_with_data = len(hold_days)

    if total_with_data == 0:
        return HoldingPeriodSummary(
            total_closed=total_closed,
            total_with_hold_data=0,
            confidence=_classify_confidence(total_closed),
        )

    sorted_days = sorted(hold_days)
    short_term_count = sum(1 for d in hold_days if d < 30)
    medium_term_count = sum(1 for d in hold_days if 30 <= d < 180)
    long_term_count = sum(1 for d in hold_days if d >= 180)

    return HoldingPeriodSummary(
        total_closed=total_closed,
        total_with_hold_data=total_with_data,
        min_days=round(sorted_days[0], 1),
        max_days=round(sorted_days[-1], 1),
        median_days=round(_percentile(sorted_days, 50), 1),
        p25_days=round(_percentile(sorted_days, 25), 1),
        p75_days=round(_percentile(sorted_days, 75), 1),
        short_term_count=short_term_count,
        medium_term_count=medium_term_count,
        long_term_count=long_term_count,
        short_term_ratio=round(short_term_count / total_with_data, 4),
        confidence=_classify_confidence(total_closed),
    )


# ---------------------------------------------------------------------------
# Public API: win/loss summary
# ---------------------------------------------------------------------------


def compute_win_loss_summary(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> WinLossSummary:
    """Compute realized win/loss statistics across all closed trades.

    Uses the same FIFO lot matching as ``compute_trade_stats_by_symbol`` but
    returns portfolio-wide win/loss metrics including profit factor and
    per-side averages.

    Parameters
    ----------
    trades : list[dict]
        Raw trade dicts (date-sorted ascending).
    fx_rates : dict[str, float]
        Currency → JPY rate mapping for fallback conversion.

    Returns
    -------
    WinLossSummary
        Portfolio-wide realized win/loss statistics.
        Returns ``WinLossSummary.empty()`` when no sell data is available.

    Notes
    -----
    ``profit_factor`` is ``None`` when either gross_profit or |gross_loss|
    is zero (no wins, or no losses), since dividing by zero would yield an
    infinite or undefined value.
    """
    _, sell_events = _run_fifo_matching(trades, fx_rates)

    if not sell_events:
        return WinLossSummary.empty()

    win_pnls = [e.pnl_jpy for e in sell_events if e.pnl_jpy >= 0]
    loss_pnls = [e.pnl_jpy for e in sell_events if e.pnl_jpy < 0]

    win_count = len(win_pnls)
    loss_count = len(loss_pnls)
    total_closed = win_count + loss_count

    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss = sum(loss_pnls) if loss_pnls else 0.0

    avg_win: float | None = gross_profit / win_count if win_count > 0 else None
    avg_loss: float | None = gross_loss / loss_count if loss_count > 0 else None

    # profit_factor = gross_profit / |gross_loss|; undefined when either side is zero
    profit_factor: float | None = None
    if gross_loss < 0 and gross_profit > 0:
        profit_factor = round(gross_profit / abs(gross_loss), 3)

    win_rate: float | None = round(win_count / total_closed, 4) if total_closed > 0 else None

    return WinLossSummary(
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        avg_win_jpy=round(avg_win, 0) if avg_win is not None else None,
        avg_loss_jpy=round(avg_loss, 0) if avg_loss is not None else None,
        gross_profit_jpy=round(gross_profit, 0),
        gross_loss_jpy=round(gross_loss, 0),
        profit_factor=profit_factor,
        confidence=_classify_confidence(total_closed),
    )


# ---------------------------------------------------------------------------
# Public API: style metrics
# ---------------------------------------------------------------------------


def compute_style_metrics(
    portfolio_stats: PortfolioTradeStats,
    trades: list[dict],
    fx_rates: dict[str, float],
) -> StyleMetrics:
    """Derive trading-style characterization from portfolio trade statistics.

    Parameters
    ----------
    portfolio_stats : PortfolioTradeStats
        Pre-computed portfolio trade statistics.
    trades : list[dict]
        Raw trade dicts for date-range and position-size calculations.
    fx_rates : dict[str, float]
        Currency → JPY rate mapping.

    Returns
    -------
    StyleMetrics
        Trading-style characterization with confidence and human-readable notes.
    """
    notes: list[str] = []
    non_cash_trades = [t for t in trades if not is_cash(t.get("symbol", ""))]
    total_trades = len(non_cash_trades)

    if total_trades == 0:
        return StyleMetrics(
            notes=["No trade data available for style analysis."],
        )

    # --- Trade frequency (trades per calendar month) ---
    trade_dates = [_parse_trade_date(t.get("date", "")) for t in non_cash_trades]
    valid_dates = [d for d in trade_dates if d is not None]

    trade_frequency = "unknown"
    if len(valid_dates) >= 2:
        first = min(valid_dates)
        last = max(valid_dates)
        span_months = max((last - first).days / 30.44, 1.0)
        tpm = total_trades / span_months
        if tpm >= 2.0:
            trade_frequency = "active"
        elif tpm >= 0.5:
            trade_frequency = "moderate"
        else:
            trade_frequency = "passive"
    elif total_trades >= 1:
        trade_frequency = "passive"
        notes.append("Date range too short to determine trade frequency precisely.")

    # --- Average buy position size ---
    buy_trades = [t for t in non_cash_trades if t.get("trade_type") in ("buy", "transfer")]
    avg_position_size_jpy = 0.0
    if buy_trades:
        total_buy_jpy = sum(_trade_amount_jpy(t, fx_rates) for t in buy_trades)
        avg_position_size_jpy = round(total_buy_jpy / len(buy_trades), 0)

    # --- Concentration score: HHI of buy volume per symbol ---
    buy_by_sym: dict[str, float] = defaultdict(float)
    for t in buy_trades:
        sym = t.get("symbol", "")
        if sym:
            buy_by_sym[sym] += _trade_amount_jpy(t, fx_rates)

    concentration_score = 0.0
    total_buy_vol = sum(buy_by_sym.values())
    if total_buy_vol > 0 and len(buy_by_sym) > 1:
        weights = [v / total_buy_vol for v in buy_by_sym.values()]
        concentration_score = round(sum(w * w for w in weights), 4)
    elif len(buy_by_sym) == 1:
        concentration_score = 1.0

    # --- Holding style ---
    holding_style = "unknown"
    if portfolio_stats.avg_hold_days is not None:
        days = portfolio_stats.avg_hold_days
        if days < 30:
            holding_style = "short_term"
        elif days < 180:
            holding_style = "medium_term"
        else:
            holding_style = "long_term"
    else:
        notes.append("Holding style unavailable: no completed sell transactions to measure.")

    # --- Confidence for style metrics ---
    confidence = portfolio_stats.confidence
    if total_trades < _MIN_SELLS_MEDIUM:
        confidence = min_confidence(confidence, ConfidenceLevel.LOW)
        notes.append(f"Style metrics based on {total_trades} trade(s) — treat as indicative only.")

    return StyleMetrics(
        trade_frequency=trade_frequency,
        avg_position_size_jpy=avg_position_size_jpy,
        concentration_score=concentration_score,
        holding_style=holding_style,
        confidence=confidence,
        notes=notes,
    )
