"""Pure-domain trade-timing analysis.

Evaluates buy/sell execution timing against locally available price-history
context.  No Streamlit, no network calls — all inputs are passed explicitly.

Typical usage::

    from src.core.behavior.timing_analysis import (
        compute_trade_timing,
        compute_portfolio_timing_insight,
    )

    result = compute_trade_timing(
        trade={"symbol": "AAPL", "date": "2024-03-15",
               "trade_type": "buy", "price": 172.5},
        history=[{"date": "2024-03-01", "close": 180.0}, ...],
    )
    print(result.timing_score, result.label)
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from src.core.behavior.models import (
    ConfidenceLevel,
    PortfolioTimingInsight,
    PriceContext,
    TradeTimingResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_RSI_PERIOD = 14
_SMA_SHORT = 20
_SMA_LONG = 50

# Default look-back window for price-percentile calculation (~1 trading year).
_PERCENTILE_WINDOW_DEFAULT = 252

# Score thresholds → human-readable label (descending threshold order).
_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (80.0, "excellent"),
    (60.0, "good"),
    (40.0, "neutral"),
    (20.0, "poor"),
]

# Confidence level ordering (mirrors trade_stats, but imported lazily to
# avoid a circular import at module load time).
_CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    ConfidenceLevel.INSUFFICIENT: 0,
    ConfidenceLevel.LOW: 1,
    ConfidenceLevel.MEDIUM: 2,
    ConfidenceLevel.HIGH: 3,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> date | None:
    """Parse a YYYY-MM-DD or YYYY/MM/DD string; return ``None`` on failure."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def _sort_history(history: list[dict]) -> list[tuple[date, float]]:
    """Parse and deduplicate price history, sorted ascending by date.

    Bars with unparseable dates, non-positive closes, or missing ``close``
    keys are silently dropped so that sparse / holiday-affected data is
    handled gracefully.

    Parameters
    ----------
    history : list[dict]
        Raw price bars, each with ``"date"`` (str) and ``"close"`` (numeric).

    Returns
    -------
    list[tuple[date, float]]
        ``(date, close_price)`` pairs sorted ascending.  May be empty.
    """
    seen: dict[date, float] = {}
    for bar in history:
        d = _parse_date(str(bar.get("date", "")))
        raw_close = bar.get("close")
        try:
            close = float(raw_close)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if d is not None and close > 0:
            seen[d] = close  # last-write wins on duplicate dates

    return sorted(seen.items())  # sorted by date (key)


def _find_idx_on_or_before(
    sorted_hist: list[tuple[date, float]],
    target: date,
) -> int | None:
    """Return the index of the latest bar on or before *target*, or ``None``."""
    best: int | None = None
    for i, (d, _) in enumerate(sorted_hist):
        if d <= target:
            best = i
        else:
            break
    return best


def _compute_sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of the last *period* values.

    Returns ``None`` when fewer than *period* values are available.
    """
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _compute_rsi(closes: list[float], period: int = _RSI_PERIOD) -> float | None:
    """RSI using simple average of gains/losses over the last *period* bars.

    Requires at least ``period + 1`` close prices (to form *period* deltas).
    Returns ``None`` when history is insufficient.

    Edge cases
    ----------
    * All moves are gains → RSI 100.
    * All moves are losses → RSI 0.
    * Zero net movement → RSI 50.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    window = deltas[-period:]

    avg_gain = sum(max(d, 0.0) for d in window) / period
    avg_loss = sum(max(-d, 0.0) for d in window) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_price_percentile(price: float, closes: list[float]) -> float:
    """Where *price* falls in *closes* range, mapped to [0, 1].

    Returns 0.5 when the range is degenerate (all values equal, or fewer
    than two bars).
    """
    if len(closes) < 2:
        return 0.5
    lo = min(closes)
    hi = max(closes)
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (price - lo) / (hi - lo)))


def _score_label(score: float) -> str:
    """Map a 0–100 timing score to a human-readable label."""
    for threshold, label in _LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "very_poor"


def _compute_timing_score(
    trade_type: str,
    price_percentile: float | None,
    rsi: float | None,
    sma_20: float | None,
    trade_price: float,
) -> tuple[float, list[str]]:
    """Compute composite timing score (0–100) for a single trade.

    Three weighted signal components are combined when available:

    * **Price percentile** (weight 0.50) — for buys, lower is better; for
      sells, higher is better.
    * **RSI(14)** (weight 0.30) — for buys, oversold (low RSI) is better;
      for sells, overbought (high RSI) is better.
    * **SMA-20 deviation** (weight 0.20) — for buys, trading below the SMA
      is better; for sells, above is better.

    When a component is unavailable the remaining components are averaged
    with equal weight, preserving the 0–100 output range.

    Parameters
    ----------
    trade_type : str
        ``"buy"`` or ``"sell"``.
    price_percentile : float | None
        Price percentile [0, 1].
    rsi : float | None
        RSI value [0, 100].
    sma_20 : float | None
        20-period SMA.
    trade_price : float
        Execution price.

    Returns
    -------
    tuple[float, list[str]]
        ``(score, notes)`` where score is 0–100 and notes contain
        human-readable context annotations.
    """
    is_buy = trade_type == "buy"
    components: list[float] = []
    notes: list[str] = []

    # --- Component 1: Price percentile (weight 0.50) ---
    if price_percentile is not None:
        pct_score = 100.0 * (1.0 - price_percentile) if is_buy else 100.0 * price_percentile
        components.append(pct_score)

        if is_buy:
            if price_percentile <= 0.20:
                notes.append("Price near period low — favorable buy entry.")
            elif price_percentile >= 0.80:
                notes.append("Price near period high — potentially late buy entry.")
        else:
            if price_percentile >= 0.80:
                notes.append("Price near period high — favorable sell exit.")
            elif price_percentile <= 0.20:
                notes.append("Price near period low — potentially early sell exit.")

    # --- Component 2: RSI (weight 0.30) ---
    if rsi is not None:
        rsi_score = 100.0 * (1.0 - rsi / 100.0) if is_buy else 100.0 * (rsi / 100.0)
        components.append(rsi_score)

        if is_buy:
            if rsi < 30:
                notes.append(f"RSI {rsi:.1f} — oversold territory, historically favorable for buys.")
            elif rsi > 70:
                notes.append(f"RSI {rsi:.1f} — overbought territory, elevated entry risk.")
        else:
            if rsi > 70:
                notes.append(f"RSI {rsi:.1f} — overbought territory, historically favorable for sells.")
            elif rsi < 30:
                notes.append(f"RSI {rsi:.1f} — oversold territory, may indicate premature exit.")

    # --- Component 3: SMA-20 deviation (weight 0.20) ---
    if sma_20 is not None and sma_20 > 0 and trade_price > 0:
        # deviation ∈ [−1, 1] mapped from price distance within ±20% of SMA
        raw_dev = (trade_price - sma_20) / sma_20
        norm_dev = max(-1.0, min(1.0, raw_dev / 0.20))

        sma_score = 100.0 * (0.5 - norm_dev / 2.0) if is_buy else 100.0 * (0.5 + norm_dev / 2.0)
        sma_score = max(0.0, min(100.0, sma_score))
        components.append(sma_score)

    if not components:
        return 50.0, ["Insufficient price history — neutral score assigned."]

    score = round(sum(components) / len(components), 1)
    return score, notes


def _classify_timing_confidence(history_bars_before_trade: int) -> ConfidenceLevel:
    """Map available history depth to a ConfidenceLevel.

    Thresholds
    ----------
    * ≥ 50 bars → MEDIUM (enough for SMA-20, RSI-14, and meaningful percentile)
    * ≥ 15 bars → LOW   (enough for RSI-14 only)
    * < 15 bars → INSUFFICIENT
    """
    if history_bars_before_trade >= 50:
        return ConfidenceLevel.MEDIUM
    if history_bars_before_trade >= _RSI_PERIOD + 1:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.INSUFFICIENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_trade_timing(
    trade: dict,
    history: list[dict],
    percentile_window: int = _PERCENTILE_WINDOW_DEFAULT,
) -> TradeTimingResult:
    """Evaluate the timing quality of a single trade against price history.

    Parameters
    ----------
    trade : dict
        Trade record.  Expected keys:

        * ``symbol`` (str) — ticker.
        * ``date`` (str) — execution date in ``YYYY-MM-DD`` or
          ``YYYY/MM/DD`` format.
        * ``trade_type`` (str) — ``"buy"`` or ``"sell"``.
        * ``price`` (float) — execution price per share.  When 0 or absent
          the nearest available close is used as a proxy.

    history : list[dict]
        Price bars for the same symbol.  Each bar needs:

        * ``date`` (str) — ``YYYY-MM-DD``.
        * ``close`` (float) — closing price (positive).

        Non-trading-day gaps and sparse data are handled gracefully; the
        nearest available bar on or before the trade date is used for
        indicator computation.

    percentile_window : int
        Maximum number of trading-day bars to include in the price-range
        (percentile) calculation.  Default is 252 (~1 year).

    Returns
    -------
    TradeTimingResult
        Timing quality result with score, price context, label, notes, and
        a reliability confidence level.

    Notes
    -----
    * If the trade date cannot be parsed, a neutral (score=50) result with
      ``INSUFFICIENT`` confidence is returned.
    * If no history bars exist on or before the trade date, the same
      neutral fallback is returned.
    """
    symbol = trade.get("symbol", "")
    trade_type = str(trade.get("trade_type", "buy"))
    date_str = str(trade.get("date", ""))
    raw_price = trade.get("price")
    try:
        trade_price = float(raw_price)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        trade_price = 0.0

    # -- Parse trade date --
    trade_d = _parse_date(date_str)
    if trade_d is None:
        return TradeTimingResult(
            symbol=symbol,
            trade_date=date_str,
            trade_type=trade_type,
            trade_price=trade_price,
            timing_score=50.0,
            price_context=PriceContext(),
            label="neutral",
            notes=["Could not parse trade date — timing score unavailable."],
            confidence=ConfidenceLevel.INSUFFICIENT,
        )

    # -- Sort and validate history --
    sorted_hist = _sort_history(history)
    total_bars = len(sorted_hist)

    # -- Find nearest available bar on/before trade date --
    trade_idx = _find_idx_on_or_before(sorted_hist, trade_d)
    if trade_idx is None:
        return TradeTimingResult(
            symbol=symbol,
            trade_date=date_str,
            trade_type=trade_type,
            trade_price=trade_price,
            timing_score=50.0,
            price_context=PriceContext(days_of_history=total_bars),
            label="neutral",
            notes=["No price history available on or before trade date."],
            confidence=ConfidenceLevel.INSUFFICIENT,
        )

    # -- Use trade price if valid, otherwise fall back to nearest close --
    ref_close = sorted_hist[trade_idx][1]
    effective_price = trade_price if trade_price > 0 else ref_close

    # Closes available up to (and including) the bar matched to trade date
    closes_to_trade = [c for _, c in sorted_hist[: trade_idx + 1]]
    bars_available = len(closes_to_trade)

    # -- Compute indicators --
    sma_20 = _compute_sma(closes_to_trade, _SMA_SHORT)
    sma_50 = _compute_sma(closes_to_trade, _SMA_LONG)
    rsi_14 = _compute_rsi(closes_to_trade, _RSI_PERIOD)

    pct_window_closes = closes_to_trade[-percentile_window:]
    price_percentile = _compute_price_percentile(effective_price, pct_window_closes)

    # -- Build PriceContext --
    context = PriceContext(
        sma_20=round(sma_20, 4) if sma_20 is not None else None,
        sma_50=round(sma_50, 4) if sma_50 is not None else None,
        rsi_14=round(rsi_14, 2) if rsi_14 is not None else None,
        price_percentile=round(price_percentile, 4),
        percentile_window_days=len(pct_window_closes),
        days_of_history=total_bars,
    )

    # -- Score --
    score, notes = _compute_timing_score(
        trade_type=trade_type,
        price_percentile=price_percentile,
        rsi=rsi_14,
        sma_20=sma_20,
        trade_price=effective_price,
    )

    confidence = _classify_timing_confidence(bars_available)
    label = _score_label(score)

    return TradeTimingResult(
        symbol=symbol,
        trade_date=date_str,
        trade_type=trade_type,
        trade_price=effective_price,
        timing_score=score,
        price_context=context,
        label=label,
        notes=notes,
        confidence=confidence,
    )


def compute_portfolio_timing_insight(
    trades: list[dict],
    history_by_symbol: dict[str, list[dict]],
    percentile_window: int = _PERCENTILE_WINDOW_DEFAULT,
) -> PortfolioTimingInsight:
    """Compute timing-quality insight across a portfolio of trades.

    Parameters
    ----------
    trades : list[dict]
        Trade records.  Each must contain at least ``symbol``, ``date``,
        ``trade_type`` (``"buy"`` or ``"sell"``), and ``price``.
        Trades with ``trade_type`` not in ``{"buy", "sell"}`` (e.g.
        transfers, dividends) are skipped.
    history_by_symbol : dict[str, list[dict]]
        Price-bar history keyed by symbol.  Missing symbols are evaluated
        with empty history (resulting in neutral, INSUFFICIENT scores).
    percentile_window : int
        Look-back window passed through to :func:`compute_trade_timing`.

    Returns
    -------
    PortfolioTimingInsight
        Aggregated result with per-trade details and average scores.
        Returns :meth:`PortfolioTimingInsight.empty` when no eligible
        trades are found.
    """
    eligible = [t for t in trades if t.get("trade_type") in ("buy", "sell")]
    if not eligible:
        return PortfolioTimingInsight.empty()

    results: list[TradeTimingResult] = []
    for trade in eligible:
        sym = trade.get("symbol", "")
        hist = history_by_symbol.get(sym, [])
        results.append(compute_trade_timing(trade, hist, percentile_window))

    buy_scores = [r.timing_score for r in results if r.trade_type == "buy"]
    sell_scores = [r.timing_score for r in results if r.trade_type == "sell"]

    avg_buy = round(sum(buy_scores) / len(buy_scores), 1) if buy_scores else None
    avg_sell = round(sum(sell_scores) / len(sell_scores), 1) if sell_scores else None

    # Portfolio confidence = median of individual confidences
    conf_values = sorted(_CONFIDENCE_ORDER[r.confidence] for r in results)
    median_val = conf_values[len(conf_values) // 2]
    portfolio_confidence = next(k for k, v in _CONFIDENCE_ORDER.items() if v == median_val)

    summary_notes: list[str] = []
    if avg_buy is not None:
        summary_notes.append(f"Average buy timing score: {avg_buy:.0f}/100.")
    if avg_sell is not None:
        summary_notes.append(f"Average sell timing score: {avg_sell:.0f}/100.")

    return PortfolioTimingInsight(
        avg_buy_timing_score=avg_buy,
        avg_sell_timing_score=avg_sell,
        trade_results=results,
        confidence=portfolio_confidence,
        notes=summary_notes,
    )
