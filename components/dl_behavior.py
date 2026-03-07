"""Behavior insight integration layer — component bridge.

Thin adapter between the pure ``src.core.behavior`` domain package and the
rest of the component infrastructure (trade history loader, FX rates).

This module is **Streamlit-free**.  It loads trade data through the existing
``components.dl_holdings._build_holdings_timeline`` helper and delegates all
computation to the domain layer.

Public API
----------
``load_behavior_insight``  — primary entry point; returns a ``BehaviorInsight``.
``load_timing_insight``    — loads timing quality insight; returns a
                             ``PortfolioTimingInsight``.

Re-exports for convenience (so callers can do
``from components.dl_behavior import BehaviorInsight``):
    ``BehaviorInsight``, ``ConfidenceLevel``, ``PortfolioTimingInsight``,
    ``PortfolioTradeStats``, ``SellRecord``, ``StyleMetrics``, ``TradeStats``.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")

_MEMO_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "押し目買い": ("押し目", "dip", "pullback", "調整", "下落"),
    "利益確定": ("利確", "利益確定", "take profit", "profit"),
    "損切り": ("損切", "cut loss", "stop loss", "ロスカット", "撤退"),
    "リバランス": ("リバランス", "rebalance", "配分調整", "比率調整"),
    "長期保有": ("長期", "積立", "core", "long term", "長く持つ"),
    "決算/イベント": ("決算", "earnings", "fomc", "event", "材料", "ニュース"),
}

from src.core.behavior import (
    BehaviorInsight,
    BiasSignal,
    ConfidenceLevel,
    PortfolioTimingInsight,
    PortfolioTradeStats,
    SellRecord,
    StyleMetrics,
    StyleProfile,
    TradeStats,
    compute_holding_period_summary,
    compute_portfolio_timing_insight,
    compute_portfolio_trade_stats,
    compute_style_metrics,
    compute_style_profile,
    compute_win_loss_summary,
    detect_biases,
    min_confidence,
)
from src.core.paths import PRICE_CACHE_DIR as _PRICE_CACHE_DIR
from src.core.portfolio.portfolio_manager import DEFAULT_CSV_PATH, get_fx_rates
from src.data import yahoo_client


def load_behavior_insight(
    base_dir: str = _DEFAULT_HISTORY_DIR,
    csv_path: str = DEFAULT_CSV_PATH,
) -> BehaviorInsight:
    """Load accumulated-trade behavior insight for the portfolio.

    This is the primary entry point for behavior analysis.  It:

    1. Loads the full trade history via ``_build_holdings_timeline``.
    2. Fetches current FX rates for JPY conversion (falls back to
       approximate rates on network failure).
    3. Delegates all computation to the pure domain layer
       (``src.core.behavior``).

    Parameters
    ----------
    base_dir : str
        Path to the history directory containing ``trade/`` JSON files.
        Defaults to ``data/history``.
    csv_path : str
        Path to ``portfolio.csv`` — used only for FX rate fallback logic
        inside ``get_fx_rates``.

    Returns
    -------
    BehaviorInsight
        Aggregated behavior insight with confidence level and notes.
        Returns ``BehaviorInsight.empty()`` when no trade data is found.
    """
    # Deferred import to avoid circular dependency at module load time.
    from components.dl_holdings import _build_holdings_timeline

    trades = _build_holdings_timeline(base_dir)
    if not trades:
        logger.info("load_behavior_insight: no trades found in %s", base_dir)
        return BehaviorInsight.empty()

    try:
        fx_rates = get_fx_rates(yahoo_client)
    except Exception as exc:
        logger.warning(
            "load_behavior_insight: FX rates unavailable (%s); using approximate fallback",
            exc,
        )
        fx_rates = {"USD": 150.0, "JPY": 1.0, "EUR": 165.0}

    portfolio_stats = compute_portfolio_trade_stats(trades, fx_rates)
    style = compute_style_metrics(portfolio_stats, trades, fx_rates)
    holding_period = compute_holding_period_summary(trades, fx_rates)
    win_loss = compute_win_loss_summary(trades, fx_rates)

    # Overall insight confidence = lower of all deterministic behavior sub-results.
    overall_confidence = min_confidence(
        min_confidence(portfolio_stats.confidence, style.confidence),
        min_confidence(holding_period.confidence, win_loss.confidence),
    )

    notes: list[str] = list(style.notes)
    if portfolio_stats.confidence == ConfidenceLevel.INSUFFICIENT:
        notes.insert(
            0,
            "Insufficient trade history for reliable behavior analysis.",
        )
    elif portfolio_stats.confidence == ConfidenceLevel.LOW:
        notes.insert(
            0,
            "Limited trade history — insights may not be fully representative.",
        )

    return BehaviorInsight(
        trade_stats=portfolio_stats,
        style_metrics=style,
        holding_period=holding_period,
        win_loss=win_loss,
        confidence=overall_confidence,
        notes=notes,
    )


def load_timing_insight(
    base_dir: str = _DEFAULT_HISTORY_DIR,
) -> PortfolioTimingInsight:
    """Load trade-timing quality insight for the portfolio.

    Loads trades from the history directory, resolves price history for each
    symbol from the **local disk price cache** (no new network calls unless
    data is already cached), and delegates computation to
    ``compute_portfolio_timing_insight``.

    Degrades gracefully when price history is sparse or missing: symbols with
    no cached price data are evaluated with empty history, receiving neutral
    (INSUFFICIENT confidence) timing scores.

    Parameters
    ----------
    base_dir : str
        Path to the history directory containing ``trade/`` JSON files.
        Defaults to ``data/history``.

    Returns
    -------
    PortfolioTimingInsight
        Aggregated timing insight.  Returns
        ``PortfolioTimingInsight.empty()`` when no trade data is found.
    """
    import pandas as pd

    # Deferred import to avoid circular dependency at module load time.
    from components.dl_holdings import _build_holdings_timeline
    from src.core.common import is_cash

    trades = _build_holdings_timeline(base_dir)
    if not trades:
        logger.info("load_timing_insight: no trades found in %s", base_dir)
        return PortfolioTimingInsight.empty()

    # Filter cash pseudo-symbols — timing analysis is only meaningful for real
    # securities; cash entries have no price history and no timing signal.
    non_cash_trades = [t for t in trades if not is_cash(str(t.get("symbol", "")))]
    if not non_cash_trades:
        logger.info("load_timing_insight: no non-cash trades in %s", base_dir)
        return PortfolioTimingInsight.empty()

    # Unique non-cash symbols that appear in the trade list.
    symbols = list({t["symbol"] for t in non_cash_trades})

    # ------------------------------------------------------------------ #
    # Build per-symbol price bar history from the disk price cache.
    # We scan candidate period files (longest look-back first) and stop
    # once every symbol is satisfied.  No new network calls are made.
    # ------------------------------------------------------------------ #
    history_by_symbol: dict[str, list[dict]] = {}

    # Prefer longer periods so timing indicators have more history depth.
    _CANDIDATE_PERIODS = ("max", "5y", "3y", "2y", "1y", "6mo")

    for period in _CANDIDATE_PERIODS:
        if len(history_by_symbol) == len(symbols):
            break  # All symbols satisfied
        cache_path = _PRICE_CACHE_DIR / f"close_{period}.csv"
        if not cache_path.exists():
            continue
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            for sym in symbols:
                if sym in history_by_symbol:
                    continue  # Already loaded from a longer-period cache
                if sym not in df.columns:
                    continue
                series = df[sym].dropna()
                if series.empty:
                    continue
                bars: list[dict] = []
                for idx, val in series.items():
                    try:
                        bars.append({"date": str(idx.date()), "close": float(val)})
                    except Exception:
                        continue
                if bars:
                    history_by_symbol[sym] = bars
        except Exception as exc:
            logger.warning(
                "load_timing_insight: failed to read price cache %s: %s",
                cache_path,
                exc,
            )

    logger.info(
        "load_timing_insight: %d trades (%d non-cash), %d/%d symbols with price history",
        len(trades),
        len(non_cash_trades),
        len(history_by_symbol),
        len(symbols),
    )

    return compute_portfolio_timing_insight(non_cash_trades, history_by_symbol)


def load_style_profile_insight(
    positions: list[dict],
    behavior_insight: BehaviorInsight,
    history_df: pd.DataFrame | None = None,
    benchmark_symbol: str | None = None,
    period: str = "1y",
) -> tuple[StyleProfile, list[BiasSignal]]:
    """Compute an ADI-style profile and detect portfolio biases.

    This is the primary entry point for style-profile analysis.  It:

    1. Computes the ``StyleProfile`` (ADI score, label, component metrics)
       from positions, ``BehaviorInsight``, and optionally portfolio history.
    2. Optionally estimates portfolio beta when a benchmark symbol is given,
       using the existing price-cache infrastructure (no new network calls
       unless the benchmark is already cached).
    3. Runs all heuristic bias detectors and returns detected
       ``BiasSignal`` instances sorted by severity.

    Degrades gracefully at every step — missing ``history_df``,
    missing benchmark, or an empty portfolio all produce valid (low-
    confidence) results rather than raising exceptions.

    Parameters
    ----------
    positions : list[dict]
        Current portfolio positions from ``get_current_snapshot()``.
        Each entry should have ``evaluation_jpy`` and ``sector``.
    behavior_insight : BehaviorInsight
        Pre-computed behavior insight from ``load_behavior_insight``.
    history_df : pd.DataFrame | None
        Portfolio history DataFrame (``build_portfolio_history`` output)
        used to derive volatility and beta.  ``None`` to skip.
    benchmark_symbol : str | None
        Yahoo Finance ticker for the benchmark (e.g. ``"^GSPC"``).
        ``None`` to skip beta estimation.
    period : str
        Price period for benchmark data (e.g. ``"1y"``).  Used only when
        ``benchmark_symbol`` is set.

    Returns
    -------
    tuple[StyleProfile, list[BiasSignal]]
        ``(style_profile, bias_signals)`` — both are always populated
        (may be empty / low-confidence when data is insufficient).
    """

    # --- Resolve benchmark series (from existing cache; no new network call
    #     if the data hasn't been downloaded this session) ---
    benchmark_series: pd.Series | None = None
    if benchmark_symbol and history_df is not None and not history_df.empty:
        try:
            from components.dl_analytics import get_benchmark_series

            benchmark_series = get_benchmark_series(benchmark_symbol, history_df, period)
        except Exception as exc:
            logger.debug("load_style_profile_insight: benchmark fetch failed (%s)", exc)

    # --- Compute style profile ---
    try:
        style_profile = compute_style_profile(
            positions=positions,
            style_metrics=behavior_insight.style_metrics,
            holding_period=behavior_insight.holding_period,
            history_df=history_df,
            benchmark_series=benchmark_series,
        )
    except Exception as exc:
        logger.warning(
            "load_style_profile_insight: compute_style_profile failed (%s); returning empty profile",
            exc,
        )
        style_profile = StyleProfile.empty()

    # --- Detect biases ---
    try:
        bias_signals = detect_biases(
            positions=positions,
            style_metrics=behavior_insight.style_metrics,
            holding_period=behavior_insight.holding_period,
            style_profile=style_profile,
        )
    except Exception as exc:
        logger.warning(
            "load_style_profile_insight: detect_biases failed (%s); returning empty bias list",
            exc,
        )
        bias_signals = []

    return style_profile, bias_signals


def load_trade_memo_context(
    base_dir: str = _DEFAULT_HISTORY_DIR,
    limit: int = 40,
) -> dict[str, object]:
    """Summarize recent trade memos into privacy-safe aggregate themes.

    Why: The optional AI retrospective should benefit from user trade notes
         without sending raw memo text or ticker symbols to Copilot.
    How: Load the newest trade history entries, count memo coverage, and map
         memo keywords to a small fixed theme set. Only aggregate counts are
         returned, so the caller can build anonymized prompts.
    """
    from src.data.history_store import load_history

    trades = load_history("trade", base_dir=base_dir)
    if limit > 0:
        trades = trades[:limit]

    memo_trades = [trade for trade in trades if isinstance(trade.get("memo"), str) and trade.get("memo", "").strip()]
    theme_counts: Counter[str] = Counter()

    for trade in memo_trades:
        memo_text = trade.get("memo", "").strip().lower()
        if not memo_text:
            continue

        matched_themes: set[str] = set()
        for theme, keywords in _MEMO_THEME_KEYWORDS.items():
            if any(keyword.lower() in memo_text for keyword in keywords):
                matched_themes.add(theme)

        for theme in matched_themes:
            theme_counts[theme] += 1

    reviewed_trade_count = len(trades)
    memo_trade_count = len(memo_trades)
    memo_coverage_pct = round(memo_trade_count / reviewed_trade_count * 100, 1) if reviewed_trade_count > 0 else 0.0

    top_themes = [{"theme": theme, "count": count} for theme, count in theme_counts.most_common(3)]

    return {
        "reviewed_trade_count": reviewed_trade_count,
        "memo_trade_count": memo_trade_count,
        "memo_coverage_pct": memo_coverage_pct,
        "top_themes": top_themes,
    }


__all__ = [
    "BehaviorInsight",
    "BiasSignal",
    "ConfidenceLevel",
    "PortfolioTimingInsight",
    "PortfolioTradeStats",
    "SellRecord",
    "StyleMetrics",
    "StyleProfile",
    "TradeStats",
    "load_behavior_insight",
    "load_style_profile_insight",
    "load_timing_insight",
    "load_trade_memo_context",
]
