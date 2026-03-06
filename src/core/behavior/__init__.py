"""Behavior domain package -- accumulated-data trade insights.

This package provides **Streamlit-free** domain logic for analyzing trader
behavior from historical trade records.  The entry points exposed here are:

    ``BehaviorInsight``                  -- top-level result dataclass
    ``BiasSignal``                       -- single detected portfolio bias
    ``ConfidenceLevel``                  -- enum for data reliability
    ``HoldingPeriodSummary``             -- holding-period distribution and stats
    ``PortfolioTimingInsight``           -- aggregate timing-analysis summary
    ``PortfolioTradeStats``              -- aggregated portfolio trade stats
    ``PriceContext``                     -- timing-analysis indicator snapshot
    ``StyleMetrics``                     -- derived style characterization
    ``StyleProfile``                     -- ADI-style aggression/defense profile
    ``TradeStats``                       -- per-symbol trade statistics
    ``TradeTimingResult``                -- timing quality for a single trade
    ``WinLossSummary``                   -- realized win/loss statistics

    ``compute_holding_period_summary``   -- holding-period distribution
    ``compute_portfolio_timing_insight`` -- aggregate timing-analysis summary
    ``compute_portfolio_trade_stats``    -- aggregated portfolio stats
    ``compute_style_metrics``            -- trading style characterization
    ``compute_style_profile``            -- ADI-style aggression/defense profile
    ``compute_trade_timing``             -- single-trade timing evaluation
    ``compute_trade_stats_by_symbol``    -- per-symbol stats computation
    ``compute_win_loss_summary``         -- realized win/loss summary
    ``detect_biases``                    -- heuristic portfolio bias detection
    ``min_confidence``                   -- helper for confidence comparisons

The UI bridge lives in ``components.dl_behavior``; this package contains
only pure domain logic with no Streamlit or network dependencies.
"""

from src.core.behavior.bias_detector import detect_biases
from src.core.behavior.models import (
    BehaviorInsight,
    BiasSignal,
    ConfidenceLevel,
    HoldingPeriodSummary,
    PortfolioTimingInsight,
    PortfolioTradeStats,
    PriceContext,
    StyleMetrics,
    StyleProfile,
    TradeStats,
    TradeTimingResult,
    WinLossSummary,
)
from src.core.behavior.style_profile import compute_style_profile
from src.core.behavior.timing_analysis import (
    compute_portfolio_timing_insight,
    compute_trade_timing,
)
from src.core.behavior.trade_stats import (
    compute_holding_period_summary,
    compute_portfolio_trade_stats,
    compute_style_metrics,
    compute_trade_stats_by_symbol,
    compute_win_loss_summary,
    min_confidence,
)

__all__ = [
    "BehaviorInsight",
    "BiasSignal",
    "ConfidenceLevel",
    "HoldingPeriodSummary",
    "PortfolioTimingInsight",
    "PortfolioTradeStats",
    "PriceContext",
    "StyleMetrics",
    "StyleProfile",
    "TradeStats",
    "TradeTimingResult",
    "WinLossSummary",
    "compute_holding_period_summary",
    "compute_portfolio_timing_insight",
    "compute_portfolio_trade_stats",
    "compute_style_metrics",
    "compute_style_profile",
    "compute_trade_stats_by_symbol",
    "compute_trade_timing",
    "compute_win_loss_summary",
    "detect_biases",
    "min_confidence",
]
