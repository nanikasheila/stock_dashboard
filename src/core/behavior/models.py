"""Behavior domain models — confidence-aware result types.

This module defines the core dataclasses and enums for accumulated-data
behavior insights.  All types are Streamlit-free and serialize via ``to_dict()``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass
class SellRecord:
    """Individual sell transaction record from FIFO matching.

    Why: Per-trade P&L data is needed for distribution analysis.
         Aggregated win/loss stats hide the distribution shape —
         histograms reveal skewness and fat tails.
    How: Extracted from FIFO matching results during trade stats computation
         and stored on both ``TradeStats`` and ``PortfolioTradeStats``.

    Attributes
    ----------
    symbol : str
        Ticker symbol of the sold position.
    sell_date : str
        Sell execution date in ``YYYY-MM-DD`` format.
        Empty string when the date could not be parsed.
    pnl_jpy : float
        Realized profit/loss for this individual sell transaction in JPY.
    holding_days : int
        Calendar days held before selling (average across consumed lots).
        Zero when date information is unavailable.
    """

    symbol: str
    sell_date: str  # YYYY-MM-DD
    pnl_jpy: float  # realized P&L in JPY
    holding_days: int  # days held before selling


class ConfidenceLevel(enum.StrEnum):
    """Data confidence level for behavior insights.

    Used to communicate how reliable an insight is, based on the
    volume and completeness of available trade history.

    Attributes
    ----------
    HIGH : str
        ≥ 20 completed sell transactions — statistically robust.
    MEDIUM : str
        5–19 sell transactions or partial data — directionally reliable.
    LOW : str
        1–4 sell transactions — treat as preliminary signals only.
    INSUFFICIENT : str
        No usable sell transaction data — insight cannot be computed.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class TradeStats:
    """Per-symbol trade statistics derived from execution history.

    Attributes
    ----------
    symbol : str
        Ticker symbol (e.g. "7203.T", "AAPL").
    buy_count : int
        Number of buy (and transfer) executions.
    sell_count : int
        Number of sell executions.
    total_buy_jpy : float
        Total buy cost in JPY (settlement-based when available).
    total_sell_jpy : float
        Total sell proceeds in JPY.
    realized_pnl_jpy : float
        FIFO-based realized profit/loss in JPY.
    win_count : int
        Number of sell executions that were profitable (realized_pnl ≥ 0).
    loss_count : int
        Number of sell executions that were unprofitable.
    avg_hold_days : float | None
        Average holding period in calendar days for closed positions.
        ``None`` when no sell transactions are available to compute it.
    confidence : ConfidenceLevel
        Reliability level for this symbol's statistics.
    """

    symbol: str
    buy_count: int = 0
    sell_count: int = 0
    total_buy_jpy: float = 0.0
    total_sell_jpy: float = 0.0
    realized_pnl_jpy: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    avg_hold_days: float | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    sell_records: list[SellRecord] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        """Total executions (buy + sell)."""
        return self.buy_count + self.sell_count

    @property
    def win_rate(self) -> float | None:
        """Win rate (0.0–1.0) based on completed sell transactions.

        Returns ``None`` when no sell data is available.
        """
        total_closed = self.win_count + self.loss_count
        if total_closed == 0:
            return None
        return self.win_count / total_closed

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "symbol": self.symbol,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "total_trades": self.total_trades,
            "total_buy_jpy": self.total_buy_jpy,
            "total_sell_jpy": self.total_sell_jpy,
            "realized_pnl_jpy": self.realized_pnl_jpy,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "avg_hold_days": self.avg_hold_days,
            "confidence": self.confidence.value,
        }


@dataclass
class PortfolioTradeStats:
    """Aggregated trade statistics across the full portfolio.

    Attributes
    ----------
    symbols_traded : list[str]
        All symbols with at least one trade, sorted alphabetically.
    total_buy_count : int
        Sum of buy executions across all symbols.
    total_sell_count : int
        Sum of sell executions across all symbols.
    total_realized_pnl_jpy : float
        Total realized profit/loss in JPY across all symbols.
    overall_win_count : int
        Total profitable sell transactions across all symbols.
    overall_loss_count : int
        Total unprofitable sell transactions across all symbols.
    avg_hold_days : float | None
        Portfolio-wide sell-count-weighted average holding period.
        ``None`` when no sell data is available.
    by_symbol : dict[str, TradeStats]
        Per-symbol breakdown.
    confidence : ConfidenceLevel
        Portfolio-wide reliability level (based on total sell count).
    """

    symbols_traded: list[str] = field(default_factory=list)
    total_buy_count: int = 0
    total_sell_count: int = 0
    total_realized_pnl_jpy: float = 0.0
    overall_win_count: int = 0
    overall_loss_count: int = 0
    avg_hold_days: float | None = None
    by_symbol: dict[str, TradeStats] = field(default_factory=dict)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    all_sell_records: list[SellRecord] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        """Total executions (buy + sell) across the portfolio."""
        return self.total_buy_count + self.total_sell_count

    @property
    def overall_win_rate(self) -> float | None:
        """Portfolio-wide win rate (0.0–1.0).

        Returns ``None`` when no sell data is available.
        """
        total_closed = self.overall_win_count + self.overall_loss_count
        if total_closed == 0:
            return None
        return self.overall_win_count / total_closed

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "symbols_traded": self.symbols_traded,
            "total_buy_count": self.total_buy_count,
            "total_sell_count": self.total_sell_count,
            "total_trades": self.total_trades,
            "total_realized_pnl_jpy": self.total_realized_pnl_jpy,
            "overall_win_count": self.overall_win_count,
            "overall_loss_count": self.overall_loss_count,
            "overall_win_rate": self.overall_win_rate,
            "avg_hold_days": self.avg_hold_days,
            "by_symbol": {k: v.to_dict() for k, v in self.by_symbol.items()},
            "confidence": self.confidence.value,
        }

    @classmethod
    def empty(cls) -> PortfolioTradeStats:
        """Return an empty result (no trade data available)."""
        return cls()


@dataclass
class HoldingPeriodSummary:
    """Distribution and summary statistics for closed-position holding periods.

    Captures how long closed trades were held, bucketed into short / medium /
    long-term categories and described by key percentiles.  A ``None`` value
    for any percentile field means insufficient date data was available to
    compute it.

    Attributes
    ----------
    total_closed : int
        Number of sell transactions processed (regardless of date availability).
    total_with_hold_data : int
        Sells where holding-period days could be computed (both lot date and
        sell date were present).
    min_days : float | None
        Shortest holding period observed.
    max_days : float | None
        Longest holding period observed.
    median_days : float | None
        50th-percentile holding period (linear interpolation).
    p25_days : float | None
        25th-percentile holding period.
    p75_days : float | None
        75th-percentile holding period.
    short_term_count : int
        Number of closed sells with hold < 30 days.
    medium_term_count : int
        Number of closed sells with 30 ≤ hold < 180 days.
    long_term_count : int
        Number of closed sells with hold ≥ 180 days.
    short_term_ratio : float | None
        ``short_term_count / total_with_hold_data`` (0.0–1.0).
        ``None`` when no hold-day data is available.
        A high value indicates a short-term trading style.
    confidence : ConfidenceLevel
        Reliability based on total closed sell count.
    """

    total_closed: int = 0
    total_with_hold_data: int = 0
    min_days: float | None = None
    max_days: float | None = None
    median_days: float | None = None
    p25_days: float | None = None
    p75_days: float | None = None
    short_term_count: int = 0
    medium_term_count: int = 0
    long_term_count: int = 0
    short_term_ratio: float | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "total_closed": self.total_closed,
            "total_with_hold_data": self.total_with_hold_data,
            "min_days": self.min_days,
            "max_days": self.max_days,
            "median_days": self.median_days,
            "p25_days": self.p25_days,
            "p75_days": self.p75_days,
            "short_term_count": self.short_term_count,
            "medium_term_count": self.medium_term_count,
            "long_term_count": self.long_term_count,
            "short_term_ratio": self.short_term_ratio,
            "confidence": self.confidence.value,
        }

    @classmethod
    def empty(cls) -> HoldingPeriodSummary:
        """Return an empty result (no hold-period data available)."""
        return cls()


@dataclass
class WinLossSummary:
    """Portfolio-wide realized win/loss statistics from FIFO-matched sells.

    Provides richer per-trade outcome statistics than the high-level win_count /
    loss_count on ``PortfolioTradeStats``.  All JPY amounts are rounded to the
    nearest whole yen.

    Attributes
    ----------
    win_count : int
        Number of sell transactions where realized PnL ≥ 0.
    loss_count : int
        Number of sell transactions where realized PnL < 0.
    win_rate : float | None
        Fraction of closed sells that were profitable (0.0–1.0).
        ``None`` when no sell data is available.
    avg_win_jpy : float | None
        Mean profit per winning sell in JPY.  ``None`` when no wins.
    avg_loss_jpy : float | None
        Mean loss per losing sell in JPY (negative value).  ``None`` when no
        losses.
    gross_profit_jpy : float
        Sum of all winning realized PnL in JPY.
    gross_loss_jpy : float
        Sum of all losing realized PnL in JPY (≤ 0).
    profit_factor : float | None
        ``gross_profit_jpy / |gross_loss_jpy|``.  ``None`` when either
        gross_profit or gross_loss is zero (avoids divide-by-zero and
        misleading infinite values).
    confidence : ConfidenceLevel
        Reliability based on total closed sell count.
    """

    win_count: int = 0
    loss_count: int = 0
    win_rate: float | None = None
    avg_win_jpy: float | None = None
    avg_loss_jpy: float | None = None
    gross_profit_jpy: float = 0.0
    gross_loss_jpy: float = 0.0
    profit_factor: float | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "avg_win_jpy": self.avg_win_jpy,
            "avg_loss_jpy": self.avg_loss_jpy,
            "gross_profit_jpy": self.gross_profit_jpy,
            "gross_loss_jpy": self.gross_loss_jpy,
            "profit_factor": self.profit_factor,
            "confidence": self.confidence.value,
        }

    @classmethod
    def empty(cls) -> WinLossSummary:
        """Return an empty result (no sell data available)."""
        return cls()


@dataclass
class StyleMetrics:
    """Trading-style characterization derived from transaction patterns.

    Attributes
    ----------
    trade_frequency : str
        "active" (≥2 trades/month avg), "moderate" (0.5–2/month),
        "passive" (<0.5/month), or "unknown" (insufficient data).
    avg_position_size_jpy : float
        Average buy amount per transaction in JPY.
    concentration_score : float
        Herfindahl–Hirschman Index (HHI) of buy volume by symbol, 0.0–1.0.
        Higher means buy volume is more concentrated in fewer symbols.
    holding_style : str
        "short_term" (avg hold < 30 days), "medium_term" (30–180 days),
        "long_term" (>180 days), or "unknown" if no sell data.
    confidence : ConfidenceLevel
        Reliability of the style characterization.
    notes : list[str]
        Human-readable caveats about data quality or computation limits.
    """

    trade_frequency: str = "unknown"
    avg_position_size_jpy: float = 0.0
    concentration_score: float = 0.0
    holding_style: str = "unknown"
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "trade_frequency": self.trade_frequency,
            "avg_position_size_jpy": self.avg_position_size_jpy,
            "concentration_score": self.concentration_score,
            "holding_style": self.holding_style,
            "confidence": self.confidence.value,
            "notes": self.notes,
        }


@dataclass
class BehaviorInsight:
    """Top-level behavior insight bundling trade stats and style analysis.

    This is the primary result type returned by the behavior integration layer
    (``components.dl_behavior.load_behavior_insight``).  It combines:
    - per-symbol and portfolio-level trade statistics
    - derived trading style characterization
    - holding-period distribution and summary
    - realized win/loss summary with profit factor
    - a unified confidence rating for the overall result quality

    Attributes
    ----------
    trade_stats : PortfolioTradeStats
        Aggregated and per-symbol trade statistics.
    style_metrics : StyleMetrics
        Derived trading style characterization.
    holding_period : HoldingPeriodSummary
        Distribution and percentile statistics for holding periods.
        Populated with defaults when no sell data is available.
    win_loss : WinLossSummary
        Realized win/loss statistics including profit factor.
        Populated with defaults when no sell data is available.
    confidence : ConfidenceLevel
        Overall confidence, taken as the minimum of sub-result confidences.
    notes : list[str]
        Human-readable notes about data quality or computation caveats.
    """

    trade_stats: PortfolioTradeStats = field(default_factory=PortfolioTradeStats)
    style_metrics: StyleMetrics = field(default_factory=StyleMetrics)
    holding_period: HoldingPeriodSummary = field(default_factory=HoldingPeriodSummary)
    win_loss: WinLossSummary = field(default_factory=WinLossSummary)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "trade_stats": self.trade_stats.to_dict(),
            "style_metrics": self.style_metrics.to_dict(),
            "holding_period": self.holding_period.to_dict(),
            "win_loss": self.win_loss.to_dict(),
            "confidence": self.confidence.value,
            "notes": self.notes,
        }

    @classmethod
    def empty(cls) -> BehaviorInsight:
        """Return an empty result for when no trade data is available."""
        return cls(
            notes=["No trade history available."],
        )


# ---------------------------------------------------------------------------
# Trade-timing result types
# ---------------------------------------------------------------------------


@dataclass
class PriceContext:
    """Price-context indicators computed from local history around a trade date.

    Attributes
    ----------
    sma_20 : float | None
        Simple moving average of the last 20 available closes before the trade.
        ``None`` when fewer than 20 bars are available.
    sma_50 : float | None
        Simple moving average of the last 50 available closes before the trade.
        ``None`` when fewer than 50 bars are available.
    rsi_14 : float | None
        RSI(14) computed from closes up to and including the trade date.
        Range 0–100. ``None`` when fewer than 15 bars are available.
    price_percentile : float | None
        Where the trade price falls in the recent price range (0 = at period
        low, 1 = at period high). ``None`` when no history is available.
    percentile_window_days : int
        Number of trading-day bars used for the percentile calculation.
    days_of_history : int
        Total bars available in the provided history for this symbol.
    """

    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    price_percentile: float | None = None
    percentile_window_days: int = 0
    days_of_history: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "rsi_14": self.rsi_14,
            "price_percentile": self.price_percentile,
            "percentile_window_days": self.percentile_window_days,
            "days_of_history": self.days_of_history,
        }


@dataclass
class TradeTimingResult:
    """Timing quality assessment for a single trade.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    trade_date : str
        Trade execution date string as supplied (YYYY-MM-DD).
    trade_type : str
        ``"buy"`` or ``"sell"``.
    trade_price : float
        Execution price per share.
    timing_score : float
        Composite timing score 0–100 (higher = better timing relative to
        local price context).
    price_context : PriceContext
        Supporting price indicators used to compute the score.
    label : str
        Human-readable quality label: ``"excellent"`` (≥80), ``"good"``
        (≥60), ``"neutral"`` (≥40), ``"poor"`` (≥20), ``"very_poor"`` (<20).
    notes : list[str]
        Explanatory notes about the score and any data limitations.
    confidence : ConfidenceLevel
        Reliability of the score based on available history depth.
    """

    symbol: str
    trade_date: str
    trade_type: str
    trade_price: float
    timing_score: float
    price_context: PriceContext
    label: str
    notes: list[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "trade_type": self.trade_type,
            "trade_price": self.trade_price,
            "timing_score": self.timing_score,
            "label": self.label,
            "price_context": self.price_context.to_dict(),
            "notes": self.notes,
            "confidence": self.confidence.value,
        }


@dataclass
class PortfolioTimingInsight:
    """Aggregated timing-quality insight across a portfolio of trades.

    Attributes
    ----------
    avg_buy_timing_score : float | None
        Mean timing score across all evaluated buy trades (0–100).
        ``None`` when no buy trades were evaluated.
    avg_sell_timing_score : float | None
        Mean timing score across all evaluated sell trades (0–100).
        ``None`` when no sell trades were evaluated.
    trade_results : list[TradeTimingResult]
        Per-trade timing results in input order.
    confidence : ConfidenceLevel
        Median confidence level across individual trade results.
    notes : list[str]
        Human-readable summary notes.
    """

    avg_buy_timing_score: float | None = None
    avg_sell_timing_score: float | None = None
    trade_results: list[TradeTimingResult] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "avg_buy_timing_score": self.avg_buy_timing_score,
            "avg_sell_timing_score": self.avg_sell_timing_score,
            "trade_results": [r.to_dict() for r in self.trade_results],
            "confidence": self.confidence.value,
            "notes": self.notes,
        }

    @classmethod
    def empty(cls) -> PortfolioTimingInsight:
        """Return an empty result when no trade data is available."""
        return cls(notes=["No trade data available for timing analysis."])


# ---------------------------------------------------------------------------
# Style-profile result types (insight-style-profile)
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """ADI-style aggression/defensiveness profile for a portfolio.

    Derived from position composition (cash ratio, concentration),
    behavioral trading patterns (frequency, holding style), and
    optional portfolio volatility metrics.

    Attributes
    ----------
    adi_score : float
        Aggression/Defense Index, 0–100.
        0 = fully defensive (高現金・低集中・長期保有・低ボラ),
        100 = fully aggressive (低現金・高集中・短期売買・高ボラ).
    label : str
        Coarse category: ``"aggressive"`` (≥62), ``"balanced"`` (38–61),
        or ``"defensive"`` (<38).
    cash_ratio : float | None
        Fraction of total portfolio value held as cash (0.0–1.0).
        ``None`` when position data is unavailable.
    concentration_hhi : float | None
        Herfindahl–Hirschman Index of non-cash position weights.
        Higher value means more concentrated in a few stocks.
        ``None`` when no equity positions are present.
    annual_volatility_pct : float | None
        Portfolio annual return volatility in % (e.g. 18.5 means 18.5%).
        ``None`` when history data is unavailable.
    beta : float | None
        Portfolio beta relative to the selected benchmark.
        ``None`` when benchmark data is unavailable.
    component_scores : dict[str, float]
        Per-component 0–100 scores used to build ``adi_score``.
        Keys: ``"cash"``, ``"concentration"``, ``"holding"``,
        ``"frequency"``, ``"volatility"`` (when available).
    confidence : ConfidenceLevel
        Reliability of the overall profile.
    notes : list[str]
        Human-readable caveats and observations.
    """

    adi_score: float = 50.0
    label: str = "balanced"
    cash_ratio: float | None = None
    concentration_hhi: float | None = None
    annual_volatility_pct: float | None = None
    beta: float | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    notes: list[str] = field(default_factory=list)

    @property
    def label_ja(self) -> str:
        """Japanese display label for the style category."""
        return {
            "aggressive": "攻め型",
            "balanced": "バランス型",
            "defensive": "守り型",
        }.get(self.label, "—")

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "adi_score": self.adi_score,
            "label": self.label,
            "label_ja": self.label_ja,
            "cash_ratio": self.cash_ratio,
            "concentration_hhi": self.concentration_hhi,
            "annual_volatility_pct": self.annual_volatility_pct,
            "beta": self.beta,
            "component_scores": self.component_scores,
            "confidence": self.confidence.value,
            "notes": self.notes,
        }

    @classmethod
    def empty(cls) -> StyleProfile:
        """Return an empty result when no data is available."""
        return cls(notes=["ポートフォリオデータが不足しています。"])


@dataclass
class BiasSignal:
    """A single detected behavioral or compositional bias.

    Represents a heuristic-based warning about a potential portfolio bias
    derived entirely from deterministic rules applied to available data.
    No speculative psychological claims are made.

    Attributes
    ----------
    bias_type : str
        Machine-readable identifier: ``"concentration"``, ``"overtrading"``,
        ``"home_bias"``, or ``"cash_drag"``.
    severity : str
        ``"high"``, ``"medium"``, or ``"low"``.
    title : str
        Short Japanese display title (e.g. ``"集中リスク"``)。
    description : str
        One-sentence Japanese description of the detected pattern.
    notes : list[str]
        Optional supporting details (quantitative context).
    """

    bias_type: str
    severity: str
    title: str
    description: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return {
            "bias_type": self.bias_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "notes": self.notes,
        }
