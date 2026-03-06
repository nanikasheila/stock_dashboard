"""Portfolio style profiling — deterministic ADI scoring.

Computes an Aggression/Defense Index (ADI) score (0–100) and a coarse
style label (``"aggressive"`` / ``"balanced"`` / ``"defensive"``) from:

- Current position composition (cash ratio, position-level HHI)
- Behavioral trade patterns (frequency, holding style from ``StyleMetrics``)
- Optional portfolio volatility from a ``history_df`` DataFrame
- Optional portfolio beta vs. a benchmark price series

All logic is **Streamlit-free** and **deterministic** — no randomness,
no network calls, no LLM.  Missing inputs degrade gracefully.

Public API
----------
compute_style_profile(positions, style_metrics, holding_period,
                      history_df=None, benchmark_series=None)
    -> StyleProfile
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from src.core.behavior.models import (
    ConfidenceLevel,
    HoldingPeriodSummary,
    StyleMetrics,
    StyleProfile,
)
from src.core.behavior.trade_stats import _CONFIDENCE_ORDER, min_confidence
from src.core.portfolio.concentration import compute_hhi

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

# Aggression threshold:  >= _AGG_THRESHOLD → "aggressive"
# Defensive threshold:   <  _DEF_THRESHOLD → "defensive"
_AGG_THRESHOLD = 62.0
_DEF_THRESHOLD = 38.0


def _cash_score(cash_ratio: float) -> float:
    """Score 0–100 based on cash ratio: less cash → more aggressive (higher score).

    Mapping:
    - 0 %  cash → 100  (fully invested, aggressive posture)
    - 20 % cash → 50   (moderate cash buffer)
    - 40 %+ cash → 0   (heavy defensiveness)
    """
    return max(0.0, min(100.0, (1.0 - cash_ratio / 0.40) * 100.0))


def _concentration_score(hhi: float) -> float:
    """Score 0–100 based on position HHI: higher concentration → higher score.

    HHI 0.0 (perfectly diversified) → 0
    HHI 1.0 (single position)       → 100
    """
    return min(100.0, max(0.0, hhi * 100.0))


def _holding_style_score(holding_style: str) -> float | None:
    """Map holding_style label to aggression score.

    Returns ``None`` when the label is unknown (treated as missing data).
    """
    mapping: dict[str, float] = {
        "short_term": 80.0,  # frequent rotation — active/aggressive
        "medium_term": 50.0,  # balanced
        "long_term": 20.0,  # patient buy-and-hold — defensive
    }
    return mapping.get(holding_style)


def _trade_frequency_score(trade_frequency: str) -> float | None:
    """Map trade_frequency label to aggression score.

    Returns ``None`` when the label is unknown.
    """
    mapping: dict[str, float] = {
        "active": 80.0,  # high turnover — aggressive
        "moderate": 50.0,  # balanced
        "passive": 25.0,  # low turnover — defensive
    }
    return mapping.get(trade_frequency)


def _volatility_score(annual_vol_pct: float) -> float:
    """Score 0–100 based on portfolio annual volatility percentage.

    Scaling: 30 %/yr → 100 (aggressive); 0 %/yr → 0 (no vol at all).
    Capped at 100.
    """
    return min(100.0, max(0.0, annual_vol_pct / 30.0 * 100.0))


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------


def _compute_beta(
    history_df: pd.DataFrame,
    benchmark_series: pd.Series,
) -> float | None:
    """Estimate portfolio beta vs a benchmark price series.

    Uses OLS-style covariance estimate:
        beta = cov(r_portfolio, r_benchmark) / var(r_benchmark)

    Both series are converted to daily returns and aligned by date index
    before computation.  Returns ``None`` when there are fewer than 20
    overlapping data points.

    Parameters
    ----------
    history_df : pd.DataFrame
        Portfolio history with a ``"total"`` column.
    benchmark_series : pd.Series
        Benchmark price series aligned on a DatetimeIndex.

    Returns
    -------
    float | None
        Portfolio beta, rounded to two decimal places, or ``None``.
    """
    try:
        import pandas as pd

        port_total = history_df["total"].dropna()
        if len(port_total) < 2:
            return None

        port_returns = port_total.pct_change().dropna()
        bench_returns = benchmark_series.dropna().pct_change().dropna()

        # Align on common dates
        combined = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()

        if len(combined) < 20:
            return None

        bench_var = combined["bench"].var()
        if bench_var <= 0:
            return None

        cov = combined["port"].cov(combined["bench"])
        return round(cov / bench_var, 2)
    except Exception as exc:  # pragma: no cover
        logger.debug("_compute_beta: failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Volatility extraction
# ---------------------------------------------------------------------------


def _compute_annual_vol(history_df: pd.DataFrame) -> float | None:
    """Compute portfolio annual return volatility (%) from history_df.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must contain a ``"total"`` column.

    Returns
    -------
    float | None
        Annual volatility in percent (e.g. 18.5), or ``None`` if insufficient
        data.
    """
    try:
        total = history_df["total"].dropna()
        if len(total) < 10:
            return None
        daily_returns = total.pct_change().dropna()
        if daily_returns.empty:
            return None
        annual_vol = float(daily_returns.std() * math.sqrt(252)) * 100.0
        return round(annual_vol, 1)
    except Exception as exc:  # pragma: no cover
        logger.debug("_compute_annual_vol: failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Cash and concentration helpers
# ---------------------------------------------------------------------------


def _cash_and_equity(
    positions: list[dict],
) -> tuple[float, float, float | None, float | None]:
    """Return (cash_jpy, equity_jpy, cash_ratio, concentration_hhi).

    ``concentration_hhi`` is computed on non-cash position weights and is
    ``None`` when no non-cash positions are present.
    ``cash_ratio`` is ``None`` when total portfolio value is zero.
    """
    cash_jpy = 0.0
    equity_positions: list[float] = []

    for pos in positions:
        val = float(pos.get("evaluation_jpy") or 0.0)
        if pos.get("sector") == "Cash":
            cash_jpy += val
        elif val > 0:
            equity_positions.append(val)

    equity_jpy = sum(equity_positions)
    total_jpy = cash_jpy + equity_jpy

    cash_ratio: float | None = None
    if total_jpy > 0:
        cash_ratio = cash_jpy / total_jpy

    concentration_hhi: float | None = None
    if equity_jpy > 0 and equity_positions:
        weights = [v / equity_jpy for v in equity_positions]
        concentration_hhi = round(compute_hhi(weights), 4)

    return cash_jpy, equity_jpy, cash_ratio, concentration_hhi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_style_profile(
    positions: list[dict],
    style_metrics: StyleMetrics,
    holding_period: HoldingPeriodSummary,
    history_df: pd.DataFrame | None = None,
    benchmark_series: pd.Series | None = None,
) -> StyleProfile:
    """Compute an ADI-style aggression/defensiveness profile.

    All inputs beyond ``positions`` are optional; missing data degrades
    gracefully — the score is computed from whatever components are available
    and the confidence level reflects the data quality.

    Parameters
    ----------
    positions : list[dict]
        Current portfolio positions.  Each entry should have
        ``evaluation_jpy`` and ``sector`` (``"Cash"`` for cash entries).
    style_metrics : StyleMetrics
        Trading-style characterization from ``compute_style_metrics``.
    holding_period : HoldingPeriodSummary
        Holding-period distribution from ``compute_holding_period_summary``.
    history_df : pd.DataFrame | None
        Portfolio history DataFrame with a ``"total"`` column, used to
        compute annual volatility.  Pass ``None`` to skip.
    benchmark_series : pd.Series | None
        Benchmark price series (DatetimeIndex, price values), used to
        estimate portfolio beta.  Pass ``None`` to skip.

    Returns
    -------
    StyleProfile
        ADI score, label, component breakdown, and supporting metrics.
    """
    notes: list[str] = []

    # --- Position-level metrics ---
    _cash_jpy, _equity_jpy, cash_ratio, concentration_hhi = _cash_and_equity(positions)
    has_positions = (_cash_jpy + _equity_jpy) > 0

    # --- Optional: portfolio volatility ---
    annual_vol_pct: float | None = None
    if history_df is not None and not getattr(history_df, "empty", True):
        if "total" in history_df.columns:
            annual_vol_pct = _compute_annual_vol(history_df)

    # --- Optional: beta ---
    beta: float | None = None
    if (
        history_df is not None
        and not getattr(history_df, "empty", True)
        and benchmark_series is not None
        and not getattr(benchmark_series, "empty", True)
    ):
        beta = _compute_beta(history_df, benchmark_series)

    # ------------------------------------------------------------------ #
    # Build component scores — each is 0-100; weight allocation is:
    #   cash           : 30 %  (always if positions present)
    #   concentration  : 25 %  (always if equity positions present)
    #   holding style  : 25 %  (from StyleMetrics, when not "unknown")
    #   trade frequency: 20 %  (from StyleMetrics, when not "unknown")
    #   volatility     : 15 %  (conditional on history_df availability)
    # When a component is unavailable, its weight is excluded and the
    # remaining weights are renormalized by their actual total.
    # ------------------------------------------------------------------ #
    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    if has_positions and cash_ratio is not None:
        components["cash"] = _cash_score(cash_ratio)
        weights["cash"] = 0.30

    if concentration_hhi is not None:
        components["concentration"] = _concentration_score(concentration_hhi)
        weights["concentration"] = 0.25

    hs = _holding_style_score(style_metrics.holding_style)
    if hs is not None:
        components["holding"] = hs
        weights["holding"] = 0.25

    fs = _trade_frequency_score(style_metrics.trade_frequency)
    if fs is not None:
        components["frequency"] = fs
        weights["frequency"] = 0.20

    if annual_vol_pct is not None:
        components["volatility"] = _volatility_score(annual_vol_pct)
        weights["volatility"] = 0.15

    # --- Compute weighted average ADI score ---
    total_weight = sum(weights.values())
    if total_weight > 0 and components:
        adi_score = round(
            sum(components[k] * weights[k] for k in components) / total_weight,
            1,
        )
    else:
        adi_score = 50.0  # neutral fallback when no data at all

    # --- Determine label ---
    if adi_score >= _AGG_THRESHOLD:
        label = "aggressive"
    elif adi_score >= _DEF_THRESHOLD:
        label = "balanced"
    else:
        label = "defensive"

    # --- Determine confidence ---
    # Position data present → at least LOW
    position_conf = ConfidenceLevel.LOW if has_positions else ConfidenceLevel.INSUFFICIENT
    # Behavior data confidence
    behavior_conf = style_metrics.confidence
    # Combined: limited by the weaker of the two
    overall_confidence = min_confidence(position_conf, behavior_conf)

    # If we have volatility data too, upgrade LOW → MEDIUM when behavior is already MEDIUM+
    if (
        annual_vol_pct is not None
        and overall_confidence == ConfidenceLevel.LOW
        and _CONFIDENCE_ORDER.get(behavior_conf, 0) >= _CONFIDENCE_ORDER.get(ConfidenceLevel.MEDIUM, 0)
    ):
        overall_confidence = ConfidenceLevel.MEDIUM

    # --- Notes ---
    if not has_positions:
        notes.append("ポートフォリオデータがありません。スタイル分析をスキップしました。")
    if cash_ratio is not None and cash_ratio > 0.35:
        notes.append(f"現金比率が高め ({cash_ratio * 100:.0f}%) — 守り型の傾向があります。")
    if concentration_hhi is not None and concentration_hhi > 0.40:
        notes.append(f"ポジション集中度が高め (HHI={concentration_hhi:.2f}) — 少数銘柄に集中しています。")
    if beta is not None:
        if beta > 1.2:
            notes.append(f"ポートフォリオβ={beta:.2f} — ベンチマークより高いボラティリティ傾向。")
        elif beta < 0.8:
            notes.append(f"ポートフォリオβ={beta:.2f} — ベンチマークより低いボラティリティ傾向。")
        else:
            notes.append(f"ポートフォリオβ={beta:.2f} — ベンチマークに近い動き。")
    if annual_vol_pct is None and history_df is not None:
        notes.append("履歴データが不足しているためボラティリティ計算をスキップしました。")
    if not components:
        notes.append("分析に必要なデータが揃っていません。トレード履歴と保有銘柄データを蓄積してください。")

    return StyleProfile(
        adi_score=adi_score,
        label=label,
        cash_ratio=round(cash_ratio, 4) if cash_ratio is not None else None,
        concentration_hhi=concentration_hhi,
        annual_volatility_pct=annual_vol_pct,
        beta=beta,
        component_scores={k: round(v, 1) for k, v in components.items()},
        confidence=overall_confidence,
        notes=notes,
    )
