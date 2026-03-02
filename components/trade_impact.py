"""Trade impact analysis module.

Why: Users need to preview how a proposed trade (buy/sell/transfer) will
     change their portfolio before committing. This module computes
     before/after metrics (weights, sector allocation, currency exposure,
     concentration HHI) and renders a Streamlit comparison UI.
How: Takes a portfolio snapshot and trade parameters, simulates the
     position change, recalculates all derived metrics, and returns a
     TradeImpact dataclass.  render_trade_impact displays the diff.
     generate_trade_commentary optionally asks Copilot for Japanese commentary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import streamlit as st

from components.copilot_client import call as copilot_call
from components.copilot_client import is_available

logger = logging.getLogger(__name__)

# =====================================================================
# 定数
# =====================================================================

HHI_THRESHOLD_DISTRIBUTED: float = 0.15
HHI_THRESHOLD_MODERATE: float = 0.25

RISK_LABEL_DISTRIBUTED: str = "分散"
RISK_LABEL_MODERATE: str = "やや集中"
RISK_LABEL_CONCENTRATED: str = "危険な集中"


# =====================================================================
# TradeImpact dataclass
# =====================================================================


@dataclass
class TradeImpact:
    """Before/after portfolio metrics for a proposed trade.

    Why: A single structured object makes it easy to pass around and render.
    How: Fields cover symbol-level weight, sector/currency allocation, and
         Herfindahl–Hirschman Index (HHI) for concentration measurement.
    """

    symbol: str
    trade_type: str  # "buy" | "sell" | "transfer"
    shares: int
    trade_value_jpy: float  # shares * price * fx_rate

    # Before / After portfolio totals
    total_value_before: float
    total_value_after: float

    # Symbol weight (0–100%)
    symbol_weight_before: float
    symbol_weight_after: float

    # Sector allocation (sector → weight%)
    sector_before: dict[str, float] = field(default_factory=dict)
    sector_after: dict[str, float] = field(default_factory=dict)

    # Currency exposure (currency → weight%)
    currency_before: dict[str, float] = field(default_factory=dict)
    currency_after: dict[str, float] = field(default_factory=dict)

    # Concentration (Herfindahl–Hirschman Index)
    hhi_before: float = 0.0
    hhi_after: float = 0.0
    risk_level_before: str = ""
    risk_level_after: str = ""


# =====================================================================
# Internal helpers
# =====================================================================


def _classify_risk(hhi: float) -> str:
    """Return a Japanese risk label based on HHI thresholds.

    Why: Users need a quick qualitative reading of portfolio concentration.
    How: Simple threshold comparison against standard HHI buckets.
    """
    if hhi < HHI_THRESHOLD_DISTRIBUTED:
        return RISK_LABEL_DISTRIBUTED
    if hhi < HHI_THRESHOLD_MODERATE:
        return RISK_LABEL_MODERATE
    return RISK_LABEL_CONCENTRATED


def _compute_hhi(evaluations: list[float], total: float) -> float:
    """Compute Herfindahl–Hirschman Index from position values.

    Why: HHI summarises portfolio concentration in a single number.
    How: Sum of squared weight fractions. Returns 0 when total is zero.
    """
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in evaluations)


def _build_allocation(
    positions: list[dict],
    total: float,
    key: str,
) -> dict[str, float]:
    """Aggregate position evaluations by *key* and return weight percentages.

    Why: Reused for both sector and currency allocation breakdowns.
    How: Sums evaluation_jpy per unique key value, divides by total.
    """
    if total <= 0:
        return {}
    buckets: dict[str, float] = {}
    for pos in positions:
        label = pos.get(key) or "不明"
        buckets[label] = buckets.get(label, 0.0) + pos["evaluation_jpy"]
    return {k: round(v / total * 100, 2) for k, v in buckets.items()}


def _currency_key(pos: dict) -> str:
    """Return the representative currency for a position.

    Why: Positions may store currency under different field names.
    How: Prefer market_currency, fall back to cost_currency, then '不明'.
    """
    return pos.get("market_currency") or pos.get("cost_currency") or "不明"


# =====================================================================
# Main computation
# =====================================================================


def compute_trade_impact(
    snapshot: dict,
    trade_type: str,
    symbol: str,
    shares: int,
    price: float,
    currency: str,
    fx_rate: float,
) -> TradeImpact:
    """Simulate a trade and return before/after portfolio metrics.

    Why: Users want to evaluate portfolio impact *before* executing a trade.
    How: Deep-copies the position list, applies the trade delta to the
         target symbol (creating a new position for a buy if needed),
         then recalculates totals, weights, allocations, and HHI.

    Parameters
    ----------
    snapshot : dict
        Output of ``get_current_snapshot()`` containing ``positions``,
        ``total_value_jpy``, and ``fx_rates``.
    trade_type : str
        ``"buy"``, ``"sell"``, or ``"transfer"``.
    symbol : str
        Ticker symbol for the trade.
    shares : int
        Number of shares in the trade.
    price : float
        Unit price in *currency*.
    currency : str
        Currency code of the trade price (e.g. ``"USD"``).
    fx_rate : float
        FX rate to convert *currency* → JPY.
    """
    trade_value_jpy = shares * price * fx_rate

    # --- snapshot "before" ---------------------------------------------------
    before_positions: list[dict] = [dict(p) for p in snapshot.get("positions", [])]
    total_before: float = snapshot.get("total_value_jpy", 0.0)

    # --- build "after" positions ---------------------------------------------
    after_positions: list[dict] = [dict(p) for p in before_positions]

    target_idx: int | None = None
    for idx, pos in enumerate(after_positions):
        if pos["symbol"] == symbol:
            target_idx = idx
            break

    if trade_type == "buy":
        if target_idx is not None:
            after_positions[target_idx]["evaluation_jpy"] += trade_value_jpy
        else:
            after_positions.append(
                {
                    "symbol": symbol,
                    "sector": "",
                    "evaluation_jpy": trade_value_jpy,
                    "market_currency": currency,
                    "cost_currency": currency,
                }
            )
    elif trade_type == "sell":
        if target_idx is not None:
            after_positions[target_idx]["evaluation_jpy"] -= trade_value_jpy
            # Why: Allow value to go to zero but not negative
            if after_positions[target_idx]["evaluation_jpy"] <= 0:
                after_positions.pop(target_idx)
    # transfer: no evaluation change

    total_after = sum(p["evaluation_jpy"] for p in after_positions)

    # --- symbol weight -------------------------------------------------------
    symbol_eval_before = 0.0
    for pos in before_positions:
        if pos["symbol"] == symbol:
            symbol_eval_before = pos["evaluation_jpy"]
            break

    symbol_eval_after = 0.0
    for pos in after_positions:
        if pos["symbol"] == symbol:
            symbol_eval_after = pos["evaluation_jpy"]
            break

    weight_before = round(symbol_eval_before / total_before * 100, 2) if total_before > 0 else 0.0
    weight_after = round(symbol_eval_after / total_after * 100, 2) if total_after > 0 else 0.0

    # --- sector allocation ---------------------------------------------------
    sector_before = _build_allocation(before_positions, total_before, "sector")
    sector_after = _build_allocation(after_positions, total_after, "sector")

    # --- currency exposure ---------------------------------------------------
    # Why: Build temporary position dicts keyed by the currency helper.
    cur_before_positions = [{"evaluation_jpy": p["evaluation_jpy"], "cur": _currency_key(p)} for p in before_positions]
    cur_after_positions = [{"evaluation_jpy": p["evaluation_jpy"], "cur": _currency_key(p)} for p in after_positions]
    currency_before = _build_allocation(
        [{"evaluation_jpy": p["evaluation_jpy"], "cur": p["cur"]} for p in cur_before_positions],
        total_before,
        "cur",
    )
    currency_after = _build_allocation(
        [{"evaluation_jpy": p["evaluation_jpy"], "cur": p["cur"]} for p in cur_after_positions],
        total_after,
        "cur",
    )

    # --- HHI -----------------------------------------------------------------
    evals_before = [p["evaluation_jpy"] for p in before_positions]
    evals_after = [p["evaluation_jpy"] for p in after_positions]

    hhi_before = _compute_hhi(evals_before, total_before)
    hhi_after = _compute_hhi(evals_after, total_after)

    return TradeImpact(
        symbol=symbol,
        trade_type=trade_type,
        shares=shares,
        trade_value_jpy=trade_value_jpy,
        total_value_before=total_before,
        total_value_after=total_after,
        symbol_weight_before=weight_before,
        symbol_weight_after=weight_after,
        sector_before=sector_before,
        sector_after=sector_after,
        currency_before=currency_before,
        currency_after=currency_after,
        hhi_before=hhi_before,
        hhi_after=hhi_after,
        risk_level_before=_classify_risk(hhi_before),
        risk_level_after=_classify_risk(hhi_after),
    )


# =====================================================================
# Streamlit UI
# =====================================================================


def render_trade_impact(impact: TradeImpact, settings: dict) -> None:
    """Render before/after trade impact comparison in Streamlit.

    Why: Visual diff of key portfolio metrics helps users make informed
         trade decisions.
    How: Displays three st.metric columns (total value, symbol weight, HHI),
         sector/currency change tables when allocations differ, and an
         optional LLM commentary button gated by settings.
    """
    st.subheader("📊 取引インパクト分析")

    # --- Key metrics in columns ----------------------------------------------
    col1, col2, col3 = st.columns(3)

    total_delta = impact.total_value_after - impact.total_value_before
    with col1:
        st.metric(
            label="ポートフォリオ総額",
            value=f"¥{impact.total_value_after:,.0f}",
            delta=f"¥{total_delta:,.0f}",
        )

    weight_delta = impact.symbol_weight_after - impact.symbol_weight_before
    with col2:
        st.metric(
            label=f"銘柄比率 ({impact.symbol})",
            value=f"{impact.symbol_weight_after:.2f}%",
            delta=f"{weight_delta:+.2f}%",
        )

    hhi_delta = impact.hhi_after - impact.hhi_before
    # Why: Higher HHI = worse diversification → invert delta colour
    with col3:
        st.metric(
            label="集中度 (HHI)",
            value=f"{impact.hhi_after:.4f} ({impact.risk_level_after})",
            delta=f"{hhi_delta:+.4f}",
            delta_color="inverse",
        )

    # --- Sector allocation changes -------------------------------------------
    if impact.sector_before != impact.sector_after:
        st.markdown("**セクター配分の変化**")
        all_sectors = sorted(set(impact.sector_before) | set(impact.sector_after))
        rows = []
        for sec in all_sectors:
            before_val = impact.sector_before.get(sec, 0.0)
            after_val = impact.sector_after.get(sec, 0.0)
            diff = after_val - before_val
            rows.append(
                {
                    "セクター": sec,
                    "Before (%)": f"{before_val:.2f}",
                    "After (%)": f"{after_val:.2f}",
                    "変化": f"{diff:+.2f}",
                }
            )
        st.table(rows)

    # --- Currency exposure changes -------------------------------------------
    if impact.currency_before != impact.currency_after:
        st.markdown("**通貨エクスポージャーの変化**")
        all_currencies = sorted(set(impact.currency_before) | set(impact.currency_after))
        rows = []
        for cur in all_currencies:
            before_val = impact.currency_before.get(cur, 0.0)
            after_val = impact.currency_after.get(cur, 0.0)
            diff = after_val - before_val
            rows.append(
                {
                    "通貨": cur,
                    "Before (%)": f"{before_val:.2f}",
                    "After (%)": f"{after_val:.2f}",
                    "変化": f"{diff:+.2f}",
                }
            )
        st.table(rows)

    # --- Optional LLM commentary ---------------------------------------------
    if settings.get("trade_preview_llm", True) and is_available():
        if st.button("🤖 AIコメント", key="trade_impact_ai_comment"):
            with st.spinner("AIが分析中..."):
                commentary = generate_trade_commentary(
                    impact,
                    model=settings.get("copilot_model"),
                    source="trade_impact",
                )
                if commentary:
                    st.info(commentary)
                else:
                    st.warning("AIコメントを取得できませんでした。")


# =====================================================================
# LLM commentary
# =====================================================================


def generate_trade_commentary(
    impact: TradeImpact,
    model: str | None = None,
    source: str = "trade_impact",
) -> str | None:
    """Ask Copilot LLM for Japanese commentary on the trade impact.

    Why: Qualitative interpretation of numeric changes helps less
         experienced investors understand portfolio implications.
    How: Builds a structured Japanese prompt with before/after metrics and
         sends it via copilot_call. Returns raw response string or None.
    """
    type_label = {"buy": "購入", "sell": "売却", "transfer": "振替"}.get(impact.trade_type, impact.trade_type)

    sector_before_str = ", ".join(f"{k}: {v:.1f}%" for k, v in impact.sector_before.items())
    sector_after_str = ", ".join(f"{k}: {v:.1f}%" for k, v in impact.sector_after.items())

    prompt = f"""\
以下の取引がポートフォリオに与える影響を日本語で簡潔に分析してください。

## 取引内容
- 種別: {type_label}
- 銘柄: {impact.symbol}
- 株数: {impact.shares}
- 取引額(円): ¥{impact.trade_value_jpy:,.0f}

## ポートフォリオ変化
- 総額: ¥{impact.total_value_before:,.0f} → ¥{impact.total_value_after:,.0f}
- 銘柄比率: {impact.symbol_weight_before:.2f}% → {impact.symbol_weight_after:.2f}%
- HHI: {impact.hhi_before:.4f} ({impact.risk_level_before}) → {impact.hhi_after:.4f} ({impact.risk_level_after})
- セクター配分(前): {sector_before_str or "なし"}
- セクター配分(後): {sector_after_str or "なし"}

## 回答項目
1. この取引の影響評価
2. リスクの変化
3. 推奨事項
"""

    try:
        kwargs: dict = {"source": source}
        if model is not None:
            kwargs["model"] = model
        return copilot_call(prompt, **kwargs)
    except Exception:
        logger.warning("LLM commentary generation failed", exc_info=True)
        return None
