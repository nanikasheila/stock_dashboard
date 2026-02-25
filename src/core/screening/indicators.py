"""Financial indicators and value-score calculation."""

from typing import Optional


def is_undervalued_per(per: Optional[float], threshold: float = 15.0) -> bool:
    """Return True if PER indicates undervaluation (0 < per < threshold)."""
    if per is None:
        return False
    return 0 < per < threshold


def is_undervalued_pbr(pbr: Optional[float], threshold: float = 1.0) -> bool:
    """Return True if PBR indicates undervaluation (0 < pbr < threshold)."""
    if pbr is None:
        return False
    return 0 < pbr < threshold


def has_good_dividend(dividend_yield: Optional[float], min_yield: float = 0.03) -> bool:
    """Return True if dividend yield meets the minimum threshold."""
    if dividend_yield is None:
        return False
    return dividend_yield >= min_yield


def has_good_roe(roe: Optional[float], min_roe: float = 0.08) -> bool:
    """Return True if ROE meets the minimum threshold."""
    if roe is None:
        return False
    return roe >= min_roe


def _score_per(per: Optional[float], per_max: float) -> float:
    """PER score (25 points max). Lower PER = higher score."""
    if per is None or per <= 0:
        return 0.0
    if per >= per_max * 2:
        return 0.0
    # Linear: PER == 0 -> 25, PER == per_max*2 -> 0
    score = max(0.0, 25.0 * (1.0 - per / (per_max * 2)))
    return round(score, 2)


def _score_pbr(pbr: Optional[float], pbr_max: float) -> float:
    """PBR score (25 points max). Lower PBR = higher score."""
    if pbr is None or pbr <= 0:
        return 0.0
    if pbr >= pbr_max * 2:
        return 0.0
    score = max(0.0, 25.0 * (1.0 - pbr / (pbr_max * 2)))
    return round(score, 2)


def _score_dividend(dividend_yield: Optional[float], div_min: float) -> float:
    """Dividend yield score (20 points max). Higher yield = higher score."""
    if dividend_yield is None or dividend_yield <= 0:
        return 0.0
    # Cap at div_min * 3 for max score
    cap = div_min * 3
    ratio = min(dividend_yield / cap, 1.0)
    return round(20.0 * ratio, 2)


def _score_roe(roe: Optional[float], roe_min: float) -> float:
    """ROE score (15 points max). Higher ROE = higher score."""
    if roe is None or roe <= 0:
        return 0.0
    # Cap at roe_min * 3 for max score
    cap = roe_min * 3
    ratio = min(roe / cap, 1.0)
    return round(15.0 * ratio, 2)


def _score_growth(revenue_growth: Optional[float]) -> float:
    """Revenue growth score (15 points max). Higher growth = higher score."""
    if revenue_growth is None:
        return 0.0
    if revenue_growth <= 0:
        return 0.0
    # Cap at 30% growth for max score
    cap = 0.30
    ratio = min(revenue_growth / cap, 1.0)
    return round(15.0 * ratio, 2)


def calculate_value_score(stock_data: dict, thresholds: Optional[dict] = None) -> float:
    """Calculate a composite value score (0-100) for a stock.

    Breakdown:
      - PER undervaluation:  25 points
      - PBR undervaluation:  25 points
      - Dividend yield:      20 points
      - ROE:                 15 points
      - Revenue growth:      15 points

    Parameters
    ----------
    stock_data : dict
        Keys: 'trailingPE' (or 'per'), 'priceToBook' (or 'pbr'),
              'dividendYield' (or 'dividend_yield'),
              'returnOnEquity' (or 'roe'), 'revenueGrowth' (or 'revenue_growth').
    thresholds : dict, optional
        Keys: 'per_max', 'pbr_max', 'dividend_yield_min', 'roe_min'.
    """
    if thresholds is None:
        thresholds = {}

    per_max = thresholds.get("per_max", 15.0)
    pbr_max = thresholds.get("pbr_max", 1.0)
    div_min = thresholds.get("dividend_yield_min", 0.03)
    roe_min = thresholds.get("roe_min", 0.08)

    # Support both yahoo raw keys and our normalised keys
    per = stock_data.get("trailingPE") or stock_data.get("per")
    pbr = stock_data.get("priceToBook") or stock_data.get("pbr")
    # Prefer trailing (actual) dividend yield, fallback to forward/predicted
    div_yield = (
        stock_data.get("dividend_yield_trailing")
        or stock_data.get("dividendYield")
        or stock_data.get("dividend_yield")
    )
    roe = stock_data.get("returnOnEquity") or stock_data.get("roe")
    growth = stock_data.get("revenueGrowth") or stock_data.get("revenue_growth")

    total = (
        _score_per(per, per_max)
        + _score_pbr(pbr, pbr_max)
        + _score_dividend(div_yield, div_min)
        + _score_roe(roe, roe_min)
        + _score_growth(growth)
    )
    return round(min(total, 100.0), 2)


def calculate_shareholder_return_history(stock: dict) -> list[dict]:
    """Calculate shareholder return for multiple fiscal years.

    Uses ``dividend_paid_history``, ``stock_repurchase_history``, and
    ``cashflow_fiscal_years`` from yahoo_client's ``get_stock_detail``.

    Returns a list of dicts (latest-first), each containing:
        fiscal_year, dividend_paid, stock_repurchase,
        total_return_amount, total_return_rate.

    Falls back to current single-period data if history is unavailable.
    """
    market_cap = stock.get("market_cap")
    div_hist: list[float] = stock.get("dividend_paid_history") or []
    rep_hist: list[float] = stock.get("stock_repurchase_history") or []
    fiscal_years: list[int] = stock.get("cashflow_fiscal_years") or []

    if not div_hist and not rep_hist:
        # Fallback: use single-period dividend_paid / stock_repurchase
        div_raw = stock.get("dividend_paid")
        rep_raw = stock.get("stock_repurchase")
        if div_raw is None and rep_raw is None:
            return []
        dividend_paid = abs(div_raw) if div_raw is not None else None
        stock_repurchase = abs(rep_raw) if rep_raw is not None else None
        total = (dividend_paid or 0.0) + (stock_repurchase or 0.0)
        total_rate = None
        if market_cap is not None and market_cap > 0:
            total_rate = total / market_cap
        return [{
            "fiscal_year": None,
            "dividend_paid": dividend_paid,
            "stock_repurchase": stock_repurchase,
            "total_return_amount": total,
            "total_return_rate": total_rate,
        }]

    n = max(len(div_hist), len(rep_hist))
    results: list[dict] = []
    for i in range(n):
        div_raw = div_hist[i] if i < len(div_hist) else None
        rep_raw = rep_hist[i] if i < len(rep_hist) else None
        fy = fiscal_years[i] if i < len(fiscal_years) else None

        dividend_paid = abs(div_raw) if div_raw is not None else None
        stock_repurchase = abs(rep_raw) if rep_raw is not None else None

        total: Optional[float] = None
        if dividend_paid is not None or stock_repurchase is not None:
            total = (dividend_paid or 0.0) + (stock_repurchase or 0.0)

        total_rate: Optional[float] = None
        if market_cap is not None and market_cap > 0 and total is not None:
            total_rate = total / market_cap

        results.append({
            "fiscal_year": fy,
            "dividend_paid": dividend_paid,
            "stock_repurchase": stock_repurchase,
            "total_return_amount": total,
            "total_return_rate": total_rate,
        })

    return results


def assess_return_stability(history: list[dict]) -> dict:
    """Assess shareholder return stability from multi-year history.

    Takes the output of ``calculate_shareholder_return_history()`` and
    classifies stability as one of:

    - ``temporary``: Latest year surged (>=2x previous) — likely one-off
    - ``increasing``: Rates rising year-over-year
    - ``decreasing``: Rates falling year-over-year
    - ``stable``: All years >= 5%
    - ``mixed``: None of the above patterns
    - ``single_high``: 1 year only, total return >= 5%
    - ``single_moderate``: 1 year only, total return 2-5%
    - ``single_low``: 1 year only, total return < 2%
    - ``no_data``: No valid return rate data at all

    Returns dict with keys: stability, label, latest_rate, avg_rate, reason.
    """
    rates = [
        e.get("total_return_rate")
        for e in history
        if e.get("total_return_rate") is not None
    ]

    # No data at all (KIK-388)
    if len(rates) == 0:
        return {
            "stability": "no_data",
            "label": "-",
            "latest_rate": None,
            "avg_rate": None,
            "reason": None,
        }

    if len(rates) < 2:
        rate = rates[0]
        if rate >= 0.05:
            return {
                "stability": "single_high",
                "label": "💰 高還元",
                "latest_rate": rate,
                "avg_rate": rate,
                "reason": f"1年データ（{rate * 100:.1f}%）",
            }
        elif rate >= 0.02:
            return {
                "stability": "single_moderate",
                "label": "💰 還元あり",
                "latest_rate": rate,
                "avg_rate": rate,
                "reason": f"1年データ（{rate * 100:.1f}%）",
            }
        else:
            return {
                "stability": "single_low",
                "label": "➖ 低還元",
                "latest_rate": rate,
                "avg_rate": rate,
                "reason": f"1年データ（{rate * 100:.1f}%）",
            }

    latest = rates[0]
    prev = rates[1]
    avg_rate = sum(rates) / len(rates)

    # Temporary: latest surged >= 2x previous AND is genuinely high (>= 8%)
    if prev > 0 and latest / prev >= 2.0 and latest >= 0.08:
        return {
            "stability": "temporary",
            "label": "⚠️ 一時的高還元",
            "latest_rate": latest,
            "avg_rate": avg_rate,
            "reason": f"前年比{latest / prev:.1f}倍に急増",
        }

    # Increasing: all years non-decreasing (latest first order)
    if all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1)):
        return {
            "stability": "increasing",
            "label": "📈 増加傾向",
            "latest_rate": latest,
            "avg_rate": avg_rate,
            "reason": f"{len(rates)}年連続増加",
        }

    # Decreasing: all years non-increasing
    if all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)):
        return {
            "stability": "decreasing",
            "label": "📉 減少傾向",
            "latest_rate": latest,
            "avg_rate": avg_rate,
            "reason": f"{len(rates)}年連続減少",
        }

    # Stable: all >= 5%
    if all(r >= 0.05 for r in rates):
        return {
            "stability": "stable",
            "label": "✅ 安定高還元",
            "latest_rate": latest,
            "avg_rate": avg_rate,
            "reason": f"{len(rates)}年平均{avg_rate * 100:.1f}%で安定",
        }

    return {
        "stability": "mixed",
        "label": "➡️ 変動あり",
        "latest_rate": latest,
        "avg_rate": avg_rate,
        "reason": f"{len(rates)}年平均: {avg_rate * 100:.1f}%",
    }


def calculate_shareholder_return(stock: dict) -> dict:
    """Calculate total shareholder return rate.

    Formula: (|dividend_paid| + |stock_repurchase|) / market_cap

    Cashflow values from yfinance are negative (outflows), so abs() is applied.

    Returns a dict with rates as ratios (e.g. 0.05 = 5%).
    """
    market_cap = stock.get("market_cap")
    div_raw = stock.get("dividend_paid")
    rep_raw = stock.get("stock_repurchase")

    dividend_paid = abs(div_raw) if div_raw is not None else None
    stock_repurchase = abs(rep_raw) if rep_raw is not None else None

    total: float | None = None
    if dividend_paid is not None or stock_repurchase is not None:
        total = (dividend_paid or 0.0) + (stock_repurchase or 0.0)

    total_rate: float | None = None
    buyback_yield: float | None = None
    if market_cap is not None and market_cap > 0:
        if total is not None:
            total_rate = total / market_cap
        if stock_repurchase is not None:
            buyback_yield = stock_repurchase / market_cap

    # Prefer trailing (actual) dividend yield, fallback to forward/predicted
    div_yield = stock.get("dividend_yield_trailing") or stock.get("dividend_yield")

    return {
        "dividend_paid": dividend_paid,
        "stock_repurchase": stock_repurchase,
        "total_return_amount": total,
        "total_return_rate": total_rate,
        "dividend_yield": div_yield,
        "buyback_yield": buyback_yield,
    }


# ---------------------------------------------------------------------------
# Consistency checks (analysis.md – 再発防止)
# ---------------------------------------------------------------------------

def check_eps_direction(stock: dict) -> Optional[dict]:
    """Check if FwdEPS < TrailEPS (earnings decline forecast).

    Returns a warning dict or None if no issue detected.
    """
    forward_eps = stock.get("forward_eps")
    trailing_eps = stock.get("eps_current")
    if forward_eps is None or trailing_eps is None:
        return None
    if trailing_eps == 0:
        return None
    growth = (forward_eps / trailing_eps - 1) * 100
    if forward_eps < trailing_eps:
        return {
            "level": "warning",
            "code": "EPS_DECLINE",
            "message": (
                f"🔴 減益予想: FwdEPS {forward_eps:.1f} < TrailEPS "
                f"{trailing_eps:.1f} ({growth:+.1f}%)"
            ),
            "forward_eps": forward_eps,
            "trailing_eps": trailing_eps,
            "growth_pct": round(growth, 2),
        }
    return None


def check_growth_consistency(stock: dict) -> Optional[dict]:
    """Check if PEG denominator (earningsGrowth) contradicts FwdEPS direction.

    earningsGrowth is backward-looking; FwdEPS is forward-looking.
    If earningsGrowth > 0 but FwdEPS < TrailEPS, PEG is misleading.
    """
    earnings_growth = stock.get("earnings_growth")
    forward_eps = stock.get("forward_eps")
    trailing_eps = stock.get("eps_current")
    forward_per = stock.get("forward_per")
    if earnings_growth is None or forward_eps is None or trailing_eps is None:
        return None
    if trailing_eps == 0:
        return None
    # Skip when trailing EPS is negative: the growth formula is
    # meaningless (e.g. -0.5 -> +7.0 = "recovery", not decline).
    if trailing_eps < 0:
        return None

    fwd_growth = (forward_eps / trailing_eps - 1)
    # earningsGrowth positive but forward decline
    if earnings_growth > 0.05 and fwd_growth < -0.02:
        peg_past = None
        if forward_per is not None and earnings_growth > 0:
            peg_past = forward_per / (earnings_growth * 100)
        msg = (
            f"⚠️ PEG矛盾: 過去earningsGrowth {earnings_growth*100:+.1f}% "
            f"だがFwdEPS成長率 {fwd_growth*100:+.1f}%。"
        )
        if peg_past is not None:
            msg += f" PEG({peg_past:.2f})は過去ベースで将来を反映していない"
        return {
            "level": "warning",
            "code": "PEG_INCONSISTENCY",
            "message": msg,
            "earnings_growth": earnings_growth,
            "fwd_eps_growth": round(fwd_growth, 4),
            "peg_past_based": peg_past,
        }
    return None


def check_margin_deterioration(stock: dict) -> Optional[dict]:
    """Check if gross margin declined significantly (>= 5pt) across fiscal years.

    Uses revenue_history and gross_profit data if available, otherwise
    falls back to operating_margin from info.
    """
    # This check requires historical gross margin data which may not always
    # be present in the standard data dict. Return None if insufficient data.
    gross_margins = stock.get("gross_margins_history")
    if gross_margins and len(gross_margins) >= 2:
        latest = gross_margins[0]
        previous = gross_margins[1]
        if latest is not None and previous is not None:
            delta_pt = (latest - previous) * 100
            if delta_pt <= -5.0:
                return {
                    "level": "warning",
                    "code": "MARGIN_DETERIORATION",
                    "message": (
                        f"⚠️ 粗利率悪化: {previous*100:.1f}% → "
                        f"{latest*100:.1f}% ({delta_pt:+.1f}pt)"
                    ),
                    "latest": latest,
                    "previous": previous,
                    "delta_pt": round(delta_pt, 2),
                }
    return None


def check_quarterly_eps_trend(stock: dict) -> Optional[dict]:
    """Check if the most recent quarterly EPS declined QoQ.

    Uses quarterly_eps list (latest-first) if available.
    """
    q_eps = stock.get("quarterly_eps")
    if not q_eps or len(q_eps) < 2:
        return None
    latest = q_eps[0]
    previous = q_eps[1]
    if latest is None or previous is None or previous == 0:
        return None
    change_pct = (latest / previous - 1) * 100
    if latest < previous:
        return {
            "level": "caution",
            "code": "EPS_DECELERATION",
            "message": (
                f"⚠️ 四半期EPS鈍化: {previous:.2f} → "
                f"{latest:.2f} ({change_pct:+.1f}% QoQ)"
            ),
            "latest_q_eps": latest,
            "previous_q_eps": previous,
            "change_pct": round(change_pct, 2),
        }
    return None


def run_consistency_checks(stock: dict) -> list[dict]:
    """Run all valuation consistency checks and return a list of warnings.

    Each warning is a dict with keys: level, code, message, and
    check-specific data fields.

    Returns an empty list if no issues detected.
    """
    checks = [
        check_eps_direction,
        check_growth_consistency,
        check_margin_deterioration,
        check_quarterly_eps_trend,
    ]
    warnings = []
    for check_fn in checks:
        result = check_fn(stock)
        if result is not None:
            warnings.append(result)
    return warnings
