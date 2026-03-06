"""Lightweight portfolio bias detection — deterministic heuristics only.

Detects four common behavioral/compositional biases from available data:

1. **Concentration bias** — portfolio too concentrated in a few positions
2. **Overtrading**        — high turnover + short holding periods
3. **Home (regional) bias** — excessive exposure to a single country/region
4. **Cash drag**          — excessive cash reducing long-term return potential

All detections are **deterministic** and based on observable data only.
No speculative psychological claims are made.  Each bias that exceeds a
heuristic threshold produces a :class:`BiasSignal` result.

Public API
----------
detect_biases(positions, style_metrics, holding_period, style_profile)
    -> list[BiasSignal]
"""

from __future__ import annotations

import logging

from src.core.behavior.models import (
    BiasSignal,
    HoldingPeriodSummary,
    StyleMetrics,
    StyleProfile,
)
from src.core.ticker_utils import infer_country

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal thresholds (documented for auditability)
# ---------------------------------------------------------------------------

# Concentration bias (position HHI)
_CONC_HIGH = 0.50  # HHI ≥ 0.50 → single dominant position
_CONC_MEDIUM = 0.35  # HHI 0.35–0.50 → moderately concentrated

# Overtrading (combined frequency + short-term ratio)
_OVERTRADE_FREQ = "active"
_OVERTRADE_SHORT_HIGH = 0.70  # >70 % short-term + active → high
_OVERTRADE_SHORT_MEDIUM = 0.45  # >45 % short-term + active → medium

# Home bias (single country weight among non-cash positions)
_HOME_HIGH = 0.90  # ≥ 90 % in one country → high
_HOME_MEDIUM = 0.75  # ≥ 75 % in one country → medium

# Cash drag
_CASH_HIGH = 0.40  # ≥ 40 % cash → high drag
_CASH_MEDIUM = 0.20  # ≥ 20 % cash → medium drag


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _detect_concentration(style_profile: StyleProfile) -> BiasSignal | None:
    """Detect concentration bias from position HHI."""
    hhi = style_profile.concentration_hhi
    if hhi is None:
        return None

    if hhi >= _CONC_HIGH:
        return BiasSignal(
            bias_type="concentration",
            severity="high",
            title="集中リスク（高）",
            description="少数銘柄に大きく集中しており、個別銘柄リスクが高い状態です。",
            notes=[
                f"ポジション集中度 HHI={hhi:.2f}（目安: 0.25未満=分散、0.50以上=高集中）",
                "銘柄数を増やすか、セクターをまたいだ分散を検討してください。",
            ],
        )
    if hhi >= _CONC_MEDIUM:
        return BiasSignal(
            bias_type="concentration",
            severity="medium",
            title="集中リスク（中）",
            description="ポジションがやや集中気味です。銘柄の追加や比率の見直しを検討できます。",
            notes=[
                f"ポジション集中度 HHI={hhi:.2f}（目安: 0.35以上=やや集中）",
            ],
        )
    return None


def _detect_overtrading(
    style_metrics: StyleMetrics,
    holding_period: HoldingPeriodSummary,
) -> BiasSignal | None:
    """Detect overtrading from trade frequency and short-term ratio."""
    if style_metrics.trade_frequency != _OVERTRADE_FREQ:
        return None
    short_ratio = holding_period.short_term_ratio
    if short_ratio is None:
        # Active frequency alone → at most low signal
        return None

    if short_ratio >= _OVERTRADE_SHORT_HIGH:
        return BiasSignal(
            bias_type="overtrading",
            severity="high",
            title="売買過多の可能性（高）",
            description=(
                "トレード頻度が高く、短期決済の比率が高めです。取引コストや税負担が累積している可能性があります。"
            ),
            notes=[
                f"月次取引頻度: アクティブ、短期（〜30日）決済比率: {short_ratio * 100:.0f}%",
                "意図した戦略に基づく頻度かどうかを確認することを推奨します。",
            ],
        )
    if short_ratio >= _OVERTRADE_SHORT_MEDIUM:
        return BiasSignal(
            bias_type="overtrading",
            severity="medium",
            title="売買過多の可能性（中）",
            description=(
                "トレード頻度がやや高く、短期売買の比率が一定以上あります。戦略と合致しているか確認してください。"
            ),
            notes=[
                f"月次取引頻度: アクティブ、短期（〜30日）決済比率: {short_ratio * 100:.0f}%",
            ],
        )
    return None


def _detect_home_bias(positions: list[dict]) -> BiasSignal | None:
    """Detect regional (home country) bias from non-cash position weights.

    Excludes cash positions and uses ``infer_country`` for region mapping.
    Reports when a single country represents ≥75 % of non-cash portfolio value.
    """
    # Build region → total equity_jpy
    region_totals: dict[str, float] = {}
    equity_total = 0.0

    for pos in positions:
        if pos.get("sector") == "Cash":
            continue
        val = float(pos.get("evaluation_jpy") or 0.0)
        if val <= 0:
            continue
        symbol = str(pos.get("symbol") or "")
        country = infer_country(symbol)
        if country == "Unknown":
            country = "その他"
        region_totals[country] = region_totals.get(country, 0.0) + val
        equity_total += val

    if equity_total <= 0 or not region_totals:
        return None

    top_country = max(region_totals, key=region_totals.__getitem__)
    top_weight = region_totals[top_country] / equity_total

    if top_weight >= _HOME_HIGH:
        return BiasSignal(
            bias_type="home_bias",
            severity="high",
            title="地域集中バイアス（高）",
            description=(
                f"株式ポートフォリオの {top_weight * 100:.0f}% が「{top_country}」に集中しています。"
                "地域リスクへの露出が高い状態です。"
            ),
            notes=[
                "海外市場や異なる地域への分散を検討してください。",
                *[
                    f"  {r}: {w / equity_total * 100:.0f}%"
                    for r, w in sorted(region_totals.items(), key=lambda x: x[1], reverse=True)[:5]
                ],
            ],
        )
    if top_weight >= _HOME_MEDIUM:
        return BiasSignal(
            bias_type="home_bias",
            severity="medium",
            title="地域集中バイアス（中）",
            description=(f"株式ポートフォリオの {top_weight * 100:.0f}% が「{top_country}」に集中しています。"),
            notes=[
                *[
                    f"  {r}: {w / equity_total * 100:.0f}%"
                    for r, w in sorted(region_totals.items(), key=lambda x: x[1], reverse=True)[:5]
                ],
            ],
        )
    return None


def _detect_cash_drag(style_profile: StyleProfile) -> BiasSignal | None:
    """Detect excessive cash drag from portfolio cash ratio."""
    cash_ratio = style_profile.cash_ratio
    if cash_ratio is None:
        return None

    if cash_ratio >= _CASH_HIGH:
        return BiasSignal(
            bias_type="cash_drag",
            severity="high",
            title="過剰現金（高）",
            description=(
                f"ポートフォリオの {cash_ratio * 100:.0f}% が現金保有です。"
                "長期的に機会損失（キャッシュドラッグ）が発生している可能性があります。"
            ),
            notes=[
                "意図的な防衛的ポジションであれば問題ありませんが、未投資資金が長期化していないか確認してください。",
            ],
        )
    if cash_ratio >= _CASH_MEDIUM:
        return BiasSignal(
            bias_type="cash_drag",
            severity="medium",
            title="現金比率がやや高め",
            description=(f"現金比率 {cash_ratio * 100:.0f}% — 一定の機会損失リスクがあります。"),
            notes=[
                "適切な現金比率は投資方針によって異なります。戦略と照らし合わせて確認してください。",
            ],
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_biases(
    positions: list[dict],
    style_metrics: StyleMetrics,
    holding_period: HoldingPeriodSummary,
    style_profile: StyleProfile,
) -> list[BiasSignal]:
    """Run all bias detectors and return the triggered signals.

    Signals are sorted by severity (high → medium → low) for display.
    Returns an empty list when no thresholds are exceeded.

    Parameters
    ----------
    positions : list[dict]
        Current portfolio positions (``evaluation_jpy``, ``sector``,
        ``symbol`` fields expected).
    style_metrics : StyleMetrics
        Trading-style characterization (frequency, holding style).
    holding_period : HoldingPeriodSummary
        Holding-period distribution including ``short_term_ratio``.
    style_profile : StyleProfile
        Computed style profile containing ``concentration_hhi``,
        ``cash_ratio``, etc.

    Returns
    -------
    list[BiasSignal]
        Detected bias signals sorted high → medium → low severity.
    """
    signals: list[BiasSignal] = []

    try:
        sig = _detect_concentration(style_profile)
        if sig is not None:
            signals.append(sig)
    except Exception as exc:  # pragma: no cover
        logger.debug("_detect_concentration failed: %s", exc)

    try:
        sig = _detect_overtrading(style_metrics, holding_period)
        if sig is not None:
            signals.append(sig)
    except Exception as exc:  # pragma: no cover
        logger.debug("_detect_overtrading failed: %s", exc)

    try:
        sig = _detect_home_bias(positions)
        if sig is not None:
            signals.append(sig)
    except Exception as exc:  # pragma: no cover
        logger.debug("_detect_home_bias failed: %s", exc)

    try:
        sig = _detect_cash_drag(style_profile)
        if sig is not None:
            signals.append(sig)
    except Exception as exc:  # pragma: no cover
        logger.debug("_detect_cash_drag failed: %s", exc)

    # Sort: high → medium → low
    _order = {"high": 0, "medium": 1, "low": 2}
    signals.sort(key=lambda s: _order.get(s.severity, 3))

    logger.debug(
        "detect_biases: %d signal(s) detected: %s",
        len(signals),
        [s.bias_type for s in signals],
    )
    return signals
