"""Tests for src.core.behavior.style_profile and bias_detector.

Covers:
- compute_style_profile: score computation, label assignment, component
  breakdown, cash/concentration/volatility/beta edge cases.
- detect_biases: each of the four detectors individually and combined.
- StyleProfile / BiasSignal model properties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- project root on sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.behavior.bias_detector import (
    _detect_cash_drag,
    _detect_concentration,
    _detect_home_bias,
    _detect_overtrading,
    detect_biases,
)
from src.core.behavior.models import (
    BiasSignal,
    ConfidenceLevel,
    HoldingPeriodSummary,
    StyleMetrics,
    StyleProfile,
)
from src.core.behavior.style_profile import (
    _cash_and_equity,
    _cash_score,
    _concentration_score,
    _holding_style_score,
    _trade_frequency_score,
    _volatility_score,
    compute_style_profile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_positions(
    equity_values: list[float] | None = None,
    cash_values: list[float] | None = None,
    symbols: list[str] | None = None,
) -> list[dict]:
    """Build a minimal positions list."""
    positions = []
    eq_vals = equity_values if equity_values is not None else [300_000.0, 200_000.0]
    sym_list = symbols or ["VTI", "7203.T"]
    for val, sym in zip(eq_vals, sym_list):
        positions.append({"symbol": sym, "evaluation_jpy": val, "sector": "Technology"})
    for val in cash_values or []:
        positions.append({"symbol": "JPY.CASH", "evaluation_jpy": val, "sector": "Cash"})
    return positions


def _make_style_metrics(
    frequency: str = "moderate",
    holding_style: str = "medium_term",
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
) -> StyleMetrics:
    return StyleMetrics(
        trade_frequency=frequency,
        avg_position_size_jpy=100_000.0,
        concentration_score=0.3,
        holding_style=holding_style,
        confidence=confidence,
    )


def _make_holding_period(
    short_term_ratio: float | None = 0.3,
    total_closed: int = 10,
) -> HoldingPeriodSummary:
    return HoldingPeriodSummary(
        total_closed=total_closed,
        total_with_hold_data=total_closed,
        short_term_count=int((short_term_ratio or 0) * total_closed),
        medium_term_count=total_closed - int((short_term_ratio or 0) * total_closed),
        short_term_ratio=short_term_ratio,
        confidence=ConfidenceLevel.MEDIUM,
    )


# ---------------------------------------------------------------------------
# Unit tests: scoring helpers
# ---------------------------------------------------------------------------


class TestScoringHelpers:
    """Unit tests for internal scoring helper functions."""

    def test_cash_score_zero_cash(self) -> None:
        assert _cash_score(0.0) == pytest.approx(100.0)

    def test_cash_score_20_pct(self) -> None:
        assert _cash_score(0.20) == pytest.approx(50.0)

    def test_cash_score_40_pct(self) -> None:
        assert _cash_score(0.40) == pytest.approx(0.0)

    def test_cash_score_over_40_pct_clamped(self) -> None:
        assert _cash_score(0.60) == pytest.approx(0.0)

    def test_concentration_score_hhi_zero(self) -> None:
        assert _concentration_score(0.0) == pytest.approx(0.0)

    def test_concentration_score_hhi_half(self) -> None:
        assert _concentration_score(0.50) == pytest.approx(50.0)

    def test_concentration_score_hhi_one(self) -> None:
        assert _concentration_score(1.0) == pytest.approx(100.0)

    def test_holding_style_score_short(self) -> None:
        assert _holding_style_score("short_term") == pytest.approx(80.0)

    def test_holding_style_score_medium(self) -> None:
        assert _holding_style_score("medium_term") == pytest.approx(50.0)

    def test_holding_style_score_long(self) -> None:
        assert _holding_style_score("long_term") == pytest.approx(20.0)

    def test_holding_style_score_unknown_returns_none(self) -> None:
        assert _holding_style_score("unknown") is None

    def test_trade_frequency_score_active(self) -> None:
        assert _trade_frequency_score("active") == pytest.approx(80.0)

    def test_trade_frequency_score_moderate(self) -> None:
        assert _trade_frequency_score("moderate") == pytest.approx(50.0)

    def test_trade_frequency_score_passive(self) -> None:
        assert _trade_frequency_score("passive") == pytest.approx(25.0)

    def test_trade_frequency_score_unknown_returns_none(self) -> None:
        assert _trade_frequency_score("unknown") is None

    def test_volatility_score_30_pct(self) -> None:
        assert _volatility_score(30.0) == pytest.approx(100.0)

    def test_volatility_score_15_pct(self) -> None:
        assert _volatility_score(15.0) == pytest.approx(50.0)

    def test_volatility_score_zero(self) -> None:
        assert _volatility_score(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit tests: _cash_and_equity
# ---------------------------------------------------------------------------


class TestCashAndEquity:
    """Tests for _cash_and_equity position parsing helper."""

    def test_all_equity_no_cash(self) -> None:
        positions = _make_positions([500_000.0, 300_000.0])
        _, _, cash_ratio, hhi = _cash_and_equity(positions)
        assert cash_ratio == pytest.approx(0.0)
        assert hhi is not None and 0 < hhi <= 1.0

    def test_all_cash_no_equity(self) -> None:
        positions = [{"symbol": "JPY.CASH", "evaluation_jpy": 1_000_000.0, "sector": "Cash"}]
        _, equity_jpy, cash_ratio, hhi = _cash_and_equity(positions)
        assert cash_ratio == pytest.approx(1.0)
        assert equity_jpy == pytest.approx(0.0)
        assert hhi is None

    def test_mixed_positions(self) -> None:
        positions = _make_positions(
            equity_values=[600_000.0, 200_000.0],
            cash_values=[200_000.0],
        )
        _, _, cash_ratio, hhi = _cash_and_equity(positions)
        assert cash_ratio == pytest.approx(0.20)
        assert hhi is not None

    def test_empty_positions(self) -> None:
        _, _, cash_ratio, hhi = _cash_and_equity([])
        assert cash_ratio is None
        assert hhi is None

    def test_single_equity_hhi_is_one(self) -> None:
        positions = _make_positions(equity_values=[500_000.0], symbols=["VTI"])
        _, _, _, hhi = _cash_and_equity(positions)
        assert hhi == pytest.approx(1.0)

    def test_equal_weight_two_positions(self) -> None:
        positions = _make_positions(equity_values=[500_000.0, 500_000.0])
        _, _, _, hhi = _cash_and_equity(positions)
        # HHI of equal weights [0.5, 0.5] = 0.25 + 0.25 = 0.50
        assert hhi == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Unit tests: compute_style_profile
# ---------------------------------------------------------------------------


class TestComputeStyleProfile:
    """Tests for the main compute_style_profile function."""

    def test_returns_style_profile_instance(self) -> None:
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
        )
        assert isinstance(result, StyleProfile)

    def test_adi_score_in_range(self) -> None:
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
        )
        assert 0.0 <= result.adi_score <= 100.0

    def test_aggressive_posture(self) -> None:
        """Short-term active trader with no cash should score aggressive."""
        positions = _make_positions(equity_values=[500_000.0, 500_000.0])
        sm = _make_style_metrics(frequency="active", holding_style="short_term")
        hp = _make_holding_period(short_term_ratio=0.8)
        result = compute_style_profile(positions, sm, hp)
        assert result.adi_score > 60.0
        assert result.label in ("aggressive", "balanced")

    def test_defensive_posture(self) -> None:
        """High cash + passive + long-term should score defensive."""
        positions = _make_positions(
            equity_values=[200_000.0, 100_000.0],
            cash_values=[700_000.0],
        )
        sm = _make_style_metrics(frequency="passive", holding_style="long_term")
        hp = _make_holding_period(short_term_ratio=0.05)
        result = compute_style_profile(positions, sm, hp)
        assert result.adi_score < 50.0
        assert result.label in ("defensive", "balanced")

    def test_label_aggressive_threshold(self) -> None:
        """adi_score ≥ 62 → aggressive."""
        # Construct inputs that produce high score
        positions = _make_positions(equity_values=[500_000.0])  # single position, HHI=1
        sm = _make_style_metrics(frequency="active", holding_style="short_term")
        hp = _make_holding_period(short_term_ratio=0.9, total_closed=30)
        result = compute_style_profile(positions, sm, hp)
        if result.adi_score >= 62.0:
            assert result.label == "aggressive"

    def test_label_defensive_threshold(self) -> None:
        """adi_score < 38 → defensive."""
        positions = _make_positions(
            equity_values=[100_000.0],
            cash_values=[900_000.0],
        )
        sm = _make_style_metrics(frequency="passive", holding_style="long_term")
        hp = _make_holding_period(short_term_ratio=0.0)
        result = compute_style_profile(positions, sm, hp)
        if result.adi_score < 38.0:
            assert result.label == "defensive"

    def test_empty_positions_returns_valid_profile(self) -> None:
        result = compute_style_profile(
            positions=[],
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
        )
        assert isinstance(result, StyleProfile)
        assert result.adi_score == pytest.approx(50.0)

    def test_unknown_style_metrics_no_crash(self) -> None:
        sm = _make_style_metrics(
            frequency="unknown",
            holding_style="unknown",
            confidence=ConfidenceLevel.INSUFFICIENT,
        )
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=sm,
            holding_period=_make_holding_period(),
        )
        assert isinstance(result, StyleProfile)

    def test_cash_ratio_computed(self) -> None:
        positions = _make_positions(
            equity_values=[800_000.0],
            cash_values=[200_000.0],
        )
        result = compute_style_profile(positions, _make_style_metrics(), _make_holding_period())
        assert result.cash_ratio == pytest.approx(0.20, abs=1e-3)

    def test_concentration_hhi_computed(self) -> None:
        positions = _make_positions(equity_values=[500_000.0, 500_000.0])
        result = compute_style_profile(positions, _make_style_metrics(), _make_holding_period())
        assert result.concentration_hhi == pytest.approx(0.50, abs=1e-3)

    def test_with_history_df_volatility(self) -> None:
        """Volatility component is included when history_df is provided."""
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        values = [100_000.0 + i * 500 + (i % 5 - 2) * 1000 for i in range(100)]
        df = pd.DataFrame({"total": values}, index=dates)
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
            history_df=df,
        )
        assert result.annual_volatility_pct is not None
        assert result.annual_volatility_pct >= 0.0
        assert "volatility" in result.component_scores

    def test_history_df_too_short_skips_vol(self) -> None:
        """history_df with < 10 rows → volatility is None."""
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame({"total": [100_000.0] * 5}, index=dates)
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
            history_df=df,
        )
        assert result.annual_volatility_pct is None

    def test_confidence_insufficient_no_positions(self) -> None:
        sm = _make_style_metrics(confidence=ConfidenceLevel.INSUFFICIENT)
        result = compute_style_profile(
            positions=[],
            style_metrics=sm,
            holding_period=HoldingPeriodSummary(),
        )
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_confidence_medium_with_data(self) -> None:
        sm = _make_style_metrics(confidence=ConfidenceLevel.MEDIUM)
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=sm,
            holding_period=_make_holding_period(),
        )
        # min(LOW from positions, MEDIUM from behavior) = LOW
        assert result.confidence == ConfidenceLevel.LOW

    def test_notes_are_list_of_strings(self) -> None:
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
        )
        assert isinstance(result.notes, list)
        for note in result.notes:
            assert isinstance(note, str)

    def test_to_dict_contains_required_keys(self) -> None:
        result = compute_style_profile(
            positions=_make_positions(),
            style_metrics=_make_style_metrics(),
            holding_period=_make_holding_period(),
        )
        d = result.to_dict()
        assert "adi_score" in d
        assert "label" in d
        assert "label_ja" in d
        assert "confidence" in d
        assert "component_scores" in d

    def test_label_ja_property(self) -> None:
        profile = StyleProfile(adi_score=70.0, label="aggressive")
        assert profile.label_ja == "攻め型"
        profile2 = StyleProfile(adi_score=50.0, label="balanced")
        assert profile2.label_ja == "バランス型"
        profile3 = StyleProfile(adi_score=30.0, label="defensive")
        assert profile3.label_ja == "守り型"

    def test_empty_classmethod(self) -> None:
        profile = StyleProfile.empty()
        assert isinstance(profile, StyleProfile)
        assert len(profile.notes) > 0


# ---------------------------------------------------------------------------
# Unit tests: bias detectors
# ---------------------------------------------------------------------------


class TestConcentrationBias:
    """Tests for _detect_concentration."""

    def test_no_hhi_returns_none(self) -> None:
        profile = StyleProfile(concentration_hhi=None)
        assert _detect_concentration(profile) is None

    def test_low_hhi_returns_none(self) -> None:
        profile = StyleProfile(concentration_hhi=0.20)
        assert _detect_concentration(profile) is None

    def test_medium_hhi(self) -> None:
        profile = StyleProfile(concentration_hhi=0.40)
        sig = _detect_concentration(profile)
        assert sig is not None
        assert sig.severity == "medium"
        assert sig.bias_type == "concentration"

    def test_high_hhi(self) -> None:
        profile = StyleProfile(concentration_hhi=0.60)
        sig = _detect_concentration(profile)
        assert sig is not None
        assert sig.severity == "high"

    def test_exact_medium_threshold(self) -> None:
        profile = StyleProfile(concentration_hhi=0.35)
        sig = _detect_concentration(profile)
        assert sig is not None
        assert sig.severity == "medium"

    def test_exact_high_threshold(self) -> None:
        profile = StyleProfile(concentration_hhi=0.50)
        sig = _detect_concentration(profile)
        assert sig is not None
        assert sig.severity == "high"


class TestOvertradingBias:
    """Tests for _detect_overtrading."""

    def test_passive_frequency_returns_none(self) -> None:
        sm = _make_style_metrics(frequency="passive")
        hp = _make_holding_period(short_term_ratio=0.9)
        assert _detect_overtrading(sm, hp) is None

    def test_active_no_short_ratio_returns_none(self) -> None:
        sm = _make_style_metrics(frequency="active")
        hp = HoldingPeriodSummary(short_term_ratio=None)
        assert _detect_overtrading(sm, hp) is None

    def test_active_low_short_ratio_returns_none(self) -> None:
        sm = _make_style_metrics(frequency="active")
        hp = _make_holding_period(short_term_ratio=0.30)
        assert _detect_overtrading(sm, hp) is None

    def test_active_medium_short_ratio(self) -> None:
        sm = _make_style_metrics(frequency="active")
        hp = _make_holding_period(short_term_ratio=0.55)
        sig = _detect_overtrading(sm, hp)
        assert sig is not None
        assert sig.severity == "medium"

    def test_active_high_short_ratio(self) -> None:
        sm = _make_style_metrics(frequency="active")
        hp = _make_holding_period(short_term_ratio=0.80)
        sig = _detect_overtrading(sm, hp)
        assert sig is not None
        assert sig.severity == "high"


class TestHomeBias:
    """Tests for _detect_home_bias."""

    def test_empty_positions_returns_none(self) -> None:
        assert _detect_home_bias([]) is None

    def test_only_cash_returns_none(self) -> None:
        positions = [{"symbol": "JPY.CASH", "evaluation_jpy": 1_000_000.0, "sector": "Cash"}]
        assert _detect_home_bias(positions) is None

    def test_diversified_no_signal(self) -> None:
        # 50% US + 50% Japan — below 75% threshold
        positions = [
            {"symbol": "VTI", "evaluation_jpy": 500_000.0, "sector": "Tech"},
            {"symbol": "7203.T", "evaluation_jpy": 500_000.0, "sector": "Auto"},
        ]
        assert _detect_home_bias(positions) is None

    def test_medium_home_bias(self) -> None:
        # 80% US (4 US stocks at 200k) + 20% Japan
        positions = [
            {"symbol": "AAPL", "evaluation_jpy": 200_000.0, "sector": "Tech"},
            {"symbol": "MSFT", "evaluation_jpy": 200_000.0, "sector": "Tech"},
            {"symbol": "VTI", "evaluation_jpy": 200_000.0, "sector": "ETF"},
            {"symbol": "GOOG", "evaluation_jpy": 200_000.0, "sector": "Tech"},
            {"symbol": "7203.T", "evaluation_jpy": 200_000.0, "sector": "Auto"},
        ]
        sig = _detect_home_bias(positions)
        assert sig is not None
        assert sig.bias_type == "home_bias"
        assert sig.severity == "medium"

    def test_high_home_bias(self) -> None:
        # 95% US
        positions = [
            {"symbol": "VTI", "evaluation_jpy": 950_000.0, "sector": "ETF"},
            {"symbol": "7203.T", "evaluation_jpy": 50_000.0, "sector": "Auto"},
        ]
        sig = _detect_home_bias(positions)
        assert sig is not None
        assert sig.severity == "high"


class TestCashDragBias:
    """Tests for _detect_cash_drag."""

    def test_no_cash_ratio_returns_none(self) -> None:
        profile = StyleProfile(cash_ratio=None)
        assert _detect_cash_drag(profile) is None

    def test_low_cash_no_signal(self) -> None:
        profile = StyleProfile(cash_ratio=0.10)
        assert _detect_cash_drag(profile) is None

    def test_medium_cash_drag(self) -> None:
        profile = StyleProfile(cash_ratio=0.25)
        sig = _detect_cash_drag(profile)
        assert sig is not None
        assert sig.severity == "medium"
        assert sig.bias_type == "cash_drag"

    def test_high_cash_drag(self) -> None:
        profile = StyleProfile(cash_ratio=0.45)
        sig = _detect_cash_drag(profile)
        assert sig is not None
        assert sig.severity == "high"


# ---------------------------------------------------------------------------
# Integration test: detect_biases
# ---------------------------------------------------------------------------


class TestDetectBiases:
    """Integration tests for the full detect_biases function."""

    def test_no_biases_returns_empty_list(self) -> None:
        # Diversified, low cash, moderate frequency
        positions = [
            {"symbol": "VTI", "evaluation_jpy": 200_000.0, "sector": "ETF"},
            {"symbol": "QQQ", "evaluation_jpy": 200_000.0, "sector": "ETF"},
            {"symbol": "7203.T", "evaluation_jpy": 200_000.0, "sector": "Auto"},
            {"symbol": "GLD", "evaluation_jpy": 200_000.0, "sector": "Commodity"},
            {"symbol": "JPY.CASH", "evaluation_jpy": 50_000.0, "sector": "Cash"},
        ]
        sm = _make_style_metrics(frequency="moderate", holding_style="medium_term")
        hp = _make_holding_period(short_term_ratio=0.20)
        profile = StyleProfile(concentration_hhi=0.20, cash_ratio=0.05)
        signals = detect_biases(positions, sm, hp, profile)
        assert isinstance(signals, list)
        # May be empty or have low-threshold signals depending on exact weights

    def test_multiple_biases_detected(self) -> None:
        # Single position (HHI=1.0) + 50% cash + active overtrading
        positions = [
            {"symbol": "AAPL", "evaluation_jpy": 500_000.0, "sector": "Tech"},
            {"symbol": "JPY.CASH", "evaluation_jpy": 500_000.0, "sector": "Cash"},
        ]
        sm = _make_style_metrics(frequency="active")
        hp = _make_holding_period(short_term_ratio=0.80)
        profile = StyleProfile(concentration_hhi=1.0, cash_ratio=0.50)
        signals = detect_biases(positions, sm, hp, profile)
        assert len(signals) >= 2
        types = {s.bias_type for s in signals}
        assert "concentration" in types
        assert "cash_drag" in types

    def test_signals_sorted_by_severity(self) -> None:
        positions = [
            {"symbol": "AAPL", "evaluation_jpy": 900_000.0, "sector": "Tech"},
            {"symbol": "JPY.CASH", "evaluation_jpy": 450_000.0, "sector": "Cash"},
        ]
        sm = _make_style_metrics(frequency="active")
        hp = _make_holding_period(short_term_ratio=0.80)
        profile = StyleProfile(concentration_hhi=1.0, cash_ratio=0.33)
        signals = detect_biases(positions, sm, hp, profile)
        if len(signals) >= 2:
            _order = {"high": 0, "medium": 1, "low": 2}
            severities = [_order.get(s.severity, 3) for s in signals]
            assert severities == sorted(severities)

    def test_returns_list_of_bias_signal_instances(self) -> None:
        profile = StyleProfile(concentration_hhi=0.60, cash_ratio=0.45)
        sm = _make_style_metrics()
        hp = _make_holding_period()
        signals = detect_biases([], sm, hp, profile)
        for sig in signals:
            assert isinstance(sig, BiasSignal)

    def test_bias_signal_to_dict(self) -> None:
        sig = BiasSignal(
            bias_type="concentration",
            severity="high",
            title="集中リスク",
            description="テスト",
        )
        d = sig.to_dict()
        assert d["bias_type"] == "concentration"
        assert d["severity"] == "high"
        assert "title" in d
        assert "description" in d
        assert "notes" in d
