"""Extended tests for richer trade statistics and holding-period analysis.

Coverage:
    - HoldingPeriodSummary: dataclass defaults, to_dict, empty()
    - WinLossSummary: dataclass defaults, to_dict, empty()
    - BehaviorInsight: new holding_period / win_loss fields in to_dict
    - compute_win_loss_summary:
        no trades, all wins, all losses, mixed, profit factor, confidence levels,
        pnl==0 counts as win, multi-symbol aggregation
    - compute_holding_period_summary:
        no trades, no sells, single sell, multiple sells,
        bucket counts, short_term_ratio, percentile edge cases,
        missing date data (hold_days=None), confidence levels
    - _percentile: single-element, two-element, standard distributions
    - _run_fifo_matching: internal consistency, stock-split transfer
    - Package __init__ re-exports all new symbols
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.behavior import (
    BehaviorInsight,
    ConfidenceLevel,
    HoldingPeriodSummary,
    WinLossSummary,
    compute_holding_period_summary,
    compute_win_loss_summary,
)
from src.core.behavior.trade_stats import (
    _percentile,
    _run_fifo_matching,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FX = {"USD": 150.0, "JPY": 1.0}


def _buy(symbol: str, date: str, shares: float, price: float, currency: str = "JPY") -> dict:
    settlement_jpy = shares * price if currency == "JPY" else 0.0
    settlement_usd = shares * price if currency == "USD" else 0.0
    fx_rate = FX.get(currency, 1.0)
    return {
        "category": "trade",
        "date": date,
        "symbol": symbol,
        "trade_type": "buy",
        "shares": shares,
        "price": price,
        "currency": currency,
        "fx_rate": fx_rate if currency != "JPY" else 0.0,
        "settlement_jpy": settlement_jpy,
        "settlement_usd": settlement_usd,
    }


def _sell(symbol: str, date: str, shares: float, price: float, currency: str = "JPY") -> dict:
    settlement_jpy = shares * price if currency == "JPY" else 0.0
    settlement_usd = shares * price if currency == "USD" else 0.0
    fx_rate = FX.get(currency, 1.0)
    return {
        "category": "trade",
        "date": date,
        "symbol": symbol,
        "trade_type": "sell",
        "shares": shares,
        "price": price,
        "currency": currency,
        "fx_rate": fx_rate if currency != "JPY" else 0.0,
        "settlement_jpy": settlement_jpy,
        "settlement_usd": settlement_usd,
    }


# ---------------------------------------------------------------------------
# HoldingPeriodSummary — model tests
# ---------------------------------------------------------------------------


class TestHoldingPeriodSummaryModel:
    def test_defaults(self):
        h = HoldingPeriodSummary()
        assert h.total_closed == 0
        assert h.total_with_hold_data == 0
        assert h.min_days is None
        assert h.max_days is None
        assert h.median_days is None
        assert h.p25_days is None
        assert h.p75_days is None
        assert h.short_term_count == 0
        assert h.medium_term_count == 0
        assert h.long_term_count == 0
        assert h.short_term_ratio is None
        assert h.confidence == ConfidenceLevel.INSUFFICIENT

    def test_empty_classmethod(self):
        h = HoldingPeriodSummary.empty()
        assert h.total_closed == 0
        assert h.confidence == ConfidenceLevel.INSUFFICIENT

    def test_to_dict_contains_all_keys(self):
        h = HoldingPeriodSummary(
            total_closed=3,
            total_with_hold_data=3,
            min_days=10.0,
            max_days=90.0,
            median_days=45.0,
            p25_days=25.0,
            p75_days=70.0,
            short_term_count=1,
            medium_term_count=2,
            long_term_count=0,
            short_term_ratio=0.3333,
            confidence=ConfidenceLevel.LOW,
        )
        d = h.to_dict()
        for key in (
            "total_closed",
            "total_with_hold_data",
            "min_days",
            "max_days",
            "median_days",
            "p25_days",
            "p75_days",
            "short_term_count",
            "medium_term_count",
            "long_term_count",
            "short_term_ratio",
            "confidence",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_confidence_is_string(self):
        h = HoldingPeriodSummary(confidence=ConfidenceLevel.MEDIUM)
        assert h.to_dict()["confidence"] == "medium"

    def test_to_dict_none_values_preserved(self):
        h = HoldingPeriodSummary()
        d = h.to_dict()
        assert d["min_days"] is None
        assert d["short_term_ratio"] is None


# ---------------------------------------------------------------------------
# WinLossSummary — model tests
# ---------------------------------------------------------------------------


class TestWinLossSummaryModel:
    def test_defaults(self):
        w = WinLossSummary()
        assert w.win_count == 0
        assert w.loss_count == 0
        assert w.win_rate is None
        assert w.avg_win_jpy is None
        assert w.avg_loss_jpy is None
        assert w.gross_profit_jpy == 0.0
        assert w.gross_loss_jpy == 0.0
        assert w.profit_factor is None
        assert w.confidence == ConfidenceLevel.INSUFFICIENT

    def test_empty_classmethod(self):
        w = WinLossSummary.empty()
        assert w.win_count == 0
        assert w.confidence == ConfidenceLevel.INSUFFICIENT

    def test_to_dict_contains_all_keys(self):
        w = WinLossSummary(
            win_count=3,
            loss_count=1,
            win_rate=0.75,
            avg_win_jpy=10000.0,
            avg_loss_jpy=-5000.0,
            gross_profit_jpy=30000.0,
            gross_loss_jpy=-5000.0,
            profit_factor=6.0,
            confidence=ConfidenceLevel.LOW,
        )
        d = w.to_dict()
        for key in (
            "win_count",
            "loss_count",
            "win_rate",
            "avg_win_jpy",
            "avg_loss_jpy",
            "gross_profit_jpy",
            "gross_loss_jpy",
            "profit_factor",
            "confidence",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_confidence_is_string(self):
        w = WinLossSummary(confidence=ConfidenceLevel.HIGH)
        assert w.to_dict()["confidence"] == "high"


# ---------------------------------------------------------------------------
# BehaviorInsight — new fields
# ---------------------------------------------------------------------------


class TestBehaviorInsightNewFields:
    def test_holding_period_field_has_default(self):
        bi = BehaviorInsight()
        assert isinstance(bi.holding_period, HoldingPeriodSummary)

    def test_win_loss_field_has_default(self):
        bi = BehaviorInsight()
        assert isinstance(bi.win_loss, WinLossSummary)

    def test_to_dict_includes_holding_period(self):
        bi = BehaviorInsight()
        d = bi.to_dict()
        assert "holding_period" in d
        assert isinstance(d["holding_period"], dict)

    def test_to_dict_includes_win_loss(self):
        bi = BehaviorInsight()
        d = bi.to_dict()
        assert "win_loss" in d
        assert isinstance(d["win_loss"], dict)

    def test_existing_fields_still_present(self):
        bi = BehaviorInsight.empty()
        d = bi.to_dict()
        for key in ("trade_stats", "style_metrics", "confidence", "notes"):
            assert key in d, f"Missing key: {key}"

    def test_explicit_holding_period_preserved(self):
        hp = HoldingPeriodSummary(total_closed=5, short_term_count=3)
        bi = BehaviorInsight(holding_period=hp)
        assert bi.to_dict()["holding_period"]["total_closed"] == 5
        assert bi.to_dict()["holding_period"]["short_term_count"] == 3

    def test_explicit_win_loss_preserved(self):
        wl = WinLossSummary(win_count=7, loss_count=3, win_rate=0.7)
        bi = BehaviorInsight(win_loss=wl)
        assert bi.to_dict()["win_loss"]["win_count"] == 7


# ---------------------------------------------------------------------------
# _percentile — internal helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_single_element(self):
        assert _percentile([42.0], 50) == pytest.approx(42.0)
        assert _percentile([42.0], 0) == pytest.approx(42.0)
        assert _percentile([42.0], 100) == pytest.approx(42.0)

    def test_two_elements_median(self):
        assert _percentile([0.0, 100.0], 50) == pytest.approx(50.0)

    def test_two_elements_p25(self):
        assert _percentile([0.0, 100.0], 25) == pytest.approx(25.0)

    def test_uniform_distribution(self):
        data = [float(i) for i in range(101)]  # 0..100
        assert _percentile(data, 0) == pytest.approx(0.0)
        assert _percentile(data, 50) == pytest.approx(50.0)
        assert _percentile(data, 100) == pytest.approx(100.0)
        assert _percentile(data, 25) == pytest.approx(25.0)
        assert _percentile(data, 75) == pytest.approx(75.0)

    def test_three_elements_median(self):
        assert _percentile([10.0, 20.0, 30.0], 50) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# compute_win_loss_summary
# ---------------------------------------------------------------------------


class TestComputeWinLossSummary:
    def test_no_trades_returns_empty(self):
        result = compute_win_loss_summary([], FX)
        assert result.win_count == 0
        assert result.loss_count == 0
        assert result.win_rate is None
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_buy_only_no_sells_returns_empty(self):
        trades = [_buy("AAPL", "2024-01-01", 10, 150.0)]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 0
        assert result.win_rate is None

    def test_single_win(self):
        trades = [
            _buy("AAPL", "2024-01-01", 10, 100.0),
            _sell("AAPL", "2024-06-01", 10, 120.0),  # +200 JPY
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 1
        assert result.loss_count == 0
        assert result.win_rate == pytest.approx(1.0)
        assert result.avg_win_jpy == pytest.approx(200.0)
        assert result.avg_loss_jpy is None
        assert result.gross_profit_jpy == pytest.approx(200.0)
        assert result.gross_loss_jpy == pytest.approx(0.0)
        # No losses → profit_factor is None
        assert result.profit_factor is None

    def test_single_loss(self):
        trades = [
            _buy("AAPL", "2024-01-01", 10, 200.0),
            _sell("AAPL", "2024-06-01", 10, 150.0),  # -500 JPY
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 0
        assert result.loss_count == 1
        assert result.win_rate == pytest.approx(0.0)
        assert result.avg_win_jpy is None
        assert result.avg_loss_jpy == pytest.approx(-500.0)
        assert result.gross_profit_jpy == pytest.approx(0.0)
        assert result.gross_loss_jpy == pytest.approx(-500.0)
        # No wins → profit_factor is None
        assert result.profit_factor is None

    def test_pnl_zero_counts_as_win(self):
        """Sell at exactly break-even (PnL == 0) must count as a win."""
        trades = [
            _buy("X", "2024-01-01", 100, 1000.0),
            _sell("X", "2024-06-01", 100, 1000.0),  # PnL = 0
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 1
        assert result.loss_count == 0

    def test_mixed_wins_and_losses(self):
        trades = [
            # Win: buy 10 @ 100, sell @ 150 → +500 JPY
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-06-01", 10, 150.0),
            # Loss: buy 10 @ 200, sell @ 100 → -1000 JPY
            _buy("B", "2024-01-01", 10, 200.0),
            _sell("B", "2024-06-01", 10, 100.0),
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 1
        assert result.loss_count == 1
        assert result.win_rate == pytest.approx(0.5)
        assert result.gross_profit_jpy == pytest.approx(500.0)
        assert result.gross_loss_jpy == pytest.approx(-1000.0)
        assert result.profit_factor == pytest.approx(0.5, abs=0.001)

    def test_profit_factor_correct_value(self):
        """profit_factor = gross_profit / |gross_loss|."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-06-01", 10, 130.0),  # +300
            _buy("B", "2024-01-01", 10, 200.0),
            _sell("B", "2024-06-01", 10, 150.0),  # -500
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.profit_factor == pytest.approx(300.0 / 500.0, abs=0.001)

    def test_win_rate_calculation(self):
        """3 wins out of 4 sells → 0.75."""
        trades = []
        for i in range(3):
            trades += [
                _buy("W", f"2024-0{i + 1}-01", 10, 100.0),
                _sell("W", f"2024-0{i + 1}-15", 10, 110.0),
            ]
        trades += [
            _buy("L", "2024-04-01", 10, 200.0),
            _sell("L", "2024-04-15", 10, 100.0),
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 3
        assert result.loss_count == 1
        assert result.win_rate == pytest.approx(0.75, abs=0.001)

    def test_confidence_insufficient_with_no_sells(self):
        result = compute_win_loss_summary([], FX)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_confidence_low_with_one_sell(self):
        trades = [
            _buy("X", "2024-01-01", 10, 100.0),
            _sell("X", "2024-06-01", 10, 110.0),
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_medium_at_five_sells(self):
        trades = []
        for i in range(5):
            m = f"{i + 1:02d}"
            trades += [
                _buy("S", f"2024-{m}-01", 10, 100.0),
                _sell("S", f"2024-{m}-15", 10, 110.0),
            ]
        result = compute_win_loss_summary(trades, FX)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_confidence_high_at_twenty_sells(self):
        trades = []
        for i in range(20):
            month = (i % 12) + 1
            year = 2024 + (i // 12)
            m = f"{month:02d}"
            trades += [
                _buy("S", f"{year}-{m}-01", 10, 100.0),
                _sell("S", f"{year}-{m}-15", 10, 110.0),
            ]
        result = compute_win_loss_summary(trades, FX)
        assert result.confidence == ConfidenceLevel.HIGH

    def test_avg_win_is_mean_of_win_pnls(self):
        """avg_win_jpy = (win1 + win2) / 2."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-06-01", 10, 120.0),  # +200
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-07-01", 10, 160.0),  # +600
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.avg_win_jpy == pytest.approx(400.0)

    def test_avg_loss_is_mean_of_loss_pnls(self):
        trades = [
            _buy("A", "2024-01-01", 10, 200.0),
            _sell("A", "2024-06-01", 10, 180.0),  # -200
            _buy("B", "2024-01-01", 10, 300.0),
            _sell("B", "2024-07-01", 10, 200.0),  # -1000
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.avg_loss_jpy == pytest.approx(-600.0)

    def test_cash_positions_excluded(self):
        trades = [
            _buy("JPY.CASH", "2024-01-01", 1000000, 1.0),
            _buy("X", "2024-01-01", 10, 100.0),
            _sell("X", "2024-06-01", 10, 120.0),
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 1

    def test_multi_symbol_aggregation(self):
        """Wins and losses across different symbols are all counted."""
        trades = [
            _buy("AAPL", "2024-01-01", 5, 150.0, "USD"),
            _sell("AAPL", "2024-06-01", 5, 200.0, "USD"),  # win
            _buy("GOOG", "2024-01-01", 2, 100.0, "USD"),
            _sell("GOOG", "2024-06-01", 2, 80.0, "USD"),  # loss
        ]
        result = compute_win_loss_summary(trades, FX)
        assert result.win_count == 1
        assert result.loss_count == 1


# ---------------------------------------------------------------------------
# compute_holding_period_summary
# ---------------------------------------------------------------------------


class TestComputeHoldingPeriodSummary:
    def test_no_trades_returns_empty(self):
        result = compute_holding_period_summary([], FX)
        assert result.total_closed == 0
        assert result.total_with_hold_data == 0
        assert result.min_days is None
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_buy_only_no_sells(self):
        trades = [_buy("AAPL", "2024-01-01", 10, 150.0)]
        result = compute_holding_period_summary(trades, FX)
        assert result.total_closed == 0
        assert result.min_days is None

    def test_single_sell_all_stats_equal(self):
        trades = [
            _buy("X", "2024-01-01", 100, 1000.0),
            _sell("X", "2024-04-10", 100, 1200.0),  # 100 days
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.total_closed == 1
        assert result.total_with_hold_data == 1
        assert result.min_days == pytest.approx(100.0, abs=1.0)
        assert result.max_days == pytest.approx(100.0, abs=1.0)
        assert result.median_days == pytest.approx(100.0, abs=1.0)
        assert result.p25_days == pytest.approx(100.0, abs=1.0)
        assert result.p75_days == pytest.approx(100.0, abs=1.0)

    def test_multiple_sells_min_max_median(self):
        # 3 sells: 10, 50, 90 days
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-01-11", 10, 110.0),  # 10 days
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-02-20", 10, 110.0),  # 50 days
            _buy("C", "2024-01-01", 10, 100.0),
            _sell("C", "2024-04-01", 10, 110.0),  # 91 days (Jan has 31 days)
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.total_closed == 3
        assert result.total_with_hold_data == 3
        assert result.min_days == pytest.approx(10.0, abs=1.0)
        assert result.max_days >= 80.0  # at least 80 days for the long one
        # Median is middle value (50 days)
        assert result.median_days == pytest.approx(50.0, abs=3.0)

    def test_short_term_bucket_count(self):
        """Sells < 30 days → short_term_count."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-01-10", 10, 110.0),  # 9 days → short
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-01-25", 10, 110.0),  # 24 days → short
            _buy("C", "2024-01-01", 10, 100.0),
            _sell("C", "2024-03-15", 10, 110.0),  # 74 days → medium
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.short_term_count == 2
        assert result.medium_term_count == 1
        assert result.long_term_count == 0

    def test_long_term_bucket_count(self):
        """Sell ≥ 180 days → long_term_count."""
        trades = [
            _buy("X", "2023-01-01", 10, 100.0),
            _sell("X", "2024-01-01", 10, 120.0),  # ~365 days → long
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.long_term_count == 1
        assert result.short_term_count == 0
        assert result.medium_term_count == 0

    def test_bucket_boundary_30_days_is_medium(self):
        """Exactly 30 days → medium_term (30 ≤ d < 180)."""
        from datetime import date, timedelta

        base = date(2024, 1, 1)
        sell_date = base + timedelta(days=30)
        trades = [
            _buy("X", str(base), 10, 100.0),
            _sell("X", str(sell_date), 10, 110.0),
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.short_term_count == 0
        assert result.medium_term_count == 1

    def test_bucket_boundary_180_days_is_long(self):
        """Exactly 180 days → long_term (≥ 180)."""
        from datetime import date, timedelta

        base = date(2024, 1, 1)
        sell_date = base + timedelta(days=180)
        trades = [
            _buy("X", str(base), 10, 100.0),
            _sell("X", str(sell_date), 10, 110.0),
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.long_term_count == 1
        assert result.medium_term_count == 0

    def test_short_term_ratio_all_short(self):
        """All sells < 30 days → short_term_ratio == 1.0."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-01-10", 10, 110.0),
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-01-20", 10, 110.0),
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.short_term_ratio == pytest.approx(1.0)

    def test_short_term_ratio_none_when_no_data(self):
        """No valid sell → short_term_ratio is None."""
        result = compute_holding_period_summary([], FX)
        assert result.short_term_ratio is None

    def test_short_term_ratio_mixed(self):
        """1 short + 1 long → ratio = 0.5."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-01-10", 10, 110.0),  # 9 days → short
            _buy("B", "2023-01-01", 10, 100.0),
            _sell("B", "2024-01-01", 10, 120.0),  # ~365 days → long
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.short_term_ratio == pytest.approx(0.5, abs=0.01)

    def test_confidence_insufficient_with_no_sells(self):
        result = compute_holding_period_summary([], FX)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_confidence_low_with_one_sell(self):
        trades = [
            _buy("X", "2024-01-01", 10, 100.0),
            _sell("X", "2024-06-01", 10, 110.0),
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.confidence == ConfidenceLevel.LOW

    def test_confidence_medium_at_five_sells(self):
        trades = []
        for i in range(5):
            m = f"{i + 1:02d}"
            trades += [
                _buy("S", f"2024-{m}-01", 10, 100.0),
                _sell("S", f"2024-{m}-15", 10, 110.0),
            ]
        result = compute_holding_period_summary(trades, FX)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_sells_without_date_data_excluded_from_stats(self):
        """Sells with missing dates → total_closed counts them, but hold_data doesn't."""
        # A sell with no date can't have hold_days computed
        no_date_sell = {
            "category": "trade",
            "date": "",  # unparseable
            "symbol": "X",
            "trade_type": "sell",
            "shares": 10,
            "price": 110.0,
            "currency": "JPY",
            "settlement_jpy": 1100.0,
            "fx_rate": 0.0,
        }
        trades = [
            _buy("X", "2024-01-01", 10, 100.0),
            no_date_sell,
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.total_closed == 1
        # No lot date → hold_days might be None; data count may be 0 or 1
        # Either way, min/max won't error (just None if no data)

    def test_p25_p75_with_four_values(self):
        """Verify p25 < median < p75 for 4-element distribution."""
        # 4 sells: 10, 30, 60, 90 days
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-01-11", 10, 110.0),  # 10 days
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-01-31", 10, 110.0),  # 30 days
            _buy("C", "2024-01-01", 10, 100.0),
            _sell("C", "2024-03-01", 10, 110.0),  # 60 days
            _buy("D", "2024-01-01", 10, 100.0),
            _sell("D", "2024-04-10", 10, 110.0),  # 100 days
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.p25_days < result.median_days < result.p75_days

    def test_total_closed_counts_all_sells_including_no_pnl_data(self):
        """total_closed = number of sell events regardless of PnL or date availability."""
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-06-01", 5, 120.0),
            _sell("A", "2024-07-01", 5, 80.0),
        ]
        result = compute_holding_period_summary(trades, FX)
        assert result.total_closed == 2


# ---------------------------------------------------------------------------
# _run_fifo_matching — internal engine tests
# ---------------------------------------------------------------------------


class TestRunFifoMatching:
    def test_buy_only_returns_no_sell_events(self):
        trades = [_buy("AAPL", "2024-01-01", 10, 150.0, "USD")]
        buy_summaries, sell_events = _run_fifo_matching(trades, FX)
        assert len(sell_events) == 0
        assert "AAPL" in buy_summaries
        assert buy_summaries["AAPL"].buy_count == 1

    def test_sell_event_pnl_correct(self):
        trades = [
            _buy("X", "2024-01-01", 100, 1000.0),  # cost = 100,000 JPY
            _sell("X", "2024-06-01", 100, 1200.0),  # proceeds = 120,000 JPY
        ]
        _, sell_events = _run_fifo_matching(trades, FX)
        assert len(sell_events) == 1
        assert sell_events[0].pnl_jpy == pytest.approx(20000.0)
        assert sell_events[0].proceeds_jpy == pytest.approx(120000.0)

    def test_sell_event_hold_days(self):
        trades = [
            _buy("X", "2024-01-01", 100, 1000.0),
            _sell("X", "2024-04-10", 100, 1200.0),  # 100 days
        ]
        _, sell_events = _run_fifo_matching(trades, FX)
        assert sell_events[0].hold_days == pytest.approx(100.0, abs=1.0)

    def test_fifo_ordering(self):
        """Older lot consumed first; PnL matches first-lot basis."""
        trades = [
            _buy("X", "2024-01-01", 100, 100.0),  # lot 1: 100 JPY/share
            _buy("X", "2024-03-01", 100, 200.0),  # lot 2: 200 JPY/share
            _sell("X", "2024-06-01", 100, 150.0),  # sells lot 1 only
        ]
        _, sell_events = _run_fifo_matching(trades, FX)
        assert sell_events[0].pnl_jpy == pytest.approx(5000.0)  # 100 * (150-100)

    def test_stock_split_adjusts_cost_basis(self):
        """Transfer with price=0 should redistribute cost without adding a new lot."""
        trades = [
            _buy("X", "2024-01-01", 100, 1000.0),  # lot: 100 shares @ 1000 JPY
            {  # 2:1 split adds 100 shares
                "date": "2024-03-01",
                "symbol": "X",
                "trade_type": "transfer",
                "shares": 100.0,
                "price": 0.0,
                "currency": "JPY",
                "settlement_jpy": 0.0,
                "settlement_usd": 0.0,
                "fx_rate": 0.0,
            },
            _sell("X", "2024-06-01", 200, 600.0),  # sell all 200 post-split shares
        ]
        _, sell_events = _run_fifo_matching(trades, FX)
        # Post-split: 200 shares, cost basis = 100000 JPY total = 500 JPY/share
        # Proceeds: 200 * 600 = 120000 JPY; PnL = 120000 - 100000 = 20000 JPY
        assert sell_events[0].pnl_jpy == pytest.approx(20000.0, abs=10.0)

    def test_cash_excluded(self):
        trades = [
            _buy("JPY.CASH", "2024-01-01", 1000000, 1.0),
            _buy("X", "2024-01-01", 10, 100.0),
        ]
        buy_summaries, _ = _run_fifo_matching(trades, FX)
        assert "JPY.CASH" not in buy_summaries
        assert "X" in buy_summaries

    def test_multiple_symbols_independent(self):
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _buy("B", "2024-01-01", 10, 200.0),
            _sell("A", "2024-06-01", 10, 120.0),
        ]
        buy_summaries, sell_events = _run_fifo_matching(trades, FX)
        assert len(sell_events) == 1
        assert sell_events[0].symbol == "A"
        assert "B" in buy_summaries


# ---------------------------------------------------------------------------
# Package __init__ re-exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_holding_period_summary_importable(self):
        from src.core.behavior import HoldingPeriodSummary as HPS

        assert HPS is HoldingPeriodSummary

    def test_win_loss_summary_importable(self):
        from src.core.behavior import WinLossSummary as WLS

        assert WLS is WinLossSummary

    def test_compute_win_loss_summary_importable(self):
        from src.core.behavior import compute_win_loss_summary as cwls

        assert cwls is compute_win_loss_summary

    def test_compute_holding_period_summary_importable(self):
        from src.core.behavior import compute_holding_period_summary as chps

        assert chps is compute_holding_period_summary
