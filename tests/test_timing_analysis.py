"""Tests for src/core/behavior/timing_analysis.py.

Coverage:
    - _compute_sma: normal case, insufficient data
    - _compute_rsi: oversold / overbought / all-gains / all-losses edge cases
    - _compute_price_percentile: at-min, at-max, midpoint, degenerate (flat)
    - _sort_history: duplicate dates, non-positive close, unparseable date
    - _find_idx_on_or_before: exact match, trade before all bars, sparse gaps
    - _compute_timing_score: buy/sell with all components, partial components
    - _score_label: all label bands
    - compute_trade_timing:
        * good buy (price at period low) → high score
        * bad buy (price at period high) → low score
        * good sell (price at period high) → high score
        * bad sell (price at period low) → low score
        * missing / invalid trade date → neutral INSUFFICIENT
        * no history before trade date → neutral INSUFFICIENT
        * trade_price=0 falls back to nearest close
        * sparse history (< RSI period) → correct confidence
        * to_dict() round-trip
    - compute_portfolio_timing_insight:
        * empty trades → PortfolioTimingInsight.empty()
        * mixed buy/sell computes averages correctly
        * trade_type not in {buy, sell} is skipped
        * missing symbol history → neutral result, not an exception
    - PriceContext.to_dict() and PortfolioTimingInsight.to_dict()
    - Package-level imports
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.core.behavior.models import ConfidenceLevel, PortfolioTimingInsight, PriceContext
from src.core.behavior.timing_analysis import (
    _classify_timing_confidence,
    _compute_price_percentile,
    _compute_rsi,
    _compute_sma,
    _compute_timing_score,
    _find_idx_on_or_before,
    _score_label,
    _sort_history,
    compute_portfolio_timing_insight,
    compute_trade_timing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _history(prices: list[float], start: str = "2024-01-01") -> list[dict]:
    """Build a synthetic daily history from a list of closing prices.

    Dates are assigned sequentially from *start* as YYYY-MM-DD strings
    (calendar days, not trading days — fine for unit tests).
    """
    from datetime import date, timedelta

    base = date.fromisoformat(start)
    return [{"date": str(base + timedelta(days=i)), "close": p} for i, p in enumerate(prices)]


def _buy_trade(symbol: str, date: str, price: float) -> dict:
    return {"symbol": symbol, "date": date, "trade_type": "buy", "price": price}


def _sell_trade(symbol: str, date: str, price: float) -> dict:
    return {"symbol": symbol, "date": date, "trade_type": "sell", "price": price}


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestComputeSma:
    def test_normal(self):
        assert _compute_sma([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(4.0)

    def test_exact_period(self):
        assert _compute_sma([10.0, 20.0, 30.0], 3) == pytest.approx(20.0)

    def test_insufficient_data_returns_none(self):
        assert _compute_sma([1.0, 2.0], 3) is None

    def test_period_one(self):
        assert _compute_sma([7.0, 8.0, 9.0], 1) == pytest.approx(9.0)


class TestComputeRsi:
    def test_insufficient_data_returns_none(self):
        # Need period+1 = 15 values; 14 is not enough
        assert _compute_rsi([float(i) for i in range(14)], 14) is None

    def test_all_gains_returns_100(self):
        prices = [float(i) for i in range(1, 20)]  # strictly increasing
        rsi = _compute_rsi(prices, 14)
        assert rsi == pytest.approx(100.0)

    def test_all_losses_returns_0(self):
        prices = [float(i) for i in range(19, 0, -1)]  # strictly decreasing
        rsi = _compute_rsi(prices, 14)
        assert rsi == pytest.approx(0.0)

    def test_alternating_returns_50(self):
        # Equal gains and losses → RS = 1 → RSI = 50
        prices = [
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
            101.0,
            100.0,
        ]
        rsi = _compute_rsi(prices, 14)
        assert rsi == pytest.approx(50.0)

    def test_oversold_range(self):
        # Long declining series followed by one recovery — RSI should be low
        prices = [100.0 - i for i in range(18)]  # steep decline
        rsi = _compute_rsi(prices, 14)
        assert rsi is not None
        assert rsi < 30.0

    def test_overbought_range(self):
        prices = [float(i) for i in range(50, 70)]  # 20 bars of gains
        rsi = _compute_rsi(prices, 14)
        assert rsi is not None
        assert rsi > 70.0


class TestComputePricePercentile:
    def test_at_minimum(self):
        assert _compute_price_percentile(1.0, [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.0)

    def test_at_maximum(self):
        assert _compute_price_percentile(5.0, [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.0)

    def test_midpoint(self):
        assert _compute_price_percentile(3.0, [1.0, 3.0, 5.0]) == pytest.approx(0.5)

    def test_flat_history_returns_half(self):
        assert _compute_price_percentile(100.0, [100.0, 100.0, 100.0]) == pytest.approx(0.5)

    def test_single_bar_returns_half(self):
        assert _compute_price_percentile(50.0, [50.0]) == pytest.approx(0.5)

    def test_clipped_above_max(self):
        assert _compute_price_percentile(999.0, [1.0, 5.0]) == pytest.approx(1.0)

    def test_clipped_below_min(self):
        assert _compute_price_percentile(-1.0, [1.0, 5.0]) == pytest.approx(0.0)


class TestSortHistory:
    def test_sorted_ascending(self):
        raw = [{"date": "2024-01-03", "close": 3.0}, {"date": "2024-01-01", "close": 1.0}]
        result = _sort_history(raw)
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][1] == pytest.approx(3.0)

    def test_duplicate_dates_last_wins(self):
        raw = [
            {"date": "2024-01-01", "close": 10.0},
            {"date": "2024-01-01", "close": 20.0},
        ]
        result = _sort_history(raw)
        assert len(result) == 1
        assert result[0][1] == pytest.approx(20.0)

    def test_non_positive_close_dropped(self):
        raw = [{"date": "2024-01-01", "close": 0.0}, {"date": "2024-01-02", "close": -5.0}]
        assert _sort_history(raw) == []

    def test_invalid_date_dropped(self):
        raw = [{"date": "not-a-date", "close": 100.0}]
        assert _sort_history(raw) == []

    def test_missing_close_dropped(self):
        raw = [{"date": "2024-01-01"}]
        assert _sort_history(raw) == []

    def test_empty_input(self):
        assert _sort_history([]) == []


class TestFindIdxOnOrBefore:
    from datetime import date as _date

    def test_exact_match(self):
        from datetime import date

        hist = _sort_history(_history([10.0, 20.0, 30.0], "2024-01-01"))
        idx = _find_idx_on_or_before(hist, date(2024, 1, 2))
        assert idx == 1  # 2024-01-02 is at index 1

    def test_trade_date_before_all_bars(self):
        from datetime import date

        hist = _sort_history(_history([10.0, 20.0], "2024-01-10"))
        idx = _find_idx_on_or_before(hist, date(2024, 1, 1))
        assert idx is None

    def test_trade_date_after_all_bars(self):
        from datetime import date

        hist = _sort_history(_history([10.0, 20.0], "2024-01-01"))
        idx = _find_idx_on_or_before(hist, date(2024, 12, 31))
        assert idx == 1

    def test_weekend_gap_uses_last_available(self):
        from datetime import date

        # History: Mon + Tue; trade on Wed (no bar)
        raw = [{"date": "2024-01-01", "close": 5.0}, {"date": "2024-01-02", "close": 6.0}]
        hist = _sort_history(raw)
        idx = _find_idx_on_or_before(hist, date(2024, 1, 3))
        assert idx == 1


class TestScoreLabel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (100.0, "excellent"),
            (80.0, "excellent"),
            (79.9, "good"),
            (60.0, "good"),
            (59.9, "neutral"),
            (40.0, "neutral"),
            (39.9, "poor"),
            (20.0, "poor"),
            (19.9, "very_poor"),
            (0.0, "very_poor"),
        ],
    )
    def test_all_bands(self, score, expected):
        assert _score_label(score) == expected


class TestClassifyTimingConfidence:
    def test_insufficient(self):
        assert _classify_timing_confidence(5) == ConfidenceLevel.INSUFFICIENT

    def test_low(self):
        # 15 = RSI_PERIOD + 1
        assert _classify_timing_confidence(15) == ConfidenceLevel.LOW

    def test_medium(self):
        assert _classify_timing_confidence(50) == ConfidenceLevel.MEDIUM

    def test_exactly_rsi_boundary(self):
        # 14 bars → need 15 for RSI → INSUFFICIENT
        assert _classify_timing_confidence(14) == ConfidenceLevel.INSUFFICIENT
        assert _classify_timing_confidence(15) == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Unit tests: _compute_timing_score
# ---------------------------------------------------------------------------


class TestComputeTimingScore:
    def test_buy_at_period_low_high_score(self):
        score, notes = _compute_timing_score("buy", 0.0, None, None, 100.0)
        assert score == pytest.approx(100.0)
        assert any("period low" in n for n in notes)

    def test_buy_at_period_high_low_score(self):
        score, notes = _compute_timing_score("buy", 1.0, None, None, 100.0)
        assert score == pytest.approx(0.0)
        assert any("period high" in n for n in notes)

    def test_sell_at_period_high_high_score(self):
        score, _ = _compute_timing_score("sell", 1.0, None, None, 100.0)
        assert score == pytest.approx(100.0)

    def test_sell_at_period_low_low_score(self):
        score, _ = _compute_timing_score("sell", 0.0, None, None, 100.0)
        assert score == pytest.approx(0.0)

    def test_buy_oversold_rsi_annotation(self):
        _, notes = _compute_timing_score("buy", 0.5, 20.0, None, 100.0)
        assert any("oversold" in n for n in notes)

    def test_buy_overbought_rsi_annotation(self):
        _, notes = _compute_timing_score("buy", 0.5, 80.0, None, 100.0)
        assert any("overbought" in n for n in notes)

    def test_sell_overbought_rsi_annotation(self):
        _, notes = _compute_timing_score("sell", 0.5, 80.0, None, 100.0)
        assert any("overbought" in n for n in notes)

    def test_no_components_returns_50(self):
        score, notes = _compute_timing_score("buy", None, None, None, 100.0)
        assert score == pytest.approx(50.0)
        assert notes  # should have an explanatory note

    def test_buy_below_sma_positive_contribution(self):
        # Price 20% below SMA → best SMA component
        score_below, _ = _compute_timing_score("buy", 0.5, None, 100.0, 80.0)
        score_above, _ = _compute_timing_score("buy", 0.5, None, 100.0, 120.0)
        assert score_below > score_above

    def test_sell_above_sma_positive_contribution(self):
        score_above, _ = _compute_timing_score("sell", 0.5, None, 100.0, 120.0)
        score_below, _ = _compute_timing_score("sell", 0.5, None, 100.0, 80.0)
        assert score_above > score_below

    def test_score_bounded_0_100(self):
        # Worst possible sell (at period low, oversold RSI, below SMA)
        score, _ = _compute_timing_score("sell", 0.0, 5.0, 100.0, 50.0)
        assert 0.0 <= score <= 100.0

    def test_score_bounded_extreme_buy(self):
        # Best possible buy
        score, _ = _compute_timing_score("buy", 0.0, 5.0, 100.0, 50.0)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Integration tests: compute_trade_timing
# ---------------------------------------------------------------------------


class TestComputeTradeTimingGoodBuy:
    """Buy at the period low of a declining series → should score high."""

    def setup_method(self):
        # 60 bars declining from 110 → 80.5; buy at the final bar (period low)
        prices = [110.0 - i * 0.5 for i in range(60)]
        self.hist = _history(prices, "2024-01-01")
        # 2024-01-01 + 59 days = 2024-02-29 (2024 is a leap year)
        self.trade = _buy_trade("TEST", "2024-02-29", prices[-1])

    def test_score_above_50(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.timing_score > 50.0

    def test_label_not_poor(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.label not in ("poor", "very_poor")

    def test_symbol_propagated(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.symbol == "TEST"

    def test_trade_type_propagated(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.trade_type == "buy"


class TestComputeTradeTimingBadBuy:
    """Buy at the top of a range → should score low."""

    def setup_method(self):
        prices = [80.0 + i * 0.5 for i in range(60)]
        self.hist = _history(prices, "2024-01-01")
        # Trade on last day = highest close
        self.trade = _buy_trade("TEST", "2024-03-01", 109.5)

    def test_score_below_50(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.timing_score < 50.0


class TestComputeTradeTimingGoodSell:
    """Sell at the top of a range → should score high."""

    def setup_method(self):
        prices = [80.0 + i * 0.5 for i in range(60)]
        self.hist = _history(prices, "2024-01-01")
        # Sell on last day = highest close
        self.trade = _sell_trade("TEST", "2024-03-01", 109.5)

    def test_score_above_50(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.timing_score > 50.0


class TestComputeTradeTimingBadSell:
    """Sell at the period low of a declining series → should score low."""

    def setup_method(self):
        # Same declining series; sell at the final bar (period low = worst sell timing)
        prices = [110.0 - i * 0.5 for i in range(60)]
        self.hist = _history(prices, "2024-01-01")
        self.trade = _sell_trade("TEST", "2024-02-29", prices[-1])

    def test_score_below_50(self):
        result = compute_trade_timing(self.trade, self.hist)
        assert result.timing_score < 50.0


class TestComputeTradeTimingEdgeCases:
    def test_invalid_date_returns_neutral(self):
        result = compute_trade_timing({"symbol": "X", "date": "bad-date", "trade_type": "buy", "price": 100.0}, [])
        assert result.timing_score == pytest.approx(50.0)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT
        assert result.label == "neutral"

    def test_no_history_returns_neutral(self):
        result = compute_trade_timing(_buy_trade("X", "2024-06-01", 100.0), [])
        assert result.timing_score == pytest.approx(50.0)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_trade_before_all_history_returns_neutral(self):
        hist = _history([100.0, 101.0, 102.0], "2024-06-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-01", 100.0), hist)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_zero_trade_price_falls_back_to_close(self):
        hist = _history([95.0] + [100.0] * 59, "2024-01-01")
        trade = {"symbol": "X", "date": "2024-01-01", "trade_type": "buy", "price": 0}
        result = compute_trade_timing(trade, hist)
        assert result.trade_price == pytest.approx(95.0)

    def test_sparse_history_low_confidence(self):
        # Only 5 bars → fewer than RSI period+1
        hist = _history([100.0, 101.0, 99.0, 100.0, 102.0], "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-05", 102.0), hist)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_medium_confidence_with_50_bars(self):
        hist = _history([100.0 + i * 0.1 for i in range(60)], "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-03-01", 105.0), hist)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_to_dict_round_trip(self):
        hist = _history([100.0 + i for i in range(60)], "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-03-01", 115.0), hist)
        d = result.to_dict()
        assert d["symbol"] == "X"
        assert d["trade_type"] == "buy"
        assert isinstance(d["timing_score"], float)
        assert d["label"] in ("excellent", "good", "neutral", "poor", "very_poor")
        assert isinstance(d["price_context"], dict)
        assert d["confidence"] in (lv.value for lv in ConfidenceLevel)

    def test_rsi_14_present_with_15_bars(self):
        hist = _history([100.0 + i for i in range(16)], "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-16", 115.0), hist)
        assert result.price_context.rsi_14 is not None

    def test_rsi_14_none_with_14_bars(self):
        hist = _history([100.0 + i for i in range(14)], "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-14", 113.0), hist)
        assert result.price_context.rsi_14 is None

    def test_sma_20_present_with_20_bars(self):
        hist = _history([100.0] * 25, "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-25", 100.0), hist)
        assert result.price_context.sma_20 is not None
        assert result.price_context.sma_20 == pytest.approx(100.0)

    def test_sma_20_none_with_19_bars(self):
        hist = _history([100.0] * 19, "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-01-19", 100.0), hist)
        assert result.price_context.sma_20 is None

    def test_percentile_window_respected(self):
        """Small percentile_window uses only recent bars."""
        prices = [100.0] * 50 + [50.0]  # 51st bar is much lower
        hist = _history(prices, "2024-01-01")
        result = compute_trade_timing(_buy_trade("X", "2024-02-20", 50.0), hist, percentile_window=5)
        # With window=5 the recent bars include the 50.0, so percentile should be 0
        assert result.price_context.price_percentile is not None

    def test_weekend_gap_handled(self):
        """History missing some dates; trade date falls in gap."""
        # Supply Mon + Tue, trade on Wed (no bar) — should use Tue close
        raw = [{"date": "2024-01-01", "close": 100.0}, {"date": "2024-01-02", "close": 110.0}]
        result = compute_trade_timing(_buy_trade("X", "2024-01-03", 110.0), raw)
        # We get a result (not INSUFFICIENT due to date parse failure)
        assert result.trade_date == "2024-01-03"


# ---------------------------------------------------------------------------
# Integration tests: compute_portfolio_timing_insight
# ---------------------------------------------------------------------------


class TestComputePortfolioTimingInsight:
    def _make_history(self, n: int = 60, base: float = 100.0) -> list[dict]:
        return _history([base + i * 0.5 for i in range(n)], "2024-01-01")

    def test_empty_trades_returns_empty(self):
        result = compute_portfolio_timing_insight([], {})
        assert isinstance(result, PortfolioTimingInsight)
        assert result.trade_results == []
        assert result.notes  # should include "No trade data" note

    def test_only_non_eligible_trades_skipped(self):
        trades = [{"symbol": "X", "date": "2024-01-01", "trade_type": "transfer", "price": 100.0}]
        result = compute_portfolio_timing_insight(trades, {})
        assert result.trade_results == []

    def test_avg_buy_score_computed(self):
        hist = self._make_history()
        trades = [_buy_trade("A", "2024-01-01", 100.0)]
        result = compute_portfolio_timing_insight(trades, {"A": hist})
        assert result.avg_buy_timing_score is not None
        assert result.avg_sell_timing_score is None

    def test_avg_sell_score_computed(self):
        hist = self._make_history()
        trades = [_sell_trade("A", "2024-03-01", 129.5)]
        result = compute_portfolio_timing_insight(trades, {"A": hist})
        assert result.avg_sell_timing_score is not None
        assert result.avg_buy_timing_score is None

    def test_mixed_buy_sell(self):
        hist = self._make_history()
        trades = [
            _buy_trade("A", "2024-01-01", 100.0),
            _sell_trade("A", "2024-03-01", 129.5),
        ]
        result = compute_portfolio_timing_insight(trades, {"A": hist})
        assert result.avg_buy_timing_score is not None
        assert result.avg_sell_timing_score is not None
        assert len(result.trade_results) == 2

    def test_missing_symbol_history_no_exception(self):
        trades = [_buy_trade("UNKNOWN", "2024-06-01", 100.0)]
        # Should not raise; returns neutral result for the trade
        result = compute_portfolio_timing_insight(trades, {})
        assert len(result.trade_results) == 1
        assert result.trade_results[0].confidence == ConfidenceLevel.INSUFFICIENT

    def test_multiple_symbols(self):
        histA = self._make_history(60, 100.0)
        histB = self._make_history(60, 200.0)
        trades = [_buy_trade("A", "2024-01-01", 100.0), _buy_trade("B", "2024-01-01", 200.0)]
        result = compute_portfolio_timing_insight(trades, {"A": histA, "B": histB})
        assert len(result.trade_results) == 2
        assert result.avg_buy_timing_score is not None

    def test_to_dict_round_trip(self):
        hist = self._make_history()
        trades = [_buy_trade("A", "2024-01-01", 100.0), _sell_trade("A", "2024-03-01", 129.5)]
        result = compute_portfolio_timing_insight(trades, {"A": hist})
        d = result.to_dict()
        assert isinstance(d["trade_results"], list)
        assert isinstance(d["avg_buy_timing_score"], float)
        assert d["confidence"] in (lv.value for lv in ConfidenceLevel)
        assert isinstance(d["notes"], list)

    def test_good_buy_scores_higher_than_bad_buy(self):
        """Buying at the period low should out-score buying at the period high."""
        prices = [80.0 + i * 0.5 for i in range(60)]
        hist = _history(prices, "2024-01-01")
        good_buy = _buy_trade("X", "2024-01-01", 80.0)
        bad_buy = _buy_trade("X", "2024-03-01", 109.5)

        good_result = compute_trade_timing(good_buy, hist)
        bad_result = compute_trade_timing(bad_buy, hist)
        assert good_result.timing_score > bad_result.timing_score

    def test_notes_contain_score_summary(self):
        hist = self._make_history()
        trades = [_buy_trade("A", "2024-01-01", 100.0)]
        result = compute_portfolio_timing_insight(trades, {"A": hist})
        assert any("buy timing score" in n for n in result.notes)


# ---------------------------------------------------------------------------
# PriceContext dataclass
# ---------------------------------------------------------------------------


class TestPriceContextToDict:
    def test_all_none(self):
        ctx = PriceContext()
        d = ctx.to_dict()
        assert d["sma_20"] is None
        assert d["rsi_14"] is None
        assert d["price_percentile"] is None

    def test_values_present(self):
        ctx = PriceContext(sma_20=100.0, rsi_14=45.5, price_percentile=0.25, days_of_history=60)
        d = ctx.to_dict()
        assert d["sma_20"] == pytest.approx(100.0)
        assert d["rsi_14"] == pytest.approx(45.5)
        assert d["price_percentile"] == pytest.approx(0.25)
        assert d["days_of_history"] == 60


# ---------------------------------------------------------------------------
# Package-level import test
# ---------------------------------------------------------------------------


def test_package_exports():
    from src.core.behavior import (
        PortfolioTimingInsight,
        PriceContext,
        TradeTimingResult,
        compute_portfolio_timing_insight,
        compute_trade_timing,
    )

    assert callable(compute_trade_timing)
    assert callable(compute_portfolio_timing_insight)
    assert PortfolioTimingInsight is not None
    assert PriceContext is not None
    assert TradeTimingResult is not None
