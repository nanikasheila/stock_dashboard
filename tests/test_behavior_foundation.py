"""Tests for the insight-domain-foundation: src/core/behavior and components/dl_behavior.

Coverage:
    - ConfidenceLevel enum values and ordering
    - TradeStats: properties (total_trades, win_rate)
    - PortfolioTradeStats: properties (total_trades, overall_win_rate), empty()
    - BehaviorInsight: to_dict(), empty()
    - compute_trade_stats_by_symbol: buy-only, sell FIFO PnL, win/loss, hold days, cash excluded
    - compute_portfolio_trade_stats: aggregation, avg_hold (weighted), confidence levels
    - compute_style_metrics: frequency, holding_style, concentration, empty-trade case
    - min_confidence helper
    - _trade_amount_jpy: all priority branches
    - load_behavior_insight (via dl_behavior): no-trades empty, smoke test with mock data
    - data_loader facade re-exports
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.behavior import (
    BehaviorInsight,
    ConfidenceLevel,
    PortfolioTradeStats,
    SellRecord,
    StyleMetrics,
    TradeStats,
    compute_portfolio_trade_stats,
    compute_style_metrics,
    compute_trade_stats_by_symbol,
    min_confidence,
)
from src.core.behavior.trade_stats import (
    _CONFIDENCE_ORDER,
    _classify_confidence,
    _trade_amount_jpy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FX = {"USD": 150.0, "JPY": 1.0}


def _buy(symbol: str, date: str, shares: float, price: float, currency: str = "JPY") -> dict:
    """Minimal buy trade dict."""
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
    """Minimal sell trade dict."""
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
# ConfidenceLevel
# ---------------------------------------------------------------------------


class TestConfidenceLevel:
    def test_values_are_strings(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.INSUFFICIENT == "insufficient"

    def test_ordering_in_confidence_order_dict(self):
        assert _CONFIDENCE_ORDER[ConfidenceLevel.INSUFFICIENT] < _CONFIDENCE_ORDER[ConfidenceLevel.LOW]
        assert _CONFIDENCE_ORDER[ConfidenceLevel.LOW] < _CONFIDENCE_ORDER[ConfidenceLevel.MEDIUM]
        assert _CONFIDENCE_ORDER[ConfidenceLevel.MEDIUM] < _CONFIDENCE_ORDER[ConfidenceLevel.HIGH]


class TestMinConfidence:
    def test_returns_lower(self):
        assert min_confidence(ConfidenceLevel.HIGH, ConfidenceLevel.LOW) == ConfidenceLevel.LOW
        assert min_confidence(ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH) == ConfidenceLevel.MEDIUM

    def test_same_level_returns_same(self):
        assert min_confidence(ConfidenceLevel.MEDIUM, ConfidenceLevel.MEDIUM) == ConfidenceLevel.MEDIUM

    def test_insufficient_always_wins(self):
        for lvl in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW):
            assert min_confidence(ConfidenceLevel.INSUFFICIENT, lvl) == ConfidenceLevel.INSUFFICIENT


class TestClassifyConfidence:
    def test_zero_sells_insufficient(self):
        assert _classify_confidence(0) == ConfidenceLevel.INSUFFICIENT

    def test_one_sell_low(self):
        assert _classify_confidence(1) == ConfidenceLevel.LOW

    def test_four_sells_low(self):
        assert _classify_confidence(4) == ConfidenceLevel.LOW

    def test_five_sells_medium(self):
        assert _classify_confidence(5) == ConfidenceLevel.MEDIUM

    def test_nineteen_sells_medium(self):
        assert _classify_confidence(19) == ConfidenceLevel.MEDIUM

    def test_twenty_sells_high(self):
        assert _classify_confidence(20) == ConfidenceLevel.HIGH

    def test_many_sells_high(self):
        assert _classify_confidence(100) == ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# _trade_amount_jpy
# ---------------------------------------------------------------------------


class TestTradeAmountJpy:
    def test_settlement_jpy_used_directly(self):
        trade = {"settlement_jpy": 300000.0, "settlement_usd": 0.0, "fx_rate": 150.0}
        assert _trade_amount_jpy(trade, FX) == 300000.0

    def test_usd_settlement_times_fx_rate(self):
        trade = {"settlement_jpy": 0.0, "settlement_usd": 1000.0, "fx_rate": 150.0}
        assert _trade_amount_jpy(trade, FX) == 150000.0

    def test_mixed_settlement(self):
        trade = {"settlement_jpy": 100000.0, "settlement_usd": 500.0, "fx_rate": 150.0}
        assert _trade_amount_jpy(trade, FX) == 175000.0

    def test_fallback_shares_times_price_times_fx(self):
        trade = {
            "settlement_jpy": 0.0,
            "settlement_usd": 0.0,
            "fx_rate": 150.0,
            "shares": 10,
            "price": 200.0,
            "currency": "USD",
        }
        assert _trade_amount_jpy(trade, FX) == pytest.approx(300000.0)

    def test_fallback_uses_portfolio_fx_rate(self):
        trade = {
            "settlement_jpy": 0.0,
            "settlement_usd": 0.0,
            "fx_rate": 0.0,
            "shares": 100,
            "price": 3000.0,
            "currency": "JPY",
        }
        assert _trade_amount_jpy(trade, FX) == pytest.approx(300000.0)


# ---------------------------------------------------------------------------
# TradeStats
# ---------------------------------------------------------------------------


class TestTradeStats:
    def test_total_trades_property(self):
        ts = TradeStats(symbol="AAPL", buy_count=3, sell_count=1)
        assert ts.total_trades == 4

    def test_win_rate_none_when_no_sells(self):
        ts = TradeStats(symbol="AAPL", buy_count=3, sell_count=0)
        assert ts.win_rate is None

    def test_win_rate_calculation(self):
        ts = TradeStats(symbol="AAPL", sell_count=4, win_count=3, loss_count=1)
        assert ts.win_rate == pytest.approx(0.75)

    def test_to_dict_contains_required_keys(self):
        ts = TradeStats(symbol="TEST", buy_count=2, sell_count=1, win_count=1, loss_count=0)
        d = ts.to_dict()
        for key in ("symbol", "buy_count", "sell_count", "total_trades", "win_rate", "confidence"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_confidence_is_string(self):
        ts = TradeStats(symbol="X", confidence=ConfidenceLevel.HIGH)
        assert ts.to_dict()["confidence"] == "high"


# ---------------------------------------------------------------------------
# PortfolioTradeStats
# ---------------------------------------------------------------------------


class TestPortfolioTradeStats:
    def test_empty_returns_defaults(self):
        ps = PortfolioTradeStats.empty()
        assert ps.symbols_traded == []
        assert ps.total_trades == 0
        assert ps.overall_win_rate is None
        assert ps.confidence == ConfidenceLevel.INSUFFICIENT

    def test_overall_win_rate_none_when_no_sells(self):
        ps = PortfolioTradeStats(overall_win_count=0, overall_loss_count=0)
        assert ps.overall_win_rate is None

    def test_overall_win_rate_calculated(self):
        ps = PortfolioTradeStats(overall_win_count=7, overall_loss_count=3)
        assert ps.overall_win_rate == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# BehaviorInsight
# ---------------------------------------------------------------------------


class TestBehaviorInsight:
    def test_empty_has_notes(self):
        bi = BehaviorInsight.empty()
        assert len(bi.notes) > 0
        assert bi.confidence == ConfidenceLevel.INSUFFICIENT

    def test_to_dict_structure(self):
        bi = BehaviorInsight.empty()
        d = bi.to_dict()
        assert "trade_stats" in d
        assert "style_metrics" in d
        assert "confidence" in d
        assert "notes" in d

    def test_to_dict_confidence_is_string(self):
        bi = BehaviorInsight(confidence=ConfidenceLevel.MEDIUM)
        assert bi.to_dict()["confidence"] == "medium"


# ---------------------------------------------------------------------------
# compute_trade_stats_by_symbol
# ---------------------------------------------------------------------------


class TestComputeTradeStatsBySymbol:
    def test_buy_only_no_sell(self):
        trades = [_buy("7203.T", "2024-01-10", 100, 2800.0)]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert "7203.T" in result
        ts = result["7203.T"]
        assert ts.buy_count == 1
        assert ts.sell_count == 0
        assert ts.realized_pnl_jpy == 0.0
        assert ts.confidence == ConfidenceLevel.INSUFFICIENT

    def test_buy_then_sell_profit(self):
        trades = [
            _buy("AAPL", "2024-01-01", 10, 150.0, "USD"),
            _sell("AAPL", "2024-06-01", 10, 200.0, "USD"),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        ts = result["AAPL"]
        assert ts.sell_count == 1
        assert ts.win_count == 1
        assert ts.loss_count == 0
        assert ts.realized_pnl_jpy > 0, "Selling at higher price should yield positive PnL"

    def test_buy_then_sell_loss(self):
        trades = [
            _buy("AAPL", "2024-01-01", 10, 200.0, "USD"),
            _sell("AAPL", "2024-06-01", 10, 150.0, "USD"),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        ts = result["AAPL"]
        assert ts.win_count == 0
        assert ts.loss_count == 1
        assert ts.realized_pnl_jpy < 0

    def test_hold_days_computed(self):
        trades = [
            _buy("7203.T", "2024-01-01", 100, 2800.0),
            _sell("7203.T", "2024-04-10", 100, 3000.0),  # 100 days later
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        ts = result["7203.T"]
        assert ts.avg_hold_days is not None
        assert ts.avg_hold_days == pytest.approx(100.0, abs=1.0)

    def test_cash_positions_excluded(self):
        trades = [
            _buy("JPY.CASH", "2024-01-01", 1000000, 1.0),
            _buy("7203.T", "2024-01-01", 100, 2800.0),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert "JPY.CASH" not in result
        assert "7203.T" in result

    def test_multiple_symbols_independent(self):
        trades = [
            _buy("AAPL", "2024-01-01", 5, 150.0, "USD"),
            _buy("GOOG", "2024-01-01", 2, 100.0, "USD"),
            _sell("AAPL", "2024-06-01", 5, 180.0, "USD"),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert result["AAPL"].sell_count == 1
        assert result["GOOG"].sell_count == 0
        assert result["GOOG"].win_count == 0

    def test_fifo_partial_sell(self):
        """Selling part of a position uses FIFO from first lot."""
        trades = [
            _buy("TEST", "2024-01-01", 100, 100.0),
            _buy("TEST", "2024-03-01", 100, 200.0),
            _sell("TEST", "2024-06-01", 100, 150.0),  # sells first lot only
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        ts = result["TEST"]
        # First lot cost 100 JPY/share, sold at 150 → profit 5000 JPY
        assert ts.realized_pnl_jpy == pytest.approx(5000.0, abs=1.0)
        assert ts.sell_count == 1
        assert ts.win_count == 1


# ---------------------------------------------------------------------------
# compute_portfolio_trade_stats
# ---------------------------------------------------------------------------


class TestComputePortfolioTradeStats:
    def test_empty_trades_returns_empty(self):
        result = compute_portfolio_trade_stats([], FX)
        assert result.total_trades == 0
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_aggregates_across_symbols(self):
        trades = [
            _buy("AAPL", "2024-01-01", 5, 150.0, "USD"),
            _buy("GOOG", "2024-01-01", 2, 100.0, "USD"),
            _sell("AAPL", "2024-06-01", 5, 180.0, "USD"),
        ]
        result = compute_portfolio_trade_stats(trades, FX)
        assert result.total_buy_count == 2
        assert result.total_sell_count == 1
        assert "AAPL" in result.symbols_traded
        assert "GOOG" in result.symbols_traded

    def test_confidence_medium_at_five_sells(self):
        # 5 sells = MEDIUM
        trades = []
        for i in range(5):
            trades.append(_buy("SYM", f"2024-0{i + 1}-01", 100, 100.0))
            trades.append(_sell("SYM", f"2024-0{i + 1}-15", 100, 110.0))
        result = compute_portfolio_trade_stats(trades, FX)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_realized_pnl_sums_across_symbols(self):
        trades = [
            _buy("A", "2024-01-01", 10, 100.0),
            _sell("A", "2024-06-01", 10, 120.0),  # +200 JPY (10 * 20 = 200)
            _buy("B", "2024-01-01", 10, 100.0),
            _sell("B", "2024-06-01", 10, 90.0),  # -100 JPY (10 * -10 = -100)
        ]
        result = compute_portfolio_trade_stats(trades, FX)
        # Net = 200 - 100 = 100 JPY
        assert result.total_realized_pnl_jpy == pytest.approx(100.0, abs=1.0)

    def test_avg_hold_days_weighted(self):
        trades = [
            _buy("FAST", "2024-01-01", 10, 100.0),
            _sell("FAST", "2024-01-11", 10, 120.0),  # 10 days
            _buy("SLOW", "2024-01-01", 10, 100.0),
            _sell("SLOW", "2024-04-10", 10, 120.0),  # 100 days
        ]
        result = compute_portfolio_trade_stats(trades, FX)
        # Both have 1 sell; weighted avg = (10 + 100) / 2 = 55
        assert result.avg_hold_days is not None
        assert result.avg_hold_days == pytest.approx(55.0, abs=2.0)


# ---------------------------------------------------------------------------
# compute_style_metrics
# ---------------------------------------------------------------------------


class TestComputeStyleMetrics:
    def test_no_trades_returns_insufficient(self):
        portfolio = PortfolioTradeStats.empty()
        style = compute_style_metrics(portfolio, [], FX)
        assert style.confidence == ConfidenceLevel.INSUFFICIENT
        assert style.trade_frequency == "unknown"

    def test_passive_frequency_few_trades(self):
        trades = [
            _buy("X", "2024-01-01", 10, 100.0),
            _buy("X", "2024-06-01", 10, 110.0),
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.trade_frequency == "passive"

    def test_active_frequency_many_trades(self):
        # Generate 2+ trades per month for 6 months = 12+ trades
        trades = []
        for month in range(1, 7):
            for day in (1, 15):
                m = f"{month:02d}"
                trades.append(_buy("FREQ", f"2024-{m}-{day:02d}", 10, 100.0))
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.trade_frequency == "active"

    def test_holding_style_long_term(self):
        trades = [
            _buy("LT", "2023-01-01", 100, 1000.0),
            _sell("LT", "2024-01-01", 100, 1200.0),  # ~365 days
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.holding_style == "long_term"

    def test_holding_style_short_term(self):
        trades = [
            _buy("ST", "2024-01-01", 100, 1000.0),
            _sell("ST", "2024-01-15", 100, 1100.0),  # 14 days
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.holding_style == "short_term"

    def test_holding_style_unknown_when_no_sells(self):
        trades = [_buy("X", "2024-01-01", 100, 1000.0)]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.holding_style == "unknown"

    def test_concentration_score_single_symbol(self):
        trades = [_buy("ONLY", "2024-01-01", 100, 1000.0)]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.concentration_score == pytest.approx(1.0)

    def test_concentration_score_uniform_two_symbols(self):
        trades = [
            _buy("A", "2024-01-01", 100, 1000.0),
            _buy("B", "2024-01-01", 100, 1000.0),
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        # Equal weight → HHI = 0.5*0.5 + 0.5*0.5 = 0.5
        assert style.concentration_score == pytest.approx(0.5, abs=0.01)

    def test_avg_position_size_computed(self):
        trades = [
            _buy("X", "2024-01-01", 100, 2000.0),  # 200,000 JPY
            _buy("Y", "2024-02-01", 100, 1000.0),  # 100,000 JPY
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        assert style.avg_position_size_jpy == pytest.approx(150000.0, abs=1.0)

    def test_notes_populated_for_sparse_data(self):
        trades = [_buy("X", "2024-01-01", 10, 100.0)]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        style = compute_style_metrics(portfolio, trades, FX)
        # With < 5 trades, notes should mention sparse data
        combined_notes = " ".join(style.notes)
        assert "indicative" in combined_notes or "sparse" in combined_notes or "1 trade" in combined_notes


# ---------------------------------------------------------------------------
# load_behavior_insight (via components.dl_behavior)
# ---------------------------------------------------------------------------


class TestLoadBehaviorInsight:
    def test_empty_when_no_trades(self, tmp_path):
        """load_behavior_insight returns BehaviorInsight.empty() when no trade files exist."""
        from components.dl_behavior import load_behavior_insight

        result = load_behavior_insight(base_dir=str(tmp_path / "history"))
        assert isinstance(result, BehaviorInsight)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT
        assert len(result.notes) > 0

    def test_returns_behavior_insight_type(self, tmp_path):
        """load_behavior_insight always returns a BehaviorInsight, never raises."""
        from components.dl_behavior import load_behavior_insight

        result = load_behavior_insight(base_dir=str(tmp_path / "empty"))
        assert isinstance(result, BehaviorInsight)

    def test_smoke_with_mocked_trades(self, tmp_path):
        """With mocked trade data, insight is computed without error."""
        from components.dl_behavior import load_behavior_insight

        sample_trades = [
            _buy("7203.T", "2024-01-10", 100, 2800.0),
            _buy("AAPL", "2024-02-01", 5, 180.0, "USD"),
            _sell("7203.T", "2024-07-10", 100, 3200.0),
        ]

        with (
            patch("components.dl_behavior.get_fx_rates", return_value=FX),
            patch(
                "components.dl_behavior.load_behavior_insight.__wrapped__"
                if hasattr(load_behavior_insight, "__wrapped__")
                else "components.dl_holdings._build_holdings_timeline",
                return_value=sample_trades,
            ),
        ):
            # Patch _build_holdings_timeline directly since it's imported inside the function
            import components.dl_holdings as dlh

            original = dlh._build_holdings_timeline
            dlh._build_holdings_timeline = lambda *a, **kw: sample_trades
            try:
                result = load_behavior_insight(
                    base_dir=str(tmp_path),
                    csv_path=str(tmp_path / "portfolio.csv"),
                )
            finally:
                dlh._build_holdings_timeline = original

        assert isinstance(result, BehaviorInsight)
        assert result.trade_stats.total_buy_count == 2
        assert result.trade_stats.total_sell_count == 1
        assert result.holding_period.total_closed == 1
        assert result.win_loss.win_count == 1
        assert result.trade_stats.overall_win_rate == pytest.approx(1.0)

    def test_uses_yahoo_client_when_loading_fx_rates(self, tmp_path):
        """FX rates are requested with the existing Yahoo client provider."""
        from components.dl_behavior import load_behavior_insight

        sample_trades = [_buy("7203.T", "2024-01-10", 100, 2800.0)]
        provider = object()

        with (
            patch("components.dl_behavior.get_fx_rates", return_value=FX) as mock_get_fx_rates,
            patch("components.dl_behavior.yahoo_client", new=provider),
        ):
            import components.dl_holdings as dlh

            original = dlh._build_holdings_timeline
            dlh._build_holdings_timeline = lambda *a, **kw: sample_trades
            try:
                load_behavior_insight(
                    base_dir=str(tmp_path),
                    csv_path=str(tmp_path / "portfolio.csv"),
                )
            finally:
                dlh._build_holdings_timeline = original

        mock_get_fx_rates.assert_called_once()
        assert mock_get_fx_rates.call_args.args == (provider,)


# ---------------------------------------------------------------------------
# data_loader facade re-exports
# ---------------------------------------------------------------------------


class TestDataLoaderFacade:
    """Verify that all new symbols are accessible via components.data_loader."""

    def test_behavior_insight_importable(self):
        from components.data_loader import BehaviorInsight as BI

        assert BI is BehaviorInsight

    def test_confidence_level_importable(self):
        from components.data_loader import ConfidenceLevel as CL

        assert CL is ConfidenceLevel

    def test_portfolio_trade_stats_importable(self):
        from components.data_loader import PortfolioTradeStats as PTS

        assert PTS is PortfolioTradeStats

    def test_style_metrics_importable(self):
        from components.data_loader import StyleMetrics as SM

        assert SM is StyleMetrics

    def test_trade_stats_importable(self):
        from components.data_loader import TradeStats as TS

        assert TS is TradeStats

    def test_load_behavior_insight_importable(self):
        from components.data_loader import load_behavior_insight as lbi

        assert callable(lbi)

    def test_existing_exports_still_work(self):
        """Ensure we did not break any pre-existing data_loader exports."""
        from components.data_loader import (
            build_portfolio_history,
            compute_risk_metrics,
            get_current_snapshot,
            get_monthly_summary,
            run_dashboard_health_check,
        )

        assert callable(build_portfolio_history)
        assert callable(compute_risk_metrics)
        assert callable(get_current_snapshot)
        assert callable(get_monthly_summary)
        assert callable(run_dashboard_health_check)

    def test_sell_record_importable(self):
        from components.data_loader import SellRecord as SR

        assert SR is SellRecord


# ---------------------------------------------------------------------------
# SellRecord — TradeStats and PortfolioTradeStats integration
# ---------------------------------------------------------------------------


class TestSellRecordIntegration:
    """Verify SellRecord is collected correctly during FIFO matching."""

    def test_sell_record_in_trade_stats(self) -> None:
        """TradeStats.sell_records is populated after a sell transaction."""
        trades = [
            _buy("A", "2024-01-01", 100, 1000.0),
            _sell("A", "2024-06-01", 100, 1200.0),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert "A" in result
        assert len(result["A"].sell_records) == 1
        assert isinstance(result["A"].sell_records[0], SellRecord)

    def test_no_sell_records_when_buy_only(self) -> None:
        """When no sells occur, sell_records must be empty."""
        trades = [_buy("B", "2024-01-01", 50, 500.0)]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert result["B"].sell_records == []

    def test_sell_record_fields_match_event(self) -> None:
        """SellRecord.pnl_jpy, sell_date, holding_days are populated from the sell event."""
        trades = [
            _buy("C", "2024-01-10", 100, 2000.0),
            _sell("C", "2024-07-10", 100, 2500.0),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        rec = result["C"].sell_records[0]
        assert rec.symbol == "C"
        assert rec.sell_date == "2024-07-10"
        assert rec.pnl_jpy == pytest.approx(50000.0)
        # 2024-01-10 → 2024-07-10 = 182 days
        assert rec.holding_days == 182

    def test_sell_records_pnl_sum_matches_realized_pnl(self) -> None:
        """Sum of SellRecord.pnl_jpy must equal TradeStats.realized_pnl_jpy."""
        trades = [
            _buy("D", "2024-01-01", 50, 1000.0),
            _sell("D", "2024-03-01", 20, 1200.0),
            _sell("D", "2024-06-01", 30, 900.0),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        ts = result["D"]
        pnl_sum = sum(r.pnl_jpy for r in ts.sell_records)
        assert pnl_sum == pytest.approx(ts.realized_pnl_jpy, abs=1.0)

    def test_multiple_sell_records_count(self) -> None:
        """Multiple sells produce multiple SellRecord entries."""
        trades = [
            _buy("E", "2024-01-01", 100, 1000.0),
            _sell("E", "2024-03-01", 30, 1100.0),
            _sell("E", "2024-05-01", 40, 1050.0),
            _sell("E", "2024-08-01", 30, 980.0),
        ]
        result = compute_trade_stats_by_symbol(trades, FX)
        assert len(result["E"].sell_records) == 3

    def test_portfolio_all_sell_records_aggregates_symbols(self) -> None:
        """PortfolioTradeStats.all_sell_records contains records from all symbols."""
        trades = [
            _buy("F", "2024-01-01", 100, 500.0),
            _sell("F", "2024-04-01", 100, 600.0),
            _buy("G", "2024-01-15", 50, 800.0),
            _sell("G", "2024-05-15", 50, 850.0),
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        assert len(portfolio.all_sell_records) == 2
        symbols_in_records = {r.symbol for r in portfolio.all_sell_records}
        assert symbols_in_records == {"F", "G"}

    def test_portfolio_all_sell_records_empty_on_no_sells(self) -> None:
        """PortfolioTradeStats.all_sell_records is empty when there are no sells."""
        trades = [
            _buy("H", "2024-01-01", 100, 1000.0),
            _buy("I", "2024-02-01", 50, 2000.0),
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        assert portfolio.all_sell_records == []

    def test_portfolio_all_sell_records_pnl_sum_consistency(self) -> None:
        """all_sell_records pnl sum equals portfolio total_realized_pnl_jpy."""
        trades = [
            _buy("J", "2024-01-01", 100, 1000.0),
            _sell("J", "2024-06-01", 100, 1300.0),
            _buy("K", "2024-02-01", 50, 2000.0),
            _sell("K", "2024-08-01", 50, 1800.0),
        ]
        portfolio = compute_portfolio_trade_stats(trades, FX)
        pnl_sum = sum(r.pnl_jpy for r in portfolio.all_sell_records)
        assert pnl_sum == pytest.approx(portfolio.total_realized_pnl_jpy, abs=1.0)

    def test_sell_record_holding_days_zero_when_no_dates(self) -> None:
        """holding_days falls back to 0 when lot date is unavailable."""
        # Trade dict without a parseable date on the buy side
        buy_no_date = {
            "category": "trade",
            "date": "",  # unparseable — lot gets date=None
            "symbol": "NODATES",
            "trade_type": "buy",
            "shares": 100,
            "price": 1000.0,
            "currency": "JPY",
            "fx_rate": 0.0,
            "settlement_jpy": 100000.0,
            "settlement_usd": 0.0,
        }
        sell_with_date = _sell("NODATES", "2024-06-01", 100, 1200.0)
        result = compute_trade_stats_by_symbol([buy_no_date, sell_with_date], FX)
        rec = result["NODATES"].sell_records[0]
        assert rec.holding_days == 0
