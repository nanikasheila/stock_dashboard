"""Tests for dashboard health check and sell alert functions.

Tests run_dashboard_health_check() and _compute_sell_alerts() in
the portfolio-dashboard data_loader module.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.data_loader import (
    _compute_sell_alerts,
    _is_nan,
    _stability_emoji,
    run_dashboard_health_check,
)
from src.core.health_check import (
    ALERT_CAUTION,
    ALERT_EXIT,
    ALERT_NONE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uptrend_hist(n: int = 300, base: float = 100.0) -> pd.DataFrame:
    """Steadily rising prices — clear uptrend."""
    prices = [base + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "Close": prices,
            "Volume": [1_000_000] * n,
        }
    )


def _make_downtrend_hist(n: int = 300, base: float = 200.0) -> pd.DataFrame:
    """Steadily falling prices — clear downtrend."""
    prices = [base - i * 0.3 for i in range(n)]
    return pd.DataFrame(
        {
            "Close": prices,
            "Volume": [1_000_000] * n,
        }
    )


def _make_flat_hist(n: int = 300, base: float = 100.0) -> pd.DataFrame:
    """Flat prices with tiny noise."""
    rng = np.random.RandomState(42)
    prices = [base + rng.uniform(-0.1, 0.1) for _ in range(n)]
    return pd.DataFrame(
        {
            "Close": prices,
            "Volume": [1_000_000] * n,
        }
    )


def _make_stock_detail(symbol="7203.T", **overrides):
    """Minimal stock detail."""
    detail = {
        "symbol": symbol,
        "quoteType": "EQUITY",
        "roe": 0.12,
        "eps_growth": 0.05,
        "dividend_yield": 0.025,
        "per": 15.0,
        "returnOnEquity": 0.12,
        "earningsGrowth": 0.05,
        "trailingEps": 100.0,
        "forwardEps": 105.0,
        "totalCashflow": {"values": [1000000]},
        "totalRevenue": {"values": [5000000, 4800000]},
        "netIncome": {"values": [500000, 480000]},
        "revenue_growth": 0.04,
        "grossMargins": 0.35,
        "operatingMargins": 0.12,
        "annualCashflow": None,
        "annualBalanceSheet": None,
    }
    detail.update(overrides)
    return detail


# ---------------------------------------------------------------------------
# Tests for _stability_emoji
# ---------------------------------------------------------------------------


class TestStabilityEmoji:
    def test_stable(self):
        assert _stability_emoji("stable") == "✅"

    def test_increasing(self):
        assert _stability_emoji("increasing") == "📈"

    def test_temporary(self):
        assert _stability_emoji("temporary") == "⚠️"

    def test_decreasing(self):
        assert _stability_emoji("decreasing") == "📉"

    def test_unknown(self):
        assert _stability_emoji("unknown") == ""

    def test_empty(self):
        assert _stability_emoji("") == ""


# ---------------------------------------------------------------------------
# Tests for _is_nan
# ---------------------------------------------------------------------------


class TestIsNan:
    def test_nan(self):
        assert _is_nan(float("nan")) is True

    def test_number(self):
        assert _is_nan(42.0) is False

    def test_none(self):
        assert _is_nan(None) is True

    def test_string(self):
        assert _is_nan("abc") is True

    def test_zero(self):
        assert _is_nan(0) is False


# ---------------------------------------------------------------------------
# Tests for _compute_sell_alerts
# ---------------------------------------------------------------------------


class TestComputeSellAlerts:
    """Test sell alert generation logic."""

    def _make_position(self, **overrides):
        """Create a position dict with defaults."""
        pos = {
            "symbol": "TEST",
            "name": "Test Corp",
            "pnl_pct": 0.0,
            "trend": "上昇",
            "rsi": 50.0,
            "alert_level": ALERT_NONE,
            "alert_reasons": [],
            "cross_signal": "none",
            "days_since_cross": None,
            "cross_date": None,
            "value_trap": False,
            "value_trap_reasons": [],
        }
        pos.update(overrides)
        return pos

    def test_no_alerts_for_healthy_position(self):
        """Healthy positions should generate no alerts."""
        positions = [self._make_position()]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) == 0

    def test_exit_alert_generates_critical(self):
        """EXIT level should generate a critical sell alert."""
        positions = [
            self._make_position(
                alert_level=ALERT_EXIT,
                alert_reasons=["デッドクロス + 変化スコア複数悪化"],
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) == 1
        assert alerts[0]["urgency"] == "critical"
        assert alerts[0]["action"] == "売却検討"

    def test_caution_with_loss_generates_critical(self):
        """CAUTION + big unrealized loss should generate critical."""
        positions = [
            self._make_position(
                alert_level=ALERT_CAUTION,
                pnl_pct=-10.0,
                alert_reasons=["変化スコア複数悪化"],
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) == 1
        assert alerts[0]["urgency"] == "critical"
        assert alerts[0]["action"] == "損切り検討"

    def test_caution_without_loss_generates_warning(self):
        """CAUTION without loss should generate warning, not critical."""
        positions = [
            self._make_position(
                alert_level=ALERT_CAUTION,
                pnl_pct=5.0,
                alert_reasons=["SMA50がSMA200に接近"],
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) >= 1
        assert alerts[0]["urgency"] == "warning"
        assert alerts[0]["action"] == "注視・一部売却検討"

    def test_big_profit_with_downtrend(self):
        """Large profit + downtrend should generate take-profit alert."""
        positions = [
            self._make_position(
                pnl_pct=25.0,
                trend="下降",
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) >= 1
        profit_alerts = [a for a in alerts if a["action"] == "利確検討"]
        assert len(profit_alerts) == 1
        assert profit_alerts[0]["urgency"] == "warning"

    def test_profit_without_downtrend_no_alert(self):
        """Large profit + uptrend should NOT generate take-profit alert."""
        positions = [
            self._make_position(
                pnl_pct=25.0,
                trend="上昇",
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) == 0

    def test_recent_death_cross_generates_warning(self):
        """Recent death cross (<=10 days) should generate warning."""
        positions = [
            self._make_position(
                cross_signal="death_cross",
                days_since_cross=5,
                cross_date="2026-02-15",
            )
        ]
        alerts = _compute_sell_alerts(positions)
        assert len(alerts) >= 1
        cross_alerts = [a for a in alerts if "トレンド転換" in a["action"]]
        assert len(cross_alerts) == 1

    def test_old_death_cross_no_alert(self):
        """Old death cross (>10 days) should NOT generate an alert on its own."""
        positions = [
            self._make_position(
                cross_signal="death_cross",
                days_since_cross=30,
                cross_date="2026-01-20",
            )
        ]
        alerts = _compute_sell_alerts(positions)
        cross_alerts = [a for a in alerts if "トレンド転換" in a.get("action", "")]
        assert len(cross_alerts) == 0

    def test_death_cross_skipped_for_exit(self):
        """Death cross alert should be skipped if EXIT already triggered."""
        positions = [
            self._make_position(
                alert_level=ALERT_EXIT,
                alert_reasons=["デッドクロス"],
                cross_signal="death_cross",
                days_since_cross=3,
            )
        ]
        alerts = _compute_sell_alerts(positions)
        # Only the EXIT alert, no duplicate death cross
        assert len(alerts) == 1
        assert alerts[0]["action"] == "売却検討"

    def test_value_trap_generates_warning(self):
        """Value trap detection should generate warning."""
        positions = [
            self._make_position(
                value_trap=True,
                value_trap_reasons=["低PER + EPS減少"],
            )
        ]
        alerts = _compute_sell_alerts(positions)
        trap_alerts = [a for a in alerts if "バリュートラップ" in a["action"]]
        assert len(trap_alerts) == 1
        assert trap_alerts[0]["urgency"] == "warning"

    def test_low_rsi_generates_info(self):
        """RSI <= 30 should generate info-level alert."""
        positions = [self._make_position(rsi=25.0)]
        alerts = _compute_sell_alerts(positions)
        rsi_alerts = [a for a in alerts if "RSI" in a["action"]]
        assert len(rsi_alerts) == 1
        assert rsi_alerts[0]["urgency"] == "info"

    def test_normal_rsi_no_alert(self):
        """Normal RSI should NOT generate alert."""
        positions = [self._make_position(rsi=55.0)]
        alerts = _compute_sell_alerts(positions)
        rsi_alerts = [a for a in alerts if "RSI" in a.get("action", "")]
        assert len(rsi_alerts) == 0

    def test_nan_rsi_no_alert(self):
        """NaN RSI should NOT generate alert."""
        positions = [self._make_position(rsi=float("nan"))]
        alerts = _compute_sell_alerts(positions)
        rsi_alerts = [a for a in alerts if "RSI" in a.get("action", "")]
        assert len(rsi_alerts) == 0

    def test_alerts_sorted_by_urgency(self):
        """Alerts should be sorted: critical > warning > info."""
        positions = [
            self._make_position(
                symbol="A",
                name="A Corp",
                rsi=25.0,
            ),
            self._make_position(
                symbol="B",
                name="B Corp",
                alert_level=ALERT_EXIT,
                alert_reasons=["EXIT"],
            ),
            self._make_position(
                symbol="C",
                name="C Corp",
                value_trap=True,
                value_trap_reasons=["trap"],
            ),
        ]
        alerts = _compute_sell_alerts(positions)
        urgencies = [a["urgency"] for a in alerts]
        expected_order = ["critical", "warning", "info"]
        # Each category should come before the next
        for i in range(len(urgencies) - 1):
            assert expected_order.index(urgencies[i]) <= expected_order.index(urgencies[i + 1])

    def test_multiple_alerts_for_same_stock(self):
        """A stock with multiple conditions can have multiple alerts."""
        positions = [
            self._make_position(
                pnl_pct=25.0,
                trend="下降",
                value_trap=True,
                value_trap_reasons=["low PER + declining EPS"],
                rsi=28.0,
            )
        ]
        alerts = _compute_sell_alerts(positions)
        # Should have profit, value_trap, and RSI alerts
        assert len(alerts) >= 3

    def test_empty_positions(self):
        """Empty positions list should return empty alerts."""
        alerts = _compute_sell_alerts([])
        assert alerts == []


# ---------------------------------------------------------------------------
# Tests for run_dashboard_health_check (with mocks)
# ---------------------------------------------------------------------------


class TestRunDashboardHealthCheck:
    """Integration tests with mocked external calls."""

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_basic_health_check(self, mock_load, mock_yc):
        """Basic health check with one stock."""
        mock_load.return_value = [
            {
                "symbol": "7203.T",
                "shares": 100,
                "cost_price": 2850.0,
                "cost_currency": "JPY",
                "memo": "トヨタ",
            },
        ]
        mock_yc.get_price_history.return_value = _make_uptrend_hist()
        mock_yc.get_stock_detail.return_value = _make_stock_detail("7203.T")

        result = run_dashboard_health_check()

        assert "positions" in result
        assert "alerts" in result
        assert "sell_alerts" in result
        assert "summary" in result
        assert result["summary"]["total"] == 1
        assert len(result["positions"]) == 1

        pos = result["positions"][0]
        assert pos["symbol"] == "7203.T"
        assert pos["name"] == "トヨタ"
        assert "alert_level" in pos
        assert "trend" in pos
        assert "rsi" in pos

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_cash_skipped(self, mock_load, mock_yc):
        """Cash positions should be skipped."""
        mock_load.return_value = [
            {
                "symbol": "JPY.CASH",
                "shares": 1,
                "cost_price": 1000000.0,
                "cost_currency": "JPY",
                "memo": "預り金",
            },
        ]
        result = run_dashboard_health_check()
        assert result["summary"]["total"] == 0
        assert len(result["positions"]) == 0

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_empty_portfolio(self, mock_load, mock_yc):
        """Empty portfolio should return empty results."""
        mock_load.return_value = []
        result = run_dashboard_health_check()
        assert result["summary"]["total"] == 0
        assert result["positions"] == []
        assert result["alerts"] == []
        assert result["sell_alerts"] == []

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_downtrend_generates_alert(self, mock_load, mock_yc):
        """Downtrend stock should generate some alert."""
        mock_load.return_value = [
            {
                "symbol": "BAD.T",
                "shares": 100,
                "cost_price": 200.0,
                "cost_currency": "JPY",
                "memo": "Bad Stock",
            },
        ]
        mock_yc.get_price_history.return_value = _make_downtrend_hist()
        detail = _make_stock_detail("BAD.T")
        # Make fundamentals bad too
        detail["earningsGrowth"] = -0.3
        detail["roe"] = 0.03
        detail["returnOnEquity"] = 0.03
        mock_yc.get_stock_detail.return_value = detail

        result = run_dashboard_health_check()

        assert result["summary"]["total"] == 1
        pos = result["positions"][0]
        # With downtrend + bad fundamentals, should have some alert
        assert pos["alert_level"] != ALERT_NONE or pos["trend"] == "下降"

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_multiple_stocks(self, mock_load, mock_yc):
        """Multiple stocks should all be checked."""
        mock_load.return_value = [
            {
                "symbol": "7203.T",
                "shares": 100,
                "cost_price": 2850.0,
                "cost_currency": "JPY",
                "memo": "トヨタ",
            },
            {
                "symbol": "AAPL",
                "shares": 10,
                "cost_price": 230.0,
                "cost_currency": "USD",
                "memo": "Apple",
            },
            {
                "symbol": "JPY.CASH",
                "shares": 1,
                "cost_price": 1000000.0,
                "cost_currency": "JPY",
                "memo": "現金",
            },
        ]
        mock_yc.get_price_history.return_value = _make_uptrend_hist()
        mock_yc.get_stock_detail.return_value = _make_stock_detail()

        result = run_dashboard_health_check()

        # Cash excluded, 2 stocks checked
        assert result["summary"]["total"] == 2
        assert len(result["positions"]) == 2
        symbols = {p["symbol"] for p in result["positions"]}
        assert symbols == {"7203.T", "AAPL"}

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_null_stock_detail(self, mock_load, mock_yc):
        """None stock detail should not crash."""
        mock_load.return_value = [
            {
                "symbol": "UNKNOWN",
                "shares": 10,
                "cost_price": 100.0,
                "cost_currency": "USD",
                "memo": "Unknown",
            },
        ]
        mock_yc.get_price_history.return_value = _make_flat_hist()
        mock_yc.get_stock_detail.return_value = None

        result = run_dashboard_health_check()
        assert result["summary"]["total"] == 1
        # Should not crash

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_pnl_calculation(self, mock_load, mock_yc):
        """PnL should be calculated from cost vs current price."""
        mock_load.return_value = [
            {
                "symbol": "7203.T",
                "shares": 100,
                "cost_price": 2000.0,
                "cost_currency": "JPY",
                "memo": "トヨタ",
            },
        ]
        hist = _make_uptrend_hist(300, base=2000.0)
        mock_yc.get_price_history.return_value = hist
        mock_yc.get_stock_detail.return_value = _make_stock_detail("7203.T")

        result = run_dashboard_health_check()
        pos = result["positions"][0]
        # Current price > cost (uptrend from 2000)
        assert pos["pnl_pct"] > 0

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_sell_alerts_in_result(self, mock_load, mock_yc):
        """sell_alerts should be present in result."""
        mock_load.return_value = [
            {
                "symbol": "7203.T",
                "shares": 100,
                "cost_price": 2850.0,
                "cost_currency": "JPY",
                "memo": "トヨタ",
            },
        ]
        mock_yc.get_price_history.return_value = _make_uptrend_hist()
        mock_yc.get_stock_detail.return_value = _make_stock_detail("7203.T")

        result = run_dashboard_health_check()
        assert isinstance(result["sell_alerts"], list)

    @patch("components.data_loader.yahoo_client")
    @patch("components.data_loader.load_portfolio")
    def test_health_result_fields(self, mock_load, mock_yc):
        """Each position result should contain all required fields."""
        mock_load.return_value = [
            {
                "symbol": "7203.T",
                "shares": 100,
                "cost_price": 2850.0,
                "cost_currency": "JPY",
                "memo": "トヨタ",
            },
        ]
        mock_yc.get_price_history.return_value = _make_uptrend_hist()
        mock_yc.get_stock_detail.return_value = _make_stock_detail("7203.T")

        result = run_dashboard_health_check()
        pos = result["positions"][0]

        required_fields = [
            "symbol",
            "name",
            "shares",
            "cost_price",
            "current_price",
            "pnl_pct",
            "trend",
            "rsi",
            "sma50",
            "sma200",
            "price_above_sma50",
            "price_above_sma200",
            "cross_signal",
            "days_since_cross",
            "cross_date",
            "change_quality",
            "change_score",
            "indicators",
            "alert_level",
            "alert_emoji",
            "alert_label",
            "alert_reasons",
            "long_term_label",
            "long_term_summary",
            "value_trap",
            "value_trap_reasons",
            "return_stability",
            "return_stability_emoji",
        ]
        for field in required_fields:
            assert field in pos, f"Missing field: {field}"
