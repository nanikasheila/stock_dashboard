"""Tests for components.trade_impact module.

Why: Validates compute_trade_impact for buy (new & existing position),
     sell, TradeImpact construction, and LLM commentary with mocked
     copilot_call.
How: Uses a deterministic mock snapshot; each test follows AAA pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.trade_impact import (
    TradeImpact,
    _classify_risk,
    _compute_hhi,
    compute_trade_impact,
    generate_trade_commentary,
)

# ---------------------------------------------------------------------------
# Common test data
# ---------------------------------------------------------------------------

_MOCK_SNAPSHOT: dict = {
    "positions": [
        {
            "symbol": "AAPL",
            "sector": "Technology",
            "evaluation_jpy": 500_000,
            "market_currency": "USD",
            "cost_currency": "USD",
        },
        {
            "symbol": "7203.T",
            "sector": "自動車",
            "evaluation_jpy": 300_000,
            "market_currency": "JPY",
            "cost_currency": "JPY",
        },
    ],
    "total_value_jpy": 800_000,
    "fx_rates": {"USD": 150.0, "JPY": 1.0},
}


# ---------------------------------------------------------------------------
# TradeImpact dataclass construction
# ---------------------------------------------------------------------------


def test_trade_impact_dataclass_construction():
    """TradeImpact can be instantiated with all required fields."""
    # Arrange & Act
    impact = TradeImpact(
        symbol="AAPL",
        trade_type="buy",
        shares=10,
        trade_value_jpy=150_000,
        total_value_before=800_000,
        total_value_after=950_000,
        symbol_weight_before=62.5,
        symbol_weight_after=68.42,
        sector_before={"Technology": 62.5, "自動車": 37.5},
        sector_after={"Technology": 68.42, "自動車": 31.58},
        currency_before={"USD": 62.5, "JPY": 37.5},
        currency_after={"USD": 68.42, "JPY": 31.58},
        hhi_before=0.53,
        hhi_after=0.57,
        risk_level_before="危険な集中",
        risk_level_after="危険な集中",
    )

    # Assert
    assert impact.symbol == "AAPL"
    assert impact.trade_type == "buy"
    assert impact.shares == 10
    assert impact.trade_value_jpy == 150_000


def test_trade_impact_default_fields():
    """TradeImpact fields with defaults are initialized correctly."""
    # Arrange & Act
    impact = TradeImpact(
        symbol="VTI",
        trade_type="buy",
        shares=5,
        trade_value_jpy=100_000,
        total_value_before=0,
        total_value_after=100_000,
        symbol_weight_before=0.0,
        symbol_weight_after=100.0,
    )

    # Assert
    assert impact.sector_before == {}
    assert impact.sector_after == {}
    assert impact.currency_before == {}
    assert impact.currency_after == {}
    assert impact.hhi_before == 0.0
    assert impact.hhi_after == 0.0
    assert impact.risk_level_before == ""
    assert impact.risk_level_after == ""


# ---------------------------------------------------------------------------
# compute_trade_impact — buy existing position
# ---------------------------------------------------------------------------


def test_compute_trade_impact_buy_existing_position():
    """Buying more of an existing symbol increases its weight and total."""
    # Arrange
    shares = 10
    price = 100.0
    fx_rate = 150.0
    expected_trade_value = shares * price * fx_rate  # 150,000

    # Act
    impact = compute_trade_impact(
        snapshot=_MOCK_SNAPSHOT,
        trade_type="buy",
        symbol="AAPL",
        shares=shares,
        price=price,
        currency="USD",
        fx_rate=fx_rate,
    )

    # Assert
    assert impact.trade_value_jpy == expected_trade_value
    assert impact.total_value_before == 800_000
    assert impact.total_value_after == 800_000 + expected_trade_value
    assert impact.symbol_weight_after > impact.symbol_weight_before
    assert impact.hhi_after > 0  # concentration exists


def test_compute_trade_impact_buy_new_position():
    """Buying a symbol not in the portfolio creates a new position."""
    # Arrange
    shares = 5
    price = 200.0
    fx_rate = 150.0
    expected_trade_value = shares * price * fx_rate  # 150,000

    # Act
    impact = compute_trade_impact(
        snapshot=_MOCK_SNAPSHOT,
        trade_type="buy",
        symbol="MSFT",
        shares=shares,
        price=price,
        currency="USD",
        fx_rate=fx_rate,
    )

    # Assert
    assert impact.symbol == "MSFT"
    assert impact.trade_value_jpy == expected_trade_value
    assert impact.symbol_weight_before == 0.0
    assert impact.symbol_weight_after > 0.0
    assert impact.total_value_after == 800_000 + expected_trade_value


# ---------------------------------------------------------------------------
# compute_trade_impact — sell
# ---------------------------------------------------------------------------


def test_compute_trade_impact_sell_partial():
    """Selling part of a position reduces its weight."""
    # Arrange — sell ¥100,000 worth of AAPL (from ¥500,000)
    shares = 1
    price = 100.0
    fx_rate = 150.0
    expected_trade_value = shares * price * fx_rate  # 15,000

    # Act
    impact = compute_trade_impact(
        snapshot=_MOCK_SNAPSHOT,
        trade_type="sell",
        symbol="AAPL",
        shares=shares,
        price=price,
        currency="USD",
        fx_rate=fx_rate,
    )

    # Assert
    assert impact.trade_value_jpy == expected_trade_value
    assert impact.total_value_after == 800_000 - expected_trade_value
    assert impact.symbol_weight_after < impact.symbol_weight_before


def test_compute_trade_impact_sell_full_position():
    """Selling the entire position removes the symbol from portfolio."""
    # Arrange — sell exactly ¥500,000 worth of AAPL
    # 500_000 / (price * fx_rate) shares needed to equal full position
    price = 100.0
    fx_rate = 1.0
    shares_to_sell = 5000  # 5000 * 100 * 1.0 = 500,000

    # Act
    impact = compute_trade_impact(
        snapshot=_MOCK_SNAPSHOT,
        trade_type="sell",
        symbol="AAPL",
        shares=shares_to_sell,
        price=price,
        currency="USD",
        fx_rate=fx_rate,
    )

    # Assert: AAPL weight should be 0 after full sale
    assert impact.symbol_weight_after == 0.0
    assert impact.total_value_after == 300_000


# ---------------------------------------------------------------------------
# compute_trade_impact — transfer
# ---------------------------------------------------------------------------


def test_compute_trade_impact_transfer_no_value_change():
    """Transfer does not change evaluation values."""
    # Act
    impact = compute_trade_impact(
        snapshot=_MOCK_SNAPSHOT,
        trade_type="transfer",
        symbol="AAPL",
        shares=10,
        price=100.0,
        currency="USD",
        fx_rate=150.0,
    )

    # Assert: total unchanged
    assert impact.total_value_before == impact.total_value_after
    assert impact.symbol_weight_before == impact.symbol_weight_after


# ---------------------------------------------------------------------------
# HHI and risk classification helpers
# ---------------------------------------------------------------------------


def test_compute_hhi_two_equal_positions():
    """Two equal positions yield HHI = 0.5."""
    # Arrange
    evaluations = [100.0, 100.0]
    total = 200.0

    # Act
    hhi = _compute_hhi(evaluations, total)

    # Assert
    assert abs(hhi - 0.5) < 1e-9


def test_compute_hhi_single_position():
    """Single position yields HHI = 1.0 (max concentration)."""
    assert abs(_compute_hhi([100.0], 100.0) - 1.0) < 1e-9


def test_compute_hhi_zero_total():
    """Zero total returns HHI = 0."""
    assert _compute_hhi([], 0.0) == 0.0


def test_classify_risk_distributed():
    assert _classify_risk(0.10) == "分散"


def test_classify_risk_moderate():
    assert _classify_risk(0.20) == "やや集中"


def test_classify_risk_concentrated():
    assert _classify_risk(0.30) == "危険な集中"


def test_classify_risk_boundary_distributed():
    """HHI exactly at 0.15 is moderate, not distributed."""
    assert _classify_risk(0.15) == "やや集中"


def test_classify_risk_boundary_moderate():
    """HHI exactly at 0.25 is concentrated, not moderate."""
    assert _classify_risk(0.25) == "危険な集中"


# ---------------------------------------------------------------------------
# generate_trade_commentary — mock copilot_call
# ---------------------------------------------------------------------------


@patch("components.trade_impact.copilot_call")
def test_generate_trade_commentary_returns_response(mock_call):
    """Commentary returns LLM response text on success."""
    # Arrange
    mock_call.return_value = "この取引は分散効果があります。"
    impact = TradeImpact(
        symbol="AAPL",
        trade_type="buy",
        shares=10,
        trade_value_jpy=150_000,
        total_value_before=800_000,
        total_value_after=950_000,
        symbol_weight_before=62.5,
        symbol_weight_after=68.42,
        sector_before={"Technology": 62.5},
        sector_after={"Technology": 68.42},
        currency_before={"USD": 62.5},
        currency_after={"USD": 68.42},
        hhi_before=0.53,
        hhi_after=0.57,
        risk_level_before="危険な集中",
        risk_level_after="危険な集中",
    )

    # Act
    result = generate_trade_commentary(impact, model="gpt-4.1", source="test")

    # Assert
    assert result == "この取引は分散効果があります。"
    mock_call.assert_called_once()
    call_kwargs = mock_call.call_args
    assert call_kwargs[1]["model"] == "gpt-4.1"
    assert call_kwargs[1]["source"] == "test"


@patch("components.trade_impact.copilot_call")
def test_generate_trade_commentary_returns_none_on_failure(mock_call):
    """Commentary returns None when copilot_call raises an exception."""
    # Arrange
    mock_call.side_effect = RuntimeError("connection error")
    impact = TradeImpact(
        symbol="AAPL",
        trade_type="sell",
        shares=5,
        trade_value_jpy=75_000,
        total_value_before=800_000,
        total_value_after=725_000,
        symbol_weight_before=62.5,
        symbol_weight_after=55.17,
        hhi_before=0.53,
        hhi_after=0.48,
        risk_level_before="危険な集中",
        risk_level_after="危険な集中",
    )

    # Act
    result = generate_trade_commentary(impact)

    # Assert
    assert result is None


@patch("components.trade_impact.copilot_call")
def test_generate_trade_commentary_no_model_uses_default(mock_call):
    """When model is None, copilot_call is invoked without model kwarg."""
    # Arrange
    mock_call.return_value = "コメント"
    impact = TradeImpact(
        symbol="VTI",
        trade_type="buy",
        shares=1,
        trade_value_jpy=30_000,
        total_value_before=800_000,
        total_value_after=830_000,
        symbol_weight_before=0.0,
        symbol_weight_after=3.61,
        hhi_before=0.5,
        hhi_after=0.47,
        risk_level_before="危険な集中",
        risk_level_after="危険な集中",
    )

    # Act
    generate_trade_commentary(impact, model=None, source="test")

    # Assert: model kwarg should NOT be passed
    call_kwargs = mock_call.call_args[1]
    assert "model" not in call_kwargs
