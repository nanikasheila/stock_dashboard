"""Tests for components.trade_form._handle_submit()."""

import datetime
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.trade_form import _handle_submit

# ---------------------------------------------------------------------------
# Common test data
# ---------------------------------------------------------------------------

_TRADE_DATE = datetime.date(2026, 1, 15)
from typing import Any

_DEFAULT_ARGS: dict[str, Any] = dict(
    trade_type="buy",
    symbol_raw="VTI",
    trade_date=_TRADE_DATE,
    shares_count=10,
    unit_price=200.0,
    currency="USD",
    fx_rate=150.0,
    settlement_jpy=300_000.0,
    settlement_usd=2_000.0,
    memo="テスト購入",
)


# ---------------------------------------------------------------------------
# Success — buy
# ---------------------------------------------------------------------------


@patch("streamlit.rerun")
@patch("streamlit.cache_data")
@patch("streamlit.success")
@patch("components.trade_form.record_trade")
def test_handle_submit_buy_calls_record_trade_with_correct_args(
    mock_record_trade, mock_st_success, mock_cache_data, mock_rerun
):
    # Arrange
    mock_record_trade.return_value = "購入を記録しました: VTI × 10株"
    mock_cache_data.clear = MagicMock()

    # Act
    _handle_submit(**_DEFAULT_ARGS)

    # Assert: record_trade called with expected arguments
    mock_record_trade.assert_called_once_with(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-01-15",
        memo="テスト購入",
        fx_rate=150.0,
        settlement_jpy=300_000.0,
        settlement_usd=2_000.0,
    )


@patch("streamlit.rerun")
@patch("streamlit.cache_data")
@patch("streamlit.success")
@patch("components.trade_form.record_trade")
def test_handle_submit_buy_success_shows_success_and_clears_cache(
    mock_record_trade, mock_st_success, mock_cache_data, mock_rerun
):
    # Arrange
    mock_record_trade.return_value = "購入を記録しました: VTI × 10株"
    mock_cache_data.clear = MagicMock()

    # Act
    _handle_submit(**_DEFAULT_ARGS)

    # Assert: success shown, cache cleared, rerun triggered
    mock_st_success.assert_called_once()
    mock_cache_data.clear.assert_called_once()
    mock_rerun.assert_called_once()


# ---------------------------------------------------------------------------
# Validation — empty symbol
# ---------------------------------------------------------------------------


@patch("streamlit.error")
@patch("components.trade_form.record_trade")
def test_handle_submit_empty_symbol_shows_error_and_skips_record_trade(mock_record_trade, mock_st_error):
    # Arrange
    args = {**_DEFAULT_ARGS, "symbol_raw": "   "}

    # Act
    _handle_submit(**args)

    # Assert: error shown, record_trade NOT called
    mock_st_error.assert_called_once()
    error_msg: str = mock_st_error.call_args[0][0]
    assert "銘柄コード" in error_msg
    mock_record_trade.assert_not_called()


@patch("streamlit.error")
@patch("components.trade_form.record_trade")
def test_handle_submit_blank_symbol_shows_error_and_skips_record_trade(mock_record_trade, mock_st_error):
    # Arrange
    args = {**_DEFAULT_ARGS, "symbol_raw": ""}

    # Act
    _handle_submit(**args)

    # Assert
    mock_st_error.assert_called_once()
    mock_record_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling — ValueError from record_trade
# ---------------------------------------------------------------------------


@patch("streamlit.rerun")
@patch("streamlit.error")
@patch("components.trade_form.record_trade")
def test_handle_submit_record_trade_value_error_shows_error(mock_record_trade, mock_st_error, mock_rerun):
    # Arrange
    mock_record_trade.side_effect = ValueError("保有株数を超えています")

    # Act
    _handle_submit(**_DEFAULT_ARGS)

    # Assert: error displayed, rerun NOT called
    mock_st_error.assert_called_once()
    error_msg: str = mock_st_error.call_args[0][0]
    assert "保有株数" in error_msg
    mock_rerun.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling — RuntimeError from record_trade
# ---------------------------------------------------------------------------


@patch("streamlit.rerun")
@patch("streamlit.error")
@patch("components.trade_form.record_trade")
def test_handle_submit_record_trade_runtime_error_shows_error(mock_record_trade, mock_st_error, mock_rerun):
    # Arrange
    mock_record_trade.side_effect = RuntimeError("ポートフォリオCSVの更新に失敗しました")

    # Act
    _handle_submit(**_DEFAULT_ARGS)

    # Assert: error displayed, rerun NOT called
    mock_st_error.assert_called_once()
    error_msg: str = mock_st_error.call_args[0][0]
    assert "ポートフォリオCSV" in error_msg
    mock_rerun.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling — unexpected exception
# ---------------------------------------------------------------------------


@patch("streamlit.rerun")
@patch("streamlit.error")
@patch("components.trade_form.record_trade")
def test_handle_submit_unexpected_error_shows_generic_error_message(mock_record_trade, mock_st_error, mock_rerun):
    # Arrange
    mock_record_trade.side_effect = Exception("network timeout")

    # Act
    _handle_submit(**_DEFAULT_ARGS)

    # Assert: error message contains "予期しないエラー"
    mock_st_error.assert_called_once()
    error_msg: str = mock_st_error.call_args[0][0]
    assert "予期しないエラー" in error_msg
    mock_rerun.assert_not_called()


# ---------------------------------------------------------------------------
# transfer — unit_price must be 0.0
# ---------------------------------------------------------------------------


@patch("streamlit.rerun")
@patch("streamlit.cache_data")
@patch("streamlit.success")
@patch("components.trade_form.record_trade")
def test_handle_submit_transfer_passes_price_zero(mock_record_trade, mock_st_success, mock_cache_data, mock_rerun):
    # Arrange: _render_form_body already forces unit_price=0.0 for transfer,
    # but _handle_submit receives unit_price as argument. Simulate the same.
    mock_record_trade.return_value = "振替を記録しました: CASH_JPY × 1株"
    mock_cache_data.clear = MagicMock()

    transfer_args = {
        "trade_type": "transfer",
        "symbol_raw": "CASH_JPY",
        "trade_date": _TRADE_DATE,
        "shares_count": 1,
        "unit_price": 0.0,  # enforced by the form UI
        "currency": "JPY",
        "fx_rate": 1.0,
        "settlement_jpy": 100_000.0,
        "settlement_usd": 0.0,
        "memo": "入金",
    }

    # Act
    _handle_submit(**transfer_args)

    # Assert: price=0.0 forwarded to record_trade
    call_kwargs = mock_record_trade.call_args[1]
    assert call_kwargs["price"] == 0.0
    assert call_kwargs["trade_type"] == "transfer"
