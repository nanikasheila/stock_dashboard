"""Tests for components.trade_writer.record_trade()."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import tempfile

from components.trade_writer import record_trade

_DUMMY_CSV = str(Path(tempfile.gettempdir()) / "portfolio.csv")
_DUMMY_JSON_PATH = "data/history/trade/2026-01-01_buy_VTI.json"


def _make_lock_mock():
    """Return a MagicMock that behaves as a context manager (no-op lock)."""
    lock = MagicMock()
    lock.__enter__ = MagicMock(return_value=None)
    lock.__exit__ = MagicMock(return_value=False)
    return lock


# ---------------------------------------------------------------------------
# buy — normal
# ---------------------------------------------------------------------------


@patch("components.trade_writer._update_cash_if_needed")
@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_calls_save_trade_and_add_position(
    mock_save_trade, mock_add_position, mock_acquire_lock, _mock_cash
):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 30}
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act
    result = record_trade(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-01-01",
        csv_path=_DUMMY_CSV,
    )

    # Assert
    mock_save_trade.assert_called_once()
    mock_add_position.assert_called_once()
    assert "購入" in result
    assert "VTI" in result


@patch("components.trade_writer._update_cash_if_needed")
@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_returns_success_message(mock_save_trade, mock_add_position, mock_acquire_lock, _mock_cash):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 10}
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act
    result = record_trade(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-01-01",
        csv_path=_DUMMY_CSV,
    )

    # Assert — message contains type label, symbol, and share count
    assert "購入を記録しました" in result
    assert "VTI" in result
    assert "10" in result


# ---------------------------------------------------------------------------
# sell — normal
# ---------------------------------------------------------------------------


@patch("components.trade_writer._update_cash_if_needed")
@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_calls_save_trade_and_sell_position(
    mock_save_trade, mock_sell_position, mock_acquire_lock, _mock_cash
):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.return_value = {"symbol": "VTI", "shares": 0}
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act
    result = record_trade(
        symbol="VTI",
        trade_type="sell",
        shares=5,
        price=210.0,
        currency="USD",
        trade_date="2026-01-02",
        csv_path=_DUMMY_CSV,
    )

    # Assert
    mock_save_trade.assert_called_once()
    mock_sell_position.assert_called_once()
    assert "売却" in result
    assert "VTI" in result


@patch("components.trade_writer._update_cash_if_needed")
@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_does_not_call_add_position(
    mock_save_trade, mock_sell_position, mock_acquire_lock, _mock_cash
):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.return_value = {"symbol": "VTI", "shares": 0}
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act
    with patch("components.trade_writer.add_position") as mock_add:
        record_trade(
            symbol="VTI",
            trade_type="sell",
            shares=5,
            price=210.0,
            currency="USD",
            trade_date="2026-01-02",
            csv_path=_DUMMY_CSV,
        )
        mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# transfer — normal
# ---------------------------------------------------------------------------


@patch("components.trade_writer._save_trade")
def test_record_trade_transfer_only_calls_save_trade(mock_save_trade):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH

    # Act
    with (
        patch("components.trade_writer.add_position") as mock_add,
        patch("components.trade_writer.sell_position") as mock_sell,
    ):
        result = record_trade(
            symbol="CASH_JPY",
            trade_type="transfer",
            shares=1,
            price=0.0,
            currency="JPY",
            trade_date="2026-01-03",
            csv_path=_DUMMY_CSV,
        )
        # Assert: CSV operations are NOT invoked for transfer
        mock_add.assert_not_called()
        mock_sell.assert_not_called()

    mock_save_trade.assert_called_once()
    assert "振替" in result
    assert "CASH_JPY" in result


@patch("components.trade_writer._save_trade")
def test_record_trade_transfer_returns_success_message(mock_save_trade):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH

    # Act
    result = record_trade(
        symbol="CASH_JPY",
        trade_type="transfer",
        shares=1,
        price=0.0,
        currency="JPY",
        trade_date="2026-01-03",
        csv_path=_DUMMY_CSV,
    )

    # Assert
    assert "振替を記録しました" in result
    assert "1株" in result


# ---------------------------------------------------------------------------
# invalid trade_type
# ---------------------------------------------------------------------------


def test_record_trade_invalid_trade_type_raises_value_error():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Invalid trade_type"):
        record_trade(
            symbol="VTI",
            trade_type="unknown",
            shares=1,
            price=100.0,
            currency="USD",
            trade_date="2026-01-01",
            csv_path=_DUMMY_CSV,
        )


# ---------------------------------------------------------------------------
# CSV update failure → RuntimeError
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_csv_failure_raises_runtime_error(mock_save_trade, mock_add_position, mock_acquire_lock):
    # Arrange: JSON save succeeds, CSV update raises an unexpected error
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.side_effect = OSError("disk full")
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act / Assert
    with pytest.raises(RuntimeError, match="ポートフォリオCSVの更新に失敗"):
        record_trade(
            symbol="VTI",
            trade_type="buy",
            shares=10,
            price=200.0,
            currency="USD",
            trade_date="2026-01-01",
            csv_path=_DUMMY_CSV,
        )


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_csv_failure_raises_runtime_error(mock_save_trade, mock_sell_position, mock_acquire_lock):
    # Arrange: JSON save succeeds, sell_position raises unexpected error
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.side_effect = OSError("disk full")
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act / Assert
    with pytest.raises(RuntimeError, match="ポートフォリオCSVの更新に失敗"):
        record_trade(
            symbol="VTI",
            trade_type="sell",
            shares=5,
            price=210.0,
            currency="USD",
            trade_date="2026-01-02",
            csv_path=_DUMMY_CSV,
        )


# ---------------------------------------------------------------------------
# sell over-holding → ValueError propagates as-is
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_over_holding_propagates_value_error(mock_save_trade, mock_sell_position, mock_acquire_lock):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.side_effect = ValueError("保有株数を超えています")
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act / Assert — ValueError must propagate unchanged (not wrapped)
    with pytest.raises(ValueError, match="保有株数を超えています"):
        record_trade(
            symbol="VTI",
            trade_type="sell",
            shares=999,
            price=210.0,
            currency="USD",
            trade_date="2026-01-02",
            csv_path=_DUMMY_CSV,
        )


# ---------------------------------------------------------------------------
# _acquire_csv_lock — called with correct path
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_acquires_lock_with_csv_path(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    # Arrange
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 10}
    mock_update_cash.return_value = {"symbol": "USD.CASH", "cost_price": 0.0}
    mock_acquire_lock.return_value = _make_lock_mock()

    # Act
    record_trade(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-01-01",
        csv_path=_DUMMY_CSV,
    )

    # Assert: lock was acquired once (position + cash in single lock block)
    mock_acquire_lock.assert_called_once_with(_DUMMY_CSV)


# ---------------------------------------------------------------------------
# Cash position update (Step 3)
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_jpy_updates_cash_with_settlement_jpy(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """buy・JPY・settlement_jpy 指定ありの場合、-settlement_jpy で預り金が更新される."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "7803.T", "shares": 100}
    mock_update_cash.return_value = {"symbol": "JPY.CASH", "cost_price": -89000.0}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="7803.T",
        trade_type="buy",
        shares=100,
        price=1000.0,
        currency="JPY",
        trade_date="2026-02-28",
        settlement_jpy=100500.0,
        csv_path=_DUMMY_CSV,
    )

    mock_update_cash.assert_called_once_with(
        csv_path=_DUMMY_CSV,
        currency="JPY",
        amount_delta=-100500.0,
        trade_date="2026-02-28",
    )


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_usd_updates_cash_with_settlement_usd(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """buy・USD・settlement_usd 指定ありの場合、-settlement_usd で預り金が更新される."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 10}
    mock_update_cash.return_value = {"symbol": "USD.CASH", "cost_price": 1000.0}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-02-28",
        settlement_usd=2050.0,
        csv_path=_DUMMY_CSV,
    )

    mock_update_cash.assert_called_once_with(
        csv_path=_DUMMY_CSV,
        currency="USD",
        amount_delta=-2050.0,
        trade_date="2026-02-28",
    )


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_jpy_adds_cash(mock_save_trade, mock_sell_position, mock_update_cash, mock_acquire_lock):
    """sell・JPY・settlement_jpy 指定ありの場合、+settlement_jpy で預り金が更新される."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.return_value = {"symbol": "7803.T", "shares": 0}
    mock_update_cash.return_value = {"symbol": "JPY.CASH", "cost_price": 200000.0}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="7803.T",
        trade_type="sell",
        shares=100,
        price=1500.0,
        currency="JPY",
        trade_date="2026-02-28",
        settlement_jpy=149500.0,
        csv_path=_DUMMY_CSV,
    )

    mock_update_cash.assert_called_once_with(
        csv_path=_DUMMY_CSV,
        currency="JPY",
        amount_delta=149500.0,
        trade_date="2026-02-28",
    )


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_buy_fallback_to_shares_times_price(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """buy・USD・settlement 両方 0 の場合、shares × price がフォールバックとして使われる."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 10}
    mock_update_cash.return_value = {"symbol": "USD.CASH", "cost_price": 0.0}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="VTI",
        trade_type="buy",
        shares=10,
        price=200.0,
        currency="USD",
        trade_date="2026-02-28",
        settlement_jpy=0.0,
        settlement_usd=0.0,
        csv_path=_DUMMY_CSV,
    )

    # Fallback: 10 * 200.0 = 2000.0 → delta = -2000.0
    mock_update_cash.assert_called_once_with(
        csv_path=_DUMMY_CSV,
        currency="USD",
        amount_delta=-2000.0,
        trade_date="2026-02-28",
    )


@patch("components.trade_writer._save_trade")
def test_record_trade_transfer_does_not_update_cash(mock_save_trade):
    """transfer 取引では update_cash_position が呼ばれない."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH

    with patch("components.trade_writer.update_cash_position") as mock_update_cash:
        record_trade(
            symbol="CASH_JPY",
            trade_type="transfer",
            shares=1,
            price=0.0,
            currency="JPY",
            trade_date="2026-02-28",
            csv_path=_DUMMY_CSV,
        )
        mock_update_cash.assert_not_called()


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_cash_symbol_does_not_update_cash(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """シンボル自体が .CASH（例: JPY.CASH を buy）の場合、重複更新しない."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "JPY.CASH", "shares": 1}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="JPY.CASH",
        trade_type="buy",
        shares=1,
        price=50000.0,
        currency="JPY",
        trade_date="2026-02-28",
        csv_path=_DUMMY_CSV,
    )

    mock_update_cash.assert_not_called()


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_cash_update_failure_raises_runtime_error(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """キャッシュ更新失敗時に RuntimeError がスローされる."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "VTI", "shares": 10}
    mock_update_cash.side_effect = OSError("disk full")
    mock_acquire_lock.return_value = _make_lock_mock()

    with pytest.raises(RuntimeError, match="ポートフォリオCSVの更新に失敗"):
        record_trade(
            symbol="VTI",
            trade_type="buy",
            shares=10,
            price=200.0,
            currency="USD",
            trade_date="2026-02-28",
            settlement_usd=2000.0,
            csv_path=_DUMMY_CSV,
        )


# ---------------------------------------------------------------------------
# W-3: sell ValueError does not trigger cash update
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.sell_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_sell_raises_does_not_update_cash(
    mock_save_trade, mock_sell_position, mock_update_cash, mock_acquire_lock
):
    """sell_position が ValueError を送出した場合、update_cash_position は呼ばれない."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_sell_position.side_effect = ValueError("保有株数を超えています")
    mock_acquire_lock.return_value = _make_lock_mock()

    with pytest.raises(ValueError, match="保有株数を超えています"):
        record_trade(
            symbol="VTI",
            trade_type="sell",
            shares=999,
            price=210.0,
            currency="USD",
            trade_date="2026-02-28",
            csv_path=_DUMMY_CSV,
        )

    mock_update_cash.assert_not_called()


# ---------------------------------------------------------------------------
# W-4: zero price and zero settlement skips cash update
# ---------------------------------------------------------------------------


@patch("components.trade_writer._acquire_csv_lock")
@patch("components.trade_writer.update_cash_position")
@patch("components.trade_writer.add_position")
@patch("components.trade_writer._save_trade")
def test_record_trade_zero_price_no_settlement_skips_cash(
    mock_save_trade, mock_add_position, mock_update_cash, mock_acquire_lock
):
    """price=0.0, settlement=0.0 の場合、update_cash_position が呼ばれない."""
    mock_save_trade.return_value = _DUMMY_JSON_PATH
    mock_add_position.return_value = {"symbol": "GIFT", "shares": 10}
    mock_acquire_lock.return_value = _make_lock_mock()

    record_trade(
        symbol="GIFT",
        trade_type="buy",
        shares=10,
        price=0.0,
        currency="USD",
        trade_date="2026-02-28",
        settlement_jpy=0.0,
        settlement_usd=0.0,
        csv_path=_DUMMY_CSV,
    )

    mock_update_cash.assert_not_called()
