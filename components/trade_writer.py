"""Trade write facade for CQRS separation (ADR-002).

Why: Trade recording involves two operations (JSON + CSV) that must stay
     consistent. Centralizing write logic here prevents duplication and
     ensures the correct ordering: JSON first (Source of Truth), CSV second.
How: Uses history_store.save_trade for JSON persistence, then dispatches to
     portfolio_manager.add_position / sell_position depending on trade_type.
     filelock guards CSV writes against concurrent Streamlit sessions.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import AbstractContextManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-level lock for CSV writes (thread-safe within a single process)
# ---------------------------------------------------------------------------
_CSV_THREAD_LOCK = threading.Lock()

try:
    from filelock import FileLock as _FileLock

    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover — filelock is in requirements.txt
    _FILELOCK_AVAILABLE = False
    logger.warning("filelock not installed; falling back to threading.Lock for CSV writes")

from src.core.common import is_cash as _is_cash
from src.core.portfolio.portfolio_manager import (
    DEFAULT_CSV_PATH,
    add_position,
    sell_position,
    update_cash_position,
)
from src.data.history_store import save_trade as _save_trade


def _acquire_csv_lock(csv_path: str) -> AbstractContextManager:
    """Return a context manager that locks the CSV path for writing.

    Why: Streamlit can run multiple sessions in the same process, creating
         race conditions on the portfolio CSV.
    How: If filelock is available, use a file-based lock (cross-process safe).
         Otherwise fall back to the module-level threading.Lock.
    """
    if _FILELOCK_AVAILABLE:
        return _FileLock(f"{csv_path}.lock", timeout=10)
    return _CSV_THREAD_LOCK


def _resolve_settlement_amount(
    currency: str,
    settlement_jpy: float,
    settlement_usd: float,
    shares: int,
    price: float,
) -> float:
    """Determine the effective settlement amount for a trade.

    Why: The trade form allows explicit JPY / USD settlement inputs to
         capture real brokerage amounts (including commissions).  When
         neither is provided, ``shares × price`` is a reasonable fallback.
    How: Pick the settlement field that matches the trade currency.
         If neither matches (e.g. EUR trades) or both are zero, fall
         back to ``shares × price``.
    """
    if currency == "JPY" and settlement_jpy > 0:
        return settlement_jpy
    if currency == "USD" and settlement_usd > 0:
        return settlement_usd
    # Why: For non-JPY/USD currencies, or when the user didn't fill in
    #      the explicit settlement, use the notional amount as fallback.
    fallback = shares * price
    return fallback


def _update_cash_if_needed(
    symbol: str,
    trade_type: str,
    currency: str,
    settlement_jpy: float,
    settlement_usd: float,
    shares: int,
    price: float,
    csv_path: str,
    trade_date: str,
) -> None:
    """Update the cash deposit balance if the trade warrants it.

    Why: Buying a stock consumes cash; selling a stock produces cash.
         Keeping {CURRENCY}.CASH rows accurate gives the user a live
         view of available cash by currency.
    How: Skip if the symbol itself is cash (avoid circular update).
         Determine the settlement amount from explicit inputs or
         fallback (shares × price).  Apply as positive delta (sell)
         or negative delta (buy).  Negative balances are allowed
         (interpreted as an implicit deposit from savings).
         Called inside the CSV lock acquired by the caller.
    """
    if _is_cash(symbol):
        return

    settlement_amount = _resolve_settlement_amount(
        currency=currency,
        settlement_jpy=settlement_jpy,
        settlement_usd=settlement_usd,
        shares=shares,
        price=price,
    )

    if settlement_amount <= 0:
        logger.warning(
            "Cash position NOT updated: settlement resolved to 0 "
            "(symbol=%s, shares=%d, price=%.4f)",
            symbol,
            shares,
            price,
        )
        return

    cash_delta = settlement_amount if trade_type == "sell" else -settlement_amount
    updated_cash = update_cash_position(
        csv_path=csv_path,
        currency=currency,
        amount_delta=cash_delta,
        trade_date=trade_date,
    )
    logger.info("Cash position updated: %s", updated_cash)


def record_trade(
    symbol: str,
    trade_type: str,
    shares: int,
    price: float,
    currency: str,
    trade_date: str,
    memo: str = "",
    fx_rate: float = 1.0,
    settlement_jpy: float = 0.0,
    settlement_usd: float = 0.0,
    csv_path: str = DEFAULT_CSV_PATH,
) -> str:
    """Record a trade to JSON history and update portfolio CSV.

    Why: Buy/sell trades must update both the trade history (JSON) and the
         live portfolio positions (CSV). Transfer trades only need history.
         JSON is the Source of Truth (ADR-002), so it is written first.
    How: Step 1 — save_trade writes the JSON record unconditionally.
         Step 2 — for buy, add_position updates the CSV under filelock.
                  for sell, sell_position updates the CSV under filelock.
                  for transfer, no CSV update is performed.
         Step 3 — for buy/sell of non-cash symbols, update the
                  corresponding {CURRENCY}.CASH deposit balance.
         Raises on any error after logging context for diagnostics.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. ``VTI``, ``7803.T``, ``CASH_JPY``.
    trade_type : str
        One of ``"buy"``, ``"sell"``, or ``"transfer"``.
    shares : int
        Number of shares traded. Must be positive.
    price : float
        Cost price per share in *currency*. Use 0.0 for transfers.
    currency : str
        ISO currency code, e.g. ``"USD"``, ``"JPY"``.
    trade_date : str
        Trade date in ``YYYY-MM-DD`` format.
    memo : str
        Optional free-text note.
    fx_rate : float
        Exchange rate to JPY (e.g. 150.5 for USD→JPY). Use 1.0 for JPY.
    settlement_jpy : float
        Settlement amount in JPY.
    settlement_usd : float
        Settlement amount in USD.
    csv_path : str
        Absolute path to the portfolio CSV file.

    Returns
    -------
    str
        Human-readable success message.

    Raises
    ------
    ValueError
        If sell_position finds the symbol is not held or shares exceed holding.
    RuntimeError
        If JSON was saved but the CSV update failed (data-inconsistency guard).
    """
    trade_type_lower = trade_type.lower()
    if trade_type_lower not in {"buy", "sell", "transfer"}:
        raise ValueError(f"Invalid trade_type: '{trade_type}'. Must be buy, sell, or transfer.")

    if not symbol or not symbol.strip():
        raise ValueError("symbol is required.")
    if shares <= 0:
        raise ValueError(f"shares must be positive, got {shares}.")

    # Step 1: Persist JSON (Source of Truth — ADR-002)
    saved_json_path = _save_trade(
        symbol=symbol,
        trade_type=trade_type_lower,
        shares=shares,
        price=price,
        currency=currency,
        date_str=trade_date,
        memo=memo,
        fx_rate=fx_rate,
        settlement_jpy=settlement_jpy,
        settlement_usd=settlement_usd,
    )
    logger.info("Trade JSON saved: %s", saved_json_path)

    # Step 2 + 3: Update portfolio CSV and cash deposit under a single lock
    # Why: Both position and cash updates operate on the same portfolio CSV.
    #      A single lock prevents other Streamlit sessions from reading an
    #      intermediate state (position changed, cash not yet updated).
    if trade_type_lower == "buy":
        try:
            lock = _acquire_csv_lock(csv_path)
            with lock:
                updated_position = add_position(
                    csv_path=csv_path,
                    symbol=symbol,
                    shares=shares,
                    cost_price=price,
                    cost_currency=currency,
                    purchase_date=trade_date,
                    memo=memo,
                )
                logger.info("Position added/updated: %s", updated_position)

                # Step 3: Cash deposit update (buy = negative delta)
                _update_cash_if_needed(
                    symbol=symbol,
                    trade_type=trade_type_lower,
                    currency=currency,
                    settlement_jpy=settlement_jpy,
                    settlement_usd=settlement_usd,
                    shares=shares,
                    price=price,
                    csv_path=csv_path,
                    trade_date=trade_date,
                )
        except Exception as exc:
            logger.error(
                "CSV update failed after JSON save (path=%s, trade=%s %s x%d): %s",
                csv_path,
                trade_type_lower,
                symbol,
                shares,
                exc,
            )
            raise RuntimeError(
                f"取引はJSONに保存されましたが、ポートフォリオCSVの更新に失敗しました: {exc}"
            ) from exc

    elif trade_type_lower == "sell":
        try:
            lock = _acquire_csv_lock(csv_path)
            with lock:
                updated_position = sell_position(
                    csv_path=csv_path,
                    symbol=symbol,
                    shares=shares,
                )
                logger.info("Position sold: %s", updated_position)

                # Step 3: Cash deposit update (sell = positive delta)
                _update_cash_if_needed(
                    symbol=symbol,
                    trade_type=trade_type_lower,
                    currency=currency,
                    settlement_jpy=settlement_jpy,
                    settlement_usd=settlement_usd,
                    shares=shares,
                    price=price,
                    csv_path=csv_path,
                    trade_date=trade_date,
                )
        except ValueError:
            # Re-raise as-is so caller can show the user-facing message
            raise
        except Exception as exc:
            logger.error(
                "CSV update failed after JSON save (path=%s, trade=%s %s x%d): %s",
                csv_path,
                trade_type_lower,
                symbol,
                shares,
                exc,
            )
            raise RuntimeError(
                f"取引はJSONに保存されましたが、ポートフォリオCSVの更新に失敗しました: {exc}"
            ) from exc

    # transfer: JSON only — no CSV change needed
    type_label = {"buy": "購入", "sell": "売却", "transfer": "振替"}[trade_type_lower]
    return f"{type_label}を記録しました: {symbol} × {shares}株"
