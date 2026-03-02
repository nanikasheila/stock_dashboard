"""Trade entry form UI component for the Streamlit dashboard.

Why: Users need to record trades (buy/sell/transfer) directly from the
     dashboard without manually editing JSON/CSV files.
How: Renders a collapsible expander containing a st.form. On submit,
     delegates to trade_writer.record_trade(). On success, clears all
     Streamlit data caches and triggers a full rerun so the updated
     trade activity and portfolio sections reflect the new trade.
"""

from __future__ import annotations

import datetime
import logging

import streamlit as st

from components.trade_writer import record_trade

logger = logging.getLogger(__name__)

# Supported currencies shown in the selectbox
_SUPPORTED_CURRENCIES: list[str] = [
    "USD",
    "JPY",
    "SGD",
    "EUR",
    "GBP",
    "HKD",
    "AUD",
    "CAD",
    "THB",
    "MYR",
    "IDR",
    "PHP",
    "KRW",
    "TWD",
    "CNY",
    "INR",
    "BRL",
]

# Default FX rates shown when the user selects a currency
# (informational defaults only — user should fill in actual rate)
_DEFAULT_FX_RATE: dict[str, float] = {
    "JPY": 1.0,
    "USD": 150.0,
    "SGD": 110.0,
    "EUR": 160.0,
    "GBP": 190.0,
    "HKD": 19.0,
    "AUD": 100.0,
    "CAD": 110.0,
    "THB": 4.2,
    "MYR": 34.0,
    "IDR": 0.0096,
    "PHP": 2.6,
    "KRW": 0.11,
    "TWD": 4.7,
    "CNY": 21.0,
    "INR": 1.8,
    "BRL": 30.0,
}


def render_trade_form(
    snapshot: dict | None = None,
    settings: dict | None = None,
) -> None:
    """Render a collapsible trade input form in the Streamlit dashboard.

    Why: Providing inline trade recording avoids context-switching to file
         editors and keeps the audit trail consistent.
    How: Uses st.expander + st.form for a single-submit UX. Validates inputs
         (symbol required, sell shares vs holdings, positive numbers) before
         calling record_trade(). On success, clears all st.cache_data entries
         and calls st.rerun() so the dashboard reflects the new data.
    """
    with st.expander("➕ 取引を記録する", expanded=False):
        _render_form_body(snapshot=snapshot, settings=settings)


def _render_form_body(
    snapshot: dict | None = None,
    settings: dict | None = None,
) -> None:
    """Render the form fields and handle submission.

    Why: Separated from render_trade_form to keep the expander wrapper thin
         and make the form logic independently testable.
    How: Builds input widgets, collects values on submit, runs validation,
         then delegates persistence to trade_writer.record_trade().
    """
    with st.form("trade_input_form", clear_on_submit=True):
        col_left, col_right = st.columns(2)

        with col_left:
            trade_type: str = st.selectbox(
                "取引タイプ",
                options=["buy", "sell", "transfer"],
                format_func=lambda t: {
                    "buy": "🟢 buy（購入）",
                    "sell": "🔴 sell（売却）",
                    "transfer": "⚪ transfer（振替）",
                }.get(t, t),
                key="trade_form_type",
            )

            symbol_raw: str = st.text_input(
                "銘柄コード",
                placeholder="例: VTI, 7803.T, CASH_JPY",
                key="trade_form_symbol",
            )

            trade_date: datetime.date = st.date_input(
                "取引日",
                value=datetime.date.today(),
                key="trade_form_date",
            )

            shares_count: int = st.number_input(
                "株数",
                min_value=1,
                value=1,
                step=1,
                key="trade_form_shares",
            )

        with col_right:
            is_transfer: bool = trade_type == "transfer"

            unit_price: float = st.number_input(
                "単価",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.4f",
                disabled=is_transfer,
                help="transfer の場合は 0 固定です。",
                key="trade_form_price",
            )

            currency: str = st.selectbox(
                "通貨",
                options=_SUPPORTED_CURRENCIES,
                index=0,
                key="trade_form_currency",
            )

            is_jpy: bool = currency == "JPY"
            default_fx: float = _DEFAULT_FX_RATE.get(currency, 1.0) if not is_jpy else 1.0
            fx_rate: float = st.number_input(
                "為替レート (→円)",
                min_value=0.0,
                value=default_fx,
                step=0.01,
                format="%.4f",
                disabled=is_jpy,
                help="JPY の場合は 1.0 固定です。",
                key="trade_form_fx_rate",
            )

            settlement_jpy: float = st.number_input(
                "円建て決済額",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.0f",
                key="trade_form_settlement_jpy",
            )

            settlement_usd: float = st.number_input(
                "USD建て決済額",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key="trade_form_settlement_usd",
            )

        memo: str = st.text_input(
            "メモ (任意)",
            placeholder="例: 積立、特定口座、NISAなど",
            key="trade_form_memo",
        )

        submitted: bool = st.form_submit_button(
            "✅ 取引を記録する",
            type="primary",
            use_container_width=True,
        )

    # --- 取引影響プレビュー ---
    _trade_preview_enabled = (settings or {}).get("trade_preview_enabled", False)
    if _trade_preview_enabled and snapshot and not is_transfer:
        _symbol_val = st.session_state.get("trade_form_symbol", "").strip().upper()
        _shares_val = st.session_state.get("trade_form_shares", 0)
        _price_val = st.session_state.get("trade_form_price", 0.0)
        _fx_val = st.session_state.get("trade_form_fx_rate", 1.0)
        _currency_val = st.session_state.get("trade_form_currency", "USD")
        _type_val = st.session_state.get("trade_form_type", "buy")

        if _symbol_val and _shares_val > 0 and _price_val > 0:
            if st.button("📊 影響分析", key="btn_trade_impact"):
                from components.trade_impact import (
                    compute_trade_impact,
                    render_trade_impact,
                )

                _impact = compute_trade_impact(
                    snapshot=snapshot,
                    trade_type=_type_val,
                    symbol=_symbol_val,
                    shares=int(_shares_val),
                    price=float(_price_val),
                    currency=_currency_val,
                    fx_rate=float(_fx_val),
                )
                render_trade_impact(_impact, settings or {})

    if submitted:
        _handle_submit(
            trade_type=trade_type,
            symbol_raw=symbol_raw,
            trade_date=trade_date,
            shares_count=int(shares_count),
            unit_price=float(unit_price) if not is_transfer else 0.0,
            currency=currency,
            fx_rate=float(fx_rate) if not is_jpy else 1.0,
            settlement_jpy=float(settlement_jpy),
            settlement_usd=float(settlement_usd),
            memo=memo,
        )


def _handle_submit(
    trade_type: str,
    symbol_raw: str,
    trade_date: datetime.date,
    shares_count: int,
    unit_price: float,
    currency: str,
    fx_rate: float,
    settlement_jpy: float,
    settlement_usd: float,
    memo: str,
) -> None:
    """Validate form inputs and persist the trade via trade_writer.

    Why: Separating validation+submission from widget rendering allows
         cleaner logic flow and makes the validation testable.
    How: Checks required fields, then calls record_trade(). On success,
         clears all st.cache_data entries and calls st.rerun(). On error,
         shows st.error() without rerunning so the user can correct input.
    """
    # --- Validation ---
    symbol: str = symbol_raw.strip()
    if not symbol:
        st.error("❌ 銘柄コードを入力してください。")
        return

    if shares_count < 1:
        st.error("❌ 株数は1以上で入力してください。")
        return

    if trade_type != "transfer" and unit_price < 0:
        st.error("❌ 単価は0以上で入力してください。")
        return

    trade_date_str: str = trade_date.strftime("%Y-%m-%d")

    # --- Persist ---
    try:
        success_message: str = record_trade(
            symbol=symbol,
            trade_type=trade_type,
            shares=shares_count,
            price=unit_price,
            currency=currency,
            trade_date=trade_date_str,
            memo=memo,
            fx_rate=fx_rate,
            settlement_jpy=settlement_jpy,
            settlement_usd=settlement_usd,
        )
    except (ValueError, RuntimeError) as exc:
        st.error(f"❌ {exc}")
        logger.warning("Trade recording failed: %s", exc)
        return
    except Exception as exc:
        st.error(f"❌ 予期しないエラーが発生しました: {exc}")
        logger.exception("Unexpected error in trade form submission: %s", exc)
        return

    # --- Success: clear all cached data and reload ---
    st.success(f"✅ {success_message}")
    st.cache_data.clear()
    st.rerun()
