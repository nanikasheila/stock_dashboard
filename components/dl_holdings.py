"""保有銘柄・取引履歴ドメイン — data_loader サブモジュール.

売買履歴から保有状況を再構築し、損益を計算するヘルパー群と
現在スナップショット取得関数を提供する。
``components.data_loader`` がこのモジュールを import して再エクスポートする。
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")

from src.core.common import is_cash
from src.core.portfolio.portfolio_manager import (
    DEFAULT_CSV_PATH,
    get_fx_rates,
    load_portfolio,
)
from src.core.ticker_utils import infer_currency
from src.data import yahoo_client
from src.data.history_store import load_history

# ---------------------------------------------------------------------------
# 0. 銘柄表示ラベル生成ユーティリティ
# ---------------------------------------------------------------------------

# 企業名から除去する法人格サフィックス（長い順にマッチ）
_CORPORATE_SUFFIXES = [
    ", Inc.",
    " Inc.",
    " Inc",
    " Corporation",
    " Corp.",
    " Corp",
    " Co., Ltd.",
    " Co.,Ltd.",
    " Co.",
    " Holdings",
    " Holding",
    " Group",
    " Limited",
    " Ltd.",
    " Ltd",
    " plc",
    " PLC",
    " SE",
    " N.V.",
    " S.A.",
    " AG",
    " SA",
    "株式会社",
    "（株）",
]


def _shorten_company_name(name: str, max_len: int = 8) -> str:
    """企業名を短縮して表示用にする.

    法人格サフィックスを除去し、長い名前は切り詰める。
    CJK 文字（漢字/ひらがな/カタカナ）が多い名前は文字数で切り詰め、
    英語名は最初の単語を使う。

    Examples
    --------
    >>> _shorten_company_name("トヨタ自動車株式会社", 6)
    'トヨタ自動車'
    >>> _shorten_company_name("Apple Inc.", 8)
    'Apple'
    >>> _shorten_company_name("Broadcom Inc.", 8)
    'Broadcom'
    """
    if not name:
        return ""

    cleaned = name
    for suffix in _CORPORATE_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip()
            break

    if not cleaned:
        cleaned = name

    # CJK 判定（漢字・ひらがな・カタカナの割合が 1/3 超）
    cjk_count = sum(
        1
        for c in cleaned
        if "\u4e00" <= c <= "\u9fff"  # 漢字
        or "\u3040" <= c <= "\u309f"  # ひらがな
        or "\u30a0" <= c <= "\u30ff"  # カタカナ
    )
    is_cjk = cjk_count > len(cleaned) / 3

    if is_cjk:
        return cleaned[:max_len] if len(cleaned) > max_len else cleaned

    # 英語: まずサフィックス除去後が収まるか確認
    if len(cleaned) <= max_len:
        return cleaned

    # 最初の単語が収まるならそれを使う
    first_word = cleaned.split()[0] if cleaned.split() else cleaned
    if len(first_word) <= max_len:
        return first_word

    return cleaned[:max_len]


def _build_symbol_labels(symbols: list[str]) -> dict[str, str]:
    """銘柄シンボルのリストから表示ラベルのマップを生成する.

    Yahoo Finance から取得済み（キャッシュ済み）の企業名を使い、
    ``短縮名(シンボル)`` 形式のラベルを返す。
    名前が取得できないシンボルはそのまま返す。

    Returns
    -------
    dict
        {raw_symbol: display_label}

    Examples
    --------
    >>> _build_symbol_labels(["7203.T"])  # doctest: +SKIP
    {"7203.T": "トヨタ(7203.T)"}
    """
    label_map: dict[str, str] = {}
    for symbol in symbols:
        try:
            info = yahoo_client.get_stock_info(symbol)
            name = info.get("name") if info else None
        except Exception as exc:
            logger.debug(
                "_build_symbol_labels: get_stock_info failed for %s: %s",
                symbol,
                exc,
            )
            name = None

        if name and name != symbol:
            short_name = _shorten_company_name(name)
            label_map[symbol] = f"{short_name}({symbol})"
        else:
            label_map[symbol] = symbol

    return label_map


# ---------------------------------------------------------------------------
# 1. 現在のスナップショット（銘柄別評価額）
# ---------------------------------------------------------------------------


def get_current_snapshot(
    csv_path: str = DEFAULT_CSV_PATH,
) -> dict:
    """現在の保有銘柄ごとの評価額を取得して dict で返す.

    Returns
    -------
    dict
        positions: list[dict]  各銘柄
        total_value_jpy: float
        fx_rates: dict
        as_of: str
    """
    positions = load_portfolio(csv_path)
    fx_rates = get_fx_rates(yahoo_client)

    result_positions: list[dict] = []
    total_value_jpy = 0.0

    for pos in positions:
        symbol = pos["symbol"]
        shares = pos["shares"]
        cost_price = pos["cost_price"]
        cost_currency = pos["cost_currency"]
        memo = pos.get("memo", "")
        purchase_date = pos.get("purchase_date", "")

        if is_cash(symbol):
            currency = symbol.replace(".CASH", "")
            rate = fx_rates.get(currency, 1.0)
            eval_jpy = shares * cost_price * rate
            result_positions.append(
                {
                    "symbol": symbol,
                    "name": memo or symbol,
                    "shares": shares,
                    "current_price": cost_price,
                    "currency": currency,
                    "evaluation_jpy": eval_jpy,
                    "cost_jpy": eval_jpy,  # Cash: 損益ゼロ
                    "pnl_jpy": 0,
                    "pnl_pct": 0,
                    "sector": "Cash",
                }
            )
            total_value_jpy += eval_jpy
            continue

        info = yahoo_client.get_stock_info(symbol)
        if not info:
            continue

        price = info.get("price", 0) or 0
        currency = info.get("currency") or infer_currency(symbol)
        rate = fx_rates.get(currency, 1.0)
        eval_jpy = shares * price * rate
        cost_rate = fx_rates.get(cost_currency, 1.0)
        cost_jpy = shares * cost_price * cost_rate

        result_positions.append(
            {
                "symbol": symbol,
                "name": info.get("name", memo or symbol),
                "shares": shares,
                "current_price": price,
                "currency": currency,
                "evaluation_jpy": eval_jpy,
                "cost_jpy": cost_jpy,
                "pnl_jpy": eval_jpy - cost_jpy,
                "pnl_pct": ((eval_jpy / cost_jpy) - 1) * 100 if cost_jpy else 0,
                "sector": info.get("sector", ""),
                "purchase_date": purchase_date,
            }
        )
        total_value_jpy += eval_jpy

    # 実現損益・含み損益の計算（総平均法）
    trades = _build_holdings_timeline()
    realized = _compute_realized_pnl(trades, fx_rates)
    pnl_ma = _compute_pnl_moving_average(trades, fx_rates, result_positions)

    return {
        "positions": result_positions,
        "total_value_jpy": total_value_jpy,
        "fx_rates": fx_rates,
        "realized_pnl": realized,
        "pnl_moving_avg": pnl_ma,
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 2. 売買履歴から時系列の保有状況を復元
# ---------------------------------------------------------------------------


def _build_holdings_timeline(
    base_dir: str | None = None,
) -> list[dict]:
    """trade 履歴を日時順にロードして返す."""
    trades = load_history("trade", base_dir=base_dir or _DEFAULT_HISTORY_DIR)
    # 取引日 (date) でソート。同一日は buy/transfer → sell の順に並べる
    _TRADE_TYPE_ORDER = {"transfer": 0, "buy": 1, "sell": 2}
    trades.sort(
        key=lambda t: (
            t.get("date", ""),
            _TRADE_TYPE_ORDER.get(t.get("trade_type", "buy"), 1),
        )
    )
    return trades


def _reconstruct_daily_holdings(
    trades: list[dict],
) -> dict[str, dict[str, int]]:
    """各取引日時点での全銘柄保有株数マップを返す.

    buy / transfer → 保有追加、sell → 保有削減。

    Returns
    -------
    dict
        date_str -> { symbol -> cumulative_shares }
    """
    cumulative: dict[str, int] = {}
    daily_snapshots: dict[str, dict[str, int]] = {}

    for trade in trades:
        symbol = trade["symbol"]
        shares = trade.get("shares", 0)
        trade_type = trade.get("trade_type", "buy")
        date_str = trade.get("date", "")

        if trade_type in ("buy", "transfer"):
            cumulative[symbol] = cumulative.get(symbol, 0) + shares
        elif trade_type == "sell":
            cumulative[symbol] = max(0, cumulative.get(symbol, 0) - shares)
            if cumulative[symbol] == 0:
                del cumulative[symbol]

        daily_snapshots[date_str] = dict(cumulative)

    return daily_snapshots


def _compute_invested_capital(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> dict[str, float]:
    """累積投資額(円換算)の推移を返す.

    Why: 投資額の推移を追跡することで月次サマリーの損益計算に使用する。
    How: buy/transfer → +投資額、sell → −売却額。
         各取引の受渡金額（settlement_jpy / settlement_usd * fx_rate）を
         優先的に使用し、フォールバックとして shares*price*fx_rate で計算。
         これにより取引時点の実際の円換算額に近い精度を実現する。

    Returns
    -------
    dict
        date_str -> cumulative_invested_jpy
    """
    cumulative = 0.0
    invested: dict[str, float] = {}

    for trade in trades:
        trade_type = trade.get("trade_type", "buy")
        date_str = trade.get("date", "")
        amount_jpy = _trade_cost_jpy(trade, fx_rates)

        if trade_type in ("buy", "transfer"):
            cumulative += amount_jpy
        elif trade_type == "sell":
            cumulative -= amount_jpy
        cumulative = max(0.0, cumulative)

        invested[date_str] = cumulative

    return invested


def _trade_cost_jpy(
    trade: dict,
    global_fx_rates: dict[str, float],
) -> float:
    """取引の約定金額をJPYで計算する.

    優先順位:
    1. settlement_jpy + settlement_usd * fx_rate (両方ある場合)
    2. settlement_jpy が正 → そのまま使用
    3. settlement_usd * fx_rate (取引日レート)
    4. shares * price * fx_rate (取引日レートで計算)
    5. shares * price * 現在のFXレート (フォールバック)
    """
    sjpy = trade.get("settlement_jpy", 0) or 0
    susd = trade.get("settlement_usd", 0) or 0
    fx = trade.get("fx_rate", 0) or 0

    if sjpy > 0 and susd > 0:
        # Mixed settlement (JPY + USD portions)
        return sjpy + susd * fx
    elif sjpy > 0:
        return sjpy
    elif susd > 0 and fx > 0:
        return susd * fx
    elif fx > 0:
        # FX rate available but no explicit settlement → use price * fx_rate
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        return shares * price * fx
    else:
        # Final fallback: use current FX rate
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        cur = trade.get("currency", "JPY")
        rate = global_fx_rates.get(cur, 1.0)
        return shares * price * rate


def _compute_realized_pnl(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> dict:
    """FIFO方式で実現損益を計算する（為替換算・株式分割対応版）.

    改善点:
    - 為替換算: CSVの受渡金額/為替レートを使い、取引時点のJPY換算で損益を算出
    - 株式分割: transfer(入庫)でprice=0の場合、既存ロットの単価を分割比率で調整
    - フォールバック: 旧形式のJSON（fx_rate/settlement未保存）は現在レートで近似

    Returns
    -------
    dict
        by_symbol: dict[str, float]  銘柄別実現損益(JPY)
        total_jpy: float  合計実現損益(JPY)
    """
    # FIFO: 銘柄ごとに購入ロットを管理
    # 各ロット: {"shares": float, "cost_jpy_per_share": float}
    lots: dict[str, list[dict]] = defaultdict(list)
    realized_by_symbol: dict[str, float] = defaultdict(float)

    for trade in trades:
        sym = trade.get("symbol", "")
        tt = trade.get("trade_type", "buy")
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)

        if is_cash(sym):
            continue

        if tt == "buy":
            total_jpy = _trade_cost_jpy(trade, fx_rates)
            cost_per_share = total_jpy / shares if shares > 0 else 0
            lots[sym].append(
                {
                    "shares": float(shares),
                    "cost_jpy_per_share": cost_per_share,
                }
            )

        elif tt == "transfer":
            if price <= 0 and lots[sym]:
                # Stock split: redistribute cost basis
                existing_shares = sum(lot["shares"] for lot in lots[sym])
                if existing_shares > 0:
                    split_ratio = (existing_shares + shares) / existing_shares
                    for lot in lots[sym]:
                        lot["cost_jpy_per_share"] /= split_ratio
                        lot["shares"] *= split_ratio
            elif price > 0:
                # Regular transfer with cost basis
                total_jpy = _trade_cost_jpy(trade, fx_rates)
                cost_per_share = total_jpy / shares if shares > 0 else 0
                lots[sym].append(
                    {
                        "shares": float(shares),
                        "cost_jpy_per_share": cost_per_share,
                    }
                )

        elif tt == "sell":
            total_jpy = _trade_cost_jpy(trade, fx_rates)
            proceeds_per_share = total_jpy / shares if shares > 0 else 0

            remaining = float(shares)
            while remaining > 0.5 and lots[sym]:
                lot = lots[sym][0]
                take = min(remaining, lot["shares"])
                pnl = take * (proceeds_per_share - lot["cost_jpy_per_share"])
                realized_by_symbol[sym] += pnl
                lot["shares"] -= take
                remaining -= take
                if lot["shares"] < 0.5:
                    lots[sym].pop(0)

    total = sum(realized_by_symbol.values())
    return {
        "by_symbol": dict(realized_by_symbol),
        "total_jpy": total,
    }


def _build_trade_activity(
    trades: list[dict],
    fx_rates: dict[str, float],
) -> pd.DataFrame:
    """月ごとの売買件数・金額をまとめた DataFrame を返す."""
    rows: list[dict] = []
    for trade in trades:
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        currency = trade.get("currency", "JPY")
        rate = fx_rates.get(currency, 1.0)
        amount = shares * price * rate
        tt = trade.get("trade_type", "buy")
        d = trade.get("date", "")
        if not d:
            continue
        month = d[:7]  # YYYY-MM
        rows.append(
            {
                "month": month,
                "trade_type": tt,
                "amount_jpy": amount,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    buy_df = (
        df[df["trade_type"].isin(["buy", "transfer"])]
        .groupby("month")
        .agg(
            buy_count=("amount_jpy", "count"),
            buy_amount=("amount_jpy", "sum"),
        )
    )
    sell_df = (
        df[df["trade_type"] == "sell"]
        .groupby("month")
        .agg(
            sell_count=("amount_jpy", "count"),
            sell_amount=("amount_jpy", "sum"),
        )
    )
    result = buy_df.join(sell_df, how="outer").fillna(0)
    result["net_flow"] = result["buy_amount"] - result["sell_amount"]
    result.index.name = None
    return result.sort_index()


def _compute_pnl_moving_average(
    trades: list[dict],
    fx_rates: dict[str, float],
    current_positions: list[dict],
) -> dict:
    """総平均法（移動平均法）で実現損益と含み損益を計算する.

    Why: 日本の証券会社（SBI証券等）は総平均法を使用しており、
         FIFO法では実現/含みの配分が証券会社の表示と一致しない。
         KPIの評価損益を証券会社の値と整合させるため、総平均法で計算する。
    How: 買い(buy/transfer)時に円建て平均取得単価を更新し、
         売り(sell)時に平均単価ベースで原価を算出して実現損益を計算。
         含み損益 = 現在評価額 − 残存保有の取得原価(総平均法ベース)。
         transfer(price=0)は株式分割として扱い、単価を按分調整する。

    Parameters
    ----------
    trades : list[dict]
        日付順にソートされた取引履歴。
    fx_rates : dict[str, float]
        現在のFXレート（フォールバック用）。
    current_positions : list[dict]
        get_current_snapshot で構築された銘柄リスト（evaluation_jpy を含む）。

    Returns
    -------
    dict
        realized_by_symbol   : dict[str, float]  銘柄別実現損益(JPY)
        realized_total_jpy   : float              合計実現損益(JPY)
        unrealized_by_symbol : dict[str, float]   銘柄別含み損益(JPY)
        unrealized_total_jpy : float              合計含み損益(JPY)
        cost_basis           : dict[str, float]   残存ポジションの取得原価(JPY)
    """
    # 銘柄ごとの累積保有株数と累積取得原価(JPY)
    holding_shares: dict[str, float] = defaultdict(float)
    holding_cost: dict[str, float] = defaultdict(float)
    realized_by_symbol: dict[str, float] = defaultdict(float)

    for trade in trades:
        sym = trade.get("symbol", "")
        trade_type = trade.get("trade_type", "buy")
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)

        if is_cash(sym):
            continue

        if trade_type in ("buy", "transfer"):
            if trade_type == "transfer" and price <= 0 and holding_shares[sym] > 0:
                # Why: price=0 の transfer は株式分割を意味する。
                # How: 既存の取得原価はそのまま維持し、株数だけ増やす。
                #      結果として1株あたり平均単価が按分される。
                holding_shares[sym] += shares
            else:
                cost_jpy = _trade_cost_jpy(trade, fx_rates)
                holding_shares[sym] += shares
                holding_cost[sym] += cost_jpy

        elif trade_type == "sell":
            proceeds_jpy = _trade_cost_jpy(trade, fx_rates)
            old_shares = holding_shares[sym]

            if old_shares > 0:
                avg_cost_per_share = holding_cost[sym] / old_shares
                cost_of_sold = avg_cost_per_share * shares
                realized_by_symbol[sym] += proceeds_jpy - cost_of_sold
                holding_shares[sym] = old_shares - shares
                holding_cost[sym] -= cost_of_sold
            else:
                # Why: 履歴不完全で売りが先に来るケース。
                realized_by_symbol[sym] += proceeds_jpy

    # 含み損益: 現在の評価額 − 総平均法での残存取得原価
    unrealized_by_symbol: dict[str, float] = {}
    for pos in current_positions:
        sym = pos["symbol"]
        if is_cash(sym):
            continue
        market_value = pos.get("evaluation_jpy", 0)
        remaining_cost = holding_cost.get(sym, 0)
        unrealized_by_symbol[sym] = market_value - remaining_cost

    realized_total = sum(realized_by_symbol.values())
    unrealized_total = sum(unrealized_by_symbol.values())

    return {
        "realized_by_symbol": dict(realized_by_symbol),
        "realized_total_jpy": realized_total,
        "unrealized_by_symbol": unrealized_by_symbol,
        "unrealized_total_jpy": unrealized_total,
        "cost_basis": {sym: holding_cost[sym] for sym in holding_shares if holding_shares[sym] > 0.5},
    }
