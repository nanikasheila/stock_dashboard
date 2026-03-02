"""ポートフォリオダッシュボード — データローダー.

既存の portfolio_manager / history_store / yahoo_client を活用して
ダッシュボード表示用のデータを組み立てる。

取引履歴（JSON/CSVインポート）から各日の保有状況を復元し、
株価履歴と掛け合わせて資産推移を構築する。
"""

from __future__ import annotations

import logging
import sys
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")
# Why: Cache path is centralized in src.core.paths to avoid hardcoded duplication.
from src.core.paths import PRICE_CACHE_DIR as _PRICE_CACHE_DIR

_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours

from src.core.common import is_cash
from src.core.health_check import (
    ALERT_CAUTION,
    ALERT_EXIT,
    ALERT_NONE,
    check_change_quality,
    check_long_term_suitability,
    check_trend_health,
    compute_alert_level,
)
from src.core.portfolio.portfolio_manager import (
    DEFAULT_CSV_PATH,
    get_fx_rates,
    load_portfolio,
)
from src.core.screening.indicators import (
    assess_return_stability,
    calculate_shareholder_return,
    calculate_shareholder_return_history,
)
from src.core.ticker_utils import infer_currency
from src.core.value_trap import detect_value_trap
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
        except Exception:
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


# ---------------------------------------------------------------------------
# 2-b. 総平均法による損益計算
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. 資産推移データの構築
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 期間 → yfinance period / start の変換
# ---------------------------------------------------------------------------

_PERIOD_MAP: dict[str, str | None] = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "1y": "1y",
    "2y": "2y",
    "3y": "3y",
    "5y": "5y",
    "max": "max",
    "all": "max",
}


def _fetch_price_history(
    symbol: str,
    period: str,
) -> pd.DataFrame | None:
    """期間指定に応じた株価履歴を取得する (個別フォールバック用)."""
    yf_period = _PERIOD_MAP.get(period, period)
    hist = yahoo_client.get_price_history(symbol, period=yf_period)
    if hist is not None and not hist.empty:
        return hist[["Close"]].rename(columns={"Close": symbol})
    return None


# ---------------------------------------------------------------------------
# 価格キャッシュ (ディスク + バッチ取得)
# ---------------------------------------------------------------------------


def _get_cache_path(period: str) -> Path:
    """期間ごとのキャッシュファイルパスを返す."""
    safe = period.replace("/", "_")
    return _PRICE_CACHE_DIR / f"close_{safe}.csv"


def _load_cached_prices(period: str) -> pd.DataFrame | None:
    """ディスクキャッシュから株価を読み込む. TTL 超過時は None."""
    path = _get_cache_path(period)
    if not path.exists():
        return None
    age = _time.time() - path.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def _save_prices_cache(prices: pd.DataFrame, period: str) -> None:
    """株価をディスクキャッシュに保存."""
    try:
        _PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(_get_cache_path(period))
    except Exception as e:
        print(f"[data_loader] Cache save error: {e}")


def clear_price_cache() -> int:
    """Delete all disk-cached price CSV files.

    Why: Manual refresh should bypass all cache layers (Streamlit + disk).
         TTL-based expiry only applies to the automatic refresh cycle;
         a user-initiated refresh must ignore it entirely.
    How: Delete all CSV files in the price cache directory so that the
         next data load triggers a fresh fetch from Yahoo Finance.
         Returns the count of deleted files for logging/display.
    """
    deleted = 0
    if _PRICE_CACHE_DIR.exists():
        for csv_file in _PRICE_CACHE_DIR.glob("*.csv"):
            try:
                csv_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning("Cache delete error for %s: %s", csv_file, e)
    return deleted


def _load_prices(symbols: list[str], period: str) -> pd.DataFrame:
    """キャッシュ優先で全銘柄の終値を一括取得.

    1. ディスクキャッシュが有効 (TTL 4h) → 即座に返す
    2. キャッシュに不足銘柄 → 不足分のみバッチ取得してマージ
    3. キャッシュなし → 全銘柄をバッチ取得 (yf.download 1 回)
    """
    yf_period = _PERIOD_MAP.get(period, period)

    # --- キャッシュヒット ---
    cached = _load_cached_prices(period)
    if cached is not None:
        missing = [s for s in symbols if s not in cached.columns]
        if not missing:
            available = [s for s in symbols if s in cached.columns]
            print(f"[data_loader] Cache hit: {period} ({len(available)} symbols)")
            return cached[available]
        # 不足銘柄のみ追加取得
        print(f"[data_loader] Cache partial: fetching {len(missing)} symbols")
        new_prices = yahoo_client.get_close_prices_batch(missing, period=yf_period)
        if new_prices is not None and not new_prices.empty:
            new_prices.index = pd.to_datetime(new_prices.index).tz_localize(None)
            merged = pd.concat([cached, new_prices], axis=1)
            _save_prices_cache(merged, period)
            available = [s for s in symbols if s in merged.columns]
            if available:
                return merged[available].sort_index().ffill()
        available = [s for s in symbols if s in cached.columns]
        return cached[available] if available else pd.DataFrame()

    # --- キャッシュミス: バッチ取得 ---
    print(f"[data_loader] Cache miss: batch-fetching {len(symbols)} symbols")
    prices = yahoo_client.get_close_prices_batch(symbols, period=yf_period)
    if prices is None or prices.empty:
        # フォールバック: 個別取得
        print("[data_loader] Batch failed, falling back to individual fetches")
        frames: dict[str, pd.Series] = {}
        for sym in symbols:
            hist = _fetch_price_history(sym, period)
            if hist is not None and sym in hist.columns:
                frames[sym] = hist[sym]
        if not frames:
            return pd.DataFrame()
        prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().ffill()
    _save_prices_cache(prices, period)
    return prices


def build_portfolio_history(
    csv_path: str = DEFAULT_CSV_PATH,
    base_dir: str = _DEFAULT_HISTORY_DIR,
    period: str = "3mo",
) -> pd.DataFrame:
    """保有銘柄の日次評価額推移を DataFrame で返す.

    売買履歴から各日の保有銘柄・株数を復元し、
    yfinance の価格履歴と掛け合わせて日次評価額を算出する。

    Parameters
    ----------
    csv_path : str
        portfolio.csv のパス
    base_dir : str | None
        history_store のベースディレクトリ
    period : str
        株価取得期間 ("1mo" .. "5y", "max", "all")

    Returns
    -------
    pd.DataFrame
        index=Date, columns=銘柄シンボル, values=円換算評価額
        + "total" カラム + "invested" カラム
    """
    trades = _build_holdings_timeline(base_dir)
    if not trades:
        return pd.DataFrame()

    daily_snapshots = _reconstruct_daily_holdings(trades)

    # 現在保有中の銘柄（直近スナップショットから）＋過去に保有していた銘柄
    all_symbols: set[str] = set()
    for snap in daily_snapshots.values():
        all_symbols.update(snap.keys())

    # CASH 銘柄は除外（為替推移のみでの表示は別途対応可能）
    stock_symbols = sorted(s for s in all_symbols if not is_cash(s))

    if not stock_symbols:
        return pd.DataFrame()

    # 為替レート取得
    fx_rates = get_fx_rates(yahoo_client)

    # 全銘柄の終値を一括取得（ディスクキャッシュ + バッチ取得）
    price_df = _load_prices(stock_symbols, period)
    if price_df.empty:
        return pd.DataFrame()

    # 売買日 → 保有株数のマッピング — 日次に展開
    dates = price_df.index
    first_trade_date = pd.Timestamp(trades[0].get("date", ""))

    # 各日の保有株数を computed
    sorted_trade_dates = sorted(daily_snapshots.keys())

    def get_holdings_at(ts: pd.Timestamp) -> dict[str, int]:
        """指定日時点の保有株数を返す."""
        result: dict[str, int] = {}
        for td in sorted_trade_dates:
            if pd.Timestamp(td) <= ts:
                result = daily_snapshots[td]
            else:
                break
        return result

    # 日次評価額・保有株数の計算
    eval_data: dict[str, list[float]] = {s: [] for s in stock_symbols}
    shares_data: dict[str, list[int]] = {s: [] for s in stock_symbols}

    for dt in dates:
        holdings = get_holdings_at(dt)
        for symbol in stock_symbols:
            shares = holdings.get(symbol, 0)
            shares_data[symbol].append(shares)
            if shares > 0 and symbol in price_df.columns:
                price_val = price_df.loc[dt, symbol]
                if pd.notna(price_val):
                    currency = infer_currency(symbol)
                    rate = fx_rates.get(currency, 1.0)
                    eval_data[symbol].append(shares * price_val * rate)
                else:
                    eval_data[symbol].append(0.0)
            else:
                eval_data[symbol].append(0.0)

    result_df = pd.DataFrame(eval_data, index=dates)

    # 取引開始前のデータを除外
    if not first_trade_date or pd.isna(first_trade_date):
        first_trade_date = dates[0]
    result_df = result_df[result_df.index >= first_trade_date]

    # 全期間ゼロの銘柄列を除外（既に売却済みで表示期間に保有がない銘柄）
    symbol_cols = [c for c in result_df.columns if c not in ("total", "invested")]
    zero_cols = [c for c in symbol_cols if (result_df[c] == 0).all()]
    if zero_cols:
        result_df = result_df.drop(columns=zero_cols)

    # 合計列（株式のみ）
    result_df["total"] = result_df.sum(axis=1)

    # 現金ポジションを合計に加算
    # Why: KPI のトータル資産と月次サマリーの月末評価額を一致させるため、
    #      現金も資産合計に含める。
    # How: portfolio.csv の現金ポジション (*.CASH) を FX 換算し定額加算する。
    #      過去日付には厳密でないが、直近値の整合性を優先する。
    _portfolio_for_cash = load_portfolio(csv_path)
    _cash_total_jpy = 0.0
    for _pos in _portfolio_for_cash:
        if is_cash(_pos["symbol"]):
            _cash_cur = _pos["symbol"].replace(".CASH", "")
            _cash_rate = fx_rates.get(_cash_cur, 1.0)
            _cash_total_jpy += _pos["shares"] * _pos["cost_price"] * _cash_rate
    if _cash_total_jpy > 0:
        result_df["total"] += _cash_total_jpy

    # 累積投資額列の追加
    invested_map = _compute_invested_capital(trades, fx_rates)
    invested_series: list[float] = []
    sorted_inv_dates = sorted(invested_map.keys())
    for dt in result_df.index:
        inv_val = 0.0
        for inv_d in sorted_inv_dates:
            if pd.Timestamp(inv_d) <= dt:
                inv_val = invested_map[inv_d]
            else:
                break
        invested_series.append(inv_val)
    result_df["invested"] = invested_series

    # 銘柄列を表示用ラベルに変換（"7203.T" → "トヨタ(7203.T)" 等）
    stock_cols = [c for c in result_df.columns if c not in ("total", "invested")]
    if stock_cols:
        label_map = _build_symbol_labels(stock_cols)
        result_df = result_df.rename(columns=label_map)

        # Why: compute_top_worst_performers compares day-over-day evaluation
        #      amounts. When shares change (buy/sell), the evaluation jump
        #      includes both price movement and position-size change.
        # How: Store a parallel DataFrame of share counts with matching
        #      index and column labels.  Downstream consumers divide
        #      evaluation by shares to isolate pure price returns.
        _shares_raw = pd.DataFrame(shares_data, index=dates)
        _shares_raw = _shares_raw[_shares_raw.index >= first_trade_date]
        if zero_cols:
            _sz = [c for c in zero_cols if c in _shares_raw.columns]
            if _sz:
                _shares_raw = _shares_raw.drop(columns=_sz)
        _shares_raw = _shares_raw.rename(columns=label_map)
        # Why: Streamlit serializes DataFrame.attrs to JSON.
        #      Storing a DataFrame in attrs causes a UserWarning.
        # How: Store index + data as plain dicts (JSON-serializable)
        #      and reconstruct the DataFrame on the consumer side.
        result_df.attrs["_shares_index"] = _shares_raw.index.strftime("%Y-%m-%d").tolist()
        result_df.attrs["_shares_data"] = _shares_raw.to_dict(orient="list")

    return result_df


# ---------------------------------------------------------------------------
# 4. セクター別集計
# ---------------------------------------------------------------------------


def get_sector_breakdown(snapshot: dict) -> pd.DataFrame:
    """スナップショットからセクター別評価額を集計."""
    rows = []
    for p in snapshot["positions"]:
        rows.append(
            {
                "sector": p.get("sector") or "Unknown",
                "evaluation_jpy": p.get("evaluation_jpy", 0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("sector")["evaluation_jpy"].sum().reset_index()


# ---------------------------------------------------------------------------
# 5. 月次集計
# ---------------------------------------------------------------------------


def get_monthly_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    """日次データから月末の total を抽出して月次テーブルを返す."""
    if history_df.empty:
        return pd.DataFrame()

    cols = ["total"]
    if "invested" in history_df.columns:
        cols.append("invested")

    monthly = history_df[cols].resample("ME").last()
    monthly.index = monthly.index.strftime("%Y-%m")
    rename = {"total": "month_end_value_jpy"}
    if "invested" in monthly.columns:
        rename["invested"] = "invested_jpy"
    monthly = monthly.rename(columns=rename)

    # 月次変動率
    monthly["change_pct"] = monthly["month_end_value_jpy"].pct_change() * 100

    # 前年同月比 (YoY)
    monthly["yoy_pct"] = monthly["month_end_value_jpy"].pct_change(periods=12) * 100

    # 含み損益
    if "invested_jpy" in monthly.columns:
        monthly["unrealized_pnl"] = monthly["month_end_value_jpy"] - monthly["invested_jpy"]

    return monthly


def get_trade_activity(
    base_dir: str = _DEFAULT_HISTORY_DIR,
) -> pd.DataFrame:
    """月ごとの売買件数・金額を返す."""
    trades = _build_holdings_timeline(base_dir)
    if not trades:
        return pd.DataFrame()
    fx_rates = get_fx_rates(yahoo_client)
    return _build_trade_activity(trades, fx_rates)


# ---------------------------------------------------------------------------
# 6. 資産推定推移（楽観/ベース/悲観）
# ---------------------------------------------------------------------------


def build_projection(
    current_value: float,
    years: int = 5,
    optimistic_rate: float | None = None,
    base_rate: float | None = None,
    pessimistic_rate: float | None = None,
    csv_path: str = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """現在の総資産から楽観/ベース/悲観の将来推定推移を生成する.

    Parameters
    ----------
    current_value : float
        現在の総資産（円）。
    years : int
        何年先まで推定するか（デフォルト5年）。
    optimistic_rate, base_rate, pessimistic_rate : float | None
        年率リターン（0.10 = 10%）。None の場合は estimate_portfolio_return から取得。
    csv_path : str
        ポートフォリオCSVパス。

    Returns
    -------
    pd.DataFrame
        index=日付, columns=[optimistic, base, pessimistic]
    """
    # リターン推定値が未指定の場合、ポートフォリオから推定
    if base_rate is None:
        try:
            from src.core.return_estimate import estimate_portfolio_return

            result = estimate_portfolio_return(csv_path, yahoo_client)
            pf = result.get("portfolio", {})
            optimistic_rate = pf.get("optimistic") or 0.15
            base_rate = pf.get("base") or 0.08
            pessimistic_rate = pf.get("pessimistic") or -0.05
        except Exception:
            optimistic_rate = 0.15
            base_rate = 0.08
            pessimistic_rate = -0.05

    if optimistic_rate is None:
        optimistic_rate = 0.15
    if pessimistic_rate is None:
        pessimistic_rate = -0.05

    # 月次ポイントで推定（years * 12 + 1 点）
    today = pd.Timestamp.now().normalize()
    months = years * 12
    dates = pd.date_range(start=today, periods=months + 1, freq="ME")
    # 先頭を今日にする
    dates = dates.insert(0, today)

    rows = []
    for d in dates:
        t_years = (d - today).days / 365.25
        rows.append(
            {
                "date": d,
                "optimistic": current_value * (1 + optimistic_rate) ** t_years,
                "base": current_value * (1 + base_rate) ** t_years,
                "pessimistic": current_value * (1 + pessimistic_rate) ** t_years,
            }
        )

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# 7. リスク指標の算出
# ---------------------------------------------------------------------------

import numpy as np


def compute_risk_metrics(history_df: pd.DataFrame) -> dict:
    """日次の資産推移からリスク指標を算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    dict
        sharpe_ratio: float   年率シャープレシオ（リスクフリーレート0.5%想定）
        max_drawdown_pct: float  最大ドローダウン（%、マイナス値）
        annual_volatility_pct: float  年率ボラティリティ（%）
        annual_return_pct: float  年率リターン（%）
        calmar_ratio: float  カルマーレシオ（年率リターン / |MDD|）
    """
    if history_df.empty or "total" not in history_df.columns:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    total = history_df["total"].dropna()
    if len(total) < 2:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    # 日次リターン
    daily_returns = total.pct_change().dropna()
    if daily_returns.empty:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annual_volatility_pct": 0.0,
            "annual_return_pct": 0.0,
            "calmar_ratio": 0.0,
        }

    trading_days = 252
    risk_free_rate = 0.005  # 0.5%

    # 年率リターン
    total_days = (total.index[-1] - total.index[0]).days
    if total_days <= 0:
        total_days = 1
    total_return = total.iloc[-1] / total.iloc[0] - 1
    annual_return = (1 + total_return) ** (365.25 / total_days) - 1

    # 年率ボラティリティ
    annual_vol = float(daily_returns.std() * np.sqrt(trading_days))

    # シャープレシオ
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # 最大ドローダウン
    cummax = total.cummax()
    drawdown = (total - cummax) / cummax
    max_dd = float(drawdown.min()) * 100  # パーセント

    # カルマーレシオ
    calmar = annual_return / abs(max_dd / 100) if max_dd != 0 else 0.0

    return {
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_dd, 1),
        "annual_volatility_pct": round(annual_vol * 100, 1),
        "annual_return_pct": round(annual_return * 100, 1),
        "calmar_ratio": round(float(calmar), 2),
    }


# ---------------------------------------------------------------------------
# 8. Top/Worst パフォーマー
# ---------------------------------------------------------------------------


def compute_top_worst_performers(
    history_df: pd.DataFrame,
    top_n: int = 3,
) -> dict:
    """直近1日の銘柄別騰落率ランキングを返す.

    Why: ユーザーがポートフォリオ内でどの銘柄が当日最も動いたかを
         一目で把握するため。
    How: 評価額を保有株数で割って1株当たりの評価額（= 株価×為替）を
         求め、前日比を純粋な価格変動率として算出する。
         株数データが attrs["_shares_df"] に存在しない場合は
         評価額ベースにフォールバックする。

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力
    top_n : int
        上位/下位何銘柄を返すか

    Returns
    -------
    dict
        top: list[dict]  (symbol, change_pct, change_jpy)
        worst: list[dict]
    """
    if history_df.empty or len(history_df) < 2:
        return {"top": [], "worst": []}

    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
    if not stock_cols:
        return {"top": [], "worst": []}

    latest = history_df.iloc[-1]
    previous = history_df.iloc[-2]

    # Why: 評価額 = 株数 × 単価 × 為替 のため、株数変動があると
    #      前日比が株価変動ではなくポジション変動を反映してしまう。
    # How: _shares_data / _shares_index があれば DataFrame を復元し
    #      株数で割って正規化し、純粋な価格騰落率を算出。
    _s_data = history_df.attrs.get("_shares_data")
    _s_index = history_df.attrs.get("_shares_index")
    shares_df: pd.DataFrame | None = None
    if _s_data and _s_index:
        shares_df = pd.DataFrame(_s_data, index=pd.to_datetime(_s_index))
    if shares_df is not None and len(shares_df) >= 2:
        latest_shares = shares_df.iloc[-1]
        prev_shares = shares_df.iloc[-2]
    else:
        latest_shares = None
        prev_shares = None

    performers = []
    for col in stock_cols:
        cur = float(latest.get(col, 0))
        prev = float(previous.get(col, 0))
        if prev > 0 and cur > 0:
            if latest_shares is not None and prev_shares is not None:
                cur_sh = float(latest_shares.get(col, 0))
                prev_sh = float(prev_shares.get(col, 0))
                if cur_sh > 0 and prev_sh > 0:
                    # 1株当たり評価額の前日比 = 純粋な価格騰落率
                    pct = ((cur / cur_sh) / (prev / prev_sh) - 1) * 100
                else:
                    pct = (cur / prev - 1) * 100
            else:
                pct = (cur / prev - 1) * 100
            change_jpy = cur - prev
            performers.append(
                {
                    "symbol": col,
                    "change_pct": round(pct, 2),
                    "change_jpy": round(change_jpy, 0),
                }
            )

    performers.sort(key=lambda x: x["change_pct"], reverse=True)

    actual_n = min(top_n, len(performers))
    return {
        "top": performers[:actual_n],
        "worst": performers[-actual_n:][::-1] if actual_n > 0 else [],
    }


# ---------------------------------------------------------------------------
# 9. 前日比計算
# ---------------------------------------------------------------------------


def compute_daily_change(history_df: pd.DataFrame) -> dict:
    """直近の前日比（金額・パーセント）を算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    dict
        daily_change_jpy: float  前日比（円）
        daily_change_pct: float  前日比（%）
    """
    if history_df.empty or "total" not in history_df.columns:
        return {"daily_change_jpy": 0.0, "daily_change_pct": 0.0}

    total = history_df["total"].dropna()
    if len(total) < 2:
        return {"daily_change_jpy": 0.0, "daily_change_pct": 0.0}

    latest = float(total.iloc[-1])
    previous = float(total.iloc[-2])
    change = latest - previous
    pct = (change / previous * 100) if previous != 0 else 0.0

    return {
        "daily_change_jpy": round(change, 0),
        "daily_change_pct": round(pct, 2),
    }


# ---------------------------------------------------------------------------
# 9. ベンチマーク超過リターン
# ---------------------------------------------------------------------------


def compute_benchmark_excess(
    history_df: pd.DataFrame,
    benchmark_series: pd.Series | None,
) -> dict | None:
    """ポートフォリオのベンチマーク超過リターンを算出する.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。
    benchmark_series : pd.Series | None
        get_benchmark_series() の出力（正規化済み）

    Returns
    -------
    dict | None
        portfolio_return_pct: float
        benchmark_return_pct: float
        excess_return_pct: float
    """
    if benchmark_series is None or history_df.empty or "total" not in history_df.columns:
        return None

    total = history_df["total"].dropna()
    bench = benchmark_series.dropna()
    if len(total) < 2 or len(bench) < 2:
        return None

    pf_return = (float(total.iloc[-1]) / float(total.iloc[0]) - 1) * 100
    bm_return = (float(bench.iloc[-1]) / float(bench.iloc[0]) - 1) * 100
    excess = pf_return - bm_return

    return {
        "portfolio_return_pct": round(pf_return, 2),
        "benchmark_return_pct": round(bm_return, 2),
        "excess_return_pct": round(excess, 2),
    }


# ---------------------------------------------------------------------------
# 10. ベンチマークデータ取得
# ---------------------------------------------------------------------------


def get_benchmark_series(
    symbol: str,
    history_df: pd.DataFrame,
    period: str = "3mo",
) -> pd.Series | None:
    """ベンチマーク銘柄の終値を取得し、PF の total 列と同じ基準に正規化する.

    PF 開始日の total 値を基準に、ベンチマークの相対パフォーマンスを
    同じ円スケールに変換して返す。

    Parameters
    ----------
    symbol : str
        ベンチマークのティッカー (e.g. "SPY", "^N225")
    history_df : pd.DataFrame
        build_portfolio_history() の出力
    period : str
        価格取得期間

    Returns
    -------
    pd.Series | None
        index=Date, values=正規化された評価額（PFと同スケール）
    """
    if history_df.empty or "total" not in history_df.columns:
        return None

    prices = _load_prices([symbol], period)
    if prices.empty or symbol not in prices.columns:
        return None

    bench = prices[symbol].dropna()
    if bench.empty:
        return None

    # PF の日付範囲に合わせる
    pf_start = history_df.index[0]
    bench = bench[bench.index >= pf_start]
    if bench.empty:
        return None

    # PF 開始日の total を基準に正規化
    pf_start_value = history_df["total"].iloc[0]
    bench_start_value = bench.iloc[0]
    if bench_start_value == 0:
        return None

    normalized = bench / bench_start_value * pf_start_value
    normalized.name = symbol
    return normalized


# ---------------------------------------------------------------------------
# 12. ドローダウン系列
# ---------------------------------------------------------------------------


def compute_drawdown_series(history_df: pd.DataFrame) -> pd.Series:
    """日次のドローダウン（ピークからの下落率 %）系列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。

    Returns
    -------
    pd.Series
        index=Date, values=ドローダウン（%、0以下の値）
    """
    if history_df.empty or "total" not in history_df.columns:
        return pd.Series(dtype=float)

    total = history_df["total"].dropna()
    if len(total) < 2:
        return pd.Series(dtype=float)

    cummax = total.cummax()
    drawdown = (total - cummax) / cummax * 100
    return drawdown


# ---------------------------------------------------------------------------
# 13. ローリングSharpe比系列
# ---------------------------------------------------------------------------


def compute_rolling_sharpe(
    history_df: pd.DataFrame,
    window: int = 60,
    risk_free_rate: float = 0.005,
) -> pd.Series:
    """ローリングSharpe比の系列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。"total" 列が必須。
    window : int
        ローリングウィンドウ（営業日数）
    risk_free_rate : float
        年率リスクフリーレート

    Returns
    -------
    pd.Series
        index=Date, values=ローリングSharpe比（年率換算）
    """
    if history_df.empty or "total" not in history_df.columns:
        return pd.Series(dtype=float)

    total = history_df["total"].dropna()
    if len(total) < window + 1:
        return pd.Series(dtype=float)

    daily_returns = total.pct_change().dropna()
    trading_days = 252
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1

    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()

    rolling_sharpe = (rolling_mean - daily_rf) / rolling_std * np.sqrt(trading_days)
    return rolling_sharpe.dropna()


# ---------------------------------------------------------------------------
# 14. 銘柄間相関行列
# ---------------------------------------------------------------------------


def compute_correlation_matrix(
    history_df: pd.DataFrame,
    min_periods: int = 20,
) -> pd.DataFrame:
    """保有銘柄間の日次リターン相関行列を返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力。銘柄ごとの列を含む。
    min_periods : int
        相関計算に必要な最低データ点数。

    Returns
    -------
    pd.DataFrame
        銘柄×銘柄の相関行列。銘柄が2つ未満の場合は空DataFrame。
    """
    if history_df.empty:
        return pd.DataFrame()

    # "total" と "invested" を除いた銘柄列のみ
    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
    if len(stock_cols) < 2:
        return pd.DataFrame()

    stock_df = history_df[stock_cols].dropna(how="all")
    if len(stock_df) < min_periods:
        return pd.DataFrame()

    # 日次リターンを計算
    daily_returns = stock_df.pct_change().dropna(how="all")

    # 相関行列
    corr = daily_returns.corr(min_periods=min_periods)
    return corr


# ---------------------------------------------------------------------------
# 15. ウェイトドリフト判定
# ---------------------------------------------------------------------------


def compute_weight_drift(
    positions: list[dict],
    total_value_jpy: float,
    target_weights: dict[str, float] | None = None,
    threshold_pct: float = 5.0,
) -> list[dict]:
    """各銘柄の現在ウェイトと目標ウェイトの乖離を計算し、閾値超過を返す.

    Parameters
    ----------
    positions : list[dict]
        get_current_snapshot()["positions"]
    total_value_jpy : float
        ポートフォリオ総額(円)
    target_weights : dict[str, float] | None
        銘柄シンボル→目標ウェイト(%)のマップ。
        None の場合は均等ウェイト（= 100 / 銘柄数）を適用。
    threshold_pct : float
        乖離警告の閾値(ポイント)。デフォルト5.0pp。

    Returns
    -------
    list[dict]
        乖離が閾値を超えた銘柄のリスト。各要素:
        - symbol: str
        - name: str
        - current_pct: float  (現在ウェイト%)
        - target_pct: float   (目標ウェイト%)
        - drift_pct: float    (乖離幅pp, 正=オーバーウェイト)
        - status: str         ("overweight" | "underweight")
    """
    if not positions or total_value_jpy <= 0:
        return []

    # Cash を除外した銘柄のみ対象
    stock_positions = [p for p in positions if p.get("sector") != "Cash"]
    if not stock_positions:
        return []

    n = len(stock_positions)
    equal_weight = 100.0 / n if n > 0 else 0

    results = []
    for p in stock_positions:
        symbol = p["symbol"]
        current_pct = p["evaluation_jpy"] / total_value_jpy * 100

        if target_weights and symbol in target_weights:
            target_pct = target_weights[symbol]
        else:
            target_pct = equal_weight

        drift = current_pct - target_pct

        if abs(drift) >= threshold_pct:
            results.append(
                {
                    "symbol": symbol,
                    "name": p.get("name", symbol),
                    "current_pct": round(current_pct, 1),
                    "target_pct": round(target_pct, 1),
                    "drift_pct": round(drift, 1),
                    "status": "overweight" if drift > 0 else "underweight",
                }
            )

    # 乖離の大きい順にソート
    results.sort(key=lambda x: abs(x["drift_pct"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# ヘルスチェック（ダッシュボード用）
# ---------------------------------------------------------------------------


def run_dashboard_health_check(
    csv_path: str = DEFAULT_CSV_PATH,
) -> dict:
    """ポートフォリオ全銘柄のヘルスチェックを実行する.

    既存の health_check.py のロジックを呼び出し、
    ダッシュボード表示用に結果を整形する。

    Returns
    -------
    dict
        positions: list[dict]  各銘柄のヘルスチェック結果
        alerts: list[dict]     アラートのある銘柄のみ
        sell_alerts: list[dict] 売りタイミング通知
        summary: dict          サマリー統計
    """
    positions = load_portfolio(csv_path)

    empty_summary = {
        "total": 0,
        "healthy": 0,
        "early_warning": 0,
        "caution": 0,
        "exit": 0,
    }

    if not positions:
        return {
            "positions": [],
            "alerts": [],
            "sell_alerts": [],
            "summary": empty_summary,
        }

    results: list[dict] = []
    alerts: list[dict] = []
    counts = {"healthy": 0, "early_warning": 0, "caution": 0, "exit": 0}

    for pos in positions:
        symbol = pos["symbol"]

        # Skip cash positions
        if is_cash(symbol):
            continue

        # 1. Trend analysis (1y price history)
        hist = yahoo_client.get_price_history(symbol, period="1y")
        trend_health = check_trend_health(hist)

        # 2. Change quality (alpha signal)
        stock_detail = yahoo_client.get_stock_detail(symbol)
        if stock_detail is None:
            stock_detail = {}
        change_quality = check_change_quality(stock_detail)

        # 3. Shareholder return stability
        sh_return = calculate_shareholder_return(stock_detail)
        sh_history = calculate_shareholder_return_history(stock_detail)
        sh_stability = assess_return_stability(sh_history)

        # 4. Alert level
        alert = compute_alert_level(
            trend_health,
            change_quality,
            stock_detail=stock_detail,
            return_stability=sh_stability,
        )

        # 5. Long-term suitability
        long_term = check_long_term_suitability(
            stock_detail,
            shareholder_return_data=sh_return,
        )

        # 6. Value trap detection
        value_trap = detect_value_trap(stock_detail)

        # PnL from portfolio
        shares = pos["shares"]
        cost_price = pos["cost_price"]
        current_price = trend_health.get("current_price", 0)
        if current_price and cost_price:
            pnl_pct = ((current_price / cost_price) - 1) * 100
        else:
            pnl_pct = 0

        result = {
            "symbol": symbol,
            "name": stock_detail.get("name") or pos.get("memo") or symbol,
            "shares": shares,
            "cost_price": cost_price,
            "current_price": current_price,
            "pnl_pct": round(pnl_pct, 2),
            "trend": trend_health.get("trend", "不明"),
            "rsi": trend_health.get("rsi", float("nan")),
            "sma50": trend_health.get("sma50", 0),
            "sma200": trend_health.get("sma200", 0),
            "price_above_sma50": trend_health.get("price_above_sma50", False),
            "price_above_sma200": trend_health.get("price_above_sma200", False),
            "cross_signal": trend_health.get("cross_signal", "none"),
            "days_since_cross": trend_health.get("days_since_cross"),
            "cross_date": trend_health.get("cross_date"),
            "change_quality": change_quality.get("quality_label", ""),
            "change_score": change_quality.get("change_score", 0),
            "indicators": change_quality.get("indicators", {}),
            "alert_level": alert["level"],
            "alert_emoji": alert["emoji"],
            "alert_label": alert["label"],
            "alert_reasons": alert["reasons"],
            "long_term_label": long_term.get("label", ""),
            "long_term_summary": long_term.get("summary", ""),
            "value_trap": value_trap.get("is_trap", False),
            "value_trap_reasons": value_trap.get("reasons", []),
            "return_stability": sh_stability.get("stability", ""),
            "return_stability_emoji": _stability_emoji(sh_stability.get("stability", "")),
            # ファンダメンタルデータ（LLMサマリー用）
            "sector": stock_detail.get("sector", ""),
            "industry": stock_detail.get("industry", ""),
            "per": stock_detail.get("per"),
            "pbr": stock_detail.get("pbr"),
            "roe": stock_detail.get("roe"),
            "roa": stock_detail.get("roa"),
            "revenue_growth": stock_detail.get("revenue_growth"),
            "earnings_growth": stock_detail.get("earnings_growth"),
            "dividend_yield": stock_detail.get("dividend_yield"),
            "market_cap": stock_detail.get("market_cap"),
            "forward_eps": stock_detail.get("forward_eps"),
            "trailing_eps": stock_detail.get("trailing_eps"),
        }
        results.append(result)

        if alert["level"] != ALERT_NONE:
            alerts.append(result)
            counts[alert["level"]] = counts.get(alert["level"], 0) + 1
        else:
            counts["healthy"] += 1

    # 売りタイミング通知を生成
    sell_alerts = _compute_sell_alerts(results)

    return {
        "positions": results,
        "alerts": alerts,
        "sell_alerts": sell_alerts,
        "summary": {
            "total": len(results),
            **counts,
        },
    }


def _stability_emoji(stability: str) -> str:
    """還元安定度のエモジを返す."""
    return {
        "stable": "✅",
        "increasing": "📈",
        "temporary": "⚠️",
        "decreasing": "📉",
    }.get(stability, "")


def _compute_sell_alerts(positions: list[dict]) -> list[dict]:
    """ヘルスチェック結果から売りタイミング通知を生成する.

    以下の条件で通知を生成:
    1. EXIT アラート → 即座に売却検討
    2. CAUTION + 含み損 → 損切り検討
    3. デッドクロス直近発生 → トレンド転換注意
    4. RSI 30以下 → 売られ過ぎ（反発 or 更なる下落）
    5. バリュートラップ検出 → 割安罠からの撤退検討
    6. 含み益が大きい + トレンド下降 → 利確検討

    Returns
    -------
    list[dict]
        各通知: symbol, name, urgency (critical/warning/info),
        action, reason, details
    """
    alerts: list[dict] = []

    for pos in positions:
        symbol = pos["symbol"]
        name = pos["name"]
        alert_level = pos["alert_level"]
        pnl_pct = pos["pnl_pct"]
        trend = pos["trend"]
        rsi = pos.get("rsi", float("nan"))
        cross_signal = pos.get("cross_signal", "none")
        days_since_cross = pos.get("days_since_cross")
        value_trap = pos.get("value_trap", False)
        reasons = pos.get("alert_reasons", [])

        # 1. EXIT → 即売却検討（最高優先度）
        if alert_level == ALERT_EXIT:
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "critical",
                    "action": "売却検討",
                    "reason": "EXIT シグナル: テクニカル崩壊 + ファンダメンタル悪化",
                    "details": reasons,
                    "pnl_pct": pnl_pct,
                }
            )
            continue  # EXIT の場合は他の通知は不要

        # 2. CAUTION + 含み損 → 損切り検討
        if alert_level == ALERT_CAUTION and pnl_pct < -5:
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "critical",
                    "action": "損切り検討",
                    "reason": f"注意アラート & 含み損 {pnl_pct:+.1f}%",
                    "details": reasons,
                    "pnl_pct": pnl_pct,
                }
            )
            continue

        # 3. CAUTION（含み損なし）→ 警告
        if alert_level == ALERT_CAUTION:
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "warning",
                    "action": "注視・一部売却検討",
                    "reason": "注意アラート発生",
                    "details": reasons,
                    "pnl_pct": pnl_pct,
                }
            )

        # 4. 含み益 +20% 以上 + トレンド下降 → 利確検討
        if pnl_pct >= 20 and trend == "下降":
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "warning",
                    "action": "利確検討",
                    "reason": f"含み益 {pnl_pct:+.1f}% だがトレンド下降中",
                    "details": [
                        f"含み益 {pnl_pct:+.1f}% を確保できるうちに一部利確を検討",
                        "トレンド転換で含み益が縮小するリスク",
                    ],
                    "pnl_pct": pnl_pct,
                }
            )

        # 5. 直近デッドクロス（10日以内）→ 注意
        if cross_signal == "death_cross" and days_since_cross is not None and days_since_cross <= 10:
            # EXIT/CAUTION で既に通知した場合はスキップ
            if alert_level not in (ALERT_EXIT, ALERT_CAUTION):
                alerts.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "urgency": "warning",
                        "action": "トレンド転換注意",
                        "reason": f"デッドクロス発生（{days_since_cross}日前）",
                        "details": [
                            f"SMA50がSMA200を下回った（{pos.get('cross_date', '')}）",
                            "中長期トレンドの下降転換シグナル",
                        ],
                        "pnl_pct": pnl_pct,
                    }
                )

        # 6. バリュートラップ検出 → 注意
        if value_trap:
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "warning",
                    "action": "バリュートラップ注意",
                    "reason": "見せかけの割安（低PER + 利益減少）",
                    "details": pos.get("value_trap_reasons", []),
                    "pnl_pct": pnl_pct,
                }
            )

        # 7. RSI 30以下 → 情報
        if not _is_nan(rsi) and rsi <= 30:
            alerts.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "urgency": "info",
                    "action": "RSI 売られ過ぎ",
                    "reason": f"RSI = {rsi:.1f}（30以下）",
                    "details": [
                        "売られ過ぎ水準 — 反発の可能性もあるが更なる下落リスクも",
                        "他の指標と合わせて判断が必要",
                    ],
                    "pnl_pct": pnl_pct,
                }
            )

    # urgency 順にソート: critical > warning > info
    _urgency_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda x: (_urgency_order.get(x["urgency"], 9), x["symbol"]))
    return alerts


def _is_nan(v) -> bool:
    """NaN 判定ヘルパー."""
    try:
        import math

        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# 経済ニュース取得 & PF影響分析
# ---------------------------------------------------------------------------

# ニュース取得対象の主要指数/商品
_NEWS_TICKERS = {
    "^GSPC": "S&P 500",
    "^N225": "日経平均",
    "^DJI": "NYダウ",
    "^TNX": "米10年債",
    "CL=F": "原油(WTI)",
    "GC=F": "金",
    "JPY=X": "USD/JPY",
}

# ニュースキーワード → 影響カテゴリマッピング
_IMPACT_KEYWORDS = {
    "金利": {
        "keywords": [
            "interest rate",
            "fed",
            "fomc",
            "rate hike",
            "rate cut",
            "利上げ",
            "利下げ",
            "金利",
            "金融政策",
            "central bank",
            "treasury",
            "yield",
            "bond",
            "boj",
            "日銀",
        ],
        "icon": "🏦",
        "label": "金利・金融政策",
    },
    "為替": {
        "keywords": [
            "dollar",
            "yen",
            "forex",
            "currency",
            "exchange rate",
            "ドル",
            "円",
            "為替",
            "円安",
            "円高",
            "ドル高",
            "ドル安",
        ],
        "icon": "💱",
        "label": "為替",
    },
    "地政学": {
        "keywords": [
            "tariff",
            "trade war",
            "sanction",
            "geopolit",
            "war",
            "conflict",
            "tension",
            "関税",
            "制裁",
            "紛争",
            "戦争",
            "地政学",
            "トランプ",
            "trump",
            "china",
            "中国",
        ],
        "icon": "🌍",
        "label": "地政学・貿易",
    },
    "景気": {
        "keywords": [
            "gdp",
            "recession",
            "inflation",
            "cpi",
            "employment",
            "jobs",
            "unemployment",
            "consumer",
            "pmi",
            "景気",
            "インフレ",
            "雇用",
            "消費",
            "gdp",
            "リセッション",
        ],
        "icon": "📊",
        "label": "景気・経済指標",
    },
    "テクノロジー": {
        "keywords": [
            "ai",
            "artificial intelligence",
            "semiconductor",
            "chip",
            "tech",
            "software",
            "cloud",
            "nvidia",
            "半導体",
            "人工知能",
            "テック",
            "クラウド",
        ],
        "icon": "💻",
        "label": "テクノロジー",
    },
    "エネルギー": {
        "keywords": [
            "oil",
            "opec",
            "energy",
            "gas",
            "crude",
            "petroleum",
            "原油",
            "石油",
            "エネルギー",
            "opec",
        ],
        "icon": "⛽",
        "label": "エネルギー",
    },
}


def _classify_news_impact(title: str) -> list[dict]:
    """ニュースのタイトルからカテゴリを推定する.

    Returns
    -------
    list[dict]
        各要素は {"category": str, "icon": str, "label": str}.
    """
    title_lower = title.lower()
    categories = []
    for cat_id, cat_info in _IMPACT_KEYWORDS.items():
        for kw in cat_info["keywords"]:
            if kw in title_lower:
                categories.append(
                    {
                        "category": cat_id,
                        "icon": cat_info["icon"],
                        "label": cat_info["label"],
                    }
                )
                break
    return categories


def _estimate_portfolio_impact(
    categories: list[dict],
    positions: list[dict],
    fx_rates: dict,
) -> dict:
    """ニュースカテゴリに基づいてPFへの影響を推定する.

    Parameters
    ----------
    categories : list[dict]
        ``_classify_news_impact`` の出力.
    positions : list[dict]
        ポートフォリオの保有銘柄リスト.
    fx_rates : dict
        為替レート辞書.

    Returns
    -------
    dict
        Keys: impact_level ("high"/"medium"/"low"/"none"),
              affected_holdings (list[str]),
              reason (str).
    """
    if not categories or not positions:
        return {
            "impact_level": "none",
            "affected_holdings": [],
            "reason": "",
        }

    cat_ids = {c["category"] for c in categories}
    affected = []
    reasons = []

    for pos in positions:
        sector = (pos.get("sector") or "").lower()
        currency = pos.get("currency", "JPY")
        symbol = pos.get("symbol", "")

        if is_cash(symbol):
            continue

        # 金利影響: 金融セクター、不動産、高ベータ銘柄
        if "金利" in cat_ids:
            if any(w in sector for w in ["financial", "real estate", "金融", "不動産"]):
                affected.append(symbol)
                reasons.append(f"{symbol}: 金利感応セクター")
            elif pos.get("beta", 1.0) and float(pos.get("beta", 1.0) or 1.0) > 1.3:
                affected.append(symbol)
                reasons.append(f"{symbol}: 高ベータ")

        # 為替影響: 外貨建て保有
        if "為替" in cat_ids:
            if currency != "JPY":
                affected.append(symbol)
                reasons.append(f"{symbol}: {currency}建て")

        # テクノロジー影響
        if "テクノロジー" in cat_ids:
            if any(w in sector for w in ["technology", "テクノロジー", "情報通信", "semiconductor", "半導体"]):
                affected.append(symbol)
                reasons.append(f"{symbol}: テクノロジーセクター")

        # エネルギー影響
        if "エネルギー" in cat_ids:
            if any(w in sector for w in ["energy", "エネルギー", "石油"]):
                affected.append(symbol)
                reasons.append(f"{symbol}: エネルギーセクター")

        # 地政学影響: 貿易関連、中国関連
        if "地政学" in cat_ids:
            if any(w in sector for w in ["industrial", "製造", "consumer", "自動車", "automobile"]):
                affected.append(symbol)
                reasons.append(f"{symbol}: 貿易影響セクター")

    # 重複除去
    affected = list(dict.fromkeys(affected))
    reasons = list(dict.fromkeys(reasons))

    # 影響度判定
    total_non_cash = len([p for p in positions if not is_cash(p.get("symbol", ""))])
    if total_non_cash == 0:
        impact_level = "none"
    elif len(affected) / total_non_cash >= 0.5:
        impact_level = "high"
    elif len(affected) / total_non_cash >= 0.2:
        impact_level = "medium"
    elif affected:
        impact_level = "low"
    else:
        impact_level = "none"

    return {
        "impact_level": impact_level,
        "affected_holdings": affected,
        "reason": "; ".join(reasons[:5]),  # 上位5件まで
    }


def fetch_economic_news(
    positions: list[dict] | None = None,
    fx_rates: dict | None = None,
    max_per_ticker: int = 3,
    *,
    llm_enabled: bool = False,
    llm_model: str | None = None,
    llm_cache_ttl: int = 3600,
) -> list[dict]:
    """主要指数の経済ニュースを取得し、PF影響を分析する.

    Parameters
    ----------
    positions : list[dict] | None
        ポートフォリオのポジションリスト（PF影響分析用）.
    fx_rates : dict | None
        為替レート辞書.
    max_per_ticker : int
        各ティッカーから取得するニュース件数.
    llm_enabled : bool
        LLM 分析を使用するかどうか.
    llm_model : str | None
        LLM モデル ID.
    llm_cache_ttl : int
        LLM 分析キャッシュの有効期間（秒）。ニュースが同一かつ TTL 内なら再分析スキップ。

    Returns
    -------
    list[dict]
        各要素のキー:
        - title (str): ニュースタイトル
        - publisher (str): 発行元
        - link (str): URL
        - publish_time (str): 発行日時
        - source_ticker (str): 取得元ティッカー
        - source_name (str): 取得元名称
        - categories (list[dict]): 影響カテゴリ
        - portfolio_impact (dict): PF影響分析結果
        - analysis_method (str): "llm" or "keyword"
    """
    all_news: list[dict] = []
    seen_titles: set[str] = set()

    for ticker, name in _NEWS_TICKERS.items():
        try:
            items = yahoo_client.get_stock_news(ticker, count=max_per_ticker)
            for item in items:
                title = item.get("title", "")
                # 重複除外
                if title in seen_titles or not title:
                    continue
                seen_titles.add(title)

                all_news.append(
                    {
                        "title": title,
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", ""),
                        "publish_time": item.get("publish_time", ""),
                        "source_ticker": ticker,
                        "source_name": name,
                        "categories": [],
                        "portfolio_impact": {
                            "impact_level": "none",
                            "affected_holdings": [],
                            "reason": "",
                        },
                        "analysis_method": "keyword",
                    }
                )
        except Exception:
            continue

    if not all_news:
        return all_news

    # ------------------------------------------------------------------
    # LLM 分析を試行 → 失敗時はキーワードベースにフォールバック
    # ------------------------------------------------------------------
    _used_llm = False
    if llm_enabled:
        try:
            from components.llm_analyzer import analyze_news_batch, is_available

            if is_available():
                llm_results = analyze_news_batch(
                    all_news,
                    positions or [],
                    model=llm_model,
                    cache_ttl=llm_cache_ttl,
                )
                if llm_results is not None:
                    _apply_llm_results(all_news, llm_results)
                    _used_llm = True
                    logger.info("[data_loader] LLM analysis applied (%d items)", len(llm_results))
        except Exception as exc:
            logger.warning("[data_loader] LLM analysis failed, falling back: %s", exc)

    # キーワードベースのフォールバック
    if not _used_llm:
        for news_item in all_news:
            categories = _classify_news_impact(news_item["title"])
            impact = _estimate_portfolio_impact(
                categories,
                positions or [],
                fx_rates or {},
            )
            news_item["categories"] = categories
            news_item["portfolio_impact"] = impact
            news_item["analysis_method"] = "keyword"

    # 日時でソート（新しい順）、影響度でサブソート
    _impact_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    all_news.sort(
        key=lambda x: (
            _impact_order.get(x["portfolio_impact"]["impact_level"], 9),
            x.get("publish_time", "") or "",
        ),
        reverse=False,
    )
    # publish_time 降順にしつつ impact が高いものを先頭に
    all_news.sort(
        key=lambda x: _impact_order.get(x["portfolio_impact"]["impact_level"], 9),
    )

    return all_news


def compute_performance_attribution(
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-stock and per-sector performance attribution.

    Why: Users need to understand which stocks/sectors drive portfolio
         returns so they can make informed rebalancing decisions.
    How: Iterate non-cash positions, compute each stock's PnL contribution
         relative to total cost, then aggregate by sector.  Results are
         sorted by contribution (biggest first).
    """
    positions: list[dict] = snapshot.get("positions", [])
    total_value_jpy: float = snapshot.get("total_value_jpy", 0.0)

    stock_positions = [p for p in positions if not is_cash(p.get("symbol", ""))]

    total_cost_jpy = sum(p.get("cost_jpy", 0.0) for p in stock_positions)
    total_pnl_jpy = sum(p.get("pnl_jpy", 0.0) for p in stock_positions)
    total_pnl_pct = (total_pnl_jpy / total_cost_jpy * 100) if total_cost_jpy else 0.0

    by_stock: list[dict[str, Any]] = []
    sector_agg: dict[str, dict[str, float]] = {}

    for p in stock_positions:
        pnl = p.get("pnl_jpy", 0.0)
        cost = p.get("cost_jpy", 0.0)
        evaluation = p.get("evaluation_jpy", 0.0)
        sector = p.get("sector", "") or "不明"

        contribution_pct = (pnl / total_cost_jpy * 100) if total_cost_jpy else 0.0
        weight_pct = (evaluation / total_value_jpy * 100) if total_value_jpy else 0.0

        by_stock.append(
            {
                "symbol": p.get("symbol", ""),
                "sector": sector,
                "pnl_jpy": pnl,
                "cost_jpy": cost,
                "contribution_pct": contribution_pct,
                "weight_pct": weight_pct,
            }
        )

        if sector not in sector_agg:
            sector_agg[sector] = {"pnl_jpy": 0.0, "contribution_pct": 0.0}
        sector_agg[sector]["pnl_jpy"] += pnl
        sector_agg[sector]["contribution_pct"] += contribution_pct

    by_stock.sort(key=lambda x: x["contribution_pct"], reverse=True)

    return {
        "by_stock": by_stock,
        "by_sector": sector_agg,
        "total_cost_jpy": total_cost_jpy,
        "total_pnl_jpy": total_pnl_jpy,
        "total_pnl_pct": total_pnl_pct,
    }


def _apply_llm_results(all_news: list[dict], llm_results: list[dict]) -> None:
    """LLM 分析結果をニュースリストに適用する."""
    # id → result のマップ
    result_map = {r["id"]: r for r in llm_results}

    for i, news_item in enumerate(all_news):
        result = result_map.get(i)
        if result is None:
            continue

        news_item["categories"] = result.get("categories", [])
        affected = result.get("affected_holdings", [])
        reason = result.get("reason", "")
        impact_level = result.get("impact_level", "none")

        # impact_level の検証
        if impact_level not in ("high", "medium", "low", "none"):
            impact_level = "none"

        news_item["portfolio_impact"] = {
            "impact_level": impact_level,
            "affected_holdings": affected,
            "reason": reason,
        }
        news_item["analysis_method"] = "llm"
