"""ポートフォリオ履歴ビルダー & 基本集計 — data_loader サブモジュール.

売買履歴と株価から日次評価額 DataFrame を構築し、
セクター別・月次・売買アクティビティの集計と将来推定を提供する。
``components.data_loader`` がこのモジュールを import して再エクスポートする。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")

from components.dl_holdings import (
    _build_holdings_timeline,
    _build_symbol_labels,
    _build_trade_activity,
    _compute_invested_capital,
    _reconstruct_daily_holdings,
)
from components.dl_prices import _load_prices
from src.core.common import is_cash
from src.core.portfolio.portfolio_manager import (
    DEFAULT_CSV_PATH,
    get_fx_rates,
    load_portfolio,
)
from src.core.ticker_utils import infer_currency
from src.data import yahoo_client

# ---------------------------------------------------------------------------
# 3. 資産推移データの構築
# ---------------------------------------------------------------------------


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
        except Exception as exc:
            logger.warning(
                "build_projection: estimate_portfolio_return failed, using default rates: %s",
                exc,
            )
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
