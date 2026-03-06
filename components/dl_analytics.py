"""分析・リスク指標 — data_loader サブモジュール.

日次評価額 DataFrame からリスク指標・ランキング・ベンチマーク比較・
相関行列・ウェイトドリフト・パフォーマンス帰属を計算する。
``components.data_loader`` がこのモジュールを import して再エクスポートする。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from components.dl_prices import _load_prices
from src.core.common import is_cash

# ---------------------------------------------------------------------------
# 7. リスク指標の算出
# ---------------------------------------------------------------------------


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
# パフォーマンス帰属
# ---------------------------------------------------------------------------


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
