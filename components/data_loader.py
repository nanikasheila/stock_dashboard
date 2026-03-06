"""ポートフォリオダッシュボード — データローダー (互換ファサード).

既存の portfolio_manager / history_store / yahoo_client を活用して
ダッシュボード表示用のデータを組み立てる。

このファイルはドメイン別サブモジュール群の再エクスポートファサードである。
app.py やテストは従来通り ``from components.data_loader import <name>`` で
インポートできる。

サブモジュール構成:
    dl_prices.py    — 価格フェッチ & ディスクキャッシュ（dl_history/dl_analytics 用）
    dl_holdings.py  — 保有銘柄・取引履歴ドメイン（純粋ヘルパー）
    dl_history.py   — ポートフォリオ履歴ビルダー & 基本集計
    dl_analytics.py — 分析・リスク指標
    dl_health.py    — 売りアラート純粋ヘルパー
    dl_news.py      — 経済ニュース純粋ヘルパー

このモジュールに直接定義される関数:
    価格キャッシュ全体 — tests が components.data_loader._PRICE_CACHE_DIR 等をパッチするため
    _build_symbol_labels  — tests が components.data_loader.yahoo_client をパッチするため
    run_dashboard_health_check — tests が components.data_loader.yahoo_client 等をパッチするため
    fetch_economic_news   — tests が components.data_loader.yahoo_client をパッチするため
"""

from __future__ import annotations

import logging
import sys
import time as _time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DEFAULT_HISTORY_DIR = str(Path(_PROJECT_ROOT) / "data" / "history")

# ---------------------------------------------------------------------------
# 外部依存 — テスト時にこれらが patch のターゲットとなる
# ---------------------------------------------------------------------------

from src.core.common import is_cash
from src.core.health_check import (
    ALERT_NONE,
    check_change_quality,
    check_long_term_suitability,
    check_trend_health,
    compute_alert_level,
)
from src.core.paths import PRICE_CACHE_DIR as _PRICE_CACHE_DIR
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
from src.core.value_trap import detect_value_trap
from src.data import yahoo_client

# ---------------------------------------------------------------------------
# 価格キャッシュ定数 & 関数
# Why: tests patch components.data_loader._PRICE_CACHE_DIR,
#      _CACHE_TTL_SECONDS, and _fetch_price_history.
#      Functions defined here see the patched values from this module's namespace.
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours

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
    except Exception as exc:
        logger.warning("_load_cached_prices: failed to read cache %s: %s", path, exc)
        return None


def _save_prices_cache(prices: pd.DataFrame, period: str) -> None:
    """株価をディスクキャッシュに保存."""
    try:
        _PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(_get_cache_path(period))
    except Exception as exc:
        logger.warning(
            "_save_prices_cache: failed to write cache for period=%s: %s",
            period,
            exc,
        )


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
            logger.debug(
                "_load_prices: cache hit for period=%s (%d symbols)",
                period,
                len(available),
            )
            return cached[available]
        # 不足銘柄のみ追加取得
        logger.debug(
            "_load_prices: cache partial for period=%s, fetching %d missing symbols",
            period,
            len(missing),
        )
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
    logger.debug(
        "_load_prices: cache miss for period=%s, batch-fetching %d symbols",
        period,
        len(symbols),
    )
    prices = yahoo_client.get_close_prices_batch(symbols, period=yf_period)
    if prices is None or prices.empty:
        # フォールバック: 個別取得
        logger.warning(
            "_load_prices: batch fetch failed for period=%s, falling back to individual fetches",
            period,
        )
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


# ---------------------------------------------------------------------------
# 銘柄表示ラベル生成
# Why: tests patch components.data_loader.yahoo_client.
#      This function must be defined here so the patch affects its calls.
# ---------------------------------------------------------------------------

from components.dl_holdings import _shorten_company_name


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
# ヘルスチェック純粋ヘルパー（dl_health から）
# ---------------------------------------------------------------------------

from components.dl_health import (
    _compute_sell_alerts,
    _is_nan,
    _stability_emoji,
)

# ---------------------------------------------------------------------------
# ダッシュボードヘルスチェック
# Why: tests patch components.data_loader.yahoo_client AND load_portfolio.
#      This function must be defined here so patches affect its calls.
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


# ---------------------------------------------------------------------------
# 経済ニュース純粋ヘルパー（dl_news から）& fetch_economic_news 本体
# Why: fetch_economic_news は tests が components.data_loader.yahoo_client を
#      パッチするため、ここで定義する。
# ---------------------------------------------------------------------------

from components.dl_news import (
    _IMPACT_KEYWORDS,
    _NEWS_TICKERS,
    _apply_llm_results,
    _classify_news_impact,
    _estimate_portfolio_impact,
)


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
        except Exception as exc:
            logger.warning(
                "get_economic_news_with_impact: failed to fetch news for ticker=%s: %s",
                ticker,
                exc,
            )
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


# ---------------------------------------------------------------------------
# 保有銘柄・取引履歴ドメイン（dl_holdings から再エクスポート）
# NOTE: _build_symbol_labels はこのモジュールで定義済み（上記）のため
#       dl_holdings からはインポートしない。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 分析・リスク指標（dl_analytics から再エクスポート）
# ---------------------------------------------------------------------------
from components.dl_analytics import (
    compute_benchmark_excess,
    compute_correlation_matrix,
    compute_daily_change,
    compute_drawdown_series,
    compute_performance_attribution,
    compute_risk_metrics,
    compute_rolling_sharpe,
    compute_top_worst_performers,
    compute_weight_drift,
    get_benchmark_series,
)

# ---------------------------------------------------------------------------
# ポートフォリオ履歴ビルダー & 基本集計（dl_history から再エクスポート）
# ---------------------------------------------------------------------------
from components.dl_history import (
    build_portfolio_history,
    build_projection,
    get_monthly_summary,
    get_sector_breakdown,
    get_trade_activity,
)
from components.dl_holdings import (
    _CORPORATE_SUFFIXES,
    _build_holdings_timeline,
    _build_trade_activity,
    _compute_invested_capital,
    _compute_pnl_moving_average,
    _compute_realized_pnl,
    _reconstruct_daily_holdings,
    _trade_cost_jpy,
    get_current_snapshot,
)

__all__ = [
    "DEFAULT_CSV_PATH",
    "_CACHE_TTL_SECONDS",
    "_CORPORATE_SUFFIXES",
    "_IMPACT_KEYWORDS",
    "_NEWS_TICKERS",
    "_PERIOD_MAP",
    "_PRICE_CACHE_DIR",
    "_apply_llm_results",
    "_build_holdings_timeline",
    "_build_symbol_labels",
    "_build_trade_activity",
    "_classify_news_impact",
    "_compute_invested_capital",
    "_compute_pnl_moving_average",
    "_compute_realized_pnl",
    "_compute_sell_alerts",
    "_estimate_portfolio_impact",
    "_fetch_price_history",
    "_get_cache_path",
    "_is_nan",
    "_load_cached_prices",
    "_load_prices",
    "_reconstruct_daily_holdings",
    "_save_prices_cache",
    "_shorten_company_name",
    "_stability_emoji",
    "_trade_cost_jpy",
    "build_portfolio_history",
    "build_projection",
    "clear_price_cache",
    "compute_benchmark_excess",
    "compute_correlation_matrix",
    "compute_daily_change",
    "compute_drawdown_series",
    "compute_performance_attribution",
    "compute_risk_metrics",
    "compute_rolling_sharpe",
    "compute_top_worst_performers",
    "compute_weight_drift",
    "fetch_economic_news",
    "get_benchmark_series",
    "get_current_snapshot",
    "get_fx_rates",
    "get_monthly_summary",
    "get_sector_breakdown",
    "get_trade_activity",
    "load_portfolio",
    "run_dashboard_health_check",
]
