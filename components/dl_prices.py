"""価格フェッチ & ディスクキャッシュ — data_loader サブモジュール.

ディスクキャッシュ（TTL 4h）付きの株価一括取得ロジックを提供する。
``components.data_loader`` がこのモジュールを import して再エクスポートする。
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

from src.core.paths import PRICE_CACHE_DIR as _PRICE_CACHE_DIR
from src.data import yahoo_client

_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours

_PERIOD_MAP: dict[str, str] = {
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
