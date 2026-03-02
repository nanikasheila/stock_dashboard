"""ウォッチリスト管理モジュール.

Why: ポートフォリオとは別に「注目銘柄」をターゲット価格付きで管理し、
     現在価格との乖離率を可視化することで購入タイミングの判断を支援する。
How: JSON ファイルで永続化した銘柄リストに対し CRUD を提供し、
     yahoo_client で価格を付加して Streamlit テーブルに表示する。
     オプションで Copilot LLM による投資判断分析を呼び出せる。
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from components.copilot_client import call as copilot_call
from components.copilot_client import is_available
from src.data.yahoo_client import get_stock_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# パス定義
# ---------------------------------------------------------------------------

WATCHLIST_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "watchlist" / "watchlist.json"

# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

WatchlistItem = dict[str, Any]

# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def load_watchlist() -> list[WatchlistItem]:
    """ウォッチリストを JSON ファイルから読み込む.

    Why: アプリ起動時・再描画時に永続化データを復元する必要がある。
    How: ファイルが存在しなければ空リストを返す。
         JSON パースエラー時はログ出力して空リストにフォールバック。
    """
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("watchlist.json is not a list, returning empty")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load watchlist: %s", exc)
        return []


def save_watchlist(watchlist: list[WatchlistItem]) -> None:
    """ウォッチリストを JSON ファイルに保存する.

    Why: 追加・削除操作の結果を永続化する。
    How: 親ディレクトリを自動作成し、UTF-8 / indent=2 で書き出す。
    """
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)


def add_to_watchlist(
    symbol: str,
    target_price: float,
    target_currency: str = "USD",
    memo: str = "",
) -> list[WatchlistItem]:
    """銘柄をウォッチリストに追加して保存済みリストを返す.

    Why: UI から1銘柄ずつ追加する操作を提供する。
    How: 既存リストをロードし、重複チェック後に末尾追加して保存する。
         同一シンボルが既に存在する場合はターゲット価格・メモを上書きする。
    """
    watchlist = load_watchlist()
    new_item: WatchlistItem = {
        "symbol": symbol.upper().strip(),
        "target_price": target_price,
        "target_currency": target_currency,
        "added_date": date.today().isoformat(),
        "memo": memo,
    }
    # 既存エントリの上書き
    for i, item in enumerate(watchlist):
        if item.get("symbol") == new_item["symbol"]:
            watchlist[i] = new_item
            save_watchlist(watchlist)
            return watchlist

    watchlist.append(new_item)
    save_watchlist(watchlist)
    return watchlist


def remove_from_watchlist(symbol: str) -> list[WatchlistItem]:
    """指定シンボルをウォッチリストから削除して保存済みリストを返す.

    Why: 不要になった銘柄を一覧から除去する操作を提供する。
    How: シンボル一致でフィルタし、結果を上書き保存する。
    """
    watchlist = load_watchlist()
    watchlist = [item for item in watchlist if item.get("symbol") != symbol.upper().strip()]
    save_watchlist(watchlist)
    return watchlist


# ---------------------------------------------------------------------------
# 価格エンリッチメント
# ---------------------------------------------------------------------------


def get_watchlist_with_prices(
    watchlist: list[WatchlistItem],
    fx_rates: dict[str, float],
) -> list[dict[str, Any]]:
    """ウォッチリスト各銘柄に現在価格・指標を付加して返す.

    Why: ターゲット価格との乖離率を計算するには現在価格が必要。
         セクターや PER/PBR/配当利回りも一覧表示に使う。
    How: yahoo_client.get_stock_info で銘柄情報を取得し、
         通貨が異なる場合は fx_rates で円換算して乖離率を算出する。
    """
    enriched: list[dict[str, Any]] = []
    for item in watchlist:
        symbol = item.get("symbol", "")
        target_price = item.get("target_price", 0.0)
        target_currency = item.get("target_currency", "USD")

        info = get_stock_info(symbol)
        current_price: float | None = None
        sector: str | None = None
        per: float | None = None
        pbr: float | None = None
        dividend_yield: float | None = None

        if info is not None:
            current_price = info.get("price")
            sector = info.get("sector")
            per = info.get("per")
            pbr = info.get("pbr")
            dividend_yield = info.get("dividend_yield")

        # 乖離率の計算（同一通貨に揃える）
        distance_pct: float | None = None
        if current_price is not None and target_price > 0:
            stock_currency = info.get("currency", target_currency) if info else target_currency
            # 通貨が異なる場合、ターゲット価格を株価通貨に換算
            adjusted_target = target_price
            if stock_currency != target_currency:
                rate_from = fx_rates.get(target_currency, 1.0)
                rate_to = fx_rates.get(stock_currency, 1.0)
                if rate_to > 0:
                    adjusted_target = target_price * rate_from / rate_to
            if adjusted_target > 0:
                distance_pct = ((current_price - adjusted_target) / adjusted_target) * 100.0

        enriched.append(
            {
                **item,
                "current_price": current_price,
                "sector": sector,
                "per": per,
                "pbr": pbr,
                "dividend_yield": dividend_yield,
                "distance_pct": distance_pct,
            }
        )
    return enriched


# ---------------------------------------------------------------------------
# LLM 分析
# ---------------------------------------------------------------------------


def analyze_watchlist_stock(
    symbol: str,
    stock_info: dict[str, Any],
    model: str = "gpt-4.1",
    source: str = "watchlist_analysis",
) -> str | None:
    """Copilot LLM を使って銘柄の投資判断分析を実行する.

    Why: ウォッチリスト銘柄について、基本指標だけでなく AI による
         総合的な投資判断を手軽に得られるようにする。
    How: 現在価格・ターゲット価格・基本指標を含む日本語プロンプトを構築し、
         copilot_call でワンショット呼び出しを行う。
         失敗時は None を返す。
    """
    current_price = stock_info.get("current_price") or stock_info.get("price", "N/A")
    target_price = stock_info.get("target_price", "N/A")
    sector = stock_info.get("sector", "N/A")
    per = stock_info.get("per", "N/A")
    pbr = stock_info.get("pbr", "N/A")
    dividend_yield = stock_info.get("dividend_yield")
    dy_str = f"{dividend_yield * 100:.2f}%" if dividend_yield is not None else "N/A"

    prompt = (
        f"以下の銘柄について、購入判断の参考となる分析を日本語で行ってください。\n\n"
        f"銘柄: {symbol}\n"
        f"セクター: {sector}\n"
        f"現在価格: {current_price}\n"
        f"ターゲット価格: {target_price}\n"
        f"PER: {per}\n"
        f"PBR: {pbr}\n"
        f"配当利回り: {dy_str}\n\n"
        f"以下の3点について簡潔に回答してください:\n"
        f"1) 概要 — 事業内容と市場でのポジション\n"
        f"2) 投資判断 — 現在の株価水準は割安か割高か、根拠を含めて\n"
        f"3) リスク要因 — 主要なリスクと注意点\n"
    )

    try:
        return copilot_call(prompt, model=model, source=source, timeout=60)
    except Exception as exc:
        logger.warning("LLM analysis failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_watchlist(
    settings: dict[str, Any],
    fx_rates: dict[str, float],
) -> None:
    """ウォッチリストの管理 UI を描画する.

    Why: 注目銘柄の追加・削除・価格確認・AI 分析を1画面で完結させる。
    How: st.form で銘柄追加フォームを表示し、価格付きテーブルを描画する。
         各行に削除ボタンを配置し、LLM が有効なら AI 分析ボタンも表示する。
    """
    st.subheader("📋 ウォッチリスト")

    watchlist = load_watchlist()

    # --- 銘柄追加フォーム ---
    with st.form("watchlist_add_form", clear_on_submit=True):
        cols = st.columns([2, 1, 1, 3])
        symbol_input = cols[0].text_input("銘柄コード", placeholder="例: AAPL")
        target_price_input = cols[1].number_input("ターゲット価格", min_value=0.0, step=0.01)
        currency_input = cols[2].selectbox("通貨", ["USD", "JPY", "EUR", "GBP", "SGD"], index=0)
        memo_input = cols[3].text_input("メモ", placeholder="購入理由など")
        submitted = st.form_submit_button("➕ 追加")

        if submitted and symbol_input:
            add_to_watchlist(
                symbol=symbol_input,
                target_price=target_price_input,
                target_currency=currency_input,
                memo=memo_input,
            )
            st.rerun()

    # --- 空リスト時のヘルプ ---
    if not watchlist:
        st.info("ウォッチリストが空です。上のフォームから銘柄を追加してください。")
        return

    # --- 価格付きリスト取得 ---
    enriched = get_watchlist_with_prices(watchlist, fx_rates)

    llm_enabled = settings.get("watchlist_llm_enabled", True) and is_available()

    # --- テーブル描画 ---
    header_cols = st.columns([1.5, 1.5, 1.2, 1.2, 1, 1, 1, 1, 0.8, 0.8])
    headers = ["銘柄", "セクター", "現在価格", "ターゲット", "乖離率(%)", "PER", "PBR", "配当利回り", "", ""]
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")

    for item in enriched:
        row_cols = st.columns([1.5, 1.5, 1.2, 1.2, 1, 1, 1, 1, 0.8, 0.8])
        row_cols[0].write(item.get("symbol", ""))
        row_cols[1].write(item.get("sector") or "—")

        # 現在価格
        cp = item.get("current_price")
        row_cols[2].write(f"{cp:,.2f}" if cp is not None else "—")

        # ターゲット価格
        tp = item.get("target_price", 0)
        tc = item.get("target_currency", "")
        row_cols[3].write(f"{tp:,.2f} {tc}" if tp else "—")

        # 乖離率（色付き）
        dist = item.get("distance_pct")
        if dist is not None:
            color = "green" if dist < 0 else "red"
            row_cols[4].markdown(f":{color}[{dist:+.1f}%]")
        else:
            row_cols[4].write("—")

        # PER
        per_val = item.get("per")
        row_cols[5].write(f"{per_val:.1f}" if per_val is not None else "—")

        # PBR
        pbr_val = item.get("pbr")
        row_cols[6].write(f"{pbr_val:.2f}" if pbr_val is not None else "—")

        # 配当利回り
        dy_val = item.get("dividend_yield")
        row_cols[7].write(f"{dy_val * 100:.2f}%" if dy_val is not None else "—")

        # 削除ボタン
        sym = item.get("symbol", "")
        if row_cols[8].button("🗑️", key=f"wl_rm_{sym}"):
            remove_from_watchlist(sym)
            st.rerun()

        # AI 分析ボタン
        if llm_enabled:
            if row_cols[9].button("🤖", key=f"wl_ai_{sym}"):
                with st.spinner(f"{sym} を分析中..."):
                    model = settings.get("llm_model", "gpt-4.1")
                    result = analyze_watchlist_stock(sym, item, model=model)
                    if result:
                        st.markdown(result)
                    else:
                        st.warning("AI 分析に失敗しました。")
