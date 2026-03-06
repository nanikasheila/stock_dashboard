"""経済ニュース取得 & ポートフォリオ影響分析 — data_loader サブモジュール.

主要指数のニュースを取得し、キーワードベースまたは LLM で
ポートフォリオへの影響を分析する。
``components.data_loader`` がこのモジュールを import して再エクスポートする。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.common import is_cash

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
