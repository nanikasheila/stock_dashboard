"""LLM 分析キャッシュ管理モジュール.

各 LLM 分析セッション（ニュース分析・要約・ヘルスチェック・統合分析・インサイト）の
キャッシュ辞書・アクセサ・クリア関数・ハッシュ計算ヘルパーを集約する。

Why: llm_analyzer.py から反復的なキャッシュ定型コードを分離し、
     プロンプト構築・レスポンス解析・オーケストレーションの可読性を高める。
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

# ---------------------------------------------------------------------------
# ニュース分析キャッシュ: ニュースが変わらなければ LLM 再呼び出しをスキップ
# ---------------------------------------------------------------------------

_analysis_cache: dict[str, Any] = {
    "hash": "",  # ニュースタイトル一覧の SHA-256
    "results": None,  # 前回の分析結果
    "timestamp": 0.0,  # 前回分析時刻 (time.time())
    "model": "",  # 前回使用モデル
}


def _compute_news_hash(news_items: list[dict]) -> str:
    """ニュースタイトルのリストから決定的ハッシュを生成する."""
    titles = sorted(item.get("title", "") for item in news_items if item.get("title"))
    return hashlib.sha256("\n".join(titles).encode()).hexdigest()


def get_cache_info() -> dict[str, Any]:
    """現在のキャッシュ状態を返す（UI 表示用）."""
    ts = _analysis_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {
        "cached": True,
        "age_sec": int(age),
        "model": _analysis_cache["model"],
    }


def clear_cache() -> None:
    """分析キャッシュを強制クリアする."""
    _analysis_cache["hash"] = ""
    _analysis_cache["results"] = None
    _analysis_cache["timestamp"] = 0.0
    _analysis_cache["model"] = ""


# ---------------------------------------------------------------------------
# サマリーキャッシュ — ニュースが変わらなければ再生成しない
# ---------------------------------------------------------------------------

_summary_cache: dict[str, Any] = {
    "hash": "",
    "result": None,
    "timestamp": 0.0,
    "model": "",
}


def get_summary_cache_info() -> dict[str, Any]:
    """サマリーキャッシュの状態を返す."""
    ts = _summary_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {"cached": True, "age_sec": int(age), "model": _summary_cache["model"]}


def clear_summary_cache() -> None:
    """サマリーキャッシュを強制クリアする."""
    _summary_cache["hash"] = ""
    _summary_cache["result"] = None
    _summary_cache["timestamp"] = 0.0
    _summary_cache["model"] = ""


# ---------------------------------------------------------------------------
# ヘルスチェックサマリーキャッシュ
# ---------------------------------------------------------------------------

_health_summary_cache: dict[str, Any] = {
    "hash": "",
    "result": None,
    "timestamp": 0.0,
    "model": "",
}


def get_health_summary_cache_info() -> dict[str, Any]:
    """ヘルスチェックサマリーキャッシュの状態を返す."""
    ts = _health_summary_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {"cached": True, "age_sec": int(age), "model": _health_summary_cache["model"]}


def clear_health_summary_cache() -> None:
    """ヘルスチェックサマリーキャッシュを強制クリアする."""
    _health_summary_cache["hash"] = ""
    _health_summary_cache["result"] = None
    _health_summary_cache["timestamp"] = 0.0
    _health_summary_cache["model"] = ""


def _compute_health_hash(health_data: dict, news_items: list[dict] | None = None) -> str:
    """ヘルスチェックデータのハッシュを生成する."""
    positions = health_data.get("positions", [])
    parts = []
    for p in positions:
        parts.append(f"{p.get('symbol', '')}:{p.get('alert_level', '')}:{p.get('pnl_pct', 0)}")
    # ニュースタイトルもハッシュに含める（ニュースが変われば再分析）
    if news_items:
        for n in news_items:
            title = n.get("title", "")
            if title:
                parts.append(f"news:{title}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 統合分析キャッシュ
# ---------------------------------------------------------------------------

_unified_cache: dict[str, Any] = {
    "hash": "",
    "result": None,
    "timestamp": 0.0,
    "model": "",
}


def get_unified_cache_info() -> dict[str, Any]:
    """統合分析キャッシュの状態を返す."""
    ts = _unified_cache["timestamp"]
    if ts == 0:
        return {"cached": False, "age_sec": 0, "model": ""}
    age = time.time() - ts
    return {"cached": True, "age_sec": int(age), "model": _unified_cache["model"]}


def clear_unified_cache() -> None:
    """統合分析キャッシュを強制クリアする."""
    _unified_cache["hash"] = ""
    _unified_cache["result"] = None
    _unified_cache["timestamp"] = 0.0
    _unified_cache["model"] = ""


def _compute_unified_hash(
    news_items: list[dict],
    health_data: dict | None = None,
) -> str:
    """統合分析のキャッシュハッシュを生成する."""
    parts: list[str] = []
    # ニュースタイトル
    for item in news_items:
        title = item.get("title", "")
        if title:
            parts.append(f"news:{title}")
    # ヘルスチェックデータ
    if health_data:
        for p in health_data.get("positions", []):
            parts.append(f"pos:{p.get('symbol', '')}:{p.get('alert_level', '')}:{p.get('pnl_pct', 0)}")
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# インサイトキャッシュ
# ---------------------------------------------------------------------------

_insights_cache: dict[str, Any] = {"hash": None, "results": None, "timestamp": 0.0}


def clear_insights_cache() -> None:
    """インサイトキャッシュを強制クリアする."""
    _insights_cache["hash"] = None
    _insights_cache["results"] = None
    _insights_cache["timestamp"] = 0.0


def _compute_insights_hash(
    snapshot: dict[str, Any],
    structure: dict[str, Any],
) -> str:
    """snapshot と structure から決定的ハッシュを生成する."""
    raw = json.dumps(
        {
            "total_value_jpy": snapshot.get("total_value_jpy"),
            "total_pnl_pct": snapshot.get("total_pnl_pct"),
            "sector_hhi": structure.get("sector_hhi"),
            "risk_level": structure.get("risk_level"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()
