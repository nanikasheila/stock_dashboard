"""ダッシュボード設定の永続化モジュール.

設定をJSONファイルに保存し、次回起動時に復元する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 設定ファイルのデフォルトパス
_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "dashboard_settings.json"

# 設定のデフォルト値
DEFAULTS: dict[str, Any] = {
    "period_label": "3ヶ月",
    "chart_style": "積み上げ面",
    "show_invested": True,
    "benchmark_label": "なし",
    "show_individual": False,
    "show_projection": True,
    "target_amount_man": 5000,
    "projection_years": 5,
    "auto_refresh_label": "5分",
    # LLM ニュース分析設定
    "llm_enabled": False,
    "llm_auto_analyze": False,  # True=自動, False=手動
    "llm_model": "gpt-4.1",
    "llm_cache_ttl_label": "1時間",
    # Copilot チャット設定
    "chat_model": "claude-sonnet-4",
    # AI インサイト設定
    "insights_enabled": True,
    # 取引影響プレビュー設定
    "trade_preview_enabled": True,
    "trade_preview_llm": True,
    # ウォッチリスト設定
    "watchlist_llm_enabled": True,
    # パフォーマンス寄与分析設定
    "attribution_llm_enabled": True,
}


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """設定ファイルを読み込む。ファイルが存在しない場合はデフォルト値を返す。"""
    p = path or _DEFAULT_SETTINGS_PATH
    settings = dict(DEFAULTS)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for key in DEFAULTS:
                    if key in saved:
                        settings[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict[str, Any], path: Path | None = None) -> None:
    """設定をJSONファイルに保存する。"""
    p = path or _DEFAULT_SETTINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    to_save = {}
    for key in DEFAULTS:
        if key in settings:
            to_save[key] = settings[key]
    with open(p, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
