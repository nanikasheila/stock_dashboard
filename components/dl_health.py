"""ヘルスチェック純粋ヘルパー — data_loader サブモジュール.

`run_dashboard_health_check` は components.data_loader に定義される
（テストが components.data_loader.yahoo_client / load_portfolio をパッチするため）。
ここには副作用のない純粋ヘルパー関数のみを収録する。
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.health_check import ALERT_CAUTION, ALERT_EXIT


def _stability_emoji(stability: str) -> str:
    """還元安定度のエモジを返す."""
    return {
        "stable": "✅",
        "increasing": "📈",
        "temporary": "⚠️",
        "decreasing": "📉",
    }.get(stability, "")


def _is_nan(v) -> bool:
    """NaN 判定ヘルパー."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


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

    _urgency_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda x: (_urgency_order.get(x["urgency"], 9), x["symbol"]))
    return alerts
