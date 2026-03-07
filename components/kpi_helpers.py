"""KPI カード HTML ビルダー — ダッシュボード全体で共有.

app.py の KPI セクションと各タブモジュールが共通で使用する
HTML 断片ジェネレーターを集約する。

Usage
-----
    from components.kpi_helpers import (
        kpi_main_card, kpi_sub_card, risk_card, alert_badge_card
    )

    st.markdown(kpi_main_card("資産", "¥1,000,000"), unsafe_allow_html=True)
    st.markdown(alert_badge_card("⚠️", "ヘルス注意", 3, detail="要確認"), unsafe_allow_html=True)
"""

from __future__ import annotations


def kpi_main_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    """大項目 KPI カード — テーマ追従 + 大きめフォント.

    Parameters
    ----------
    label:  カード上部のラベル文字列
    value:  メインの数値 / テキスト
    sub:    補足テキスト（省略可）
    color:  value と sub に適用する CSS 色（省略で継承）
    """
    color_style = f"color:{color};" if color else ""
    sub_html = f'<div style="font-size:0.92rem; {color_style} margin-top:4px; opacity:0.85;">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card kpi-main" role="group" aria-label="{label}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="{color_style}">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


def kpi_sub_card(label: str, value: str, color: str = "") -> str:
    """小項目 KPI カード — テーマ追従 + コンパクト.

    Parameters
    ----------
    label:  カード上部のラベル文字列
    value:  数値 / テキスト
    color:  value に適用する CSS 色（省略で継承）
    """
    color_style = f"color:{color};" if color else ""
    return (
        f'<div class="kpi-card kpi-sub" role="group" aria-label="{label}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value-sub" style="{color_style}">{value}</div>'
        f"</div>"
    )


def risk_card(label: str, value: str, color: str = "") -> str:
    """リスク指標カード — テーマ追従 + 最小サイズ.

    Parameters
    ----------
    label:  ラベル（長い場合は省略記号で切り捨て）
    value:  数値 / テキスト
    color:  value に適用する CSS 色（省略で継承）
    """
    color_style = f"color:{color};" if color else ""
    return (
        f'<div class="kpi-card kpi-risk" role="group" aria-label="{label}">'
        f'<div class="kpi-label" style="white-space:nowrap;'
        f' overflow:hidden; text-overflow:ellipsis;">{label}</div>'
        f'<div class="kpi-value-risk" style="{color_style}">{value}</div>'
        f"</div>"
    )


def alert_badge_card(
    icon: str,
    label: str,
    count: int,
    *,
    detail: str = "",
    color: str = "",
) -> str:
    """アラートバッジカード — ヘッドラインストリップ用コンパクト表示.

    タブを横断する「要確認」情報を一目で伝えるための小型カード。
    件数が 0 の場合は em dash を表示して「問題なし」を示す。

    Parameters
    ----------
    icon:   状態を示すアイコン（絵文字など）
    label:  ラベル文字列
    count:  件数（0 のとき "—" 表示）
    detail: 補足テキスト（省略可）
    color:  CSS カラー（省略で継承）
    """
    color_style = f"color:{color};" if color else ""
    border_style = f"border-left:3px solid {color};" if color else "border-left:3px solid transparent;"
    count_str = str(count) if count > 0 else "—"
    detail_html = (
        f'<div style="font-size:0.72rem; opacity:0.62; margin-top:2px;'
        f" white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
        f'">{detail}</div>'
        if detail
        else ""
    )
    return (
        f'<div class="kpi-card kpi-risk" role="group" aria-label="{label}: {count_str}" '
        f'style="{border_style} padding:8px 14px; text-align:center;">'
        f'<div style="font-size:1.15rem; line-height:1.1;">{icon}</div>'
        f'<div class="kpi-label" style="margin-top:3px; white-space:nowrap;'
        f' overflow:hidden; text-overflow:ellipsis;">{label}</div>'
        f'<div class="kpi-value-risk" style="{color_style} font-size:1.1rem; font-weight:700;">'
        f"{count_str}</div>"
        f"{detail_html}"
        f"</div>"
    )
