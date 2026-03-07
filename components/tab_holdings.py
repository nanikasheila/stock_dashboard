"""保有構成タブ描画モジュール.

``app.py`` の ``with _tab_holdings:`` ブロックを切り出したモジュール。
銘柄別評価額テーブル / セクター構成 / 通貨配分 /
ツリーマップ / ウェイトドリフト / 銘柄間相関を描画する。

公開 API
--------
render_holdings_tab(...)
    保有構成タブのコンテンツを描画する。
"""

from __future__ import annotations

import html as _html_mod
import time

import pandas as pd
import streamlit as st

from components.charts import (
    build_correlation_chart,
    build_currency_chart,
    build_sector_chart,
    build_treemap_chart,
)
from components.data_loader import (
    compute_correlation_matrix,
    compute_weight_drift,
    get_sector_breakdown,
)


def render_holdings_tab(
    *,
    snapshot: dict,
    positions: list[dict],
    total_value: float,
    history_df,
) -> None:
    """保有構成タブのコンテンツを描画する.

    Parameters
    ----------
    snapshot:    ポートフォリオスナップショット（``get_current_snapshot()`` 戻り値）
    positions:   保有銘柄リスト
    total_value: 総資産（円換算）
    history_df:  ポートフォリオ価格履歴 DataFrame
    """
    # =====================================================================
    # 現在の保有構成
    # =====================================================================
    st.markdown('<div id="holdings" role="region" aria-label="保有銘柄一覧"></div>', unsafe_allow_html=True)
    _holdings_as_of = snapshot.get("as_of", "")[:16].replace("T", " ") or "—"
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🏢 銘柄別 評価額")
        st.caption(
            f"保有銘柄ごとの評価額・損益率を確認。構成比の偏りや損益の大きい銘柄を把握できます。｜ 🕐 データ取得: {_holdings_as_of}"
        )

        holdings_df = pd.DataFrame(
            [
                {
                    "銘柄": f"{p['name']} ({p['symbol']})",
                    "保有数": p["shares"],
                    "現在価格": f"{p['current_price']:,.2f} {p.get('currency', '')}",
                    "評価額(円)": p["evaluation_jpy"],
                    "構成比": p["evaluation_jpy"] / total_value * 100 if total_value else 0,
                    "損益(円)": p.get("pnl_jpy", 0),
                    "損益率(%)": p.get("pnl_pct", 0),
                    "通貨": p.get("currency", ""),
                    "セクター": p.get("sector", ""),
                }
                for p in positions
            ]
        )

        if not holdings_df.empty:
            # 評価額でソート
            holdings_df = holdings_df.sort_values("評価額(円)", ascending=False)

            st.dataframe(
                holdings_df.style.format(
                    {
                        "評価額(円)": "¥{:,.0f}",
                        "構成比": "{:.1f}%",
                        "損益(円)": "¥{:,.0f}",
                        "損益率(%)": "{:+.1f}%",
                    }
                )
                .background_gradient(
                    subset=["損益率(%)"],
                    cmap="RdYlGn",
                    vmin=-30,
                    vmax=30,
                )
                .map(
                    lambda v: (
                        "color: #4ade80"
                        if isinstance(v, int | float) and v > 0
                        else ("color: #f87171" if isinstance(v, int | float) and v < 0 else "")
                    ),
                    subset=["損益(円)"],
                ),
                width="stretch",
                height=400,
            )

            # CSVダウンロード
            csv_data = holdings_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 保有一覧をCSVダウンロード",
                data=csv_data,
                file_name=f"holdings_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with col_right:
        st.markdown("### 🥧 セクター構成")
        st.caption("セクター別の配分比率。特定業種への偏りがないか確認しましょう。")

        sector_df = get_sector_breakdown(snapshot)
        if not sector_df.empty:
            fig_sector = build_sector_chart(sector_df)
            st.plotly_chart(fig_sector, key="chart_sector")
        else:
            st.info("セクターデータなし")

        # 通貨別エクスポージャー
        st.markdown("### 💱 通貨別配分")
        st.caption("通貨エクスポージャーの確認。為替リスクの偏りを把握できます。")
        fig_cur = build_currency_chart(positions)
        if fig_cur is not None:
            st.plotly_chart(fig_cur, key="chart_currency")

    # --- 構成比ツリーマップ（フルワイド表示） ---
    st.markdown("### 🌳 構成比ツリーマップ")
    st.caption("銘柄の評価額を面積で表現。大きいほど構成比が高く、ポートフォリオ全体像を直感的に把握できます。")
    fig_treemap = build_treemap_chart(positions)
    if fig_treemap is not None:
        st.plotly_chart(fig_treemap, width="stretch", key="chart_treemap")
    else:
        st.info("ツリーマップの表示に必要なデータがありません")

    # --- ウェイトドリフト警告 ---
    drift_alerts = compute_weight_drift(positions, total_value)
    if drift_alerts:
        st.markdown("### ⚖️ ウェイトドリフト警告")
        st.caption("均等配分からの乖離が大きい銘柄を表示。値上がりで膨らんだ銘柄のリバランス検討に活用できます。")
        drift_cols = st.columns(min(len(drift_alerts), 4))
        for i, alert in enumerate(drift_alerts[:4]):
            with drift_cols[i]:
                if alert["status"] == "overweight":
                    icon = "🔺"
                    color = "#f59e0b"
                    label = "オーバーウェイト"
                else:
                    icon = "🔻"
                    color = "#6366f1"
                    label = "アンダーウェイト"
                st.markdown(
                    f'<div class="kpi-card kpi-risk" style="text-align:center;">'
                    f'<span style="font-size:0.8rem; opacity:0.7;">{icon} {label}</span><br>'
                    f'<span style="font-size:1.1rem; font-weight:600;">{_html_mod.escape(str(alert["name"]))}</span><br>'
                    f'<span style="font-size:0.85rem;">現在 {alert["current_pct"]:.1f}% '
                    f"→ 目標 {alert['target_pct']:.1f}%</span><br>"
                    f'<span style="font-size:1.0rem; font-weight:600; color:{color};">'
                    f"{alert['drift_pct']:+.1f}pp</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # --- 銘柄間相関ヒートマップ — プログレッシブ開示（折りたたみ）
    # Why: 相関行列は保有銘柄が多いほど大きく、毎回表示すると画面が長くなる。
    #      折りたたみでデフォルト非表示にし、分散リスクを確認したいときだけ展開。
    if not history_df.empty:
        corr_matrix = compute_correlation_matrix(history_df)
        if not corr_matrix.empty:
            _n = len(corr_matrix)
            _high_corr = int((corr_matrix.where(corr_matrix >= 0.7).stack().dropna() != 1.0).sum() // 2)
            _corr_label = f"🔗 銘柄間 日次リターン相関（{_n}×{_n}）"
            if _high_corr > 0:
                _corr_label += f" — ⚠️ 高相関ペア {_high_corr}"
            with st.expander(_corr_label, expanded=False):
                st.caption(
                    "銘柄同士の値動きの連動性を表示。相関が高い銘柄が多いと分散効果が薄れるため、確認が重要です。"
                    "（相関 ≥ 0.7 のセルは橙色〜赤色で強調）"
                )
                fig_corr = build_correlation_chart(corr_matrix)
                if fig_corr is not None:
                    st.plotly_chart(fig_corr, width="stretch", key="chart_correlation")
