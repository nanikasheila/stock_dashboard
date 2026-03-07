"""月次サマリー & 売買アクティビティ タブ描画モジュール.

``app.py`` の ``with _tab_monthly:`` ブロックを切り出したモジュール。
月次収支チャート / 月次サマリーテーブル / 売買アクティビティ / 取引フォームを描画する。

公開 API
--------
render_monthly_tab(...)
    月次サマリー & 売買タブのコンテンツを描画する。
"""

from __future__ import annotations

import streamlit as st

from components.charts import build_monthly_chart, build_trade_flow_chart
from components.data_loader import get_monthly_summary
from components.trade_form import render_trade_form


def render_monthly_tab(
    *,
    history_df,
    snapshot: dict,
    trade_act_df,
    settings: dict,
) -> None:
    """月次サマリー & 売買タブのコンテンツを描画する.

    Parameters
    ----------
    history_df:    ポートフォリオ価格履歴 DataFrame
    snapshot:      ポートフォリオスナップショット
    trade_act_df:  月次売買アクティビティ DataFrame（``get_trade_activity()`` 戻り値）
    settings:      現在の設定 dict（``render_trade_form`` に渡す）
    """
    # =====================================================================
    # 月次サマリー
    # =====================================================================
    st.markdown('<div id="monthly" role="region" aria-label="月次収支"></div>', unsafe_allow_html=True)
    st.markdown("### 📅 月次サマリー")
    st.caption("月末時点の評価額と前月比変動率を一覧表示。月単位でのパフォーマンス傾向を確認できます。")

    if not history_df.empty:
        monthly_df = get_monthly_summary(history_df)
        if not monthly_df.empty:
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig_monthly = build_monthly_chart(monthly_df)
                st.plotly_chart(fig_monthly, key="chart_monthly")

            with col_table:
                display_cols = ["month_end_value_jpy", "change_pct"]
                col_names = {"month_end_value_jpy": "月末評価額(円)", "change_pct": "前月比(%)"}
                fmt = {"月末評価額(円)": "¥{:,.0f}", "前月比(%)": "{:+.1f}%"}
                if "invested_jpy" in monthly_df.columns:
                    display_cols.insert(1, "invested_jpy")
                    col_names["invested_jpy"] = "投資額(円)"
                    fmt["投資額(円)"] = "¥{:,.0f}"
                if "yoy_pct" in monthly_df.columns:
                    display_cols.append("yoy_pct")
                    col_names["yoy_pct"] = "前年同月比(%)"
                    fmt["前年同月比(%)"] = "{:+.1f}%"
                if "unrealized_pnl" in monthly_df.columns:
                    display_cols.append("unrealized_pnl")
                    col_names["unrealized_pnl"] = "含み損益(円)"
                    fmt["含み損益(円)"] = "¥{:,.0f}"
                display_monthly = monthly_df[display_cols].rename(columns=col_names)
                st.dataframe(
                    display_monthly.style.format(fmt),
                    width="stretch",
                )
                # 月次CSVダウンロード
                import time

                monthly_csv = display_monthly.to_csv().encode("utf-8-sig")
                st.download_button(
                    "📥 月次サマリーをCSVダウンロード",
                    data=monthly_csv,
                    file_name=f"monthly_summary_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("月次データなし（データ期間が短い可能性があります）")
    else:
        st.info("履歴データがありません")

    # =====================================================================
    # 取引アクティビティ
    # =====================================================================
    st.markdown('<div id="trade-activity" role="region" aria-label="売買入力"></div>', unsafe_allow_html=True)
    st.markdown("### 🔄 月次売買アクティビティ")
    _trade_as_of = str(trade_act_df.index[-1])[:7] if not trade_act_df.empty else "—"
    st.caption(
        f"月ごとの売買件数・金額フローを表示。投資ペースや資金の出入りを振り返るのに便利です。｜ 🕐 最終データ月: {_trade_as_of}"
    )
    if not trade_act_df.empty:
        col_flow, col_tbl = st.columns([2, 1])

        with col_flow:
            fig_flow = build_trade_flow_chart(trade_act_df)
            st.plotly_chart(fig_flow, key="chart_trade_flow")

        with col_tbl:
            display_act = trade_act_df.copy()
            display_act.columns = ["購入件数", "購入額(円)", "売却件数", "売却額(円)", "ネット(円)"]
            st.dataframe(
                display_act.style.format(
                    {
                        "購入件数": "{:.0f}",
                        "購入額(円)": "¥{:,.0f}",
                        "売却件数": "{:.0f}",
                        "売却額(円)": "¥{:,.0f}",
                        "ネット(円)": "¥{:,.0f}",
                    }
                ),
                width="stretch",
            )
            # 売買アクティビティ CSVダウンロード
            import time as _time_act

            act_csv = display_act.to_csv().encode("utf-8-sig")
            st.download_button(
                "📥 売買アクティビティをCSVダウンロード",
                data=act_csv,
                file_name=f"trade_activity_{_time_act.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_trade_activity",
            )
    else:
        st.info("取引データがありません")

    render_trade_form(snapshot=snapshot, settings=settings)
