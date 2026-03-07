"""チャート分析タブ描画モジュール.

``app.py`` の ``with _tab_charts:`` ブロック（2 箇所）を統合して切り出したモジュール。
総資産推移 / ドローダウン / ローリングSharpe / 投資額比較 / 将来推定 /
パフォーマンス寄与 / 銘柄別個別チャートを描画する。

公開 API
--------
render_charts_tab(...)
    チャート分析タブのコンテンツを描画する。
"""

from __future__ import annotations

import streamlit as st

from components.charts import (
    build_attribution_chart,
    build_drawdown_chart,
    build_individual_chart,
    build_invested_chart,
    build_projection_chart,
    build_rolling_sharpe_chart,
    build_total_chart,
)
from components.data_loader import (
    build_projection,
    compute_drawdown_series,
    compute_performance_attribution,
    compute_rolling_sharpe,
    get_benchmark_series,
)
from components.llm_analyzer import generate_attribution_summary
from components.llm_analyzer import is_available as llm_is_available


def render_charts_tab(
    *,
    history_df,
    snapshot: dict,
    total_value: float,
    positions: list[dict],
    period: str,
    chart_style: str,
    show_invested: bool,
    show_projection: bool,
    target_amount: float,
    projection_years: int,
    benchmark_symbol: str | None,
    benchmark_label: str,
    attribution_llm_enabled: bool,
    llm_enabled: bool,
    llm_model: str,
    show_individual: bool = True,
) -> None:
    """チャート分析タブのコンテンツを描画する.

    Parameters
    ----------
    history_df:              ポートフォリオ価格履歴 DataFrame
    snapshot:                ポートフォリオスナップショット
    total_value:             総資産（円換算）
    positions:               保有銘柄リスト
    period:                  表示期間識別子（例: "1mo", "1y"）
    chart_style:             チャートスタイル（"積み上げ面" / "折れ線" / "積み上げ棒"）
    show_invested:           投資額 vs 評価額チャートの表示フラグ
    show_projection:         将来推定チャートの表示フラグ
    show_individual:         銘柄別個別チャートの表示フラグ（デフォルト True）
    target_amount:           目標資産額（円）
    projection_years:        将来推定期間（年）
    benchmark_symbol:        ベンチマーク銘柄コード（None で無効）
    benchmark_label:         ベンチマーク表示名
    attribution_llm_enabled: パフォーマンス寄与 LLM 要因分析の有効フラグ
    llm_enabled:             LLM 機能全体の有効フラグ
    llm_model:               使用 LLM モデル識別子
    """
    # =====================================================================
    # 総資産推移グラフ
    # =====================================================================
    st.markdown('<div id="total-chart" role="region" aria-label="チャート"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 総資産推移")
    _history_as_of = str(history_df.index[-1])[:10] if not history_df.empty else "—"
    st.caption(
        f"資産全体の値動きを時系列で確認。ドローダウンやシャープレシオの推移も合わせて表示します。｜ 🕐 最終データ日: {_history_as_of}"
    )

    if not history_df.empty:
        # ベンチマーク系列の取得
        bench_series = None
        if benchmark_symbol:
            bench_series = get_benchmark_series(benchmark_symbol, history_df, period)

        fig_total = build_total_chart(history_df, chart_style, bench_series, benchmark_label)
        st.plotly_chart(fig_total, key="chart_total")

        # ---------------------------------------------------------------
        # ドローダウンチャート
        # ---------------------------------------------------------------
        _dd_series = compute_drawdown_series(history_df)
        if not _dd_series.empty:
            fig_dd = build_drawdown_chart(_dd_series)
            st.plotly_chart(fig_dd, key="chart_drawdown")

        # ---------------------------------------------------------------
        # ローリングSharpe比
        # ---------------------------------------------------------------
        _rolling_window = 60
        _rolling_sharpe = compute_rolling_sharpe(history_df, window=_rolling_window)
        if not _rolling_sharpe.empty:
            fig_rs = build_rolling_sharpe_chart(_rolling_sharpe, window=_rolling_window)
            st.plotly_chart(fig_rs, key="chart_rolling_sharpe")

        # ---------------------------------------------------------------
        # 投資額 vs 評価額
        # ---------------------------------------------------------------
        if show_invested and "invested" in history_df.columns:
            st.markdown('<div id="invested-chart"></div>', unsafe_allow_html=True)
            st.markdown("### 💰 投資額 vs 評価額")
            st.caption("累計投資額と現在の評価額を比較し、投入資金に対するリターンを視覚的に確認できます。")
            fig_inv = build_invested_chart(history_df)
            st.plotly_chart(fig_inv, key="chart_invested")

        # ---------------------------------------------------------------
        # 目標ライン & 将来推定推移
        # ---------------------------------------------------------------
        if show_projection:
            st.markdown('<div id="projection"></div>', unsafe_allow_html=True)
            st.markdown("### 🔮 総資産推移 & 将来推定")
            st.caption("過去のリターン実績をもとに、楽観・基本・悲観の3シナリオで将来の資産推移を推計します。")

            projection_df = build_projection(
                current_value=total_value,
                years=projection_years,
            )

            fig_proj = build_projection_chart(history_df, projection_df, target_amount)
            st.plotly_chart(fig_proj, key="chart_projection")

            # 推定リターンのサマリー
            opt_val = projection_df["optimistic"].iloc[-1]
            base_val = projection_df["base"].iloc[-1]
            pess_val = projection_df["pessimistic"].iloc[-1]
            opt_rate = (opt_val / total_value - 1) * 100
            base_rate_pct = (base_val / total_value - 1) * 100
            pess_rate = (pess_val / total_value - 1) * 100

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown(
                    f'<div style="text-align:center; padding:8px;">'
                    f'<span style="font-size:0.85rem; opacity:0.7;">🟢 楽観（{projection_years}年後）</span><br>'
                    f'<span style="font-size:1.3rem; font-weight:600; color:#4ade80;">'
                    f"¥{opt_val:,.0f}</span><br>"
                    f'<span style="font-size:0.8rem; color:#4ade80;">{opt_rate:+.1f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with scol2:
                st.markdown(
                    f'<div style="text-align:center; padding:8px;">'
                    f'<span style="font-size:0.85rem; opacity:0.7;">🟣 ベース（{projection_years}年後）</span><br>'
                    f'<span style="font-size:1.3rem; font-weight:600; color:#a78bfa;">'
                    f"¥{base_val:,.0f}</span><br>"
                    f'<span style="font-size:0.8rem; color:#a78bfa;">{base_rate_pct:+.1f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with scol3:
                st.markdown(
                    f'<div style="text-align:center; padding:8px;">'
                    f'<span style="font-size:0.85rem; opacity:0.7;">🔴 悲観（{projection_years}年後）</span><br>'
                    f'<span style="font-size:1.3rem; font-weight:600; color:#f87171;">'
                    f"¥{pess_val:,.0f}</span><br>"
                    f'<span style="font-size:0.8rem; color:#f87171;">{pess_rate:+.1f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------------------
        # パフォーマンス寄与分析 — プログレッシブ開示（折りたたみ）
        # Why: 寄与分析チャートは詳細レビュー時のみ必要。折りたたみで
        #      主要チャートへの集中を維持しつつ展開時に全情報を提供。
        # ---------------------------------------------------------------
        with st.expander("📈 パフォーマンス寄与分析", expanded=False):
            st.caption("ポートフォリオ全体のリターンに対する各銘柄・セクターの寄与度を確認します。")

            _attribution = compute_performance_attribution(snapshot)
            if _attribution and _attribution.get("by_stock"):
                _attr_col1, _attr_col2 = st.columns(2)
                with _attr_col1:
                    st.markdown("#### 銘柄別")
                    fig_attr_stock = build_attribution_chart(_attribution, by="stock")
                    st.plotly_chart(fig_attr_stock, key="chart_attr_stock", use_container_width=True)
                with _attr_col2:
                    st.markdown("#### セクター別")
                    fig_attr_sector = build_attribution_chart(_attribution, by="sector")
                    st.plotly_chart(fig_attr_sector, key="chart_attr_sector", use_container_width=True)

                if attribution_llm_enabled and llm_enabled and llm_is_available():
                    if st.button("🤖 AI 要因分析", key="btn_attr_llm"):
                        with st.spinner("AI が要因を分析中..."):
                            _attr_summary = generate_attribution_summary(_attribution)
                            if _attr_summary:
                                st.info(_attr_summary)
                            else:
                                st.warning("要因分析の生成に失敗しました。")
            else:
                st.info("寄与分析に必要なデータがありません。")

        # ---------------------------------------------------------------
        # 銘柄別個別チャート — プログレッシブ開示（折りたたみ）
        # Why: 銘柄数が多い場合にページが長くなりすぎる。デフォルト折りたたみで
        #      メインチャートへの視線集中を維持しつつ、必要なときだけ展開できる。
        # ---------------------------------------------------------------
        if show_individual and not history_df.empty:
            stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]
            _ind_label = f"📉 銘柄別 個別推移（{len(stock_cols)} 銘柄）"
            with st.expander(_ind_label, expanded=False):
                st.caption("各銘柄の評価額推移を個別に確認。特定銘柄の値動きパターンを詳しく見たいときに。")
                cols_per_row = 2
                for i in range(0, len(stock_cols), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col_widget in enumerate(cols):
                        idx = i + j
                        if idx >= len(stock_cols):
                            break
                        symbol = stock_cols[idx]
                        with col_widget:
                            fig_ind = build_individual_chart(history_df, symbol)
                            st.plotly_chart(fig_ind, key=f"chart_ind_{symbol}")

    else:
        st.warning("株価履歴データが取得できませんでした。")
