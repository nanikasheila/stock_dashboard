"""ポートフォリオダッシュボード — チャート構築モジュール.

Plotly の Figure を構築して返す関数群。
st.plotly_chart() の呼び出しは app.py 側で行う。
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================================
# 総資産推移チャート
# =====================================================================

def build_total_chart(
    history_df: pd.DataFrame,
    chart_style: str,
    benchmark_series: pd.Series | None = None,
    benchmark_label: str = "",
) -> go.Figure:
    """総資産推移チャートを構築して返す.

    Parameters
    ----------
    history_df : pd.DataFrame
        build_portfolio_history() の出力
    chart_style : str
        "積み上げ面" | "折れ線" | "積み上げ棒"
    benchmark_series : pd.Series | None
        ベンチマークの正規化済み系列
    benchmark_label : str
        ベンチマーク表示名

    Returns
    -------
    go.Figure
    """
    stock_cols = [c for c in history_df.columns if c not in ("total", "invested")]

    if chart_style == "積み上げ面":
        fig = go.Figure()
        for col in stock_cols:
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[col],
                mode="lines",
                stackgroup="one",
                name=col,
                hovertemplate="%{x}<br>%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig.update_layout(
            title="保有銘柄別 評価額推移（積み上げ面グラフ）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    elif chart_style == "折れ線":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["total"],
            mode="lines",
            name="合計",
            line=dict(width=3, color="#fbbf24"),
            hovertemplate="合計: ¥%{y:,.0f}<extra></extra>",
        ))
        for col in stock_cols:
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df[col],
                mode="lines",
                name=col,
                hovertemplate="%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig.update_layout(
            title="保有銘柄別 評価額推移（折れ線グラフ）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    else:  # 積み上げ棒
        weekly = history_df[stock_cols].resample("W").last().ffill()
        fig = go.Figure()
        for col in stock_cols:
            fig.add_trace(go.Bar(
                x=weekly.index,
                y=weekly[col],
                name=col,
                hovertemplate="%{fullData.name}: ¥%{y:,.0f}<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            title="保有銘柄別 評価額推移（積み上げ棒グラフ・週次）",
            xaxis_title="日付",
            yaxis_title="評価額（円）",
            hovermode="x unified",
            height=500,
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

    # ベンチマーク重ね描き
    if benchmark_series is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_series.index,
            y=benchmark_series.values,
            mode="lines",
            name=f"BM: {benchmark_label}",
            line=dict(width=2, color="#94a3b8", dash="dash"),
            hovertemplate=f"{benchmark_label}: ¥%{{y:,.0f}}<extra></extra>",
        ))

    return fig


# =====================================================================
# 投資額 vs 評価額チャート
# =====================================================================

def build_invested_chart(history_df: pd.DataFrame) -> go.Figure:
    """投資額 vs 評価額チャートを構築して返す."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["total"],
        mode="lines",
        name="評価額",
        line=dict(width=2, color="#60a5fa"),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.15)",
        hovertemplate="評価額: ¥%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["invested"],
        mode="lines",
        name="累積投資額",
        line=dict(width=2, color="#f59e0b", dash="dot"),
        hovertemplate="投資額: ¥%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="日付",
        yaxis_title="金額（円）",
        hovermode="x unified",
        height=400,
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# =====================================================================
# プロジェクションチャート
# =====================================================================

def build_projection_chart(
    history_df: pd.DataFrame,
    projection_df: pd.DataFrame,
    target_amount: float = 0,
) -> go.Figure:
    """将来推定推移チャートを構築して返す."""
    fig = go.Figure()

    # 過去の実績
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df["total"],
        mode="lines",
        name="実績（総資産）",
        line=dict(width=2.5, color="#60a5fa"),
        hovertemplate="実績: ¥%{y:,.0f}<extra></extra>",
    ))

    # 投資額
    if "invested" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df["invested"],
            mode="lines",
            name="累積投資額",
            line=dict(width=1.5, color="#f59e0b", dash="dot"),
            hovertemplate="投資額: ¥%{y:,.0f}<extra></extra>",
        ))

    # 楽観シナリオ
    fig.add_trace(go.Scatter(
        x=projection_df.index,
        y=projection_df["optimistic"],
        mode="lines",
        name="楽観",
        line=dict(width=1.5, color="#4ade80", dash="dash"),
        hovertemplate="楽観: ¥%{y:,.0f}<extra></extra>",
    ))

    # 悲観シナリオ（fill between 用に先に追加）
    fig.add_trace(go.Scatter(
        x=projection_df.index,
        y=projection_df["pessimistic"],
        mode="lines",
        name="悲観",
        line=dict(width=1.5, color="#f87171", dash="dash"),
        fill="tonexty",
        fillcolor="rgba(148, 163, 184, 0.1)",
        hovertemplate="悲観: ¥%{y:,.0f}<extra></extra>",
    ))

    # ベースシナリオ
    fig.add_trace(go.Scatter(
        x=projection_df.index,
        y=projection_df["base"],
        mode="lines",
        name="ベース",
        line=dict(width=2, color="#a78bfa"),
        hovertemplate="ベース: ¥%{y:,.0f}<extra></extra>",
    ))

    # 目標資産ライン
    if target_amount > 0:
        x_all_start = history_df.index[0]
        x_all_end = projection_df.index[-1]
        fig.add_trace(go.Scatter(
            x=[x_all_start, x_all_end],
            y=[target_amount, target_amount],
            mode="lines",
            name=f"目標: ¥{target_amount:,.0f}",
            line=dict(width=2, color="#fbbf24", dash="dashdot"),
            hovertemplate="目標: ¥%{y:,.0f}<extra></extra>",
        ))

        # 目標到達予想時期
        reach_date = None
        for d, row in projection_df.iterrows():
            if row["base"] >= target_amount:
                reach_date = d
                break
        if reach_date is not None:
            fig.add_annotation(
                x=reach_date,
                y=target_amount,
                text=f"目標到達予想: {reach_date.strftime('%Y/%m')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#fbbf24",
                font=dict(size=11, color="#fbbf24"),
                ax=0,
                ay=-35,
            )

    fig.update_layout(
        xaxis_title="日付",
        yaxis_title="金額（円）",
        hovermode="x unified",
        height=550,
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


# =====================================================================
# セクター構成パイチャート
# =====================================================================

def build_sector_chart(sector_df: pd.DataFrame) -> go.Figure:
    """セクター構成ドーナツチャートを構築して返す."""
    fig = px.pie(
        sector_df,
        values="evaluation_jpy",
        names="sector",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


# =====================================================================
# 通貨別配分パイチャート
# =====================================================================

def build_currency_chart(positions: list[dict]) -> go.Figure | None:
    """通貨別エクスポージャーのドーナツチャートを構築して返す."""
    currency_data: dict[str, float] = {}
    for p in positions:
        cur = p.get("currency", "JPY") or "JPY"
        currency_data[cur] = currency_data.get(cur, 0) + p.get("evaluation_jpy", 0)
    if not currency_data:
        return None

    cur_df = pd.DataFrame([
        {"currency": k, "evaluation_jpy": v}
        for k, v in currency_data.items()
    ])
    _cur_colors = {"JPY": "#60a5fa", "USD": "#4ade80", "SGD": "#fbbf24", "HKD": "#f87171"}
    fig = px.pie(
        cur_df,
        values="evaluation_jpy",
        names="currency",
        hole=0.45,
        color="currency",
        color_discrete_map=_cur_colors,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="%{label}<br>¥%{value:,.0f}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(height=300, showlegend=False)
    return fig


# =====================================================================
# 銘柄別個別チャート
# =====================================================================

def build_individual_chart(
    history_df: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    """個別銘柄の評価額推移チャートを構築して返す."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df[symbol],
        mode="lines",
        fill="tozeroy",
        name=symbol,
        line=dict(width=2),
        hovertemplate="¥%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=symbol,
        height=250,
        margin=dict(l=40, r=20, t=40, b=30),
        yaxis=dict(tickformat=","),
        showlegend=False,
    )
    return fig


# =====================================================================
# 月次サマリーチャート
# =====================================================================

def build_monthly_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """月次サマリーの棒グラフを構築して返す."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_df.index,
        y=monthly_df["month_end_value_jpy"],
        name="月末評価額",
        marker_color=[
            "#4ade80" if v >= 0 else "#f87171"
            for v in monthly_df["change_pct"].fillna(0)
        ],
        hovertemplate="月末資産: ¥%{y:,.0f}<extra></extra>",
    ))
    if "invested_jpy" in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df.index,
            y=monthly_df["invested_jpy"],
            name="累積投資額",
            mode="lines",
            line=dict(width=2, color="#f59e0b", dash="dot"),
            hovertemplate="投資額: ¥%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        title="月末資産額の推移",
        xaxis_title="月",
        yaxis_title="評価額（円）",
        height=350,
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    return fig


# =====================================================================
# 売買フローチャート
# =====================================================================

def build_trade_flow_chart(trade_act_df: pd.DataFrame) -> go.Figure:
    """月次売買フローチャートを構築して返す."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trade_act_df.index,
        y=trade_act_df["buy_amount"],
        name="購入額",
        marker_color="#60a5fa",
        hovertemplate="購入: ¥%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=trade_act_df.index,
        y=-trade_act_df["sell_amount"],
        name="売却額",
        marker_color="#f87171",
        hovertemplate="売却: ¥%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trade_act_df.index,
        y=trade_act_df["net_flow"],
        name="ネットフロー",
        mode="lines+markers",
        line=dict(color="#fbbf24", width=2),
        hovertemplate="ネット: ¥%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="月次売買フロー",
        xaxis_title="月",
        yaxis_title="金額（円）",
        barmode="relative",
        height=350,
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    return fig


# =====================================================================
# ドローダウンチャート
# =====================================================================

def build_drawdown_chart(drawdown_series: "pd.Series") -> go.Figure:
    """ドローダウン（水面下）チャートを構築して返す.

    Parameters
    ----------
    drawdown_series : pd.Series
        compute_drawdown_series() の出力（%、0以下の値）

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode="lines",
        fill="tozeroy",
        name="ドローダウン",
        line=dict(color="#f87171", width=1.5),
        fillcolor="rgba(248,113,113,0.2)",
        hovertemplate="%{x}<br>DD: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="ドローダウン（ピークからの下落率）",
        xaxis_title="日付",
        yaxis_title="ドローダウン（%）",
        height=280,
        yaxis=dict(ticksuffix="%"),
        showlegend=False,
    )
    return fig


# =====================================================================
# ローリングSharpe比チャート
# =====================================================================

def build_rolling_sharpe_chart(
    rolling_sharpe: "pd.Series",
    window: int = 60,
) -> go.Figure:
    """ローリングSharpe比のチャートを構築して返す.

    Parameters
    ----------
    rolling_sharpe : pd.Series
        compute_rolling_sharpe() の出力
    window : int
        ウィンドウサイズ（表示用）

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode="lines",
        name=f"Sharpe ({window}日)",
        line=dict(color="#a78bfa", width=2),
        hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    # 基準ライン
    fig.add_hline(y=1.0, line_dash="dash", line_color="#4ade80",
                  annotation_text="1.0（良好）", annotation_position="bottom right")
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(148,163,184,0.5)")
    fig.update_layout(
        title=f"ローリングSharpe比（{window}日）",
        xaxis_title="日付",
        yaxis_title="Sharpe比",
        height=280,
        showlegend=False,
    )
    return fig


# =====================================================================
# 構成比ツリーマップ
# =====================================================================

def build_treemap_chart(positions: list[dict]) -> go.Figure | None:
    """保有銘柄のセクター×銘柄ツリーマップを構築して返す.

    Parameters
    ----------
    positions : list[dict]
        get_current_snapshot()["positions"]

    Returns
    -------
    go.Figure | None
        銘柄が無い場合は None
    """
    if not positions:
        return None

    labels = []
    parents = []
    values = []
    colors = []

    # セクターを集約
    sector_map: dict[str, list[dict]] = {}
    for p in positions:
        sector = p.get("sector") or "不明"
        sector_map.setdefault(sector, []).append(p)

    for sector, stocks in sector_map.items():
        # セクターノード（親はルート ""）
        labels.append(sector)
        parents.append("")
        sector_value = sum(s["evaluation_jpy"] for s in stocks)
        values.append(sector_value)
        colors.append(0)  # セクターノード自体の色は中立

        for s in stocks:
            label = s.get("name", s["symbol"])
            # 長すぎる名前は短縮
            if len(label) > 15:
                label = label[:13] + "…"
            labels.append(label)
            parents.append(sector)
            values.append(max(s["evaluation_jpy"], 0))
            colors.append(s.get("pnl_pct", 0))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale="RdYlGn",
            cmid=0,
            showscale=True,
            colorbar=dict(title="損益率%", ticksuffix="%"),
        ),
        texttemplate="<b>%{label}</b><br>¥%{value:,.0f}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "評価額: ¥%{value:,.0f}<br>"
            "損益率: %{color:+.1f}%<extra></extra>"
        ),
        branchvalues="total",
    ))
    fig.update_layout(
        title="構成比ツリーマップ（セクター × 銘柄）",
        height=450,
        margin=dict(t=40, l=5, r=5, b=5),
    )
    return fig


# =====================================================================
# 銘柄間相関ヒートマップ
# =====================================================================

def build_correlation_chart(corr_matrix: pd.DataFrame) -> go.Figure | None:
    """銘柄間の日次リターン相関ヒートマップを構築して返す.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        compute_correlation_matrix() の出力

    Returns
    -------
    go.Figure | None
        銘柄が2つ未満の場合は None
    """
    if corr_matrix.empty or len(corr_matrix) < 2:
        return None

    # 小数点2桁の注釈テキスト
    text_vals = [[f"{v:.2f}" for v in row] for row in corr_matrix.values]

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="相関"),
        hovertemplate=(
            "%{x} × %{y}<br>"
            "相関: %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="銘柄間 日次リターン相関",
        height=max(350, len(corr_matrix) * 40 + 100),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig