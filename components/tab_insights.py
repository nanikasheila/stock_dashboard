"""インサイトタブ描画モジュール.

蓄積されたトレード履歴・行動分析から得られたポートフォリオ洞察を表示する。
``src.core.behavior`` ドメインの ``BehaviorInsight`` と
``PortfolioTimingInsight`` を直接受け取り、confidence-aware な UI を描画する。

公開 API
--------
render_insights_tab(...)
    インサイトタブのコンテンツを描画する。

セクション構成
--------------
1. トレード統計（Trade Statistics）
   - 売買回数・勝率・プロフィットファクター・平均保有期間・短期売買比率など
2. タイミングレビュー（Timing Review）
   - エントリー/エグジットのタイミング分析スコアと銘柄別レビュー表
3. スタイルプロファイル（Style Profile）
   - 投資スタイルの傾向（軽量プレースホルダー）
4. 長期ホライゾンインサイト（Long-Horizon Insights）
   - 長期視点でのポートフォリオ基本指標
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from state_keys import SK

from components.dl_analytics import (
    compute_benchmark_excess,
    compute_monthly_seasonality,
    compute_rolling_sharpe_trend,
)
from src.core.behavior.models import (
    BehaviorInsight,
    BiasSignal,
    ConfidenceLevel,
    PortfolioTimingInsight,
    StyleProfile,
)

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Confidence-level UI helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_MESSAGES: dict[str, tuple[str, str]] = {
    ConfidenceLevel.INSUFFICIENT: (
        "📭 トレード履歴が不足しています。",
        "売買記録が蓄積されると、ここに詳細な統計が表示されます。",
    ),
    ConfidenceLevel.LOW: (
        "⚠️ データ量が少なく、精度は限定的です。",
        "売買回数が増えるほど、より信頼性の高いインサイトが得られます。",
    ),
    ConfidenceLevel.MEDIUM: (
        "📊 一定量のデータに基づく分析です。",
        "さらに売買を積み重ねると、精度が向上します。",
    ),
    ConfidenceLevel.HIGH: (
        "✅ 十分なデータに基づく信頼性の高い分析です。",
        "",
    ),
}


def _render_confidence_badge(confidence: ConfidenceLevel) -> None:
    """データ信頼度バッジを描画する."""
    msg, sub = _CONFIDENCE_MESSAGES.get(confidence, ("", ""))
    if confidence in (ConfidenceLevel.INSUFFICIENT, ConfidenceLevel.LOW):
        st.info(f"{msg}{('  ' + sub) if sub else ''}", icon=None)
    elif confidence == ConfidenceLevel.MEDIUM and sub:
        st.caption(f"{msg}  {sub}")


# ---------------------------------------------------------------------------
# セクション内部ヘルパー
# ---------------------------------------------------------------------------


def _render_trade_statistics_section(behavior_insight: BehaviorInsight) -> None:
    """トレード統計セクションを描画する.

    Parameters
    ----------
    behavior_insight:
        ``load_behavior_insight()`` が返す ``BehaviorInsight`` オブジェクト。
        confidence が ``INSUFFICIENT`` のときはデータ未蓄積メッセージを表示する。
    """
    st.markdown("#### 📊 トレード統計")
    st.caption("過去トレードの集計サマリー — 勝率・プロフィットファクター・平均保有期間など")

    ts = behavior_insight.trade_stats
    wl = behavior_insight.win_loss
    hp = behavior_insight.holding_period

    if behavior_insight.confidence == ConfidenceLevel.INSUFFICIENT:
        st.info(
            "📭 トレード履歴が蓄積されると、ここに統計情報が表示されます。\n\n"
            "**項目:** 総トレード数 / 勝率 / プロフィットファクター / 平均保有期間 / 短期売買比率",
            icon=None,
        )
        return

    # --- サマリーメトリクス (2行 × 3列) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("売買回数（合計）", ts.total_trades)
    with c2:
        st.metric("買い回数", ts.total_buy_count)
    with c3:
        st.metric("売り回数", ts.total_sell_count)

    st.markdown("")

    c4, c5, c6 = st.columns(3)
    with c4:
        wr = ts.overall_win_rate
        st.metric(
            "勝率",
            f"{wr * 100:.1f}%" if wr is not None else "—",
            help="売りトレードのうち利益が出たものの割合",
        )
    with c5:
        pf = wl.profit_factor
        _pf_delta = None
        if pf is not None:
            _pf_delta = "良好" if pf >= 1.5 else ("基準以上" if pf >= 1.0 else "要改善")
        st.metric(
            "プロフィットファクター",
            f"{pf:.2f}" if pf is not None else "—",
            delta=_pf_delta,
            help="総利益 ÷ 総損失。1.0 以上で全体として利益",
        )
    with c6:
        ahd = ts.avg_hold_days
        st.metric(
            "平均保有期間",
            f"{ahd:.0f} 日" if ahd is not None else "—",
            help="クローズポジションの平均保有カレンダー日数",
        )

    st.markdown("")

    # --- 保有期間分布・勝敗内訳 ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**⏳ 保有期間分布**")
        if hp.total_with_hold_data > 0:
            st.caption(
                f"短期（〜30日）: **{hp.short_term_count}** 件 ／ "
                f"中期（30〜180日）: **{hp.medium_term_count}** 件 ／ "
                f"長期（180日〜）: **{hp.long_term_count}** 件"
            )
            if hp.short_term_ratio is not None:
                st.progress(
                    hp.short_term_ratio,
                    text=f"短期売買比率: {hp.short_term_ratio * 100:.0f}%",
                )
            if hp.median_days is not None:
                st.caption(
                    f"中央値: **{hp.median_days:.0f}** 日  （P25: {hp.p25_days:.0f} 日 ／ P75: {hp.p75_days:.0f} 日）"
                    if hp.p25_days is not None and hp.p75_days is not None
                    else f"中央値: **{hp.median_days:.0f}** 日"
                )
        else:
            st.caption("保有期間データが不足しています。")

    with col_b:
        st.markdown("**🏆 勝敗内訳**")
        total_closed = wl.win_count + wl.loss_count
        if total_closed > 0:
            st.caption(f"勝ち: **{wl.win_count}** 件 ／ 負け: **{wl.loss_count}** 件")
            st.progress(
                wl.win_count / total_closed,
                text=f"勝ち比率: {wl.win_count}/{total_closed}",
            )
            if wl.avg_win_jpy is not None:
                st.caption(f"平均利益: ¥{wl.avg_win_jpy:+,.0f}")
            if wl.avg_loss_jpy is not None:
                st.caption(f"平均損失: ¥{wl.avg_loss_jpy:+,.0f}")
            pnl_color = "normal" if ts.total_realized_pnl_jpy >= 0 else "inverse"
            st.metric(
                "累積実現損益",
                f"¥{ts.total_realized_pnl_jpy:+,.0f}",
                delta_color=pnl_color,
            )
        else:
            st.caption("クローズ済みトレードがありません。")

    # --- 信頼度バッジ ---
    _render_confidence_badge(behavior_insight.confidence)


def _render_timing_review_section(timing_insight: PortfolioTimingInsight) -> None:
    """タイミングレビューセクションを描画する.

    Parameters
    ----------
    timing_insight:
        ``load_timing_insight()`` が返す ``PortfolioTimingInsight`` オブジェクト。
        confidence が ``INSUFFICIENT`` のときはデータ未蓄積メッセージを表示する。
    """
    st.markdown("#### ⏱️ タイミングレビュー")
    st.caption(
        "エントリー・エグジットのタイミング品質を 0–100 スコアで評価します。スコアが高いほど理想的なタイミングに近い。"
    )

    if not timing_insight.trade_results:
        st.info(
            "📭 トレード履歴が蓄積されると、ここにタイミング分析が表示されます。\n\n"
            "**項目:** 買い/売りタイミングスコア（平均）/ 銘柄別レビュー表 / タイミング改善ヒント",
            icon=None,
        )
        return

    # --- サマリースコア ---
    buy_score = timing_insight.avg_buy_timing_score
    sell_score = timing_insight.avg_sell_timing_score

    cs1, cs2 = st.columns(2)
    with cs1:
        if buy_score is not None:
            _buy_delta = "良好" if buy_score >= 60 else ("要改善" if buy_score < 40 else None)
            st.metric(
                "平均買いタイミングスコア",
                f"{buy_score:.0f} / 100",
                delta=_buy_delta,
                help="0=最悪のタイミング、100=最良のタイミング（価格パーセンタイル・RSI・SMA乖離の加重平均）",
            )
        else:
            st.metric("平均買いタイミングスコア", "—")
    with cs2:
        if sell_score is not None:
            _sell_delta = "良好" if sell_score >= 60 else ("要改善" if sell_score < 40 else None)
            st.metric(
                "平均売りタイミングスコア",
                f"{sell_score:.0f} / 100",
                delta=_sell_delta,
            )
        else:
            st.metric("平均売りタイミングスコア", "—")

    # --- サマリーノート ---
    for note in timing_insight.notes:
        st.caption(f"📌 {note}")

    st.markdown("")

    # --- 銘柄別タイミング表（最新 20 件）---
    results = timing_insight.trade_results
    if results:
        st.markdown("**📋 直近トレードのタイミングレビュー**（最新 20 件）")

        # Sort by date descending (latest first), take top 20
        sorted_results = sorted(results, key=lambda r: r.trade_date, reverse=True)[:20]

        _LABEL_JA = {
            "excellent": "🟢 優秀",
            "good": "🟩 良好",
            "neutral": "🟡 中立",
            "poor": "🟠 要改善",
            "very_poor": "🔴 不良",
        }
        _TYPE_JA = {"buy": "買い", "sell": "売り"}

        rows = []
        for r in sorted_results:
            rows.append(
                {
                    "日付": r.trade_date,
                    "銘柄": r.symbol,
                    "区分": _TYPE_JA.get(r.trade_type, r.trade_type),
                    "約定価格": f"{r.trade_price:,.2f}",
                    "スコア": f"{r.timing_score:.0f}",
                    "評価": _LABEL_JA.get(r.label, r.label),
                    "信頼度": r.confidence.value,
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    # --- 信頼度バッジ ---
    _render_confidence_badge(timing_insight.confidence)


def _render_style_profile_section(
    positions: list[dict],
    behavior_insight: BehaviorInsight,
    style_profile: StyleProfile | None = None,
    style_biases: list[BiasSignal] | None = None,
) -> None:
    """スタイルプロファイルセクションを描画する.

    ADI スコア・スタイルラベル・コンポーネント内訳・バイアス検出結果を
    confidence-aware に表示する。

    Parameters
    ----------
    positions:
        現在の保有銘柄リスト。銘柄数表示に使用する。
    behavior_insight:
        ``BehaviorInsight`` オブジェクト。``style_metrics`` から
        基本スタイル情報を取得する。
    style_profile:
        ``load_style_profile_insight()`` が返す ``StyleProfile``。
        ``None`` の場合は基本情報のみ表示する。
    style_biases:
        ``load_style_profile_insight()`` が返す ``list[BiasSignal]``。
        ``None`` または空リストの場合はバイアスセクションを省略する。
    """
    st.markdown("#### 🎨 スタイルプロファイル")
    st.caption("投資スタイルの傾向（攻め/守り傾向・集中度・現金比率・ボラティリティ）を分析します。")

    sm = behavior_insight.style_metrics
    ts = behavior_insight.trade_stats

    # プロファイルデータが何もない場合
    if style_profile is None and behavior_insight.confidence == ConfidenceLevel.INSUFFICIENT and not positions:
        st.info(
            "📭 保有銘柄またはトレード履歴が蓄積されると、スタイル分析が表示されます。\n\n"
            "**項目:** 攻め/守り傾向スコア / 現金比率 / 集中度 / ポジションバイアス検出",
            icon=None,
        )
        return

    # ------------------------------------------------------------------ #
    # Section A: ADI スコアと基本情報
    # ------------------------------------------------------------------ #
    if style_profile is not None:
        _render_adi_section(style_profile)
    else:
        # Fallback: basic style info from BehaviorInsight
        _render_basic_style_section(positions, sm, ts)

    st.markdown("")

    # ------------------------------------------------------------------ #
    # Section B: コンポーネント内訳
    # ------------------------------------------------------------------ #
    if style_profile is not None and style_profile.component_scores:
        _render_component_breakdown(style_profile)
        st.markdown("")

    # ------------------------------------------------------------------ #
    # Section C: バイアス検出
    # ------------------------------------------------------------------ #
    if style_biases:
        _render_bias_signals(style_biases)
    elif style_profile is not None and style_profile.confidence != ConfidenceLevel.INSUFFICIENT:
        st.caption("✅ 主要なバイアスは検出されませんでした。")

    # 信頼度バッジ
    conf = style_profile.confidence if style_profile is not None else behavior_insight.confidence
    _render_confidence_badge(conf)


# ---------------------------------------------------------------------------
# Style profile sub-renderers
# ---------------------------------------------------------------------------


def _render_adi_section(style_profile: StyleProfile) -> None:
    """ADI スコア・ラベル・主要指標を描画する."""
    _LABEL_EMOJI = {
        "aggressive": "🔴",
        "balanced": "🟡",
        "defensive": "🔵",
    }
    emoji = _LABEL_EMOJI.get(style_profile.label, "⚪")

    col_score, col_metrics = st.columns([1, 2])

    with col_score:
        st.metric(
            "ADI スコア（攻め度）",
            f"{style_profile.adi_score:.0f} / 100",
            delta=f"{emoji} {style_profile.label_ja}",
            help=(
                "Aggression/Defense Index: 0=守り型（低リスク）、100=攻め型（高リスク）。"
                "現金比率・集中度・売買頻度・保有スタイル・ボラティリティから算出。"
            ),
        )

    with col_metrics:
        if style_profile.cash_ratio is not None:
            st.caption(f"💴 現金比率: **{style_profile.cash_ratio * 100:.1f}%**")
        if style_profile.concentration_hhi is not None:
            _hhi_label = (
                "高集中"
                if style_profile.concentration_hhi >= 0.50
                else "やや集中"
                if style_profile.concentration_hhi >= 0.35
                else "分散"
            )
            st.caption(f"📐 ポジション集中度 (HHI): **{style_profile.concentration_hhi:.2f}** — {_hhi_label}")
        if style_profile.annual_volatility_pct is not None:
            st.caption(f"📉 年率ボラティリティ: **{style_profile.annual_volatility_pct:.1f}%**")
        if style_profile.beta is not None:
            st.caption(f"📊 ポートフォリオβ: **{style_profile.beta:.2f}**")

    # ADI ゲージバー（progress bar を流用）
    st.progress(
        style_profile.adi_score / 100.0,
        text=f"守り型 ← {style_profile.adi_score:.0f}/100 → 攻め型",
    )

    # Notes
    for note in style_profile.notes:
        st.caption(f"📌 {note}")


def _render_basic_style_section(
    positions: list[dict],
    sm: object,
    ts: object,
) -> None:
    """BehaviorInsight のみ利用できる場合の基本スタイル情報を描画する."""
    col1, col2 = st.columns(2)
    _freq_ja = {
        "active": "アクティブ（月2回以上）",
        "moderate": "中程度（月0.5〜2回）",
        "passive": "パッシブ（月0.5回未満）",
        "unknown": "—",
    }
    _style_ja = {
        "short_term": "短期（平均保有 〜30日）",
        "medium_term": "中期（平均保有 30〜180日）",
        "long_term": "長期（平均保有 180日〜）",
        "unknown": "—",
    }
    with col1:
        st.caption(f"📋 保有銘柄数: **{len(positions)}**")
        if hasattr(ts, "symbols_traded") and ts.symbols_traded:
            st.caption(f"🔄 取引銘柄数: **{len(ts.symbols_traded)}**")
    with col2:
        freq = getattr(sm, "trade_frequency", "unknown")
        style = getattr(sm, "holding_style", "unknown")
        st.caption(f"📈 売買頻度: **{_freq_ja.get(freq, freq)}**")
        st.caption(f"⏳ 保有スタイル: **{_style_ja.get(style, style)}**")


def _render_component_breakdown(style_profile: StyleProfile) -> None:
    """ADI コンポーネント内訳をコンパクトに描画する."""
    st.markdown("**🔍 ADI スコア内訳**")

    _comp_labels = {
        "cash": "💴 現金比率",
        "concentration": "📐 集中度",
        "holding": "⏳ 保有スタイル",
        "frequency": "🔄 売買頻度",
        "volatility": "📉 ボラティリティ",
    }
    scores = style_profile.component_scores
    n_components = len(scores)
    if n_components == 0:
        return

    # 最大3列で表示
    n_cols = min(3, n_components)
    cols = st.columns(n_cols)
    for i, (key, score) in enumerate(scores.items()):
        label = _comp_labels.get(key, key)
        with cols[i % n_cols]:
            st.metric(label, f"{score:.0f}/100")


def _render_bias_signals(biases: list[BiasSignal]) -> None:
    """検出されたバイアスシグナルを severity 別にレンダリングする."""
    st.markdown("**⚠️ バイアス検出**")

    _SEVERITY_COLOR = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🔵",
    }

    for bias in biases:
        icon = _SEVERITY_COLOR.get(bias.severity, "⚪")
        with st.expander(f"{icon} {bias.title}", expanded=(bias.severity == "high")):
            st.caption(bias.description)
            for note in bias.notes:
                st.caption(f"  • {note}")


_MONTH_NAMES_JA = {
    1: "1月",
    2: "2月",
    3: "3月",
    4: "4月",
    5: "5月",
    6: "6月",
    7: "7月",
    8: "8月",
    9: "9月",
    10: "10月",
    11: "11月",
    12: "12月",
}


def _render_long_horizon_section(
    total_value: float,
    realized_pnl: float,
    unrealized_pnl: float,
    *,
    history_df: pd.DataFrame | None = None,
    benchmark_series: pd.Series | None = None,
    benchmark_label: str = "ベンチマーク",
) -> None:
    """長期ホライゾンインサイトセクションを描画する.

    Parameters
    ----------
    total_value:
        現在の総資産額（円換算）。長期目標達成率の計算に使用する。
    realized_pnl:
        実現損益（円）。累積リターン計算に使用する。
    unrealized_pnl:
        含み損益（円）。現在のポートフォリオ状況の把握に使用する。
    history_df:
        build_portfolio_history() の出力。None の場合は基本指標のみ表示する。
    benchmark_series:
        get_benchmark_series() の出力（正規化済み）。None の場合はベンチマーク比較を省略。
    benchmark_label:
        ベンチマーク名称（UI 表示用）。
    """
    st.markdown("#### 🌅 長期ホライゾンインサイト")
    st.caption("長期視点でのポートフォリオ評価・季節性パターン・ベンチマーク比較・Sharpe安定性を提供します。")

    if total_value <= 0:
        st.info(
            "📭 ポートフォリオデータが揃うと、長期インサイトが表示されます。\n\n"
            "**将来の項目:** 目標資産額達成率 / 累積リターン推移 / リスク調整後リターン / FIRE試算",
            icon=None,
        )
        return

    # ------------------------------------------------------------------ #
    # 基本指標（既存）
    # ------------------------------------------------------------------ #
    _total_pnl = realized_pnl + unrealized_pnl
    _col1, _col2, _col3 = st.columns(3)
    with _col1:
        st.metric(label="総資産", value=f"¥{total_value:,.0f}")
    with _col2:
        _pnl_label = "累積損益（実現＋含み）"
        st.metric(
            label=_pnl_label,
            value=f"¥{_total_pnl:,.0f}",
            delta=f"{_total_pnl / (total_value - _total_pnl) * 100:+.1f}%" if (total_value - _total_pnl) > 0 else None,
        )
    with _col3:
        st.metric(label="実現損益", value=f"¥{realized_pnl:,.0f}")

    if history_df is None or history_df.empty:
        st.info(
            "⚙️ 履歴データが蓄積されると、季節性・ベンチマーク比較・Sharpe安定性シグナルが表示されます。",
            icon=None,
        )
        return

    st.markdown("")

    # ------------------------------------------------------------------ #
    # A. 月次季節性パターン
    # ------------------------------------------------------------------ #
    _render_seasonality_subsection(history_df)

    st.markdown("")

    # ------------------------------------------------------------------ #
    # B. ベンチマーク比較
    # ------------------------------------------------------------------ #
    _render_benchmark_comparison_subsection(history_df, benchmark_series, benchmark_label)

    st.markdown("")

    # ------------------------------------------------------------------ #
    # C. ローリングSharpe 安定性シグナル
    # ------------------------------------------------------------------ #
    _render_sharpe_stability_subsection(history_df)


def _render_seasonality_subsection(history_df: pd.DataFrame) -> None:
    """月次季節性パターンを描画する.

    Why: 暦月ごとのリターン傾向を把握することで、ポジション調整の
         参考情報として活用できる。
    How: compute_monthly_seasonality() で月次平均リターンと年間リターンを算出し、
         コンパクトなテーブル形式で表示する。12ヶ月未満のデータでは
         gracefully degrade してデータ不足メッセージを表示する。
    """
    st.markdown("**📅 月次季節性パターン**")
    seasonality = compute_monthly_seasonality(history_df)

    months_of_data = seasonality["months_of_data"]

    if months_of_data == 0:
        st.caption("履歴データが見つかりません。")
        return

    if not seasonality["has_sufficient_data"]:
        st.caption(
            f"⚠️ 現在 **{months_of_data}ヶ月** 分のデータがあります。"
            " 季節性パターンの信頼性を高めるには **12ヶ月以上** 必要です。"
        )

    monthly_avg = seasonality["monthly_avg_returns"]
    if monthly_avg:
        # Build a 2-row × 6-col display (Jan-Jun / Jul-Dec)
        st.caption("月次平均リターン（全期間の月次リターン平均）")
        _rows_monthly = []
        for m in range(1, 13):
            if m in monthly_avg:
                pct = monthly_avg[m]
                sign = "+" if pct >= 0 else ""
                _rows_monthly.append({"月": _MONTH_NAMES_JA[m], "平均リターン": f"{sign}{pct:.2f}%"})
        if _rows_monthly:
            # Display in two rows of 6 columns each for compactness
            _half = len(_rows_monthly) // 2
            _first_half = _rows_monthly[:_half] if _half > 0 else _rows_monthly
            _second_half = _rows_monthly[_half:] if _half > 0 else []

            _cols_a = st.columns(len(_first_half)) if _first_half else []
            for _ci, _row in enumerate(_first_half):
                with _cols_a[_ci]:
                    try:
                        _v = float(_row["平均リターン"].replace("%", "").replace("+", ""))
                    except (ValueError, AttributeError):
                        _v = 0.0
                    _color = "normal" if _v >= 0 else "inverse"
                    st.metric(_row["月"], _row["平均リターン"], delta_color=_color)

            if _second_half:
                _cols_b = st.columns(len(_second_half))
                for _ci, _row in enumerate(_second_half):
                    with _cols_b[_ci]:
                        try:
                            _v = float(_row["平均リターン"].replace("%", "").replace("+", ""))
                        except (ValueError, AttributeError):
                            _v = 0.0
                        _color = "normal" if _v >= 0 else "inverse"
                        st.metric(_row["月"], _row["平均リターン"], delta_color=_color)

    year_returns = seasonality["year_returns"]
    if year_returns:
        st.caption("年次リターン（月次リターンの複利合算）")
        _yr_cols = st.columns(min(len(year_returns), 5))
        for _yi, (yr, ret) in enumerate(sorted(year_returns.items())):
            with _yr_cols[_yi % len(_yr_cols)]:
                _yr_sign = "+" if ret >= 0 else ""
                st.metric(f"{yr}年", f"{_yr_sign}{ret:.1f}%")


def _render_benchmark_comparison_subsection(
    history_df: pd.DataFrame,
    benchmark_series: pd.Series | None,
    benchmark_label: str,
) -> None:
    """ベンチマーク比較サブセクションを描画する.

    Why: ポートフォリオのリターンをベンチマーク（市場インデックス）と
         比較することで、超過リターン（アルファ）を定量化できる。
    How: compute_benchmark_excess() でポートフォリオとベンチマークの
         期間リターンを算出し、3列形式で表示する。ベンチマーク未設定の場合は
         サイドバーで設定するよう案内する。
    """
    st.markdown("**📏 ベンチマーク比較**")

    if benchmark_series is None:
        st.caption(
            "ベンチマークが設定されていません。"
            " サイドバーの「📏 ベンチマーク比較」でベンチマークを選択すると比較が表示されます。"
        )
        return

    excess = compute_benchmark_excess(history_df, benchmark_series)
    if excess is None:
        st.caption("ベンチマークとのリターン比較に十分なデータがありません。")
        return

    bm_disp = benchmark_label if benchmark_label and benchmark_label != "なし" else "ベンチマーク"

    _bc1, _bc2, _bc3 = st.columns(3)
    with _bc1:
        _pf_r = excess["portfolio_return_pct"]
        st.metric(
            "PFリターン（期間）",
            f"{_pf_r:+.2f}%",
            help="選択期間のポートフォリオ総リターン",
        )
    with _bc2:
        _bm_r = excess["benchmark_return_pct"]
        st.metric(
            f"{bm_disp}リターン",
            f"{_bm_r:+.2f}%",
            help=f"同期間の {bm_disp} リターン（ポートフォリオ開始日を基準に正規化）",
        )
    with _bc3:
        _ex_r = excess["excess_return_pct"]
        _ex_sign = "+" if _ex_r >= 0 else ""
        _ex_note = "市場平均超過" if _ex_r >= 0 else "市場平均未満"
        st.metric(
            "超過リターン（α）",
            f"{_ex_sign}{_ex_r:.2f}%",
            delta=_ex_note,
            help="PFリターン − ベンチマークリターン。正値が市場平均超過（アルファ）。",
        )


def _render_sharpe_stability_subsection(history_df: pd.DataFrame) -> None:
    """ローリングSharpe安定性シグナルを描画する.

    Why: 単体のSharpe比よりもその時系列トレンドを見ることで、
         ポートフォリオのリスク調整後リターンが改善しているか劣化しているかを
         早期に把握できる。
    How: compute_rolling_sharpe_trend() で直近60日間のローリングSharpe比と
         30日前の値を比較し、「改善傾向 / 安定 / 悪化傾向」のシグナルを返す。
    """
    st.markdown("**📈 Sharpe安定性シグナル**")

    trend_info = compute_rolling_sharpe_trend(history_df, window=60, trend_points=30)

    if trend_info["trend"] == "insufficient":
        st.caption("⚠️ ローリングSharpe比を計算するにはデータが不足しています。 （60日間以上の履歴が必要です）")
        return

    _trend_emoji = {
        "improving": "🟢",
        "stable": "🟡",
        "declining": "🔴",
    }
    emoji = _trend_emoji.get(trend_info["trend"], "⚪")

    _sc1, _sc2, _sc3 = st.columns(3)
    with _sc1:
        st.metric(
            "ローリングSharpe（直近60日）",
            f"{trend_info['latest']:.2f}",
            help="直近60日間のリターンで算出したSharpe比（年率換算、無リスクレート0.5%想定）",
        )
    with _sc2:
        st.metric(
            "30日前との変化",
            f"{trend_info['delta']:+.2f}",
            delta=f"{emoji} {trend_info['trend_ja']}",
            help="30日前のローリングSharpe比との差分。±0.2以上で傾向判定。",
        )
    with _sc3:
        st.metric(
            "30日前のSharpe",
            f"{trend_info['prev']:.2f}",
            help="比較基準点（30日前）のローリングSharpe比",
        )

    st.caption(trend_info["description"])


# ---------------------------------------------------------------------------
# AI レトロスペクティブ（任意実行）― ヘルパー関数
# ---------------------------------------------------------------------------

_RETRO_PRIVACY_NOTICE = (
    "このボタンを押すと、以下のデータが **GitHub Copilot** に送信されます。 "
    "個人情報・口座情報・証券コードの実名や生のメモ本文は含まれません。 "
    "送信内容は下の「送信データプレビュー」で事前確認できます。 "
    "実行は任意です。キャンセルしたい場合はこのエクスパンダーを閉じてください。"
)


def _build_retro_payload(
    behavior: BehaviorInsight,
    timing: PortfolioTimingInsight,
    style_profile: StyleProfile | None,
    style_biases: list[BiasSignal] | None,
    positions: list[dict],
    realized_pnl: float,
    unrealized_pnl: float,
    total_value: float,
    retro_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ローカルで決定論的なレトロスペクティブ用サマリーペイロードを構築する.

    Why: LLM に送信する前に、何が送られるかをユーザーが確認できるよう、
         UI レンダリングとは独立した純粋な辞書として構築する。
    How: 既存の behavior/timing/style オブジェクトから主要指標を抽出し、
         銘柄コードを「N銘柄」形式に置き換えてプライバシーを保護する。
    """
    ts = behavior.trade_stats
    wl = behavior.win_loss
    hp = behavior.holding_period
    retro_context = retro_context or {}
    memo_themes = [
        str(item.get("theme", "")) for item in retro_context.get("top_themes", []) if str(item.get("theme", "")).strip()
    ]

    payload: dict[str, Any] = {
        "total_buy": ts.total_buy_count if ts else 0,
        "total_sell": ts.total_sell_count if ts else 0,
        "symbols_traded": len(ts.symbols_traded) if ts and ts.symbols_traded else 0,
        "win_rate": round(wl.win_rate * 100, 1) if wl and wl.win_rate is not None else None,
        "profit_factor": round(wl.profit_factor, 2) if wl and wl.profit_factor is not None else None,
        "avg_hold_days": round(hp.median_days, 1) if hp and hp.median_days is not None else None,
        "short_term_ratio": round(hp.short_term_ratio * 100, 1) if hp and hp.short_term_ratio is not None else None,
        "realized_pnl_jpy": int(realized_pnl),
        "unrealized_pnl_jpy": int(unrealized_pnl),
        "total_value_jpy": int(total_value),
        "positions_held": len(positions),
        "avg_buy_timing": round(timing.avg_buy_timing_score, 1) if timing.avg_buy_timing_score is not None else None,
        "avg_sell_timing": round(timing.avg_sell_timing_score, 1) if timing.avg_sell_timing_score is not None else None,
        "adi_score": round(style_profile.adi_score, 1)
        if style_profile and style_profile.adi_score is not None
        else None,
        "adi_label": style_profile.label if style_profile else None,
        "bias_signals": [b.title for b in style_biases] if style_biases else [],
        "memo_trade_count": int(retro_context.get("memo_trade_count", 0) or 0),
        "memo_coverage_pct": retro_context.get("memo_coverage_pct"),
        "memo_themes": memo_themes,
        "confidence": behavior.confidence,
    }
    return payload


def _build_retro_prompt(payload: dict[str, Any]) -> str:
    """レトロスペクティブ用LLMプロンプトをペイロードから構築する.

    Why: プロンプトはペイロードから決定論的に生成され、ユーザーが
         送信内容を事前確認できるようにする。
    How: 主要指標を箇条書きで列挙し、日本語での振り返りレポートを要求する。
    """
    total_trades = payload["total_buy"] + payload["total_sell"]
    pnl_sign = "+" if payload["realized_pnl_jpy"] >= 0 else ""
    upnl_sign = "+" if payload["unrealized_pnl_jpy"] >= 0 else ""

    lines = [
        "あなたは個人投資家のパフォーマンスコーチです。",
        "以下の匿名化された投資行動サマリーに基づき、日本語で振り返りレポートを作成してください。",
        "",
        "## 投資行動サマリー",
        f"- 総取引数: {total_trades}件（買{payload['total_buy']} / 売{payload['total_sell']}）",
        f"- 取引銘柄数: {payload['symbols_traded']}銘柄",
        f"- 現在保有銘柄数: {payload['positions_held']}銘柄",
    ]

    if payload["win_rate"] is not None:
        lines.append(f"- 勝率: {payload['win_rate']}%")
    if payload["profit_factor"] is not None:
        lines.append(f"- プロフィットファクター: {payload['profit_factor']}")
    if payload["avg_hold_days"] is not None:
        lines.append(f"- 平均保有期間（中央値）: {payload['avg_hold_days']}日")
    if payload["short_term_ratio"] is not None:
        lines.append(f"- 短期売買比率（30日以内）: {payload['short_term_ratio']}%")

    lines += [
        f"- 実現損益: {pnl_sign}{payload['realized_pnl_jpy']:,}円",
        f"- 含み損益: {upnl_sign}{payload['unrealized_pnl_jpy']:,}円",
    ]

    if payload["avg_buy_timing"] is not None:
        lines.append(f"- 平均エントリータイミングスコア: {payload['avg_buy_timing']}/100")
    if payload["avg_sell_timing"] is not None:
        lines.append(f"- 平均エグジットタイミングスコア: {payload['avg_sell_timing']}/100")
    if payload["adi_score"] is not None:
        label_map = {
            "aggressive": "積極型",
            "defensive": "守備型",
            "balanced": "バランス型",
        }
        adi_label = label_map.get(payload["adi_label"] or "", payload["adi_label"] or "")
        lines.append(f"- 投資スタイルスコア（ADI）: {payload['adi_score']}/100（{adi_label}）")
    if payload["bias_signals"]:
        lines.append(f"- 検出されたバイアス: {', '.join(payload['bias_signals'])}")
    if payload["memo_trade_count"] > 0:
        lines.append(f"- メモ付き取引数（直近）: {payload['memo_trade_count']}件")
        if payload["memo_coverage_pct"] is not None:
            lines.append(f"- メモ記録率（直近）: {payload['memo_coverage_pct']}%")
        if payload["memo_themes"]:
            lines.append(f"- 取引メモの主要テーマ: {', '.join(payload['memo_themes'])}")

    lines += [
        "",
        "## 指示",
        "以下の構成でレポートを作成してください。",
        "1. **良かった点**: 数値データから読み取れる強みを2〜3点",
        "2. **改善点**: 気をつけるべきパターンや課題を2〜3点",
        "3. **次のアクション**: 具体的な改善提案を1〜2点",
        "",
        "回答は日本語で、箇条書きを使い、簡潔に（合計400字程度）まとめてください。",
        "銘柄コードや個人情報は一切含まれていないため、内容に基づいて分析してください。",
    ]

    return "\n".join(lines)


def _render_retrospective_section(
    *,
    behavior: BehaviorInsight,
    timing: PortfolioTimingInsight,
    style_profile: StyleProfile | None,
    style_biases: list[BiasSignal] | None,
    positions: list[dict],
    realized_pnl: float,
    unrealized_pnl: float,
    total_value: float,
    retro_context: dict[str, Any] | None = None,
) -> None:
    """AI レトロスペクティブ（任意実行）セクションを描画する.

    Why: ユーザーが明示的にオプトインしたときだけ LLM 呼び出しを行う。
         自動呼び出しは一切しない。
    How: ペイロードをローカルで構築 → ユーザーへの事前通知 → ボタン押下で
         copilot_client.call() を起動 → 結果をセッション state に保存。
    """
    # --- セッション state の初期化 ---
    if SK.RETRO_RESULT not in st.session_state:
        st.session_state[SK.RETRO_RESULT] = None
    if SK.RETRO_ERROR not in st.session_state:
        st.session_state[SK.RETRO_ERROR] = None

    # --- ペイロードとプロンプトをローカルで確定 ---
    payload = _build_retro_payload(
        behavior=behavior,
        timing=timing,
        style_profile=style_profile,
        style_biases=style_biases,
        retro_context=retro_context,
        positions=positions,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_value=total_value,
    )
    prompt = _build_retro_prompt(payload)

    # --- データ量チェック（最低限のトレード数がないと意味がない） ---
    total_trades = payload["total_buy"] + payload["total_sell"]

    st.markdown("#### 🤖 AI レトロスペクティブ（任意実行）")
    st.caption(
        "蓄積されたトレード行動・スタイル・タイミングデータをもとに、"
        "AI が投資行動の振り返りレポートを生成します。"
        "実行は任意です。"
    )

    if total_trades == 0:
        st.info("📭 レトロスペクティブを生成するには、売買履歴が必要です。取引を記録すると利用できます。")
        return

    # --- プライバシー通知 ---
    st.warning(_RETRO_PRIVACY_NOTICE, icon="🔒")
    if payload["memo_trade_count"] > 0:
        _memo_caption = f"直近のメモ付き取引 {payload['memo_trade_count']} 件を匿名テーマ化して要約に利用します。"
        if payload["memo_themes"]:
            _memo_caption += f" 主要テーマ: {', '.join(payload['memo_themes'])}"
        st.caption(_memo_caption)

    # --- 送信データプレビュー ---
    with st.expander("📋 送信データプレビュー（クリックして確認）", expanded=False):
        st.caption("以下のテキストのみが Copilot に送信されます。銘柄コード・個人情報・生のメモ本文は含まれません。")
        st.code(prompt, language="markdown")

    # --- 実行・クリアボタン ---
    col_run, col_clear = st.columns([2, 1])
    with col_run:
        run_clicked = st.button(
            "▶ レトロスペクティブを実行",
            key="retro_run_btn",
            type="primary",
            help="GitHub Copilot に接続して振り返りレポートを生成します。",
        )
    with col_clear:
        clear_clicked = st.button(
            "🗑 クリア",
            key="retro_clear_btn",
            help="生成済みのレポートを削除します。",
        )

    if clear_clicked:
        st.session_state[SK.RETRO_RESULT] = None
        st.session_state[SK.RETRO_ERROR] = None
        st.rerun()

    if run_clicked:
        # --- 遅延インポート: importlib.import_module を使い sys.modules 経由で解決する ---
        # Why: from components import copilot_client はパッケージ属性キャッシュを参照するため、
        #      テスト時の sys.modules パッチが効かない場合がある。
        #      importlib.import_module は sys.modules を直接参照し、パッチが確実に効く。
        import importlib

        try:
            copilot_client = importlib.import_module("components.copilot_client")
        except Exception:
            st.session_state[SK.RETRO_RESULT] = None
            st.session_state[SK.RETRO_ERROR] = "Copilot クライアントを読み込めませんでした。"
        else:
            if not copilot_client.is_available():
                st.session_state[SK.RETRO_RESULT] = None
                st.session_state[SK.RETRO_ERROR] = (
                    "GitHub Copilot が利用できません。 Copilot CLI のインストールと認証を確認してください。"
                )
            else:
                with st.spinner("Copilot でレポートを生成中…"):
                    try:
                        result = copilot_client.call(
                            prompt,
                            timeout=90,
                            source="insight_retrospective",
                        )
                    except Exception as exc:
                        result = None
                        st.session_state[SK.RETRO_ERROR] = f"エラーが発生しました: {exc}"
                    else:
                        if result is None:
                            st.session_state[SK.RETRO_ERROR] = (
                                "Copilot から応答を取得できませんでした。しばらく後に再試行してください。"
                            )
                        else:
                            st.session_state[SK.RETRO_RESULT] = result
                            st.session_state[SK.RETRO_ERROR] = None

    # --- 結果 / エラー表示 ---
    retro_result: str | None = st.session_state.get(SK.RETRO_RESULT)
    retro_error: str | None = st.session_state.get(SK.RETRO_ERROR)

    if retro_error:
        st.error(f"⚠️ {retro_error}")

    if retro_result:
        st.divider()
        st.markdown("##### 📝 AI レトロスペクティブ レポート")
        st.markdown(retro_result)
        st.caption("※ このレポートは AI が生成したものです。投資判断の参考情報としてご利用ください。")


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def render_insights_tab(
    *,
    positions: list[dict],
    total_value: float,
    unrealized_pnl: float,
    realized_pnl: float,
    history_df: pd.DataFrame | None = None,
    benchmark_series: pd.Series | None = None,
    benchmark_label: str = "ベンチマーク",
    behavior_insight: BehaviorInsight | None = None,
    timing_insight: PortfolioTimingInsight | None = None,
    style_profile: StyleProfile | None = None,
    style_biases: list[BiasSignal] | None = None,
    retro_context: dict[str, Any] | None = None,
) -> None:
    """インサイトタブのコンテンツを描画する.

    Parameters
    ----------
    positions:
        現在の保有銘柄リスト（``get_current_snapshot()`` から取得した
        ``list[dict]`` 形式。各要素は ``symbol``, ``name``,
        ``evaluation_jpy``, ``sector`` などのキーを含む）。
    total_value:
        現在の総資産額（円換算）。長期インサイトセクションで使用する。
    unrealized_pnl:
        含み損益（円）。長期インサイトセクションで使用する。
    realized_pnl:
        実現損益（円）。トレード統計・長期インサイトセクションで使用する。
    history_df:
        build_portfolio_history() の出力。長期インサイトの季節性・Sharpe分析に使用する。
        None の場合は長期インサイトの基本指標のみ表示する。
    benchmark_series:
        get_benchmark_series() の出力（正規化済み）。長期インサイトのベンチマーク比較に使用する。
        None の場合はベンチマーク比較を省略する。
    benchmark_label:
        ベンチマーク名称（UI 表示用）。
    behavior_insight:
        ``load_behavior_insight()`` から得た ``BehaviorInsight`` オブジェクト。
        ``None`` の場合は空の ``BehaviorInsight`` を使用する。
    timing_insight:
        ``load_timing_insight()`` から得た ``PortfolioTimingInsight`` オブジェクト。
        ``None`` の場合は空の ``PortfolioTimingInsight`` を使用する。
    style_profile:
        ``load_style_profile_insight()`` から得た ``StyleProfile``。
        ``None`` の場合は基本的なスタイル情報のみを表示する。
    style_biases:
        ``load_style_profile_insight()`` から得た ``list[BiasSignal]``。
        ``None`` の場合はバイアスセクションを省略する。
    retro_context:
        ``load_trade_memo_context()`` から得た匿名化メモ要約。
        ``None`` の場合はメモ関連集計を省略する。
    """
    _behavior = behavior_insight if behavior_insight is not None else BehaviorInsight.empty()
    _timing = timing_insight if timing_insight is not None else PortfolioTimingInsight.empty()

    st.markdown(
        '<div id="insights" role="region" aria-label="インサイト"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 💡 インサイト")
    st.caption(
        "蓄積されたトレード履歴・保有データから、投資行動パターンと改善ポイントを分析します。"
        "｜ データが蓄積されるほど、より精度の高いインサイトが得られます。"
    )

    # ------------------------------------------------------------------ #
    # セクション 1: トレード統計
    # ------------------------------------------------------------------ #
    with st.expander("📊 トレード統計", expanded=True):
        _render_trade_statistics_section(_behavior)

    st.divider()

    # ------------------------------------------------------------------ #
    # セクション 2: タイミングレビュー
    # ------------------------------------------------------------------ #
    with st.expander("⏱️ タイミングレビュー", expanded=False):
        _render_timing_review_section(_timing)

    st.divider()

    # ------------------------------------------------------------------ #
    # セクション 3: スタイルプロファイル
    # ------------------------------------------------------------------ #
    with st.expander("🎨 スタイルプロファイル", expanded=False):
        _render_style_profile_section(
            positions,
            _behavior,
            style_profile=style_profile,
            style_biases=style_biases,
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # セクション 4: 長期ホライゾンインサイト
    # ------------------------------------------------------------------ #
    with st.expander("🌅 長期ホライゾンインサイト", expanded=False):
        _render_long_horizon_section(
            total_value=total_value,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            history_df=history_df,
            benchmark_series=benchmark_series,
            benchmark_label=benchmark_label,
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # セクション 5: AI レトロスペクティブ（任意実行）
    # ------------------------------------------------------------------ #
    with st.expander("🤖 AI レトロスペクティブ（任意実行）", expanded=False):
        _render_retrospective_section(
            behavior=_behavior,
            timing=_timing,
            style_profile=style_profile,
            style_biases=style_biases,
            retro_context=retro_context,
            positions=positions,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_value=total_value,
        )
