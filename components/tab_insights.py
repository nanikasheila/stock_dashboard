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

import streamlit as st

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.behavior.models import (
    BehaviorInsight,
    BiasSignal,
    ConfidenceLevel,
    PortfolioTimingInsight,
    StyleProfile,
)

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


def _render_long_horizon_section(
    total_value: float,
    realized_pnl: float,
    unrealized_pnl: float,
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
    """
    st.markdown("#### 🌅 長期ホライゾンインサイト")
    st.caption("長期視点でのポートフォリオ評価・目標達成予測・リスク分散状況を提供します。")

    if total_value <= 0:
        st.info(
            "📭 ポートフォリオデータが揃うと、長期インサイトが表示されます。\n\n"
            "**将来の項目:** 目標資産額達成率 / 累積リターン推移 / リスク調整後リターン / FIRE試算",
            icon=None,
        )
        return

    # --- 将来: 長期インサイトの表示エリア（現在は基本指標のみ） ---
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

    st.info(
        "⚙️ FIRE試算・長期目標達成率・リスク調整後リターンの詳細分析は次のフェーズで追加されます。",
        icon=None,
    )


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def render_insights_tab(
    *,
    positions: list[dict],
    total_value: float,
    unrealized_pnl: float,
    realized_pnl: float,
    behavior_insight: BehaviorInsight | None = None,
    timing_insight: PortfolioTimingInsight | None = None,
    style_profile: StyleProfile | None = None,
    style_biases: list[BiasSignal] | None = None,
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
        )
