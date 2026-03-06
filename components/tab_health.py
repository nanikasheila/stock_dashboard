"""ヘルスチェック & 経済ニュース タブ描画モジュール.

``app.py`` の ``with _tab_health:`` ブロックを切り出したモジュール。

公開 API
--------
render_health_tab(...)
    ヘルスチェック + 売りアラート + LLM 統合分析 + 経済ニュースを描画する。

依存
----
- ``state_keys.SK`` — session_state キー定数
- ``components.kpi_helpers.risk_card`` — リスク指標 HTML カード
- ``components.llm_analyzer`` — LLM 統合分析
- ``components.copilot_client`` — CLI 実行ログ
"""

from __future__ import annotations

import html as _html_mod
import math
import time

import pandas as pd
import streamlit as st
from state_keys import SK

from components.copilot_client import (
    clear_execution_logs as copilot_clear_logs,
)
from components.copilot_client import (
    get_execution_logs as copilot_get_logs,
)
from components.kpi_helpers import risk_card
from components.llm_analyzer import apply_news_analysis, run_unified_analysis


def render_health_tab(
    *,
    snapshot: dict,
    positions: list[dict],
    health_data: dict | None,
    econ_news: list[dict],
    llm_enabled: bool,
    llm_auto_analyze: bool,
    llm_model: str,
    llm_cache_ttl_sec: int,
) -> None:
    """ヘルスチェック & 経済ニュースタブを描画する.

    Parameters
    ----------
    snapshot:         ポートフォリオスナップショット（``get_current_snapshot()`` 戻り値）
    positions:        保有銘柄リスト
    health_data:      ヘルスチェック結果（``run_dashboard_health_check()`` 戻り値、失敗時 None）
    econ_news:        経済ニュースリスト（キーワードベース、LLM 未適用版を想定）
    llm_enabled:      LLM 分析機能の ON/OFF
    llm_auto_analyze: 自動分析モードの ON/OFF
    llm_model:        使用 LLM モデル識別子
    llm_cache_ttl_sec: LLM 分析キャッシュ有効期間（秒）
    """
    # =====================================================================
    # ヘルスチェック & 売りアラート
    # =====================================================================
    st.markdown('<div id="health-check" role="region" aria-label="ヘルスチェック"></div>', unsafe_allow_html=True)
    st.markdown("### 🏥 ヘルスチェック")
    _hc_as_of = st.session_state.get(SK.LAST_REFRESH, "—")[:16]
    st.caption(
        f"各銘柄のトレンド・テクニカル指標をチェックし、売りタイミングや注意が必要な銘柄を自動検出します。｜ 🕐 データ取得: {_hc_as_of}"
    )

    if health_data is not None:
        hc_summary = health_data["summary"]
        hc_positions = health_data["positions"]
        sell_alerts = health_data["sell_alerts"]

        # --- サマリーカード ---
        hc_cols = st.columns(5)
        _hc_items = [
            ("合計", hc_summary["total"], ""),
            ("✅ 健全", hc_summary["healthy"], "#4ade80"),
            ("⚡ 早期警告", hc_summary["early_warning"], "#fbbf24"),
            ("⚠️ 注意", hc_summary["caution"], "#fb923c"),
            ("🚨 撤退", hc_summary["exit"], "#f87171"),
        ]
        for i, (label, count, color) in enumerate(_hc_items):
            with hc_cols[i]:
                st.markdown(risk_card(label, str(count), color), unsafe_allow_html=True)

        # --- LLM ヘルスチェック分析（売りアラート通知より先に実行） ---
        _hc_llm_summary: dict | None = None
        _hc_llm_assessment_map: dict[str, dict] = {}

        # 手動モードの場合は session_state から前回結果を復元
        if llm_enabled and not llm_auto_analyze:
            _hc_llm_summary = st.session_state.get(SK.LLM_HC_SUMMARY)
            if _hc_llm_summary:
                for _sa in _hc_llm_summary.get("stock_assessments", []):
                    _sa_sym = _sa.get("symbol", "")
                    if _sa_sym:
                        _hc_llm_assessment_map[_sa_sym] = _sa

        # 自動モードの場合は統合分析（1セッション）で実行
        if llm_enabled and llm_auto_analyze:
            # 統合分析: ニュース分類 + 要約 + ヘルスチェック を 1 セッションで実行
            _unified_result = run_unified_analysis(
                econ_news,
                positions,
                health_data,
                model=llm_model,
                timeout=180,
                cache_ttl=llm_cache_ttl_sec,
            )
            if _unified_result:
                # ニュース分析結果を適用して session_state へ保存
                _analyzed_news = apply_news_analysis(econ_news, _unified_result.get("news_analysis", []))
                st.session_state[SK.LLM_NEWS_RESULTS] = _analyzed_news
                st.session_state[SK.LLM_ANALYZED_AT] = time.time()

                # ニュースサマリーを session_state へ保存
                _unified_news_summary = _unified_result.get("news_summary")
                if _unified_news_summary:
                    st.session_state[SK.LLM_NEWS_SUMMARY] = _unified_news_summary

                # ヘルスチェックサマリー
                _hc_llm_summary = _unified_result.get("health_summary")
                if _hc_llm_summary:
                    st.session_state[SK.LLM_HC_SUMMARY] = _hc_llm_summary
                    for _sa in _hc_llm_summary.get("stock_assessments", []):
                        _sa_sym = _sa.get("symbol", "")
                        if _sa_sym:
                            _hc_llm_assessment_map[_sa_sym] = _sa

        # --- クリティカルアラートのトースト通知 ---
        # Why: ユーザーが他タブを閲覧中でも緊急アラートに気付ける必要がある
        # How: st.toast() はタブスコープ外のオーバーレイで表示される
        _critical_alerts = [a for a in sell_alerts if a["urgency"] == "critical"]
        for _ca in _critical_alerts:
            st.toast(f"🚨 {_ca['name']}: {_ca['action']}", icon="🚨")

        # --- 売りアラート通知 ---
        if sell_alerts:
            st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔔 売りタイミング通知")

            for alert in sell_alerts:
                urgency = alert["urgency"]
                _urgency_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                _urgency_label = {"critical": "緊急", "warning": "注意", "info": "参考"}

                # Build detail HTML
                detail_html = ""
                for d in alert.get("details", []):
                    detail_html += f'<div class="sell-alert-detail">• {d}</div>'

                # LLM 分析コメントを付加
                _alert_sym = alert.get("symbol", "")
                _llm_sa = _hc_llm_assessment_map.get(_alert_sym)
                if _llm_sa:
                    _llm_text = _llm_sa.get("assessment", "")
                    if _llm_text:
                        detail_html += f'<div class="sell-alert-ai">🤖 <strong>AI分析</strong>: {_llm_text}</div>'

                pnl = alert.get("pnl_pct", 0)
                pnl_color = "#4ade80" if pnl >= 0 else "#f87171"
                pnl_text = f'<span style="color:{pnl_color}; font-weight:600;">{pnl:+.1f}%</span>'

                st.markdown(
                    f'<div class="sell-alert sell-alert-{urgency}" role="alert"'
                    f' aria-label="{_urgency_label.get(urgency, "")} {alert["name"]}">'
                    f'<div class="sell-alert-header">'
                    f"{_urgency_emoji.get(urgency, '')} "
                    f"[{_urgency_label.get(urgency, '')}] "
                    f"{alert['name']} ({alert['symbol']}) "
                    f"— {alert['action']} "
                    f"(含み損益: {pnl_text})"
                    f"</div>"
                    f'<div class="sell-alert-reason">{alert["reason"]}</div>'
                    f"{detail_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("🟢 現在、売りタイミングの通知はありません")

        # --- LLM ヘルスチェックサマリー表示 ---
        if _hc_llm_summary:
            st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

            _hcs_html = '<div class="health-summary-card">'
            _hcs_html += '<div class="health-summary-header">🤖 ヘルスチェックサマリー</div>'

            _hcs_overview = _hc_llm_summary.get("overview", "")
            if _hcs_overview:
                _hcs_html += f'<div class="health-summary-overview">{_hcs_overview}</div>'

            _hcs_warning = _hc_llm_summary.get("risk_warning", "")
            if _hcs_warning:
                _hcs_html += f'<div class="health-summary-warning">⚠️ <strong>リスク注意</strong>: {_hcs_warning}</div>'

            _hcs_assessments = _hc_llm_summary.get("stock_assessments", [])
            if _hcs_assessments:
                # アラートレベルを持つ銘柄マップ
                _hc_alert_map: dict[str, str] = {}
                for _hcp in hc_positions:
                    _hc_alert_map[_hcp.get("symbol", "")] = _hcp.get("alert_level", "none")

                _hcs_html += '<details class="health-summary-stocks-toggle">'
                _hcs_html += f"<summary>📋 銘柄別コメント（{len(_hcs_assessments)}件）</summary>"

                for _sa in _hcs_assessments:
                    _sa_sym = _sa.get("symbol", "")
                    _sa_name = _sa.get("name", _sa_sym)
                    _sa_assessment = _sa.get("assessment", "")
                    _sa_action = _sa.get("action", "")
                    _sa_alert = _hc_alert_map.get(_sa_sym, "none")
                    _sa_level_class = f" health-summary-stock-{_sa_alert}" if _sa_alert != "none" else ""
                    _action_badge = f'<span class="health-summary-action">{_sa_action}</span>' if _sa_action else ""
                    _hcs_html += (
                        f'<div class="health-summary-stock{_sa_level_class}">'
                        f'<div class="health-summary-stock-name">'
                        f"{_sa_name} ({_sa_sym}){_action_badge}</div>"
                        f'<div class="health-summary-stock-text">{_sa_assessment}</div>'
                        f"</div>"
                    )

                _hcs_html += "</details>"

            _hcs_html += "</div>"
            st.markdown(_hcs_html, unsafe_allow_html=True)

        # --- 銘柄別ヘルスチェック詳細 ---
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

        with st.expander("📋 銘柄別ヘルスチェック詳細", expanded=False):
            if hc_positions:
                # テーブル表示
                hc_table_data = []
                for pos in hc_positions:
                    alert_level = pos["alert_level"]
                    _level_display = {
                        "none": "✅ 健全",
                        "early_warning": "⚡ 早期警告",
                        "caution": "⚠️ 注意",
                        "exit": "🚨 撤退",
                    }
                    _trend_emoji = {
                        "上昇": "📈",
                        "横ばい": "➡️",
                        "下降": "📉",
                        "不明": "❓",
                    }
                    rsi_val = pos.get("rsi", float("nan"))
                    try:
                        rsi_str = f"{rsi_val:.1f}" if not math.isnan(rsi_val) else "N/A"
                    except (TypeError, ValueError):
                        rsi_str = "N/A"

                    stability_emoji = pos.get("return_stability_emoji", "")
                    long_term = pos.get("long_term_label", "")

                    reasons_str = " / ".join(pos.get("alert_reasons", [])) if pos.get("alert_reasons") else "-"

                    hc_table_data.append(
                        {
                            "銘柄": f"{pos['name']}",
                            "シンボル": pos["symbol"],
                            "判定": _level_display.get(alert_level, alert_level),
                            "トレンド": f"{_trend_emoji.get(pos['trend'], '')} {pos['trend']}",
                            "RSI": rsi_str,
                            "変化品質": pos.get("change_quality", ""),
                            "長期適性": long_term,
                            "還元安定度": stability_emoji,
                            "含み損益(%)": pos.get("pnl_pct", 0),
                            "理由": reasons_str,
                        }
                    )

                hc_df = pd.DataFrame(hc_table_data)

                # アラートレベルでソート（exit > caution > early_warning > none）
                _sort_order = {"🚨 撤退": 0, "⚠️ 注意": 1, "⚡ 早期警告": 2, "✅ 健全": 3}
                hc_df["_sort"] = hc_df["判定"].map(_sort_order).fillna(9)
                hc_df = hc_df.sort_values("_sort").drop(columns=["_sort"])

                st.dataframe(
                    hc_df.style.format(
                        {
                            "含み損益(%)": "{:+.1f}%",
                        }
                    ).map(
                        lambda v: (
                            "color: #4ade80"
                            if isinstance(v, int | float) and v > 0
                            else ("color: #f87171" if isinstance(v, int | float) and v < 0 else "")
                        ),
                        subset=["含み損益(%)"],
                    ),
                    width="stretch",
                    height=min(400, 60 + len(hc_table_data) * 38),
                )

                # --- 個別銘柄カード（アラートのみ展開） ---
                alert_positions = [p for p in hc_positions if p["alert_level"] != "none"]
                if alert_positions:
                    st.markdown("##### ⚡ アラート銘柄の詳細")
                    for pos in alert_positions:
                        alert_level = pos["alert_level"]
                        _card_border_color = {
                            "early_warning": "#fbbf24",
                            "caution": "#fb923c",
                            "exit": "#f87171",
                        }.get(alert_level, "#94a3b8")

                        indicators = pos.get("indicators", {})
                        ind_parts = []
                        for ind_name, ind_val in indicators.items():
                            _ind_labels = {
                                "accruals": "アクルーアルズ",
                                "revenue_acceleration": "売上加速",
                                "fcf_yield": "FCF利回り",
                                "roe_trend": "ROE趨勢",
                            }
                            label = _ind_labels.get(ind_name, ind_name)
                            if isinstance(ind_val, bool):
                                emoji = "✅" if ind_val else "❌"
                                ind_parts.append(f"{emoji} {label}")
                            elif isinstance(ind_val, int | float):
                                emoji = "✅" if ind_val > 0 else "❌"
                                ind_parts.append(f"{emoji} {label}")

                        ind_html = " &nbsp;|&nbsp; ".join(ind_parts) if ind_parts else ""

                        trap_html = ""
                        if pos.get("value_trap"):
                            trap_reasons = " / ".join(pos.get("value_trap_reasons", []))
                            trap_html = (
                                f'<div style="margin-top:6px; padding:6px 10px;'
                                f" background:rgba(248,113,113,0.1); border-radius:6px;"
                                f' font-size:0.82rem;">'
                                f"🪤 バリュートラップ: {trap_reasons}</div>"
                            )

                        reasons_html = ""
                        for r in pos.get("alert_reasons", []):
                            reasons_html += f'<div style="font-size:0.82rem; padding:1px 0;">• {r}</div>'

                        cross_html = ""
                        cross_signal = pos.get("cross_signal", "none")
                        if cross_signal != "none":
                            _cross_emoji = "🟡" if cross_signal == "golden_cross" else "💀"
                            _cross_label = "ゴールデンクロス" if cross_signal == "golden_cross" else "デッドクロス"
                            days = pos.get("days_since_cross", "?")
                            cross_html = f" | {_cross_emoji} {_cross_label}（{days}日前）"

                        st.markdown(
                            f'<div class="health-card health-card-{alert_level}">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                            f'<span style="font-weight:700; font-size:1.0rem;">'
                            f"{pos['alert_emoji']} {pos['name']} ({pos['symbol']})</span>"
                            f'<span style="font-size:0.85rem; opacity:0.8;">'
                            f"{pos['alert_label']}</span>"
                            f"</div>"
                            f'<div style="font-size:0.85rem; margin-top:6px; opacity:0.8;">'
                            f"トレンド: {pos['trend']} | RSI: {pos.get('rsi', 0):.1f} "
                            f"| SMA50: {pos.get('sma50', 0):,.1f} "
                            f"| SMA200: {pos.get('sma200', 0):,.1f}"
                            f"{cross_html}"
                            f"</div>"
                            f'<div style="font-size:0.85rem; margin-top:4px;">{ind_html}</div>'
                            f'<div style="margin-top:6px;">{reasons_html}</div>'
                            f"{trap_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            else:
                st.info("保有銘柄データがありません")

    # =====================================================================
    # 経済ニュース & PF影響
    # =====================================================================
    st.markdown('<div id="economic-news" role="region" aria-label="経済ニュース"></div>', unsafe_allow_html=True)
    st.markdown("### 📰 経済ニュース & PF影響")
    _news_as_of = st.session_state.get(SK.LAST_REFRESH, "—")[:16]
    st.caption(
        f"主要指数・商品に関する最新ニュースと、ポートフォリオへの影響度を自動分析します。｜ 🕐 データ取得: {_news_as_of}"
    )

    # セッション内に LLM 分析結果があれば置換（手動・自動共通）
    if econ_news and llm_enabled and SK.LLM_NEWS_RESULTS in st.session_state:
        econ_news = st.session_state[SK.LLM_NEWS_RESULTS]

    if econ_news:
        # 分析方法の表示
        _any_llm = any(n.get("analysis_method") == "llm" for n in econ_news)

        # --- 手動モード: AI分析ボタン ---
        if llm_enabled and not llm_auto_analyze:
            _manual_col1, _manual_col2 = st.columns([3, 1])
            with _manual_col1:
                if _any_llm:
                    st.caption("🤖 AI分析（" + llm_model + "）")
                elif SK.LLM_ANALYZED_AT in st.session_state:
                    import datetime as _dt_mn

                    _mn_at = _dt_mn.datetime.fromtimestamp(st.session_state[SK.LLM_ANALYZED_AT]).strftime("%H:%M")
                    st.caption(f"🤖 AI分析済み（{_mn_at}）— 🔑 ニュースはキーワードベース")
                else:
                    st.caption("🔑 キーワードベース分析（AI分析は手動実行）")
            with _manual_col2:
                if health_data is not None and st.button(
                    "🤖 AI分析を実行", key="manual_llm_run", help="LLM でニュース・ヘルスチェックを分析します"
                ):
                    with st.spinner("AI分析中..."):
                        # 統合分析: 1回の LLM 呼び出しでニュース分析+サマリー+ヘルスチェックを実行
                        _unified = run_unified_analysis(
                            econ_news,
                            positions,
                            health_data,
                            model=llm_model,
                            cache_ttl=llm_cache_ttl_sec,
                        )
                        if _unified:
                            # ニュース分析結果を適用
                            _analyzed = apply_news_analysis(econ_news, _unified.get("news_analysis", []))
                            st.session_state[SK.LLM_NEWS_RESULTS] = _analyzed
                            st.session_state[SK.LLM_ANALYZED_AT] = time.time()
                            # ニュースサマリー
                            _ns = _unified.get("news_summary")
                            if _ns:
                                st.session_state[SK.LLM_NEWS_SUMMARY] = _ns
                            # ヘルスチェックサマリー
                            _hcs = _unified.get("health_summary")
                            if _hcs:
                                st.session_state[SK.LLM_HC_SUMMARY] = _hcs

                    st.rerun()
        elif _any_llm:
            from components.llm_analyzer import get_cache_info as llm_get_cache_info

            _cache_info = llm_get_cache_info()
            if _cache_info["cached"] and _cache_info["age_sec"] > 10:
                _age_m = _cache_info["age_sec"] // 60
                st.caption(f"🤖 AI分析（{llm_model}）— 📦 キャッシュ済み（{_age_m}分前）")
            else:
                st.caption("🤖 AI分析（" + llm_model + "）")
        else:
            st.caption("🔑 キーワードベース分析")

        # --- サマリーカード: 影響度別件数 ---
        _n_high = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "high")
        _n_med = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "medium")
        _n_low = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "low")
        _n_none = sum(1 for n in econ_news if n["portfolio_impact"]["impact_level"] == "none")

        ncol1, ncol2, ncol3, ncol4 = st.columns(4)
        with ncol1:
            st.markdown(risk_card("🔴 高影響", str(_n_high), "#f87171" if _n_high > 0 else ""), unsafe_allow_html=True)
        with ncol2:
            st.markdown(risk_card("🟡 中影響", str(_n_med), "#fbbf24" if _n_med > 0 else ""), unsafe_allow_html=True)
        with ncol3:
            st.markdown(risk_card("🔵 低影響", str(_n_low), "#60a5fa" if _n_low > 0 else ""), unsafe_allow_html=True)
        with ncol4:
            st.markdown(risk_card("⚪ 影響なし", str(_n_none), ""), unsafe_allow_html=True)

        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

        # --- LLM サマリー ---
        # 自動/手動共通: session_state から復元（統合分析で取得済み）
        _summary: dict | None = None
        if llm_enabled:
            _summary = st.session_state.get(SK.LLM_NEWS_SUMMARY)

        if _summary:
            _overview = _summary.get("overview", "")
            _key_points = _summary.get("key_points", [])
            _pf_alert = _summary.get("portfolio_alert", "")

            # サマリーカード
            _summary_html = '<div class="news-summary-card">'
            _summary_html += '<div class="news-summary-header">📋 ニュースサマリー</div>'
            if _overview:
                _summary_html += f'<div class="news-summary-overview">{_overview}</div>'

            if _key_points:
                _summary_html += '<div class="news-summary-points">'
                for _kp in _key_points:
                    _icon = _kp.get("icon", "📌")
                    _label = _kp.get("label", _kp.get("category", ""))
                    _kp_summary = _kp.get("summary", "")
                    _news_ids = _kp.get("news_ids", [])
                    _ids_str = ""
                    if _news_ids:
                        _id_links = [f'<span class="news-ref">#{nid + 1}</span>' for nid in _news_ids]
                        _ids_str = f' <span class="news-refs">{", ".join(_id_links)}</span>'
                    _summary_html += (
                        f'<div class="news-summary-point">'
                        f'<span class="news-summary-cat">{_icon} {_label}</span>'
                        f'<span class="news-summary-text">{_kp_summary}{_ids_str}</span>'
                        f"</div>"
                    )
                _summary_html += "</div>"

            if _pf_alert:
                _summary_html += f'<div class="news-summary-alert">⚠️ <strong>PF注意</strong>: {_pf_alert}</div>'

            _summary_html += "</div>"
            st.markdown(_summary_html, unsafe_allow_html=True)
            st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

        # --- ニュースカード表示 ---
        # PF影響ありのニュースを先に表示
        _impact_news = [n for n in econ_news if n["portfolio_impact"]["impact_level"] != "none"]
        _other_news = [n for n in econ_news if n["portfolio_impact"]["impact_level"] == "none"]

        # ニュースにインデックス番号を付与（サマリーからのトレース用）
        for _disp_num, _news in enumerate(econ_news, 1):
            _news["_display_number"] = _disp_num

        if _impact_news:
            with st.expander(f"⚡ PF影響のあるニュース（{len(_impact_news)}件）", expanded=False):
                for news_item in _impact_news:
                    _impact = news_item["portfolio_impact"]
                    _impact_level = _impact["impact_level"]
                    _impact_labels = {"high": "高影響", "medium": "中影響", "low": "低影響"}
                    _impact_colors = {"high": "impact-high", "medium": "impact-medium", "low": "impact-low"}

                    # カテゴリバッジ
                    _cat_badges = ""
                    for cat in news_item.get("categories", []):
                        _cat_badges += (
                            f'<span class="news-badge news-badge-category">{cat["icon"]} {cat["label"]}</span>'
                        )

                    # 影響度バッジ
                    _impact_badge = (
                        f'<span class="news-badge news-badge-{_impact_colors.get(_impact_level, "")}">'
                        f"{_impact_labels.get(_impact_level, '')} — "
                        f"{len(_impact['affected_holdings'])}銘柄</span>"
                    )

                    # 影響銘柄リスト
                    _affected_html = ""
                    if _impact["affected_holdings"]:
                        _syms = ", ".join(_impact["affected_holdings"][:8])
                        _affected_html = f'<div class="news-affected">📌 影響銘柄: {_syms}</div>'

                    # LLM分析の理由（あれば表示）
                    _reason_html = ""
                    _reason = _html_mod.escape(_impact.get("reason", ""))
                    if _reason and news_item.get("analysis_method") == "llm":
                        _reason_html = (
                            f'<div style="font-size:0.82rem; margin-top:4px; opacity:0.85;">💡 {_reason}</div>'
                        )

                    # タイトルリンク
                    # Why: ニュースタイトル・リンクは外部ソース由来のため XSS 防止
                    _link = _html_mod.escape(news_item.get("link", ""))
                    _safe_title = _html_mod.escape(news_item.get("title", ""))
                    _disp_no = news_item.get("_display_number", "")
                    _num_badge = f'<span class="news-number">#{_disp_no}</span>' if _disp_no else ""
                    _title_html = (
                        f'<a href="{_link}" target="_blank" rel="noopener noreferrer">{_safe_title}</a>'
                        if _link
                        else _safe_title
                    )

                    # 発行元・日時
                    _pub = _html_mod.escape(news_item.get("publisher", ""))
                    _time = news_item.get("publish_time", "")
                    _source = _html_mod.escape(news_item.get("source_name", ""))
                    _meta_parts = [p for p in [_pub, _source, _time[:16] if _time else ""] if p]
                    _meta = " · ".join(_meta_parts)
                    _meta_html = f'<div class="news-meta">{_meta}</div>' if _meta else ""

                    st.markdown(
                        f'<div class="news-card news-{_impact_colors.get(_impact_level, "impact-none")}">'
                        f'<div class="news-title">{_num_badge}{_title_html}</div>'
                        f"{_meta_html}"
                        f"{_affected_html}"
                        f"{_reason_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        if _other_news:
            with st.expander(f"📋 その他のニュース（{len(_other_news)}件）", expanded=False):
                for news_item in _other_news:
                    # Why: 外部ソース由来テキストの XSS 防止
                    _link = _html_mod.escape(news_item.get("link", ""))
                    _safe_title = _html_mod.escape(news_item.get("title", ""))
                    _disp_no = news_item.get("_display_number", "")
                    _num_badge = f'<span class="news-number">#{_disp_no}</span>' if _disp_no else ""
                    _title_html = (
                        f'<a href="{_link}" target="_blank" rel="noopener noreferrer">{_safe_title}</a>'
                        if _link
                        else _safe_title
                    )
                    _pub = _html_mod.escape(news_item.get("publisher", ""))
                    _time = news_item.get("publish_time", "")
                    _source = _html_mod.escape(news_item.get("source_name", ""))
                    _meta_parts = [p for p in [_pub, _source, _time[:16] if _time else ""] if p]
                    _meta = " · ".join(_meta_parts)

                    _cat_badges = ""
                    for cat in news_item.get("categories", []):
                        _cat_badges += (
                            f'<span class="news-badge news-badge-category">{cat["icon"]} {cat["label"]}</span>'
                        )

                    st.markdown(
                        f'<div class="news-card news-impact-none">'
                        f'<div class="news-title">{_num_badge}{_title_html}</div>'
                        f'<div class="news-meta">{_meta}</div>'
                        f"<div>{_cat_badges}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("📰 経済ニュースの取得なし（ネットワーク接続を確認してください）")

    # --- Copilot CLI 実行ログ ---
    _cli_logs = copilot_get_logs()
    if _cli_logs:
        with st.expander(f"🔍 Copilot CLI 実行ログ（{len(_cli_logs)}件）", expanded=False):
            _log_col1, _log_col2 = st.columns([6, 1])
            with _log_col2:
                if st.button("🗑️ クリア", key="clear_cli_logs"):
                    copilot_clear_logs()
                    st.rerun()
            for _log in _cli_logs:
                import datetime as _dt

                _ts = _dt.datetime.fromtimestamp(_log.timestamp).strftime("%H:%M:%S")
                _status = "✅" if _log.success else "❌"
                _src = f" [{_log.source}]" if _log.source else ""
                _header = f"{_status} {_ts} — {_log.model} ({_log.duration_sec:.1f}s){_src}"
                if _log.success:
                    _detail = (
                        f"**プロンプト** (先頭150文字):\n```\n{_log.prompt_preview}\n```\n\n"
                        f"**応答** ({_log.response_length}文字):\n```\n{_log.response_preview}\n```"
                    )
                else:
                    _detail = (
                        f"**プロンプト** (先頭150文字):\n```\n{_log.prompt_preview}\n```\n\n**エラー**: `{_log.error}`"
                    )
                with st.expander(_header, expanded=False):
                    st.markdown(_detail)
