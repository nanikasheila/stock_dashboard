"""Copilot チャットタブ描画モジュール.

``app.py`` の ``with _tab_copilot:`` ブロックを切り出したモジュール。
ダッシュボードコンテキスト構築 / チャット履歴表示 / Copilot CLI 呼び出しを担当する。

公開 API
--------
render_copilot_tab(...)
    Copilot チャットタブのコンテンツを描画する。

build_chat_context(...)
    ダッシュボードデータをプロンプトコンテキスト文字列に変換する。
    単体テストや将来の拡張用に公開している。
"""

from __future__ import annotations

import html as _html_mod

import streamlit as st
from state_keys import SK

from components.copilot_client import call_with_session
from components.copilot_client import get_available_models as _get_copilot_models
from components.data_loader import compute_risk_metrics


def build_chat_context(
    *,
    total_value: float,
    daily_change_jpy: float,
    daily_change_pct: float,
    unrealized_pnl: float,
    unrealized_pnl_pct: float,
    realized_pnl: float,
    positions: list[dict],
    history_df,
    health_data: dict | None,
    econ_news: list[dict],
) -> str:
    """ダッシュボード上の全情報をプロンプトコンテキストとして構築する.

    Parameters
    ----------
    total_value:        総資産（円換算）
    daily_change_jpy:   前日比（円）
    daily_change_pct:   前日比（%）
    unrealized_pnl:     含み損益（円）
    unrealized_pnl_pct: 含み損益率（%）
    realized_pnl:       実現損益（円）
    positions:          保有銘柄リスト
    history_df:         ポートフォリオ価格履歴 DataFrame
    health_data:        ヘルスチェック結果（失敗時 None）
    econ_news:          経済ニュースリスト

    Returns
    -------
    str
        Copilot CLI へ送るプロンプトに埋め込むコンテキスト文字列。
    """
    parts: list[str] = []
    parts.append("## ポートフォリオ概要")
    parts.append(f"総資産: ¥{total_value:,.0f}")
    parts.append(f"前日比: ¥{daily_change_jpy:+,.0f} ({daily_change_pct:+.1f}%)")
    parts.append(f"含み損益: ¥{unrealized_pnl:,.0f} ({unrealized_pnl_pct:+.1f}%)")
    parts.append(f"実現損益: ¥{realized_pnl:,.0f}")
    parts.append(f"トータル損益: ¥{unrealized_pnl + realized_pnl:,.0f}")
    parts.append(f"銘柄数: {len(positions)}")

    # リスク指標
    if not history_df.empty:
        try:
            _ctx_risk = compute_risk_metrics(history_df)
            parts.append("\n## リスク指標")
            parts.append(f"シャープレシオ: {_ctx_risk['sharpe_ratio']:.2f}")
            parts.append(f"ボラティリティ: {_ctx_risk['volatility_pct']:.1f}%")
            parts.append(f"最大ドローダウン: {_ctx_risk['max_drawdown_pct']:.1f}%")
        except Exception:
            pass

    # Holdings summary: sector aggregation + top-5 individual entries.
    # Why: sending every position inflates the prompt for large portfolios;
    #      sector roll-ups preserve the strategic picture at lower token cost.
    parts.append("\n## 保有銘柄サマリー")
    parts.append(f"銘柄数: {len(positions)}, 総資産: ¥{total_value:,.0f}")

    _sector_totals: dict[str, dict] = {}
    for _p in positions:
        _p_sector = _p.get("sector", "その他") or "その他"
        _p_eval = _p.get("evaluation_jpy", 0)
        if _p_sector not in _sector_totals:
            _sector_totals[_p_sector] = {"total_jpy": 0.0, "count": 0}
        _sector_totals[_p_sector]["total_jpy"] += _p_eval
        _sector_totals[_p_sector]["count"] += 1

    _sector_parts: list[str] = []
    for _sector_name, _sector_data in sorted(
        _sector_totals.items(), key=lambda item: item[1]["total_jpy"], reverse=True
    ):
        _sector_weight = (_sector_data["total_jpy"] / total_value * 100) if total_value else 0
        _sector_parts.append(f"{_sector_name} {_sector_weight:.1f}%({_sector_data['count']}銘柄)")
    parts.append("セクター構成: " + ", ".join(_sector_parts))

    _sorted_positions = sorted(positions, key=lambda item: item.get("evaluation_jpy", 0), reverse=True)
    parts.append("\n上位5銘柄（構成比順）:")
    for _p in _sorted_positions[:5]:
        _sym = _p.get("symbol", "")
        _name = _p.get("name", "")
        _pnl = _p.get("pnl_pct", 0)
        _eval_jpy = _p.get("evaluation_jpy", 0)
        _sector = _p.get("sector", "")
        _weight = (_eval_jpy / total_value * 100) if total_value else 0
        parts.append(
            f"- {_name} ({_sym}): 評価額¥{_eval_jpy:,.0f} 構成比{_weight:.1f}% 損益{_pnl:+.1f}% セクター:{_sector}"
        )

    # ヘルスチェック結果
    if health_data is not None:
        _hc_pos = health_data["positions"]
        _hc_alerts_list = health_data["sell_alerts"]
        _alert_pos = [p for p in _hc_pos if p.get("alert_level") != "none"]
        if _alert_pos:
            parts.append("\nアラート対象:")
            for _hp in _alert_pos:
                _hp_sym = _hp.get("symbol", "")
                _hp_name = _hp.get("name", "")
                _hp_level = _hp.get("alert_level", "")
                _hp_reasons = ", ".join(_hp.get("alert_reasons", []))
                _hp_trend = _hp.get("trend", "")
                parts.append(f"- {_hp_name} ({_hp_sym}): [{_hp_level}] {_hp_reasons} トレンド:{_hp_trend}")

        # 売りアラート
        if _hc_alerts_list:
            parts.append("\n## 売りタイミング通知")
            for _sa_ctx in _hc_alerts_list:
                parts.append(
                    f"- {_sa_ctx.get('name', '')} ({_sa_ctx.get('symbol', '')}): {_sa_ctx.get('action', '')} — {_sa_ctx.get('reason', '')}"
                )

    # LLM ヘルスサマリー（session_stateに格納されていれば利用）
    _chat_hc_summary = st.session_state.get(SK.HC_LLM_SUMMARY_DATA)
    if _chat_hc_summary:
        parts.append("\n## AI ヘルスチェック分析")
        _overview_ctx = _chat_hc_summary.get("overview", "")
        if _overview_ctx:
            parts.append(_overview_ctx)
        _warning_ctx = _chat_hc_summary.get("risk_warning", "")
        if _warning_ctx:
            parts.append(f"リスク注意: {_warning_ctx}")

    # 経済ニュース
    if econ_news:
        _impact_items = [n for n in econ_news if n.get("portfolio_impact", {}).get("impact_level") != "none"]
        if _impact_items:
            parts.append("\n## 経済ニュース（PF影響あり）")
            for _ni in _impact_items[:5]:  # limit to 5 to keep context concise
                _ni_title = _ni.get("title", "")
                _ni_impact = _ni.get("portfolio_impact", {})
                _ni_level = _ni_impact.get("impact_level", "")
                _ni_reason = _ni_impact.get("reason", "")
                _ni_url = _ni.get("link", "")
                _ni_url_part = f" URL:{_ni_url}" if _ni_url else ""
                parts.append(f"- [{_ni_level}] {_ni_title}: {_ni_reason}{_ni_url_part}")

    return "\n".join(parts)


def render_copilot_tab(
    *,
    snapshot: dict,
    history_df,
    positions: list[dict],
    total_value: float,
    unrealized_pnl: float,
    unrealized_pnl_pct: float,
    realized_pnl: float,
    daily_change_jpy: float,
    daily_change_pct: float,
    health_data: dict | None,
    econ_news: list[dict],
    chat_model: str,
) -> None:
    """Copilot チャットタブのコンテンツを描画する.

    Parameters
    ----------
    snapshot:           ポートフォリオスナップショット
    history_df:         ポートフォリオ価格履歴 DataFrame
    positions:          保有銘柄リスト
    total_value:        総資産（円換算）
    unrealized_pnl:     含み損益（円）
    unrealized_pnl_pct: 含み損益率（%）
    realized_pnl:       実現損益（円）
    daily_change_jpy:   前日比（円）
    daily_change_pct:   前日比（%）
    health_data:        ヘルスチェック結果（失敗時 None）
    econ_news:          経済ニュースリスト
    chat_model:         Copilot チャットモデル識別子
    """
    st.markdown('<div id="copilot-chat" role="region" aria-label="Copilot チャット"></div>', unsafe_allow_html=True)
    st.markdown("### 💬 Copilot に相談")
    st.caption("ダッシュボードの全データを踏まえて、Copilot に自由に質問できます。")

    # チャット履歴の初期化
    if SK.COPILOT_CHAT_MESSAGES not in st.session_state:
        st.session_state[SK.COPILOT_CHAT_MESSAGES] = []
    if SK.COPILOT_SESSION_ID not in st.session_state:
        st.session_state[SK.COPILOT_SESSION_ID] = None

    # コンテキストバッジ
    _ctx_items = []
    _ctx_items.append(f"銘柄 {len(positions)}")
    if health_data is not None:
        _n_alerts = sum(1 for p in health_data["positions"] if p.get("alert_level") != "none")
        if _n_alerts:
            _ctx_items.append(f"アラート {_n_alerts}")
        if health_data["sell_alerts"]:
            _ctx_items.append(f"売り通知 {len(health_data['sell_alerts'])}")
    if st.session_state.get(SK.HC_LLM_SUMMARY_DATA):
        _ctx_items.append("AI分析")
    if econ_news:
        _ctx_items.append(f"ニュース {len(econ_news)}")

    _badges_html = " ".join(f'<span class="copilot-chat-context-badge">{item}</span>' for item in _ctx_items)
    st.markdown(
        f'<div style="margin-bottom:10px;">'
        f'<span style="font-size:0.82rem; opacity:0.7;">📎 自動添付コンテキスト:</span> '
        f"{_badges_html}</div>",
        unsafe_allow_html=True,
    )

    # Model display, new-session button, and full-clear button.
    # Why: users need to start a fresh CLI session without losing visible
    #      history (new-session) or wipe everything at once (clear).
    _chat_col_model, _chat_col_new, _chat_col_clear = st.columns([3, 1, 1])
    with _chat_col_model:
        _chat_model_ids = [m[0] for m in _get_copilot_models()]
        _chat_model_labels = [m[1] for m in _get_copilot_models()]
        _chat_model_current_idx = _chat_model_ids.index(chat_model) if chat_model in _chat_model_ids else 0
        st.caption(f"🧠 モデル: **{_chat_model_labels[_chat_model_current_idx]}**（設定で変更可能）")
        if st.session_state.get(SK.COPILOT_SESSION_ID) is not None:
            st.caption("🔗 セッション継続中")
    with _chat_col_new:
        if st.button("🔄 新規セッション", key="copilot_chat_new_session"):
            # Reset session ID only; keep chat history for reference.
            st.session_state[SK.COPILOT_SESSION_ID] = None
            st.session_state[SK.COPILOT_CHAT_MESSAGES].append({"role": "assistant", "content": "---"})
            st.rerun()
    with _chat_col_clear:
        if st.button("🗑️ クリア", key="copilot_chat_clear"):
            # Full reset: history + session ID.
            st.session_state[SK.COPILOT_CHAT_MESSAGES] = []
            st.session_state[SK.COPILOT_SESSION_ID] = None
            st.rerun()

    # チャット履歴表示
    for _msg in st.session_state[SK.COPILOT_CHAT_MESSAGES]:
        if _msg["role"] == "user":
            st.markdown(
                f'<div class="copilot-chat-msg copilot-chat-msg-user">'
                f'<div class="copilot-chat-msg-role">👤 あなた</div>'
                f'<div class="copilot-chat-msg-text">{_html_mod.escape(str(_msg["content"]))}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="copilot-chat-msg copilot-chat-msg-ai">'
                '<div class="copilot-chat-msg-role">🤖 Copilot</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(_msg["content"])

    # 入力欄
    _chat_input = st.chat_input(
        "ダッシュボードについて質問...",
        key="copilot_chat_input",
    )

    if _chat_input:
        st.session_state[SK.COPILOT_CHAT_MESSAGES].append({"role": "user", "content": _chat_input})

        _dashboard_ctx = build_chat_context(
            total_value=total_value,
            daily_change_jpy=daily_change_jpy,
            daily_change_pct=daily_change_pct,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            realized_pnl=realized_pnl,
            positions=positions,
            history_df=history_df,
            health_data=health_data,
            econ_news=econ_news,
        )

        # When a session is active, the CLI already holds earlier context
        # via --resume, so re-sending the full history wastes tokens.
        _has_session = st.session_state.get(SK.COPILOT_SESSION_ID) is not None

        _chat_prompt = (
            "あなたはポートフォリオ分析の専門家です。\n"
            "以下のダッシュボード情報を踏まえて、ユーザーの質問に日本語で回答してください。\n"
            "回答は簡潔かつ具体的に。数値データを活用してください。\n"
            "ニュースURLが提供されている場合は、必要に応じてURLにアクセスして最新情報を確認してください。\n\n"
            f"--- ダッシュボードデータ ---\n{_dashboard_ctx}\n\n"
        )

        # Include conversation history only for the first turn (no session yet).
        if not _has_session:
            _recent_msgs = st.session_state[SK.COPILOT_CHAT_MESSAGES][-10:]
            if len(_recent_msgs) > 1:
                _chat_prompt += "--- 会話履歴 ---\n"
                for _hm in _recent_msgs[:-1]:  # exclude the latest user message
                    _hm_role = "ユーザー" if _hm["role"] == "user" else "アシスタント"
                    _chat_prompt += f"{_hm_role}: {_hm['content']}\n"
                _chat_prompt += "\n"

        _chat_prompt += f"--- ユーザーの質問 ---\n{_chat_input}"

        with st.spinner("🤖 Copilot が考えています..."):
            _result = call_with_session(
                _chat_prompt,
                model=chat_model,
                timeout=120,
                source="dashboard_chat",
                session_id=st.session_state.get(SK.COPILOT_SESSION_ID),
            )

        # Persist the session ID for subsequent turns.
        if _result.session_id:
            st.session_state[SK.COPILOT_SESSION_ID] = _result.session_id

        if _result.response:
            st.session_state[SK.COPILOT_CHAT_MESSAGES].append({"role": "assistant", "content": _result.response})
        else:
            st.session_state[SK.COPILOT_CHAT_MESSAGES].append(
                {"role": "assistant", "content": "⚠️ 応答を取得できませんでした。Copilot CLI の状態を確認してください。"}
            )
        st.rerun()
