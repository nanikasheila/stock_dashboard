"""ポートフォリオダッシュボード — Streamlit アプリ.

総資産推移 / 銘柄別評価額 / セクター構成 / 月次サマリー を
インタラクティブなグラフで表示する。

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- コンポーネントを import ---
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from state_keys import SK

from components.copilot_client import (
    get_available_models as _get_copilot_models,
)
from components.copilot_client import (
    get_available_models as _get_llm_models,
)
from components.data_loader import (
    build_portfolio_history,
    clear_price_cache,
    compute_benchmark_excess,
    compute_daily_change,
    compute_risk_metrics,
    compute_top_worst_performers,
    compute_weight_drift,
    fetch_economic_news,
    get_benchmark_series,
    get_current_snapshot,
    get_sector_breakdown,
    get_trade_activity,
    run_dashboard_health_check,
)
from components.kpi_helpers import alert_badge_card, kpi_main_card, kpi_sub_card, risk_card
from components.llm_analyzer import (
    CACHE_TTL_OPTIONS as LLM_CACHE_OPTIONS,
)
from components.llm_analyzer import (
    clear_cache as llm_clear_cache,
)
from components.llm_analyzer import (
    clear_health_summary_cache as llm_clear_health_summary_cache,
)
from components.llm_analyzer import (
    clear_summary_cache as llm_clear_summary_cache,
)
from components.llm_analyzer import (
    clear_unified_cache as llm_clear_unified_cache,
)
from components.llm_analyzer import (
    generate_insights,
)
from components.llm_analyzer import (
    get_cache_info as llm_get_cache_info,
)
from components.llm_analyzer import (
    is_available as llm_is_available,
)
from components.settings_store import load_settings, save_settings
from components.tab_charts import render_charts_tab
from components.tab_copilot import render_copilot_tab
from components.tab_health import render_health_tab
from components.tab_holdings import render_holdings_tab
from components.tab_insights import render_insights_tab
from components.tab_monthly import render_monthly_tab
from components.watchlist import render_watchlist

# =====================================================================
# ページ設定
# =====================================================================
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS — 外部ファイルから読み込み
# Why: 約400行のCSSをapp.pyに埋め込むと保守性が低い
# How: static/dashboard.css に分離し、起動時に読み込む
_CSS_PATH = _SCRIPT_DIR / "static" / "dashboard.css"
_custom_css = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""
st.markdown(f"<style>{_custom_css}</style>", unsafe_allow_html=True)


# =====================================================================
# ポートフォリオ変更検知 — CSV / 取引履歴の更新を自動反映
# =====================================================================
def _get_portfolio_fingerprint() -> str:
    """portfolio.csv と取引履歴ファイルの mtime を結合した文字列を返す.

    ファイルが更新されると値が変わるので、キャッシュ無効化のトリガーに使う。
    """
    from src.core.portfolio.portfolio_manager import DEFAULT_CSV_PATH as _CSV_PATH

    parts: list[str] = []
    # portfolio.csv の mtime
    csv_p = Path(_CSV_PATH)
    if csv_p.exists():
        parts.append(f"csv:{csv_p.stat().st_mtime}")
    # 取引履歴ディレクトリ内の最新 mtime
    trade_dir = Path(_SCRIPT_DIR).resolve() / "data" / "history" / "trade"
    if trade_dir.is_dir():
        json_files = list(trade_dir.glob("*.json"))
        if json_files:
            latest = max(f.stat().st_mtime for f in json_files)
            parts.append(f"trade:{latest}:{len(json_files)}")
    return "|".join(parts) if parts else "empty"


def _check_portfolio_changed() -> bool:
    """portfolio.csv / trade 履歴が前回読込時から変更されたか判定.

    変更があれば全キャッシュをクリアして True を返す。
    """
    current_fp = _get_portfolio_fingerprint()
    prev_fp = st.session_state.get(SK.PORTFOLIO_FINGERPRINT, None)
    if prev_fp is not None and current_fp != prev_fp:
        # ファイルが変更された → キャッシュを全クリアして再読込
        load_snapshot.clear()
        load_history.clear()
        load_trade_activity.clear()
        load_health_check.clear()
        load_economic_news.clear()
        load_behavior_insight_cached.clear()
        load_timing_insight_cached.clear()
        st.session_state[SK.PORTFOLIO_FINGERPRINT] = current_fp
        st.session_state[SK.LAST_REFRESH] = time.strftime("%Y-%m-%d %H:%M:%S")
        return True
    st.session_state[SK.PORTFOLIO_FINGERPRINT] = current_fp
    return False


# =====================================================================
# データ取得（キャッシュ付き）— サイドバーより先に定義
# =====================================================================
@st.cache_data(ttl=300, show_spinner="データを取得中...")
def load_snapshot():
    return get_current_snapshot()


@st.cache_data(ttl=300, show_spinner="株価履歴を取得中...")
def load_history(period_val: str):
    return build_portfolio_history(period=period_val)


@st.cache_data(ttl=300, show_spinner="取引データを集計中...")
def load_trade_activity():
    return get_trade_activity()


@st.cache_data(ttl=600, show_spinner="ヘルスチェック実行中...")
def load_health_check():
    return run_dashboard_health_check()


@st.cache_data(ttl=300, show_spinner="行動インサイトを計算中...")
def load_behavior_insight_cached():
    from components.dl_behavior import load_behavior_insight

    try:
        return load_behavior_insight()
    except Exception as _exc:
        import logging

        logging.getLogger(__name__).warning("load_behavior_insight failed: %s", _exc)
        from src.core.behavior.models import BehaviorInsight

        return BehaviorInsight.empty()


@st.cache_data(ttl=300, show_spinner="タイミング分析を計算中...")
def load_timing_insight_cached():
    from components.dl_behavior import load_timing_insight

    try:
        return load_timing_insight()
    except Exception as _exc:
        import logging

        logging.getLogger(__name__).warning("load_timing_insight failed: %s", _exc)
        from src.core.behavior.models import PortfolioTimingInsight

        return PortfolioTimingInsight.empty()


@st.cache_data(ttl=600, show_spinner="経済ニュースを取得中...")
def load_economic_news(
    _positions_key: str,
    positions: list,
    fx_rates: dict,
    llm_enabled: bool = False,
    llm_model: str | None = None,
    llm_cache_ttl: int = 3600,
):
    """経済ニュースを取得してPF影響を分析する.

    _positions_key はキャッシュキー用（保有銘柄が変わったら再取得）。
    llm_enabled / llm_model でLLM分析の有無・モデルをキャッシュキーに含む。
    llm_cache_ttl はLLM分析結果のキャッシュ有効期間（秒）。
    """
    return fetch_economic_news(
        positions=positions,
        fx_rates=fx_rates,
        llm_enabled=llm_enabled,
        llm_model=llm_model,
        llm_cache_ttl=llm_cache_ttl,
    )


# =====================================================================
# サイドバー（タブ: 目次 / 設定）
# =====================================================================
st.sidebar.title("📊 Portfolio Dashboard")

_tab_toc, _tab_settings, _tab_help = st.sidebar.tabs(["📑 目次", "⚙️ 設定", "❓ 用語集"])

# --- 目次タブ ---
# Why: 静的テキストのTOCはクリック不可でサイドバーを圧迫し UX を低下させる
# How: components.html で JS 付きボタンを描画し、クリックでメインタブを切り替える。
#      タブのラベルテキストで DOM 上の button[data-baseweb="tab"] を検索して click() する。
with _tab_toc:
    import streamlit.components.v1 as _stc_v1

    _toc_nav_items = [
        ("🏥", "ヘルス & ニュース"),
        ("📊", "チャート分析"),
        ("🏢", "保有構成"),
        ("📅", "月次 & 売買"),
        ("👀", "ウォッチリスト"),
        ("💬", "Copilot"),
        ("📈", "インサイト"),
    ]
    _toc_buttons_html = ""
    for _toc_icon, _toc_label in _toc_nav_items:
        _toc_buttons_html += (
            f'<button class="toc-nav" onclick="switchTab(\'{_toc_label}\')">{_toc_icon} {_toc_label}</button>'
        )
    _stc_v1.html(
        """
        <style>
        .toc-hint{font-size:.78rem;opacity:.55;margin-bottom:6px}
        .toc-nav{
            display:block;width:100%;text-align:left;
            padding:7px 10px;margin:3px 0;
            background:transparent;border:1px solid rgba(128,128,128,.18);
            border-radius:6px;cursor:pointer;font-size:.88rem;
            color:inherit;transition:background .15s,border-color .15s;
            font-family:inherit;
        }
        .toc-nav:hover{background:rgba(99,102,241,.12);border-color:rgba(99,102,241,.4)}
        </style>
        <div class="toc-hint">📈 サマリーは常時表示</div>
        """
        + _toc_buttons_html
        + """
        <script>
        function switchTab(label){
            const tabs=window.parent.document.querySelectorAll(
                'button[data-baseweb="tab"]'
            );
            for(const t of tabs){
                if(t.textContent.includes(label)){t.click();break;}
            }
        }
        </script>
        """,
        height=280,
    )

# --- 設定の読み込み ---
if SK.SAVED_SETTINGS not in st.session_state:
    st.session_state[SK.SAVED_SETTINGS] = load_settings()
_saved = st.session_state[SK.SAVED_SETTINGS]

# --- 設定タブ ---
# Why: 設定項目が多く一覧性が悪い
# How: expander で論理グループに折りたたみ、基本設定のみデフォルト展開
with _tab_settings:
    _PERIOD_OPTIONS = [
        ("1ヶ月", "1mo"),
        ("3ヶ月", "3mo"),
        ("6ヶ月", "6mo"),
        ("1年", "1y"),
        ("2年", "2y"),
        ("3年", "3y"),
        ("5年", "5y"),
        ("全期間", "max"),
    ]
    _period_labels = [label for label, _ in _PERIOD_OPTIONS]
    _period_saved_idx = _period_labels.index(_saved["period_label"]) if _saved["period_label"] in _period_labels else 1

    period_label = st.selectbox(
        "📅 表示期間",
        options=_period_labels,
        index=_period_saved_idx,
        help="株価履歴の取得期間",
    )
    period = dict(_PERIOD_OPTIONS)[period_label]

    _chart_styles = ["積み上げ面", "折れ線", "積み上げ棒"]
    _chart_saved_idx = _chart_styles.index(_saved["chart_style"]) if _saved["chart_style"] in _chart_styles else 0

    chart_style = st.radio(
        "🎨 チャートスタイル",
        options=_chart_styles,
        index=_chart_saved_idx,
    )

    show_invested = st.checkbox(
        "投資額 vs 評価額を表示",
        value=_saved["show_invested"],
    )

    # ベンチマーク選択
    _BENCHMARK_OPTIONS = {
        "なし": None,
        "S&P 500 (SPY)": "SPY",
        "VTI (米国全体)": "VTI",
        "日経225 (^N225)": "^N225",
        "TOPIX (^TPX)": "1306.T",
    }
    _bench_labels = list(_BENCHMARK_OPTIONS.keys())
    _bench_saved_idx = (
        _bench_labels.index(_saved["benchmark_label"]) if _saved["benchmark_label"] in _bench_labels else 0
    )

    benchmark_label = st.selectbox(
        "📏 ベンチマーク比較",
        options=_bench_labels,
        index=_bench_saved_idx,
        help="総資産推移にベンチマークのパフォーマンスを重ねて表示",
    )
    benchmark_symbol = _BENCHMARK_OPTIONS[benchmark_label]

    show_individual = st.checkbox(
        "銘柄別の個別チャートを表示",
        value=_saved["show_individual"],
    )

    with st.expander("🎯 目標・将来推定", expanded=False):
        show_projection = st.checkbox(
            "目標ライン & 将来推定を表示",
            value=_saved["show_projection"],
        )

        target_amount = (
            st.number_input(
                "🎯 目標資産額（万円）",
                min_value=0,
                max_value=100000,
                value=_saved["target_amount_man"],
                step=500,
                help="総資産推移グラフに水平ラインとして表示",
            )
            * 10000
        )  # 万円→円

        projection_years = st.slider(
            "📅 推定期間（年）",
            min_value=1,
            max_value=20,
            value=_saved["projection_years"],
            help="現在の保有銘柄のリターン推定に基づく将来推移",
        )

    with st.expander("🔄 データ更新", expanded=False):
        _REFRESH_OPTIONS = [
            ("なし（手動のみ）", 0),
            ("1分", 60),
            ("5分", 300),
            ("15分", 900),
            ("30分", 1800),
            ("1時間", 3600),
        ]
        _refresh_labels = [label for label, _ in _REFRESH_OPTIONS]
        _refresh_saved_idx = (
            _refresh_labels.index(_saved["auto_refresh_label"])
            if _saved["auto_refresh_label"] in _refresh_labels
            else 2
        )

        auto_refresh_label = st.selectbox(
            "⏱ 自動更新間隔",
            options=_refresh_labels,
            index=_refresh_saved_idx,
            help="選択した間隔でダッシュボードを自動リロードします",
        )
        auto_refresh_sec = dict(_REFRESH_OPTIONS)[auto_refresh_label]

    # --- 手動更新ボタン ---
    if st.button("📥 今すぐ更新", help="キャッシュを無視してデータを即座に再取得します"):
        # a. Streamlit インメモリキャッシュをクリア
        load_snapshot.clear()
        load_history.clear()
        load_trade_activity.clear()
        load_health_check.clear()
        load_economic_news.clear()
        load_behavior_insight_cached.clear()
        load_timing_insight_cached.clear()
        # b. ディスクキャッシュ（価格履歴 CSV）を削除
        _deleted = clear_price_cache()
        # c. 手動更新タイムスタンプを記録
        _now = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state[SK.LAST_MANUAL_REFRESH] = _now
        st.session_state[SK.LAST_REFRESH] = _now
        # d. 即座にリロード
        st.rerun()

    _last_manual = st.session_state.get(SK.LAST_MANUAL_REFRESH)
    if _last_manual:
        st.caption(f"最終手動更新: {_last_manual}")

    with st.expander("🤖 AI 設定", expanded=False):
        _llm_available = llm_is_available()

        llm_enabled = st.checkbox(
            "LLMでニュースを分析",
            value=_saved.get("llm_enabled", False),
            help=(
                "GitHub Copilot CLI を使ってニュースのカテゴリ分類・PF影響を"
                "AIで分析します。`copilot` CLI のインストールが必要です。"
            ),
            disabled=not _llm_available,
        )

        if not _llm_available:
            st.caption("⚠️ `copilot` CLI が見つかりません。GitHub Copilot CLI をインストールしてください")

        # LLM 分析トリガーモード（自動 / 手動）
        llm_auto_analyze = st.checkbox(
            "ページ更新時に自動分析",
            value=_saved.get("llm_auto_analyze", False),
            help=("OFF にすると LLM 分析はボタンクリック時のみ実行されます。Premium Request の過剰消費を防ぎます。"),
            disabled=not llm_enabled,
        )

        _model_ids = [m[0] for m in _get_llm_models()]
        _model_labels = [m[1] for m in _get_llm_models()]
        _saved_model = _saved.get("llm_model", "gpt-4.1")
        _model_saved_idx = _model_ids.index(_saved_model) if _saved_model in _model_ids else 1

        llm_model_label = st.selectbox(
            "🧠 分析モデル",
            options=_model_labels,
            index=_model_saved_idx,
            help="ニュース分析に使用するLLMモデル",
            disabled=not llm_enabled,
        )
        llm_model = _model_ids[_model_labels.index(llm_model_label)]

        # LLM 分析キャッシュ TTL
        _ttl_labels = [t[0] for t in LLM_CACHE_OPTIONS]
        _ttl_values = [t[1] for t in LLM_CACHE_OPTIONS]
        _saved_ttl_label = _saved.get("llm_cache_ttl_label", "1時間")
        _ttl_saved_idx = _ttl_labels.index(_saved_ttl_label) if _saved_ttl_label in _ttl_labels else 0

        llm_cache_ttl_label = st.selectbox(
            "⏳ 分析キャッシュ保持",
            options=_ttl_labels,
            index=_ttl_saved_idx,
            help=("同じニュースに対して LLM 再分析をスキップする期間。Premium Request の消費を抑えます。"),
            disabled=not llm_enabled,
        )
        llm_cache_ttl_sec = _ttl_values[_ttl_labels.index(llm_cache_ttl_label)]

        # --- Copilot チャットモデル ---
        st.markdown("---")
        _chat_model_ids = [m[0] for m in _get_copilot_models()]
        _chat_model_labels = [m[1] for m in _get_copilot_models()]
        _saved_chat_model = _saved.get("chat_model", "claude-sonnet-4")
        _chat_model_saved_idx = _chat_model_ids.index(_saved_chat_model) if _saved_chat_model in _chat_model_ids else 0
        chat_model_label = st.selectbox(
            "🧠 チャットモデル",
            options=_chat_model_labels,
            index=_chat_model_saved_idx,
            help="Copilot チャットで使用するモデル（分析モデルとは独立）",
        )
        chat_model = _chat_model_ids[_chat_model_labels.index(chat_model_label)]

        # --- 機能別 AI ON/OFF ---
        st.markdown("---")
        st.markdown("**機能別 AI 設定**")

        insights_enabled = st.checkbox(
            "💡 AIインサイト",
            value=_saved.get("insights_enabled", True),
            help="KPIサマリーの下にAI生成のアクション可能なインサイトを表示",
            disabled=not _llm_available,
        )

        trade_preview_enabled = st.checkbox(
            "📊 取引影響プレビュー",
            value=_saved.get("trade_preview_enabled", True),
            help="取引フォームで影響分析ボタンを表示",
        )

        trade_preview_llm = st.checkbox(
            "🤖 取引プレビューAIコメント",
            value=_saved.get("trade_preview_llm", True),
            help="取引影響プレビューでLLMコメントを表示",
            disabled=not _llm_available or not trade_preview_enabled,
        )

        watchlist_llm_enabled = st.checkbox(
            "👀 ウォッチリストAI分析",
            value=_saved.get("watchlist_llm_enabled", True),
            help="ウォッチリストの個別銘柄に対してAI分析ボタンを表示",
            disabled=not _llm_available,
        )

        attribution_llm_enabled = st.checkbox(
            "📈 パフォーマンス寄与LLMサマリー",
            value=_saved.get("attribution_llm_enabled", True),
            help="パフォーマンス寄与分析にLLMの要因分析コメントを表示",
            disabled=not _llm_available,
        )

    # キャッシュ状態を表示
    if llm_enabled:
        _ci = llm_get_cache_info()
        if _ci["cached"]:
            _age_min = _ci["age_sec"] // 60
            if _age_min < 60:
                _age_str = f"{_age_min}分前"
            else:
                _age_str = f"{_age_min // 60}時間{_age_min % 60}分前"
            st.caption(f"💾 キャッシュあり（{_age_str}に {_ci['model']} で分析済み）")
            if st.button("🔄 今すぐ再分析", key="llm_reanalyze", help="キャッシュを破棄して LLM 分析をやり直します"):
                llm_clear_cache()
                llm_clear_summary_cache()
                llm_clear_health_summary_cache()
                llm_clear_unified_cache()
                # session_state のLLM結果もクリア
                for _ss_key in [
                    "_llm_news_results",
                    "_llm_news_summary",
                    "_llm_hc_summary",
                    "_llm_analyzed_at",
                ]:
                    st.session_state.pop(_ss_key, None)
                st.rerun()
        else:
            if llm_auto_analyze:
                st.caption("💾 キャッシュなし（次回更新時に自動で LLM 分析を実行）")
            else:
                st.caption("💾 キャッシュなし（手動モード: ボタンで分析を実行）")

        # session_state に前回の分析結果があれば表示
        if SK.LLM_ANALYZED_AT in st.session_state:
            import datetime as _dt_ss

            _ss_at = _dt_ss.datetime.fromtimestamp(st.session_state[SK.LLM_ANALYZED_AT]).strftime("%H:%M:%S")
            st.caption(f"📌 セッション内分析: {_ss_at}")

    # --- 設定の自動保存 ---
    _current_settings = {
        "period_label": period_label,
        "chart_style": chart_style,
        "show_invested": show_invested,
        "benchmark_label": benchmark_label,
        "show_individual": show_individual,
        "show_projection": show_projection,
        "target_amount_man": int(target_amount // 10000),
        "projection_years": projection_years,
        "auto_refresh_label": auto_refresh_label,
        "llm_enabled": llm_enabled,
        "llm_auto_analyze": llm_auto_analyze,
        "llm_model": llm_model,
        "llm_cache_ttl_label": llm_cache_ttl_label,
        "chat_model": chat_model,
        "insights_enabled": insights_enabled,
        "trade_preview_enabled": trade_preview_enabled,
        "trade_preview_llm": trade_preview_llm,
        "watchlist_llm_enabled": watchlist_llm_enabled,
        "attribution_llm_enabled": attribution_llm_enabled,
    }
    if _current_settings != _saved:
        save_settings(_current_settings)
        st.session_state[SK.SAVED_SETTINGS] = _current_settings

# --- 用語集タブ ---
with _tab_help:
    _GLOSSARY = {
        "📈 パフォーマンス指標": {
            "評価額": "保有株数 × 現在株価で算出した現在の資産価値。",
            "損益率": "（現在評価額 − 投資額）÷ 投資額 × 100。投資に対するリターン。",
            "ドローダウン": "直近の最高値からの下落率。リスク管理の重要指標で、ポートフォリオがピークから何%下がったかを示す。",
            "シャープレシオ": "リスク（値動きのばらつき）1単位あたりのリターン。1以上で良好、2以上で優秀。リスク調整後のパフォーマンスを測る。",
            "ベンチマーク": "運用成果の比較基準となる指数（S&P500、日経225等）。ベンチマークを上回っていれば市場平均以上の成績。",
        },
        "🔍 テクニカル指標": {
            "SMA（単純移動平均）": "過去N日間の終値の平均。SMA50（短期）とSMA200（長期）がよく使われる。トレンドの方向を判断する基本指標。",
            "ゴールデンクロス": "短期移動平均（SMA50）が長期移動平均（SMA200）を下から上に突き抜けること。上昇トレンドへの転換を示唆。",
            "デッドクロス": "短期移動平均（SMA50）が長期移動平均（SMA200）を上から下に突き抜けること。下降トレンドへの転換を示唆し、売り検討のサイン。",
            "RSI（相対力指数）": "0〜100で買われすぎ・売られすぎを判定。70以上で買われすぎ（売り検討）、30以下で売られすぎ（買い検討）。",
            "ボリンジャーバンド": "移動平均の上下に標準偏差の帯を描いたもの。バンド外に出ると異常値で、反転の可能性。",
        },
        "📊 ポートフォリオ分析": {
            "ウェイトドリフト": "各銘柄の構成比が均等配分からどれだけズレているか。値上がりした銘柄が膨らみ過ぎていないか確認する指標。",
            "相関係数": "2銘柄の値動きの連動性。+1で完全連動、−1で逆の動き、0で無関係。相関が高い銘柄ばかりだと分散効果が薄れる。",
            "セクター構成": "銘柄を業種別（テクノロジー、金融、ヘルスケア等）に分類した配分比率。特定セクターへの集中を防ぐために確認。",
            "通貨エクスポージャー": "保有資産の通貨別の配分。為替変動によるリスクの偏りを確認するための指標。",
        },
        "🏥 ヘルスチェック": {
            "アラートレベル": "銘柄の健全性を4段階で判定。✅ 正常 → ⚡ 早期警告 → ⚠️ 注意 → 🚨 EXIT（売却検討）。",
            "バリュートラップ": "PERが低く割安に見えるが、業績悪化が原因で株価が下がり続ける銘柄。見せかけの割安に注意。",
            "還元安定度": "配当や自社株買いの継続性を評価。✅安定 / 📈増加 / ⚠️一時的 / 📉低下の4段階。",
        },
        "💰 バリュエーション": {
            "PER（株価収益率）": "株価 ÷ 1株利益(EPS)。株価が利益の何倍かを示す。低いほど割安だが、業績悪化による低PERには注意。",
            "PBR（株価純資産倍率）": "株価 ÷ 1株純資産。1倍以下は解散価値割れで割安とされるが、万能ではない。",
            "配当利回り": "年間配当金 ÷ 株価 × 100。高いほどインカム収入が多い。ただし株価下落による見かけの高利回りに注意。",
            "総還元率": "（配当金 + 自社株買い）÷ 時価総額。配当だけでなく自社株買いも含めた株主還元の総合指標。",
        },
        "🔮 将来推定": {
            "楽観 / 基本 / 悲観": "過去リターンの平均±標準偏差で3パターンの将来推移を推計。基本＝平均、楽観＝+1σ、悲観＝−1σ。",
            "目標ライン": "設定した目標資産額を水平線で表示。将来推定と重ねて達成時期の目安を確認できる。",
        },
    }

    for _cat_name, _terms in _GLOSSARY.items():
        with st.expander(_cat_name, expanded=False):
            for _term, _desc in _terms.items():
                st.markdown(f"**{_term}**")
                st.caption(_desc)

# 自動更新タイマー（タブ外に配置）
if auto_refresh_sec > 0:
    _refresh_count = st_autorefresh(
        interval=auto_refresh_sec * 1000,
        limit=0,  # 無制限
        key="auto_refresh",
    )
else:
    _refresh_count = 0

# ポートフォリオ変更検知 — CSV / 取引履歴が更新されたら自動で再読込
if _check_portfolio_changed():
    st.rerun()

# 手動更新ボタン（タブ外に配置）
if st.sidebar.button("🔄 今すぐ更新", width="stretch"):
    load_snapshot.clear()
    load_history.clear()
    load_trade_activity.clear()
    load_health_check.clear()
    load_economic_news.clear()
    load_behavior_insight_cached.clear()
    load_timing_insight_cached.clear()
    _cache_dir = Path(_SCRIPT_DIR).resolve() / "data" / "cache" / "price_history"
    if _cache_dir.exists():
        for f in _cache_dir.glob("*.csv"):
            f.unlink(missing_ok=True)
    st.rerun()

# 最終更新時刻を session_state で管理
if SK.LAST_REFRESH not in st.session_state:
    st.session_state[SK.LAST_REFRESH] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state[SK.PREV_REFRESH_COUNT] = 0

if _refresh_count > st.session_state.get(SK.PREV_REFRESH_COUNT, 0):
    load_snapshot.clear()
    load_history.clear()
    load_trade_activity.clear()
    load_health_check.clear()
    load_economic_news.clear()
    load_behavior_insight_cached.clear()
    load_timing_insight_cached.clear()
    st.session_state[SK.LAST_REFRESH] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state[SK.PREV_REFRESH_COUNT] = _refresh_count

st.sidebar.caption(f"最終更新: {st.session_state[SK.LAST_REFRESH]}\n\nData Source: yfinance + portfolio.csv")


# =====================================================================
# メインコンテンツ
# =====================================================================
st.title("💼 ポートフォリオダッシュボード")

# --- データ読み込み ---
try:
    with st.spinner("ポートフォリオデータを読み込み中..."):
        snapshot = load_snapshot()
        history_df = load_history(period)
except Exception as _data_err:
    st.error(f"⚠️ データ取得に失敗しました: {_data_err}")
    st.info("ネットワーク接続を確認するか、「🔄 今すぐ更新」ボタンで再試行してください。")
    st.stop()

# FXレート表示（サイドバー下部）
_fx = snapshot.get("fx_rates", {})
_fx_display = {k: v for k, v in _fx.items() if k != "JPY" and v != 1.0}
if _fx_display:
    with st.sidebar.expander("💱 為替レート", expanded=False):
        for cur, rate in sorted(_fx_display.items()):
            st.caption(f"{cur}/JPY: ¥{rate:,.2f}")

# =====================================================================
# KPI メトリクスカード
# =====================================================================
st.markdown('<div id="summary" role="region" aria-label="サマリー"></div>', unsafe_allow_html=True)
st.markdown("### 📈 サマリー")
_summary_as_of = snapshot.get("as_of", "")[:16].replace("T", " ") or "—"
st.caption(
    f"ポートフォリオ全体の現在価値・損益・リスク指標を一目で把握するセクションです。｜ 🕐 データ取得: {_summary_as_of}"
)

positions = snapshot["positions"]
total_value = snapshot["total_value_jpy"]
num_holdings = len([p for p in positions if p.get("sector") != "Cash"])

# --- 損益計算: 総平均法（移動平均法）で直接計算 ---
# Why: 日本の証券会社（SBI等）は総平均法で評価損益を表示する。
#      invested - realized の間接計算では現金残高の扱いが不整合になり
#      正確な含み損益が出なかった。
# How: _compute_pnl_moving_average() で取引履歴から銘柄ごとの
#      平均取得単価(JPY)を追跡し、現在評価額との差分で含み損益を算出。
#      実現損益も同じ総平均法で一貫して計算する。
_pnl_ma = snapshot.get("pnl_moving_avg", {})
unrealized_pnl = _pnl_ma.get("unrealized_total_jpy", 0)
realized_pnl = _pnl_ma.get("realized_total_jpy", 0)
_cost_basis_total = sum(_pnl_ma.get("cost_basis", {}).values())
unrealized_pnl_pct = (unrealized_pnl / _cost_basis_total * 100) if _cost_basis_total else 0

# KPI カードヘルパーは components.kpi_helpers から import 済み
# _kpi_main / _kpi_sub / _risk_card は後方互換エイリアスとして保持
_kpi_main = kpi_main_card
_kpi_sub = kpi_sub_card
_risk_card = risk_card

_unr_color = "#4ade80" if unrealized_pnl >= 0 else "#f87171"
_unr_sign = "+" if unrealized_pnl >= 0 else ""

# 前日比の算出
_daily = compute_daily_change(history_df)
_dc_jpy = _daily["daily_change_jpy"]
_dc_pct = _daily["daily_change_pct"]
_dc_sign = "+" if _dc_jpy >= 0 else ""
_dc_color = "#4ade80" if _dc_jpy >= 0 else "#f87171"
_dc_text = f"{_dc_sign}¥{_dc_jpy:,.0f}（{_dc_pct:+.2f}%）" if _dc_jpy != 0 else "--"
_dc_sub = f'<span style="color:{_dc_color};">前日比 {_dc_text}</span>' if _dc_jpy != 0 else ""

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(_kpi_main("トータル資産（円換算）", f"¥{total_value:,.0f}", sub=_dc_sub), unsafe_allow_html=True)
with col2:
    st.markdown(
        _kpi_main(
            "評価損益（含み）",
            f"{_unr_sign}¥{unrealized_pnl:,.0f}",
            sub=f"{unrealized_pnl_pct:+.2f}%",
            color=_unr_color,
        ),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        _kpi_main(
            "保有銘柄数",
            f"{num_holdings}",
            sub=f"更新: {snapshot['as_of'][:10]}",
            color="#60a5fa",
        ),
        unsafe_allow_html=True,
    )

# --- 小項目: 損益 ---
total_pnl = unrealized_pnl + realized_pnl
realized_sign = "+" if realized_pnl >= 0 else ""
total_pnl_sign = "+" if total_pnl >= 0 else ""
realized_color = "#4ade80" if realized_pnl >= 0 else "#f87171"
total_pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"

st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

sub_col1, sub_col2 = st.columns(2)
with sub_col1:
    st.markdown(
        _kpi_sub(
            "トータル損益（実現＋含み）",
            f"{total_pnl_sign}¥{total_pnl:,.0f}",
            color=total_pnl_color,
        ),
        unsafe_allow_html=True,
    )
with sub_col2:
    st.markdown(
        _kpi_sub(
            "実現損益（確定済）",
            f"{realized_sign}¥{realized_pnl:,.0f}",
            color=realized_color,
        ),
        unsafe_allow_html=True,
    )

# --- リスク指標 ---
if not history_df.empty:
    risk = compute_risk_metrics(history_df)

    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    _sharpe_color = (
        "#4ade80" if risk["sharpe_ratio"] >= 1.0 else ("#fbbf24" if risk["sharpe_ratio"] >= 0.5 else "#f87171")
    )
    _mdd_color = (
        "#4ade80" if risk["max_drawdown_pct"] > -10 else ("#fbbf24" if risk["max_drawdown_pct"] > -20 else "#f87171")
    )

    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    with rcol1:
        st.markdown(
            _risk_card(
                "年率リターン",
                f"{risk['annual_return_pct']:+.1f}%",
                "#4ade80" if risk["annual_return_pct"] > 0 else "#f87171",
            ),
            unsafe_allow_html=True,
        )
    with rcol2:
        st.markdown(_risk_card("ボラティリティ", f"{risk['annual_volatility_pct']:.1f}%"), unsafe_allow_html=True)
    with rcol3:
        st.markdown(_risk_card("Sharpe", f"{risk['sharpe_ratio']:.2f}", _sharpe_color), unsafe_allow_html=True)
    with rcol4:
        st.markdown(_risk_card("最大DD", f"{risk['max_drawdown_pct']:.1f}%", _mdd_color), unsafe_allow_html=True)
    with rcol5:
        st.markdown(_risk_card("Calmar", f"{risk['calmar_ratio']:.2f}"), unsafe_allow_html=True)

# --- ベンチマーク超過リターン ---
if benchmark_symbol and not history_df.empty:
    _bench_for_excess = get_benchmark_series(benchmark_symbol, history_df, period)
    _excess = compute_benchmark_excess(history_df, _bench_for_excess)
    if _excess is not None:
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
        _ex_color = "#4ade80" if _excess["excess_return_pct"] >= 0 else "#f87171"
        _ex_sign = "+" if _excess["excess_return_pct"] >= 0 else ""
        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.markdown(
                _risk_card(
                    "PFリターン",
                    f"{_excess['portfolio_return_pct']:+.1f}%",
                    "#4ade80" if _excess["portfolio_return_pct"] > 0 else "#f87171",
                ),
                unsafe_allow_html=True,
            )
        with ecol2:
            st.markdown(
                _risk_card(
                    f"{benchmark_label}リターン",
                    f"{_excess['benchmark_return_pct']:+.1f}%",
                    "#60a5fa",
                ),
                unsafe_allow_html=True,
            )
        with ecol3:
            st.markdown(
                _risk_card(
                    "超過リターン",
                    f"{_ex_sign}{_excess['excess_return_pct']:.1f}%",
                    _ex_color,
                ),
                unsafe_allow_html=True,
            )

# --- Top / Worst パフォーマー ---
if not history_df.empty:
    _performers = compute_top_worst_performers(history_df, top_n=3)
    _top = _performers["top"]
    _worst = _performers["worst"]
    if _top or _worst:
        st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            _top_html = '<div class="kpi-card kpi-sub" style="text-align:left;">'
            _top_html += '<div class="kpi-label">🟢 本日 Best</div>'
            for p in _top:
                _c = "#4ade80" if p["change_pct"] >= 0 else "#f87171"
                _top_html += (
                    f'<div style="display:flex; justify-content:space-between;'
                    f' padding:3px 0; font-size:0.9rem;">'
                    f"<span>{p['symbol']}</span>"
                    f'<span style="color:{_c}; font-weight:600;">'
                    f"{p['change_pct']:+.2f}%</span></div>"
                )
            _top_html += "</div>"
            st.markdown(_top_html, unsafe_allow_html=True)
        with pcol2:
            _worst_html = '<div class="kpi-card kpi-sub" style="text-align:left;">'
            _worst_html += '<div class="kpi-label">🔴 本日 Worst</div>'
            for p in _worst:
                _c = "#4ade80" if p["change_pct"] >= 0 else "#f87171"
                _worst_html += (
                    f'<div style="display:flex; justify-content:space-between;'
                    f' padding:3px 0; font-size:0.9rem;">'
                    f"<span>{p['symbol']}</span>"
                    f'<span style="color:{_c}; font-weight:600;">'
                    f"{p['change_pct']:+.2f}%</span></div>"
                )
            _worst_html += "</div>"
            st.markdown(_worst_html, unsafe_allow_html=True)


# =====================================================================
# データ事前読み込み（AI インサイト + タブ共通）
# Why: health_data は AI インサイトとヘルスタブの両方で使う。
#      キャッシュ済みなので2回呼んでも実コストは発生しないが、
#      1か所でロードして変数を共有することで意図が明確になる。
# =====================================================================
try:
    health_data = load_health_check()
except Exception:
    health_data = None

# =====================================================================
# AI インサイトパネル
# Why: ダッシュボードを開いた瞬間に「今注目すべきこと」を把握する
# How: LLM がスナップショット・ヘルスチェック等から 3-5 個のインサイトを生成
# =====================================================================
if insights_enabled and llm_enabled and llm_is_available():
    # ヘルスデータとセクター情報を事前取得（キャッシュ済みなのでコスト無し）
    _ins_health_results: list[dict] = []
    _ins_sell_alerts: list[dict] = []
    if health_data is not None:
        _ins_health_results = health_data.get("positions", [])
        _ins_sell_alerts = health_data.get("sell_alerts", [])

    _ins_sector_df = get_sector_breakdown(snapshot)
    _ins_sector_bd: dict[str, float] = {}
    if _ins_sector_df is not None and not _ins_sector_df.empty and "sector" in _ins_sector_df.columns:
        _ins_total = _ins_sector_df["evaluation_jpy"].sum()
        if _ins_total > 0:
            _ins_sector_bd = {
                row["sector"]: round(row["evaluation_jpy"] / _ins_total * 100, 1)
                for _, row in _ins_sector_df.iterrows()
            }

    _ins_positions = snapshot.get("positions", [])
    _ins_currency_bd: dict[str, int] = {}
    for _p in _ins_positions:
        _cur = _p.get("currency", "JPY")
        _ins_currency_bd[_cur] = _ins_currency_bd.get(_cur, 0) + 1

    _ins_structure = {
        "sector_breakdown": _ins_sector_bd,
        "currency_breakdown": _ins_currency_bd,
    }

    _insight_results = generate_insights(
        snapshot,
        _ins_structure,
        health_results=_ins_health_results,
        sell_alerts=_ins_sell_alerts,
    )
    if _insight_results:
        st.markdown("### 💡 AI インサイト")
        for _insight_text in _insight_results:
            st.info(_insight_text, icon=None)


# =====================================================================
# ヘッドライン・アラートストリップ
# Why: タブをまたぐ「要確認アラート」を一か所に集約し、ダッシュボードを
#      開いた瞬間に対処が必要な項目を把握できるようにする。
#      ヘルス & ドリフトいずれかにアラートがあるときタブバッジも更新して
#      視認性を高める（alert-oriented discoverability）。
#      全て既計算データを再利用するため追加の通信は発生しない。
# =====================================================================

# --- ウェイトドリフト（tab_holdings と同じ関数＝計算コストなし） ---
_hl_drift_alerts = compute_weight_drift(positions, total_value)
_hl_drift_count = len(_hl_drift_alerts)

# --- ヘルスチェック集計 ---
_hl_health_alert_count = 0  # early_warning + caution + exit
_hl_exit_count = 0  # exit のみ（撤退候補）
if health_data:
    for _hp in health_data.get("positions", []):
        if _hp.get("alert_level", "none") != "none":
            _hl_health_alert_count += 1
    _hl_exit_count = len(health_data.get("sell_alerts", []))

# --- ヘッドラインストリップ描画 ---
st.markdown(
    '<div id="headline-strip" role="region" aria-label="デイリーアラートサマリー"></div>', unsafe_allow_html=True
)
_hl_col1, _hl_col2, _hl_col3, _hl_col4 = st.columns(4)

with _hl_col1:
    if _hl_exit_count > 0:
        _hc_icon, _hc_color = "🚨", "#f87171"
        _hc_detail = f"うち撤退候補 {_hl_exit_count} 銘柄"
    elif _hl_health_alert_count > 0:
        _hc_icon, _hc_color = "⚠️", "#fbbf24"
        _hc_detail = "🏥 ヘルスタブを確認"
    else:
        _hc_icon, _hc_color = "✅", "#4ade80"
        _hc_detail = "全銘柄 問題なし"
    st.markdown(
        alert_badge_card(
            _hc_icon,
            "ヘルス注意",
            _hl_health_alert_count,
            detail=_hc_detail,
            color=_hc_color,
        ),
        unsafe_allow_html=True,
    )

with _hl_col2:
    if _hl_exit_count > 0:
        _ex_icon, _ex_color = "🚨", "#f87171"
        _ex_detail = "🏥 ヘルスタブを確認"
    else:
        _ex_icon, _ex_color = "✅", "#4ade80"
        _ex_detail = "撤退候補なし"
    st.markdown(
        alert_badge_card(
            _ex_icon,
            "撤退候補",
            _hl_exit_count,
            detail=_ex_detail,
            color=_ex_color,
        ),
        unsafe_allow_html=True,
    )

with _hl_col3:
    if _hl_drift_count > 0:
        _dr_icon, _dr_color = "⚖️", "#fbbf24"
        _dr_detail = "🏢 保有構成タブを確認"
    else:
        _dr_icon, _dr_color = "✅", "#4ade80"
        _dr_detail = "配分バランス良好"
    st.markdown(
        alert_badge_card(
            _dr_icon,
            "ドリフト",
            _hl_drift_count,
            detail=_dr_detail,
            color=_dr_color,
        ),
        unsafe_allow_html=True,
    )

with _hl_col4:
    # トータルアラート数（ヘルス + ドリフトを合算したスコア）
    _total_alerts = _hl_health_alert_count + _hl_drift_count
    if _hl_exit_count > 0:
        _ta_icon, _ta_color = "🔴", "#f87171"
        _ta_detail = "要対応あり"
    elif _total_alerts > 0:
        _ta_icon, _ta_color = "🟡", "#fbbf24"
        _ta_detail = "確認推奨"
    else:
        _ta_icon, _ta_color = "🟢", "#4ade80"
        _ta_detail = "今日は問題なし"
    st.markdown(
        alert_badge_card(
            _ta_icon,
            "総アラート数",
            _total_alerts,
            detail=_ta_detail,
            color=_ta_color,
        ),
        unsafe_allow_html=True,
    )

# --- ダイナミックタブラベル — アラートバッジ付き ---
# Why: タブバーに件数を表示することで、開かずに要確認タブを特定できる
#      (alert-oriented discoverability)。TOC サイドバーの JS は
#      textContent.includes(partial_label) でマッチするため既存の
#      ナビゲーションを壊さない。
_hl_health_tab_lbl = "🏥 ヘルス & ニュース"
if _hl_exit_count > 0:
    _hl_health_tab_lbl += f"  🚨{_hl_exit_count}"
elif _hl_health_alert_count > 0:
    _hl_health_tab_lbl += f"  ⚠️{_hl_health_alert_count}"

_hl_holdings_tab_lbl = "🏢 保有構成"
if _hl_drift_count > 0:
    _hl_holdings_tab_lbl += f"  🔺{_hl_drift_count}"


# =====================================================================
# タブ共通データ — キャッシュ済み関数の結果を 1 回だけ取得
# =====================================================================
# 経済ニュース（キーワードベース、LLM 分析は render_health_tab() 内で実行）
try:
    _pos_key = ",".join(sorted(p.get("symbol", "") for p in positions if p.get("sector") != "Cash"))
    _fx_for_news = snapshot.get("fx_rates", {})
    econ_news: list[dict] = load_economic_news(
        _pos_key,
        positions,
        _fx_for_news,
        llm_enabled=False,
        llm_model=llm_model,
        llm_cache_ttl=llm_cache_ttl_sec,
    )
except Exception:
    econ_news = []

# 月次売買アクティビティ
try:
    trade_act_df = load_trade_activity()
except Exception:
    import pandas as _pd

    trade_act_df = _pd.DataFrame()

# =====================================================================
# タブ構造 — セクションをタブで整理して情報過負荷を解消
# Why: 各タブの描画責務は components/tab_*.py モジュールに委譲する
# How: app.py はオーケストレーターとして st.tabs() とデータ受け渡しを担う
# =====================================================================
_tab_health, _tab_charts, _tab_holdings, _tab_monthly, _tab_watchlist, _tab_copilot, _tab_insights = st.tabs(
    [
        _hl_health_tab_lbl,
        "📊 チャート分析",
        _hl_holdings_tab_lbl,
        "📅 月次 & 売買",
        "👀 ウォッチリスト",
        "💬 Copilot",
        "📈 インサイト",
    ]
)

with _tab_health:
    render_health_tab(
        snapshot=snapshot,
        positions=positions,
        health_data=health_data,
        econ_news=econ_news,
        llm_enabled=llm_enabled,
        llm_auto_analyze=llm_auto_analyze,
        llm_model=llm_model,
        llm_cache_ttl_sec=llm_cache_ttl_sec,
    )

with _tab_charts:
    render_charts_tab(
        history_df=history_df,
        snapshot=snapshot,
        total_value=total_value,
        positions=positions,
        period=period,
        chart_style=chart_style,
        show_invested=show_invested,
        show_projection=show_projection,
        target_amount=target_amount,
        projection_years=projection_years,
        benchmark_symbol=benchmark_symbol,
        benchmark_label=benchmark_label,
        attribution_llm_enabled=attribution_llm_enabled,
        llm_enabled=llm_enabled,
        llm_model=llm_model,
        show_individual=show_individual,
    )

with _tab_holdings:
    render_holdings_tab(
        snapshot=snapshot,
        positions=positions,
        total_value=total_value,
        history_df=history_df,
    )

with _tab_monthly:
    render_monthly_tab(
        history_df=history_df,
        snapshot=snapshot,
        trade_act_df=trade_act_df,
        settings=_current_settings,
    )


with _tab_watchlist:
    # =====================================================================
    # ウォッチリスト
    # Why: 購入検討中の銘柄を追跡し、意思決定を支援する
    # How: render_watchlist() が CRUD + 価格表示 + AI 分析を一括描画
    # =====================================================================
    st.markdown('<div id="watchlist" role="region" aria-label="ウォッチリスト"></div>', unsafe_allow_html=True)
    _fx_rates = snapshot.get("fx_rates", {})
    render_watchlist(_current_settings, _fx_rates)


with _tab_copilot:
    render_copilot_tab(
        snapshot=snapshot,
        history_df=history_df,
        positions=positions,
        total_value=total_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        realized_pnl=realized_pnl,
        daily_change_jpy=_dc_jpy,
        daily_change_pct=_dc_pct,
        health_data=health_data,
        econ_news=econ_news,
        chat_model=chat_model,
    )


with _tab_insights:
    # =====================================================================
    # インサイトタブ — 蓄積データに基づく行動・タイミング分析
    # Why: 過去のトレード履歴から投資行動パターンと改善ポイントを可視化する
    # How: load_behavior_insight_cached / load_timing_insight_cached でデータを
    #      取得し render_insights_tab() に渡す。両ローダーとも Streamlit-free で
    #      ディスクキャッシュ優先のため追加のネットワーク呼び出しは発生しない。
    # =====================================================================
    _behavior_insight = load_behavior_insight_cached()
    _timing_insight = load_timing_insight_cached()

    # スタイルプロファイル + バイアス検出（positions/history_df が必要なため
    # キャッシュ外でインラインに計算する。内部の重い処理はすでに上記でキャッシュ済み）
    try:
        from components.dl_behavior import load_style_profile_insight

        _style_profile, _style_biases = load_style_profile_insight(
            positions=positions,
            behavior_insight=_behavior_insight,
            history_df=history_df if not history_df.empty else None,
            benchmark_symbol=benchmark_symbol,
            period=period,
        )
    except Exception as _sp_exc:
        import logging as _logging

        _logging.getLogger(__name__).warning("load_style_profile_insight failed: %s", _sp_exc)
        from src.core.behavior.models import StyleProfile

        _style_profile = StyleProfile.empty()
        _style_biases = []

    render_insights_tab(
        positions=positions,
        total_value=total_value,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        behavior_insight=_behavior_insight,
        timing_insight=_timing_insight,
        style_profile=_style_profile,
        style_biases=_style_biases,
    )


st.caption(
    "Data provided by Yahoo Finance via yfinance. Values are estimates and may differ from actual brokerage accounts."
)
