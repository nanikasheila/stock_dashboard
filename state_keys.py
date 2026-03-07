"""Streamlit ``session_state`` key constants for the portfolio dashboard.

Centralises every ``st.session_state`` key used by ``app.py`` so that:

- key names are easy to rename in one place instead of hunting string literals,
- future module splits can import this contract rather than hardcoding strings,
- accidental typos produce a clear ``AttributeError`` at import time instead of
  silently creating phantom keys at runtime.

Usage
-----
::

    from state_keys import SK

    st.session_state[SK.LAST_REFRESH] = time.strftime("%Y-%m-%d %H:%M:%S")
    ts = st.session_state.get(SK.LLM_ANALYZED_AT)
    if SK.SAVED_SETTINGS not in st.session_state:
        st.session_state[SK.SAVED_SETTINGS] = load_settings()
"""

from __future__ import annotations


class SK:
    """Namespace of ``str`` constants for ``st.session_state`` keys.

    All attributes are ``str`` class variables.  The class is intentionally
    **not** instantiated; import and use as ``SK.SOME_KEY`` directly.
    """

    # ------------------------------------------------------------------
    # Portfolio fingerprinting
    # Used to detect changes in portfolio.csv / trade-history files and
    # trigger a full cache-clear + reload automatically.
    # ------------------------------------------------------------------

    PORTFOLIO_FINGERPRINT: str = "_portfolio_fingerprint"
    """Internal hash of ``portfolio.csv`` mtime and trade-history file count.

    Written on every render cycle; compared to the previous value to decide
    whether caches need to be invalidated.
    """

    # ------------------------------------------------------------------
    # Refresh timestamps
    # ------------------------------------------------------------------

    LAST_REFRESH: str = "last_refresh"
    """Human-readable ``"%Y-%m-%d %H:%M:%S"`` string shown in the sidebar.

    Updated on: auto-refresh tick, manual 手動更新 click, and file-change
    detection.
    """

    LAST_MANUAL_REFRESH: str = "last_manual_refresh"
    """Timestamp of the most-recent 手動更新 (manual-refresh) button click.

    Only set when the user explicitly presses the button; ``None``/absent
    otherwise.
    """

    PREV_REFRESH_COUNT: str = "_prev_refresh_count"
    """``st_autorefresh`` counter value from the previous render cycle.

    Used to detect whether the auto-refresh timer has fired since last render
    and trigger a data reload when it has.
    """

    # ------------------------------------------------------------------
    # User settings persistence
    # ------------------------------------------------------------------

    SAVED_SETTINGS: str = "_saved_settings"
    """Settings ``dict`` loaded from / persisted to ``settings_store`` (JSON).

    Initialised once per session from disk; updated whenever the user changes
    a settings widget.
    """

    # ------------------------------------------------------------------
    # LLM news analysis
    # ------------------------------------------------------------------

    LLM_ANALYZED_AT: str = "_llm_analyzed_at"
    """``time.time()`` float recorded when the latest LLM news analysis ran.

    Used to display "analysed at HH:MM" captions and to skip redundant re-runs.
    """

    LLM_NEWS_RESULTS: str = "_llm_news_results"
    """``list[dict]`` of news items enriched with LLM classification data.

    Replaces the raw keyword-based news list when LLM analysis is available.
    """

    LLM_NEWS_SUMMARY: str = "_llm_news_summary"
    """LLM-generated plain-text summary ``dict`` for portfolio-relevant news.

    Schema: ``{"overview": str, "key_points": list[str], "portfolio_alert": str}``.
    """

    LLM_HC_SUMMARY: str = "_llm_hc_summary"
    """LLM health-check summary ``dict`` from the unified analysis run.

    Schema: ``{"stock_assessments": list[dict], "overview": str, ...}``.
    Written by both the auto-analysis path and the manual 「AI分析を実行」 button.
    """

    # ------------------------------------------------------------------
    # Health-check LLM data (read by the Copilot context builder)
    # ------------------------------------------------------------------

    HC_LLM_SUMMARY_DATA: str = "_hc_llm_summary_data"
    """Health-check LLM summary written by an external component.

    ``app.py`` only *reads* this key (for the Copilot chat context); it is
    written by a component not yet integrated into the main app flow.
    Schema: ``{"overview": str, "risk_warning": str, ...}``.
    """

    # ------------------------------------------------------------------
    # Copilot chat
    # ------------------------------------------------------------------

    COPILOT_CHAT_MESSAGES: str = "copilot_chat_messages"
    """``list[dict]`` of ``{"role": str, "content": str}`` chat messages.

    Initialised to ``[]`` on first render; appended to on every user / assistant
    turn; reset to ``[]`` by the 🗑️ クリア button.
    """

    COPILOT_SESSION_ID: str = "copilot_session_id"
    """Active GitHub Copilot CLI session ID string, or ``None``.

    ``None`` ⟹ next call starts a fresh CLI session.
    Non-``None`` ⟹ ``--resume <id>`` is passed so the CLI retains context.
    Reset to ``None`` by both the 🔄 新規セッション and 🗑️ クリア buttons.
    """

    # ------------------------------------------------------------------
    # AI レトロスペクティブ（インサイトタブ・任意実行）
    # ------------------------------------------------------------------

    RETRO_RESULT: str = "_retro_result"
    """AI retrospective response text (``str``), or ``None`` if not yet run.

    Session-scoped only; never persisted to disk.
    Written when the user clicks 「レトロスペクティブを実行」 and a response
    is received from Copilot.  Reset to ``None`` by the 🗑️ クリア button.
    """

    RETRO_ERROR: str = "_retro_error"
    """Error message (``str``) from the last retrospective run, or ``None``.

    Set when Copilot is unavailable or the call fails.  Reset alongside
    ``RETRO_RESULT`` when the user clears the result.
    """
