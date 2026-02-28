"""汎用 GitHub Copilot CLI クライアント.

``copilot`` CLI を任意のモジュールから呼び出すための共通インターフェース。
ニュース分析・銘柄分析・その他の LLM 用途で横断的に再利用する。

責務
----
- CLI の存在確認 (``is_available``)
- プロンプトを渡して応答テキストを取得 (``call``)
- 利用可能モデルの定義 (``AVAILABLE_MODELS``)
- 実行ログの記録と公開 (``get_execution_logs``)

キャッシュはドメイン固有の上位モジュール（例: ``llm_analyzer``）で行う。
本モジュールは純粋な CLI 実行層として薄く保つ。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =====================================================================
# モデル定義
# =====================================================================

DEFAULT_MODEL: str = "gpt-4.1"

# サイドバー / 設定画面に表示するモデル選択肢: (model_id, display_label)
# ``copilot --model`` の choices に準拠
AVAILABLE_MODELS: list[tuple[str, str]] = [
    # --- コスト効率（非 Premium / 低コスト） ---
    ("gpt-4.1", "GPT-4.1（バランス型・低コスト）"),
    ("gpt-5-mini", "GPT-5 Mini（高速・低コスト）"),
    ("claude-haiku-4.5", "Claude Haiku 4.5（高速・低コスト）"),
    # --- 標準 ---
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5.2", "GPT-5.2（高精度）"),
    ("claude-sonnet-4", "Claude Sonnet 4"),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5"),
    ("claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("gemini-3-pro-preview", "Gemini 3 Pro"),
    # --- Premium ---
    ("gpt-5.1-codex", "GPT-5.1 Codex ⚡Premium"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini ⚡Premium"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max ⚡Premium"),
    ("gpt-5.2-codex", "GPT-5.2 Codex ⚡Premium"),
    ("gpt-5.3-codex", "GPT-5.3 Codex ⚡Premium"),
    ("claude-opus-4.5", "Claude Opus 4.5 ⚡Premium"),
    ("claude-opus-4.6", "Claude Opus 4.6 ⚡Premium"),
    ("claude-opus-4.6-fast", "Claude Opus 4.6 Fast ⚡Premium"),
]

# =====================================================================
# 実行ログ
# =====================================================================

MAX_LOG_ENTRIES: int = 50


@dataclass
class CLICallLog:
    """1 回の CLI 呼び出しの記録."""

    timestamp: float
    model: str
    prompt_preview: str  # プロンプトの先頭部分
    success: bool
    duration_sec: float
    response_length: int  # 応答の文字数
    response_preview: str  # 応答の先頭部分
    error: str  # エラーメッセージ（成功時は空）
    source: str  # 呼び出し元の識別子（例: "news_analysis"）


_execution_logs: list[CLICallLog] = []
_log_lock = threading.Lock()


def get_execution_logs() -> list[CLICallLog]:
    """記録済みの実行ログを返す（新しい順）.

    Why: Streamlit はマルチセッションでプロセスを共有するため、
         ロックで読み取り一貫性を保証する。
    How: _log_lock で排他制御し、スナップショットを逆順で返す。
    """
    with _log_lock:
        return list(reversed(_execution_logs))


def clear_execution_logs() -> None:
    """実行ログをクリアする."""
    with _log_lock:
        _execution_logs.clear()


def _record_log(
    *,
    model: str,
    prompt: str,
    success: bool,
    duration: float,
    response: str | None,
    error: str,
    source: str,
) -> None:
    """実行ログを追記する.

    Why: 複数スレッド/セッションから同時書き込みされる可能性がある。
    How: _log_lock で排他制御し、append + 上限チェックを安全に行う。
    """
    entry = CLICallLog(
        timestamp=time.time(),
        model=model,
        prompt_preview=prompt[:150].replace("\n", " "),
        success=success,
        duration_sec=round(duration, 2),
        response_length=len(response) if response else 0,
        response_preview=(response[:200].replace("\n", " ") if response else ""),
        error=error[:300] if error else "",
        source=source,
    )
    with _log_lock:
        _execution_logs.append(entry)
        # 上限を超えたら古いものを破棄
        while len(_execution_logs) > MAX_LOG_ENTRIES:
            _execution_logs.pop(0)


# =====================================================================
# CLI 存在確認
# =====================================================================


def is_available() -> bool:
    """GitHub Copilot CLI (``copilot`` コマンド) が利用可能か判定する.

    ``shutil.which`` で見つからない場合（WinGet App Execution Alias 等）は
    ``copilot --version`` を実行して確認する。
    """
    if shutil.which("copilot") is not None:
        return True
    try:
        result = subprocess.run(
            ["copilot", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# =====================================================================
# CLI 呼び出し
# =====================================================================


def call(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 60,
    source: str = "",
    session_id: str | None = None,
    allow_urls: bool = False,
    allow_tools: bool = False,
) -> str | None:
    """Invoke Copilot CLI and return the text response.

    Why: Multiple callers (llm_analyzer, dashboard chat) need a single
         thin CLI execution layer with optional session-continuation and
         tool-access flags.
    How: Build the base command, then conditionally append --resume,
         --allow-all-urls, and --allow-all-tools before subprocess.run.
         All new parameters default to off so existing callers are
         unaffected (backward compatible).

    Parameters
    ----------
    prompt : str
        Prompt string sent to the LLM.
    model : str | None
        Model ID (``copilot --model``). Falls back to ``DEFAULT_MODEL``.
    timeout : int
        Subprocess timeout in seconds.
    source : str
        Caller identifier recorded in execution logs.
        e.g. ``"news_analysis"``, ``"stock_analysis"``
    session_id : str | None
        When provided, passes ``--resume <session_id>`` so the CLI
        continues an existing conversation context.
    allow_urls : bool
        When True, appends ``--allow-all-urls`` to let Copilot fetch URLs.
    allow_tools : bool
        When True, appends ``--allow-all-tools`` to grant full tool access.

    Returns
    -------
    str | None
        stdout text from CLI, or ``None`` on any failure.
    """
    mdl = model or DEFAULT_MODEL

    cmd = [
        "copilot",
        "-p",
        prompt,
        "-s",
        "--model",
        mdl,
    ]

    if session_id is not None:
        cmd.extend(["--resume", session_id])
    if allow_urls:
        cmd.append("--allow-all-urls")
    if allow_tools:
        cmd.append("--allow-all-tools")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
        duration = time.time() - t0

        if result.returncode != 0:
            stderr = result.stderr.strip()
            err_msg = f"rc={result.returncode}: {stderr[:200]}"
            logger.warning("[copilot_client] CLI error: %s", err_msg)
            _record_log(
                model=mdl,
                prompt=prompt,
                success=False,
                duration=duration,
                response=None,
                error=err_msg,
                source=source,
            )
            return None

        output = result.stdout.strip()
        logger.info(
            "[copilot_client] success model=%s duration=%.1fs len=%d source=%s",
            mdl,
            duration,
            len(output),
            source,
        )
        _record_log(
            model=mdl,
            prompt=prompt,
            success=True,
            duration=duration,
            response=output,
            error="",
            source=source,
        )
        return output

    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        err_msg = f"timeout ({timeout}s)"
        logger.warning("[copilot_client] %s", err_msg)
        _record_log(
            model=mdl,
            prompt=prompt,
            success=False,
            duration=duration,
            response=None,
            error=err_msg,
            source=source,
        )
        return None

    except FileNotFoundError:
        duration = time.time() - t0
        err_msg = "copilot command not found"
        logger.warning("[copilot_client] %s", err_msg)
        _record_log(
            model=mdl,
            prompt=prompt,
            success=False,
            duration=duration,
            response=None,
            error=err_msg,
            source=source,
        )
        return None

    except Exception as exc:
        duration = time.time() - t0
        err_msg = f"unexpected: {exc}"
        logger.warning("[copilot_client] %s", err_msg)
        _record_log(
            model=mdl,
            prompt=prompt,
            success=False,
            duration=duration,
            response=None,
            error=err_msg,
            source=source,
        )
        return None


# =====================================================================
# Chat-oriented call with session tracking
# =====================================================================


@dataclass
class ChatCallResult:
    """Result of a chat-oriented CLI call that carries session context.

    Why: The dashboard chat needs to propagate a session_id across turns
         so the CLI can maintain conversation context via --resume.
         Returning both fields together avoids a separate session-lookup.
    """

    response: str | None
    session_id: str | None


def call_with_session(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 120,
    source: str = "dashboard_chat",
    session_id: str | None = None,
    allow_urls: bool = True,
    allow_tools: bool = True,
) -> ChatCallResult:
    """Call Copilot CLI with session continuation and return result + session_id.

    Why: The multi-turn dashboard chat requires a stable session identifier
         so the CLI can recall earlier context without re-sending the full
         history every turn (reduces token cost and latency).
    How: If no session_id is provided, a new UUID4 is generated and passed
         as --resume so the CLI anchors a new session to that ID.  The
         generated or provided session_id is always echoed back in the
         returned ChatCallResult, letting the caller persist it in
         st.session_state for subsequent turns.

    Parameters
    ----------
    prompt : str
        Prompt string sent to the LLM.
    model : str | None
        Model ID. Falls back to ``DEFAULT_MODEL``.
    timeout : int
        Subprocess timeout in seconds.
    source : str
        Caller identifier recorded in execution logs.
    session_id : str | None
        Existing session ID to resume.  When None, a new UUID is created.
    allow_urls : bool
        Passed through to ``call()``; defaults True for chat use-case.
    allow_tools : bool
        Passed through to ``call()``; defaults True for chat use-case.

    Returns
    -------
    ChatCallResult
        ``.response`` is the CLI output (or None on failure).
        ``.session_id`` is the session ID that was used (new or provided).
    """
    effective_session_id = session_id if session_id is not None else str(uuid.uuid4())

    logger.info(
        "[copilot_client] call_with_session source=%s session_id=%s",
        source,
        effective_session_id,
    )

    response = call(
        prompt,
        model=model,
        timeout=timeout,
        source=source,
        session_id=effective_session_id,
        allow_urls=allow_urls,
        allow_tools=allow_tools,
    )

    return ChatCallResult(response=response, session_id=effective_session_id)
