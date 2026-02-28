"""汎用 GitHub Copilot SDK クライアント.

GitHub Copilot SDK (``github-copilot-sdk``) を通じて Copilot CLI サーバーと
JSON-RPC で通信する共通インターフェース。
ニュース分析・銘柄分析・ダッシュボードチャット等で横断的に再利用する。

責務
----
- SDK / CLI の利用可能判定 (``is_available``)
- プロンプトを渡して応答テキストを取得 (``call``)
- 利用可能モデルの動的取得 (``get_available_models``)
- チャットセッション管理 (``call_with_session``)
- 実行ログの記録と公開 (``get_execution_logs``)

Why: subprocess.run で CLI を直接呼び出す方式から、SDK の JSON-RPC 通信に移行する。
     SDK はプロセスライフサイクル管理・セッション管理・ツール実行を内蔵しており、
     再接続やエラー回復が堅牢。
How: CopilotClient シングルトンを専用バックグラウンドスレッドの asyncio ループで管理し、
     同期呼び出し元（Streamlit）からは run_coroutine_threadsafe でブリッジする。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from copilot import CopilotClient, ModelInfo, PermissionHandler
from copilot.generated.session_events import SessionEventType

logger = logging.getLogger(__name__)

# =====================================================================
# モデル定義
# =====================================================================

DEFAULT_MODEL: str = "gpt-4.1"

# SDK 未初期化時のフォールバック用ハードコードリスト
_FALLBACK_MODELS: list[tuple[str, str]] = [
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-5-mini", "GPT-5 Mini"),
    ("claude-haiku-4.5", "Claude Haiku 4.5"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5.2", "GPT-5.2"),
    ("claude-sonnet-4", "Claude Sonnet 4"),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5"),
    ("claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("gemini-3-pro-preview", "Gemini 3 Pro"),
]

# re-export: 後方互換のためフォールバックリストを公開
AVAILABLE_MODELS: list[tuple[str, str]] = list(_FALLBACK_MODELS)

# =====================================================================
# 非同期イベントループ管理
# =====================================================================

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """バックグラウンドスレッドで asyncio ループを起動・再利用する.

    Why: Copilot SDK は async 専用。Streamlit は同期的に動作するため、
         専用スレッドのイベントループ経由でブリッジする。
    How: daemon スレッドで run_forever() を実行。
         スレッドセーフにシングルトンとして初期化する。
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True, name="copilot-sdk-loop")
        _loop_thread.start()
        return _loop


def _run_async(coro: Any, *, timeout: float | None = None) -> Any:
    """async コルーチンを同期的に実行して結果を返す.

    Why: 全公開 API は同期シグネチャを維持する必要がある。
    How: バックグラウンドループに submit し、Future.result() でブロッキング待ち。
    """
    loop = _ensure_event_loop()
    future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


# =====================================================================
# SDK クライアントシングルトン
# =====================================================================

_client: CopilotClient | None = None
_client_lock = threading.Lock()


async def _get_client() -> CopilotClient:
    """CopilotClient を遅延初期化して返す.

    Why: 初回呼び出し時にのみ CLI サーバーを起動する（起動コスト回避）。
    How: ダブルチェックロッキングで排他的に初期化する。
         auto_start=True なので create_session 時に自動接続される。
    """
    global _client
    if _client is not None:
        return _client
    # NOTE: _client_lock は同期ロックだが、この関数は _run_async 経由で
    # バックグラウンドスレッドから呼ばれる。初期化は同期ロック内で行い、
    # start() は非同期で実行する。
    with _client_lock:
        if _client is not None:
            return _client
        _client = CopilotClient(
            {
                "log_level": "warning",
                "auto_start": True,
                "auto_restart": True,
            }
        )
        await _client.start()
        logger.info("[copilot_client] SDK client started")
    return _client


# =====================================================================
# 実行ログ
# =====================================================================

MAX_LOG_ENTRIES: int = 50


@dataclass
class CLICallLog:
    """1 回の SDK 呼び出しの記録."""

    timestamp: float
    model: str
    prompt_preview: str
    success: bool
    duration_sec: float
    response_length: int
    response_preview: str
    error: str
    source: str


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
        while len(_execution_logs) > MAX_LOG_ENTRIES:
            _execution_logs.pop(0)


# =====================================================================
# CLI / SDK 存在確認
# =====================================================================


def is_available() -> bool:
    """GitHub Copilot SDK が利用可能か判定する.

    Why: SDK は copilot CLI のインストールを前提とする。
         CLI が存在しない環境では graceful に無効化する。
    How: まず SDK パッケージの import を確認（モジュールレベルで済み）、
         次に CLI バイナリの存在を確認する。
    """
    try:
        # SDK パッケージはモジュールレベルで import 済み
        # CLI バイナリの存在確認
        if shutil.which("copilot") is not None:
            return True
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
# モデル一覧の動的取得
# =====================================================================

_models_cache: list[tuple[str, str]] | None = None
_models_cache_lock = threading.Lock()
_models_cache_timestamp: float = 0.0
_MODELS_CACHE_TTL: float = 300.0  # 5分


def get_available_models() -> list[tuple[str, str]]:
    """利用可能なモデルの一覧を ``(model_id, display_name)`` で返す.

    Why: ハードコードリストではなく SDK から動的に取得することで、
         CLI アップデートで追加されたモデルも自動的に反映される。
    How: client.list_models() を呼び出し、結果を 5 分間キャッシュする。
         SDK 未初期化やエラー時はフォールバックリストを返す。
    """
    global _models_cache, _models_cache_timestamp
    with _models_cache_lock:
        now = time.time()
        if _models_cache is not None and (now - _models_cache_timestamp) < _MODELS_CACHE_TTL:
            return list(_models_cache)

    if not is_available():
        return list(_FALLBACK_MODELS)

    try:
        models: list[ModelInfo] = _run_async(_fetch_models(), timeout=30)
        result = [(m.id, m.name) for m in models]
        with _models_cache_lock:
            _models_cache = result
            _models_cache_timestamp = time.time()
        # グローバルの AVAILABLE_MODELS も更新（後方互換）
        AVAILABLE_MODELS.clear()
        AVAILABLE_MODELS.extend(result)
        return list(result)
    except Exception as exc:
        logger.warning("[copilot_client] list_models failed, using fallback: %s", exc)
        return list(_FALLBACK_MODELS)


async def _fetch_models() -> list[ModelInfo]:
    """SDK からモデル一覧を取得する（内部 async ヘルパー）."""
    client = await _get_client()
    return await client.list_models()


# =====================================================================
# SDK 呼び出し（ワンショットセッション）
# =====================================================================


async def _async_call(
    prompt: str,
    *,
    model: str,
    timeout: int,
    source: str,
) -> str | None:
    """SDK セッションを作成してプロンプトを送信し、最終応答を返す.

    Why: 分析用途では1回のプロンプト→レスポンスで完結する。
    How: create_session → send → assistant.message イベントで応答取得 →
         session.idle で完了を検知 → destroy。
         infinite_sessions を無効化してオーバーヘッドを削減。
    """
    client = await _get_client()

    session = await client.create_session(
        {
            "model": model,
            "on_permission_request": PermissionHandler.approve_all,
            "infinite_sessions": {"enabled": False},
        }
    )

    try:
        result_text: str | None = None
        done = asyncio.Event()
        error_msg: str | None = None

        def on_event(event: Any) -> None:
            nonlocal result_text, error_msg
            if event.type == SessionEventType.ASSISTANT_MESSAGE:
                result_text = event.data.content
            elif event.type == SessionEventType.SESSION_IDLE:
                done.set()
            elif event.type == SessionEventType.SESSION_ERROR:
                error_msg = getattr(event.data, "message", None) or "unknown SDK error"
                done.set()

        session.on(on_event)
        await session.send({"prompt": prompt})
        await asyncio.wait_for(done.wait(), timeout=timeout)

        if error_msg:
            logger.warning("[copilot_client] SDK session error: %s", error_msg)
            return None

        return result_text
    finally:
        try:
            await session.destroy()
        except Exception:
            pass


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
    """Copilot SDK 経由でプロンプトを送信し応答テキストを返す.

    Why: llm_analyzer 等の複数モジュールが共通で使うワンショット呼び出し。
    How: バックグラウンドループで _async_call を実行し、結果を同期で返す。
         session_id / allow_urls / allow_tools は後方互換のために残すが、
         SDK ではセッション作成時に設定するため現在は未使用パラメータ。

    Parameters
    ----------
    prompt : str
        LLM に送信するプロンプト文字列。
    model : str | None
        モデル ID。省略時は ``DEFAULT_MODEL``。
    timeout : int
        応答待ちタイムアウト秒数。
    source : str
        実行ログに記録する呼び出し元識別子。
    session_id : str | None
        後方互換パラメータ（SDK では create_session で管理）。
    allow_urls : bool
        後方互換パラメータ。
    allow_tools : bool
        後方互換パラメータ。

    Returns
    -------
    str | None
        LLM の応答テキスト。失敗時は ``None``。
    """
    mdl = model or DEFAULT_MODEL
    t0 = time.time()

    try:
        output = _run_async(
            _async_call(prompt, model=mdl, timeout=timeout, source=source),
            timeout=timeout + 30,  # async ブリッジのマージン
        )
        duration = time.time() - t0

        if output is None:
            _record_log(
                model=mdl,
                prompt=prompt,
                success=False,
                duration=duration,
                response=None,
                error="no response from SDK",
                source=source,
            )
            return None

        output = output.strip()
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

    except TimeoutError:
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
    """Result of a chat-oriented SDK call that carries session context.

    Why: The dashboard chat needs to propagate a session_id across turns
         so the SDK can maintain conversation context via resume_session.
         Returning both fields together avoids a separate session-lookup.
    """

    response: str | None
    session_id: str | None


async def _async_call_with_session(
    prompt: str,
    *,
    model: str,
    timeout: int,
    source: str,
    session_id: str | None,
) -> ChatCallResult:
    """SDK セッションを使ったチャット呼び出し（内部 async）.

    Why: マルチターンのチャットではセッションを再利用することで、
         会話コンテキストを維持しつつトークンコストを削減する。
    How: session_id が None なら create_session で新規セッション、
         あれば resume_session で既存セッションを再開する。
         チャットセッションは infinite_sessions を有効にして
         長い会話でもコンテキストを自動圧縮する。
    """
    client = await _get_client()

    effective_session_id = session_id if session_id is not None else str(uuid.uuid4())

    session_config: dict[str, Any] = {
        "model": model,
        "on_permission_request": PermissionHandler.approve_all,
    }

    try:
        if session_id is not None:
            session = await client.resume_session(effective_session_id, session_config)
        else:
            session_config["session_id"] = effective_session_id
            session = await client.create_session(session_config)
    except Exception as exc:
        logger.warning("[copilot_client] session create/resume failed: %s", exc)
        return ChatCallResult(response=None, session_id=effective_session_id)

    try:
        result_text: str | None = None
        done = asyncio.Event()

        def on_event(event: Any) -> None:
            nonlocal result_text
            if event.type == SessionEventType.ASSISTANT_MESSAGE:
                result_text = event.data.content
            elif event.type in (SessionEventType.SESSION_IDLE, SessionEventType.SESSION_ERROR):
                done.set()

        session.on(on_event)
        await session.send({"prompt": prompt})
        await asyncio.wait_for(done.wait(), timeout=timeout)

        return ChatCallResult(response=result_text, session_id=effective_session_id)
    except TimeoutError:
        logger.warning("[copilot_client] chat session timeout (%ds)", timeout)
        return ChatCallResult(response=None, session_id=effective_session_id)
    except Exception as exc:
        logger.warning("[copilot_client] chat session error: %s", exc)
        return ChatCallResult(response=None, session_id=effective_session_id)
    # NOTE: チャットセッションは destroy しない（次のターンで再利用するため）


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
    """Copilot SDK でチャットセッション付き呼び出しを行い結果を返す.

    Why: マルチターンのダッシュボードチャットでセッション ID を伝搬し、
         会話コンテキストを維持する。
    How: SDK の resume_session / create_session でセッションを管理し、
         send() で応答を取得する。allow_urls / allow_tools は
         後方互換のために残すが SDK では設定不要。

    Parameters
    ----------
    prompt : str
        LLM に送信するプロンプト文字列。
    model : str | None
        モデル ID。省略時は ``DEFAULT_MODEL``。
    timeout : int
        応答待ちタイムアウト秒数。
    source : str
        実行ログに記録する呼び出し元識別子。
    session_id : str | None
        既存セッション ID。None の場合は新規 UUID を生成。
    allow_urls : bool
        後方互換パラメータ。
    allow_tools : bool
        後方互換パラメータ。

    Returns
    -------
    ChatCallResult
        ``.response`` は LLM 応答（失敗時 None）。
        ``.session_id`` は使用されたセッション ID。
    """
    mdl = model or DEFAULT_MODEL
    effective_session_id = session_id if session_id is not None else str(uuid.uuid4())

    logger.info(
        "[copilot_client] call_with_session source=%s session_id=%s",
        source,
        effective_session_id,
    )

    t0 = time.time()
    try:
        result = _run_async(
            _async_call_with_session(
                prompt,
                model=mdl,
                timeout=timeout,
                source=source,
                session_id=session_id,
            ),
            timeout=timeout + 30,
        )
        duration = time.time() - t0
        _record_log(
            model=mdl,
            prompt=prompt,
            success=result.response is not None,
            duration=duration,
            response=result.response,
            error="" if result.response else "no response",
            source=source,
        )
        return result

    except Exception as exc:
        duration = time.time() - t0
        err_msg = f"unexpected: {exc}"
        logger.warning("[copilot_client] call_with_session error: %s", err_msg)
        _record_log(
            model=mdl,
            prompt=prompt,
            success=False,
            duration=duration,
            response=None,
            error=err_msg,
            source=source,
        )
        return ChatCallResult(response=None, session_id=effective_session_id)
