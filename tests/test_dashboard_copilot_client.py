"""Tests for copilot_client — SDK-based Copilot client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from components.copilot_client import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ChatCallResult,
    call,
    call_with_session,
    clear_execution_logs,
    get_available_models,
    get_execution_logs,
    is_available,
)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------
class TestIsAvailable:
    def test_found_by_which(self):
        with patch("components.copilot_client.shutil.which", return_value="/usr/bin/copilot"):
            assert is_available() is True

    def test_found_by_subprocess(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            assert is_available() is True

    def test_not_found(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            assert is_available() is False


# ---------------------------------------------------------------------------
# AVAILABLE_MODELS (fallback list)
# ---------------------------------------------------------------------------
class TestModels:
    def test_has_models(self):
        """Fallback list has some models."""
        assert len(AVAILABLE_MODELS) >= 5

    def test_models_are_tuples(self):
        for m in AVAILABLE_MODELS:
            assert isinstance(m, tuple) and len(m) == 2

    def test_default_model_in_list(self):
        ids = [m[0] for m in AVAILABLE_MODELS]
        assert DEFAULT_MODEL in ids


# ---------------------------------------------------------------------------
# get_available_models()
# ---------------------------------------------------------------------------
class TestGetAvailableModels:
    def test_returns_fallback_when_unavailable(self):
        """When CLI is not available, fallback list is returned."""
        with patch("components.copilot_client.is_available", return_value=False):
            models = get_available_models()
        assert len(models) >= 5
        assert all(isinstance(m, tuple) and len(m) == 2 for m in models)

    def test_returns_dynamic_models_on_success(self):
        """When SDK succeeds, dynamic models are returned."""
        from types import SimpleNamespace

        import components.copilot_client as _cc

        saved = list(_cc.AVAILABLE_MODELS)
        mock_model1 = SimpleNamespace(id="gpt-4.1", name="GPT-4.1")
        mock_model2 = SimpleNamespace(id="claude-sonnet-4", name="Claude Sonnet 4")
        try:
            with (
                patch("components.copilot_client.is_available", return_value=True),
                patch("components.copilot_client._run_async", return_value=[mock_model1, mock_model2]),
                patch("components.copilot_client._models_cache", None),
                patch("components.copilot_client._models_cache_timestamp", 0.0),
            ):
                models = get_available_models()
            assert ("gpt-4.1", "GPT-4.1") in models
            assert ("claude-sonnet-4", "Claude Sonnet 4") in models
        finally:
            _cc.AVAILABLE_MODELS.clear()
            _cc.AVAILABLE_MODELS.extend(saved)

    def test_returns_fallback_on_sdk_error(self):
        """When SDK raises, fallback list is returned."""
        with (
            patch("components.copilot_client.is_available", return_value=True),
            patch("components.copilot_client._run_async", side_effect=RuntimeError("SDK error")),
            patch("components.copilot_client._models_cache", None),
            patch("components.copilot_client._models_cache_timestamp", 0.0),
        ):
            models = get_available_models()
        assert len(models) >= 5


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------
class TestCall:
    def test_success(self):
        """call() returns stripped response text on success."""
        with patch("components.copilot_client._run_async", return_value="hello world\n"):
            result = call("test prompt", source="test")
        assert result == "hello world"

    def test_returns_none_on_sdk_none(self):
        """call() returns None when SDK returns None."""
        with patch("components.copilot_client._run_async", return_value=None):
            result = call("test prompt")
        assert result is None

    def test_returns_none_on_timeout(self):
        """call() returns None on timeout."""
        with patch("components.copilot_client._run_async", side_effect=TimeoutError("timeout")):
            result = call("test prompt", timeout=60)
        assert result is None

    def test_returns_none_on_exception(self):
        """call() returns None on unexpected exception."""
        with patch("components.copilot_client._run_async", side_effect=RuntimeError("boom")):
            result = call("test prompt")
        assert result is None

    def test_uses_default_model(self):
        """call() passes DEFAULT_MODEL when model is not specified."""
        with patch("components.copilot_client._run_async", return_value="ok"):
            call("test prompt")
        # The coroutine passed to _run_async is _async_call with model=DEFAULT_MODEL
        # We verify via execution logs
        logs = get_execution_logs()
        assert any(log.model == DEFAULT_MODEL for log in logs)

    def test_uses_specified_model(self):
        """call() passes the specified model."""
        clear_execution_logs()
        with patch("components.copilot_client._run_async", return_value="ok"):
            call("test prompt", model="claude-sonnet-4.6")
        logs = get_execution_logs()
        assert logs[0].model == "claude-sonnet-4.6"


# ---------------------------------------------------------------------------
# Execution logs
# ---------------------------------------------------------------------------
class TestExecutionLogs:
    def setup_method(self):
        clear_execution_logs()

    def test_empty_initially(self):
        assert get_execution_logs() == []

    def test_records_success(self):
        with patch("components.copilot_client._run_async", return_value="response text"):
            call("test prompt", source="test_source")
        logs = get_execution_logs()
        assert len(logs) == 1
        log = logs[0]
        assert log.success is True
        assert log.source == "test_source"
        assert log.response_length == len("response text")
        assert log.error == ""

    def test_records_failure(self):
        with patch("components.copilot_client._run_async", return_value=None):
            call("test prompt", source="fail_test")
        logs = get_execution_logs()
        assert len(logs) == 1
        assert logs[0].success is False

    def test_records_timeout(self):
        with patch("components.copilot_client._run_async", side_effect=TimeoutError("timeout")):
            call("test prompt", timeout=30)
        logs = get_execution_logs()
        assert len(logs) == 1
        assert logs[0].success is False
        assert "timeout" in logs[0].error

    def test_newest_first(self):
        """get_execution_logs は新しい順で返す."""
        with patch("components.copilot_client._run_async", return_value="r1"):
            call("first", source="first")
            call("second", source="second")
        logs = get_execution_logs()
        assert logs[0].source == "second"
        assert logs[1].source == "first"

    def test_clear(self):
        with patch("components.copilot_client._run_async", return_value="ok"):
            call("test")
        assert len(get_execution_logs()) == 1
        clear_execution_logs()
        assert len(get_execution_logs()) == 0

    def test_prompt_preview_truncated(self):
        long_prompt = "x" * 500
        with patch("components.copilot_client._run_async", return_value="ok"):
            call(long_prompt)
        logs = get_execution_logs()
        assert len(logs[0].prompt_preview) <= 150

    def test_log_has_duration(self):
        with patch("components.copilot_client._run_async", return_value="ok"):
            call("test")
        logs = get_execution_logs()
        assert logs[0].duration_sec >= 0


# ---------------------------------------------------------------------------
# call_with_session() and ChatCallResult
# ---------------------------------------------------------------------------
class TestCallWithSession:
    def setup_method(self):
        clear_execution_logs()

    def test_chat_call_result_dataclass(self):
        """ChatCallResult stores response and session_id."""
        result = ChatCallResult(response="hello", session_id="abc-123")
        assert result.response == "hello"
        assert result.session_id == "abc-123"

    def test_call_with_session_creates_new_id(self):
        """When session_id is None, a new session_id is generated and returned."""
        generated_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        mock_result = ChatCallResult(response="answer", session_id=generated_id)
        with patch("components.copilot_client._run_async", return_value=mock_result):
            result = call_with_session("prompt", session_id=None)
        assert result.session_id is not None
        assert result.response == "answer"

    def test_call_with_session_preserves_id(self):
        """When session_id is provided, the same ID is echoed in the result."""
        mock_result = ChatCallResult(response="answer", session_id="existing-session-id")
        with patch("components.copilot_client._run_async", return_value=mock_result):
            result = call_with_session("prompt", session_id="existing-session-id")
        assert result.session_id == "existing-session-id"
        assert result.response == "answer"

    def test_call_with_session_records_log(self):
        """call_with_session records execution log."""
        mock_result = ChatCallResult(response="answer", session_id="s1")
        with patch("components.copilot_client._run_async", return_value=mock_result):
            call_with_session("test", source="chat_test")
        logs = get_execution_logs()
        assert len(logs) == 1
        assert logs[0].source == "chat_test"
        assert logs[0].success is True
