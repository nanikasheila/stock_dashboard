"""Tests for copilot_client — generic Copilot CLI client."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from components.copilot_client import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    call,
    clear_execution_logs,
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
# AVAILABLE_MODELS
# ---------------------------------------------------------------------------
class TestModels:
    def test_has_many_models(self):
        assert len(AVAILABLE_MODELS) >= 10

    def test_models_are_tuples(self):
        for m in AVAILABLE_MODELS:
            assert isinstance(m, tuple) and len(m) == 2

    def test_default_model_in_list(self):
        ids = [m[0] for m in AVAILABLE_MODELS]
        assert DEFAULT_MODEL in ids

    def test_has_premium_models(self):
        labels = [m[1] for m in AVAILABLE_MODELS]
        assert any("Premium" in label for label in labels)

    def test_has_low_cost_models(self):
        labels = [m[1] for m in AVAILABLE_MODELS]
        assert any("低コスト" in label for label in labels)


# ---------------------------------------------------------------------------
# call()
# ---------------------------------------------------------------------------
class TestCall:
    def test_success(self):
        mock_result = MagicMock(returncode=0, stdout="hello world\n", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            result = call("test prompt", source="test")
        assert result == "hello world"

    def test_returns_none_on_error(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="error msg")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            result = call("test prompt")
        assert result is None

    def test_returns_none_on_timeout(self):
        import subprocess

        with patch(
            "components.copilot_client.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="copilot", timeout=60)
        ):
            result = call("test prompt", timeout=60)
        assert result is None

    def test_returns_none_when_not_found(self):
        with patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError):
            result = call("test prompt")
        assert result is None

    def test_uses_default_model(self):
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result) as mock_run:
            call("test prompt")
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == DEFAULT_MODEL

    def test_uses_specified_model(self):
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result) as mock_run:
            call("test prompt", model="claude-sonnet-4.6")
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4.6"


# ---------------------------------------------------------------------------
# Execution logs
# ---------------------------------------------------------------------------
class TestExecutionLogs:
    def setup_method(self):
        clear_execution_logs()

    def test_empty_initially(self):
        assert get_execution_logs() == []

    def test_records_success(self):
        mock_result = MagicMock(returncode=0, stdout="response text", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call("test prompt", source="test_source")
        logs = get_execution_logs()
        assert len(logs) == 1
        log = logs[0]
        assert log.success is True
        assert log.source == "test_source"
        assert log.response_length == len("response text")
        assert log.error == ""

    def test_records_failure(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="some error")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call("test prompt", source="fail_test")
        logs = get_execution_logs()
        assert len(logs) == 1
        assert logs[0].success is False
        assert "some error" in logs[0].error

    def test_records_timeout(self):
        import subprocess

        with patch(
            "components.copilot_client.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="copilot", timeout=30)
        ):
            call("test prompt", timeout=30)
        logs = get_execution_logs()
        assert len(logs) == 1
        assert logs[0].success is False
        assert "timeout" in logs[0].error

    def test_newest_first(self):
        """get_execution_logs は新しい順で返す."""
        mock_result = MagicMock(returncode=0, stdout="r1", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call("first", source="first")
            call("second", source="second")
        logs = get_execution_logs()
        assert logs[0].source == "second"
        assert logs[1].source == "first"

    def test_clear(self):
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call("test")
        assert len(get_execution_logs()) == 1
        clear_execution_logs()
        assert len(get_execution_logs()) == 0

    def test_prompt_preview_truncated(self):
        long_prompt = "x" * 500
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call(long_prompt)
        logs = get_execution_logs()
        assert len(logs[0].prompt_preview) <= 150

    def test_log_has_duration(self):
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("components.copilot_client.subprocess.run", return_value=mock_result):
            call("test")
        logs = get_execution_logs()
        assert logs[0].duration_sec >= 0
