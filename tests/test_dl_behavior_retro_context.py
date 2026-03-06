"""Trade memo context aggregation tests for the AI retrospective helper.

Why: Memo-aware retrospectives should use only anonymized aggregates, so the
     loader must summarize coverage and themes without leaking raw memo text.
How: Patch ``history_store.load_history`` with representative trade records and
     verify that ``load_trade_memo_context`` returns only counts and theme names.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import components.dl_behavior as _dlb


class TestLoadTradeMemoContext:
    """``load_trade_memo_context`` の要約ロジックを検証する."""

    def test_returns_empty_summary_when_trade_history_is_empty(self) -> None:
        """履歴が空ならゼロ件サマリーを返すこと."""
        with patch("src.data.history_store.load_history", return_value=[]):
            summary = _dlb.load_trade_memo_context()

        assert summary["reviewed_trade_count"] == 0
        assert summary["memo_trade_count"] == 0
        assert summary["memo_coverage_pct"] == 0.0
        assert summary["top_themes"] == []

    def test_aggregates_recent_memo_themes(self) -> None:
        """メモ本文から匿名テーマの件数が集計されること."""
        trades = [
            {"memo": "押し目なので買い増し"},
            {"memo": "利益確定を一部実施"},
            {"memo": "押し目からのリバランス"},
            {"memo": ""},
        ]
        with patch("src.data.history_store.load_history", return_value=trades):
            summary = _dlb.load_trade_memo_context(limit=3)

        top_themes = {item["theme"]: item["count"] for item in summary["top_themes"]}
        assert summary["reviewed_trade_count"] == 3
        assert summary["memo_trade_count"] == 3
        assert summary["memo_coverage_pct"] == 100.0
        assert top_themes["押し目買い"] == 2
        assert top_themes["利益確定"] == 1
        assert top_themes["リバランス"] == 1

    def test_blank_memos_reduce_coverage_but_do_not_create_themes(self) -> None:
        """空メモはカバレッジだけに影響し、テーマ集計には含まれないこと."""
        trades = [
            {"memo": "長期保有の継続"},
            {"memo": "   "},
            {"memo": ""},
            {"memo": None},
        ]
        with patch("src.data.history_store.load_history", return_value=trades):
            summary = _dlb.load_trade_memo_context(limit=4)

        assert summary["reviewed_trade_count"] == 4
        assert summary["memo_trade_count"] == 1
        assert summary["memo_coverage_pct"] == 25.0
        assert summary["top_themes"] == [{"theme": "長期保有", "count": 1}]
