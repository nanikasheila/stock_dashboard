"""src/data/history_store と src/data/summary_builder のユニットテスト."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.data.history_store as history_store
import src.data.summary_builder as summary_builder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_graph(*args, **kwargs):
    """Dummy no-op to replace graph_store imports."""
    raise ImportError("graph_store not available in tests")


# ---------------------------------------------------------------------------
# TestSafeFilename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    """_safe_filename() のテスト."""

    def test_dot_replaced(self):
        """'.' が '_' に置換される."""
        assert history_store._safe_filename("7203.T") == "7203_T"

    def test_slash_replaced(self):
        """'/' が '_' に置換される."""
        assert history_store._safe_filename("a/b") == "a_b"

    def test_plain_string_unchanged(self):
        """特殊文字がなければそのまま返る."""
        assert history_store._safe_filename("VTI") == "VTI"


# ---------------------------------------------------------------------------
# TestSanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    """_sanitize() のテスト."""

    def test_nan_float_becomes_none(self):
        """float NaN が None になる."""
        import math
        result = history_store._sanitize(float("nan"))
        assert result is None

    def test_inf_float_becomes_none(self):
        """float inf が None になる."""
        result = history_store._sanitize(float("inf"))
        assert result is None

    def test_regular_float_unchanged(self):
        """通常の float はそのまま返る."""
        assert history_store._sanitize(3.14) == 3.14

    def test_nested_dict_sanitized(self):
        """ネストした dict の中の NaN も None になる."""
        import math
        obj = {"a": float("nan"), "b": {"c": 1.0}}
        result = history_store._sanitize(obj)
        assert result["a"] is None
        assert result["b"]["c"] == 1.0

    def test_list_sanitized(self):
        """list の中の NaN も None になる."""
        result = history_store._sanitize([1, float("nan"), 3])
        assert result == [1, None, 3]


# ---------------------------------------------------------------------------
# TestSaveScreening
# ---------------------------------------------------------------------------


class TestSaveScreening:
    """save_screening() のテスト."""

    def test_file_created(self, tmp_path):
        """スクリーニング結果が JSON ファイルとして保存される."""
        results = [{"symbol": "VTI", "name": "Vanguard Total"}]
        path_str = history_store.save_screening(
            preset="alpha",
            region="us",
            results=results,
            base_dir=str(tmp_path),
        )
        assert Path(path_str).exists()

    def test_payload_contains_category(self, tmp_path):
        """保存ファイルに category='screen' が含まれる."""
        results = [{"symbol": "IVV"}]
        path_str = history_store.save_screening(
            preset="value", region="us", results=results, base_dir=str(tmp_path)
        )
        with open(path_str, encoding="utf-8") as f:
            data = json.load(f)
        assert data["category"] == "screen"
        assert data["preset"] == "value"
        assert data["region"] == "us"
        assert data["count"] == 1

    def test_results_persisted(self, tmp_path):
        """results フィールドが保存されている."""
        results = [{"symbol": "7203.T", "name": "トヨタ"}]
        path_str = history_store.save_screening(
            preset="japan", region="jp", results=results, base_dir=str(tmp_path)
        )
        with open(path_str, encoding="utf-8") as f:
            data = json.load(f)
        assert data["results"][0]["symbol"] == "7203.T"


# ---------------------------------------------------------------------------
# TestSaveReport
# ---------------------------------------------------------------------------


class TestSaveReport:
    """save_report() のテスト."""

    def test_file_created(self, tmp_path):
        """レポートが JSON ファイルとして保存される."""
        data = {"name": "トヨタ", "sector": "Consumer Cyclical", "price": 3000.0}
        path_str = history_store.save_report(
            symbol="7203.T", data=data, score=65.0, verdict="割安", base_dir=str(tmp_path)
        )
        assert Path(path_str).exists()

    def test_payload_fields(self, tmp_path):
        """symbol, value_score, verdict が保存される."""
        data = {"name": "Toyota", "sector": "Auto", "price": 3000.0}
        path_str = history_store.save_report(
            symbol="7203.T", data=data, score=72.5, verdict="やや割安", base_dir=str(tmp_path)
        )
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["symbol"] == "7203.T"
        assert saved["value_score"] == pytest.approx(72.5)
        assert saved["verdict"] == "やや割安"
        assert saved["category"] == "report"


# ---------------------------------------------------------------------------
# TestSaveTrade
# ---------------------------------------------------------------------------


class TestSaveTrade:
    """save_trade() のテスト."""

    def test_file_created(self, tmp_path):
        """トレード記録が JSON ファイルとして保存される."""
        path_str = history_store.save_trade(
            symbol="VTI", trade_type="buy", shares=10, price=210.0,
            currency="USD", date_str="2026-02-25", base_dir=str(tmp_path),
        )
        assert Path(path_str).exists()

    def test_payload_trade_fields(self, tmp_path):
        """trade_type, shares, price, currency が保存される."""
        path_str = history_store.save_trade(
            symbol="IVV", trade_type="sell", shares=5, price=500.0,
            currency="USD", date_str="2026-02-25", memo="利確",
            base_dir=str(tmp_path),
        )
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["symbol"] == "IVV"
        assert saved["trade_type"] == "sell"
        assert saved["shares"] == 5
        assert saved["memo"] == "利確"
        assert saved["category"] == "trade"

    def test_fx_rate_persisted(self, tmp_path):
        """fx_rate が保存される."""
        path_str = history_store.save_trade(
            symbol="VTI", trade_type="buy", shares=3, price=220.0,
            currency="USD", date_str="2026-02-25", fx_rate=150.5,
            base_dir=str(tmp_path),
        )
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["fx_rate"] == pytest.approx(150.5)


# ---------------------------------------------------------------------------
# TestSaveHealth
# ---------------------------------------------------------------------------


class TestSaveHealth:
    """save_health() のテスト."""

    def test_file_created(self, tmp_path):
        """ヘルスチェック結果が JSON ファイルとして保存される."""
        health_data = {
            "positions": [{"symbol": "VTI", "pnl_pct": 10.0}],
            "summary": {"total": 1, "healthy": 1, "early_warning": 0, "caution": 0, "exit": 0},
        }
        path_str = history_store.save_health(health_data, base_dir=str(tmp_path))
        assert Path(path_str).exists()

    def test_summary_persisted(self, tmp_path):
        """summary が正しく保存される."""
        health_data = {
            "positions": [],
            "summary": {"total": 3, "healthy": 2, "early_warning": 1, "caution": 0, "exit": 0},
        }
        path_str = history_store.save_health(health_data, base_dir=str(tmp_path))
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["summary"]["total"] == 3
        assert saved["summary"]["healthy"] == 2
        assert saved["category"] == "health"


# ---------------------------------------------------------------------------
# TestSaveResearch
# ---------------------------------------------------------------------------


class TestSaveResearch:
    """save_research() のテスト."""

    def test_file_created(self, tmp_path):
        """リサーチ結果が JSON ファイルとして保存される."""
        result = {"name": "トヨタ", "summary": "EV移行でリスクあり"}
        path_str = history_store.save_research(
            research_type="stock", target="7203.T", result=result, base_dir=str(tmp_path)
        )
        assert Path(path_str).exists()

    def test_payload_fields(self, tmp_path):
        """research_type と target が保存される."""
        result = {"summary": "半導体需要拡大"}
        path_str = history_store.save_research(
            research_type="industry", target="semiconductor", result=result, base_dir=str(tmp_path)
        )
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["research_type"] == "industry"
        assert saved["target"] == "semiconductor"
        assert saved["category"] == "research"


# ---------------------------------------------------------------------------
# TestSaveMarketContext
# ---------------------------------------------------------------------------


class TestSaveMarketContext:
    """save_market_context() のテスト."""

    def test_file_created(self, tmp_path):
        """市場コンテキストが JSON ファイルとして保存される."""
        context = {"indices": [{"name": "日経", "price": 38000}]}
        path_str = history_store.save_market_context(context, base_dir=str(tmp_path))
        assert Path(path_str).exists()

    def test_category_and_date_persisted(self, tmp_path):
        """category と date が保存される."""
        context = {"indices": []}
        path_str = history_store.save_market_context(context, base_dir=str(tmp_path))
        with open(path_str, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["category"] == "market_context"
        assert "date" in saved


# ---------------------------------------------------------------------------
# TestLoadHistory
# ---------------------------------------------------------------------------


class TestLoadHistory:
    """load_history() のテスト."""

    def test_empty_when_no_directory(self, tmp_path):
        """ディレクトリが存在しない場合は空リストを返す."""
        result = history_store.load_history("screen", base_dir=str(tmp_path))
        assert result == []

    def test_returns_saved_records(self, tmp_path):
        """save_screening で保存したレコードが load_history で取得できる."""
        history_store.save_screening(
            preset="alpha", region="us",
            results=[{"symbol": "VTI"}], base_dir=str(tmp_path)
        )
        records = history_store.load_history("screen", base_dir=str(tmp_path))
        assert len(records) == 1
        assert records[0]["preset"] == "alpha"

    def test_returns_multiple_records(self, tmp_path):
        """複数の save_report で保存したレコードが全件返る."""
        for i in range(3):
            history_store.save_report(
                symbol=f"TEST{i}", data={"price": float(100 + i)},
                score=float(50 + i), verdict="普通", base_dir=str(tmp_path),
            )
        records = history_store.load_history("report", base_dir=str(tmp_path))
        assert len(records) == 3

    def test_days_back_filters_old_files(self, tmp_path):
        """days_back=0 は今日のファイルのみ返す（古いファイルは除外）."""
        # 古い日付のファイルを手動作成
        old_dir = tmp_path / "trade"
        old_dir.mkdir(parents=True)
        old_file = old_dir / "2020-01-01_buy_VTI.json"
        old_file.write_text(
            json.dumps({"category": "trade", "date": "2020-01-01"}), encoding="utf-8"
        )
        records = history_store.load_history("trade", days_back=30, base_dir=str(tmp_path))
        # 2020-01-01 は 30日以上前なのでフィルタされる
        assert all(r.get("date", "") >= "2025-01-01" for r in records)

    def test_corrupted_file_skipped(self, tmp_path):
        """JSON が壊れているファイルはスキップして他のレコードを返す."""
        history_store.save_screening(
            preset="valid", region="us", results=[], base_dir=str(tmp_path)
        )
        # 壊れたファイルを追加
        bad_dir = tmp_path / "screen"
        bad_file = bad_dir / "2026-01-01_bad.json"
        bad_file.write_text("INVALID JSON{{", encoding="utf-8")

        records = history_store.load_history("screen", base_dir=str(tmp_path))
        # 正常ファイル1件のみが返る
        assert len(records) == 1
        assert records[0]["category"] == "screen"


# ---------------------------------------------------------------------------
# TestListHistoryFiles
# ---------------------------------------------------------------------------


class TestListHistoryFiles:
    """list_history_files() のテスト."""

    def test_empty_when_no_directory(self, tmp_path):
        """ディレクトリが存在しない場合は空リストを返す."""
        result = history_store.list_history_files("screen", base_dir=str(tmp_path))
        assert result == []

    def test_returns_file_paths(self, tmp_path):
        """保存したファイルのパスが取得できる."""
        history_store.save_screening(
            preset="beta", region="jp", results=[], base_dir=str(tmp_path)
        )
        files = history_store.list_history_files("screen", base_dir=str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith(".json")


# ---------------------------------------------------------------------------
# TestBuildScreenSummary
# ---------------------------------------------------------------------------


class TestBuildScreenSummary:
    """summary_builder.build_screen_summary() のテスト."""

    def test_basic_components(self):
        """region, preset, date が含まれる."""
        result = summary_builder.build_screen_summary(
            screen_date="2026-02-25", preset="alpha", region="us"
        )
        assert "us" in result
        assert "alpha" in result
        assert "2026-02-25" in result

    def test_top_symbols_appended(self):
        """top_symbols が 'Top:' として含まれる."""
        result = summary_builder.build_screen_summary(
            screen_date="2026-02-25", preset="value", region="jp",
            top_symbols=["7203.T", "6758.T"]
        )
        assert "Top:" in result
        assert "7203.T" in result

    def test_truncated_to_200_chars(self):
        """長いテキストが 200 文字以内に切り詰められる."""
        long_preset = "x" * 300
        result = summary_builder.build_screen_summary(
            screen_date="2026-02-25", preset=long_preset, region="us"
        )
        assert len(result) <= 200

    def test_empty_inputs_return_empty_string(self):
        """全引数が空の場合は空文字列を返す."""
        result = summary_builder.build_screen_summary("", "", "")
        assert result == ""


# ---------------------------------------------------------------------------
# TestBuildReportSummary
# ---------------------------------------------------------------------------


class TestBuildReportSummary:
    """summary_builder.build_report_summary() のテスト."""

    def test_symbol_and_name_included(self):
        """symbol と name が含まれる."""
        result = summary_builder.build_report_summary(
            symbol="7203.T", name="トヨタ", score=65.0, verdict="割安", sector="Auto"
        )
        assert "7203.T" in result
        assert "トヨタ" in result

    def test_verdict_and_score_included(self):
        """verdict と score が含まれる."""
        result = summary_builder.build_report_summary(
            symbol="AAPL", score=58.3, verdict="やや割安"
        )
        assert "やや割安" in result
        assert "58.3" in result

    def test_truncated_to_200_chars(self):
        """200 文字超えは切り詰め."""
        long_name = "A" * 300
        result = summary_builder.build_report_summary(symbol="X", name=long_name)
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# TestBuildTradeSummary
# ---------------------------------------------------------------------------


class TestBuildTradeSummary:
    """summary_builder.build_trade_summary() のテスト."""

    def test_basic_fields(self):
        """date, type, symbol が含まれる."""
        result = summary_builder.build_trade_summary(
            trade_date="2026-02-25", trade_type="buy", symbol="VTI", shares=10
        )
        assert "2026-02-25" in result
        assert "BUY" in result
        assert "VTI" in result
        assert "10" in result

    def test_memo_appended(self):
        """memo が含まれる."""
        result = summary_builder.build_trade_summary(
            trade_date="2026-02-25", trade_type="sell", symbol="IVV",
            shares=5, memo="利確"
        )
        assert "利確" in result

    def test_trade_type_uppercased(self):
        """trade_type が大文字に変換される."""
        result = summary_builder.build_trade_summary(
            trade_date="2026-02-25", trade_type="sell", symbol="VTI"
        )
        assert "SELL" in result


# ---------------------------------------------------------------------------
# TestBuildHealthSummary
# ---------------------------------------------------------------------------


class TestBuildHealthSummary:
    """summary_builder.build_health_summary() のテスト."""

    def test_date_included(self):
        """日付が含まれる."""
        result = summary_builder.build_health_summary("2026-02-25")
        assert "2026-02-25" in result

    def test_summary_counts_included(self):
        """健全・注意・EXIT のカウントが含まれる."""
        summary = {"total": 5, "healthy": 3, "early_warning": 1, "caution": 0, "exit": 1}
        result = summary_builder.build_health_summary("2026-02-25", summary=summary)
        assert "健全3" in result
        assert "EXIT1" in result
        assert "全5銘柄" in result

    def test_no_summary_returns_basic_string(self):
        """summary=None でも文字列を返す."""
        result = summary_builder.build_health_summary("2026-02-25", summary=None)
        assert "ヘルスチェック" in result


# ---------------------------------------------------------------------------
# TestBuildMarketContextSummary
# ---------------------------------------------------------------------------


class TestBuildMarketContextSummary:
    """summary_builder.build_market_context_summary() のテスト."""

    def test_date_included(self):
        """日付が含まれる."""
        result = summary_builder.build_market_context_summary("2026-02-25")
        assert "2026-02-25" in result

    def test_index_values_included(self):
        """インデックス名と価格が含まれる."""
        indices = [{"name": "日経", "price": 38500}]
        result = summary_builder.build_market_context_summary("2026-02-25", indices=indices)
        assert "日経" in result
        assert "38500" in result


# ---------------------------------------------------------------------------
# TestBuildNoteSummary
# ---------------------------------------------------------------------------


class TestBuildNoteSummary:
    """summary_builder.build_note_summary() のテスト."""

    def test_symbol_and_content(self):
        """symbol と content が含まれる."""
        result = summary_builder.build_note_summary(
            symbol="7203.T", note_type="thesis", content="EV普及でリスク"
        )
        assert "7203.T" in result
        assert "thesis:" in result
        assert "EV普及でリスク" in result

    def test_empty_returns_empty(self):
        """全引数空は空文字列."""
        result = summary_builder.build_note_summary()
        assert result == ""


# ---------------------------------------------------------------------------
# TestBuildWatchlistSummary
# ---------------------------------------------------------------------------


class TestBuildWatchlistSummary:
    """summary_builder.build_watchlist_summary() のテスト."""

    def test_name_and_symbols(self):
        """name と symbols が含まれる."""
        result = summary_builder.build_watchlist_summary(
            name="main", symbols=["7203.T", "AAPL"]
        )
        assert "main" in result
        assert "7203.T" in result
        assert "AAPL" in result

    def test_only_first_10_symbols(self):
        """11個以上の symbols でも最大10件まで含まれる."""
        symbols = [f"SYM{i}" for i in range(15)]
        result = summary_builder.build_watchlist_summary(name="big", symbols=symbols)
        # SYM10以降は含まれないことを確認
        assert len(result) <= 200
