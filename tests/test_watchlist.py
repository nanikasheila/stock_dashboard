"""ウォッチリスト CRUD・価格エンリッチメント・LLM 分析のテスト."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# CRUD テスト
# ---------------------------------------------------------------------------


class TestLoadSaveWatchlist:
    """load_watchlist / save_watchlist の読み書きテスト."""

    def test_load_empty_when_file_missing(self, tmp_path: Path) -> None:
        """ファイルが存在しない場合は空リストを返す."""
        with patch("components.watchlist.WATCHLIST_PATH", tmp_path / "missing.json"):
            from components.watchlist import load_watchlist

            assert load_watchlist() == []

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """保存したデータがロードで復元される."""
        wl_path = tmp_path / "watchlist.json"
        items = [
            {
                "symbol": "AAPL",
                "target_price": 150.0,
                "target_currency": "USD",
                "added_date": "2026-01-01",
                "memo": "test",
            },
        ]
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import load_watchlist, save_watchlist

            save_watchlist(items)
            loaded = load_watchlist()
            assert len(loaded) == 1
            assert loaded[0]["symbol"] == "AAPL"
            assert loaded[0]["target_price"] == 150.0

    def test_load_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        """不正な JSON ファイルの場合は空リストを返す."""
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text("NOT JSON", encoding="utf-8")
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import load_watchlist

            assert load_watchlist() == []

    def test_load_returns_empty_on_non_list(self, tmp_path: Path) -> None:
        """JSON がリストでない場合は空リストを返す."""
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text('{"not": "a list"}', encoding="utf-8")
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import load_watchlist

            assert load_watchlist() == []


class TestAddRemoveWatchlist:
    """add_to_watchlist / remove_from_watchlist のテスト."""

    def test_add_new_item(self, tmp_path: Path) -> None:
        """新規銘柄を追加できる."""
        wl_path = tmp_path / "watchlist.json"
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import add_to_watchlist

            result = add_to_watchlist("MSFT", 400.0, "USD", "AI growth")
            assert len(result) == 1
            assert result[0]["symbol"] == "MSFT"
            assert result[0]["target_price"] == 400.0

    def test_add_overwrites_existing(self, tmp_path: Path) -> None:
        """同一シンボルの追加は上書きされる."""
        wl_path = tmp_path / "watchlist.json"
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import add_to_watchlist

            add_to_watchlist("MSFT", 400.0, "USD", "old memo")
            result = add_to_watchlist("MSFT", 450.0, "USD", "new memo")
            assert len(result) == 1
            assert result[0]["target_price"] == 450.0
            assert result[0]["memo"] == "new memo"

    def test_remove_existing(self, tmp_path: Path) -> None:
        """既存銘柄を削除できる."""
        wl_path = tmp_path / "watchlist.json"
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import add_to_watchlist, remove_from_watchlist

            add_to_watchlist("AAPL", 150.0)
            add_to_watchlist("MSFT", 400.0)
            result = remove_from_watchlist("AAPL")
            assert len(result) == 1
            assert result[0]["symbol"] == "MSFT"

    def test_remove_nonexistent_is_noop(self, tmp_path: Path) -> None:
        """存在しない銘柄の削除は何も変更しない."""
        wl_path = tmp_path / "watchlist.json"
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import add_to_watchlist, remove_from_watchlist

            add_to_watchlist("AAPL", 150.0)
            result = remove_from_watchlist("TSLA")
            assert len(result) == 1
            assert result[0]["symbol"] == "AAPL"

    def test_add_normalizes_symbol_case(self, tmp_path: Path) -> None:
        """シンボルは大文字に正規化される."""
        wl_path = tmp_path / "watchlist.json"
        with patch("components.watchlist.WATCHLIST_PATH", wl_path):
            from components.watchlist import add_to_watchlist

            result = add_to_watchlist("aapl", 150.0)
            assert result[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# 価格エンリッチメント テスト
# ---------------------------------------------------------------------------


class TestGetWatchlistWithPrices:
    """get_watchlist_with_prices のテスト."""

    def _make_stock_info(
        self,
        *,
        price: float = 180.0,
        sector: str = "Technology",
        per: float = 28.5,
        pbr: float = 12.3,
        dividend_yield: float = 0.005,
        currency: str = "USD",
    ) -> dict[str, Any]:
        """テスト用の stock_info を生成する."""
        return {
            "symbol": "AAPL",
            "price": price,
            "sector": sector,
            "per": per,
            "pbr": pbr,
            "dividend_yield": dividend_yield,
            "currency": currency,
        }

    @patch("components.watchlist.get_stock_info")
    def test_enriches_with_price_and_metrics(self, mock_get: MagicMock) -> None:
        """現在価格・指標が付加される."""
        from components.watchlist import get_watchlist_with_prices

        mock_get.return_value = self._make_stock_info(price=180.0)
        watchlist = [
            {"symbol": "AAPL", "target_price": 150.0, "target_currency": "USD", "memo": ""},
        ]
        result = get_watchlist_with_prices(watchlist, {"USD": 1.0})
        assert len(result) == 1
        assert result[0]["current_price"] == 180.0
        assert result[0]["sector"] == "Technology"
        assert result[0]["per"] == 28.5

    @patch("components.watchlist.get_stock_info")
    def test_distance_pct_positive_when_above_target(self, mock_get: MagicMock) -> None:
        """現在価格 > ターゲットで正の乖離率."""
        from components.watchlist import get_watchlist_with_prices

        mock_get.return_value = self._make_stock_info(price=200.0)
        watchlist = [
            {"symbol": "AAPL", "target_price": 150.0, "target_currency": "USD"},
        ]
        result = get_watchlist_with_prices(watchlist, {"USD": 1.0})
        dist = result[0]["distance_pct"]
        assert dist is not None
        assert dist > 0  # 200 > 150 → positive

    @patch("components.watchlist.get_stock_info")
    def test_distance_pct_negative_when_below_target(self, mock_get: MagicMock) -> None:
        """現在価格 < ターゲットで負の乖離率（買い時）."""
        from components.watchlist import get_watchlist_with_prices

        mock_get.return_value = self._make_stock_info(price=120.0)
        watchlist = [
            {"symbol": "AAPL", "target_price": 150.0, "target_currency": "USD"},
        ]
        result = get_watchlist_with_prices(watchlist, {"USD": 1.0})
        dist = result[0]["distance_pct"]
        assert dist is not None
        assert dist < 0  # 120 < 150 → negative

    @patch("components.watchlist.get_stock_info")
    def test_handles_none_stock_info(self, mock_get: MagicMock) -> None:
        """yahoo_client が None を返してもクラッシュしない."""
        from components.watchlist import get_watchlist_with_prices

        mock_get.return_value = None
        watchlist = [
            {"symbol": "FAKE", "target_price": 100.0, "target_currency": "USD"},
        ]
        result = get_watchlist_with_prices(watchlist, {"USD": 1.0})
        assert len(result) == 1
        assert result[0]["current_price"] is None
        assert result[0]["distance_pct"] is None

    @patch("components.watchlist.get_stock_info")
    def test_cross_currency_distance(self, mock_get: MagicMock) -> None:
        """異通貨間の乖離率計算が正しく動作する."""
        from components.watchlist import get_watchlist_with_prices

        # 株価は USD 建てで 150、ターゲットは JPY 建てで 22500
        mock_get.return_value = self._make_stock_info(price=150.0, currency="USD")
        watchlist = [
            {"symbol": "AAPL", "target_price": 22500.0, "target_currency": "JPY"},
        ]
        # JPY=1.0, USD=150.0 なので 22500 JPY = 150 USD → 乖離率 0%
        result = get_watchlist_with_prices(watchlist, {"JPY": 1.0, "USD": 150.0})
        dist = result[0]["distance_pct"]
        assert dist is not None
        assert abs(dist) < 0.01  # ≈ 0%


# ---------------------------------------------------------------------------
# LLM 分析テスト
# ---------------------------------------------------------------------------


class TestAnalyzeWatchlistStock:
    """analyze_watchlist_stock のテスト."""

    @patch("components.watchlist.copilot_call")
    def test_returns_response_on_success(self, mock_call: MagicMock) -> None:
        """LLM 呼び出しが成功した場合に文字列を返す."""
        from components.watchlist import analyze_watchlist_stock

        mock_call.return_value = "概要: テスト銘柄は...\n投資判断: 割安...\nリスク: ..."
        stock_info = {
            "current_price": 180.0,
            "target_price": 150.0,
            "sector": "Technology",
            "per": 28.5,
            "pbr": 12.3,
            "dividend_yield": 0.005,
        }
        result = analyze_watchlist_stock("AAPL", stock_info, model="gpt-4.1")
        assert result is not None
        assert "概要" in result
        mock_call.assert_called_once()

    @patch("components.watchlist.copilot_call")
    def test_returns_none_on_failure(self, mock_call: MagicMock) -> None:
        """LLM 呼び出しが None を返した場合は None を返す."""
        from components.watchlist import analyze_watchlist_stock

        mock_call.return_value = None
        result = analyze_watchlist_stock("AAPL", {"price": 100}, model="gpt-4.1")
        assert result is None

    @patch("components.watchlist.copilot_call")
    def test_returns_none_on_exception(self, mock_call: MagicMock) -> None:
        """LLM 呼び出しが例外を投げた場合は None を返す."""
        from components.watchlist import analyze_watchlist_stock

        mock_call.side_effect = RuntimeError("connection failed")
        result = analyze_watchlist_stock("AAPL", {"price": 100}, model="gpt-4.1")
        assert result is None

    @patch("components.watchlist.copilot_call")
    def test_prompt_includes_symbol_and_metrics(self, mock_call: MagicMock) -> None:
        """プロンプトにシンボルと指標が含まれる."""
        from components.watchlist import analyze_watchlist_stock

        mock_call.return_value = "response"
        stock_info = {
            "current_price": 180.0,
            "target_price": 150.0,
            "sector": "Technology",
            "per": 28.5,
            "pbr": 12.3,
            "dividend_yield": 0.005,
        }
        analyze_watchlist_stock("MSFT", stock_info, model="gpt-4.1")
        prompt_arg = mock_call.call_args[0][0]
        assert "MSFT" in prompt_arg
        assert "Technology" in prompt_arg
        assert "28.5" in prompt_arg
