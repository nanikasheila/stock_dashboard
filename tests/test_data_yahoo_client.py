"""src/data/yahoo_client のユニットテスト."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.data.yahoo_client as yahoo_client

# ---------------------------------------------------------------------------
# TestSafeGet
# ---------------------------------------------------------------------------


class TestSafeGet:
    """_safe_get() のテスト."""

    def test_returns_value_for_existing_key(self):
        """存在するキーの値を返す."""
        assert yahoo_client._safe_get({"price": 100.0}, "price") == 100.0

    def test_returns_none_for_missing_key(self):
        """存在しないキーは None を返す."""
        assert yahoo_client._safe_get({}, "price") is None

    def test_returns_none_for_none_value(self):
        """値が None のキーは None を返す."""
        assert yahoo_client._safe_get({"price": None}, "price") is None

    def test_returns_none_for_nan(self):
        """float NaN は None を返す."""
        assert yahoo_client._safe_get({"val": float("nan")}, "val") is None

    def test_returns_none_for_inf(self):
        """float inf は None を返す."""
        assert yahoo_client._safe_get({"val": float("inf")}, "val") is None

    def test_returns_none_for_negative_inf(self):
        """-inf は None を返す."""
        assert yahoo_client._safe_get({"val": float("-inf")}, "val") is None

    def test_returns_string_value(self):
        """文字列値はそのまま返す."""
        assert yahoo_client._safe_get({"name": "Toyota"}, "name") == "Toyota"

    def test_returns_zero_as_value(self):
        """0 は None でなく 0 を返す."""
        assert yahoo_client._safe_get({"count": 0}, "count") == 0


# ---------------------------------------------------------------------------
# TestNormalizeRatio
# ---------------------------------------------------------------------------


class TestNormalizeRatio:
    """_normalize_ratio() のテスト."""

    def test_none_returns_none(self):
        """None は None を返す."""
        assert yahoo_client._normalize_ratio(None) is None

    def test_divides_by_100(self):
        """値を 100 で割って返す."""
        assert yahoo_client._normalize_ratio(3.87) == pytest.approx(0.0387)

    def test_zero_returns_zero(self):
        """0 は 0 を返す."""
        assert yahoo_client._normalize_ratio(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestSanitizeAnomalies
# ---------------------------------------------------------------------------


class TestSanitizeAnomalies:
    """_sanitize_anomalies() のテスト."""

    def test_excessive_dividend_yield_cleared(self):
        """dividend_yield > 15% は None にする."""
        data = {"dividend_yield": 0.20}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["dividend_yield"] is None

    def test_normal_dividend_yield_kept(self):
        """dividend_yield <= 15% はそのまま."""
        data = {"dividend_yield": 0.03}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["dividend_yield"] == pytest.approx(0.03)

    def test_extreme_low_pbr_cleared(self):
        """pbr < 0.05 は None にする."""
        data = {"pbr": 0.01}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["pbr"] is None

    def test_normal_pbr_kept(self):
        """通常の pbr はそのまま."""
        data = {"pbr": 1.5}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["pbr"] == pytest.approx(1.5)

    def test_extreme_roe_cleared(self):
        """roe > 200% は None にする."""
        data = {"roe": 3.5}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["roe"] is None

    def test_normal_roe_kept(self):
        """通常の roe はそのまま."""
        data = {"roe": 0.15}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["roe"] == pytest.approx(0.15)

    def test_low_per_cleared(self):
        """per が 0 < per < 1 は None にする."""
        data = {"per": 0.5}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["per"] is None

    def test_normal_per_kept(self):
        """通常の per はそのまま."""
        data = {"per": 15.0}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["per"] == pytest.approx(15.0)

    def test_missing_keys_no_error(self):
        """該当キーが無くてもエラーなし."""
        data = {"price": 100.0}
        result = yahoo_client._sanitize_anomalies(data)
        assert result["price"] == 100.0


# ---------------------------------------------------------------------------
# TestCacheFunctions
# ---------------------------------------------------------------------------


class TestCacheFunctions:
    """_read_cache() / _write_cache() のテスト."""

    def test_cache_miss_returns_none(self, tmp_path):
        """キャッシュファイルが存在しない場合は None を返す."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            result = yahoo_client._read_cache("AAPL")
        assert result is None

    def test_cache_write_and_read(self, tmp_path):
        """書き込んだキャッシュが読み込める."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            data = {"symbol": "AAPL", "price": 180.0}
            yahoo_client._write_cache("AAPL", data)
            result = yahoo_client._read_cache("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 180.0

    def test_expired_cache_returns_none(self, tmp_path):
        """TTL 超過したキャッシュは None を返す."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            # 古いタイムスタンプでキャッシュを書き込む
            old_time = (datetime.now() - timedelta(hours=25)).isoformat()
            data = {"symbol": "AAPL", "price": 180.0, "_cached_at": old_time}
            path = yahoo_client._cache_path("AAPL")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = yahoo_client._read_cache("AAPL")
        assert result is None

    def test_corrupted_cache_returns_none(self, tmp_path):
        """破損したキャッシュは None を返す."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            path = yahoo_client._cache_path("AAPL")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("INVALID{JSON", encoding="utf-8")
            result = yahoo_client._read_cache("AAPL")
        assert result is None

    def test_cache_path_safe_filename(self, tmp_path):
        """ドット含みシンボルのキャッシュパスが安全なファイル名になる."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            path = yahoo_client._cache_path("7203.T")
        assert "." not in path.stem


# ---------------------------------------------------------------------------
# TestGetStockInfo
# ---------------------------------------------------------------------------


class TestGetStockInfo:
    """get_stock_info() のテスト."""

    def _make_mock_ticker(self, info: dict) -> MagicMock:
        """モック yf.Ticker を生成するヘルパー."""
        mock_ticker = MagicMock()
        mock_ticker.info = info
        return mock_ticker

    def test_returns_none_when_no_market_price(self, tmp_path):
        """regularMarketPrice が無い場合は None を返す."""
        mock_info = {"shortName": "Test Corp"}  # regularMarketPrice なし
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", return_value=self._make_mock_ticker(mock_info)),
        ):
            result = yahoo_client.get_stock_info("TEST")
        assert result is None

    def test_returns_none_on_exception(self, tmp_path):
        """yfinance が例外を投げた場合は None を返す."""
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", side_effect=Exception("network error")),
        ):
            result = yahoo_client.get_stock_info("TEST")
        assert result is None

    def test_returns_stock_data_from_api(self, tmp_path):
        """yfinance から正常にデータが取得できる."""
        mock_info = {
            "regularMarketPrice": 180.0,
            "shortName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "USD",
            "marketCap": 2_800_000_000_000,
            "trailingPE": 28.5,
            "priceToBook": 45.0,
            "dividendYield": 0.55,
            "returnOnEquity": 1.60,
        }
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", return_value=self._make_mock_ticker(mock_info)),
        ):
            result = yahoo_client.get_stock_info("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["price"] == pytest.approx(180.0)
        assert result["per"] == pytest.approx(28.5)

    def test_dividend_yield_normalized_to_ratio(self, tmp_path):
        """dividendYield が 100 で割られてレシオ形式になる."""
        mock_info = {
            "regularMarketPrice": 100.0,
            "dividendYield": 3.87,  # yfinance の percent 形式
        }
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", return_value=self._make_mock_ticker(mock_info)),
        ):
            result = yahoo_client.get_stock_info("VTI")

        assert result is not None
        assert result["dividend_yield"] == pytest.approx(0.0387)

    def test_returns_cached_data_without_api_call(self, tmp_path):
        """キャッシュがある場合は yfinance を呼ばない."""
        cached = {
            "symbol": "CACHED",
            "price": 99.0,
            "_cached_at": datetime.now().isoformat(),
        }
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path), patch("yfinance.Ticker") as mock_ticker_cls:
            # キャッシュを事前に書き込む
            path = yahoo_client._cache_path("CACHED")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            result = yahoo_client.get_stock_info("CACHED")

        mock_ticker_cls.assert_not_called()
        assert result is not None
        assert result["price"] == 99.0

    def test_result_written_to_cache(self, tmp_path):
        """取得成功後に結果がキャッシュに書き込まれる."""
        mock_info = {
            "regularMarketPrice": 200.0,
            "shortName": "Test Inc.",
        }
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", return_value=self._make_mock_ticker(mock_info)),
        ):
            yahoo_client.get_stock_info("TESTWRITE")

        cache_path = tmp_path / "TESTWRITE.json"
        assert cache_path.exists()


# ---------------------------------------------------------------------------
# TestGetMultipleStocks
# ---------------------------------------------------------------------------


class TestGetMultipleStocks:
    """get_multiple_stocks() のテスト."""

    def test_returns_dict_for_each_symbol(self, tmp_path):
        """各 symbol に対応する dict が返る."""
        mock_info = {"regularMarketPrice": 100.0}
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch("time.sleep"),
        ):  # スリープをスキップ
            results = yahoo_client.get_multiple_stocks(["A", "B"])

        assert "A" in results
        assert "B" in results

    def test_none_for_failed_symbol(self, tmp_path):
        """取得失敗した symbol は None になる."""
        with (
            patch.object(yahoo_client, "CACHE_DIR", tmp_path),
            patch("yfinance.Ticker", side_effect=Exception("fail")),
            patch("time.sleep"),
        ):
            results = yahoo_client.get_multiple_stocks(["FAIL"])

        assert results["FAIL"] is None


# ---------------------------------------------------------------------------
# TestCachePath
# ---------------------------------------------------------------------------


class TestCachePath:
    """_cache_path() / _detail_cache_path() のテスト."""

    def test_cache_path_returns_json_file(self, tmp_path):
        """キャッシュパスは .json 拡張子を持つ."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            path = yahoo_client._cache_path("AAPL")
        assert path.suffix == ".json"

    def test_detail_cache_path_has_detail_suffix(self, tmp_path):
        """ディテールキャッシュは _detail.json 形式."""
        with patch.object(yahoo_client, "CACHE_DIR", tmp_path):
            path = yahoo_client._detail_cache_path("AAPL")
        assert path.name.endswith("_detail.json")
