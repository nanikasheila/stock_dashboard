"""src/core/return_estimate の ユニットテスト.

対象:
  - _is_etf()               : ETF判定ロジック
  - _compute_buyback_yield() : 自社株買い利回り計算
  - _estimate_from_analyst() : アナリスト目標株価ベースの推定
  - _estimate_from_history() : ヒストリカルリターンベースの推定
  - _empty_estimate()        : 空推定値の返却
  - estimate_stock_return()  : 銘柄単位のリターン推定
  - estimate_portfolio_return(): ポートフォリオ全体のリターン推定
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.return_estimate import (
    RETURN_CAP,
    _compute_buyback_yield,
    _empty_estimate,
    _estimate_from_analyst,
    _estimate_from_history,
    _is_etf,
    estimate_portfolio_return,
    estimate_stock_return,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CSV_HEADER = "symbol,shares,cost_price,cost_currency,purchase_date,memo\n"


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(_CSV_HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def _make_analyst_detail(
    price: float = 100.0,
    target_low: float = 90.0,
    target_mean: float = 110.0,
    target_high: float = 130.0,
    dividend_yield: float = 0.02,
    analyst_count: int = 5,
) -> dict:
    """Return a minimal stock_detail dict for analyst method tests."""
    return {
        "price": price,
        "target_low_price": target_low,
        "target_mean_price": target_mean,
        "target_high_price": target_high,
        "dividend_yield": dividend_yield,
        "number_of_analyst_opinions": analyst_count,
        "recommendation_mean": 2.0,
        "forward_per": 20.0,
        "quoteType": "EQUITY",
        "sector": "Technology",
    }


def _make_price_history(n: int = 60, trend: float = 1.005) -> list[float]:
    """Generate a synthetic price history list of length *n*."""
    price = 100.0
    history = [price]
    for _ in range(n - 1):
        price *= trend
        history.append(round(price, 4))
    return history


# ===========================================================================
# _is_etf
# ===========================================================================


class TestIsEtf:
    """_is_etf() のテスト."""

    def test_stock_with_target_price_is_not_etf(self):
        """target_mean_price があればアナリストカバレッジ株 → ETF ではない."""
        detail = {"target_mean_price": 150.0, "quoteType": "ETF"}
        assert _is_etf(detail) is False

    def test_quotetype_etf_without_target_price_is_etf(self):
        """quoteType=ETF かつ target_mean_price なし → ETF."""
        detail = {"quoteType": "ETF", "target_mean_price": None}
        assert _is_etf(detail) is True

    def test_equity_with_sector_is_not_etf(self):
        """quoteType=EQUITY かつ sector あり → ETF ではない."""
        detail = {
            "quoteType": "EQUITY",
            "target_mean_price": None,
            "sector": "Technology",
        }
        assert _is_etf(detail) is False


# ===========================================================================
# _compute_buyback_yield
# ===========================================================================


class TestComputeBuybackYield:
    """_compute_buyback_yield() のテスト."""

    def test_normal_case(self):
        """正常入力で buyback_yield が計算される."""
        detail = {"stock_repurchase": -5_000_000, "market_cap": 100_000_000}
        result = _compute_buyback_yield(detail)
        assert result == pytest.approx(0.05)

    def test_no_repurchase_returns_zero(self):
        """stock_repurchase が None なら 0.0 を返す."""
        detail = {"stock_repurchase": None, "market_cap": 100_000_000}
        assert _compute_buyback_yield(detail) == 0.0

    def test_no_market_cap_returns_zero(self):
        """market_cap が None なら 0.0 を返す."""
        detail = {"stock_repurchase": -5_000_000, "market_cap": None}
        assert _compute_buyback_yield(detail) == 0.0

    def test_zero_market_cap_returns_zero(self):
        """market_cap が 0 のとき 0.0 を返す（ゼロ除算ガード）."""
        detail = {"stock_repurchase": -5_000_000, "market_cap": 0}
        assert _compute_buyback_yield(detail) == 0.0

    def test_positive_repurchase_treated_as_absolute(self):
        """stock_repurchase が正の値でも abs() で計算される."""
        detail = {"stock_repurchase": 5_000_000, "market_cap": 100_000_000}
        assert _compute_buyback_yield(detail) == pytest.approx(0.05)


# ===========================================================================
# _estimate_from_analyst
# ===========================================================================


class TestEstimateFromAnalyst:
    """_estimate_from_analyst() のテスト."""

    def test_all_targets_present(self):
        """high/mean/low が全て揃っていると3シナリオ全て返る."""
        detail = _make_analyst_detail(
            price=100.0,
            target_low=90.0,
            target_mean=110.0,
            target_high=130.0,
            dividend_yield=0.0,
            analyst_count=5,
        )
        result = _estimate_from_analyst(detail)
        assert result["method"] == "analyst"
        assert result["optimistic"] == pytest.approx((130 - 100) / 100)
        assert result["base"] == pytest.approx((110 - 100) / 100)
        assert result["pessimistic"] == pytest.approx((90 - 100) / 100)

    def test_dividend_yield_added(self):
        """dividend_yield が各シナリオに加算される."""
        detail = _make_analyst_detail(
            price=100.0,
            target_mean=110.0,
            target_low=90.0,
            target_high=130.0,
            dividend_yield=0.03,
            analyst_count=5,
        )
        result = _estimate_from_analyst(detail)
        assert result["base"] == pytest.approx((110 - 100) / 100 + 0.03)

    def test_zero_price_returns_empty(self):
        """price=0 のとき全フィールドが None の空推定を返す."""
        detail = {"price": 0.0, "target_mean_price": 110.0}
        result = _estimate_from_analyst(detail)
        assert result["base"] is None
        assert result["optimistic"] is None
        assert result["pessimistic"] is None

    def test_analyst_count_populated(self):
        """analyst_count フィールドが正しく取得される."""
        detail = _make_analyst_detail(analyst_count=8)
        result = _estimate_from_analyst(detail)
        assert result["analyst_count"] == 8

    def test_few_analysts_adds_spread(self):
        """アナリスト数が 2 以下のとき spread が付与される."""
        detail = _make_analyst_detail(
            price=100.0,
            target_low=110.0,
            target_mean=110.0,
            target_high=110.0,
            analyst_count=2,
        )
        result = _estimate_from_analyst(detail)
        # spread が付いているので optimistic != pessimistic
        assert result["optimistic"] != result["pessimistic"]

    def test_identical_targets_adds_spread(self):
        """high == low のとき spread が付与される."""
        detail = _make_analyst_detail(
            price=100.0,
            target_low=110.0,
            target_mean=110.0,
            target_high=110.0,
            analyst_count=5,
        )
        result = _estimate_from_analyst(detail)
        assert result["optimistic"] != result["pessimistic"]

    def test_missing_target_low_uses_fallback(self):
        """target_low がない場合はフォールバックで pessimistic が補完される."""
        detail = {
            "price": 100.0,
            "target_mean_price": 110.0,
            "target_high_price": 130.0,
            "target_low_price": None,
            "dividend_yield": 0.0,
            "number_of_analyst_opinions": 5,
        }
        result = _estimate_from_analyst(detail)
        assert result["pessimistic"] is not None


# ===========================================================================
# _estimate_from_history
# ===========================================================================


class TestEstimateFromHistory:
    """_estimate_from_history() のテスト."""

    def test_insufficient_data_returns_empty(self):
        """価格履歴が 22 本未満なら空推定を返す."""
        detail = {"price_history": [100.0] * 10}
        result = _estimate_from_history(detail)
        assert result["base"] is None

    def test_no_price_history_returns_empty(self):
        """price_history なしなら空推定を返す."""
        result = _estimate_from_history({})
        assert result["base"] is None

    def test_trending_market_base_positive(self):
        """上昇トレンドの履歴からは正の base 推定が返る."""
        detail = {"price_history": _make_price_history(n=252, trend=1.003)}
        result = _estimate_from_history(detail)
        assert result["method"] == "historical"
        assert result["base"] is not None
        assert result["base"] > 0

    def test_scenarios_ordered(self):
        """optimistic >= base >= pessimistic の順序が保たれる."""
        detail = {"price_history": _make_price_history(n=252, trend=1.002)}
        result = _estimate_from_history(detail)
        if result["base"] is not None:
            assert result["optimistic"] >= result["base"]
            assert result["base"] >= result["pessimistic"]

    def test_return_cap_respected(self):
        """推定値が RETURN_CAP (0.30) を超えない."""
        # 極端な上昇トレンドでもキャップ
        detail = {"price_history": _make_price_history(n=252, trend=1.02)}
        result = _estimate_from_history(detail)
        if result["optimistic"] is not None:
            assert result["optimistic"] <= RETURN_CAP
        if result["pessimistic"] is not None:
            assert result["pessimistic"] >= -RETURN_CAP

    def test_data_months_field_present(self):
        """data_months フィールドが返る."""
        detail = {"price_history": _make_price_history(n=252)}
        result = _estimate_from_history(detail)
        if result["base"] is not None:
            assert "data_months" in result
            assert result["data_months"] > 0


# ===========================================================================
# _empty_estimate
# ===========================================================================


class TestEmptyEstimate:
    """_empty_estimate() のテスト."""

    def test_all_scenario_fields_are_none(self):
        """全シナリオフィールドが None になる."""
        result = _empty_estimate("analyst")
        assert result["optimistic"] is None
        assert result["base"] is None
        assert result["pessimistic"] is None

    def test_method_field_propagated(self):
        """method フィールドに渡した文字列が入る."""
        result = _empty_estimate("historical")
        assert result["method"] == "historical"


# ===========================================================================
# estimate_stock_return
# ===========================================================================


class TestEstimateStockReturn:
    """estimate_stock_return() のテスト."""

    def test_analyst_method_for_covered_stock(self):
        """アナリストカバレッジ株には analyst メソッドが使われる."""
        detail = _make_analyst_detail(price=100.0)
        result = estimate_stock_return("AAPL", detail)
        assert result["method"] == "analyst"
        assert result["symbol"] == "AAPL"

    def test_historical_method_for_etf(self):
        """ETF（target_mean_price なし & quoteType=ETF）には historical メソッドを使う."""
        detail = {
            "price": 200.0,
            "quoteType": "ETF",
            "target_mean_price": None,
            "dividend_yield": 0.01,
            "currency": "USD",
            "name": "Vanguard Total Stock Market ETF",
            "price_history": _make_price_history(n=252, trend=1.002),
        }
        result = estimate_stock_return("VTI", detail)
        assert result["method"] == "historical"

    def test_fallback_to_history_when_no_analyst_data(self):
        """アナリスト推定が空（target_mean_price なし株式）で price_history ある場合 historical にフォールバック."""
        detail = {
            "price": 100.0,
            "quoteType": "EQUITY",
            "target_mean_price": None,
            "target_high_price": None,
            "target_low_price": None,
            "dividend_yield": 0.0,
            "number_of_analyst_opinions": 0,
            "sector": "Technology",
            "price_history": _make_price_history(n=252, trend=1.001),
        }
        result = estimate_stock_return("7203.T", detail)
        # base should not be None due to fallback
        assert result["base"] is not None

    def test_required_fields_present(self):
        """返り値に必須フィールドが全て含まれる."""
        detail = _make_analyst_detail()
        result = estimate_stock_return("AAPL", detail, news=[{"title": "Apple news"}])
        required_keys = [
            "symbol",
            "name",
            "price",
            "currency",
            "optimistic",
            "base",
            "pessimistic",
            "method",
            "analyst_count",
            "dividend_yield",
            "news",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_news_passed_through(self):
        """news リストがそのまま返り値に含まれる."""
        detail = _make_analyst_detail()
        news = [{"title": "headline"}]
        result = estimate_stock_return("AAPL", detail, news=news)
        assert result["news"] == news

    def test_x_sentiment_passed_through(self):
        """x_sentiment がそのまま返り値に含まれる."""
        detail = _make_analyst_detail()
        sentiment = {"positive": 0.7}
        result = estimate_stock_return("AAPL", detail, x_sentiment=sentiment)
        assert result["x_sentiment"] == sentiment

    def test_buyback_yield_field_present(self):
        """buyback_yield フィールドが返る."""
        detail = _make_analyst_detail()
        detail["stock_repurchase"] = -2_000_000
        detail["market_cap"] = 100_000_000
        result = estimate_stock_return("AAPL", detail)
        assert "buyback_yield" in result
        assert result["buyback_yield"] == pytest.approx(0.02)


# ===========================================================================
# estimate_portfolio_return
# ===========================================================================


class TestEstimatePortfolioReturn:
    """estimate_portfolio_return() のテスト."""

    def test_empty_portfolio_returns_none_scenarios(self, tmp_path: Path):
        """空ポートフォリオでは portfolio シナリオが全て None."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(_CSV_HEADER, encoding="utf-8")

        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 1.0}

        result = estimate_portfolio_return(str(csv_file), mock_client)
        assert result["portfolio"]["optimistic"] is None
        assert result["portfolio"]["base"] is None
        assert result["portfolio"]["pessimistic"] is None
        assert result["positions"] == []

    def test_single_stock_portfolio(self, tmp_path: Path):
        """1銘柄ポートフォリオでもポートフォリオ推定が返る."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["AAPL,10,180,USD,2024-01-01,"])

        mock_client = MagicMock()
        # FX rates
        mock_client.get_stock_info.return_value = {"price": 150.0}
        # get_stock_detail
        mock_client.get_stock_detail.return_value = {
            **_make_analyst_detail(price=200.0),
            "currency": "USD",
            "name": "Apple Inc.",
        }
        mock_client.get_stock_news.return_value = []

        result = estimate_portfolio_return(str(csv_file), mock_client)
        assert len(result["positions"]) == 1
        assert result["portfolio"]["base"] is not None
        assert result["total_value_jpy"] > 0

    def test_cash_position_handled(self, tmp_path: Path):
        """現金ポジションが method=cash で含まれる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["JPY.CASH,1,500000,JPY,2024-01-01,"])

        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 1.0}

        result = estimate_portfolio_return(str(csv_file), mock_client)
        assert len(result["positions"]) == 1
        cash_pos = result["positions"][0]
        assert cash_pos["method"] == "cash"
        assert cash_pos["base"] == 0.0

    def test_no_price_position_method_is_no_data(self, tmp_path: Path):
        """価格が取得できない銘柄は method=no_data で含まれる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["VTI,5,200,USD,2024-01-01,"])

        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 150.0}
        mock_client.get_stock_detail.return_value = None  # no data

        result = estimate_portfolio_return(str(csv_file), mock_client)
        assert result["positions"][0]["method"] == "no_data"

    def test_fx_rates_returned(self, tmp_path: Path):
        """fx_rates が返り値に含まれ JPY=1.0 になっている."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(_CSV_HEADER, encoding="utf-8")

        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 150.0}

        result = estimate_portfolio_return(str(csv_file), mock_client)
        assert "fx_rates" in result
        assert result["fx_rates"]["JPY"] == 1.0

    def test_weighted_average_calculation(self, tmp_path: Path):
        """ポートフォリオ加重平均が計算される（2銘柄）."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(
            csv_file,
            [
                "AAPL,10,180,USD,2024-01-01,",
                "VTI,5,200,USD,2024-01-01,",
            ],
        )

        def _detail_side_effect(symbol: str):
            return {
                **_make_analyst_detail(price=200.0),
                "currency": "USD",
                "name": symbol,
            }

        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 150.0}
        mock_client.get_stock_detail.side_effect = _detail_side_effect
        mock_client.get_stock_news.return_value = []

        result = estimate_portfolio_return(str(csv_file), mock_client)
        # Portfolio should have valid weighted returns
        assert result["portfolio"]["base"] is not None
        assert len(result["positions"]) == 2
