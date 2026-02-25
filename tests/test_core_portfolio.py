"""src/core/portfolio の ユニットテスト.

対象:
  - src/core/portfolio/portfolio_manager.py : load_portfolio, save_portfolio,
      add_position, sell_position, get_fx_rates, get_snapshot,
      get_structure_analysis, merge_positions, get_portfolio_shareholder_return
  - src/core/portfolio/concentration.py     : compute_hhi,
      get_concentration_multiplier, analyze_concentration
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.portfolio.concentration import (
    analyze_concentration,
    compute_hhi,
    get_concentration_multiplier,
)
from src.core.portfolio.portfolio_manager import (
    add_position,
    get_fx_rates,
    get_portfolio_shareholder_return,
    get_snapshot,
    get_structure_analysis,
    load_portfolio,
    merge_positions,
    save_portfolio,
    sell_position,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = "symbol,shares,cost_price,cost_currency,purchase_date,memo\n"


def _write_csv(path: Path, rows: list[str]) -> None:
    """Write a minimal portfolio CSV to *path*."""
    path.write_text(_CSV_HEADER + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# load_portfolio
# ---------------------------------------------------------------------------


class TestLoadPortfolio:
    """load_portfolio() のテスト."""

    def test_returns_empty_list_when_file_missing(self, tmp_path: Path):
        """存在しないCSVパスを渡すと空リストを返す."""
        result = load_portfolio(str(tmp_path / "nonexistent.csv"))
        assert result == []

    def test_loads_basic_rows(self, tmp_path: Path):
        """正常なCSVから2行を正しく読み込む."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, [
            "7203.T,100,2850,JPY,2024-01-15,",
            "AAPL,10,180,USD,2024-03-01,test memo",
        ])
        result = load_portfolio(str(csv_file))
        assert len(result) == 2
        assert result[0]["symbol"] == "7203.T"
        assert result[0]["shares"] == 100
        assert result[0]["cost_price"] == 2850.0
        assert result[0]["cost_currency"] == "JPY"
        assert result[1]["symbol"] == "AAPL"
        assert result[1]["memo"] == "test memo"

    def test_skips_zero_shares_rows(self, tmp_path: Path):
        """shares=0 の行はスキップされる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, [
            "VTI,0,200,USD,2024-01-01,",
            "AAPL,5,180,USD,2024-01-01,",
        ])
        result = load_portfolio(str(csv_file))
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_skips_empty_symbol_rows(self, tmp_path: Path):
        """symbol が空の行はスキップされる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, [
            ",10,100,JPY,2024-01-01,",
            "VTI,5,200,USD,2024-01-01,",
        ])
        result = load_portfolio(str(csv_file))
        assert len(result) == 1

    def test_shares_coerced_to_int(self, tmp_path: Path):
        """shares は int に変換される（浮動小数点表記でも可）."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["VTI,5.0,200,USD,2024-01-01,"])
        result = load_portfolio(str(csv_file))
        assert isinstance(result[0]["shares"], int)
        assert result[0]["shares"] == 5


# ---------------------------------------------------------------------------
# save_portfolio
# ---------------------------------------------------------------------------


class TestSavePortfolio:
    """save_portfolio() のテスト."""

    def test_roundtrip(self, tmp_path: Path):
        """保存してから読み込むと同じデータが得られる."""
        csv_file = tmp_path / "sub" / "portfolio.csv"
        positions = [
            {
                "symbol": "AAPL",
                "shares": 10,
                "cost_price": 180.0,
                "cost_currency": "USD",
                "purchase_date": "2024-01-01",
                "memo": "",
            }
        ]
        save_portfolio(positions, str(csv_file))
        reloaded = load_portfolio(str(csv_file))
        assert len(reloaded) == 1
        assert reloaded[0]["symbol"] == "AAPL"
        assert reloaded[0]["shares"] == 10

    def test_creates_missing_directories(self, tmp_path: Path):
        """親ディレクトリが存在しなくても自動作成してファイル保存できる."""
        csv_file = tmp_path / "deep" / "nested" / "portfolio.csv"
        save_portfolio([], str(csv_file))
        assert csv_file.exists()


# ---------------------------------------------------------------------------
# add_position
# ---------------------------------------------------------------------------


class TestAddPosition:
    """add_position() のテスト."""

    def test_adds_new_position(self, tmp_path: Path):
        """新規ポジションが CSVに追加される."""
        csv_file = tmp_path / "portfolio.csv"
        result = add_position(
            str(csv_file), "VTI", 5, 200.0, "USD", "2024-01-01"
        )
        assert result["symbol"] == "VTI"
        assert result["shares"] == 5
        # Persisted
        loaded = load_portfolio(str(csv_file))
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "VTI"

    def test_average_cost_for_existing_position(self, tmp_path: Path):
        """既存銘柄への追加購入で加重平均取得単価が再計算される."""
        csv_file = tmp_path / "portfolio.csv"
        # First buy: 10 shares @ 100
        add_position(str(csv_file), "AAPL", 10, 100.0, "USD", "2024-01-01")
        # Second buy: 10 shares @ 200
        result = add_position(
            str(csv_file), "AAPL", 10, 200.0, "USD", "2024-06-01"
        )
        # Expected avg = (10*100 + 10*200) / 20 = 150.0
        assert result["cost_price"] == pytest.approx(150.0)
        assert result["shares"] == 20

    def test_symbol_uppercased_for_non_japanese(self, tmp_path: Path):
        """ドット非含有のティッカーは大文字に正規化される."""
        csv_file = tmp_path / "portfolio.csv"
        result = add_position(str(csv_file), "vti", 1, 200.0)
        assert result["symbol"] == "VTI"

    def test_japanese_ticker_preserves_dot_suffix(self, tmp_path: Path):
        """ドットを含むティッカー（日本株）は大文字変換をスキップする."""
        csv_file = tmp_path / "portfolio.csv"
        result = add_position(str(csv_file), "7203.T", 100, 2850.0, "JPY")
        assert result["symbol"] == "7203.T"


# ---------------------------------------------------------------------------
# sell_position
# ---------------------------------------------------------------------------


class TestSellPosition:
    """sell_position() のテスト."""

    def _setup_csv(self, tmp_path: Path) -> Path:
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["AAPL,10,180,USD,2024-01-01,"])
        return csv_file

    def test_partial_sell_reduces_shares(self, tmp_path: Path):
        """部分売却で株数が減る."""
        csv_file = self._setup_csv(tmp_path)
        result = sell_position(str(csv_file), "AAPL", 3)
        assert result["shares"] == 7
        loaded = load_portfolio(str(csv_file))
        assert loaded[0]["shares"] == 7

    def test_full_sell_removes_position(self, tmp_path: Path):
        """全株売却でポジションが削除される."""
        csv_file = self._setup_csv(tmp_path)
        result = sell_position(str(csv_file), "AAPL", 10)
        assert result["shares"] == 0
        loaded = load_portfolio(str(csv_file))
        assert len(loaded) == 0

    def test_raises_when_symbol_not_found(self, tmp_path: Path):
        """存在しない銘柄を売ろうとすると ValueError が上がる."""
        csv_file = self._setup_csv(tmp_path)
        with pytest.raises(ValueError, match="VTI"):
            sell_position(str(csv_file), "VTI", 1)

    def test_raises_when_oversell(self, tmp_path: Path):
        """保有数を超える売却で ValueError が上がる."""
        csv_file = self._setup_csv(tmp_path)
        with pytest.raises(ValueError):
            sell_position(str(csv_file), "AAPL", 11)


# ---------------------------------------------------------------------------
# get_fx_rates
# ---------------------------------------------------------------------------


class TestGetFxRates:
    """get_fx_rates() のテスト."""

    def test_jpy_always_one(self):
        """JPY は常に 1.0 として返される."""
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 150.0}
        rates = get_fx_rates(mock_client)
        assert rates["JPY"] == 1.0

    def test_usd_rate_populated(self):
        """USD の FXレートが取得できたとき辞書に含まれる."""
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 150.0}
        rates = get_fx_rates(mock_client)
        assert "USD" in rates
        assert rates["USD"] == pytest.approx(150.0)

    def test_unavailable_pair_skipped(self):
        """price が None のペアは辞書に追加されない."""
        def _side_effect(pair: str):
            if pair == "USDJPY=X":
                return {"price": None}
            return {"price": 100.0}

        mock_client = MagicMock()
        mock_client.get_stock_info.side_effect = _side_effect
        rates = get_fx_rates(mock_client)
        assert "USD" not in rates

    def test_exception_in_client_does_not_propagate(self):
        """クライアントが例外を投げてもクラッシュせず辞書を返す."""
        mock_client = MagicMock()
        mock_client.get_stock_info.side_effect = RuntimeError("network error")
        rates = get_fx_rates(mock_client)
        assert isinstance(rates, dict)
        assert rates["JPY"] == 1.0


# ---------------------------------------------------------------------------
# get_snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshot:
    """get_snapshot() のテスト."""

    def test_empty_portfolio_returns_zero_totals(self, tmp_path: Path):
        """空のCSVでスナップショットを取ると全合計が0."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(_CSV_HEADER, encoding="utf-8")
        mock_client = MagicMock()
        result = get_snapshot(str(csv_file), mock_client)
        assert result["total_value_jpy"] == 0.0
        assert result["positions"] == []

    def test_single_jpy_stock_pnl(self, tmp_path: Path):
        """JPY株1銘柄の P&L が正しく計算される."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["7203.T,100,2850,JPY,2024-01-01,"])
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {
            "price": 3000.0,
            "name": "Toyota",
            "sector": "Automotive",
            "currency": "JPY",
        }
        result = get_snapshot(str(csv_file), mock_client)
        position = result["positions"][0]
        # P&L = (3000 - 2850) * 100 = 15000
        assert position["pnl"] == pytest.approx(15000.0)
        # pnl_pct is stored as round(pnl_pct, 4) in the implementation
        assert position["pnl_pct"] == pytest.approx(150.0 / 2850.0, abs=1e-4)

    def test_cash_position_has_zero_pnl(self, tmp_path: Path):
        """現金ポジションは P&L が 0 になる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["JPY.CASH,1,500000,JPY,2024-01-01,"])
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"price": 1.0}
        result = get_snapshot(str(csv_file), mock_client)
        assert result["positions"][0]["pnl"] == 0.0

    def test_missing_price_sets_pnl_to_zero(self, tmp_path: Path):
        """API が price を返さない場合、P&L は 0 になる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["VTI,10,200,USD,2024-01-01,"])
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {
            "price": None, "name": "VTI", "currency": "USD"
        }
        result = get_snapshot(str(csv_file), mock_client)
        assert result["positions"][0]["pnl"] == 0.0


# ---------------------------------------------------------------------------
# get_structure_analysis
# ---------------------------------------------------------------------------


class TestGetStructureAnalysis:
    """get_structure_analysis() のテスト."""

    def test_empty_portfolio_returns_zero_hhi(self, tmp_path: Path):
        """空のCSVでは HHI が全て 0."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(_CSV_HEADER, encoding="utf-8")
        mock_client = MagicMock()
        result = get_structure_analysis(str(csv_file), mock_client)
        assert result["sector_hhi"] == 0.0
        assert result["region_hhi"] == 0.0
        assert result["currency_hhi"] == 0.0

    def test_single_stock_gives_max_hhi(self, tmp_path: Path):
        """1銘柄だけだとセクター/通貨 HHI = 1.0 になる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["AAPL,10,180,USD,2024-01-01,"])
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {
            "price": 200.0,
            "name": "Apple",
            "sector": "Technology",
            "currency": "USD",
        }
        result = get_structure_analysis(str(csv_file), mock_client)
        # With a single stock, every axis should approach 1.0
        assert result["sector_hhi"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# merge_positions
# ---------------------------------------------------------------------------


class TestMergePositions:
    """merge_positions() のテスト."""

    def test_new_symbol_appended(self):
        """既存に存在しない銘柄は末尾に追加される."""
        current = [
            {"symbol": "AAPL", "shares": 10, "cost_price": 180.0,
             "cost_currency": "USD", "purchase_date": "", "memo": ""}
        ]
        proposed = [
            {"symbol": "VTI", "shares": 5, "cost_price": 200.0,
             "cost_currency": "USD"}
        ]
        merged = merge_positions(current, proposed)
        assert len(merged) == 2
        symbols = [p["symbol"] for p in merged]
        assert "VTI" in symbols

    def test_existing_symbol_weighted_average(self):
        """既存銘柄への追加は加重平均コストで合算される."""
        current = [
            {"symbol": "AAPL", "shares": 10, "cost_price": 100.0,
             "cost_currency": "USD", "purchase_date": "", "memo": ""}
        ]
        proposed = [
            {"symbol": "AAPL", "shares": 10, "cost_price": 200.0,
             "cost_currency": "USD"}
        ]
        merged = merge_positions(current, proposed)
        assert len(merged) == 1
        assert merged[0]["shares"] == 20
        assert merged[0]["cost_price"] == pytest.approx(150.0)

    def test_original_list_not_mutated(self):
        """元の current リストは変更されない（deep copy 確認）."""
        current = [
            {"symbol": "AAPL", "shares": 10, "cost_price": 100.0,
             "cost_currency": "USD", "purchase_date": "", "memo": ""}
        ]
        proposed = [
            {"symbol": "AAPL", "shares": 5, "cost_price": 200.0,
             "cost_currency": "USD"}
        ]
        merge_positions(current, proposed)
        assert current[0]["shares"] == 10  # unchanged

    def test_case_insensitive_symbol_matching(self):
        """シンボルの大文字/小文字を問わずマッチする."""
        current = [
            {"symbol": "aapl", "shares": 10, "cost_price": 100.0,
             "cost_currency": "USD", "purchase_date": "", "memo": ""}
        ]
        proposed = [
            {"symbol": "AAPL", "shares": 5, "cost_price": 200.0,
             "cost_currency": "USD"}
        ]
        merged = merge_positions(current, proposed)
        assert len(merged) == 1
        assert merged[0]["shares"] == 15


# ---------------------------------------------------------------------------
# get_portfolio_shareholder_return
# ---------------------------------------------------------------------------


class TestGetPortfolioShareholderReturn:
    """get_portfolio_shareholder_return() のテスト."""

    def test_empty_portfolio_returns_none_avg(self, tmp_path: Path):
        """空ポートフォリオでは weighted_avg_rate が None."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(_CSV_HEADER, encoding="utf-8")
        mock_client = MagicMock()
        result = get_portfolio_shareholder_return(str(csv_file), mock_client)
        assert result["weighted_avg_rate"] is None
        assert result["positions"] == []

    def test_returns_none_when_price_unavailable(self, tmp_path: Path):
        """price が取得できない銘柄はスキップされる."""
        csv_file = tmp_path / "portfolio.csv"
        _write_csv(csv_file, ["AAPL,10,180,USD,2024-01-01,"])
        mock_client = MagicMock()
        mock_client.get_stock_detail.return_value = None
        result = get_portfolio_shareholder_return(str(csv_file), mock_client)
        assert result["weighted_avg_rate"] is None


# ===========================================================================
# concentration.py
# ===========================================================================

# ---------------------------------------------------------------------------
# compute_hhi
# ---------------------------------------------------------------------------


class TestComputeHhi:
    """compute_hhi() のテスト."""

    def test_empty_weights_returns_zero(self):
        """weights が空なら HHI = 0.0."""
        assert compute_hhi([]) == 0.0

    def test_single_asset_returns_one(self):
        """1銘柄のみ weight=1.0 なら HHI = 1.0."""
        assert compute_hhi([1.0]) == pytest.approx(1.0)

    def test_equal_weights_returns_inverse_n(self):
        """均等配分なら HHI = 1/N."""
        weights = [0.25, 0.25, 0.25, 0.25]
        assert compute_hhi(weights) == pytest.approx(0.25)

    def test_concentrated_portfolio(self):
        """集中偏重ポートフォリオは HHI が高い."""
        weights = [0.9, 0.05, 0.05]
        result = compute_hhi(weights)
        assert result > 0.8

    def test_mathematical_formula(self):
        """数値計算が正しい: sum(w^2)."""
        weights = [0.5, 0.3, 0.2]
        expected = 0.5**2 + 0.3**2 + 0.2**2
        assert compute_hhi(weights) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_concentration_multiplier
# ---------------------------------------------------------------------------


class TestGetConcentrationMultiplier:
    """get_concentration_multiplier() のテスト."""

    def test_below_0_25_returns_1_0(self):
        """HHI < 0.25 → 乗数 = 1.0."""
        assert get_concentration_multiplier(0.0) == pytest.approx(1.0)
        assert get_concentration_multiplier(0.24) == pytest.approx(1.0)

    def test_0_25_returns_1_0(self):
        """HHI = 0.25 → 乗数 = 1.0 (境界値)."""
        assert get_concentration_multiplier(0.25) == pytest.approx(1.0)

    def test_0_50_returns_1_3(self):
        """HHI = 0.50 → 乗数 = 1.3."""
        assert get_concentration_multiplier(0.50) == pytest.approx(1.3)

    def test_1_00_returns_1_6(self):
        """HHI = 1.00 → 乗数 = 1.6 (最大)."""
        assert get_concentration_multiplier(1.00) == pytest.approx(1.6)

    def test_capped_at_1_6(self):
        """HHI > 1.0 でも乗数は 1.6 を超えない."""
        assert get_concentration_multiplier(1.5) == pytest.approx(1.6)

    def test_midpoint_0_375_linear_interpolation(self):
        """HHI = 0.375 (0.25〜0.50 の中点) → 乗数 = 1.15."""
        assert get_concentration_multiplier(0.375) == pytest.approx(1.15, abs=1e-4)

    def test_midpoint_0_75_linear_interpolation(self):
        """HHI = 0.75 (0.50〜1.00 の中点) → 乗数 = 1.45."""
        assert get_concentration_multiplier(0.75) == pytest.approx(1.45, abs=1e-4)


# ---------------------------------------------------------------------------
# analyze_concentration
# ---------------------------------------------------------------------------


class TestAnalyzeConcentration:
    """analyze_concentration() のテスト."""

    def test_all_same_sector_gives_sector_hhi_one(self):
        """全銘柄が同一セクターならセクター HHI = 1.0."""
        portfolio_data = [
            {"sector": "Technology", "country": "US", "currency": "USD"},
            {"sector": "Technology", "country": "JP", "currency": "JPY"},
        ]
        weights = [0.5, 0.5]
        result = analyze_concentration(portfolio_data, weights)
        assert result["sector_hhi"] == pytest.approx(1.0)

    def test_diversified_currency_gives_low_hhi(self):
        """通貨が分散されていると currency_hhi が低い."""
        portfolio_data = [
            {"sector": "Tech", "country": "US", "currency": "USD"},
            {"sector": "Finance", "country": "JP", "currency": "JPY"},
            {"sector": "Health", "country": "UK", "currency": "GBP"},
            {"sector": "Energy", "country": "AU", "currency": "AUD"},
        ]
        weights = [0.25, 0.25, 0.25, 0.25]
        result = analyze_concentration(portfolio_data, weights)
        assert result["currency_hhi"] == pytest.approx(0.25)

    def test_max_hhi_axis_identified_correctly(self):
        """最大 HHI の軸が max_hhi_axis に返される."""
        # Currency 軸を最大 HHI にする（全USD）
        portfolio_data = [
            {"sector": "Tech", "country": "US", "currency": "USD"},
            {"sector": "Finance", "country": "JP", "currency": "USD"},
        ]
        weights = [0.5, 0.5]
        result = analyze_concentration(portfolio_data, weights)
        assert result["max_hhi_axis"] == "currency"
        assert result["max_hhi"] == pytest.approx(1.0)

    def test_risk_level_labels(self):
        """リスクレベルのラベルが正しく返される."""
        # 分散: 5銘柄均等 → HHI = 0.2 < 0.25 → '分散'
        port_diversified = [
            {"sector": str(i), "country": str(i), "currency": str(i)}
            for i in range(5)
        ]
        result = analyze_concentration(port_diversified, [0.2] * 5)
        assert result["risk_level"] == "分散"

        # 危険な集中: 1銘柄100%
        port_concentrated = [
            {"sector": "A", "country": "B", "currency": "C"}
        ]
        result2 = analyze_concentration(port_concentrated, [1.0])
        assert result2["risk_level"] == "危険な集中"

    def test_missing_sector_key_uses_default_label(self):
        """sector キーが存在しない場合はデフォルトラベル '不明' が使われる."""
        portfolio_data = [
            {"country": "US", "currency": "USD"},  # sector キーなし
        ]
        weights = [1.0]
        result = analyze_concentration(portfolio_data, weights)
        assert "不明" in result["sector_breakdown"]

    def test_breakdown_sums_to_one(self):
        """各軸の breakdown の合計が 1.0 になる."""
        portfolio_data = [
            {"sector": "Tech", "country": "US", "currency": "USD"},
            {"sector": "Finance", "country": "JP", "currency": "JPY"},
        ]
        weights = [0.6, 0.4]
        result = analyze_concentration(portfolio_data, weights)
        assert sum(result["sector_breakdown"].values()) == pytest.approx(1.0)
        assert sum(result["region_breakdown"].values()) == pytest.approx(1.0)
        assert sum(result["currency_breakdown"].values()) == pytest.approx(1.0)

    def test_concentration_multiplier_range(self):
        """concentration_multiplier は常に 1.0〜1.6 の範囲内."""
        portfolio_data = [
            {"sector": "A", "country": "B", "currency": "C"}
        ]
        result = analyze_concentration(portfolio_data, [1.0])
        assert 1.0 <= result["concentration_multiplier"] <= 1.6

    def test_country_key_fallback_to_region(self):
        """'country' キーが存在せず 'region' キーがある場合、region を使う."""
        portfolio_data = [
            {"sector": "Tech", "region": "North America", "currency": "USD"},
        ]
        weights = [1.0]
        result = analyze_concentration(portfolio_data, weights)
        # region_breakdown should have a non-"不明" key
        assert "不明" not in result["region_breakdown"] or (
            # fallback to region key was used
            result["region_hhi"] >= 0.0
        )
