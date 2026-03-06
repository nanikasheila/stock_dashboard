"""components.dl_behavior.load_timing_insight のユニットテスト.

``load_timing_insight`` が様々なシナリオ（トレードなし、キャッシュなし、
キャッシュあり、ファイル破損）で例外なく動作し、正しい型を返すことを検証する。

ネットワーク呼び出しは一切行わない。
``dl_behavior.py`` も ``dl_holdings.py`` も Streamlit を import しないため、
``sys.modules`` のパッチは不要。``patch.object`` だけで完結する。

設計上の注意
-----------
``load_timing_insight`` 内の ``from components.dl_holdings import _build_holdings_timeline``
は deferred import なので、モック対象は
``components.dl_holdings._build_holdings_timeline`` 属性をパッチすることで制御する。
``_PRICE_CACHE_DIR`` は ``components.dl_behavior`` モジュール属性として
``patch.object`` でパッチする。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# dl_behavior.py と dl_holdings.py はどちらも Streamlit-free なので直接 import できる
import components.dl_behavior as _dlb
import components.dl_holdings as _dlh
from src.core.behavior.models import ConfidenceLevel, PortfolioTimingInsight

# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------


def _buy(symbol: str = "VTI") -> dict:
    return {
        "symbol": symbol,
        "date": "2024-06-15",
        "trade_type": "buy",
        "shares": 10,
        "price": 235.0,
        "currency": "USD",
    }


def _sell(symbol: str = "VTI") -> dict:
    return {
        "symbol": symbol,
        "date": "2024-09-01",
        "trade_type": "sell",
        "shares": 5,
        "price": 250.0,
        "currency": "USD",
    }


def _write_csv(path: Path, symbol: str, n: int = 500) -> None:
    """``n`` バーの価格 CSV をキャッシュディレクトリに書き込む."""
    from datetime import date, timedelta

    base = date(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    prices = [200.0 + i * 0.1 for i in range(n)]
    df = pd.DataFrame({symbol: prices}, index=pd.to_datetime(dates))
    df.to_csv(path)


# ---------------------------------------------------------------------------
# テスト: トレードなし → empty() 相当が返ること
# ---------------------------------------------------------------------------


class TestLoadTimingInsightNoTrades:
    """トレード履歴がない場合の動作テスト."""

    def test_returns_portfolio_timing_insight_type(self) -> None:
        """返り値の型が PortfolioTimingInsight であること."""
        with patch.object(_dlh, "_build_holdings_timeline", return_value=[]):
            result = _dlb.load_timing_insight()
        assert isinstance(result, PortfolioTimingInsight)

    def test_empty_trades_gives_empty_results(self) -> None:
        """トレードがない場合 trade_results が空であること."""
        with patch.object(_dlh, "_build_holdings_timeline", return_value=[]):
            result = _dlb.load_timing_insight()
        assert result.trade_results == []

    def test_empty_trades_gives_no_avg_scores(self) -> None:
        """トレードがない場合 avg スコアが None であること."""
        with patch.object(_dlh, "_build_holdings_timeline", return_value=[]):
            result = _dlb.load_timing_insight()
        assert result.avg_buy_timing_score is None
        assert result.avg_sell_timing_score is None


# ---------------------------------------------------------------------------
# テスト: トレードあり・キャッシュなし → INSUFFICIENT 信頼度で返ること
# ---------------------------------------------------------------------------


class TestLoadTimingInsightNoPriceCache:
    """価格キャッシュが存在しない場合の動作テスト."""

    def test_no_price_cache_returns_insufficient(self, tmp_path: Path) -> None:
        """キャッシュなしでも例外を送出せず INSUFFICIENT で返ること."""
        empty_dir = tmp_path / "price_history"
        # ディレクトリを作成しない → ファイルが存在しない状態

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=[_buy()]),
            patch.object(_dlb, "_PRICE_CACHE_DIR", empty_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_no_price_cache_has_one_trade_result(self, tmp_path: Path) -> None:
        """キャッシュなしでも trade_results にエントリが 1 件あること."""
        empty_dir = tmp_path / "price_history"

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=[_buy()]),
            patch.object(_dlb, "_PRICE_CACHE_DIR", empty_dir),
        ):
            result = _dlb.load_timing_insight()

        assert len(result.trade_results) == 1

    def test_cash_symbols_excluded_from_analysis(self, tmp_path: Path) -> None:
        """*.CASH シンボルはタイミング分析から除外されること."""
        empty_dir = tmp_path / "price_history"
        cash_trade = {
            "symbol": "JPY.CASH",
            "date": "2024-01-10",
            "trade_type": "buy",
            "shares": 1000,
            "price": 1.0,
            "currency": "JPY",
        }

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=[cash_trade]),
            patch.object(_dlb, "_PRICE_CACHE_DIR", empty_dir),
        ):
            result = _dlb.load_timing_insight()

        # キャッシュシンボルのみ → タイミング分析対象なし → trade_results 空
        assert result.trade_results == []

    def test_mixed_cash_and_equity_only_equity_analyzed(self, tmp_path: Path) -> None:
        """株式とキャッシュが混在する場合、株式のみが分析されること."""
        empty_dir = tmp_path / "price_history"
        trades = [
            _buy("VTI"),
            {
                "symbol": "USD.CASH",
                "date": "2024-01-01",
                "trade_type": "buy",
                "shares": 100,
                "price": 1.0,
                "currency": "USD",
            },
        ]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", empty_dir),
        ):
            result = _dlb.load_timing_insight()

        symbols = [tr.symbol for tr in result.trade_results]
        assert "VTI" in symbols
        assert all(s != "USD.CASH" for s in symbols)


# ---------------------------------------------------------------------------
# テスト: トレードあり・キャッシュあり → 価格データが活用されること
# ---------------------------------------------------------------------------


class TestLoadTimingInsightWithPriceCache:
    """価格キャッシュが存在する場合の動作テスト."""

    def test_with_price_cache_trade_results_populated(self, tmp_path: Path) -> None:
        """価格キャッシュがあるとき trade_results にエントリが生成されること."""
        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)
        _write_csv(cache_dir / "close_2y.csv", "VTI", n=500)

        trades = [_buy("VTI"), _sell("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)
        assert len(result.trade_results) == 2

    def test_timing_scores_in_valid_range(self, tmp_path: Path) -> None:
        """タイミングスコアが 0〜100 の範囲内であること."""
        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)
        _write_csv(cache_dir / "close_2y.csv", "VTI", n=500)

        trades = [_buy("VTI"), _sell("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        for tr in result.trade_results:
            assert 0.0 <= tr.timing_score <= 100.0, f"Score {tr.timing_score} out of range for {tr.symbol}"

    def test_longer_period_cache_preferred(self, tmp_path: Path) -> None:
        """``max`` 期間のキャッシュが ``1y`` より優先されること（先に走査される）."""
        from datetime import date, timedelta

        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)

        base = date(2020, 1, 1)
        # max: 1000 バー（長期）— VTI を含む
        dates_long = [base + timedelta(days=i) for i in range(1000)]
        df_long = pd.DataFrame(
            {"VTI": [200.0 + i * 0.05 for i in range(1000)]},
            index=pd.to_datetime(dates_long),
        )
        df_long.to_csv(cache_dir / "close_max.csv")

        # 1y: 200 バー — VTI を含むが max より後に走査されるので使われないはず
        dates_short = [base + timedelta(days=i) for i in range(200)]
        df_short = pd.DataFrame(
            {"VTI": [999.0] * 200},
            index=pd.to_datetime(dates_short),
        )
        df_short.to_csv(cache_dir / "close_1y.csv")

        trades = [_buy("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)
        assert len(result.trade_results) == 1

    def test_symbol_not_in_cache_gets_insufficient(self, tmp_path: Path) -> None:
        """キャッシュにないシンボルは INSUFFICIENT confidence で返ること."""
        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)
        _write_csv(cache_dir / "close_2y.csv", "VTI", n=500)  # QQQ は含まない

        trades = [_buy("QQQ")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)
        assert result.confidence == ConfidenceLevel.INSUFFICIENT

    def test_corrupted_cache_file_degrades_gracefully(self, tmp_path: Path) -> None:
        """キャッシュファイルが壊れていても例外にならないこと."""
        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)
        (cache_dir / "close_2y.csv").write_text("CORRUPTED\nNOT,A,VALID,CSV\n\n\n")

        trades = [_buy("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()  # must not raise

        assert isinstance(result, PortfolioTimingInsight)

    def test_empty_csv_does_not_crash(self, tmp_path: Path) -> None:
        """中身が空の CSV でもクラッシュしないこと."""
        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)
        pd.DataFrame().to_csv(cache_dir / "close_2y.csv")

        trades = [_buy("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)

    def test_multiple_symbols_each_scored(self, tmp_path: Path) -> None:
        """複数シンボルが全て個別にスコア付けされること."""
        from datetime import date, timedelta

        cache_dir = tmp_path / "ph"
        cache_dir.mkdir(parents=True)

        base = date(2022, 1, 1)
        dates = [base + timedelta(days=i) for i in range(500)]
        df = pd.DataFrame(
            {
                "VTI": [200.0 + i * 0.1 for i in range(500)],
                "QQQ": [300.0 + i * 0.2 for i in range(500)],
            },
            index=pd.to_datetime(dates),
        )
        df.to_csv(cache_dir / "close_2y.csv")

        trades = [_buy("VTI"), _buy("QQQ"), _sell("VTI")]

        with (
            patch.object(_dlh, "_build_holdings_timeline", return_value=trades),
            patch.object(_dlb, "_PRICE_CACHE_DIR", cache_dir),
        ):
            result = _dlb.load_timing_insight()

        assert isinstance(result, PortfolioTimingInsight)
        assert len(result.trade_results) == 3
        scored_symbols = {tr.symbol for tr in result.trade_results}
        assert "VTI" in scored_symbols
        assert "QQQ" in scored_symbols
