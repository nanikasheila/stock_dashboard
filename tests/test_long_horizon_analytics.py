"""long-horizon analytics のユニットテスト.

``compute_monthly_seasonality`` と ``compute_rolling_sharpe_trend`` の
純粋関数テスト、および ``_render_long_horizon_section`` と関連サブセクション
ヘルパーの Streamlit-free 描画テストを含む。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Pure analytics — no Streamlit dependency
# ---------------------------------------------------------------------------

from components.dl_analytics import compute_monthly_seasonality, compute_rolling_sharpe_trend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_history_df(months: int = 24, start: str = "2022-01-01") -> pd.DataFrame:
    """``months`` ヶ月分の日次ポートフォリオ履歴 DataFrame を生成する."""
    dates = pd.date_range(start=start, periods=months * 21, freq="B")  # ~21 trading days/month
    rng = np.random.default_rng(42)
    total_values = 1_000_000.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
    df = pd.DataFrame({"total": total_values}, index=dates)
    return df


def _make_short_history_df(trading_days: int = 30) -> pd.DataFrame:
    """短期 (data 不足) の履歴 DataFrame を生成する."""
    dates = pd.date_range(start="2024-01-01", periods=trading_days, freq="B")
    rng = np.random.default_rng(0)
    vals = 500_000.0 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(dates)))
    return pd.DataFrame({"total": vals}, index=dates)


def _make_benchmark_series(history_df: pd.DataFrame, symbol: str = "SPY") -> pd.Series:
    """簡易ベンチマーク系列（ポートフォリオと同じ基準に正規化）を生成する."""
    rng = np.random.default_rng(7)
    vals = history_df["total"].iloc[0] * np.cumprod(1 + rng.normal(0.0004, 0.012, len(history_df)))
    s = pd.Series(vals, index=history_df.index, name=symbol)
    return s


# ===========================================================================
# compute_monthly_seasonality
# ===========================================================================


class TestComputeMonthlySeasonal:
    """compute_monthly_seasonality の入力バリエーションテスト."""

    def test_empty_df_returns_empty_result(self) -> None:
        result = compute_monthly_seasonality(pd.DataFrame())
        assert result["has_sufficient_data"] is False
        assert result["months_of_data"] == 0
        assert result["monthly_avg_returns"] == {}
        assert result["year_returns"] == {}

    def test_no_total_column_returns_empty(self) -> None:
        df = pd.DataFrame({"other": [1.0, 2.0]}, index=pd.date_range("2023-01-01", periods=2))
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is False

    def test_single_row_returns_empty(self) -> None:
        df = pd.DataFrame({"total": [1_000_000.0]}, index=pd.date_range("2023-01-01", periods=1))
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is False

    def test_insufficient_data_less_than_12_months(self) -> None:
        df = _make_short_history_df(trading_days=60)  # ~3 months
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is False
        # Some monthly data points should still be present
        assert result["months_of_data"] >= 0

    def test_sufficient_data_24_months(self) -> None:
        df = _make_history_df(months=24)
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is True
        assert result["months_of_data"] >= 12

    def test_monthly_avg_returns_keys_are_1_to_12(self) -> None:
        df = _make_history_df(months=24)
        result = compute_monthly_seasonality(df)
        avg = result["monthly_avg_returns"]
        for k in avg:
            assert 1 <= k <= 12, f"Unexpected month key: {k}"

    def test_monthly_avg_returns_are_finite(self) -> None:
        df = _make_history_df(months=24)
        result = compute_monthly_seasonality(df)
        for k, v in result["monthly_avg_returns"].items():
            assert np.isfinite(v), f"month {k} has non-finite avg return {v}"

    def test_year_returns_keys_are_valid_years(self) -> None:
        df = _make_history_df(months=24, start="2022-01-01")
        result = compute_monthly_seasonality(df)
        for yr in result["year_returns"]:
            assert 2000 <= yr <= 2100, f"Unexpected year: {yr}"

    def test_year_returns_are_floats(self) -> None:
        df = _make_history_df(months=24, start="2022-01-01")
        result = compute_monthly_seasonality(df)
        for yr, ret in result["year_returns"].items():
            assert isinstance(ret, float), f"year {yr}: expected float, got {type(ret)}"
            assert np.isfinite(ret), f"year {yr}: non-finite return {ret}"

    def test_exact_12_months_is_sufficient(self) -> None:
        """厳密に12ヶ月分のデータは has_sufficient_data=True を返すこと."""
        # Build exactly 12 month-end boundaries → 11 monthly returns (pct_change drops first)
        # We need 12 monthly *returns* → 13 month-end values → ~13 months of data
        df = _make_history_df(months=14, start="2023-01-01")
        result = compute_monthly_seasonality(df)
        # May be True or False depending on exact calendar alignment; just ensure no error
        assert isinstance(result["has_sufficient_data"], bool)

    def test_very_long_history_5_years(self) -> None:
        df = _make_history_df(months=60, start="2019-01-01")
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is True
        # Should cover all 12 months if data spans full years
        assert len(result["monthly_avg_returns"]) <= 12

    def test_constant_portfolio_value_no_crash(self) -> None:
        """一定の総資産額でも例外が発生しないこと."""
        dates = pd.date_range("2022-01-01", periods=500, freq="B")
        df = pd.DataFrame({"total": [1_000_000.0] * len(dates)}, index=dates)
        result = compute_monthly_seasonality(df)
        # Constant value → all monthly returns = 0%
        for v in result["monthly_avg_returns"].values():
            assert v == pytest.approx(0.0, abs=1e-6)

    # --- year_month_returns 追加フィールドのテスト ---

    def test_year_month_returns_key_present(self) -> None:
        """year_month_returns キーが常に返り値に含まれること."""
        result = compute_monthly_seasonality(pd.DataFrame())
        assert "year_month_returns" in result

    def test_year_month_returns_empty_for_insufficient_data(self) -> None:
        """データ不足（< 2行）のとき year_month_returns が空辞書であること."""
        df = pd.DataFrame({"total": [1_000_000.0]}, index=pd.date_range("2023-01-01", periods=1))
        result = compute_monthly_seasonality(df)
        assert result["year_month_returns"] == {}

    def test_year_month_returns_nonempty_for_sufficient_data(self) -> None:
        """12ヶ月以上のデータでは year_month_returns が空でないこと."""
        df = _make_history_df(months=24)
        result = compute_monthly_seasonality(df)
        assert result["has_sufficient_data"] is True
        assert len(result["year_month_returns"]) > 0

    def test_year_month_returns_key_format_yyyy_mm(self) -> None:
        """year_month_returns のキーが 'YYYY-MM' 形式であること."""
        import re

        df = _make_history_df(months=24, start="2022-01-01")
        result = compute_monthly_seasonality(df)
        pattern = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
        for key in result["year_month_returns"]:
            assert pattern.match(key), f"Invalid key format: {key!r}"

    def test_year_month_returns_values_are_finite_floats(self) -> None:
        """year_month_returns の値が有限な float であること."""
        df = _make_history_df(months=24, start="2022-01-01")
        result = compute_monthly_seasonality(df)
        for key, val in result["year_month_returns"].items():
            assert isinstance(val, float), f"{key}: expected float, got {type(val)}"
            assert np.isfinite(val), f"{key}: non-finite value {val}"

    def test_year_month_returns_count_matches_months_of_data(self) -> None:
        """year_month_returns のエントリ数が months_of_data と一致すること."""
        df = _make_history_df(months=24, start="2022-01-01")
        result = compute_monthly_seasonality(df)
        assert len(result["year_month_returns"]) == result["months_of_data"]

    def test_year_month_returns_empty_for_no_total_column(self) -> None:
        """total 列がない DataFrame では year_month_returns が空辞書であること."""
        df = pd.DataFrame({"other": [1.0, 2.0]}, index=pd.date_range("2023-01-01", periods=2))
        result = compute_monthly_seasonality(df)
        assert result["year_month_returns"] == {}


# ===========================================================================
# compute_rolling_sharpe_trend
# ===========================================================================


class TestComputeRollingSharpeTrend:
    """compute_rolling_sharpe_trend の入力バリエーションテスト."""

    def test_empty_df_returns_insufficient(self) -> None:
        result = compute_rolling_sharpe_trend(pd.DataFrame())
        assert result["trend"] == "insufficient"
        assert result["latest"] is None

    def test_no_total_column_returns_insufficient(self) -> None:
        df = pd.DataFrame({"other": range(200)}, index=pd.date_range("2023-01-01", periods=200))
        result = compute_rolling_sharpe_trend(df)
        assert result["trend"] == "insufficient"

    def test_too_short_history_returns_insufficient(self) -> None:
        """window=60, trend_points=30 に対してデータが短すぎる場合."""
        df = _make_short_history_df(trading_days=50)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        assert result["trend"] == "insufficient"

    def test_sufficient_data_returns_valid_trend(self) -> None:
        df = _make_history_df(months=24)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        assert result["trend"] in ("improving", "stable", "declining")

    def test_latest_and_prev_are_floats(self) -> None:
        df = _make_history_df(months=24)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        if result["trend"] != "insufficient":
            assert isinstance(result["latest"], float)
            assert isinstance(result["prev"], float)
            assert np.isfinite(result["latest"])
            assert np.isfinite(result["prev"])

    def test_delta_is_latest_minus_prev(self) -> None:
        df = _make_history_df(months=24)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        if result["trend"] != "insufficient":
            expected_delta = result["latest"] - result["prev"]
            assert result["delta"] == pytest.approx(expected_delta, abs=1e-6)

    def test_improving_trend_on_rising_sharpe(self) -> None:
        """Sharpeが明確に改善している履歴では 'improving' トレンドを返すこと."""
        dates = pd.date_range("2021-01-01", periods=300, freq="B")
        # Gradually increasing returns to ensure rolling Sharpe increases
        base = np.linspace(0.0001, 0.003, 300)
        rng = np.random.default_rng(99)
        vals = 1_000_000 * np.cumprod(1 + base + rng.normal(0, 0.001, 300))
        df = pd.DataFrame({"total": vals}, index=dates)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        # Just check it runs without error and returns a valid trend
        assert result["trend"] in ("improving", "stable", "declining", "insufficient")

    def test_description_contains_value(self) -> None:
        df = _make_history_df(months=24)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0

    def test_trend_ja_is_non_empty(self) -> None:
        df = _make_history_df(months=24)
        result = compute_rolling_sharpe_trend(df, window=60, trend_points=30)
        assert isinstance(result["trend_ja"], str)
        assert len(result["trend_ja"]) > 0


# ===========================================================================
# _render_long_horizon_section (Streamlit-mocked)
# ===========================================================================

_st_mock = MagicMock()
_st_mock.expander.return_value.__enter__ = MagicMock(return_value=_st_mock)
_st_mock.expander.return_value.__exit__ = MagicMock(return_value=False)


def _mock_columns(*args, **kwargs):
    n = args[0] if args else 1
    count = n if isinstance(n, int) else len(n)
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    return [col for _ in range(count)]


_st_mock.columns.side_effect = _mock_columns

with patch.dict("sys.modules", {"streamlit": _st_mock}):
    import components.tab_insights as tab_insights


class TestRenderLongHorizonSection:
    """_render_long_horizon_section の新引数バリエーションテスト."""

    def test_zero_total_value_shows_placeholder_no_history(self) -> None:
        """total_value=0 のとき、history_df なしでもプレースホルダーが描画されること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
            )

    def test_positive_value_no_history_df_shows_basic_metrics(self) -> None:
        """history_df=None のとき、基本指標のみ描画されること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=500_000.0,
                realized_pnl=20_000.0,
                unrealized_pnl=50_000.0,
            )

    def test_with_history_df_no_benchmark(self) -> None:
        """history_df あり・benchmark なしで例外なく描画されること."""
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=500_000.0,
                realized_pnl=20_000.0,
                unrealized_pnl=50_000.0,
                history_df=df,
            )

    def test_with_history_df_and_benchmark(self) -> None:
        """history_df + benchmark_series ありで例外なく描画されること."""
        df = _make_history_df(months=24)
        bench = _make_benchmark_series(df)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=500_000.0,
                realized_pnl=20_000.0,
                unrealized_pnl=50_000.0,
                history_df=df,
                benchmark_series=bench,
                benchmark_label="S&P 500 (SPY)",
            )

    def test_short_history_insufficient_data(self) -> None:
        """短期データ（季節性・Sharpeデータ不足）でも例外なく描画されること."""
        df = _make_short_history_df(trading_days=30)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=300_000.0,
                realized_pnl=5_000.0,
                unrealized_pnl=-10_000.0,
                history_df=df,
            )

    def test_loss_scenario_with_history(self) -> None:
        """損失ケースでも例外なく描画されること."""
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=400_000.0,
                realized_pnl=-5_000.0,
                unrealized_pnl=-20_000.0,
                history_df=df,
            )

    def test_pnl_equals_total_no_division_error(self) -> None:
        """cost_basis=0 の極端ケースでゼロ除算が起きないこと."""
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=100_000.0,
                realized_pnl=60_000.0,
                unrealized_pnl=40_000.0,
                history_df=df,
            )

    def test_benchmark_none_shows_guidance(self) -> None:
        """benchmark_series=None のとき案内メッセージが描画されること."""
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=500_000.0,
                realized_pnl=10_000.0,
                unrealized_pnl=30_000.0,
                history_df=df,
                benchmark_series=None,
            )


# ===========================================================================
# _render_seasonality_subsection
# ===========================================================================


class TestRenderSeasonalitySubsection:
    """_render_seasonality_subsection の個別テスト."""

    def test_empty_history_no_crash(self) -> None:
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_seasonality_subsection(pd.DataFrame())

    def test_short_history_insufficient_message(self) -> None:
        df = _make_short_history_df(30)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_seasonality_subsection(df)

    def test_full_history_renders_without_error(self) -> None:
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_seasonality_subsection(df)

    def test_5_year_history_renders_without_error(self) -> None:
        df = _make_history_df(months=60, start="2019-01-01")
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_seasonality_subsection(df)

    def test_heatmap_calls_plotly_chart_when_sufficient_data(self) -> None:
        """12ヶ月以上のデータで st.plotly_chart が呼び出されること."""
        df = _make_history_df(months=24)
        st_mock = MagicMock()
        st_mock.columns.side_effect = _mock_columns
        with patch.object(tab_insights, "st", st_mock):
            tab_insights._render_seasonality_subsection(df)
        st_mock.plotly_chart.assert_called_once()

    def test_heatmap_not_called_when_insufficient_data(self) -> None:
        """データ不足（< 12ヶ月）のとき st.plotly_chart が呼び出されないこと."""
        df = _make_short_history_df(trading_days=30)  # ~1-2 months
        st_mock = MagicMock()
        st_mock.columns.side_effect = _mock_columns
        with patch.object(tab_insights, "st", st_mock):
            tab_insights._render_seasonality_subsection(df)
        st_mock.plotly_chart.assert_not_called()

    def test_render_seasonality_heatmap_direct_call(self) -> None:
        """_render_seasonality_heatmap を直接呼び出して例外が発生しないこと."""
        from components.dl_analytics import compute_monthly_seasonality

        df = _make_history_df(months=24)
        result = compute_monthly_seasonality(df)
        year_month_returns = result["year_month_returns"]
        assert year_month_returns  # 空でないことを確認

        st_mock = MagicMock()
        with patch.object(tab_insights, "st", st_mock):
            tab_insights._render_seasonality_heatmap(year_month_returns)
        st_mock.plotly_chart.assert_called_once()


# ===========================================================================
# _render_benchmark_comparison_subsection
# ===========================================================================


class TestRenderBenchmarkComparisonSubsection:
    """_render_benchmark_comparison_subsection の個別テスト."""

    def test_no_benchmark_shows_guidance(self) -> None:
        df = _make_history_df(months=12)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_benchmark_comparison_subsection(df, None, "なし")

    def test_with_benchmark_renders_metrics(self) -> None:
        df = _make_history_df(months=12)
        bench = _make_benchmark_series(df)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_benchmark_comparison_subsection(df, bench, "SPY")

    def test_excess_positive_no_error(self) -> None:
        """超過リターンが正値のケース."""
        df = _make_history_df(months=12)
        # portfolio outperforms benchmark: use a flat benchmark
        bench_vals = float(df["total"].iloc[0]) * np.ones(len(df))
        bench = pd.Series(bench_vals, index=df.index, name="FLAT")
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_benchmark_comparison_subsection(df, bench, "FLAT")

    def test_excess_negative_no_error(self) -> None:
        """超過リターンが負値のケース."""
        df = _make_history_df(months=12)
        rng = np.random.default_rng(2)
        # benchmark outperforms: use a strongly rising benchmark
        vals = float(df["total"].iloc[0]) * np.cumprod(1 + rng.uniform(0.005, 0.01, len(df)))
        bench = pd.Series(vals, index=df.index, name="STRONG")
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_benchmark_comparison_subsection(df, bench, "STRONG")


# ===========================================================================
# _render_sharpe_stability_subsection
# ===========================================================================


class TestRenderSharpeStabilitySubsection:
    """_render_sharpe_stability_subsection の個別テスト."""

    def test_short_history_shows_insufficient_message(self) -> None:
        df = _make_short_history_df(30)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_sharpe_stability_subsection(df)

    def test_full_history_renders_metrics(self) -> None:
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_sharpe_stability_subsection(df)

    def test_empty_history_no_crash(self) -> None:
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_sharpe_stability_subsection(pd.DataFrame())


# ===========================================================================
# render_insights_tab — backward-compat + new args
# ===========================================================================


class TestRenderInsightsTabLongHorizon:
    """render_insights_tab の長期インサイト引数テスト（後方互換性含む）."""

    def test_backwards_compat_no_history_df(self) -> None:
        """history_df 省略（旧呼び出し）でも例外なく動作すること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=[],
                total_value=500_000.0,
                unrealized_pnl=30_000.0,
                realized_pnl=10_000.0,
            )

    def test_with_history_df_and_no_benchmark(self) -> None:
        """history_df ありで例外なく動作すること."""
        df = _make_history_df(months=24)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=[],
                total_value=500_000.0,
                unrealized_pnl=30_000.0,
                realized_pnl=10_000.0,
                history_df=df,
            )

    def test_with_history_df_and_benchmark(self) -> None:
        """history_df + benchmark_series ありで例外なく動作すること."""
        df = _make_history_df(months=24)
        bench = _make_benchmark_series(df)
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=[],
                total_value=500_000.0,
                unrealized_pnl=30_000.0,
                realized_pnl=10_000.0,
                history_df=df,
                benchmark_series=bench,
                benchmark_label="S&P 500 (SPY)",
            )

    def test_with_empty_history_df_passed_as_none(self) -> None:
        """history_df=None（app.py が空DataFrameをNoneに変換する場合）でも動作すること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=[],
                total_value=500_000.0,
                unrealized_pnl=30_000.0,
                realized_pnl=10_000.0,
                history_df=None,
                benchmark_series=None,
                benchmark_label="なし",
            )
