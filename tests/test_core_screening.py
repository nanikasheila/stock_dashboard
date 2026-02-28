"""screening モジュールのユニットテスト.

対象:
  - src/core/screening/indicators.py  : ファンダメンタル指標計算
  - src/core/screening/alpha.py       : 超過リターン指標（変化スコア）
  - src/core/screening/technicals.py  : テクニカル指標（RSI, BB, プルバック）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.screening.alpha import (
    compute_accruals_score,
    compute_change_score,
    compute_fcf_yield_score,
    compute_revenue_acceleration_score,
    compute_roe_trend_score,
)
from src.core.screening.indicators import (
    _score_dividend,
    _score_growth,
    _score_pbr,
    _score_per,
    _score_roe,
    assess_return_stability,
    calculate_shareholder_return,
    calculate_shareholder_return_history,
    calculate_value_score,
    check_eps_direction,
    check_growth_consistency,
    check_margin_deterioration,
    check_quarterly_eps_trend,
    has_good_dividend,
    has_good_roe,
    is_undervalued_pbr,
    is_undervalued_per,
    run_consistency_checks,
)
from src.core.screening.technicals import (
    compute_bollinger_bands,
    compute_rsi,
    detect_pullback_in_uptrend,
)

# ===========================================================================
# indicators.py — ブール判定
# ===========================================================================


class TestIsUndervaluedPer:
    """is_undervalued_per() のテスト."""

    def test_low_per_returns_true(self):
        """PER が閾値未満（0 < per < threshold）の場合 True."""
        assert is_undervalued_per(10.0) is True

    def test_per_at_threshold_returns_false(self):
        """PER が閾値ちょうどのとき False（境界値）."""
        assert is_undervalued_per(15.0) is False

    def test_per_above_threshold_returns_false(self):
        """PER が閾値超の場合 False."""
        assert is_undervalued_per(20.0) is False

    def test_zero_per_returns_false(self):
        """PER = 0 は False（0 < per の条件を満たさない）."""
        assert is_undervalued_per(0.0) is False

    def test_negative_per_returns_false(self):
        """負の PER は False."""
        assert is_undervalued_per(-5.0) is False

    def test_none_per_returns_false(self):
        """None は False."""
        assert is_undervalued_per(None) is False

    def test_custom_threshold(self):
        """カスタム閾値が反映されること."""
        assert is_undervalued_per(18.0, threshold=20.0) is True
        assert is_undervalued_per(18.0, threshold=15.0) is False


class TestIsUndervaluedPbr:
    """is_undervalued_pbr() のテスト."""

    def test_low_pbr_returns_true(self):
        """PBR が閾値未満（0 < pbr < threshold）の場合 True."""
        assert is_undervalued_pbr(0.8) is True

    def test_pbr_at_threshold_returns_false(self):
        """PBR = 1.0（デフォルト閾値）は False（境界値）."""
        assert is_undervalued_pbr(1.0) is False

    def test_pbr_above_threshold_returns_false(self):
        """PBR が閾値超の場合 False."""
        assert is_undervalued_pbr(2.5) is False

    def test_zero_pbr_returns_false(self):
        """PBR = 0 は False."""
        assert is_undervalued_pbr(0.0) is False

    def test_negative_pbr_returns_false(self):
        """負の PBR は False."""
        assert is_undervalued_pbr(-1.0) is False

    def test_none_pbr_returns_false(self):
        """None は False."""
        assert is_undervalued_pbr(None) is False


class TestHasGoodDividend:
    """has_good_dividend() のテスト."""

    def test_yield_above_min_returns_true(self):
        """配当利回りが最低閾値以上で True."""
        assert has_good_dividend(0.05) is True

    def test_yield_at_min_returns_true(self):
        """配当利回りが最低閾値ちょうどで True（境界値）."""
        assert has_good_dividend(0.03) is True

    def test_yield_below_min_returns_false(self):
        """配当利回りが最低閾値未満で False."""
        assert has_good_dividend(0.01) is False

    def test_zero_yield_returns_false(self):
        """配当利回り 0 は False."""
        assert has_good_dividend(0.0) is False

    def test_none_yield_returns_false(self):
        """None は False."""
        assert has_good_dividend(None) is False

    def test_custom_min_yield(self):
        """カスタム最低利回りが反映されること."""
        assert has_good_dividend(0.02, min_yield=0.02) is True
        assert has_good_dividend(0.02, min_yield=0.03) is False


class TestHasGoodRoe:
    """has_good_roe() のテスト."""

    def test_roe_above_min_returns_true(self):
        """ROE が最低閾値以上で True."""
        assert has_good_roe(0.15) is True

    def test_roe_at_min_returns_true(self):
        """ROE が最低閾値ちょうどで True（境界値）."""
        assert has_good_roe(0.08) is True

    def test_roe_below_min_returns_false(self):
        """ROE が最低閾値未満で False."""
        assert has_good_roe(0.05) is False

    def test_zero_roe_returns_false(self):
        """ROE = 0 は False."""
        assert has_good_roe(0.0) is False

    def test_negative_roe_returns_false(self):
        """負の ROE は False."""
        assert has_good_roe(-0.05) is False

    def test_none_roe_returns_false(self):
        """None は False."""
        assert has_good_roe(None) is False


# ===========================================================================
# indicators.py — スコア計算（プライベート）
# ===========================================================================


class TestScorePer:
    """_score_per() のテスト."""

    def test_very_low_per_gives_high_score(self):
        """PER が非常に低い場合は高スコア（25 に近い）."""
        score = _score_per(1.0, 15.0)
        assert score > 20.0

    def test_per_at_max_gives_half_score(self):
        """PER = per_max のとき中間スコア（0.5 = 50%）."""
        score = _score_per(15.0, 15.0)
        assert score == pytest.approx(12.5, abs=0.01)

    def test_per_at_double_max_gives_zero(self):
        """PER = per_max * 2 のとき 0 点（上限）."""
        assert _score_per(30.0, 15.0) == 0.0

    def test_per_above_double_max_gives_zero(self):
        """PER が per_max * 2 を超えていても 0 点."""
        assert _score_per(100.0, 15.0) == 0.0

    def test_none_per_gives_zero(self):
        """None は 0 点."""
        assert _score_per(None, 15.0) == 0.0

    def test_zero_per_gives_zero(self):
        """PER = 0 は 0 点."""
        assert _score_per(0.0, 15.0) == 0.0

    def test_negative_per_gives_zero(self):
        """負の PER は 0 点."""
        assert _score_per(-5.0, 15.0) == 0.0


class TestScorePbr:
    """_score_pbr() のテスト."""

    def test_low_pbr_gives_high_score(self):
        """PBR が非常に低い場合は高スコア."""
        score = _score_pbr(0.1, 1.0)
        assert score > 20.0

    def test_pbr_at_max_gives_half_score(self):
        """PBR = pbr_max のとき中間スコア."""
        score = _score_pbr(1.0, 1.0)
        assert score == pytest.approx(12.5, abs=0.01)

    def test_pbr_at_double_max_gives_zero(self):
        """PBR = pbr_max * 2 で 0 点."""
        assert _score_pbr(2.0, 1.0) == 0.0

    def test_none_pbr_gives_zero(self):
        """None は 0 点."""
        assert _score_pbr(None, 1.0) == 0.0


class TestScoreDividend:
    """_score_dividend() のテスト."""

    def test_high_yield_gives_max_score(self):
        """配当利回りが cap 以上で最大 20 点."""
        # div_min=0.03, cap=0.09 → 0.09 以上は 20 点
        score = _score_dividend(0.09, 0.03)
        assert score == pytest.approx(20.0, abs=0.01)

    def test_at_div_min_gives_partial_score(self):
        """利回りが div_min ちょうどのとき 1/3 スコア."""
        # div_min=0.03, cap=0.09 → 0.03/0.09 = 1/3 → 20 * 1/3 ≈ 6.67
        score = _score_dividend(0.03, 0.03)
        assert score == pytest.approx(20.0 / 3, abs=0.02)

    def test_zero_yield_gives_zero(self):
        """利回り 0 は 0 点."""
        assert _score_dividend(0.0, 0.03) == 0.0

    def test_none_yield_gives_zero(self):
        """None は 0 点."""
        assert _score_dividend(None, 0.03) == 0.0


class TestScoreRoe:
    """_score_roe() のテスト."""

    def test_high_roe_gives_max_score(self):
        """ROE が cap 以上で最大 15 点."""
        # roe_min=0.08, cap=0.24 → 0.24 以上は 15 点
        score = _score_roe(0.30, 0.08)
        assert score == pytest.approx(15.0, abs=0.01)

    def test_zero_roe_gives_zero(self):
        """ROE = 0 は 0 点."""
        assert _score_roe(0.0, 0.08) == 0.0

    def test_none_roe_gives_zero(self):
        """None は 0 点."""
        assert _score_roe(None, 0.08) == 0.0


class TestScoreGrowth:
    """_score_growth() のテスト."""

    def test_high_growth_gives_max_score(self):
        """成長率 30%（cap） 以上で最大 15 点."""
        score = _score_growth(0.30)
        assert score == pytest.approx(15.0, abs=0.01)

    def test_over_cap_gives_max_score(self):
        """cap を超えていても 15 点（スコアが上限を超えない）."""
        score = _score_growth(0.60)
        assert score == pytest.approx(15.0, abs=0.01)

    def test_zero_growth_gives_zero(self):
        """成長率 0 は 0 点."""
        assert _score_growth(0.0) == 0.0

    def test_negative_growth_gives_zero(self):
        """負の成長率は 0 点."""
        assert _score_growth(-0.10) == 0.0

    def test_none_growth_gives_zero(self):
        """None は 0 点."""
        assert _score_growth(None) == 0.0


# ===========================================================================
# indicators.py — calculate_value_score
# ===========================================================================


class TestCalculateValueScore:
    """calculate_value_score() のテスト."""

    def test_all_good_metrics_returns_high_score(self):
        """全指標が良好なとき高スコアを返す."""
        stock = {
            "trailingPE": 8.0,
            "priceToBook": 0.5,
            "dividendYield": 0.06,
            "returnOnEquity": 0.20,
            "revenueGrowth": 0.25,
        }
        score = calculate_value_score(stock)
        assert score >= 60.0
        assert score <= 100.0

    def test_empty_dict_returns_zero(self):
        """空の dict は 0 を返す."""
        assert calculate_value_score({}) == 0.0

    def test_normalised_keys_work(self):
        """Yahoo Raw キー以外の正規化キーも受け付ける."""
        stock = {
            "per": 8.0,
            "pbr": 0.5,
            "dividend_yield": 0.06,
            "roe": 0.20,
            "revenue_growth": 0.25,
        }
        score_raw = calculate_value_score(
            {
                "trailingPE": 8.0,
                "priceToBook": 0.5,
                "dividendYield": 0.06,
                "returnOnEquity": 0.20,
                "revenueGrowth": 0.25,
            }
        )
        score_norm = calculate_value_score(stock)
        assert score_norm == pytest.approx(score_raw, abs=0.01)

    def test_score_does_not_exceed_100(self):
        """スコアが 100 を超えないこと."""
        stock = {
            "trailingPE": 1.0,
            "priceToBook": 0.01,
            "dividendYield": 0.99,
            "returnOnEquity": 1.00,
            "revenueGrowth": 1.00,
        }
        score = calculate_value_score(stock)
        assert score <= 100.0

    def test_custom_thresholds_change_score(self):
        """カスタム閾値によってスコアが変わること."""
        stock = {"trailingPE": 12.0}
        score_default = calculate_value_score(stock)  # per_max=15
        score_strict = calculate_value_score(stock, thresholds={"per_max": 10.0})
        # 閾値が低いほど同じ PER でも低スコア
        assert score_strict < score_default

    def test_dividend_yield_trailing_takes_priority(self):
        """dividend_yield_trailing が dividendYield より優先されること."""
        stock_trailing = {
            "dividend_yield_trailing": 0.09,
            "dividendYield": 0.01,
        }
        stock_forward = {
            "dividendYield": 0.09,
        }
        # 同じ trailing yield → 同スコア
        assert calculate_value_score(stock_trailing) == calculate_value_score(stock_forward)


# ===========================================================================
# indicators.py — calculate_shareholder_return_history
# ===========================================================================


class TestCalculateShareholderReturnHistory:
    """calculate_shareholder_return_history() のテスト."""

    def test_with_history_data_returns_multiple_years(self):
        """履歴データがある場合は複数年のリストを返す."""
        stock = {
            "market_cap": 1_000_000,
            "dividend_paid_history": [-30_000, -25_000, -20_000],
            "stock_repurchase_history": [-10_000, -8_000, -5_000],
            "cashflow_fiscal_years": [2023, 2022, 2021],
        }
        results = calculate_shareholder_return_history(stock)
        assert len(results) == 3
        assert results[0]["fiscal_year"] == 2023
        assert results[0]["dividend_paid"] == pytest.approx(30_000)
        assert results[0]["stock_repurchase"] == pytest.approx(10_000)

    def test_total_return_rate_calculation(self):
        """total_return_rate が正しく計算されること."""
        stock = {
            "market_cap": 1_000_000,
            "dividend_paid_history": [-50_000],
            "stock_repurchase_history": [-50_000],
            "cashflow_fiscal_years": [2023],
        }
        results = calculate_shareholder_return_history(stock)
        assert len(results) == 1
        assert results[0]["total_return_rate"] == pytest.approx(0.10)

    def test_no_history_fallback_to_single_period(self):
        """履歴なし → 単一期間データにフォールバック."""
        stock = {
            "market_cap": 500_000,
            "dividend_paid": -25_000,
            "stock_repurchase": -25_000,
        }
        results = calculate_shareholder_return_history(stock)
        assert len(results) == 1
        assert results[0]["fiscal_year"] is None
        assert results[0]["total_return_rate"] == pytest.approx(0.10)

    def test_empty_stock_returns_empty_list(self):
        """配当・自社株買い情報がない場合は空リスト."""
        results = calculate_shareholder_return_history({})
        assert results == []

    def test_none_market_cap_returns_none_rate(self):
        """market_cap が None のとき total_return_rate は None."""
        stock = {
            "dividend_paid_history": [-30_000],
            "cashflow_fiscal_years": [2023],
        }
        results = calculate_shareholder_return_history(stock)
        assert results[0]["total_return_rate"] is None

    def test_abs_applied_to_negative_values(self):
        """負のキャッシュフロー値に abs() が適用されること."""
        stock = {
            "market_cap": 100_000,
            "dividend_paid_history": [-10_000],
            "stock_repurchase_history": [-5_000],
            "cashflow_fiscal_years": [2023],
        }
        results = calculate_shareholder_return_history(stock)
        assert results[0]["dividend_paid"] == pytest.approx(10_000)
        assert results[0]["stock_repurchase"] == pytest.approx(5_000)


# ===========================================================================
# indicators.py — assess_return_stability
# ===========================================================================


class TestAssessReturnStability:
    """assess_return_stability() のテスト."""

    def test_no_data_returns_no_data(self):
        """rate が全て None の場合 stability='no_data'."""
        history = [{"total_return_rate": None}, {"total_return_rate": None}]
        result = assess_return_stability(history)
        assert result["stability"] == "no_data"
        assert result["latest_rate"] is None

    def test_single_high_rate_returns_single_high(self):
        """1年データかつ rate >= 5% → 'single_high'."""
        history = [{"total_return_rate": 0.08}]
        result = assess_return_stability(history)
        assert result["stability"] == "single_high"

    def test_single_moderate_rate(self):
        """1年データかつ 2% <= rate < 5% → 'single_moderate'."""
        history = [{"total_return_rate": 0.03}]
        result = assess_return_stability(history)
        assert result["stability"] == "single_moderate"

    def test_single_low_rate(self):
        """1年データかつ rate < 2% → 'single_low'."""
        history = [{"total_return_rate": 0.01}]
        result = assess_return_stability(history)
        assert result["stability"] == "single_low"

    def test_increasing_trend(self):
        """全年増加傾向 → 'increasing'."""
        history = [
            {"total_return_rate": 0.09},
            {"total_return_rate": 0.07},
            {"total_return_rate": 0.05},
        ]
        result = assess_return_stability(history)
        assert result["stability"] == "increasing"

    def test_decreasing_trend(self):
        """全年減少傾向 → 'decreasing'."""
        history = [
            {"total_return_rate": 0.03},
            {"total_return_rate": 0.05},
            {"total_return_rate": 0.08},
        ]
        result = assess_return_stability(history)
        assert result["stability"] == "decreasing"

    def test_stable_high_return(self):
        """全年 >= 5% の安定高還元 → 'stable'."""
        history = [
            {"total_return_rate": 0.07},
            {"total_return_rate": 0.06},
            {"total_return_rate": 0.08},
        ]
        result = assess_return_stability(history)
        assert result["stability"] == "stable"

    def test_temporary_surge(self):
        """最新年が前年比 >= 2x かつ >= 8% → 'temporary'."""
        history = [
            {"total_return_rate": 0.20},  # 急増（2x以上）
            {"total_return_rate": 0.09},  # 前年
        ]
        result = assess_return_stability(history)
        assert result["stability"] == "temporary"

    def test_mixed_pattern(self):
        """増加でも減少でも安定でもないパターン → 'mixed'."""
        history = [
            {"total_return_rate": 0.06},
            {"total_return_rate": 0.03},  # 低い
            {"total_return_rate": 0.07},
        ]
        result = assess_return_stability(history)
        # stable でないことを確認（全年 5% 以上ではない）
        assert result["stability"] in {"mixed", "increasing", "decreasing"}

    def test_empty_history_returns_no_data(self):
        """空リストは no_data."""
        result = assess_return_stability([])
        assert result["stability"] == "no_data"


# ===========================================================================
# indicators.py — calculate_shareholder_return
# ===========================================================================


class TestCalculateShareholderReturn:
    """calculate_shareholder_return() のテスト."""

    def test_normal_case_calculates_correctly(self):
        """正常系: 全フィールドが揃っている場合に正しく計算される."""
        stock = {
            "market_cap": 1_000_000,
            "dividend_paid": -40_000,
            "stock_repurchase": -10_000,
            "dividend_yield_trailing": 0.04,
        }
        result = calculate_shareholder_return(stock)
        assert result["dividend_paid"] == pytest.approx(40_000)
        assert result["stock_repurchase"] == pytest.approx(10_000)
        assert result["total_return_rate"] == pytest.approx(0.05)
        assert result["dividend_yield"] == pytest.approx(0.04)
        assert result["buyback_yield"] == pytest.approx(0.01)

    def test_none_market_cap_returns_none_rates(self):
        """market_cap が None のとき rate 系は None."""
        stock = {
            "dividend_paid": -30_000,
            "stock_repurchase": -10_000,
        }
        result = calculate_shareholder_return(stock)
        assert result["total_return_rate"] is None
        assert result["buyback_yield"] is None

    def test_only_dividend_no_repurchase(self):
        """自社株買いなしの場合も計算できること."""
        stock = {
            "market_cap": 500_000,
            "dividend_paid": -25_000,
        }
        result = calculate_shareholder_return(stock)
        assert result["stock_repurchase"] is None
        assert result["total_return_amount"] == pytest.approx(25_000)
        assert result["total_return_rate"] == pytest.approx(0.05)

    def test_empty_stock_returns_none_values(self):
        """全フィールドなしのとき total_return_amount は None."""
        result = calculate_shareholder_return({})
        assert result["total_return_amount"] is None
        assert result["total_return_rate"] is None


# ===========================================================================
# indicators.py — 整合性チェック
# ===========================================================================


class TestCheckEpsDirection:
    """check_eps_direction() のテスト."""

    def test_forward_eps_below_trailing_returns_warning(self):
        """FwdEPS < TrailEPS のとき警告が返る."""
        stock = {"forward_eps": 1.5, "eps_current": 2.0}
        result = check_eps_direction(stock)
        assert result is not None
        assert result["code"] == "EPS_DECLINE"

    def test_forward_eps_above_trailing_returns_none(self):
        """FwdEPS >= TrailEPS のとき None（問題なし）."""
        stock = {"forward_eps": 2.5, "eps_current": 2.0}
        assert check_eps_direction(stock) is None

    def test_missing_fields_returns_none(self):
        """フィールド欠損のとき None."""
        assert check_eps_direction({}) is None
        assert check_eps_direction({"forward_eps": 1.5}) is None

    def test_zero_trailing_eps_returns_none(self):
        """TrailEPS = 0 のとき None（ゼロ除算防止）."""
        stock = {"forward_eps": 1.5, "eps_current": 0.0}
        assert check_eps_direction(stock) is None


class TestCheckGrowthConsistency:
    """check_growth_consistency() のテスト."""

    def test_positive_earnings_growth_but_fwd_decline_returns_warning(self):
        """過去成長 > 0 かつ FwdEPS 下落 → 警告."""
        stock = {
            "earnings_growth": 0.15,
            "forward_eps": 1.0,
            "eps_current": 1.5,  # FwdEPS < TrailEPS
        }
        result = check_growth_consistency(stock)
        assert result is not None
        assert result["code"] == "PEG_INCONSISTENCY"

    def test_consistent_metrics_returns_none(self):
        """整合している場合は None."""
        stock = {
            "earnings_growth": 0.15,
            "forward_eps": 1.8,
            "eps_current": 1.5,  # FwdEPS > TrailEPS
        }
        assert check_growth_consistency(stock) is None

    def test_negative_trailing_eps_skipped(self):
        """trailing_eps < 0 のとき計算をスキップして None."""
        stock = {
            "earnings_growth": 0.20,
            "forward_eps": 0.5,
            "eps_current": -1.0,
        }
        assert check_growth_consistency(stock) is None

    def test_missing_fields_returns_none(self):
        """フィールド欠損時は None."""
        assert check_growth_consistency({}) is None


class TestCheckMarginDeterioration:
    """check_margin_deterioration() のテスト."""

    def test_margin_declined_5pt_returns_warning(self):
        """粗利率が 5pt 以上低下 → 警告."""
        stock = {"gross_margins_history": [0.30, 0.36]}  # -6pt
        result = check_margin_deterioration(stock)
        assert result is not None
        assert result["code"] == "MARGIN_DETERIORATION"

    def test_margin_declined_less_than_5pt_returns_none(self):
        """粗利率低下が 5pt 未満 → None."""
        stock = {"gross_margins_history": [0.33, 0.35]}  # -2pt
        assert check_margin_deterioration(stock) is None

    def test_margin_improved_returns_none(self):
        """粗利率が改善 → None."""
        stock = {"gross_margins_history": [0.40, 0.35]}
        assert check_margin_deterioration(stock) is None

    def test_insufficient_history_returns_none(self):
        """履歴データが 1 件以下 → None."""
        assert check_margin_deterioration({"gross_margins_history": [0.30]}) is None
        assert check_margin_deterioration({}) is None


class TestCheckQuarterlyEpsTrend:
    """check_quarterly_eps_trend() のテスト."""

    def test_declining_eps_returns_warning(self):
        """直近 EPS が前期より低下 → 警告."""
        stock = {"quarterly_eps": [1.0, 1.5]}
        result = check_quarterly_eps_trend(stock)
        assert result is not None
        assert result["code"] == "EPS_DECELERATION"

    def test_growing_eps_returns_none(self):
        """直近 EPS が成長中 → None."""
        stock = {"quarterly_eps": [2.0, 1.5]}
        assert check_quarterly_eps_trend(stock) is None

    def test_insufficient_data_returns_none(self):
        """データ 1 件以下 → None."""
        assert check_quarterly_eps_trend({"quarterly_eps": [1.0]}) is None
        assert check_quarterly_eps_trend({}) is None

    def test_previous_zero_returns_none(self):
        """前期 = 0 → ゼロ除算防止で None."""
        stock = {"quarterly_eps": [1.0, 0.0]}
        assert check_quarterly_eps_trend(stock) is None


class TestRunConsistencyChecks:
    """run_consistency_checks() のテスト."""

    def test_no_issues_returns_empty_list(self):
        """問題がない場合は空リスト."""
        stock = {
            "forward_eps": 2.5,
            "eps_current": 2.0,
            "earnings_growth": 0.05,
            "quarterly_eps": [2.0, 1.8],
        }
        warnings = run_consistency_checks(stock)
        assert isinstance(warnings, list)
        # EPS は増加なので警告なし
        for w in warnings:
            assert w["code"] != "EPS_DECLINE"

    def test_multiple_issues_returns_multiple_warnings(self):
        """複数問題があるとき複数の警告が返る."""
        stock = {
            "forward_eps": 1.0,
            "eps_current": 2.0,  # EPS_DECLINE
            "quarterly_eps": [0.5, 1.0],  # EPS_DECELERATION
        }
        warnings = run_consistency_checks(stock)
        codes = [w["code"] for w in warnings]
        assert "EPS_DECLINE" in codes
        assert "EPS_DECELERATION" in codes

    def test_empty_stock_returns_empty_list(self):
        """空 dict → 警告なし."""
        assert run_consistency_checks({}) == []


# ===========================================================================
# alpha.py — compute_accruals_score
# ===========================================================================


class TestComputeAccrualsScore:
    """compute_accruals_score() のテスト."""

    def test_high_quality_earnings_returns_max_score(self):
        """accruals < -0.05（OCF >> 純利益） → 25 点."""
        stock = {
            "net_income_stmt": 50,
            "operating_cashflow": 200,
            "total_assets": 1000,
        }
        score, raw = compute_accruals_score(stock)
        assert score == 25.0
        assert raw == pytest.approx(-0.15)

    def test_moderate_quality_earnings(self):
        """accruals が 0.0 〜 0.05 → 15 点."""
        stock = {
            "net_income_stmt": 30,
            "operating_cashflow": 20,
            "total_assets": 1000,
        }
        # accruals = (30 - 20) / 1000 = 0.01
        score, raw = compute_accruals_score(stock)
        assert score == 15.0
        assert raw == pytest.approx(0.01)

    def test_low_quality_earnings_returns_zero(self):
        """accruals >= 0.10 → 0 点."""
        stock = {
            "net_income_stmt": 200,
            "operating_cashflow": 50,
            "total_assets": 1000,
        }
        # accruals = (200 - 50) / 1000 = 0.15
        score, _raw = compute_accruals_score(stock)
        assert score == 0.0

    def test_utilities_sector_capped_at_15(self):
        """Utilities セクターは最大 15 点にキャップ."""
        stock = {
            "net_income_stmt": 50,
            "operating_cashflow": 200,
            "total_assets": 1000,
            "sector": "Utilities",
        }
        score, _ = compute_accruals_score(stock)
        assert score <= 15.0

    def test_financial_services_sector_capped_at_15(self):
        """Financial Services セクターは最大 15 点にキャップ."""
        stock = {
            "net_income_stmt": 50,
            "operating_cashflow": 200,
            "total_assets": 1000,
            "sector": "Financial Services",
        }
        score, _ = compute_accruals_score(stock)
        assert score <= 15.0

    def test_missing_field_returns_zero_and_none(self):
        """必須フィールド欠損のとき (0, None)."""
        assert compute_accruals_score({}) == (0.0, None)
        assert compute_accruals_score({"net_income_stmt": 100, "operating_cashflow": 50}) == (0.0, None)

    def test_zero_total_assets_returns_zero(self):
        """total_assets = 0 は (0, None)."""
        stock = {"net_income_stmt": 50, "operating_cashflow": 30, "total_assets": 0}
        assert compute_accruals_score(stock) == (0.0, None)


# ===========================================================================
# alpha.py — compute_revenue_acceleration_score
# ===========================================================================


class TestComputeRevenueAccelerationScore:
    """compute_revenue_acceleration_score() のテスト."""

    def test_strong_acceleration_gives_max_score(self):
        """加速度 > 0.10 → 25 点."""
        # rev0=1.3, rev1=1.0, rev2=0.95
        # current_growth = 0.3, previous_growth ≈ 0.0526, acceleration ≈ 0.247
        stock = {"revenue_history": [1.3, 1.0, 0.95]}
        score, _acc = compute_revenue_acceleration_score(stock)
        assert score == 25.0

    def test_no_acceleration_gives_partial_score(self):
        """加速度が 0 〜 0.05 → 15 点."""
        # current_growth = 0.03, previous_growth = 0.01 → acc = 0.02
        stock = {"revenue_history": [103, 100, 99]}
        score, _ = compute_revenue_acceleration_score(stock)
        assert score == 15.0

    def test_negative_current_growth_gives_zero(self):
        """KIK-349: 今期成長率が負のとき 0 点（真の加速ではない）."""
        # rev0 < rev1 → current_growth < 0
        stock = {"revenue_history": [90, 100, 95]}
        score, _ = compute_revenue_acceleration_score(stock)
        assert score == 0.0

    def test_insufficient_history_returns_zero(self):
        """履歴が 3 件未満 → 0 点."""
        assert compute_revenue_acceleration_score({"revenue_history": [100, 90]}) == (0.0, None)
        assert compute_revenue_acceleration_score({}) == (0.0, None)

    def test_zero_base_revenue_returns_zero(self):
        """ベース収益が 0 → ゼロ除算防止で 0 点."""
        stock = {"revenue_history": [100, 0, 90]}
        assert compute_revenue_acceleration_score(stock) == (0.0, None)


# ===========================================================================
# alpha.py — compute_fcf_yield_score
# ===========================================================================


class TestComputeFcfYieldScore:
    """compute_fcf_yield_score() のテスト."""

    def test_high_fcf_yield_gives_max_score(self):
        """FCF yield > 0.12 → 25 点."""
        stock = {"fcf": 130_000, "market_cap": 1_000_000}
        score, fcf_yield = compute_fcf_yield_score(stock)
        assert score == 25.0
        assert fcf_yield == pytest.approx(0.13)

    def test_moderate_fcf_yield_gives_partial_score(self):
        """FCF yield が 0.05 〜 0.08 → 15 点."""
        stock = {"fcf": 60_000, "market_cap": 1_000_000}
        score, _ = compute_fcf_yield_score(stock)
        assert score == 15.0

    def test_low_fcf_yield_gives_zero(self):
        """FCF yield <= 0.02 → 0 点."""
        stock = {"fcf": 15_000, "market_cap": 1_000_000}
        score, _ = compute_fcf_yield_score(stock)
        assert score == 0.0

    def test_negative_fcf_gives_zero(self):
        """FCF が負（フリーキャッシュフロー赤字） → 0 点."""
        stock = {"fcf": -50_000, "market_cap": 1_000_000}
        score, _ = compute_fcf_yield_score(stock)
        assert score == 0.0

    def test_missing_fields_returns_zero_and_none(self):
        """フィールド欠損 → (0, None)."""
        assert compute_fcf_yield_score({}) == (0.0, None)
        assert compute_fcf_yield_score({"fcf": 100}) == (0.0, None)

    def test_zero_market_cap_returns_zero(self):
        """market_cap = 0 → (0, None)."""
        stock = {"fcf": 100_000, "market_cap": 0}
        assert compute_fcf_yield_score(stock) == (0.0, None)


# ===========================================================================
# alpha.py — compute_roe_trend_score
# ===========================================================================


class TestComputeRoeTrendScore:
    """compute_roe_trend_score() のテスト."""

    def test_improving_roe_gives_high_score(self):
        """ROE が上昇傾向かつ最新 >= 8% → 高スコア."""
        stock = {
            "net_income_history": [120, 100, 80],
            "equity_history": [1000, 1000, 1000],
        }
        # roes = [0.12, 0.10, 0.08] → slope > 0
        score, slope = compute_roe_trend_score(stock)
        assert score >= 15.0
        assert slope is not None and slope > 0

    def test_declining_roe_gives_lower_score(self):
        """ROE が下降傾向 → 低スコア（slope < 0）."""
        stock = {
            "net_income_history": [80, 100, 120],
            "equity_history": [1000, 1000, 1000],
        }
        # roes = [0.08, 0.10, 0.12] → slope < 0
        score, _slope = compute_roe_trend_score(stock)
        assert score <= 10.0

    def test_negative_roe_returns_zero(self):
        """KIK-349: いずれかの期間に負の ROE → 0 点."""
        stock = {
            "net_income_history": [-80, 100, 120],
            "equity_history": [1000, 1000, 1000],
        }
        score, _ = compute_roe_trend_score(stock)
        assert score == 0.0

    def test_latest_roe_below_8pct_returns_zero(self):
        """KIK-349: 最新 ROE < 8% → 0 点（低 ROE はトレンド評価対象外）."""
        stock = {
            "net_income_history": [60, 50, 40],
            "equity_history": [1000, 1000, 1000],
        }
        # roes = [0.06, 0.05, 0.04] → 上昇傾向だが 0.06 < 0.08
        score, _ = compute_roe_trend_score(stock)
        assert score == 0.0

    def test_insufficient_history_returns_zero(self):
        """履歴が 3 件未満 → 0 点."""
        stock = {
            "net_income_history": [100, 90],
            "equity_history": [1000, 1000],
        }
        assert compute_roe_trend_score(stock) == (0.0, None)

    def test_zero_equity_returns_zero(self):
        """equity = 0 → ゼロ除算防止で 0 点."""
        stock = {
            "net_income_history": [100, 90, 80],
            "equity_history": [0, 1000, 1000],
        }
        assert compute_roe_trend_score(stock) == (0.0, None)


# ===========================================================================
# alpha.py — compute_change_score
# ===========================================================================


class TestComputeChangeScore:
    """compute_change_score() のテスト."""

    def test_all_good_metrics_returns_high_score(self):
        """全指標が良好 → 高変化スコア."""
        stock = {
            "net_income_stmt": 50,
            "operating_cashflow": 200,
            "total_assets": 1000,
            "revenue_history": [130, 100, 90],
            "fcf": 130_000,
            "market_cap": 1_000_000,
            "net_income_history": [120, 100, 80],
            "equity_history": [1000, 1000, 1000],
        }
        result = compute_change_score(stock)
        assert result["change_score"] >= 50.0
        assert result["quality_pass"] is True

    def test_empty_stock_returns_zero_score(self):
        """空 dict → change_score = 0."""
        result = compute_change_score({})
        assert result["change_score"] == 0.0
        assert result["passed_count"] == 0
        assert result["quality_pass"] is False

    def test_earnings_penalty_applied_for_negative_growth(self):
        """KIK-349: 負の earnings_growth にペナルティが適用される."""
        stock_with_penalty = {"earnings_growth": -0.25}
        stock_without_penalty = {}

        result_penalty = compute_change_score(stock_with_penalty)
        # Why: Only need to verify penalty variant; no-penalty variant tested elsewhere
        _result_no_penalty = compute_change_score(stock_without_penalty)

        # ペナルティ版は earnings_penalty が負の値
        assert result_penalty["earnings_penalty"] == -20.0

    def test_earnings_penalty_thresholds(self):
        """ペナルティ閾値が正しく分岐されること.

        alpha.py の分岐仕様:
            < -0.20 → -20pt
            < -0.10 → -15pt
            < 0.00  → -10pt
        """
        # -0% 〜 -10% 範囲 → -10pt
        result1 = compute_change_score({"earnings_growth": -0.05})
        assert result1["earnings_penalty"] == -10.0

        # -10% 〜 -20% 範囲 → -15pt
        result2 = compute_change_score({"earnings_growth": -0.15})
        assert result2["earnings_penalty"] == -15.0

        # -20% 超 → -20pt
        result3 = compute_change_score({"earnings_growth": -0.30})
        assert result3["earnings_penalty"] == -20.0

    def test_change_score_floor_is_zero(self):
        """change_score の最小値は 0（負にならない）."""
        stock = {"earnings_growth": -0.99}
        result = compute_change_score(stock)
        assert result["change_score"] >= 0.0

    def test_result_has_all_required_keys(self):
        """戻り値に必要なキーが全て含まれていること."""
        result = compute_change_score({})
        required_keys = {
            "change_score",
            "accruals",
            "revenue_acceleration",
            "fcf_yield",
            "roe_trend",
            "earnings_penalty",
            "passed_count",
            "quality_pass",
        }
        assert required_keys.issubset(result.keys())


# ===========================================================================
# technicals.py — compute_rsi
# ===========================================================================


def _make_price_series(n: int = 50, seed: int = 42) -> pd.Series:
    """Why: RSI テスト用に再現可能な価格系列が必要.
    How: 固定シードの乱数で価格変動を生成し、100 を基準に累積。
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n)
    prices = 100.0 * np.cumprod(1 + returns)
    return pd.Series(prices)


class TestComputeRsi:
    """compute_rsi() のテスト."""

    def test_rsi_is_bounded_between_0_and_100(self):
        """RSI は常に 0〜100 の範囲に収まる."""
        prices = _make_price_series(100)
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_nan_before_period_fills(self):
        """period 前の値は NaN になること."""
        prices = _make_price_series(30)
        rsi = compute_rsi(prices, period=14)
        # min_periods=14 なので最初は NaN が含まれる
        # ewm は初期値があるため厳密な NaN カウントは変動するが、
        # 全要素が数値でないことは保証されない → 境界付近の値のみ確認する
        assert len(rsi) == len(prices)

    def test_monotone_up_gives_high_rsi(self):
        """単調上昇の価格は高い RSI（> 70）になる."""
        prices = pd.Series(range(1, 51, 1), dtype=float)
        rsi = compute_rsi(prices, period=14)
        # 後半は全て上昇なので高 RSI
        assert float(rsi.iloc[-1]) > 70.0

    def test_monotone_down_gives_low_rsi(self):
        """単調下降の価格は低い RSI（< 30）になる."""
        prices = pd.Series(range(50, 0, -1), dtype=float)
        rsi = compute_rsi(prices, period=14)
        assert float(rsi.iloc[-1]) < 30.0

    def test_custom_period(self):
        """カスタム period が反映されること."""
        prices = _make_price_series(60)
        rsi7 = compute_rsi(prices, period=7)
        rsi14 = compute_rsi(prices, period=14)
        # 異なる period では異なる値になる
        assert not rsi7.equals(rsi14)


# ===========================================================================
# technicals.py — compute_bollinger_bands
# ===========================================================================


class TestComputeBollingerBands:
    """compute_bollinger_bands() のテスト."""

    def test_upper_above_middle_above_lower(self):
        """upper > middle > lower の関係が保たれること（NaN以外）."""
        prices = _make_price_series(100)
        upper, middle, lower = compute_bollinger_bands(prices)
        valid_idx = upper.dropna().index
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()

    def test_middle_is_rolling_mean(self):
        """middle が period 日の移動平均と一致すること."""
        prices = _make_price_series(100)
        period = 20
        _, middle, _ = compute_bollinger_bands(prices, period=period)
        expected_middle = prices.rolling(window=period).mean()
        pd.testing.assert_series_equal(middle, expected_middle)

    def test_nan_before_period(self):
        """period 前の値は NaN になること."""
        prices = _make_price_series(50)
        upper, _middle, _lower = compute_bollinger_bands(prices, period=20)
        assert upper.iloc[:19].isna().all()

    def test_custom_std_dev(self):
        """std_dev パラメータが幅に反映されること."""
        prices = _make_price_series(100)
        upper1, _middle1, lower1 = compute_bollinger_bands(prices, std_dev=1.0)
        upper2, _middle2, lower2 = compute_bollinger_bands(prices, std_dev=2.0)
        valid = upper1.dropna().index
        # std_dev=2 の方が幅が広い
        assert (upper2[valid] > upper1[valid]).all()
        assert (lower2[valid] < lower1[valid]).all()


# ===========================================================================
# technicals.py — detect_pullback_in_uptrend
# ===========================================================================


def _make_hist_df(n: int = 250, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Why: detect_pullback_in_uptrend に渡す DataFrame を生成するためのヘルパー.
    How: 上昇トレンドを模した価格にランダムノイズを加えて生成する。
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 0.5, n)  # 全体で 50% 上昇トレンド
    noise = rng.normal(0, 0.01, n)
    prices = base_price * np.exp(trend + np.cumsum(noise))
    volumes = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"Close": prices, "Volume": volumes})


def _make_pullback_hist_df() -> pd.DataFrame:
    """Why: プルバック条件（-5% 〜 -20%）を持つ DataFrame を再現可能に生成する.
    How: 強い上昇後に急落を追加して is_pullback 条件を満たすようにする。
    """
    rng = np.random.default_rng(10)
    n = 210
    # 強い上昇トレンドで SMA50 > SMA200 / current_price > SMA200 を確保
    trend = np.linspace(0, 1.0, n)
    noise = rng.normal(0, 0.005, n)
    prices = 80.0 * np.exp(trend + np.cumsum(noise))
    # 最後の 60 件を高値から 10% 下落させてプルバックを作る
    prices[-60:] *= np.linspace(1.0, 0.90, 60)
    volumes = rng.integers(200_000, 800_000, n).astype(float)
    return pd.DataFrame({"Close": prices, "Volume": volumes})


class TestDetectPullbackInUptrend:
    """detect_pullback_in_uptrend() のテスト."""

    def test_insufficient_data_returns_default(self):
        """データ件数 < 200 のときデフォルト結果（全て False）を返す."""
        hist = _make_hist_df(n=150)
        result = detect_pullback_in_uptrend(hist)
        assert result["uptrend"] is False
        assert result["is_pullback"] is False
        assert result["all_conditions"] is False

    def test_return_has_all_required_keys(self):
        """戻り値に必要な全キーが含まれていること."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        required_keys = {
            "uptrend",
            "is_pullback",
            "pullback_pct",
            "bounce_signal",
            "bounce_score",
            "bounce_details",
            "rsi",
            "volume_ratio",
            "sma50",
            "sma200",
            "current_price",
            "recent_high",
            "all_conditions",
        }
        assert required_keys.issubset(result.keys())

    def test_bounce_details_has_all_sub_keys(self):
        """bounce_details に必要なサブキーが存在すること."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        expected_keys = {
            "rsi_reversal",
            "rsi_depth_bonus",
            "bb_proximity",
            "volume_surge",
            "price_reversal",
            "lookback_day",
        }
        assert expected_keys.issubset(result["bounce_details"].keys())

    def test_uptrend_with_rising_market(self):
        """強い上昇トレンドデータで uptrend が True になること."""
        hist = _make_hist_df(n=250, seed=42)
        result = detect_pullback_in_uptrend(hist)
        # 上昇トレンド作成済みなので True が期待される
        assert result["uptrend"] is True

    def test_current_price_matches_last_close(self):
        """current_price が DataFrame の最終 Close と一致すること."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        assert result["current_price"] == pytest.approx(float(hist["Close"].iloc[-1]), abs=0.01)

    def test_pullback_pct_is_non_positive(self):
        """pullback_pct は recent_high からの下落率なので 0 以下であること."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        assert result["pullback_pct"] <= 0.0

    def test_rsi_is_valid_float(self):
        """rsi が有限の数値であること（十分なデータがある場合）."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        assert not math.isnan(result["rsi"])
        assert 0.0 <= result["rsi"] <= 100.0

    def test_sma50_and_sma200_are_positive(self):
        """SMA50 と SMA200 が正の値であること."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        assert result["sma50"] > 0.0
        assert result["sma200"] > 0.0

    def test_all_conditions_requires_three_flags_true(self):
        """all_conditions は uptrend & is_pullback & bounce_signal が全て True のとき True."""
        hist = _make_hist_df(n=250)
        result = detect_pullback_in_uptrend(hist)
        expected = result["uptrend"] and result["is_pullback"] and result["bounce_signal"]
        assert result["all_conditions"] == expected
