"""src/core の主要モジュールに対するユニットテスト.

対象:
  - src/core/models.py   : Position, ForecastResult, HealthResult,
                           RebalanceAction, YearlySnapshot, SimulationResult
  - src/core/common.py   : is_cash(), is_etf(), safe_float()
  - src/core/ticker_utils.py : infer_currency(), infer_country(), cash_currency()
  - src/core/value_trap.py   : detect_value_trap(), _finite_or_none()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.common import is_cash, is_etf, safe_float
from src.core.models import (
    ForecastResult,
    HealthResult,
    Position,
    RebalanceAction,
    SimulationResult,
    YearlySnapshot,
)
from src.core.ticker_utils import cash_currency, infer_country, infer_currency
from src.core.value_trap import _finite_or_none, detect_value_trap

# ---------------------------------------------------------------------------
# src/core/common.py
# ---------------------------------------------------------------------------


class TestIsCash:
    """is_cash() のテスト."""

    def test_jpy_cash_returns_true(self):
        """JPY.CASH は現金ポジションとして True を返す."""
        assert is_cash("JPY.CASH") is True

    def test_usd_cash_returns_true(self):
        """USD.CASH は現金ポジションとして True を返す."""
        assert is_cash("USD.CASH") is True

    def test_lowercase_cash_returns_true(self):
        """小文字の jpy.cash でも True を返す（大文字変換処理）."""
        assert is_cash("jpy.cash") is True

    def test_regular_ticker_returns_false(self):
        """通常株式ティッカーは False を返す."""
        assert is_cash("AAPL") is False

    def test_japanese_stock_returns_false(self):
        """日本株サフィックス付き (.T) は False を返す."""
        assert is_cash("7203.T") is False

    def test_empty_string_returns_false(self):
        """空文字列は False を返す."""
        assert is_cash("") is False

    def test_cash_without_currency_prefix_returns_true(self):
        """.CASH のみでも True を返す."""
        assert is_cash(".CASH") is True


class TestIsEtf:
    """is_etf() のテスト."""

    def test_quotetype_etf_returns_true(self):
        """quoteType が 'ETF' の場合は True を返す."""
        assert is_etf({"quoteType": "ETF"}) is True

    def test_quotetype_equity_with_sector_returns_false(self):
        """quoteType が EQUITY でセクター情報あり → False."""
        assert is_etf({"quoteType": "EQUITY", "info": {"sector": "Technology"}}) is False

    def test_no_fundamentals_at_all_returns_true(self):
        """セクター・財務データが全て欠落している場合は ETF 扱いで True."""
        assert is_etf({}) is True

    def test_has_sector_returns_false(self):
        """セクター情報が存在する場合は False."""
        assert is_etf({"info": {"sector": "Healthcare"}}) is False

    def test_has_net_income_returns_false(self):
        """net_income_stmt があれば False."""
        assert is_etf({"net_income_stmt": [100, 200]}) is False

    def test_has_operating_cashflow_returns_false(self):
        """operating_cashflow があれば False."""
        assert is_etf({"operating_cashflow": 500}) is False

    def test_has_revenue_history_returns_false(self):
        """revenue_history があれば False."""
        assert is_etf({"revenue_history": [1000, 2000]}) is False

    def test_quotetype_equity_no_fundamentals_returns_true(self):
        """quoteType が EQUITY でも財務データなしなら True（ETF 扱い）."""
        assert is_etf({"quoteType": "EQUITY"}) is True


class TestSafeFloat:
    """safe_float() のテスト."""

    def test_integer_input_converts_to_float(self):
        """整数入力は正常に float に変換される."""
        assert safe_float(42) == 42.0

    def test_string_number_converts_correctly(self):
        """数値文字列は float に変換される."""
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_default(self):
        """None は default 値を返す."""
        assert safe_float(None) == 0.0

    def test_none_with_custom_default(self):
        """None に対してカスタム default 値を返す."""
        assert safe_float(None, default=-1.0) == -1.0

    def test_nan_returns_default(self):
        """NaN は default 値を返す."""
        assert safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self):
        """正の無限大は default 値を返す."""
        assert safe_float(float("inf")) == 0.0

    def test_negative_inf_returns_default(self):
        """負の無限大は default 値を返す."""
        assert safe_float(float("-inf")) == 0.0

    def test_non_numeric_string_returns_default(self):
        """数値に変換できない文字列は default 値を返す."""
        assert safe_float("abc") == 0.0

    def test_zero_returns_zero(self):
        """0 は 0.0 を返す."""
        assert safe_float(0) == 0.0

    def test_negative_number_converts_correctly(self):
        """負の数は正常に変換される."""
        assert safe_float(-5.5) == pytest.approx(-5.5)

    def test_empty_string_returns_default(self):
        """空文字列は default 値を返す."""
        assert safe_float("") == 0.0


# ---------------------------------------------------------------------------
# src/core/ticker_utils.py
# ---------------------------------------------------------------------------


class TestCashCurrency:
    """cash_currency() のテスト."""

    def test_jpy_cash_returns_jpy(self):
        """JPY.CASH から JPY を抽出する."""
        assert cash_currency("JPY.CASH") == "JPY"

    def test_usd_cash_returns_usd(self):
        """USD.CASH から USD を抽出する."""
        assert cash_currency("USD.CASH") == "USD"

    def test_lowercase_input_normalizes_to_upper(self):
        """小文字入力は大文字に正規化される."""
        assert cash_currency("eur.cash") == "EUR"

    def test_sgd_cash_returns_sgd(self):
        """SGD.CASH から SGD を抽出する."""
        assert cash_currency("SGD.CASH") == "SGD"


class TestInferCurrency:
    """infer_currency() のテスト."""

    def test_japanese_stock_returns_jpy(self):
        """日本株 (.T suffix) は JPY を返す."""
        assert infer_currency("7203.T") == "JPY"

    def test_us_stock_no_suffix_returns_usd(self):
        """サフィックスなしは USD を返す."""
        assert infer_currency("AAPL") == "USD"

    def test_uk_stock_returns_gbp(self):
        """英国株 (.L suffix) は GBP を返す."""
        assert infer_currency("HSBA.L") == "GBP"

    def test_german_stock_returns_eur(self):
        """ドイツ株 (.DE suffix) は EUR を返す."""
        assert infer_currency("SAP.DE") == "EUR"

    def test_jpy_cash_returns_jpy(self):
        """JPY.CASH は JPY を返す."""
        assert infer_currency("JPY.CASH") == "JPY"

    def test_usd_cash_returns_usd(self):
        """USD.CASH は USD を返す."""
        assert infer_currency("USD.CASH") == "USD"

    def test_info_currency_takes_priority(self):
        """info 辞書に currency がある場合、サフィックス推論より優先される."""
        assert infer_currency("7203.T", info={"currency": "USD"}) == "USD"

    def test_info_without_currency_falls_back_to_suffix(self):
        """info に currency キーがない場合はサフィックス推論にフォールバック."""
        assert infer_currency("7203.T", info={}) == "JPY"

    def test_australian_stock_returns_aud(self):
        """オーストラリア株 (.AX suffix) は AUD を返す."""
        assert infer_currency("BHP.AX") == "AUD"

    def test_unknown_suffix_returns_usd(self):
        """未知のサフィックスは USD を返す（ドット有だがマッピング外）."""
        assert infer_currency("XYZ.ZZ") == "USD"

    def test_korean_stock_ks_returns_krw(self):
        """韓国株 (.KS suffix) は KRW を返す."""
        assert infer_currency("005930.KS") == "KRW"


class TestInferCountry:
    """infer_country() のテスト."""

    def test_japanese_stock_returns_japan(self):
        """日本株 (.T suffix) は Japan を返す."""
        assert infer_country("7203.T") == "Japan"

    def test_us_stock_no_suffix_returns_united_states(self):
        """サフィックスなしは United States を返す."""
        assert infer_country("MSFT") == "United States"

    def test_uk_stock_returns_united_kingdom(self):
        """英国株 (.L suffix) は United Kingdom を返す."""
        assert infer_country("BP.L") == "United Kingdom"

    def test_info_country_takes_priority(self):
        """info に country がある場合は優先される."""
        assert infer_country("7203.T", info={"country": "Germany"}) == "Germany"

    def test_info_region_fallback(self):
        """info に country がないが region がある場合は region を返す."""
        assert infer_country("AAPL", info={"region": "US"}) == "US"

    def test_jpy_cash_returns_japan(self):
        """JPY.CASH は Japan を返す."""
        assert infer_country("JPY.CASH") == "Japan"

    def test_usd_cash_returns_united_states(self):
        """USD.CASH は United States を返す."""
        assert infer_country("USD.CASH") == "United States"

    def test_unknown_suffix_returns_unknown(self):
        """未知のサフィックスは Unknown を返す."""
        assert infer_country("XYZ.ZZ") == "Unknown"

    def test_indian_stock_ns_returns_india(self):
        """インド株 (.NS suffix) は India を返す."""
        assert infer_country("RELIANCE.NS") == "India"

    def test_canadian_stock_returns_canada(self):
        """カナダ株 (.TO suffix) は Canada を返す."""
        assert infer_country("RY.TO") == "Canada"


# ---------------------------------------------------------------------------
# src/core/value_trap.py
# ---------------------------------------------------------------------------


class TestFiniteOrNone:
    """_finite_or_none() のテスト."""

    def test_normal_float_returns_value(self):
        """通常の float 値はそのまま返す."""
        assert _finite_or_none(3.14) == pytest.approx(3.14)

    def test_integer_returns_float(self):
        """整数入力は float として返す."""
        assert _finite_or_none(10) == 10.0

    def test_none_returns_none(self):
        """None は None を返す."""
        assert _finite_or_none(None) is None

    def test_nan_returns_none(self):
        """NaN は None を返す."""
        assert _finite_or_none(float("nan")) is None

    def test_positive_inf_returns_none(self):
        """正の無限大は None を返す."""
        assert _finite_or_none(float("inf")) is None

    def test_negative_inf_returns_none(self):
        """負の無限大は None を返す."""
        assert _finite_or_none(float("-inf")) is None

    def test_string_number_converts_and_returns(self):
        """数値文字列は float に変換して返す."""
        assert _finite_or_none("2.5") == pytest.approx(2.5)

    def test_non_numeric_string_returns_none(self):
        """変換不可能な文字列は None を返す."""
        assert _finite_or_none("abc") is None

    def test_zero_returns_zero(self):
        """0 は 0.0 を返す（None ではない）."""
        assert _finite_or_none(0) == 0.0

    def test_negative_value_returns_value(self):
        """負の値はそのまま返す."""
        assert _finite_or_none(-99.9) == pytest.approx(-99.9)


class TestDetectValueTrap:
    """detect_value_trap() のテスト."""

    def test_none_input_returns_no_trap(self):
        """None 入力は is_trap=False の結果を返す."""
        result = detect_value_trap(None)
        assert result["is_trap"] is False
        assert result["reasons"] == []

    def test_empty_dict_returns_no_trap(self):
        """空の辞書はトラップなしを返す."""
        result = detect_value_trap({})
        assert result["is_trap"] is False

    def test_condition_a_low_per_negative_eps_growth(self):
        """条件A: PER<8 かつ EPS成長率<0 でトラップ検出."""
        result = detect_value_trap({"per": 5.0, "eps_growth": -0.1})
        assert result["is_trap"] is True
        assert "低PERだが利益減少中" in result["reasons"]

    def test_condition_a_not_triggered_when_per_above_threshold(self):
        """条件A: PER>=8 なら EPS成長率が負でもトラップなし（本条件では）."""
        result = detect_value_trap({"per": 9.0, "eps_growth": -0.5})
        # 条件A は発動しない（条件B が発動するかは revenue_growth 次第）
        assert "低PERだが利益減少中" not in result["reasons"]

    def test_condition_b_low_per_revenue_decline(self):
        """条件B: PER<10 かつ 売上成長率<=-0.05 でトラップ検出."""
        result = detect_value_trap({"per": 7.0, "revenue_growth": -0.10})
        assert result["is_trap"] is True
        assert "低PER+売上減少トレンド" in result["reasons"]

    def test_condition_b_not_triggered_when_revenue_decline_small(self):
        """条件B: 売上減少が -0.05 未満ならトラップなし."""
        result = detect_value_trap({"per": 7.0, "revenue_growth": -0.04})
        assert "低PER+売上減少トレンド" not in result["reasons"]

    def test_condition_c_low_pbr_low_roe_negative_eps(self):
        """条件C: PBR<0.8 かつ ROE<0.05 かつ EPS成長率<0 でトラップ検出."""
        result = detect_value_trap({"pbr": 0.5, "roe": 0.03, "eps_growth": -0.2})
        assert result["is_trap"] is True
        assert "低PBRだがROE低下・利益減少" in result["reasons"]

    def test_condition_c_not_triggered_when_roe_sufficient(self):
        """条件C: ROE >= 0.05 ならトラップなし."""
        result = detect_value_trap({"pbr": 0.5, "roe": 0.06, "eps_growth": -0.2})
        assert "低PBRだがROE低下・利益減少" not in result["reasons"]

    def test_multiple_conditions_can_trigger_simultaneously(self):
        """複数条件が同時に満たされると理由が複数列挙される."""
        result = detect_value_trap(
            {
                "per": 5.0,
                "eps_growth": -0.2,
                "revenue_growth": -0.15,
                "pbr": 0.4,
                "roe": 0.02,
            }
        )
        assert result["is_trap"] is True
        assert len(result["reasons"]) >= 2

    def test_healthy_stock_returns_no_trap(self):
        """健全な指標の銘柄はトラップなしを返す."""
        result = detect_value_trap(
            {
                "per": 20.0,
                "pbr": 2.0,
                "roe": 0.15,
                "eps_growth": 0.10,
                "revenue_growth": 0.08,
            }
        )
        assert result["is_trap"] is False
        assert result["reasons"] == []

    def test_nan_metrics_do_not_trigger_conditions(self):
        """NaN 値の指標は条件を発動させない."""
        result = detect_value_trap({"per": float("nan"), "eps_growth": -0.5})
        assert "低PERだが利益減少中" not in result["reasons"]

    def test_per_exactly_at_boundary_8(self):
        """PER=8 は条件A の境界値（< 8 でないので発動しない）."""
        result = detect_value_trap({"per": 8.0, "eps_growth": -0.3})
        assert "低PERだが利益減少中" not in result["reasons"]

    def test_revenue_growth_exactly_at_boundary_minus_5pct(self):
        """revenue_growth=-0.05 は条件B の境界値（<= -0.05 なので発動する）."""
        result = detect_value_trap({"per": 7.0, "revenue_growth": -0.05})
        assert "低PER+売上減少トレンド" in result["reasons"]


# ---------------------------------------------------------------------------
# src/core/models.py — Position
# ---------------------------------------------------------------------------


class TestPosition:
    """Position データクラスのテスト."""

    def _make_position(self, **kwargs) -> Position:
        defaults = {
            "symbol": "7203.T",
            "shares": 100,
            "cost_price": 7000.0,
            "cost_currency": "JPY",
        }
        defaults.update(kwargs)
        return Position(**defaults)

    def test_required_fields_create_instance(self):
        """必須フィールドのみで Position を生成できる."""
        pos = self._make_position()
        assert pos.symbol == "7203.T"
        assert pos.shares == 100

    def test_optional_fields_have_defaults(self):
        """オプションフィールドのデフォルト値が正しい."""
        pos = self._make_position()
        assert pos.current_price == 0.0
        assert pos.value_jpy == 0.0
        assert pos.sector == ""
        assert pos.country == ""
        assert pos.memo == ""

    def test_is_cash_property_true_for_cash_symbol(self):
        """is_cash プロパティが現金シンボルに対して True を返す."""
        pos = self._make_position(symbol="JPY.CASH", cost_currency="JPY")
        assert pos.is_cash is True

    def test_is_cash_property_false_for_stock(self):
        """is_cash プロパティが株式シンボルに対して False を返す."""
        pos = self._make_position(symbol="AAPL", cost_currency="USD")
        assert pos.is_cash is False

    def test_to_dict_returns_all_fields(self):
        """to_dict() が全フィールドを含む辞書を返す."""
        pos = self._make_position(sector="Automotive", country="Japan")
        d = pos.to_dict()
        assert d["symbol"] == "7203.T"
        assert d["sector"] == "Automotive"
        assert d["country"] == "Japan"

    def test_from_dict_round_trip(self):
        """from_dict(to_dict()) でラウンドトリップが成立する."""
        pos = self._make_position(
            current_price=8000.0,
            value_jpy=800_000.0,
            sector="Technology",
        )
        pos2 = Position.from_dict(pos.to_dict())
        assert pos2.symbol == pos.symbol
        assert pos2.shares == pos.shares
        assert pos2.current_price == pos.current_price
        assert pos2.sector == pos.sector

    def test_from_dict_uses_evaluation_jpy_fallback(self):
        """from_dict() が evaluation_jpy キーにフォールバックできる."""
        d = {
            "symbol": "VTI",
            "shares": 10,
            "cost_price": 200.0,
            "cost_currency": "USD",
            "evaluation_jpy": 300_000.0,
        }
        pos = Position.from_dict(d)
        assert pos.value_jpy == 300_000.0

    def test_from_dict_missing_optional_fields_use_empty_string(self):
        """from_dict() でオプションフィールドが欠けていても空文字列になる."""
        pos = Position.from_dict({"symbol": "AAPL", "shares": 5, "cost_price": 100.0, "cost_currency": "USD"})
        assert pos.sector == ""
        assert pos.memo == ""


# ---------------------------------------------------------------------------
# src/core/models.py — ForecastResult
# ---------------------------------------------------------------------------


class TestForecastResult:
    """ForecastResult データクラスのテスト."""

    def test_required_fields_create_instance(self):
        """必須フィールドのみで ForecastResult を生成できる."""
        fr = ForecastResult(symbol="AAPL", method="analyst")
        assert fr.symbol == "AAPL"
        assert fr.method == "analyst"

    def test_optional_returns_default_to_none(self):
        """オプションの収益予測フィールドのデフォルトは None."""
        fr = ForecastResult(symbol="AAPL", method="no_data")
        assert fr.base is None
        assert fr.optimistic is None
        assert fr.pessimistic is None

    def test_to_dict_includes_none_values(self):
        """to_dict() は None 値も含んで返す."""
        fr = ForecastResult(symbol="VTI", method="historical", base=0.07)
        d = fr.to_dict()
        assert d["symbol"] == "VTI"
        assert d["base"] == pytest.approx(0.07)
        assert d["optimistic"] is None

    def test_from_dict_round_trip(self):
        """from_dict(to_dict()) でラウンドトリップが成立する."""
        fr = ForecastResult(symbol="IVV", method="analyst", base=0.08, optimistic=0.12, pessimistic=0.04)
        fr2 = ForecastResult.from_dict(fr.to_dict())
        assert fr2.symbol == fr.symbol
        assert fr2.method == fr.method
        assert fr2.base == pytest.approx(fr.base)

    def test_from_dict_missing_keys_use_defaults(self):
        """from_dict() でキーが欠落している場合はデフォルト値を使う."""
        fr = ForecastResult.from_dict({"symbol": "CASH"})
        assert fr.method == "no_data"
        assert fr.base is None


# ---------------------------------------------------------------------------
# src/core/models.py — HealthResult
# ---------------------------------------------------------------------------


class TestHealthResult:
    """HealthResult データクラスのテスト."""

    def test_defaults_are_empty(self):
        """デフォルトフィールドは空文字列・空リスト."""
        hr = HealthResult(symbol="7203.T")
        assert hr.trend == ""
        assert hr.quality_label == ""
        assert hr.alert_level == ""
        assert hr.reasons == []

    def test_to_dict_serializes_correctly(self):
        """to_dict() が正しくシリアライズする."""
        hr = HealthResult(symbol="AAPL", trend="上昇", alert_level="early_warning", reasons=["PER高騰"])
        d = hr.to_dict()
        assert d["trend"] == "上昇"
        assert d["alert_level"] == "early_warning"
        assert d["reasons"] == ["PER高騰"]

    def test_from_dict_parses_nested_alert(self):
        """from_dict() がネストされた alert 辞書を正しくパースする."""
        raw = {
            "symbol": "7203.T",
            "trend_health": {"trend": "下降"},
            "change_quality": {"quality_label": "複数悪化"},
            "alert": {"level": "caution", "reasons": ["ROE低下"]},
        }
        hr = HealthResult.from_dict(raw)
        assert hr.trend == "下降"
        assert hr.quality_label == "複数悪化"
        assert hr.alert_level == "caution"
        assert "ROE低下" in hr.reasons

    def test_from_dict_missing_nested_keys_use_defaults(self):
        """from_dict() でネストキー欠落時はデフォルト値."""
        hr = HealthResult.from_dict({"symbol": "VTI"})
        assert hr.trend == ""
        assert hr.alert_level == ""
        assert hr.reasons == []


# ---------------------------------------------------------------------------
# src/core/models.py — RebalanceAction
# ---------------------------------------------------------------------------


class TestRebalanceAction:
    """RebalanceAction データクラスのテスト."""

    def test_required_fields_create_instance(self):
        """必須フィールドのみで RebalanceAction を生成できる."""
        ra = RebalanceAction(action="sell", symbol="IXJ")
        assert ra.action == "sell"
        assert ra.symbol == "IXJ"

    def test_optional_defaults(self):
        """オプションフィールドのデフォルト値が正しい."""
        ra = RebalanceAction(action="buy", symbol="VTI")
        assert ra.ratio == 0.0
        assert ra.amount_jpy == 0.0
        assert ra.reason == ""
        assert ra.priority == 99

    def test_to_dict_serializes_all_fields(self):
        """to_dict() が全フィールドを含む辞書を返す."""
        ra = RebalanceAction(
            action="reduce",
            symbol="LIT",
            name="Lithium ETF",
            ratio=0.5,
            reason="過剰集中",
            priority=1,
        )
        d = ra.to_dict()
        assert d["action"] == "reduce"
        assert d["ratio"] == pytest.approx(0.5)
        assert d["priority"] == 1


# ---------------------------------------------------------------------------
# src/core/models.py — YearlySnapshot
# ---------------------------------------------------------------------------


class TestYearlySnapshot:
    """YearlySnapshot データクラスのテスト."""

    def test_create_and_to_dict(self):
        """YearlySnapshot を生成して to_dict() が正しく動作する."""
        snap = YearlySnapshot(
            year=2025,
            value=10_000_000.0,
            cumulative_input=8_000_000.0,
            capital_gain=1_500_000.0,
            cumulative_dividends=500_000.0,
        )
        d = snap.to_dict()
        assert d["year"] == 2025
        assert d["value"] == pytest.approx(10_000_000.0)
        assert d["cumulative_dividends"] == pytest.approx(500_000.0)


# ---------------------------------------------------------------------------
# src/core/models.py — SimulationResult
# ---------------------------------------------------------------------------


class TestSimulationResult:
    """SimulationResult データクラスのテスト."""

    def _make_snapshot(self, year: int = 2025) -> YearlySnapshot:
        return YearlySnapshot(
            year=year,
            value=10_000_000.0,
            cumulative_input=8_000_000.0,
            capital_gain=1_500_000.0,
            cumulative_dividends=500_000.0,
        )

    def test_empty_returns_empty_result(self):
        """empty() クラスメソッドが空の SimulationResult を返す."""
        result = SimulationResult.empty()
        assert result.scenarios == {}
        assert result.target is None
        assert result.required_monthly is None
        assert result.dividend_effect == 0.0

    def test_to_dict_serializes_scenarios(self):
        """to_dict() がシナリオ内の YearlySnapshot を辞書化する."""
        snap = self._make_snapshot()
        sim = SimulationResult(
            scenarios={"base": [snap]},
            target=20_000_000.0,
            target_year_base=2035.0,
            target_year_optimistic=2032.0,
            target_year_pessimistic=None,
            required_monthly=50_000.0,
            dividend_effect=500_000.0,
            dividend_effect_pct=0.05,
        )
        d = sim.to_dict()
        assert "base" in d["scenarios"]
        assert isinstance(d["scenarios"]["base"][0], dict)
        assert d["scenarios"]["base"][0]["year"] == 2025
        assert d["target"] == pytest.approx(20_000_000.0)

    def test_to_dict_includes_all_top_level_keys(self):
        """to_dict() が全トップレベルキーを含む."""
        result = SimulationResult.empty()
        d = result.to_dict()
        expected_keys = {
            "scenarios",
            "target",
            "target_year_base",
            "target_year_optimistic",
            "target_year_pessimistic",
            "required_monthly",
            "dividend_effect",
            "dividend_effect_pct",
            "years",
            "monthly_add",
            "reinvest_dividends",
            "current_value",
            "portfolio_return_base",
            "dividend_yield",
        }
        assert expected_keys.issubset(d.keys())

    def test_default_fields_on_empty(self):
        """empty() のデフォルトフィールド値が正しい."""
        result = SimulationResult.empty()
        assert result.years == 0
        assert result.monthly_add == 0.0
        assert result.reinvest_dividends is True
        assert result.current_value == 0.0
        assert result.portfolio_return_base is None
