"""components/tab_insights のユニットテスト（実インサイト統合版）.

``render_insights_tab`` および各内部セクションヘルパーが、
``BehaviorInsight`` / ``PortfolioTimingInsight`` オブジェクトの
あり/なしの両方のケースで例外なく呼び出せることを確認する。
Streamlit の描画関数は ``unittest.mock.patch`` でスタブ化する。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- プロジェクトルートを sys.path に追加 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Streamlit をモックしてから tab_insights をインポートする
# ---------------------------------------------------------------------------

_st_mock = MagicMock()

# expander はコンテキストマネージャとして使われるため __enter__/__exit__ を設定
_st_mock.expander.return_value.__enter__ = MagicMock(return_value=_st_mock)
_st_mock.expander.return_value.__exit__ = MagicMock(return_value=False)


# st.columns(n) は n 個のコンテキストマネージャリストを返す必要がある。
# 引数の数に応じて正しいサイズのリストを返すよう side_effect で設定する。
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

from src.core.behavior.models import (
    BehaviorInsight,
    ConfidenceLevel,
    HoldingPeriodSummary,
    PortfolioTimingInsight,
    PortfolioTradeStats,
    PriceContext,
    StyleMetrics,
    TradeStats,
    TradeTimingResult,
    WinLossSummary,
)

# ---------------------------------------------------------------------------
# フィクスチャヘルパー
# ---------------------------------------------------------------------------


def _make_positions() -> list[dict]:
    """最小限の保有銘柄リストを生成する."""
    return [
        {
            "symbol": "VTI",
            "name": "Vanguard Total",
            "sector": "Diversified",
            "evaluation_jpy": 300_000.0,
            "currency": "USD",
            "pnl_pct": 15.0,
        },
        {
            "symbol": "7203.T",
            "name": "トヨタ",
            "sector": "Consumer Cyclical",
            "evaluation_jpy": 200_000.0,
            "currency": "JPY",
            "pnl_pct": -3.0,
        },
    ]


def _make_behavior_insight_empty() -> BehaviorInsight:
    """データなし状態の BehaviorInsight を生成する."""
    return BehaviorInsight.empty()


def _make_behavior_insight_with_data() -> BehaviorInsight:
    """一定量のデータがある BehaviorInsight を生成する."""
    ts = PortfolioTradeStats(
        symbols_traded=["VTI", "7203.T"],
        total_buy_count=10,
        total_sell_count=5,
        total_realized_pnl_jpy=50_000.0,
        overall_win_count=3,
        overall_loss_count=2,
        avg_hold_days=45.0,
        confidence=ConfidenceLevel.MEDIUM,
        by_symbol={
            "VTI": TradeStats(
                symbol="VTI",
                buy_count=6,
                sell_count=3,
                total_buy_jpy=600_000.0,
                total_sell_jpy=700_000.0,
                realized_pnl_jpy=45_000.0,
                win_count=2,
                loss_count=1,
                avg_hold_days=40.0,
                confidence=ConfidenceLevel.MEDIUM,
            )
        },
    )
    wl = WinLossSummary(
        win_count=3,
        loss_count=2,
        win_rate=0.6,
        avg_win_jpy=20_000.0,
        avg_loss_jpy=-5_000.0,
        gross_profit_jpy=60_000.0,
        gross_loss_jpy=-10_000.0,
        profit_factor=6.0,
        confidence=ConfidenceLevel.MEDIUM,
    )
    hp = HoldingPeriodSummary(
        total_closed=5,
        total_with_hold_data=5,
        min_days=10.0,
        max_days=90.0,
        median_days=45.0,
        p25_days=20.0,
        p75_days=70.0,
        short_term_count=1,
        medium_term_count=3,
        long_term_count=1,
        short_term_ratio=0.2,
        confidence=ConfidenceLevel.MEDIUM,
    )
    sm = StyleMetrics(
        trade_frequency="moderate",
        avg_position_size_jpy=60_000.0,
        concentration_score=0.4,
        holding_style="medium_term",
        confidence=ConfidenceLevel.MEDIUM,
    )
    return BehaviorInsight(
        trade_stats=ts,
        style_metrics=sm,
        holding_period=hp,
        win_loss=wl,
        confidence=ConfidenceLevel.MEDIUM,
    )


def _make_timing_insight_empty() -> PortfolioTimingInsight:
    """データなし状態の PortfolioTimingInsight を生成する."""
    return PortfolioTimingInsight.empty()


def _make_timing_insight_with_data() -> PortfolioTimingInsight:
    """一定量のデータがある PortfolioTimingInsight を生成する."""
    result = TradeTimingResult(
        symbol="VTI",
        trade_date="2024-03-15",
        trade_type="buy",
        trade_price=230.0,
        timing_score=72.5,
        price_context=PriceContext(
            sma_20=225.0,
            rsi_14=42.0,
            price_percentile=0.35,
            days_of_history=100,
        ),
        label="good",
        notes=["Price below SMA-20 — favorable buy entry."],
        confidence=ConfidenceLevel.MEDIUM,
    )
    return PortfolioTimingInsight(
        avg_buy_timing_score=72.5,
        avg_sell_timing_score=None,
        trade_results=[result],
        confidence=ConfidenceLevel.MEDIUM,
        notes=["Average buy timing score: 72/100."],
    )


# ---------------------------------------------------------------------------
# render_insights_tab — 公開 API
# ---------------------------------------------------------------------------


class TestRenderInsightsTab:
    """render_insights_tab の呼び出しテスト."""

    def test_renders_without_optional_data(self) -> None:
        """insight 引数なし・最小データで例外が発生しないこと."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=[],
                total_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

    def test_renders_with_empty_insights(self) -> None:
        """空の insight オブジェクトで例外が発生しないこと."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=_make_positions(),
                total_value=500_000.0,
                unrealized_pnl=50_000.0,
                realized_pnl=20_000.0,
                behavior_insight=_make_behavior_insight_empty(),
                timing_insight=_make_timing_insight_empty(),
            )

    def test_renders_with_full_insights(self) -> None:
        """データ有りの insight オブジェクトで例外が発生しないこと."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=_make_positions(),
                total_value=500_000.0,
                unrealized_pnl=50_000.0,
                realized_pnl=20_000.0,
                behavior_insight=_make_behavior_insight_with_data(),
                timing_insight=_make_timing_insight_with_data(),
            )

    def test_renders_with_negative_pnl(self) -> None:
        """損失ケースでも例外が発生しないこと."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=_make_positions(),
                total_value=400_000.0,
                unrealized_pnl=-30_000.0,
                realized_pnl=-10_000.0,
                behavior_insight=_make_behavior_insight_with_data(),
                timing_insight=_make_timing_insight_with_data(),
            )


# ---------------------------------------------------------------------------
# 内部セクションヘルパー — 境界値テスト
# ---------------------------------------------------------------------------


class TestTradeStatisticsSection:
    """_render_trade_statistics_section の境界値テスト."""

    def test_insufficient_data_shows_placeholder(self) -> None:
        """INSUFFICIENT confidence でプレースホルダーが描画されること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_trade_statistics_section(_make_behavior_insight_empty())

    def test_with_medium_confidence_data(self) -> None:
        """MEDIUM confidence でメトリクスが描画されること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_trade_statistics_section(_make_behavior_insight_with_data())

    def test_low_confidence_data(self) -> None:
        """LOW confidence でも例外なく描画できること."""
        bi = _make_behavior_insight_with_data()
        bi.confidence = ConfidenceLevel.LOW
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_trade_statistics_section(bi)

    def test_zero_profit_factor(self) -> None:
        """profit_factor が None のとき例外なく描画できること."""
        bi = _make_behavior_insight_with_data()
        bi.win_loss.profit_factor = None
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_trade_statistics_section(bi)

    def test_no_hold_data(self) -> None:
        """保有期間データがないとき例外なく描画できること."""
        bi = _make_behavior_insight_with_data()
        bi.holding_period = HoldingPeriodSummary()  # empty
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_trade_statistics_section(bi)


class TestTimingReviewSection:
    """_render_timing_review_section の境界値テスト."""

    def test_empty_timing_shows_placeholder(self) -> None:
        """トレード結果なしでプレースホルダーが描画されること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_timing_review_section(_make_timing_insight_empty())

    def test_with_trade_results(self) -> None:
        """トレード結果ありで例外なく描画できること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_timing_review_section(_make_timing_insight_with_data())

    def test_only_sell_scores(self) -> None:
        """avg_buy_timing_score が None でも例外なく描画できること."""
        ti = _make_timing_insight_with_data()
        ti.avg_buy_timing_score = None
        ti.avg_sell_timing_score = 65.0
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_timing_review_section(ti)

    def test_many_results_truncated(self) -> None:
        """25 件以上のトレードがあっても例外なく描画できること（最新20件に切り詰め）."""
        result_template = TradeTimingResult(
            symbol="VTI",
            trade_date="2024-01-01",
            trade_type="buy",
            trade_price=100.0,
            timing_score=50.0,
            price_context=PriceContext(),
            label="neutral",
        )
        results = [result_template] * 25
        ti = PortfolioTimingInsight(
            avg_buy_timing_score=50.0,
            trade_results=results,
            confidence=ConfidenceLevel.LOW,
        )
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_timing_review_section(ti)


class TestStyleProfileSection:
    """_render_style_profile_section の境界値テスト."""

    def test_no_data_shows_placeholder(self) -> None:
        """ポジションもデータもないとき例外なく描画できること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_style_profile_section([], _make_behavior_insight_empty())

    def test_positions_with_behavior_data(self) -> None:
        """ポジションあり・behavior_insight ありで例外なく描画できること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_style_profile_section(_make_positions(), _make_behavior_insight_with_data())

    def test_positions_with_style_profile(self) -> None:
        """StyleProfile ありで例外なく描画できること."""
        from src.core.behavior.models import StyleProfile

        profile = StyleProfile(
            adi_score=65.0,
            label="aggressive",
            cash_ratio=0.05,
            concentration_hhi=0.40,
            annual_volatility_pct=18.5,
            component_scores={"cash": 87.5, "concentration": 40.0, "holding": 50.0},
            confidence=ConfidenceLevel.MEDIUM,
        )
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_style_profile_section(
                _make_positions(),
                _make_behavior_insight_with_data(),
                style_profile=profile,
                style_biases=[],
            )

    def test_positions_with_style_profile_and_biases(self) -> None:
        """StyleProfile + BiasSignal ありで例外なく描画できること."""
        from src.core.behavior.models import BiasSignal, StyleProfile

        profile = StyleProfile(
            adi_score=72.0,
            label="aggressive",
            cash_ratio=0.45,
            concentration_hhi=0.60,
            confidence=ConfidenceLevel.LOW,
        )
        biases = [
            BiasSignal(
                bias_type="concentration",
                severity="high",
                title="集中リスク（高）",
                description="少数銘柄に集中しています。",
            ),
            BiasSignal(
                bias_type="cash_drag",
                severity="medium",
                title="現金比率がやや高め",
                description="現金比率 45%。",
            ),
        ]
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_style_profile_section(
                _make_positions(),
                _make_behavior_insight_with_data(),
                style_profile=profile,
                style_biases=biases,
            )

    def test_style_profile_defensive(self) -> None:
        """守り型プロファイルで例外なく描画できること."""
        from src.core.behavior.models import StyleProfile

        profile = StyleProfile(
            adi_score=25.0,
            label="defensive",
            cash_ratio=0.50,
            concentration_hhi=0.15,
            confidence=ConfidenceLevel.LOW,
        )
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_style_profile_section(
                _make_positions(),
                _make_behavior_insight_with_data(),
                style_profile=profile,
            )


class TestRenderInsightsTabWithStyleProfile:
    """render_insights_tab の style_profile/style_biases 引数テスト."""

    def test_renders_with_style_profile(self) -> None:
        """StyleProfile + biases ありで例外が発生しないこと."""
        from src.core.behavior.models import BiasSignal, StyleProfile

        profile = StyleProfile(
            adi_score=55.0,
            label="balanced",
            cash_ratio=0.10,
            concentration_hhi=0.30,
            confidence=ConfidenceLevel.MEDIUM,
        )
        biases = [
            BiasSignal(
                bias_type="home_bias",
                severity="medium",
                title="地域集中バイアス（中）",
                description="株式の 80% が米国に集中しています。",
            )
        ]
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=_make_positions(),
                total_value=500_000.0,
                unrealized_pnl=50_000.0,
                realized_pnl=20_000.0,
                behavior_insight=_make_behavior_insight_with_data(),
                timing_insight=_make_timing_insight_with_data(),
                style_profile=profile,
                style_biases=biases,
            )

    def test_renders_without_style_profile_arg(self) -> None:
        """style_profile/style_biases 省略で例外が発生しないこと（後方互換性）."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights.render_insights_tab(
                positions=_make_positions(),
                total_value=500_000.0,
                unrealized_pnl=50_000.0,
                realized_pnl=20_000.0,
                behavior_insight=_make_behavior_insight_with_data(),
                timing_insight=_make_timing_insight_with_data(),
            )


class TestLongHorizonSection:
    """_render_long_horizon_section の境界値テスト."""

    def test_zero_total_value_shows_placeholder(self) -> None:
        """total_value が 0 のとき例外なく描画できること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
            )

    def test_positive_total_value(self) -> None:
        """通常データで例外なく描画できること."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=500_000.0,
                realized_pnl=20_000.0,
                unrealized_pnl=50_000.0,
            )

    def test_loss_scenario(self) -> None:
        """損失ケースで例外なく描画できること（cost_basis ゼロ除算の回避確認）."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            tab_insights._render_long_horizon_section(
                total_value=400_000.0,
                realized_pnl=-5_000.0,
                unrealized_pnl=-20_000.0,
            )

    def test_pnl_equals_total_value_no_division_error(self) -> None:
        """cost_basis が 0 になる極端なケースでゼロ除算が起きないこと."""
        with patch.dict("sys.modules", {"streamlit": _st_mock}):
            # total_value == unrealized_pnl + realized_pnl → cost_basis = 0
            tab_insights._render_long_horizon_section(
                total_value=100_000.0,
                realized_pnl=60_000.0,
                unrealized_pnl=40_000.0,
            )
