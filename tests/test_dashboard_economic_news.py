"""Tests for economic news & PF impact analysis in dashboard data_loader."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure dashboard components are importable
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from components.data_loader import (
    _apply_llm_results,
    _classify_news_impact,
    _estimate_portfolio_impact,
    fetch_economic_news,
)

# ---------------------------------------------------------------------------
# _classify_news_impact tests
# ---------------------------------------------------------------------------


class TestClassifyNewsImpact:
    """Test news title classification into impact categories."""

    def test_interest_rate_keywords(self):
        cats = _classify_news_impact("Fed raises interest rate by 25 basis points")
        cat_ids = {c["category"] for c in cats}
        assert "金利" in cat_ids

    def test_japanese_interest_rate(self):
        cats = _classify_news_impact("日銀が利上げを決定")
        cat_ids = {c["category"] for c in cats}
        assert "金利" in cat_ids

    def test_forex_keywords(self):
        cats = _classify_news_impact("Dollar strengthens against yen")
        cat_ids = {c["category"] for c in cats}
        assert "為替" in cat_ids

    def test_geopolitical_keywords(self):
        cats = _classify_news_impact("Trump announces new tariff on China imports")
        cat_ids = {c["category"] for c in cats}
        assert "地政学" in cat_ids

    def test_economy_keywords(self):
        cats = _classify_news_impact("US GDP growth slows, recession fears mount")
        cat_ids = {c["category"] for c in cats}
        assert "景気" in cat_ids

    def test_tech_keywords(self):
        cats = _classify_news_impact("AI semiconductor demand surges")
        cat_ids = {c["category"] for c in cats}
        assert "テクノロジー" in cat_ids

    def test_energy_keywords(self):
        cats = _classify_news_impact("OPEC cuts oil production, crude prices rise")
        cat_ids = {c["category"] for c in cats}
        assert "エネルギー" in cat_ids

    def test_no_category_match(self):
        cats = _classify_news_impact("Company XYZ releases quarterly earnings")
        assert cats == []

    def test_multiple_categories(self):
        cats = _classify_news_impact("Fed rate cut weakens dollar against yen")
        cat_ids = {c["category"] for c in cats}
        assert "金利" in cat_ids
        assert "為替" in cat_ids

    def test_empty_title(self):
        cats = _classify_news_impact("")
        assert cats == []


# ---------------------------------------------------------------------------
# _estimate_portfolio_impact tests
# ---------------------------------------------------------------------------


class TestEstimatePortfolioImpact:
    """Test portfolio impact estimation based on news categories."""

    @pytest.fixture
    def sample_positions(self):
        return [
            {"symbol": "7203.T", "sector": "Consumer Cyclical", "currency": "JPY"},
            {"symbol": "AAPL", "sector": "Technology", "currency": "USD"},
            {"symbol": "8306.T", "sector": "Financial Services", "currency": "JPY"},
            {"symbol": "XOM", "sector": "Energy", "currency": "USD"},
        ]

    def test_no_categories(self, sample_positions):
        result = _estimate_portfolio_impact([], sample_positions, {})
        assert result["impact_level"] == "none"
        assert result["affected_holdings"] == []

    def test_no_positions(self):
        cats = [{"category": "金利", "icon": "🏦", "label": "金利"}]
        result = _estimate_portfolio_impact(cats, [], {})
        assert result["impact_level"] == "none"

    def test_interest_rate_affects_financial(self, sample_positions):
        cats = [{"category": "金利", "icon": "🏦", "label": "金利"}]
        result = _estimate_portfolio_impact(cats, sample_positions, {})
        assert "8306.T" in result["affected_holdings"]

    def test_forex_affects_usd_holdings(self, sample_positions):
        cats = [{"category": "為替", "icon": "💱", "label": "為替"}]
        result = _estimate_portfolio_impact(cats, sample_positions, {})
        assert "AAPL" in result["affected_holdings"]
        assert "XOM" in result["affected_holdings"]

    def test_tech_affects_technology_sector(self, sample_positions):
        cats = [{"category": "テクノロジー", "icon": "💻", "label": "テック"}]
        result = _estimate_portfolio_impact(cats, sample_positions, {})
        assert "AAPL" in result["affected_holdings"]

    def test_energy_affects_energy_sector(self, sample_positions):
        cats = [{"category": "エネルギー", "icon": "⛽", "label": "エネルギー"}]
        result = _estimate_portfolio_impact(cats, sample_positions, {})
        assert "XOM" in result["affected_holdings"]

    def test_geopolitical_affects_industrial(self, sample_positions):
        cats = [{"category": "地政学", "icon": "🌍", "label": "地政学"}]
        result = _estimate_portfolio_impact(cats, sample_positions, {})
        # Consumer Cyclical contains "consumer" keyword match
        assert "7203.T" in result["affected_holdings"]

    def test_high_impact_when_many_affected(self):
        """When >= 50% of holdings affected, impact should be 'high'."""
        positions = [
            {"symbol": "AAPL", "sector": "Technology", "currency": "USD"},
            {"symbol": "MSFT", "sector": "Technology", "currency": "USD"},
        ]
        cats = [{"category": "テクノロジー", "icon": "💻", "label": "テック"}]
        result = _estimate_portfolio_impact(cats, positions, {})
        assert result["impact_level"] == "high"

    def test_cash_positions_excluded(self):
        positions = [
            {"symbol": "JPY_CASH", "sector": "Cash", "currency": "JPY"},
            {"symbol": "AAPL", "sector": "Technology", "currency": "USD"},
        ]
        cats = [{"category": "テクノロジー", "icon": "💻", "label": "テック"}]
        result = _estimate_portfolio_impact(cats, positions, {})
        assert result["impact_level"] == "high"  # 1/1 non-cash affected
        assert "JPY_CASH" not in result["affected_holdings"]

    def test_impact_level_medium(self):
        """When 20-50% affected → medium."""
        positions = [
            {"symbol": "AAPL", "sector": "Technology", "currency": "USD"},
            {"symbol": "JNJ", "sector": "Healthcare", "currency": "USD"},
            {"symbol": "PG", "sector": "Consumer Defensive", "currency": "USD"},
            {"symbol": "8306.T", "sector": "Financial Services", "currency": "JPY"},
        ]
        cats = [{"category": "テクノロジー", "icon": "💻", "label": "テック"}]
        result = _estimate_portfolio_impact(cats, positions, {})
        # 1/4 = 25% → medium
        assert result["impact_level"] == "medium"


# ---------------------------------------------------------------------------
# fetch_economic_news tests
# ---------------------------------------------------------------------------


class TestFetchEconomicNews:
    """Test the main fetch function with mocked Yahoo client."""

    @patch("components.data_loader.yahoo_client")
    def test_returns_deduplicated_news(self, mock_yc):
        """Same title from different tickers should be deduplicated."""
        mock_yc.get_stock_news.return_value = [
            {
                "title": "Markets Rally on Fed Decision",
                "publisher": "Reuters",
                "link": "https://example.com/1",
                "publish_time": "2026-02-22T10:00:00",
            },
        ]
        result = fetch_economic_news()
        # Even though we query multiple tickers, deduplication should work
        titles = [n["title"] for n in result]
        assert titles.count("Markets Rally on Fed Decision") == 1

    @patch("components.data_loader.yahoo_client")
    def test_empty_when_no_news(self, mock_yc):
        mock_yc.get_stock_news.return_value = []
        result = fetch_economic_news()
        assert result == []

    @patch("components.data_loader.yahoo_client")
    def test_includes_source_info(self, mock_yc):
        mock_yc.get_stock_news.return_value = [
            {"title": "Test News", "publisher": "AP", "link": "https://example.com", "publish_time": "2026-02-22"},
        ]
        result = fetch_economic_news()
        assert len(result) >= 1
        item = result[0]
        assert "source_ticker" in item
        assert "source_name" in item
        assert "categories" in item
        assert "portfolio_impact" in item

    @patch("components.data_loader.yahoo_client")
    def test_impact_analysis_with_positions(self, mock_yc):
        mock_yc.get_stock_news.return_value = [
            {"title": "Fed raises interest rate", "publisher": "Reuters", "link": "", "publish_time": "2026-02-22"},
        ]
        positions = [
            {"symbol": "8306.T", "sector": "Financial Services", "currency": "JPY"},
        ]
        result = fetch_economic_news(positions=positions)
        rate_news = [n for n in result if n["title"] == "Fed raises interest rate"]
        if rate_news:
            impact = rate_news[0]["portfolio_impact"]
            assert "8306.T" in impact["affected_holdings"]

    @patch("components.data_loader.yahoo_client")
    def test_handles_api_error_gracefully(self, mock_yc):
        mock_yc.get_stock_news.side_effect = Exception("API error")
        result = fetch_economic_news()
        assert result == []

    @patch("components.data_loader.yahoo_client")
    def test_sorted_by_impact(self, mock_yc):
        """High impact news should appear before low/none impact."""

        def mock_news(symbol, count=3):
            if symbol == "^GSPC":
                return [
                    {
                        "title": "Generic company report",
                        "publisher": "AP",
                        "link": "",
                        "publish_time": "2026-02-22T10:00",
                    },
                ]
            elif symbol == "^TNX":
                return [
                    {
                        "title": "Fed interest rate decision shocks markets",
                        "publisher": "WSJ",
                        "link": "",
                        "publish_time": "2026-02-22T09:00",
                    },
                ]
            return []

        mock_yc.get_stock_news.side_effect = mock_news
        positions = [
            {"symbol": "8306.T", "sector": "Financial Services", "currency": "JPY"},
        ]
        result = fetch_economic_news(positions=positions)
        if len(result) >= 2:
            # High-impact news should come first
            first_impact = result[0]["portfolio_impact"]["impact_level"]
            assert first_impact != "none" or all(n["portfolio_impact"]["impact_level"] == "none" for n in result)


# ---------------------------------------------------------------------------
# _apply_llm_results tests
# ---------------------------------------------------------------------------


class TestApplyLlmResults:
    """Test merging LLM analysis results into news items."""

    def test_apply_basic_results(self):
        news = [
            {
                "title": "Fed raises rates",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
        ]
        llm_results = [
            {
                "id": 0,
                "categories": [
                    {"category": "金利", "icon": "🏦", "label": "金利・金融政策"},
                ],
                "impact_level": "high",
                "affected_holdings": ["8306.T"],
                "reason": "金融セクター直撃",
            },
        ]
        _apply_llm_results(news, llm_results)
        assert news[0]["analysis_method"] == "llm"
        assert news[0]["portfolio_impact"]["impact_level"] == "high"
        assert "8306.T" in news[0]["portfolio_impact"]["affected_holdings"]
        assert news[0]["portfolio_impact"]["reason"] == "金融セクター直撃"

    def test_unmatched_ids_not_affected(self):
        news = [
            {
                "title": "Test",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
        ]
        llm_results = [
            {"id": 99, "categories": [], "impact_level": "high", "affected_holdings": [], "reason": ""},
        ]
        _apply_llm_results(news, llm_results)
        assert news[0]["analysis_method"] == "keyword"

    def test_invalid_impact_level_normalized(self):
        news = [
            {
                "title": "Test",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
        ]
        llm_results = [
            {"id": 0, "categories": [], "impact_level": "critical", "affected_holdings": [], "reason": ""},
        ]
        _apply_llm_results(news, llm_results)
        assert news[0]["portfolio_impact"]["impact_level"] == "none"
        assert news[0]["analysis_method"] == "llm"

    @patch("components.data_loader.yahoo_client")
    def test_fetch_with_llm_enabled_fallback(self, mock_yc):
        """When llm_enabled=True but copilot CLI unavailable, falls back to keyword."""
        mock_yc.get_stock_news.return_value = [
            {"title": "Fed raises interest rate", "publisher": "Reuters", "link": "", "publish_time": "2026-02-22"},
        ]
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            result = fetch_economic_news(llm_enabled=True)
        if result:
            assert result[0]["analysis_method"] == "keyword"

    @patch("components.data_loader.yahoo_client")
    def test_fetch_with_llm_disabled(self, mock_yc):
        """When llm_enabled=False, keyword analysis is used."""
        mock_yc.get_stock_news.return_value = [
            {
                "title": "Oil prices surge on OPEC cuts",
                "publisher": "Reuters",
                "link": "",
                "publish_time": "2026-02-22",
            },
        ]
        result = fetch_economic_news(llm_enabled=False)
        if result:
            assert result[0]["analysis_method"] == "keyword"
