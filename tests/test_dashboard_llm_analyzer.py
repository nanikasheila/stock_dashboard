"""Tests for LLM-based news analysis (llm_analyzer module)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure dashboard components are importable
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from components.llm_analyzer import (
    AVAILABLE_MODELS,
    CACHE_TTL_OPTIONS,
    DEFAULT_CACHE_TTL_SEC,
    _build_analysis_prompt,
    _build_health_summary_prompt,
    _build_portfolio_summary,
    # Unified analysis (KIK: 3→1 LLM session)
    _build_unified_prompt,
    _compute_health_hash,
    _compute_news_hash,
    _compute_unified_hash,
    _extract_json_text,
    _parse_health_summary_response,
    _parse_response,
    _parse_summary_response,
    _parse_unified_response,
    analyze_news_batch,
    apply_news_analysis,
    clear_cache,
    clear_health_summary_cache,
    clear_insights_cache,
    clear_summary_cache,
    clear_unified_cache,
    generate_attribution_summary,
    generate_health_summary,
    generate_insights,
    generate_news_summary,
    get_cache_info,
    get_health_summary_cache_info,
    get_summary_cache_info,
    get_unified_cache_info,
    is_available,
    run_unified_analysis,
)

# ---------------------------------------------------------------------------
# is_available tests (delegated to copilot_client)
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_when_copilot_found_by_which(self):
        with (
            patch("components.copilot_client._SDK_AVAILABLE", True),
            patch("components.copilot_client.shutil.which", return_value="/usr/bin/copilot"),
        ):
            assert is_available() is True

    def test_available_when_copilot_found_by_subprocess(self):
        """shutil.which が見つけられなくても subprocess で検出できる."""
        with (
            patch("components.copilot_client._SDK_AVAILABLE", True),
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            assert is_available() is True

    def test_unavailable_when_copilot_not_found(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            assert is_available() is False


# ---------------------------------------------------------------------------
# _build_portfolio_summary tests
# ---------------------------------------------------------------------------


class TestBuildPortfolioSummary:
    def test_basic_summary(self):
        positions = [
            {"symbol": "7203.T", "sector": "Consumer Cyclical", "currency": "JPY", "weight_pct": 25.0},
            {"symbol": "AAPL", "sector": "Technology", "currency": "USD", "weight_pct": 30.0},
        ]
        result = _build_portfolio_summary(positions)
        assert "7203.T" in result
        assert "AAPL" in result
        assert "Technology" in result

    def test_excludes_cash(self):
        positions = [
            {"symbol": "CASH_JPY", "sector": "Cash", "currency": "JPY", "weight_pct": 10.0},
            {"symbol": "7203.T", "sector": "Consumer Cyclical", "currency": "JPY", "weight_pct": 90.0},
        ]
        result = _build_portfolio_summary(positions)
        assert "CASH" not in result
        assert "7203.T" in result

    def test_empty_positions(self):
        result = _build_portfolio_summary([])
        assert "保有銘柄なし" in result


# ---------------------------------------------------------------------------
# _build_analysis_prompt tests
# ---------------------------------------------------------------------------


class TestBuildAnalysisPrompt:
    def test_includes_news_and_portfolio(self):
        news_list = [{"id": 0, "title": "Fed raises rates", "publisher": "Reuters", "source": "S&P 500"}]
        pf_summary = "- 7203.T: セクター=Consumer Cyclical, 通貨=JPY, 比率=50.0%"
        prompt = _build_analysis_prompt(news_list, pf_summary)
        assert "Fed raises rates" in prompt
        assert "7203.T" in prompt
        assert "Consumer Cyclical" in prompt

    def test_includes_category_definitions(self):
        prompt = _build_analysis_prompt([], "（保有銘柄なし）")
        assert "金利" in prompt
        assert "為替" in prompt
        assert "地政学" in prompt


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_plain_json_array(self):
        raw = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["金利"],
                    "impact_level": "medium",
                    "affected_holdings": ["7203.T"],
                    "reason": "テスト理由",
                },
            ]
        )
        result = _parse_response(raw, 1)
        assert result is not None
        assert len(result) == 1
        assert result[0]["impact_level"] == "medium"
        assert result[0]["affected_holdings"] == ["7203.T"]
        assert result[0]["reason"] == "テスト理由"

    def test_json_in_code_block(self):
        raw = '```json\n[{"id": 0, "categories": ["景気"], "impact_level": "low", "affected_holdings": [], "reason": ""}]\n```'
        result = _parse_response(raw, 1)
        assert result is not None
        assert len(result) == 1
        assert result[0]["categories"][0]["category"] == "景気"

    def test_json_in_generic_code_block(self):
        raw = '```\n[{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]\n```'
        result = _parse_response(raw, 1)
        assert result is not None
        assert len(result) == 1

    def test_category_string_to_dict_conversion(self):
        raw = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["金利", "為替"],
                    "impact_level": "high",
                    "affected_holdings": [],
                    "reason": "",
                },
            ]
        )
        result = _parse_response(raw, 1)
        assert result is not None
        cats = result[0]["categories"]
        assert len(cats) == 2
        assert cats[0]["icon"] == "🏦"
        assert cats[0]["label"] == "金利・金融政策"
        assert cats[1]["icon"] == "💱"

    def test_category_dict_format(self):
        """LLM may return categories as dicts directly."""
        raw = json.dumps(
            [
                {
                    "id": 0,
                    "categories": [{"category": "テクノロジー", "icon": "💻", "label": "テク"}],
                    "impact_level": "low",
                    "affected_holdings": [],
                    "reason": "",
                },
            ]
        )
        result = _parse_response(raw, 1)
        assert result is not None
        cats = result[0]["categories"]
        assert len(cats) == 1
        assert cats[0]["category"] == "テクノロジー"
        # icon/label are normalized from our mapping, not LLM's
        assert cats[0]["icon"] == "💻"
        assert cats[0]["label"] == "テクノロジー"

    def test_unknown_category_filtered(self):
        raw = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["不明カテゴリ", "金利"],
                    "impact_level": "low",
                    "affected_holdings": [],
                    "reason": "",
                },
            ]
        )
        result = _parse_response(raw, 1)
        assert result is not None
        cats = result[0]["categories"]
        # Only "金利" should remain, not "不明カテゴリ"
        assert len(cats) == 1
        assert cats[0]["category"] == "金利"

    def test_invalid_json_returns_none(self):
        result = _parse_response("this is not json", 1)
        assert result is None

    def test_empty_response(self):
        result = _parse_response("", 0)
        assert result is None

    def test_json_with_prefix_text(self):
        raw = 'Here is the analysis:\n[{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]'
        result = _parse_response(raw, 1)
        assert result is not None
        assert len(result) == 1

    def test_invalid_impact_level_preserved(self):
        """Unknown impact_level is kept as-is in _parse_response (validation happens downstream)."""
        raw = json.dumps(
            [
                {"id": 0, "categories": [], "impact_level": "unknown", "affected_holdings": [], "reason": ""},
            ]
        )
        result = _parse_response(raw, 1)
        assert result is not None
        assert result[0]["impact_level"] == "unknown"

    def test_multiple_items(self):
        raw = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["金利"],
                    "impact_level": "high",
                    "affected_holdings": ["8306.T"],
                    "reason": "金融セクター影響",
                },
                {
                    "id": 1,
                    "categories": ["テクノロジー"],
                    "impact_level": "medium",
                    "affected_holdings": ["AAPL"],
                    "reason": "テック関連",
                },
            ]
        )
        result = _parse_response(raw, 2)
        assert result is not None
        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1

    def test_unclosed_code_fence_with_valid_json(self):
        """Opening ```json with no closing ``` should not crash; valid JSON is still parsed."""
        raw = '```json\n[{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]'
        result = _parse_response(raw, 1)
        assert result is not None
        assert len(result) == 1
        assert result[0]["impact_level"] == "none"

    def test_unclosed_code_fence_with_invalid_json(self):
        """Opening ```json with no closing ``` and invalid JSON should return None."""
        raw = "```json\nthis is not valid json at all"
        result = _parse_response(raw, 1)
        assert result is None


# ---------------------------------------------------------------------------
# analyze_news_batch tests
# ---------------------------------------------------------------------------


class TestAnalyzeNewsBatch:
    def test_returns_none_when_copilot_not_found(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            result = analyze_news_batch(
                [{"title": "test", "publisher": "AP", "source_name": "test"}],
                [],
            )
            assert result is None

    def test_returns_empty_for_no_news(self):
        with patch("components.llm_analyzer.is_available", return_value=True):
            result = analyze_news_batch([], [])
            assert result == []

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_successful_cli_call(self, _mock_avail, mock_copilot):
        llm_response = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["金利"],
                    "impact_level": "high",
                    "affected_holdings": ["8306.T"],
                    "reason": "利上げで金融セクター影響",
                },
            ]
        )
        mock_copilot.return_value = llm_response

        news = [{"title": "Fed raises rates", "publisher": "Reuters", "source_name": "S&P 500"}]
        positions = [{"symbol": "8306.T", "sector": "Financial Services", "currency": "JPY", "weight_pct": 30.0}]

        result = analyze_news_batch(news, positions)
        assert result is not None
        assert len(result) == 1
        assert result[0]["impact_level"] == "high"
        assert "8306.T" in result[0]["affected_holdings"]

    @patch("components.llm_analyzer.copilot_call", return_value=None)
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cli_timeout_returns_none(self, _mock_avail, _mock_copilot):
        result = analyze_news_batch(
            [{"title": "test", "publisher": "AP", "source_name": "test"}],
            [],
        )
        assert result is None

    @patch("components.llm_analyzer.copilot_call", return_value=None)
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cli_error_returns_none(self, _mock_avail, _mock_copilot):
        result = analyze_news_batch(
            [{"title": "test", "publisher": "AP", "source_name": "test"}],
            [],
        )
        assert result is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_invalid_json_response_returns_none(self, _mock_avail, mock_copilot):
        mock_copilot.return_value = "Sorry, I cannot process that."

        result = analyze_news_batch(
            [{"title": "test", "publisher": "AP", "source_name": "test"}],
            [],
        )
        assert result is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_model_passed_to_cli(self, _mock_avail, mock_copilot):
        """Verify model param is used in CLI call."""
        mock_copilot.return_value = "[]"
        analyze_news_batch(
            [{"title": "test news", "publisher": "AP", "source_name": "test"}],
            [],
            model="claude-sonnet-4",
        )
        assert mock_copilot.call_count == 1
        _, kwargs = mock_copilot.call_args
        assert kwargs.get("model") == "claude-sonnet-4"


# ---------------------------------------------------------------------------
# AVAILABLE_MODELS sanity check
# ---------------------------------------------------------------------------


class TestAvailableModels:
    def test_has_entries(self):
        """Fallback list has some models (dynamic list may have more)."""
        assert len(AVAILABLE_MODELS) >= 5

    def test_entries_are_tuples(self):
        for m in AVAILABLE_MODELS:
            assert isinstance(m, tuple)
            assert len(m) == 2
            assert isinstance(m[0], str)  # model_id
            assert isinstance(m[1], str)  # display label

    def test_includes_common_models(self):
        ids = [m[0] for m in AVAILABLE_MODELS]
        assert "gpt-4.1" in ids
        assert "claude-sonnet-4" in ids


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCacheMechanism:
    def setup_method(self):
        clear_cache()

    def test_cache_info_empty_initially(self):
        info = get_cache_info()
        assert info["cached"] is False
        assert info["age_sec"] == 0

    def test_compute_news_hash_deterministic(self):
        items = [{"title": "A"}, {"title": "B"}]
        h1 = _compute_news_hash(items)
        h2 = _compute_news_hash(items)
        assert h1 == h2

    def test_compute_news_hash_order_independent(self):
        items1 = [{"title": "A"}, {"title": "B"}]
        items2 = [{"title": "B"}, {"title": "A"}]
        assert _compute_news_hash(items1) == _compute_news_hash(items2)

    def test_compute_news_hash_changes_with_content(self):
        items1 = [{"title": "A"}]
        items2 = [{"title": "B"}]
        assert _compute_news_hash(items1) != _compute_news_hash(items2)

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cache_hit_skips_cli_call(self, _mock_avail, mock_copilot):
        """Second call with same news should not invoke CLI."""
        llm_response = json.dumps(
            [
                {
                    "id": 0,
                    "categories": ["金利"],
                    "impact_level": "high",
                    "affected_holdings": ["8306.T"],
                    "reason": "利上げ",
                },
            ]
        )
        mock_copilot.return_value = llm_response

        news = [{"title": "Fed raises rates", "publisher": "R", "source_name": "SP"}]
        pos = [{"symbol": "8306.T", "sector": "Financial", "currency": "JPY", "weight_pct": 100}]

        # First call: CLI invoked
        result1 = analyze_news_batch(news, pos, cache_ttl=3600)
        assert result1 is not None
        assert mock_copilot.call_count == 1

        # Second call: cache hit, CLI NOT invoked
        result2 = analyze_news_batch(news, pos, cache_ttl=3600)
        assert result2 is not None
        assert mock_copilot.call_count == 1  # still 1
        assert result1 == result2

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cache_miss_on_different_news(self, _mock_avail, mock_copilot):
        """Different news titles should trigger new CLI call."""
        mock_copilot.return_value = json.dumps(
            [{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]
        )

        news1 = [{"title": "News A", "publisher": "R", "source_name": "SP"}]
        news2 = [{"title": "News B", "publisher": "R", "source_name": "SP"}]

        analyze_news_batch(news1, [], cache_ttl=3600)
        assert mock_copilot.call_count == 1

        analyze_news_batch(news2, [], cache_ttl=3600)
        assert mock_copilot.call_count == 2

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cache_miss_on_model_change(self, _mock_avail, mock_copilot):
        """Changing model should invalidate cache."""
        mock_copilot.return_value = json.dumps(
            [{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]
        )

        news = [{"title": "Same News", "publisher": "R", "source_name": "SP"}]
        analyze_news_batch(news, [], model="gpt-4.1", cache_ttl=3600)
        assert mock_copilot.call_count == 1

        analyze_news_batch(news, [], model="claude-sonnet-4", cache_ttl=3600)
        assert mock_copilot.call_count == 2

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cache_disabled_with_zero_ttl(self, _mock_avail, mock_copilot):
        """cache_ttl=0 should always call CLI."""
        mock_copilot.return_value = json.dumps(
            [{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]
        )

        news = [{"title": "Same News", "publisher": "R", "source_name": "SP"}]
        analyze_news_batch(news, [], cache_ttl=0)
        analyze_news_batch(news, [], cache_ttl=0)
        assert mock_copilot.call_count == 2

    def test_clear_cache(self):
        from components.llm_analyzer import _analysis_cache

        _analysis_cache["hash"] = "abc"
        _analysis_cache["results"] = [{}]
        _analysis_cache["timestamp"] = 12345.0
        _analysis_cache["model"] = "gpt-4.1"

        clear_cache()
        info = get_cache_info()
        assert info["cached"] is False

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_cache_info_after_analysis(self, _mock_avail, mock_copilot):
        mock_copilot.return_value = json.dumps(
            [{"id": 0, "categories": [], "impact_level": "none", "affected_holdings": [], "reason": ""}]
        )
        news = [{"title": "Test", "publisher": "R", "source_name": "SP"}]
        analyze_news_batch(news, [], model="gpt-4.1", cache_ttl=3600)

        info = get_cache_info()
        assert info["cached"] is True
        assert info["age_sec"] >= 0
        assert info["model"] == "gpt-4.1"


# ---------------------------------------------------------------------------
# Cache TTL options
# ---------------------------------------------------------------------------


class TestCacheTTLOptions:
    def test_has_options(self):
        assert len(CACHE_TTL_OPTIONS) >= 3

    def test_options_are_tuples(self):
        for opt in CACHE_TTL_OPTIONS:
            assert isinstance(opt, tuple)
            assert len(opt) == 2
            assert isinstance(opt[0], str)  # label
            assert isinstance(opt[1], int)  # seconds

    def test_default_ttl_matches_first_option(self):
        assert DEFAULT_CACHE_TTL_SEC == CACHE_TTL_OPTIONS[0][1]


# ---------------------------------------------------------------------------
# _parse_summary_response tests
# ---------------------------------------------------------------------------


class TestParseSummaryResponse:
    """Test _parse_summary_response parsing logic."""

    def test_valid_json(self):
        raw = json.dumps(
            {
                "overview": "今日はリスクオフムード",
                "key_points": [
                    {
                        "category": "金利",
                        "summary": "FRBの利上げ示唆",
                        "news_ids": [0, 2],
                    },
                ],
                "portfolio_alert": "金利上昇で債券ポジションに注意",
            }
        )
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["overview"] == "今日はリスクオフムード"
        assert len(result["key_points"]) == 1
        assert result["key_points"][0]["category"] == "金利"
        assert result["key_points"][0]["icon"] == "🏦"
        assert result["key_points"][0]["news_ids"] == [0, 2]
        assert result["portfolio_alert"] == "金利上昇で債券ポジションに注意"

    def test_json_in_code_block(self):
        raw = '```json\n{"overview": "概要", "key_points": [], "portfolio_alert": ""}\n```'
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["overview"] == "概要"

    def test_json_with_preamble(self):
        raw = 'Here is the summary:\n{"overview": "test", "key_points": [], "portfolio_alert": ""}'
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["overview"] == "test"

    def test_invalid_json(self):
        result = _parse_summary_response("this is not json at all")
        assert result is None

    def test_array_instead_of_object(self):
        result = _parse_summary_response('[{"id": 0}]')
        assert result is None

    def test_unknown_category_gets_default_icon(self):
        raw = json.dumps(
            {
                "overview": "",
                "key_points": [
                    {"category": "不動産", "summary": "test", "news_ids": [0]},
                ],
                "portfolio_alert": "",
            }
        )
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["key_points"][0]["icon"] == "📌"

    def test_known_categories_get_correct_icons(self):
        categories_icons = {
            "金利": "🏦",
            "為替": "💱",
            "地政学": "🌍",
            "景気": "📊",
            "テクノロジー": "💻",
            "エネルギー": "⛽",
        }
        for cat_name, expected_icon in categories_icons.items():
            raw = json.dumps(
                {
                    "overview": "",
                    "key_points": [
                        {"category": cat_name, "summary": "t", "news_ids": []},
                    ],
                    "portfolio_alert": "",
                }
            )
            result = _parse_summary_response(raw)
            assert result["key_points"][0]["icon"] == expected_icon

    def test_empty_key_points(self):
        raw = json.dumps(
            {
                "overview": "概要のみ",
                "key_points": [],
                "portfolio_alert": "",
            }
        )
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["key_points"] == []

    def test_unclosed_code_fence_with_valid_json(self):
        """Opening ```json with no closing ``` should not crash; valid JSON is still parsed."""
        raw = '```json\n{"overview": "概要", "key_points": [], "portfolio_alert": ""}'
        result = _parse_summary_response(raw)
        assert result is not None
        assert result["overview"] == "概要"

    def test_unclosed_code_fence_with_invalid_json(self):
        """Opening ```json with no closing ``` and invalid JSON should return None."""
        raw = "```json\nnot valid json"
        result = _parse_summary_response(raw)
        assert result is None


# ---------------------------------------------------------------------------
# generate_news_summary tests
# ---------------------------------------------------------------------------


class TestGenerateNewsSummary:
    """Test generate_news_summary function."""

    def setup_method(self):
        clear_summary_cache()

    def _make_news(self, n=3):
        return [
            {
                "title": f"News {i}",
                "publisher": "Reuters",
                "source_name": "SP500",
                "categories": [{"category": "景気", "icon": "📊", "label": "景気・経済指標"}],
                "portfolio_impact": {
                    "impact_level": "medium" if i == 0 else "low",
                    "affected_holdings": ["VTI"] if i == 0 else [],
                    "reason": f"理由{i}",
                },
                "analysis_method": "llm",
            }
            for i in range(n)
        ]

    def test_returns_none_when_not_available(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            result = generate_news_summary(self._make_news(), [])
            assert result is None

    def test_success(self):
        summary_response = json.dumps(
            {
                "overview": "全体概要テスト",
                "key_points": [
                    {"category": "景気", "summary": "景気関連", "news_ids": [0, 1]},
                ],
                "portfolio_alert": "注意点テスト",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=summary_response),
        ):
            result = generate_news_summary(self._make_news(), [])
            assert result is not None
            assert result["overview"] == "全体概要テスト"
            assert len(result["key_points"]) == 1
            assert result["portfolio_alert"] == "注意点テスト"

    def test_cache_hit(self):
        summary_response = json.dumps(
            {
                "overview": "cached",
                "key_points": [],
                "portfolio_alert": "",
            }
        )
        news = self._make_news()
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=summary_response) as mock_copilot,
        ):
            # First call
            result1 = generate_news_summary(news, [], cache_ttl=3600)
            # Second call — should use cache
            result2 = generate_news_summary(news, [], cache_ttl=3600)
            assert result1 == result2
            assert mock_copilot.call_count == 1  # Only called once

    def test_cache_info(self):
        info = get_summary_cache_info()
        assert info["cached"] is False

        summary_response = json.dumps(
            {
                "overview": "x",
                "key_points": [],
                "portfolio_alert": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=summary_response),
        ):
            generate_news_summary(self._make_news(), [], cache_ttl=3600)

        info = get_summary_cache_info()
        assert info["cached"] is True

    def test_clear_cache(self):
        summary_response = json.dumps(
            {
                "overview": "x",
                "key_points": [],
                "portfolio_alert": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=summary_response),
        ):
            generate_news_summary(self._make_news(), [], cache_ttl=3600)

        assert get_summary_cache_info()["cached"] is True
        clear_summary_cache()
        assert get_summary_cache_info()["cached"] is False

    def test_cli_failure_returns_none(self):
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=None),
        ):
            result = generate_news_summary(self._make_news(), [])
            assert result is None

    def test_source_is_news_summary(self):
        """Verify the CLI call uses source='news_summary'."""
        summary_response = json.dumps(
            {
                "overview": "x",
                "key_points": [],
                "portfolio_alert": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=summary_response) as mock_copilot,
        ):
            generate_news_summary(self._make_news(), [])

        assert mock_copilot.call_count >= 1
        _, kwargs = mock_copilot.call_args
        assert kwargs.get("source") == "news_summary"


# ---------------------------------------------------------------------------
# Health summary parser tests
# ---------------------------------------------------------------------------


class TestParseHealthSummaryResponse:
    def test_valid_json(self):
        raw = json.dumps(
            {
                "overview": "PF全体は健全",
                "stock_assessments": [
                    {
                        "symbol": "7203.T",
                        "name": "トヨタ",
                        "assessment": "下降トレンド入り",
                        "action": "注視",
                    },
                ],
                "risk_warning": "テック偏重リスク",
            }
        )
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert result["overview"] == "PF全体は健全"
        assert len(result["stock_assessments"]) == 1
        assert result["stock_assessments"][0]["symbol"] == "7203.T"
        assert result["stock_assessments"][0]["action"] == "注視"
        assert result["risk_warning"] == "テック偏重リスク"

    def test_json_in_code_block(self):
        raw = '```json\n{"overview": "ok", "stock_assessments": [], "risk_warning": ""}\n```'
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert result["overview"] == "ok"
        assert result["stock_assessments"] == []

    def test_json_with_preamble(self):
        raw = 'Here is the result:\n{"overview": "x", "stock_assessments": [], "risk_warning": ""}'
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert result["overview"] == "x"

    def test_invalid_json(self):
        assert _parse_health_summary_response("not json at all") is None

    def test_array_rejected(self):
        raw = json.dumps([{"symbol": "X"}])
        assert _parse_health_summary_response(raw) is None

    def test_empty_assessments(self):
        raw = json.dumps(
            {
                "overview": "全銘柄健全",
                "stock_assessments": [],
                "risk_warning": "",
            }
        )
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert result["stock_assessments"] == []
        assert result["risk_warning"] == ""

    def test_multiple_assessments(self):
        raw = json.dumps(
            {
                "overview": "一部注意",
                "stock_assessments": [
                    {"symbol": "AAPL", "name": "Apple", "assessment": "OK", "action": "保有継続"},
                    {"symbol": "7203.T", "name": "トヨタ", "assessment": "下降", "action": "損切り検討"},
                ],
                "risk_warning": "",
            }
        )
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert len(result["stock_assessments"]) == 2
        assert result["stock_assessments"][1]["action"] == "損切り検討"

    def test_unclosed_code_fence_with_valid_json(self):
        """Opening ```json with no closing ``` should not crash; valid JSON is still parsed."""
        raw = '```json\n{"overview": "ok", "stock_assessments": [], "risk_warning": ""}'
        result = _parse_health_summary_response(raw)
        assert result is not None
        assert result["overview"] == "ok"

    def test_unclosed_code_fence_with_invalid_json(self):
        """Opening ```json with no closing ``` and invalid JSON should return None."""
        raw = "```json\nnot valid json"
        result = _parse_health_summary_response(raw)
        assert result is None


# ---------------------------------------------------------------------------
# Health summary generation tests
# ---------------------------------------------------------------------------


class TestGenerateHealthSummary:
    @staticmethod
    def _make_health_data():
        return {
            "summary": {"total": 2, "healthy": 1, "early_warning": 0, "caution": 1, "exit": 0},
            "positions": [
                {
                    "symbol": "AAPL",
                    "name": "Apple",
                    "alert_level": "none",
                    "trend": "上昇",
                    "rsi": 55.0,
                    "pnl_pct": 12.5,
                    "alert_reasons": [],
                    "value_trap": False,
                    "cross_signal": "none",
                    "change_quality": "良好",
                    "return_stability": "stable",
                },
                {
                    "symbol": "7203.T",
                    "name": "トヨタ",
                    "alert_level": "caution",
                    "trend": "下降",
                    "rsi": 35.0,
                    "pnl_pct": -8.2,
                    "alert_reasons": ["RSI低下", "SMA200割れ"],
                    "value_trap": False,
                    "cross_signal": "death_cross",
                    "days_since_cross": 5,
                    "change_quality": "悪化",
                    "return_stability": "decreasing",
                },
            ],
            "sell_alerts": [
                {
                    "symbol": "7203.T",
                    "name": "トヨタ",
                    "urgency": "warning",
                    "action": "損切り検討",
                    "reason": "注意アラート & 含み損",
                },
            ],
            "alerts": [],
        }

    def test_not_available(self):
        with (
            patch("components.copilot_client.shutil.which", return_value=None),
            patch("components.copilot_client.subprocess.run", side_effect=FileNotFoundError),
        ):
            result = generate_health_summary(self._make_health_data())
            assert result is None

    def test_success(self):
        response = json.dumps(
            {
                "overview": "トヨタに注意が必要",
                "stock_assessments": [
                    {"symbol": "7203.T", "name": "トヨタ", "assessment": "下降トレンド", "action": "損切り検討"},
                ],
                "risk_warning": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=response),
        ):
            result = generate_health_summary(self._make_health_data())
            assert result is not None
            assert "トヨタ" in result["overview"]
            assert len(result["stock_assessments"]) == 1

    def test_cache_hit(self):
        clear_health_summary_cache()
        response = json.dumps(
            {
                "overview": "cached",
                "stock_assessments": [],
                "risk_warning": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=response) as mock_copilot,
        ):
            data = self._make_health_data()
            r1 = generate_health_summary(data, cache_ttl=3600)
            r2 = generate_health_summary(data, cache_ttl=3600)
            assert r1 == r2
            assert mock_copilot.call_count == 1  # second call uses cache

    def test_cache_info_and_clear(self):
        clear_health_summary_cache()
        assert get_health_summary_cache_info()["cached"] is False

        response = json.dumps(
            {
                "overview": "x",
                "stock_assessments": [],
                "risk_warning": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=response),
        ):
            generate_health_summary(self._make_health_data(), cache_ttl=3600)

        assert get_health_summary_cache_info()["cached"] is True
        clear_health_summary_cache()
        assert get_health_summary_cache_info()["cached"] is False

    def test_cli_failure_returns_none(self):
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=None),
        ):
            result = generate_health_summary(self._make_health_data())
            assert result is None

    def test_source_is_health_summary(self):
        """Verify the CLI call uses source='health_summary'."""
        response = json.dumps(
            {
                "overview": "ok",
                "stock_assessments": [],
                "risk_warning": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=response) as mock_copilot,
        ):
            generate_health_summary(self._make_health_data())

        assert mock_copilot.call_count >= 1
        _, kwargs = mock_copilot.call_args
        assert kwargs.get("source") == "health_summary"

    def test_with_news_items(self):
        """Verify generate_health_summary accepts news_items parameter."""
        clear_health_summary_cache()
        news = [
            {
                "title": "米国利下げ観測で市場反発",
                "portfolio_impact": {
                    "impact_level": "high",
                    "affected_holdings": ["AAPL"],
                    "reason": "米国株全般にポジティブ",
                },
            },
        ]
        mock_copilot_response = json.dumps(
            {
                "overview": "ニュースを踏まえると好転可能性あり",
                "stock_assessments": [
                    {"symbol": "7203.T", "name": "トヨタ", "assessment": "下降中", "action": "注視"},
                ],
                "risk_warning": "",
            }
        )
        with (
            patch("components.llm_analyzer.is_available", return_value=True),
            patch("components.llm_analyzer.copilot_call", return_value=mock_copilot_response),
        ):
            result = generate_health_summary(self._make_health_data(), news_items=news)
            assert result is not None
            assert "ニュース" in result["overview"]

    def test_news_changes_cache_hash(self):
        """Verify that different news items produce different cache hashes."""
        clear_health_summary_cache()
        data = self._make_health_data()
        news1 = [{"title": "ニュースA"}]
        news2 = [{"title": "ニュースB"}]
        hash_no_news = _compute_health_hash(data)
        hash_with_news1 = _compute_health_hash(data, news1)
        hash_with_news2 = _compute_health_hash(data, news2)
        assert hash_no_news != hash_with_news1
        assert hash_with_news1 != hash_with_news2


# ---------------------------------------------------------------------------
# Health summary prompt building tests
# ---------------------------------------------------------------------------


class TestBuildHealthSummaryPrompt:
    """_build_health_summary_prompt のプロンプト構築テスト."""

    @staticmethod
    def _make_health_data_with_fundamentals():
        return {
            "summary": {"total": 2, "healthy": 1, "early_warning": 0, "caution": 1, "exit": 0},
            "positions": [
                {
                    "symbol": "AAPL",
                    "name": "Apple",
                    "alert_level": "none",
                    "trend": "上昇",
                    "rsi": 55.0,
                    "pnl_pct": 12.5,
                    "alert_reasons": [],
                    "value_trap": False,
                    "cross_signal": "none",
                    "change_quality": "良好",
                    "return_stability": "stable",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "per": 28.5,
                    "pbr": 45.2,
                    "roe": 0.175,
                    "revenue_growth": 0.08,
                    "earnings_growth": 0.12,
                    "dividend_yield": 0.005,
                    "forward_eps": 7.50,
                    "trailing_eps": 6.95,
                },
                {
                    "symbol": "7203.T",
                    "name": "トヨタ",
                    "alert_level": "caution",
                    "trend": "下降",
                    "rsi": 35.0,
                    "pnl_pct": -8.2,
                    "alert_reasons": ["RSI低下", "SMA200割れ"],
                    "value_trap": True,
                    "cross_signal": "death_cross",
                    "days_since_cross": 5,
                    "change_quality": "悪化",
                    "return_stability": "decreasing",
                    "sector": "Consumer Cyclical",
                    "industry": "Auto Manufacturers",
                    "per": 8.2,
                    "pbr": 0.95,
                    "roe": 0.11,
                    "revenue_growth": -0.03,
                    "earnings_growth": -0.15,
                    "dividend_yield": 0.032,
                    "forward_eps": 180.0,
                    "trailing_eps": 210.0,
                },
            ],
            "sell_alerts": [],
            "alerts": [],
        }

    def test_prompt_includes_fundamentals(self):
        """ファンダメンタルデータがプロンプトに含まれることを検証."""
        data = self._make_health_data_with_fundamentals()
        prompt = _build_health_summary_prompt(data)
        assert "PER=" in prompt
        assert "PBR=" in prompt
        assert "ROE=" in prompt
        assert "売上成長=" in prompt
        assert "利益成長=" in prompt
        assert "EPS方向=" in prompt
        assert "ファンダ=[" in prompt

    def test_prompt_includes_sector(self):
        """セクター情報がプロンプトに含まれることを検証."""
        data = self._make_health_data_with_fundamentals()
        prompt = _build_health_summary_prompt(data)
        assert "Technology/Consumer Electronics" in prompt
        assert "Consumer Cyclical/Auto Manufacturers" in prompt

    def test_prompt_includes_news(self):
        """ニュース情報がプロンプトに含まれることを検証."""
        data = self._make_health_data_with_fundamentals()
        news = [
            {
                "title": "日銀が利上げを決定",
                "portfolio_impact": {
                    "impact_level": "high",
                    "affected_holdings": ["7203.T"],
                    "reason": "円高で輸出企業に逆風",
                },
            },
            {
                "title": "Apple新製品発表",
                "portfolio_impact": {
                    "impact_level": "medium",
                    "affected_holdings": ["AAPL"],
                    "reason": "売上増期待",
                },
            },
            {
                "title": "原油価格安定",
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
            },
        ]
        prompt = _build_health_summary_prompt(data, news_items=news)
        assert "関連ニュース" in prompt
        assert "日銀が利上げを決定" in prompt
        assert "Apple新製品発表" in prompt
        assert "影響銘柄: 7203.T" in prompt
        assert "[high]" in prompt
        assert "[medium]" in prompt
        assert "[参考]" in prompt  # none impact_level news

    def test_prompt_without_news(self):
        """ニュースなしでもプロンプトが正常に生成されることを検証."""
        data = self._make_health_data_with_fundamentals()
        prompt = _build_health_summary_prompt(data)
        assert "## 関連ニュース" not in prompt
        # 基本的なプロンプト構造は維持
        assert "サマリー統計" in prompt
        assert "各銘柄のヘルスチェック結果" in prompt

    def test_prompt_with_empty_news(self):
        """空のニュースリストでもプロンプトが正常に生成されることを検証."""
        data = self._make_health_data_with_fundamentals()
        prompt = _build_health_summary_prompt(data, news_items=[])
        assert "## 関連ニュース" not in prompt

    def test_prompt_without_fundamentals(self):
        """ファンダメンタルデータが無い場合でもプロンプトが正常に生成されることを検証."""
        data = {
            "summary": {"total": 1, "healthy": 0, "early_warning": 1, "caution": 0, "exit": 0},
            "positions": [
                {
                    "symbol": "TEST",
                    "name": "Test Stock",
                    "alert_level": "early_warning",
                    "trend": "横ばい",
                    "rsi": 45.0,
                    "pnl_pct": 2.0,
                    "alert_reasons": ["テスト理由"],
                    "value_trap": False,
                    "cross_signal": "none",
                    "change_quality": "",
                    "return_stability": "",
                    # No fundamentals (per, pbr, etc.)
                },
            ],
            "sell_alerts": [],
        }
        prompt = _build_health_summary_prompt(data)
        assert "Test Stock" in prompt
        assert "ファンダ=[" not in prompt  # No fundamentals section

    def test_prompt_mentions_three_perspectives(self):
        """プロンプトがテクニカル・ファンダメンタル・ニュースの3観点を指示していることを検証."""
        data = self._make_health_data_with_fundamentals()
        prompt = _build_health_summary_prompt(data)
        assert "テクニカル" in prompt
        assert "ファンダメンタル" in prompt
        assert "ニュース" in prompt


# ---------------------------------------------------------------------------
# _extract_json_text tests
# ---------------------------------------------------------------------------


class TestExtractJsonText:
    """応答テキストからJSON部分を抽出する共通ヘルパーのテスト."""

    def test_plain_json(self):
        result = _extract_json_text('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json_text(text)
        assert result == '{"key": "value"}'

    def test_json_in_generic_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = _extract_json_text(text)
        assert result == '{"key": "value"}'

    def test_json_with_prefix_text(self):
        text = 'Here is the result:\n{"key": "value"}'
        result = _extract_json_text(text)
        assert result == '{"key": "value"}'

    def test_array_json(self):
        text = '[{"id": 1}, {"id": 2}]'
        result = _extract_json_text(text)
        assert result == '[{"id": 1}, {"id": 2}]'

    def test_no_json(self):
        result = _extract_json_text("This is plain text with no JSON")
        assert result is None

    def test_json_with_trailing_text(self):
        text = '{"key": "value"}\nSome trailing text'
        result = _extract_json_text(text)
        assert result == '{"key": "value"}'

    def test_unclosed_code_fence_with_valid_json(self):
        """Opening ```json with no closing ``` should not crash; valid JSON is still extracted."""
        text = '```json\n{"key": "value"}'
        result = _extract_json_text(text)
        assert result is not None
        assert result == '{"key": "value"}'

    def test_unclosed_code_fence_with_invalid_content(self):
        """Opening ```json with no closing ``` and no JSON structure should return None."""
        text = "```json\nplain text with no braces"
        result = _extract_json_text(text)
        assert result is None


# ---------------------------------------------------------------------------
# _compute_unified_hash tests
# ---------------------------------------------------------------------------


class TestComputeUnifiedHash:
    """統合分析ハッシュのテスト."""

    def test_deterministic(self):
        news = [{"title": "テストニュース"}]
        h1 = _compute_unified_hash(news)
        h2 = _compute_unified_hash(news)
        assert h1 == h2

    def test_different_news_different_hash(self):
        h1 = _compute_unified_hash([{"title": "ニュースA"}])
        h2 = _compute_unified_hash([{"title": "ニュースB"}])
        assert h1 != h2

    def test_health_data_changes_hash(self):
        news = [{"title": "テスト"}]
        h1 = _compute_unified_hash(news)
        h2 = _compute_unified_hash(
            news, health_data={"positions": [{"symbol": "7203.T", "alert_level": "caution", "pnl_pct": -5.0}]}
        )
        assert h1 != h2

    def test_no_health_data(self):
        news = [{"title": "テスト"}]
        h = _compute_unified_hash(news, health_data=None)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_empty_news(self):
        h = _compute_unified_hash([])
        assert isinstance(h, str)


# ---------------------------------------------------------------------------
# Unified cache tests
# ---------------------------------------------------------------------------


class TestUnifiedCache:
    """統合分析キャッシュのテスト."""

    def setup_method(self):
        clear_unified_cache()

    def test_initial_state(self):
        info = get_unified_cache_info()
        assert info["cached"] is False
        assert info["age_sec"] == 0
        assert info["model"] == ""

    def test_clear_cache(self):
        clear_unified_cache()
        info = get_unified_cache_info()
        assert info["cached"] is False


# ---------------------------------------------------------------------------
# _build_unified_prompt tests
# ---------------------------------------------------------------------------


class TestBuildUnifiedPrompt:
    """統合分析プロンプトの構築テスト."""

    def _make_positions(self):
        return [
            {"symbol": "7203.T", "name": "トヨタ自動車", "sector": "Auto", "weight_pct": 30.0},
            {"symbol": "AAPL", "name": "Apple Inc", "sector": "Technology", "weight_pct": 20.0},
        ]

    def _make_news_list(self):
        return [
            {"id": 0, "title": "日銀利上げ検討", "publisher": "NHK", "source": ""},
            {"id": 1, "title": "Apple新製品発表", "publisher": "Reuters", "source": ""},
        ]

    def _make_health_data(self):
        return {
            "summary": {"total": 2, "healthy": 1, "early_warning": 1, "caution": 0, "exit": 0},
            "positions": [
                {
                    "symbol": "7203.T",
                    "name": "トヨタ自動車",
                    "alert_level": "early_warning",
                    "trend": "下降",
                    "rsi": 35.0,
                    "pnl_pct": -3.5,
                    "alert_reasons": ["RSI低下"],
                    "value_trap": False,
                    "cross_signal": "none",
                    "change_quality": "",
                    "return_stability": "",
                },
            ],
            "sell_alerts": [],
        }

    def test_prompt_includes_three_tasks(self):
        prompt = _build_unified_prompt(self._make_news_list(), self._make_positions())
        assert "T1" in prompt
        assert "T2" in prompt
        assert "分類" in prompt
        assert "要約" in prompt

    def test_prompt_includes_portfolio(self):
        prompt = _build_unified_prompt(self._make_news_list(), self._make_positions())
        assert "7203.T" in prompt
        assert "AAPL" in prompt

    def test_prompt_includes_news(self):
        prompt = _build_unified_prompt(self._make_news_list(), self._make_positions())
        assert "日銀利上げ検討" in prompt
        assert "Apple新製品発表" in prompt

    def test_prompt_with_health_data(self):
        prompt = _build_unified_prompt(
            self._make_news_list(),
            self._make_positions(),
            health_data=self._make_health_data(),
        )
        assert "T3" in prompt
        assert "HC" in prompt
        assert "early_warning" in prompt
        assert "health_summary" in prompt

    def test_prompt_without_health_data(self):
        prompt = _build_unified_prompt(
            self._make_news_list(),
            self._make_positions(),
            health_data=None,
        )
        assert "T3" not in prompt
        assert '"health_summary":null' in prompt

    def test_prompt_with_fundamentals(self):
        health_data = self._make_health_data()
        health_data["positions"][0]["per"] = 12.5
        health_data["positions"][0]["pbr"] = 1.2
        health_data["positions"][0]["roe"] = 0.15
        prompt = _build_unified_prompt(
            self._make_news_list(),
            self._make_positions(),
            health_data=health_data,
        )
        assert "PE12" in prompt or "PE13" in prompt
        assert "PB1.2" in prompt
        assert "ROE15%" in prompt

    def test_prompt_with_cross_signal(self):
        health_data = self._make_health_data()
        health_data["positions"][0]["cross_signal"] = "golden_cross"
        health_data["positions"][0]["days_since_cross"] = 3
        prompt = _build_unified_prompt(
            self._make_news_list(),
            self._make_positions(),
            health_data=health_data,
        )
        assert "GC(3d)" in prompt


# ---------------------------------------------------------------------------
# _parse_unified_response tests
# ---------------------------------------------------------------------------


class TestParseUnifiedResponse:
    """統合分析応答のパーステスト."""

    def _make_valid_response(self, *, include_health=True):
        data = {
            "news_analysis": [
                {
                    "id": 0,
                    "categories": ["金利"],
                    "impact_level": "high",
                    "affected_holdings": ["7203.T"],
                    "reason": "利上げで自動車ローン金利上昇",
                },
                {
                    "id": 1,
                    "categories": ["テクノロジー"],
                    "impact_level": "low",
                    "affected_holdings": ["AAPL"],
                    "reason": "新製品は織り込み済み",
                },
            ],
            "news_summary": {
                "overview": "テスト概要",
                "key_points": [
                    {"category": "金利", "summary": "日銀利上げ", "news_ids": [0]},
                ],
                "portfolio_alert": "利上げ注意",
            },
        }
        if include_health:
            data["health_summary"] = {
                "overview": "概ね健全",
                "stock_assessments": [
                    {
                        "symbol": "7203.T",
                        "name": "トヨタ",
                        "assessment": "やや軟調",
                        "action": "様子見",
                    },
                ],
                "risk_warning": "為替リスクに注意",
            }
        else:
            data["health_summary"] = None
        return json.dumps(data, ensure_ascii=False)

    def test_parse_valid_response(self):
        raw = self._make_valid_response()
        result = _parse_unified_response(raw, 2)
        assert result is not None
        assert len(result["news_analysis"]) == 2
        assert result["news_analysis"][0]["impact_level"] == "high"
        assert result["news_summary"]["overview"] == "テスト概要"
        assert result["health_summary"]["overview"] == "概ね健全"

    def test_parse_with_code_block(self):
        raw = "```json\n" + self._make_valid_response() + "\n```"
        result = _parse_unified_response(raw, 2)
        assert result is not None
        assert len(result["news_analysis"]) == 2

    def test_parse_normalizes_categories(self):
        raw = self._make_valid_response()
        result = _parse_unified_response(raw, 2)
        cat = result["news_analysis"][0]["categories"][0]
        assert "category" in cat
        assert "icon" in cat
        assert "label" in cat

    def test_parse_normalizes_key_points(self):
        raw = self._make_valid_response()
        result = _parse_unified_response(raw, 2)
        kp = result["news_summary"]["key_points"][0]
        assert "icon" in kp
        assert "label" in kp

    def test_parse_health_summary_none(self):
        raw = self._make_valid_response(include_health=False)
        result = _parse_unified_response(raw, 2)
        assert result is not None
        assert result["health_summary"] is None

    def test_parse_invalid_json(self):
        result = _parse_unified_response("not json at all", 2)
        assert result is None

    def test_parse_empty_string(self):
        result = _parse_unified_response("", 2)
        assert result is None

    def test_normalizes_health_assessments(self):
        raw = self._make_valid_response()
        result = _parse_unified_response(raw, 2)
        sa = result["health_summary"]["stock_assessments"]
        assert len(sa) == 1
        assert sa[0]["symbol"] == "7203.T"
        assert sa[0]["action"] == "様子見"


# ---------------------------------------------------------------------------
# apply_news_analysis tests
# ---------------------------------------------------------------------------


class TestApplyNewsAnalysis:
    """統合分析結果のニュースへの適用テスト."""

    def _make_news_items(self):
        return [
            {
                "title": "日銀利上げ",
                "publisher": "NHK",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
            {
                "title": "Apple新製品",
                "publisher": "Reuters",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
        ]

    def _make_analysis(self):
        return [
            {
                "id": 0,
                "categories": [{"category": "金利", "icon": "🏦", "label": "金利・金融政策"}],
                "impact_level": "high",
                "affected_holdings": ["7203.T"],
                "reason": "利上げ影響",
            },
            {
                "id": 1,
                "categories": [{"category": "テクノロジー", "icon": "💻", "label": "テクノロジー"}],
                "impact_level": "low",
                "affected_holdings": ["AAPL"],
                "reason": "新製品は織り込み済み",
            },
        ]

    def test_applies_analysis(self):
        news = self._make_news_items()
        result = apply_news_analysis(news, self._make_analysis())
        assert len(result) == 2
        # Should be sorted by impact: high first
        assert result[0]["portfolio_impact"]["impact_level"] == "high"
        assert result[0]["analysis_method"] == "llm"

    def test_does_not_mutate_original(self):
        news = self._make_news_items()
        result = apply_news_analysis(news, self._make_analysis())
        assert news[0]["analysis_method"] == "keyword"  # Original unchanged
        assert result[0]["analysis_method"] == "llm"

    def test_sorts_by_impact(self):
        news = self._make_news_items()
        analysis = self._make_analysis()
        result = apply_news_analysis(news, analysis)
        levels = [n["portfolio_impact"]["impact_level"] for n in result]
        assert levels == ["high", "low"]

    def test_empty_analysis(self):
        news = self._make_news_items()
        result = apply_news_analysis(news, [])
        assert len(result) == 2
        # Should keep original analysis_method
        assert result[0]["analysis_method"] == "keyword"

    def test_partial_analysis(self):
        news = self._make_news_items()
        analysis = [self._make_analysis()[0]]  # Only first item
        result = apply_news_analysis(news, analysis)
        # First item analyzed, second unchanged
        lmm_items = [n for n in result if n.get("analysis_method") == "llm"]
        assert len(lmm_items) == 1

    def test_invalid_impact_level_normalized(self):
        news = self._make_news_items()
        analysis = [{"id": 0, "categories": [], "impact_level": "invalid", "affected_holdings": [], "reason": ""}]
        result = apply_news_analysis(news, analysis)
        analyzed = [n for n in result if n.get("analysis_method") == "llm"]
        assert analyzed[0]["portfolio_impact"]["impact_level"] == "none"


# ---------------------------------------------------------------------------
# generate_insights tests
# ---------------------------------------------------------------------------


class TestGenerateInsights:
    """AI Insights パネル用のインサイト生成テスト."""

    _mock_snapshot: dict = {
        "total_value_jpy": 10000000,
        "total_pnl_jpy": 500000,
        "total_pnl_pct": 5.0,
        "positions": [{"symbol": "AAPL", "sector": "Technology"}],
    }
    _mock_structure: dict = {
        "sector_breakdown": {"Technology": 0.5, "自動車": 0.3},
        "currency_breakdown": {"USD": 0.7, "JPY": 0.3},
        "risk_level": "やや集中",
        "sector_hhi": 0.34,
    }

    def setup_method(self):
        clear_insights_cache()

    @patch("components.llm_analyzer.is_available", return_value=False)
    def test_returns_none_when_copilot_unavailable(self, _mock_avail):
        result = generate_insights(self._mock_snapshot, self._mock_structure)
        assert result is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_returns_insights_list(self, _mock_avail, mock_call):
        insights = [
            "🟢 ポートフォリオ全体で+5.0%の含み益。利確タイミングを検討",
            "🟡 Technology セクターが50%を占め集中リスクに注意",
            "💱 USD比率70%: 円高局面でのヘッジを検討",
        ]
        mock_call.return_value = json.dumps(insights, ensure_ascii=False)
        result = generate_insights(self._mock_snapshot, self._mock_structure)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0].startswith("🟢")

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_returns_none_on_invalid_json(self, _mock_avail, mock_call):
        mock_call.return_value = "This is not valid JSON at all"
        result = generate_insights(self._mock_snapshot, self._mock_structure)
        assert result is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_caching(self, _mock_avail, mock_call):
        insights = ["🟢 テスト用インサイト"]
        mock_call.return_value = json.dumps(insights, ensure_ascii=False)
        result1 = generate_insights(self._mock_snapshot, self._mock_structure)
        result2 = generate_insights(self._mock_snapshot, self._mock_structure)
        assert result1 == result2
        assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# generate_attribution_summary tests
# ---------------------------------------------------------------------------


class TestGenerateAttributionSummary:
    """パフォーマンス寄与分析 LLM サマリーのテスト."""

    _mock_attribution: dict = {
        "total_pnl_pct": 5.0,
        "stocks": [
            {"symbol": "AAPL", "name": "Apple", "contribution_pct": 3.0, "pnl_pct": 15.0, "sector": "Technology"},
            {"symbol": "7203.T", "name": "トヨタ", "contribution_pct": 1.5, "pnl_pct": 8.0, "sector": "自動車"},
            {"symbol": "MSFT", "name": "Microsoft", "contribution_pct": 0.5, "pnl_pct": 5.0, "sector": "Technology"},
            {"symbol": "9984.T", "name": "SBG", "contribution_pct": -1.0, "pnl_pct": -10.0, "sector": "通信"},
        ],
    }

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_returns_summary_string(self, _mock_avail, mock_call):
        mock_call.return_value = "Apple が最大の寄与銘柄であり、Technology セクターが牽引しています。"
        result = generate_attribution_summary(self._mock_attribution)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("components.llm_analyzer.is_available", return_value=False)
    def test_returns_none_when_unavailable(self, _mock_avail):
        result = generate_attribution_summary(self._mock_attribution)
        assert result is None


# ---------------------------------------------------------------------------
# run_unified_analysis tests
# ---------------------------------------------------------------------------


class TestRunUnifiedAnalysis:
    """統合分析のエンドツーエンドテスト."""

    def setup_method(self):
        clear_unified_cache()

    def _make_news(self):
        return [
            {
                "title": "日銀利上げ",
                "publisher": "NHK",
                "categories": [],
                "portfolio_impact": {
                    "impact_level": "none",
                    "affected_holdings": [],
                    "reason": "",
                },
                "analysis_method": "keyword",
            },
        ]

    def _make_positions(self):
        return [
            {"symbol": "7203.T", "name": "トヨタ", "sector": "Auto", "weight_pct": 50.0},
        ]

    def _make_valid_llm_response(self):
        return json.dumps(
            {
                "news_analysis": [
                    {
                        "id": 0,
                        "categories": ["金利"],
                        "impact_level": "medium",
                        "affected_holdings": ["7203.T"],
                        "reason": "利上げ",
                    },
                ],
                "news_summary": {
                    "overview": "日銀利上げに注目",
                    "key_points": [
                        {"category": "金利", "summary": "利上げ", "news_ids": [0]},
                    ],
                    "portfolio_alert": "",
                },
                "health_summary": None,
            },
            ensure_ascii=False,
        )

    @patch("components.llm_analyzer.is_available", return_value=False)
    def test_returns_none_when_unavailable(self, _mock):
        result = run_unified_analysis(self._make_news(), self._make_positions())
        assert result is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_returns_parsed_result(self, _avail, mock_call):
        mock_call.return_value = self._make_valid_llm_response()
        result = run_unified_analysis(self._make_news(), self._make_positions())
        assert result is not None
        assert len(result["news_analysis"]) == 1
        assert result["news_summary"]["overview"] == "日銀利上げに注目"
        assert result["health_summary"] is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_caches_result(self, _avail, mock_call):
        mock_call.return_value = self._make_valid_llm_response()
        news = self._make_news()
        pos = self._make_positions()
        # First call
        r1 = run_unified_analysis(news, pos, cache_ttl=600)
        assert r1 is not None
        assert mock_call.call_count == 1
        # Second call should use cache
        r2 = run_unified_analysis(news, pos, cache_ttl=600)
        assert r2 is not None
        assert mock_call.call_count == 1  # No additional call

    @patch("components.llm_analyzer.copilot_call", return_value=None)
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_returns_none_on_call_failure(self, _avail, _call):
        result = run_unified_analysis(self._make_news(), self._make_positions())
        assert result is None

    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_empty_news_returns_empty_analysis(self, _avail):
        result = run_unified_analysis([], self._make_positions())
        assert result is not None
        assert result["news_analysis"] == []
        assert result["news_summary"] is None

    @patch("components.llm_analyzer.copilot_call")
    @patch("components.llm_analyzer.is_available", return_value=True)
    def test_with_health_data(self, _avail, mock_call):
        response = json.dumps(
            {
                "news_analysis": [
                    {
                        "id": 0,
                        "categories": ["金利"],
                        "impact_level": "high",
                        "affected_holdings": ["7203.T"],
                        "reason": "利上げ",
                    },
                ],
                "news_summary": {
                    "overview": "概要",
                    "key_points": [],
                    "portfolio_alert": "",
                },
                "health_summary": {
                    "overview": "概ね健全",
                    "stock_assessments": [
                        {"symbol": "7203.T", "name": "トヨタ", "assessment": "やや軟調", "action": "様子見"},
                    ],
                    "risk_warning": "",
                },
            },
            ensure_ascii=False,
        )
        mock_call.return_value = response
        health_data = {
            "summary": {"total": 1, "healthy": 1, "early_warning": 0, "caution": 0, "exit": 0},
            "positions": [
                {
                    "symbol": "7203.T",
                    "name": "トヨタ",
                    "alert_level": "none",
                    "trend": "上昇",
                    "rsi": 55,
                    "pnl_pct": 5.0,
                    "alert_reasons": [],
                    "value_trap": False,
                    "cross_signal": "none",
                    "change_quality": "",
                    "return_stability": "",
                },
            ],
            "sell_alerts": [],
        }
        result = run_unified_analysis(self._make_news(), self._make_positions(), health_data)
        assert result is not None
        assert result["health_summary"] is not None
        assert result["health_summary"]["overview"] == "概ね健全"
