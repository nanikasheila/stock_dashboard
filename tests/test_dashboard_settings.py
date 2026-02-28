"""settings_store のユニットテスト."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# settings_store を import できるよう path を追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from components.settings_store import DEFAULTS, load_settings, save_settings


class TestLoadSettings:
    """load_settings のテスト."""

    def test_returns_defaults_when_no_file(self, tmp_path):
        """ファイルが存在しない場合デフォルト値を返す."""
        p = tmp_path / "nonexistent.json"
        result = load_settings(p)
        assert result == DEFAULTS

    def test_loads_saved_values(self, tmp_path):
        """保存済みの値を正しく読み込む."""
        p = tmp_path / "settings.json"
        saved = {"period_label": "1年", "chart_style": "折れ線"}
        p.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

        result = load_settings(p)
        assert result["period_label"] == "1年"
        assert result["chart_style"] == "折れ線"
        # 保存されていないキーはデフォルト値
        assert result["show_invested"] == DEFAULTS["show_invested"]

    def test_ignores_unknown_keys(self, tmp_path):
        """不明なキーは無視する."""
        p = tmp_path / "settings.json"
        saved = {"unknown_key": "value", "period_label": "6ヶ月"}
        p.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

        result = load_settings(p)
        assert "unknown_key" not in result
        assert result["period_label"] == "6ヶ月"

    def test_handles_corrupt_json(self, tmp_path):
        """壊れたJSONでもデフォルト値を返す."""
        p = tmp_path / "settings.json"
        p.write_text("{invalid json", encoding="utf-8")

        result = load_settings(p)
        assert result == DEFAULTS

    def test_handles_non_dict_json(self, tmp_path):
        """JSONがdictでない場合デフォルト値を返す."""
        p = tmp_path / "settings.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        result = load_settings(p)
        assert result == DEFAULTS

    def test_partial_saved_values_merged_with_defaults(self, tmp_path):
        """一部のみ保存されている場合、残りはデフォルトで補完される."""
        p = tmp_path / "settings.json"
        saved = {"target_amount_man": 10000, "projection_years": 10}
        p.write_text(json.dumps(saved), encoding="utf-8")

        result = load_settings(p)
        assert result["target_amount_man"] == 10000
        assert result["projection_years"] == 10
        assert result["period_label"] == DEFAULTS["period_label"]
        assert result["chart_style"] == DEFAULTS["chart_style"]


class TestSaveSettings:
    """save_settings のテスト."""

    def test_saves_and_loads_roundtrip(self, tmp_path):
        """save → load のラウンドトリップが正しく動作する."""
        p = tmp_path / "settings.json"
        settings = {
            "period_label": "1年",
            "chart_style": "折れ線",
            "show_invested": False,
            "benchmark_label": "VTI (米国全体)",
            "show_individual": True,
            "show_projection": False,
            "target_amount_man": 8000,
            "projection_years": 10,
            "auto_refresh_label": "1分",
            "llm_enabled": True,
            "llm_auto_analyze": False,
            "llm_model": "claude-sonnet-4",
            "llm_cache_ttl_label": "3時間",
            "chat_model": "gpt-4.1",
        }
        save_settings(settings, p)
        loaded = load_settings(p)
        assert loaded == settings

    def test_creates_parent_directory(self, tmp_path):
        """親ディレクトリが無くても自動作成する."""
        p = tmp_path / "sub" / "dir" / "settings.json"
        save_settings(DEFAULTS, p)
        assert p.exists()

    def test_only_saves_known_keys(self, tmp_path):
        """DEFAULTSに無いキーは保存しない."""
        p = tmp_path / "settings.json"
        settings = dict(DEFAULTS)
        settings["extra_key"] = "should_not_be_saved"
        save_settings(settings, p)

        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "extra_key" not in raw

    def test_overwrites_existing_file(self, tmp_path):
        """既存ファイルを上書き保存する."""
        p = tmp_path / "settings.json"
        save_settings({"period_label": "1年"}, p)
        save_settings({"period_label": "3年"}, p)

        loaded = load_settings(p)
        assert loaded["period_label"] == "3年"

    def test_saves_utf8_content(self, tmp_path):
        """日本語が正しくUTF-8で保存される."""
        p = tmp_path / "settings.json"
        save_settings(DEFAULTS, p)

        content = p.read_text(encoding="utf-8")
        assert "3ヶ月" in content
        assert "積み上げ面" in content
