"""Tests for components.data_loader.clear_price_cache()."""

import sys
from pathlib import Path
from unittest.mock import patch

# --- プロジェクトルートを sys.path に追加 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.data_loader import clear_price_cache

# ---------------------------------------------------------------------------
# Tests using tmp_path fixture (real filesystem, no network I/O)
# ---------------------------------------------------------------------------


def test_clear_price_cache_deletes_csv_files_and_returns_count(tmp_path):
    # Arrange: create a fake cache directory with CSV files
    cache_dir = tmp_path / "price_history"
    cache_dir.mkdir()
    (cache_dir / "close_3mo.csv").write_text("date,VTI\n2026-01-01,250.0\n")
    (cache_dir / "close_1y.csv").write_text("date,VTI\n2025-01-01,230.0\n")

    # Patch _PRICE_CACHE_DIR to point at the temp directory
    with patch("components.data_loader._PRICE_CACHE_DIR", cache_dir):
        # Act
        deleted_count = clear_price_cache()

    # Assert: both CSV files were removed
    assert deleted_count == 2
    assert not any(cache_dir.glob("*.csv"))


def test_clear_price_cache_empty_directory_returns_zero(tmp_path):
    # Arrange: empty cache directory (no CSV files)
    cache_dir = tmp_path / "price_history"
    cache_dir.mkdir()

    with patch("components.data_loader._PRICE_CACHE_DIR", cache_dir):
        # Act
        deleted_count = clear_price_cache()

    # Assert: no files deleted, no exception raised
    assert deleted_count == 0


def test_clear_price_cache_nonexistent_directory_returns_zero(tmp_path):
    # Arrange: directory does not exist
    cache_dir = tmp_path / "nonexistent_cache"

    with patch("components.data_loader._PRICE_CACHE_DIR", cache_dir):
        # Act
        deleted_count = clear_price_cache()

    # Assert: graceful no-op
    assert deleted_count == 0


def test_clear_price_cache_ignores_non_csv_files(tmp_path):
    # Arrange: cache directory with mixed file types
    cache_dir = tmp_path / "price_history"
    cache_dir.mkdir()
    (cache_dir / "close_3mo.csv").write_text("date,VTI\n")
    (cache_dir / "README.txt").write_text("this should not be deleted")
    (cache_dir / "metadata.json").write_text("{}")

    with patch("components.data_loader._PRICE_CACHE_DIR", cache_dir):
        # Act
        deleted_count = clear_price_cache()

    # Assert: only the CSV was deleted; other files remain
    assert deleted_count == 1
    assert (cache_dir / "README.txt").exists()
    assert (cache_dir / "metadata.json").exists()


def test_clear_price_cache_single_file_returns_one(tmp_path):
    # Arrange
    cache_dir = tmp_path / "price_history"
    cache_dir.mkdir()
    (cache_dir / "close_3mo.csv").write_text("date,VTI\n2026-01-01,250.0\n")

    with patch("components.data_loader._PRICE_CACHE_DIR", cache_dir):
        # Act
        deleted_count = clear_price_cache()

    # Assert
    assert deleted_count == 1
