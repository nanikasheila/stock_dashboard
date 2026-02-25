"""Centralized project paths.

Why: Cache paths and data directories were hardcoded in multiple modules
     (yahoo_client, data_loader), making path changes require edits in
     several locations.
How: Define all paths relative to PROJECT_ROOT. Modules import from here
     instead of computing paths independently.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
PRICE_CACHE_DIR: Path = CACHE_DIR / "price_history"
HISTORY_DIR: Path = DATA_DIR / "history"
PORTFOLIO_DIR: Path = DATA_DIR / "portfolio"

# ---------------------------------------------------------------------------
# Overrides via environment variables
# ---------------------------------------------------------------------------

_env_cache = os.environ.get("STOCK_DASHBOARD_CACHE_DIR")
if _env_cache:
    CACHE_DIR = Path(_env_cache)
    PRICE_CACHE_DIR = CACHE_DIR / "price_history"
