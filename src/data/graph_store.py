"""Backward-compatible facade for graph store (split into src/data/graph/).

Why: External callers import from ``src.data.graph_store``.
     This facade preserves those import paths after the 3-way split.
How: Re-export all public names from the ``src.data.graph`` package.
"""

from src.data.graph import *  # noqa: F403
