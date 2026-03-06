# Contributing

## Prerequisites

- Python 3.11 or newer
- Git
- Optional services only when you need them:
  - GitHub Copilot CLI / SDK for AI features
  - Neo4j for graph persistence
  - TEI for embedding generation

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

- Copy `.env.example` to `.env` if you need to override cache paths or connect optional services.
- See `docs/external-services.md` for Neo4j, TEI, and Copilot-specific setup notes.

## Running Tests

```bash
python -m pytest tests/ -q --tb=short
```

## Linting and Formatting

```bash
ruff check .
ruff format --check .
python -m mypy src components
```

## pre-commit

```bash
pre-commit install
pre-commit install --hook-type pre-push
pre-commit run --all-files
```

## Insights & AI Guardrails

The `📈 インサイト` tab uses **deterministic, accumulated-data analytics** with no automatic LLM calls.
Keep these invariants when contributing to `src/core/behavior/`, `components/dl_behavior.py`, or `components/tab_insights.py`:

- **Confidence-aware UI**: every computation returns a `ConfidenceLevel`. Surface it in the UI—show a
  "データ不足" notice rather than hiding results or silently returning zeros.
- **No always-on external calls**: `src/core/behavior/` must remain free of network I/O. All price data
  is passed explicitly; the layer never calls `yahoo_client` directly.
- **Opt-in-only retrospective**: the AI retrospective is gated behind an explicit user action
  (`RETRO_RESULT` / `RETRO_ERROR` in `state_keys.py`). Do not add automatic triggers or background tasks
  that call Copilot on page load.
- **Anonymized prompt payloads**: when constructing a retrospective prompt, include only aggregated
  summary text (statistics, style labels, bias signals, memo theme counts). Do not embed raw ticker lists,
  memo bodies, cost prices, or personally identifying portfolio metadata.
- **Layer boundary**: `src/core/behavior/` has zero Streamlit imports. Streamlit session-state handling
  belongs in `components/` only.

## Pull Request Checklist

- Run `python -m pytest tests/ -q --tb=short`
- Run `ruff check .` and `ruff format --check .`
- Run `python -m mypy src components`
- Update README / docs when behavior or setup changes
- Keep optional integrations optional (`NEO4J_MODE=off`, no Copilot SDK, no TEI)
