# External Services

This dashboard can run without any external service beyond its Python dependencies. The services below are optional integrations.

## Neo4j (optional)

- Purpose: store graph-oriented history and relationship data
- Environment variables:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
  - `NEO4J_MODE`
- Fallback: set `NEO4J_MODE=off` to disable graph writes entirely

## TEI (optional)

- Purpose: generate embeddings through a Text Embeddings Inference service
- Environment variable: `TEI_URL`
- Expected health endpoint: `GET {TEI_URL}/health`
- Fallback: if TEI is unavailable, embedding calls return `None` and the app continues

## GitHub Copilot CLI / SDK (optional)

- Purpose: power the dashboard's AI-assisted analysis and chat features
- Requirements:
  - `copilot` CLI installed and authenticated
  - Python package support for the Copilot SDK available in the active environment
- Fallback: if the SDK is unavailable, the dashboard still imports cleanly and AI-only features should degrade gracefully

## Cache configuration

- `YFINANCE_CACHE_TTL_HOURS`: TTL for Yahoo Finance cache files
- `STOCK_DASHBOARD_CACHE_DIR`: override the default cache location under `data/cache`
