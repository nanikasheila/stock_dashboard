# コードレビュー結果

実施日: 2026-02-25
レビュアー: reviewer エージェント

## 総合評価

**品質レベル: B**

実装は完成度が高くドメインロジックも丁寧に設計されているが、ファイル肥大化・コード重複・ロギング統一性の3点が構造的負債として蓄積している。

## 修正済み（本レビューで対応）

| # | ファイル | 修正内容 |
|---|---|---|
| 1 | `src/core/health_check.py` | `_finite_or_none()` の重複定義を削除し `value_trap.py` からインポートに変更 |
| 2 | `src/core/return_estimate.py` | `health_check` のプライベート関数 `_detect_value_trap` のクロスインポートを `value_trap.detect_value_trap` の直接インポートに変更 |
| 3 | `src/data/graph_store.py` | `NEO4J_PASSWORD` のデフォルト値 `"password"` を空文字 `""` に変更（セキュリティ修正） |

## 未修正（将来の改善タスク）

### 🥇 優先度1: 大規模ファイルの分割

ガイドライン（500行以下）を超過するファイル:

| ファイル | 行数 | 推奨分割方針 |
|---|---|---|
| `app.py` | 2,158行 | Streamlit multipage app 構造（`pages/`）への分割 |
| `components/data_loader.py` | 1,683行 | データ取得・キャッシュ・ビルド関数群を機能別ファイルに |
| `components/llm_analyzer.py` | 1,153行 | 分析タイプ（news, stock, portfolio等）ごとに分割 |
| `src/data/graph_store.py` | 1,092行 | 接続層 / スキーマ / エンティティ別 CRUD に分割 |
| `src/data/yahoo_client.py` | 942行 | 財務データ取得部を `_fetch_fundamentals()` 等に分解 |
| `src/core/portfolio/portfolio_manager.py` | 674行 | `snapshot.py` / `rebalance.py` に分割 |
| `src/core/health_check.py` | 656行 | `check_trend_health` 等を別ファイルに分離 |
| `src/data/history_store.py` | 626行 | `summary_builder` との重複解消後に再評価 |
| `components/charts.py` | 569行 | `charts_portfolio.py` / `charts_simulation.py` 等 |
| `src/core/screening/indicators.py` | 507行 | `value_score.py` / `shareholder_return.py` 等 |

### 🥈 優先度2: コード品質改善

| 項目 | 対象 | 内容 |
|---|---|---|
| 関数サイズ超過 (>100行) | `check_trend_health()`, `compute_alert_level()`, `get_snapshot()`, `get_stock_detail()`, `estimate_portfolio_return()`, `detect_pullback_in_uptrend()` | 内部ロジックをヘルパー関数に分離 |
| `print()` → `logging` 統一 | `yahoo_client.py`, `portfolio_manager.py` | `copilot_client.py` の `logging.getLogger(__name__)` パターンに統一 |
| ベアな `except Exception` | `yahoo_client.py`, `history_store.py`, `embedding_client.py` | 具体的な例外型に変更、`logging.warning()` で記録 |
| Why/How docstring 不足 | 大半のパブリック関数 | ガイドラインの Why/How 形式に更新 |
| コード重複 | `history_store._build_research_summary()` ↔ `summary_builder.build_research_summary()` | `history_store` 側を削除して `summary_builder` に一本化 |
| ミュータブルグローバル変数 | `copilot_client.py` の `_execution_logs`, `embedding_client.py` の `_available` | `threading.Lock` での保護またはクラスでカプセル化 |

### 🥉 優先度3: セキュリティ

| 項目 | 対象 | 内容 |
|---|---|---|
| Cypher インジェクションリスク | `graph_store.py` の `merge_trade()` | `rel_type` を f-string で直接埋め込み。現状は安全だが将来の拡張時に注意 |

## テストカバレッジ

本レビューで追加されたテスト:

| テストファイル | テスト数 | 対象 |
|---|---|---|
| `tests/test_core_models.py` | 99 | models, common, ticker_utils, value_trap |
| `tests/test_core_screening.py` | 141 | indicators, alpha, technicals |
| `tests/test_core_portfolio.py` | 48 | portfolio_manager, concentration |
| `tests/test_core_return_estimate.py` | 39 | return_estimate |
| `tests/test_data_stores.py` | 49 | history_store, summary_builder |
| `tests/test_data_yahoo_client.py` | 35 | yahoo_client |
| `tests/test_dashboard_charts.py` | 34 | charts |
| **合計追加** | **445** | |
| **全テスト合計** | **767** | 全パス |
