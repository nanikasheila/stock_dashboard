# 構造評価レポート — 2026-02-25

| 項目 | 内容 |
|---|---|
| **評価日** | 2026-02-25 |
| **評価者** | architect エージェント |
| **評価対象** | `stock_dashboard` リポジトリ全体 |
| **評価手法** | ペースレイヤリング分析・単一責任原則・依存方向検証 |

---

## 概要

本レポートは `stock_dashboard` の構造的健全性を評価し、課題・推奨対応・ペースレイヤリング分析を記録する。
個々のコード品質ではなく、**モジュール間の関係性・データの流れ・責務の配置**を評価対象とする。

---

## ペースレイヤリング評価

変化速度の異なるコードが正しい層に配置されているかを評価する。

| 層 | 該当モジュール | 変化速度 | 状態 |
|---|---|---|---|
| インフラ | yfinance、Neo4j、Azure OpenAI | 低 | ✅ 外部化 |
| データアクセス | `src/data/` | 低〜中 | ✅ 正しく分離 |
| ドメインモデル | `src/core/models.py`, `common.py` | 低 | ✅ 安定 |
| ドメインロジック | `src/core/portfolio/`, `screening/`, `health_check.py` | 中 | ✅ 正しく配置 |
| **アプリサービス層** | ❌ 存在しない | — | 🔴 欠落 |
| プレゼンテーション | `app.py`, `charts.py` | 高 | ⚠️ 肥大化 |
| アダプタ | `data_loader.py`（1683行） | 高 | 🔴 責務過多 |

### 所見

`src/core/` と `src/data/` は正しく分離されており、ドメインモデルおよびデータアクセス層の構造は健全である。
一方、**アプリサービス層が存在しない**ため、ユースケース調整ロジックが `components/data_loader.py` に集積している。
また `app.py`（2158行）と `data_loader.py`（1683行）はそれぞれ単一責任を逸脱した肥大化が顕著である。

---

## 課題一覧

重大度: **Critical**（設計上の欠陥・整合性リスク）/ **Warning**（保守性・拡張性の阻害）/ **Info**（改善推奨）

| # | 重大度 | 課題 | 影響範囲 | 推奨対応 |
|---|---|---|---|---|
| 1 | 🔴 Critical | アプリケーションサービス層の欠落 | `data_loader.py` 全体、`app.py` | `src/usecase/` を新設し、ユースケース調整ロジックを移動 |
| 2 | 🔴 Critical | Streamlit グローバルミュータブル状態（マルチセッション汚染） | `copilot_client._execution_logs`, `embedding_client._available` | `st.session_state` への移行 |
| 3 | 🔴 Critical | 二重永続化の整合性保証なし（JSON vs Neo4j） | `history_store`, `graph_store` | Source of Truth を宣言し、HistoryRepository で統一 |
| 4 | ⚠️ Warning | コアドメイン計算が `components/` に漏出 | `data_loader.py` のリスク指標群 | `src/core/analytics/` を新設して移動 |
| 5 | ⚠️ Warning | 3層キャッシュの非同期問題 | `yahoo_client`, `data_loader`, `@st.cache_data` | 統一キャッシュ戦略を実装 |
| 6 | ⚠️ Warning | `summary_builder` vs `history_store._build_research_summary` 重複 | `src/data/` 内 | `history_store` が `summary_builder` を呼ぶよう統一 |
| 7 | ⚠️ Warning | `app.py` の UI コンポーネント未分解（2158行） | `app.py` 全体 | Streamlit `pages/` 分割 + `components/ui/` |
| 8 | ⚠️ Warning | `graph_store.py` の単一責任違反（1092行） | `src/data/graph_store.py` | 接続管理・スキーマ・リポジトリに3分割 |
| 9 | ℹ️ Info | `portfolio_manager.get_fx_rates(client)` の非明示インターフェース | `src/core/portfolio/` | `FxRateProvider` Protocol を定義 |
| 10 | ℹ️ Info | キャッシュパスのハードコード分散 | `yahoo_client`, `data_loader` | `settings_store` 経由で一元管理 |

---

## 課題詳細

### #1 — アプリケーションサービス層の欠落（Critical）

`components/data_loader.py`（1683行）が UI アダプタ・ユースケースオーケストレーション・ドメイン計算を兼務している。
ユースケース調整ロジックが Streamlit に依存しているため、純粋な unittest が書けない状態である。

**推奨対応**: `src/usecase/` を新設し、以下のユースケースを配置する。

```
src/usecase/
  __init__.py
  snapshot_usecase.py
  portfolio_history_usecase.py
  health_check_usecase.py
  economic_news_usecase.py
```

`data_loader.py` はこれらのユースケースを呼び出す薄いアダプタ（目標 200行程度）に縮小する。

→ 詳細: [ADR-001](adr/ADR-001.md)

---

### #2 — Streamlit グローバルミュータブル状態（Critical）

`copilot_client._execution_logs`（クラス変数）および `embedding_client._available`（クラス変数）がモジュールレベルで保持されている。
Streamlit はマルチユーザー環境でプロセスを共有するため、あるユーザーの操作が別ユーザーのセッション状態に影響を与える可能性がある。

**推奨対応**:
- `_execution_logs` → `st.session_state["execution_logs"]` へ移行
- `_available` フラグ → セッションスコープに閉じ込めるか、接続チェックを呼び出し時に都度実行する

---

### #3 — 二重永続化の整合性保証なし（Critical）

`history_store`（JSON）と `graph_store`（Neo4j）が同一エンティティを並行永続化している。
書き込みはトランザクション管理されておらず、一方が失敗した場合に不整合が生じる。

**推奨対応**: JSON を Source of Truth と宣言し、Neo4j をセカンダリ検索インデックスとして位置づける。

→ 詳細: [ADR-002](adr/ADR-002.md)

---

### #4 — コアドメイン計算の漏出（Warning）

ボラティリティ、最大ドローダウン、シャープレシオなどのリスク指標計算が `components/data_loader.py` に実装されており、ドメインロジックが UI アダプタ層に漏出している。
これらの計算は UI に依存しない純粋なドメイン計算であるため、`src/core/` に配置すべきである。

**推奨対応**: `src/core/analytics/` を新設し、リスク指標計算群を移動する。

```
src/core/analytics/
  __init__.py
  risk_metrics.py   # volatility, max_drawdown, sharpe_ratio 等
```

---

### #5 — 3層キャッシュの非同期問題（Warning）

以下の3層でキャッシュが独立して管理されており、キャッシュの一貫性が保証されていない。

| 層 | 実装 | TTL |
|---|---|---|
| `yahoo_client` | ディスクキャッシュ（`data/cache/`） | 不明 |
| `data_loader` | メモリキャッシュ（内部） | 不明 |
| `app.py` | `@st.cache_data` | TTL 設定あり |

**推奨対応**: キャッシュ戦略を `settings_store` 経由で一元管理し、各層のキャッシュ TTL を統一する。

---

### #6 — サマリービルダーの重複（Warning）

`src/data/summary_builder.py` と `src/data/history_store._build_research_summary()` が類似のサマリー生成ロジックを重複実装している。

**推奨対応**: `history_store` が `summary_builder` を呼び出す形に統一し、重複を排除する。

---

### #7 — `app.py` の肥大化（Warning）

`app.py`（2158行）が全ページの UI を単一ファイルで管理しており、変更の局所化が困難である。

**推奨対応**:
- Streamlit の `pages/` ディレクトリ機能を活用してページを分割する
- 再利用可能な UI コンポーネントは `components/ui/` に抽出する

---

### #8 — `graph_store.py` の単一責任違反（Warning）

`src/data/graph_store.py`（1092行）が接続管理・スキーマ定義・リポジトリ操作を一手に担っている。

**推奨対応**: 以下の3ファイルに分割する。

```
src/data/graph/
  connection.py    # Neo4j 接続・セッション管理
  schema.py        # ノード/エッジ定義・制約
  repository.py    # クエリ・CRUD 操作
```

---

### #9 — `FxRateProvider` の非明示インターフェース（Info）

`portfolio_manager.get_fx_rates(client)` の `client` 引数の型が明示されていない。
他クライアントへの差し替え時に暗黙のインターフェース違反が発生しやすい。

**推奨対応**: `FxRateProvider` Protocol を `src/core/portfolio/` に定義する。

```python
class FxRateProvider(Protocol):
    def get_fx_rate(self, from_currency: str, to_currency: str) -> float: ...
```

---

### #10 — キャッシュパスのハードコード分散（Info）

`yahoo_client` と `data_loader` でキャッシュパス（`data/cache/price_history/` 等）がハードコードされており、パス変更時に複数箇所の修正が必要になる。

**推奨対応**: `settings_store` でキャッシュパスを一元管理し、各モジュールから参照する。

---

## 優先対応ロードマップ

### フェーズ 1（Critical — 早期対応推奨）

| タスク | 対象 ADR |
|---|---|
| `src/usecase/` の新設と `data_loader.py` の縮小 | ADR-001 |
| JSON を SoT とする二重永続化の整理 | ADR-002 |
| `st.session_state` へのグローバル状態移行 | — |

### フェーズ 2（Warning — 中期対応）

| タスク |
|---|
| `src/core/analytics/` の新設とリスク指標の移動 |
| `app.py` の `pages/` 分割 |
| `graph_store.py` の3分割 |
| `summary_builder` の重複排除 |
| 統一キャッシュ戦略の実装 |

### フェーズ 3（Info — 長期対応）

| タスク |
|---|
| `FxRateProvider` Protocol の定義 |
| キャッシュパスの `settings_store` への集約 |

---

## 関連ドキュメント

- [ADR-001](adr/ADR-001.md) — アプリケーションサービス層の導入
- [ADR-002](adr/ADR-002.md) — Source of Truth の宣言（JSON vs Neo4j）
- [module-map.md](module-map.md) — モジュール層の対応表
- [data-flow.md](data-flow.md) — データフロー定義
