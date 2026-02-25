# モジュールマップ

最終更新: 2026-02-25

## 概要

`stock_dashboard` は Streamlit ベースのポートフォリオ管理ダッシュボードである。
コードは **UI 層 → コンポーネント層 → コアロジック層 / データアクセス層** の3層に分かれており、
依存は常に上位層から下位層への一方向である。

## 層の対応

| 層 | ディレクトリ / ファイル | 責務 |
|---|---|---|
| エントリポイント | `run.py`, `app.py` | CLI ランチャー / Streamlit UI |
| コンポーネント層 | `components/` | UI とコアの橋渡し・LLM 実行 |
| コアロジック層 | `src/core/` | ドメインロジック・計算エンジン |
| データアクセス層 | `src/data/` | 外部 API・ファイル I/O の抽象化 |
| データディレクトリ | `data/` | CSV・JSON・キャッシュの永続化 |
| テスト | `tests/` | 各層のユニット・統合テスト |

---

## エントリポイント

### `run.py`

- **責務**: CLI からアプリを起動するランチャー。`streamlit run app.py` をサブプロセスで実行する。
- **依存**: なし（標準ライブラリのみ）

### `app.py`

- **責務**: Streamlit UI の全体制御。ページ構成・タブ切り替え・ウィジェット配置を担う。
- **依存**: `components/` のすべての公開関数・クラス

---

## コンポーネント層（`components/`）

UI とコアモジュールを接続する中間層。UI 固有の状態管理（`st.session_state` 等）はここに閉じ込める。

### `data_loader.py`

- **責務**: ダッシュボード表示用データの組み立て。
  - ポートフォリオスナップショット生成（`get_current_snapshot`）
  - 資産推移時系列構築（`build_portfolio_history`）
  - セクター・月次サマリー・リスク指標・ベンチマーク比較を提供
- **依存**:
  - `src.core.portfolio.portfolio_manager` — CSV 読込・ポジション構築
  - `src.core.models` — `Position` 等のデータモデル
  - `src.core.health_check` — ヘルスチェックエンジン
  - `src.core.value_trap` — バリュートラップ検出
  - `src.core.screening.*` — スクリーニング指標
  - `src.data.yahoo_client` — 株価取得（キャッシュ経由）
  - `src.data.history_store` — 取引履歴 JSON 読み書き

### `charts.py`

- **責務**: Plotly チャートオブジェクトの生成。データ加工は行わず、受け取った DataFrame・辞書をグラフに変換する。
- **依存**: `components.data_loader`（間接的に `app.py` 経由で呼ばれる）

### `llm_analyzer.py`

- **責務**: LLM を使った経済ニュース分析。プロンプト構築・メモリキャッシュ（TTL 設定可能）・複数モデル対応。
- **依存**: `components.copilot_client`

### `copilot_client.py`

- **責務**: GitHub Copilot CLI のラッパー。サブプロセス経由で LLM 推論を実行し、結果を返す。
- **依存**: 標準ライブラリ（`subprocess`）のみ

### `settings_store.py`

- **責務**: ユーザー設定の永続化（JSON ファイル読み書き）。モデル選択・TTL 等のダッシュボード設定を管理する。
- **依存**: 標準ライブラリのみ

---

## コアロジック層（`src/core/`）

ビジネスルールを表現する純粋なロジック層。ファイル I/O や HTTP 通信を直接行わない。
外部データが必要な場合は引数として受け取るか、`data_loader.py` に委譲する。

### `models.py`

- **責務**: ドメインオブジェクトの型定義（`@dataclass`）。
  - `Position` — 1銘柄の保有状態
  - `ForecastResult` — リターン推計結果（3シナリオ）
  - `HealthResult` — ヘルスチェック結果
  - `RebalanceAction` — リバランス提案
  - `SimulationResult` — 複利シミュレーション結果
- **依存**: `src.core.common`（`is_cash` のみ）

### `common.py`

- **責務**: 層横断の軽量ユーティリティ。`is_cash`, `is_etf` 等の判定関数を提供する。
- **依存**: なし（標準ライブラリのみ）

### `health_check.py`

- **責務**: 保有銘柄の健全性診断エンジン。
  - `check_trend_health` — SMA・RSI によるトレンド評価
  - `check_change_quality` — ファンダメンタル変質チェック
  - `check_long_term_suitability` — 長期保有適性評価
  - `compute_alert_level` — アラートレベル（none / early_warning / caution / exit）集約
- **依存**: `src.core.common`, `src.core.screening.indicators`

### `return_estimate.py`

- **責務**: 銘柄ごとのリターン推定（アナリスト予想 / ヒストリカル / 配当利回り）。`ForecastResult` を返す。
- **依存**: `src.core.models`

### `ticker_utils.py`

- **責務**: ティッカーシンボルから取引所・通貨・国を推定するルールベースマッピング。
- **依存**: なし

### `value_trap.py`

- **責務**: バリュートラップ（割安に見える問題銘柄）の検出ロジック。
- **依存**: `src.core.models`

### `portfolio/portfolio_manager.py`

- **責務**: `data/portfolio/portfolio.csv` を読み込み、FX レートを適用して `Position` オブジェクトリストを生成する。
- **依存**: `src.core.models`, `src.core.ticker_utils`, `src.data.yahoo_client`（FX 取得）

### `portfolio/concentration.py`

- **責務**: ポートフォリオの集中度リスク分析（銘柄・セクター・通貨の偏りを定量化）。
- **依存**: `src.core.models`

### `screening/indicators.py`

- **責務**: 株主還元率・配当安定性等のファンダメンタル指標を計算する。
- **依存**: なし（pandas / numpy のみ）

### `screening/alpha.py`

- **責務**: 超過リターン（アルファ）指標の計算。
- **依存**: なし

### `screening/technicals.py`

- **責務**: SMA・RSI 等テクニカル指標の計算。
- **依存**: なし

---

## データアクセス層（`src/data/`）

外部 API・ファイルシステムへのアクセスを抽象化する。コアロジック層はこの層を直接参照しない。

### `yahoo_client.py`

- **責務**: Yahoo Finance API のラッパー。株価・財務データ取得にキャッシュ（`data/cache/price_history/`、TTL: 4時間）を適用する。
- **依存**: `yfinance` ライブラリ

### `history_store.py`

- **責務**: `data/history/` 以下の JSON ファイルを読み書きする。取引履歴・LLM 分析履歴・ヘルスチェック履歴を管理する。
- **依存**: 標準ライブラリ（`json`, `pathlib`）

### `graph_store.py`

- **責務**: グラフ構造データ（銘柄間の関係性等）の永続化。
- **依存**: 標準ライブラリ

### `summary_builder.py`

- **責務**: ポートフォリオ状態の要約テキスト生成（LLM プロンプトへの入力向け）。
- **依存**: `src.core.models`

### `embedding_client.py`

- **責務**: テキストの埋め込みベクトル生成クライアント（類似銘柄検索等に利用）。
- **依存**: 外部 API（設定依存）

---

## データディレクトリ（`data/`）

コードではなく生データを格納する。バージョン管理対象外のファイルを含む。

| パス | 内容 |
|---|---|
| `data/portfolio/portfolio.csv` | ポートフォリオの Source of Truth（銘柄・株数・取得価格・通貨） |
| `data/history/trade/*.json` | 取引履歴（1取引1ファイル、`YYYY-MM-DD_action_TICKER.json` 形式） |
| `data/history/health/` | ヘルスチェック結果の履歴 JSON |
| `data/history/report/` | 生成済みレポートの履歴 JSON |
| `data/cache/price_history/` | Yahoo Finance から取得した株価のローカルキャッシュ（TTL: 4時間） |

---

## 依存方向の概要

```
app.py
  └─→ components/data_loader.py
  └─→ components/charts.py
  └─→ components/llm_analyzer.py
        └─→ components/copilot_client.py
  └─→ components/settings_store.py

components/data_loader.py
  └─→ src/core/portfolio/portfolio_manager.py
  └─→ src/core/models.py
  └─→ src/core/health_check.py
  └─→ src/core/value_trap.py
  └─→ src/core/screening/indicators.py
  └─→ src/data/yahoo_client.py
  └─→ src/data/history_store.py

src/core/portfolio/portfolio_manager.py
  └─→ src/core/models.py
  └─→ src/core/ticker_utils.py
  └─→ src/data/yahoo_client.py   ← FX レート取得のみ

src/core/health_check.py
  └─→ src/core/common.py
  └─→ src/core/screening/indicators.py

src/data/yahoo_client.py
  └─→ data/cache/price_history/  (ファイルシステム)
  └─→ Yahoo Finance API           (外部)
```

> **ルール**: `src/core/` は `src/data/` を直接呼び出してはならない（`portfolio_manager.py` の FX 取得を唯一の例外とする）。データの取得・結合は `components/data_loader.py` が担う。
