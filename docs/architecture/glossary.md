# 用語集（Glossary）

最終更新: 2026-02-25

本プロジェクトで使用するドメイン固有の用語を定義する。
コードとドキュメント間で表記が揺れないよう、新しい概念を導入する際は必ずここに追記すること。

---

## Position（ポジション）

ポートフォリオ内の1銘柄の保有状態を表すドメインオブジェクト。

| フィールド | 型 | 説明 |
|---|---|---|
| `symbol` | `str` | ティッカーシンボル（例: `7203.T`, `AAPL`, `JPY.CASH`） |
| `shares` | `int` | 保有株数 |
| `cost_price` | `float` | 1株あたりの平均取得価格（`cost_currency` 建て） |
| `cost_currency` | `str` | 取得価格の通貨（例: `JPY`, `USD`） |
| `current_price` | `float` | 最新市場価格（`market_currency` 建て） |
| `value_jpy` | `float` | 現在の評価額（JPY 換算済み） |
| `sector` | `str` | GICS セクター名 |
| `country` | `str` | 銘柄の上場国 |
| `market_currency` | `str` | 取引所の取引通貨 |
| `name` | `str` | 銘柄の表示名 |

実装: `src/core/models.py` の `Position` dataclass。

---

## Ticker / Symbol（ティッカー / シンボル）

銘柄を一意に識別するコード文字列。

- **米国株・ETF**: サフィックスなし（例: `AAPL`, `VTI`, `MSFT`）
- **東証上場銘柄**: `.T` サフィックス（例: `7203.T` = トヨタ自動車）
- **シンガポール上場**: `.SI` サフィックス
- **現金ポジション**: `{CURRENCY}.CASH` 形式（例: `JPY.CASH`, `USD.CASH`）

サフィックスから取引所・通貨・国を推定するロジックは `src/core/ticker_utils.py` に集約している。

---

## HealthCheck（ヘルスチェック）

保有銘柄の投資テーゼが引き続き有効かを診断する機能。3つの観点で評価する。

| 観点 | 関数 | 評価内容 |
|---|---|---|
| トレンド | `check_trend_health()` | SMA50/SMA200・RSI によるモメンタム |
| 変質チェック | `check_change_quality()` | ファンダメンタル指標の悪化度 |
| 長期適性 | `check_long_term_suitability()` | 長期保有に値するかの総合判断 |

3観点の結果を `compute_alert_level()` が集約し、`AlertLevel` を返す。
実装: `src/core/health_check.py`。

---

## AlertLevel（アラートレベル）

ヘルスチェックの結果を4段階で表す定数。

| 値 | 定数名 | 意味 |
|---|---|---|
| `"none"` | `ALERT_NONE` | 問題なし |
| `"early_warning"` | `ALERT_EARLY_WARNING` | 初期警戒（SMA50 割れ・RSI 低下など） |
| `"caution"` | `ALERT_CAUTION` | 要注意（デッドクロス接近・指標悪化） |
| `"exit"` | `ALERT_EXIT` | 売却検討（デッドクロス・複数指標の同時悪化） |

深刻度の昇順: `none` → `early_warning` → `caution` → `exit`。
実装: `src/core/health_check.py`（`ALERT_*` 定数）。

---

## ValueTrap（バリュートラップ）

PBR・PER 等の指標では割安に見えるが、業績悪化・構造的問題などにより株価が上昇しない銘柄のこと。

本システムでは `src/core/value_trap.py` の `detect_value_trap()` が
ファンダメンタル指標の複合条件でバリュートラップを検出する。

---

## ForecastResult（リターン推計結果）

1銘柄に対するリターン予測を3シナリオで保持するデータオブジェクト。

| フィールド | 説明 |
|---|---|
| `base` | ベースシナリオの期待リターン（%） |
| `optimistic` | 楽観シナリオ |
| `pessimistic` | 悲観シナリオ |
| `source` | 推計根拠（`analyst` / `historical` / `dividend`） |

実装: `src/core/models.py` の `ForecastResult` dataclass。
生成: `src/core/return_estimate.py`。

---

## RebalanceAction（リバランス提案）

ポートフォリオのリバランス時に生成される提案アクション。

| 値 | 意味 |
|---|---|
| `"buy"` | 新規購入 |
| `"increase"` | 買い増し |
| `"reduce"` | 一部売却 |
| `"sell"` | 全売却 |

実装: `src/core/models.py` の `RebalanceAction` dataclass。

---

## SimulationResult（シミュレーション結果）

複利効果を考慮した将来資産額のシミュレーション結果を保持するオブジェクト。
期間・リターン率・拠出額のパラメータに基づいて `src/core/models.py` が生成する。

---

## Drawdown（ドローダウン）

過去の高値から現在価格への下落率。ポートフォリオのリスク指標として使用する。

$$\text{Drawdown} = \frac{\text{current\_price} - \text{peak\_price}}{\text{peak\_price}}$$

`components/data_loader.py` の `compute_drawdown_series()` が日次系列を計算する。

---

## Sharpe Ratio（シャープレシオ）

リスク調整後リターン指標。超過リターンをリターンの標準偏差で割った値。

$$\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$$

- $R_p$: ポートフォリオリターン
- $R_f$: リスクフリーレート
- $\sigma_p$: リターンの標準偏差

`components/data_loader.py` の `compute_rolling_sharpe()` がローリング計算を提供する。

---

## WeightDrift（ウェイトドリフト）

初期配分比率（ターゲットウェイト）と現在の実際の比率の乖離。
乖離が閾値を超えた銘柄がリバランス候補となる。

`components/data_loader.py` の `compute_weight_drift()` が計算する。

---

## Sector / GICS（セクター / GICS）

**GICS**（Global Industry Classification Standard）は S&P と MSCI が定義した業種分類体系。
銘柄を11セクターに分類する（例: `Technology`, `Financials`, `Energy` 等）。

本システムでは `Position.sector` フィールドに GICS セクター名を格納し、
セクター別の資産配分分析に使用する。

---

## FX Rate（為替レート）

複数通貨建ての保有銘柄を JPY に統一換算するための為替レート。

- `src/core/portfolio/portfolio_manager.py` の `get_fx_rates()` が Yahoo Finance からリアルタイム取得する。
- 通貨ペアの命名: `{FROM}JPY=X`（例: `USDJPY=X`, `SGDJPY=X`）

---

## Cash Position（現金ポジション）

現金保有を `Position` オブジェクトとして表現する特殊ティッカー形式。

| ティッカー | 通貨 |
|---|---|
| `JPY.CASH` | 日本円現金 |
| `USD.CASH` | 米ドル現金 |

`src/core/common.py` の `is_cash(symbol)` で判定する。
`shares` フィールドが保有金額（例: 100000 = 10万円）として機能する。
