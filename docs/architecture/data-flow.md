# データフロー

最終更新: 2026-02-26

## 概要

本ドキュメントは `stock_dashboard` における主要データの流れ・Source of Truth・変換ポイントを記述する。

---

## 1. ポートフォリオデータフロー

### Source of Truth

```
data/portfolio/portfolio.csv
  列: symbol, shares, cost_price, cost_currency, purchase_date, sector, memo, ...
```

CSV が唯一の正規データ源である。手動編集で保有内容を更新する。

### フロー

```
data/portfolio/portfolio.csv
  │  (CSV 読込)
  ▼
src/core/portfolio/portfolio_manager.py
  ├─ load_portfolio()        → list[Position]
  └─ get_fx_rates()          → dict[str, float]
          │ (Yahoo Finance でリアルタイム FX 取得)
          ▼
        src/data/yahoo_client.py
components/data_loader.py
  ├─ get_current_snapshot()
  │    │ portfolio_manager.load_portfolio()
  │    │ yahoo_client で現在株価取得
  │    │ FX レートで JPY 換算
  │    └─ Position.value_jpy を付与
  └─ get_sector_breakdown()
       └─ Position.sector を集計
  │
  ▼
app.py
  └─ Plotly チャート（components/charts.py 経由）で表示
```

### 変換ポイント

| ステップ | 変換内容 |
|---|---|
| CSV → `Position` | `portfolio_manager.load_portfolio()` が型付きオブジェクトに変換 |
| `Position` + 株価 | `data_loader` が `current_price` と `value_jpy` を付与 |
| 多通貨 → JPY | `get_fx_rates()` で取得した FX レートで統一換算 |

---

## 2. 取引履歴フロー

### Source

```
data/history/trade/
  ファイル命名: YYYY-MM-DD_{action}_{TICKER}.json
  例: 2024-01-10_buy_VTI.json
```

1取引1ファイル形式。`action` は `buy` / `sell` / `transfer` のいずれか。

### フロー

```
data/history/trade/*.json
  │  (JSON 読込)
  ▼
src/data/history_store.py
  └─ load_history(category="trade")  → list[dict]
  │
  ▼
components/data_loader.py
  └─ build_portfolio_history()
       │ 取引を日付順にソート
       │ 各日の保有株数を累積計算
       │ yahoo_client で日次株価取得
       └─ 日次 value_jpy を展開 → DataFrame（index=date, columns=symbol）
  │
  ▼
app.py
  └─ 資産推移グラフとして表示
```

### 変換ポイント

| ステップ | 変換内容 |
|---|---|
| JSON → dict | `history_store.load_history()` がファイルをパース |
| 取引列 → 保有状態時系列 | `data_loader` が累積計算で各日の `shares` を復元 |
| 保有状態 × 日次株価 | `yahoo_client` の履歴データと結合して `value_jpy` 時系列を生成 |

---

## 2a. 取引書き込みフロー

### 概要

ダッシュボードから新規取引（買い/売り/転送）を記録する書き込みフロー。
CQRS パターンにより、読み取り専用の `data_loader.py` と分離されている（ADR-003）。

### フロー

```
app.py
  └─ render_trade_form()  ← ユーザー入力
       │
       ▼
components/trade_form.py
  └─ _handle_submit()  — バリデーション・送信
       │
       ▼
components/trade_writer.py
  └─ record_trade()
       │
       ├─ Step 1: src/data/history_store.save_trade()
       │    → data/history/trade/YYYY-MM-DD_{type}_{TICKER}.json
       │    (JSON = Source of Truth, ADR-002)
       │
       ├─ Step 2 (buy):  portfolio_manager.add_position()
       │                   → data/portfolio/portfolio.csv (filelock)
       │
       ├─ Step 2 (sell): portfolio_manager.sell_position()
       │                   → data/portfolio/portfolio.csv (filelock)
       │
       └─ Step 2 (transfer): Skip (履歴記録のみ)
       │
       ▼
  st.cache_data.clear() + st.rerun()  — ダッシュボード全体を再描画
```

### 変換ポイント

| ステップ | 変換内容 |
|---|---|
| フォーム入力 → dict | `trade_form` がスキーマに従った取引レコード dict を構築 |
| dict → JSON ファイル | `history_store.save_trade()` がファイルに書き込む |
| dict → CSV 行 | `portfolio_manager.add_position()` / `sell_position()` が filelock で CSV を書き換え |

---

## 3. 株価データフロー

### 外部 API

Yahoo Finance (`yfinance` ライブラリ経由)。ティッカーシンボル・期間を指定して OHLCV データを取得する。

### キャッシュ戦略

```
src/data/yahoo_client.py
  ├─ キャッシュ確認: data/cache/price_history/{ticker}.json
  │    TTL: 4 時間（_CACHE_TTL_SECONDS = 14400）
  │    ヒット → キャッシュから返却
  │    ミス  → Yahoo Finance API 呼び出し → キャッシュ書込 → 返却
  └─ 返却値: pd.DataFrame（columns: Open, High, Low, Close, Volume）
```

**手動更新**: ユーザーが「📥 今すぐ更新」ボタンを押すと `clear_price_cache()` がディスクキャッシュ（`data/cache/price_history/*.csv`）を全削除し、`st.cache_data.clear()` でメモリキャッシュもクリアする。次回アクセス時に Yahoo Finance API から再取得される（ADR-004）。

### フロー

```
app.py / components/data_loader.py
  │  (株価リクエスト)
  ▼
src/data/yahoo_client.py
  │
  ├─[キャッシュヒット]→ data/cache/price_history/{ticker}.json → DataFrame
  │
  └─[キャッシュミス] → Yahoo Finance API
                          │
                          ▼
                    data/cache/price_history/{ticker}.json に書込
                          │
                          ▼
                       DataFrame を返却
```

### 変換ポイント

| ステップ | 変換内容 |
|---|---|
| API レスポンス → DataFrame | `yfinance` が pandas DataFrame に変換 |
| DataFrame → JSON キャッシュ | `yahoo_client` が `to_json()` でファイルに書込 |
| キャッシュ → DataFrame | `read_json()` で復元 |

---

## 4. LLM 分析フロー

### 概要

経済ニュースを取得し、LLM（GitHub Copilot CLI）でポートフォリオへの影響を分析する。
分析結果はメモリキャッシュに保存され、TTL を超えるまで再利用される。

### フロー

```
外部ニュース API / フィード
  │
  ▼
components/data_loader.py
  └─ fetch_economic_news()  → list[dict]（ヘッドライン・本文・日時）
  │
  ▼
components/llm_analyzer.py
  ├─ run_unified_analysis()
  │    ├─ キャッシュ確認（メモリ内、TTL 設定可能）
  │    │    ヒット → キャッシュから返却
  │    │    ミス  → プロンプト構築 → copilot_client 呼び出し
  │    └─ 結果をメモリキャッシュに保存
  │
  └─ apply_news_analysis()  → ニュース × ポートフォリオの影響マトリクス
  │
  ▼
components/copilot_client.py
  └─ サブプロセスで GitHub Copilot CLI を実行
       └─ stdout を解析 → 文字列レスポンスを返却
  │
  ▼
app.py
  └─ LLM 分析結果タブに表示
```

### キャッシュ設定

| キャッシュ種別 | 場所 | TTL |
|---|---|---|
| ニュース分析（unified） | メモリ（`llm_analyzer` モジュール変数） | ユーザー設定（`settings_store` 経由） |
| ヘルスサマリー | メモリ | ユーザー設定 |
| ポートフォリオサマリー | メモリ | ユーザー設定 |

### 変換ポイント

| ステップ | 変換内容 |
|---|---|
| ニュースリスト + `Position` リスト → プロンプト | `llm_analyzer` がテンプレートに埋め込む |
| CLI stdout → 構造化レスポンス | `copilot_client` が文字列をパース |
| LLM テキスト → 表示用 dict | `apply_news_analysis` が銘柄ごとに影響を分類 |

---

## 全体俯瞰

```
[外部]                [ファイルシステム]           [メモリ]              [UI]
Yahoo Finance API  → data/cache/          → DataFrame       →
                    price_history/                            components/
                                                              data_loader  → app.py
                   data/portfolio/        → list[Position]  →             → Plotly
                   portfolio.csv                               charts.py
                                                                           表示
                   data/history/          → list[dict]      →
                   trade/*.json

Copilot CLI ←── components/llm_analyzer ──────────────────────────────────→ app.py
(サブプロセス)   (メモリキャッシュ)                                          (LLM タブ)
```
