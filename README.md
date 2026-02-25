# Portfolio Dashboard

ポートフォリオの資産推移・リスク指標・構造分析をブラウザ上でインタラクティブに可視化するダッシュボード。

## 概要

Streamlit + Plotly ベースの Web ダッシュボードで、以下を可視化・分析します：

- **KPI カード**: 総資産額・評価損益・日次変動・リスク指標
- **総資産推移**: 積み上げ面 / 折れ線 / 積み上げ棒（3スタイル切替）
- **ドローダウン推移**: 高値からの下落率
- **ローリングシャープレシオ**: 60日ウィンドウでの推移
- **投資元本 vs 評価額**: 累積投資額と評価額の比較
- **将来予測**: 3シナリオの将来推計
- **セクター/通貨構成**: ドーナツチャート
- **ツリーマップ**: 銘柄の評価額を可視化
- **相関ヒートマップ**: 銘柄間の日次リターン相関
- **ウェイトドリフト警告**: 初期配分と現在配分の乖離
- **ヘルスチェック & 売りタイミング通知**: 保有銘柄の健全性診断
- **経済ニュース & PF影響分析**: 主要指標から自動取得＋影響度判定
- **Copilot チャット**: AI による対話的なPF分析

## セットアップ

### 前提条件

- Python 3.10+

### インストール

```bash
cd stock_dashboard
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### データ準備

`data/portfolio/portfolio.csv` にポートフォリオデータを配置してください。

CSV フォーマット:
```csv
symbol,shares,cost_price,cost_currency,purchase_date,memo
7203.T,100,2850,JPY,2024-01-15,
AAPL,10,180,USD,2024-03-01,
```

取引履歴は `data/history/trade/` に JSON ファイルとして配置します。

## 起動

```bash
python run.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

### オプション

| パラメータ | デフォルト | 説明 |
|:---|:---|:---|
| `--port` | 8501 | サーバーポート番号 |
| `--no-browser` | false | ブラウザを自動で開かない |

## テスト

```bash
python -m pytest tests/ -q
```

## プロジェクト構造

```
stock_dashboard/
├── run.py                    # ランチャー
├── app.py                    # Streamlit メインアプリ
├── requirements.txt
├── components/               # UI コンポーネント
│   ├── charts.py             # Plotly チャート構築
│   ├── copilot_client.py     # GitHub Copilot CLI クライアント
│   ├── data_loader.py        # データ取得・加工
│   ├── llm_analyzer.py       # LLM ニュース分析
│   └── settings_store.py     # 設定の永続化
├── src/                      # コアモジュール
│   ├── core/
│   │   ├── common.py         # 共通ユーティリティ
│   │   ├── models.py         # データモデル
│   │   ├── health_check.py   # ヘルスチェックエンジン
│   │   ├── return_estimate.py # リターン推定
│   │   ├── ticker_utils.py   # ティッカー/通貨判定
│   │   ├── value_trap.py     # バリュートラップ検出
│   │   ├── portfolio/        # ポートフォリオ管理
│   │   └── screening/        # スクリーニング指標
│   └── data/
│       ├── yahoo_client.py   # yfinance ラッパー
│       ├── history_store.py  # 履歴データ管理
│       └── ...
├── data/                     # データディレクトリ
│   ├── portfolio/            # ポートフォリオCSV
│   ├── cache/                # API キャッシュ
│   └── history/              # 取引・分析履歴
└── tests/                    # テスト
```

## 元プロジェクト

このダッシュボードは [stock_skills](../stock_skills) プロジェクトから独立させたものです。  
コアモジュール (`src/`) は stock_skills から必要なものをコピーしています。

## ライセンス

Data provided by Yahoo Finance via yfinance.
