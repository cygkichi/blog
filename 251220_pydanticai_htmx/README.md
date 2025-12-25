# PydanticAI + HTMX チャットアプリケーション

FastAPI、PydanticAI、HTMXを使用したシンプルなAIチャットアプリケーションです。

## 機能

- リアルタイムチャットインターフェース
- HTMX による動的UI更新
- PydanticAI を使用した AI エージェント (Google Gemini 2.5 Flash Lite)
- フィードバック機能 (👍👎)
- Logfire によるロギング
- レスポンシブデザイン

## セットアップ

### 前提条件

- Python 3.13以上
- uv (推奨) または pip

### インストール

1. 依存関係のインストール:
```bash
uv sync
```

または

```bash
pip install -e .
```

2. 環境変数の設定:
`.env` ファイルを作成し、必要なAPI キーを設定してください:
```bash
GOOGLE_API_KEY=your_google_api_key_here
LOGFIRE_TOKEN=your_logfire_token_here
```

## 実行方法

開発サーバーの起動:
```bash
uvicorn app:app --reload
```

ブラウザで `http://localhost:8000` を開いてください。

## プロジェクト構造

```
.
├── app.py              # FastAPI アプリケーション
├── templates/
│   └── index.html      # チャットインターフェース
├── static/
│   └── style.css       # スタイルシート
├── pyproject.toml      # プロジェクト設定
└── README.md           # このファイル
```

## 技術スタック

- **FastAPI**: Webフレームワーク
- **PydanticAI**: AIエージェントフレームワーク
- **HTMX**: 動的UIライブラリ
- **Logfire**: ロギングと監視
- **Google Gemini**: AI モデル

## ライセンス

MIT
