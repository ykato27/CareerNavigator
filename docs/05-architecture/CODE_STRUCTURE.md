# コード構造ガイド

このドキュメントでは、現行の CareerNavigator のコード構造を Cloudflare / WebUI 前提で整理します。

## プロジェクト構造

```text
CareerNavigator/
├── frontend/
│   ├── functions/                  # Pages Functions API
│   ├── migrations/                 # D1 migrations
│   ├── shared/                     # 推薦ロジック共有層
│   └── src/                        # React UI
├── backend/                        # 旧 FastAPI 実装
├── skillnote_recommendation/       # Python ドメインロジック
├── docs/                           # ドキュメント
└── tests/                          # Python テスト
```

## 主要モジュール

### frontend/

- `src/`: 現行 UI
- `functions/api/[[path]].ts`: `/api/*` を処理する Pages Functions
- `shared/cloudflare-engine.ts`: UI と Functions で共有する推薦エンジン

### backend/

- 旧 FastAPI 実装です
- Cloudflare 本番経路では使いません
- 過去ロジックの参照先として残しています

### skillnote_recommendation/

- Python 側の分析コード
- コアデータ処理、ML、グラフ分析、可視化ユーティリティを含みます

## 実行経路

### ローカル簡易起動

- `frontend/src` を Vite で起動
- `localhost` ではブラウザ内フォールバック API を使用

### Cloudflare 本番

- `frontend/src` を Pages で配信
- `frontend/functions/api/[[path]].ts` が API を処理
- D1 と R2 を永続化先として利用

## 関連ドキュメント

- [README.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/README.md)
- [WEBUI_GUIDE.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/docs/03-webui/WEBUI_GUIDE.md)
- [ARCHITECTURE.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/docs/05-architecture/ARCHITECTURE.md)
