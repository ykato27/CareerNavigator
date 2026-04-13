# CareerNavigator

CareerNavigator は、6種類の CSV をアップロードしてキャリア推薦と組織スキル分析を行う Web アプリです。

現在の実行系は `frontend` を中心とした Cloudflare Pages / Pages Functions 構成です。  
Python の `backend` はローカル確認や過去ロジック参照用として残しています。

Cloudflare 本番版の学習は `cloudflare-approx` モードです。  
無料枠で運用するため、既存の Python / LiNGAM 学習をそのまま本番で動かす構成ではありません。

## 構成

- `frontend/`
  - React + TypeScript + Vite
  - Cloudflare Pages Functions
- `backend/`
  - FastAPI の参照実装
- `skillnote_recommendation/`
  - Python 側の分析ロジック

## 入力ファイル

アップロード対象は次の 6 種類です。

- `members`
- `skills`
- `education`
- `license`
- `categories`
- `acquired`

補足:

- アップロードした CSV や一時生成物は Git 管理しません
- `backend/temp_uploads/` や coverage レポートはローカル確認用です

## ローカル実行

### 1. フロントエンドだけ起動する

Cloudflare なしで画面を確認する最短手順です。

```bash
cd frontend
npm install
npm run dev
```

- URL: `http://localhost:5173`
- `localhost` ではブラウザ内フォールバックが有効です
- D1 / R2 がなくても基本操作を確認できます

### 2. バックエンドも起動する

FastAPI の参照実装も一緒に起動したい場合の手順です。

```bash
cd C:\Users\加藤裕樹\Desktop\CareerNavigator
uv run --extra web uvicorn backend.main:app --reload --port 8000
```

`backend` は `backend.*` の絶対 import を使っているため、プロジェクトルートから起動してください。
`backend` ディレクトリ内で `uvicorn main:app` を実行すると、`ModuleNotFoundError: No module named 'backend'` が発生します。

別ターミナルでフロントエンドを起動します。

```bash
cd frontend
npm install
npm run dev
```

補足:

- フロントエンド: `http://localhost:5173`
- バックエンド: `http://localhost:8000`
- API ドキュメント: `http://localhost:8000/api/docs`

### 3. 型チェックとビルド

```bash
cd frontend
npx tsc -b
npm run build
```

出力先:

- `frontend/dist/`

## Cloudflare で実行

### 前提

- Cloudflare アカウント
- `wrangler` にログイン済み

```bash
npx wrangler login
```

### 1. D1 migration

```bash
cd frontend
npx wrangler d1 migrations apply career-navigator-db --remote
```

現在の migration:

- [0001_init.sql](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/migrations/0001_init.sql)
- [0002_storage_objects.sql](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/migrations/0002_storage_objects.sql)
- [0003_operational_metadata.sql](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/migrations/0003_operational_metadata.sql)

### 2. build

```bash
cd frontend
npm install
npm run build
```

### 3. deploy

```bash
cd frontend
npx wrangler pages deploy dist --project-name career-navigator --commit-dirty=true
```

本番 URL:

- `https://career-navigator-5b0.pages.dev/`

### 4. 本番運用で使う環境変数

Pages / Functions では次の変数を設定できます。

- `SESSION_TTL_DAYS`
  - デフォルト `30`
  - セッション保持日数
- `CLEANUP_TOKEN`
  - 管理用 cleanup API の認証トークン

期限切れセッションを手動削除する場合:

```bash
curl -X POST "https://career-navigator-5b0.pages.dev/api/admin/cleanup" \
  -H "x-admin-token: <CLEANUP_TOKEN>"
```

## 永続化

現在の保存先は次の優先順位です。

1. `R2` が有効なら `R2`
2. `R2` が無い場合は `D1.storage_objects`

そのため、`R2` を有効化しなくても今の実装は動作します。  
ただし長期運用やデータ増加を考えると、最終的には `R2` を使う構成が望ましいです。

## 本番運用メモ

- セッションには保持期限があります。デフォルトは `30日` です
- `/api/train` は `training_mode: cloudflare-approx` を返します
- モデル本体には `artifact_version`, `training_mode`, `source_storage` を保存します
- `R2` 未有効時は `D1.storage_objects` に保存されるため、無料枠では容量監視が重要です

## 主要パス

- [frontend/functions/api/[[path]].ts](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/functions/api/[[path]].ts)
- [frontend/shared/cloudflare-engine.ts](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/shared/cloudflare-engine.ts)
- [frontend/wrangler.jsonc](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/wrangler.jsonc)
