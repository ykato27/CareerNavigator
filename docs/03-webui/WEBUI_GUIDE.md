# WebUI Guide

CareerNavigator の現行 UI は、Vite + React + TypeScript と Cloudflare Pages Functions を前提にしています。

## Runtime Layout

```text
Browser
  -> frontend/src
  -> /api/*
Cloudflare Pages Functions
  -> frontend/functions/api/[[path]].ts
  -> frontend/shared/cloudflare-engine.ts
Cloudflare D1
  -> metadata / fallback object storage
Cloudflare R2
  -> session/model object storage when enabled
```

## Local Modes

### UI only

```bash
cd frontend
npm install
npm run dev -- --port 5174
```

`localhost` ではブラウザ内フォールバックが有効なので、Cloudflare の設定なしでも基本操作を確認できます。
Python backend を使う場合は `http://localhost:8010` を API 接続先にし、画面は `http://localhost:5174` を開きます。

### Cloudflare-like

```bash
cd frontend
npm run build
npx wrangler pages dev dist
```

## Production Deployment

本番手順は root の [README.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/README.md) を参照してください。
