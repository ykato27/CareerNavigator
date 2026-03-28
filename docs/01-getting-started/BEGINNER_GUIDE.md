# 初心者向けコード理解ガイド

このガイドは、現在の CareerNavigator をどこから読めばよいかを短く整理したものです。現行の実行系は Streamlit ではなく、`frontend` と Cloudflare Pages Functions です。

## まず見るファイル

### UI の入口

- [main.tsx](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/src/main.tsx)
- [App.tsx](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/src/App.tsx)

### API の入口

- [functions/api/[[path]].ts](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/functions/api/[[path]].ts)

### 推薦ロジックの中心

- [cloudflare-engine.ts](/C:/Users/加藤裕樹/Desktop/CareerNavigator/frontend/shared/cloudflare-engine.ts)

### 旧 Python 実装

- [backend/main.py](/C:/Users/加藤裕樹/Desktop/CareerNavigator/backend/main.py)
- [skillnote_recommendation/](/C:/Users/加藤裕樹/Desktop/CareerNavigator/skillnote_recommendation)

## 現在の読み方

1. UI の画面遷移を知りたいなら `frontend/src/App.tsx`
2. API の入口を知りたいなら `frontend/functions/api/[[path]].ts`
3. 推薦ロジックを知りたいなら `frontend/shared/cloudflare-engine.ts`
4. 過去の Python 側設計を知りたいなら `backend/` と `skillnote_recommendation/`

## 補足

- ローカル起動手順は [README.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/README.md) を参照してください
- Cloudflare 配備手順も root README に集約しています
