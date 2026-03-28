# クイックスタートガイド

## 最短起動

現行 UI は `frontend` から起動します。

```bash
cd CareerNavigator/frontend
npm install
npm run dev
```

- 起動先: `http://localhost:5173`
- `localhost` ではブラウザ内フォールバックが有効です
- Cloudflare の D1/R2 がなくても基本操作を確認できます

## Cloudflare 向けビルド

```bash
cd CareerNavigator/frontend
npx tsc -b
npm run build
```

## Cloudflare に寄せたローカル確認

```bash
cd CareerNavigator/frontend
npm run build
npx wrangler pages dev dist
```

## Python 側のセットアップ

Python の分析コードやテストを扱う場合だけ `uv` を使ってください。

```bash
cd CareerNavigator
uv sync
uv run pytest
```

## 次のステップ

- [README.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/README.md)
- [WEBUI_GUIDE.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/docs/03-webui/WEBUI_GUIDE.md)
- [CODE_STRUCTURE.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/docs/05-architecture/CODE_STRUCTURE.md)
- [TEST_DESIGN.md](/C:/Users/加藤裕樹/Desktop/CareerNavigator/docs/06-testing/TEST_DESIGN.md)
