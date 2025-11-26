# CareerNavigator - Web Application

AIによる因果推論を用いたキャリア推薦システムのWeb版実装です。

## 新機能（Web版）

### フロントエンド (React + Vite + Tailwind CSS v4)
- **SkillNote風のモダンなUI**
- **データアップロード**: 6種類のCSVファイルをアップロード
- **因果モデル学習**: LiNGAMによる因果構造学習UI
  - パラメータ設定（最小メンバー数、相関閾値）
  - 重みモード選択（デフォルト/手動/自動最適化）
- **推薦表示**: 3軸スコア詳細（Readiness, Bayesian, Utility）
- **因果グラフ可視化**: インタラクティブなグラフ表示
  - エゴネットワーク
  - 全体グラフ

### バックエンド (FastAPI)
- **データアップロードAPI** (`POST /api/upload`)
- **モデル学習API** (`POST /api/train`) - LiNGAM使用
- **推薦API** (`POST /api/recommend`) - 詳細スコア付き
- **重み調整API**:
  - 手動更新 (`POST /api/update-weights`)
  - 自動最適化 (`POST /api/optimize-weights`) - ベイズ最適化
- **グラフ可視化API**:
  - エゴネットワーク (`POST /api/graph/ego`)
  - 全体グラフ (`POST /api/graph/full`)

## セットアップ

### バックエンド
```bash
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### フロントエンド
```bash
cd frontend
npm install
npm run dev
```

### アクセス
- **フロントエンド**: http://localhost:5173
- **バックエンド API**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

## 使用方法

1. **データアップロード**:
   - サイドバー「データ管理」から6種類のCSVをアップロード

2. **因果モデル学習**:
   - サイドバー「Career」をクリック
   - 学習パラメータと重みモードを選択
   - 「因果モデルを学習開始」ボタンをクリック

3. **推薦実行**:
   - 学習完了後、メンバーIDを入力
   - 「分析実行」ボタンで推薦結果を表示

4. **グラフ可視化**:
   - 推薦結果の下にあるグラフセクションで可視化
   - エゴネットワーク/全体グラフをタブで切り替え

## 技術スタック

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS v4, Axios
- **Backend**: FastAPI, LiNGAM, Pyvis, NetworkX, Optuna
- **ML**: 因果推論（LiNGAM）、ベイジアンネットワーク、ベイズ最適化

## 旧Streamlit版との互換性

全てのStreamlit版の因果推論推薦機能を新しいWeb版に移植済みです。

---

## Original Streamlit Version

The original Streamlit implementation is still available in the repository.
Please refer to the previous commits for the Streamlit-based interface.
