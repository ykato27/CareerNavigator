# WebUI完全ガイド

CareerNavigatorのWebUIは、React + FastAPIによるモダンなWebアプリケーションで、従業員キャリアダッシュボードとデータ可視化機能を提供します。

## アーキテクチャ

```
クライアント (Browser)
    ↓ HTTP/REST
フロントエンド (React + TypeScript)
    ↓ API Calls
バックエンド (FastAPI)
    ↓ Data Access
Core Library (skillnote_recommendation)
    ↓
CSV Data Files
```

## セットアップ

### 前提条件

- **Node.js**: v18以上
- **Python**: 3.11以上
- **uv**: Pythonパッケージマネージャー

### フロントエンドのセットアップ

```bash
# frontendディレクトリに移動
cd frontend

# 依存関係のインストール
npm install

# 開発サーバーの起動
npm run dev
```

デフォルトで `http://localhost:5173` で起動します。

### バックエンドのセットアップ

```bash
# プロジェクトルートに戻る
cd ..

# FastAPIサーバーの起動
cd backend
uv run uvicorn main:app --reload --port 8000
```

デフォルトで `http://localhost:8000` で起動します。

### 両方を同時に起動

開発時は2つのターミナルで別々に起動：

**ターミナル1 (Backend)**:
```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

**ターミナル2 (Frontend)**:
```bash
cd frontend
npm run dev
```

フロントエンドが自動的にバックエンドAPIに接続します。

---

## フロントエンド構成

### 技術スタック

- **React 18**: UIライブラリ
- **TypeScript**: 型安全性
- **Vite**: ビルドツール
- **Tailwind CSS**: スタイリング
- **Axios**: HTTP クライアント
- **Recharts**: データ可視化
- **Lucide React**: アイコン

### ディレクトリ構造

```
frontend/
├── src/
│   ├── pages/                      # ページコンポーネント
│   │   ├── DataUpload.tsx          # データアップロードページ
│   │   ├── ModelTraining.tsx       # モデル学習ページ
│   │   ├── Recommendation.tsx      # 推薦ページ
│   │   └── EmployeeCareerDashboard.tsx  # キャリアダッシュボード
│   │
│   ├── components/                 # 再利用可能コンポーネント
│   │   ├── Header.tsx              # ヘッダーコンポーネント
│   │   ├── FileUploader.tsx        # ファイルアップローダー
│   │   ├── RecommendationCard.tsx  # 推薦カード
│   │   └── ...
│   │
│   ├── App.tsx                     # メインアプリ
│   ├── main.tsx                    # エントリーポイント
│   └── index.css                   # グローバルスタイル
│
├── index.html                      # HTMLテンプレート
├── package.json                    # 依存関係
├── tsconfig.json                   # TypeScript設定
├── vite.config.ts                  # Vite設定
└── tailwind.config.js              # Tailwind設定
```

### 主要コンポーネント

#### EmployeeCareerDashboard

従業員向けキャリアダッシュボードのメインコンポーネント。

**機能**:
- メンバー選択
- 目標設定（ロールモデル/役職ベース）
- ギャップ分析
- キャリアパス生成
- ロードマップ可視化

**状態管理**:
```typescript
interface DashboardState {
  selectedMember: string;
  targetSelectionMode: 'role-model' | 'role-based';
  targetRole?: string;
  targetMember?: string;
  gapAnalysis?: GapAnalysisResult;
  careerPath?: CareerPathResult;
}
```

**主要API呼び出し**:
- `GET /api/career/members` - メンバー一覧取得
- `GET /api/career/roles` - 役職一覧取得
- `POST /api/career/gap-analysis` - ギャップ分析
- `POST /api/career/career-path` - キャリアパス生成
- `POST /api/career/role/role-skills` - 役職スキル統計

---

## バックエンド構成

### 技術スタック

- **FastAPI**: Webフレームワーク
- **Pydantic**: データバリデーション
- **Uvicorn**: ASGIサーバー
- **Core Library**: skillnote_recommendation パッケージ

### ディレクトリ構造

```
backend/
├── api/                            # APIエンドポイント
│   ├── career_dashboard.py         # キャリアダッシュボードAPI
│   ├── role_based_dashboard.py     # 役職ベース分析API
│   └── __init__.py

**基本エンドポイント**:

```python
# メンバー一覧取得
GET /api/career/members
Response: List[{member_code: str, member_name: str, role: str}]

# 役職一覧取得
GET /api/career/roles
Response: List[{role: str, member_count: int}]

# ギャップ分析
POST /api/career/gap-analysis
Request: {
    source_member: str,
    target_member?: str,
    target_role?: str,
    min_frequency?: float
}
Response: {
    source_member: str,
    target_member: str,
    missing_competences: List[CompetenceInfo],
    ...
}

# キャリアパス生成
POST /api/career/career-path
Request: {
    source_member: str,
    target_member?: str,
    target_role?: str,
    min_frequency?: float,
    min_total_score?: float
}
Response: {
    recommended_skills: List[RecommendedSkill],
    dependencies: Dict,
    ...
}
```

#### 役職ベース分析API (`role_based_dashboard.py`)

**役職スキル分析**:

```python
# 役職スキル統計取得
POST /api/career/role/role-skills
Request: {
    role: str,
    min_frequency: float
}
Response: {
    role: str,
    total_members: int,
    target_skills: List[SkillFrequency],
    priority_distribution: Dict
}

# 役職ベースギャップ分析
POST /api/career/role/gap-analysis
Request: {
    source_member: str,
    target_role: str,
    min_frequency: float
}
Response: GapAnalysisResult

# 役職ベースキャリアパス
POST /api/career/role/career-path
Request: {
    source_member: str,
    target_role: str,
    min_frequency: float,
    min_total_score: float
}
Response: CareerPathResult
```

### データモデル

#### Pydanticモデル

```python
class CompetenceInfo(BaseModel):
    competence_code: str
    competence_name: str
    category: str
    competence_type: str

class RecommendedSkill(BaseModel):
    competence_code: str
    competence_name: str
    category: str
    total_score: float
    readiness_score: float
    bayesian_score: float
    utility_score: float
    readiness_reasons: List[Tuple[str, float]]
    utility_reasons: List[Tuple[str, float]]

class GapAnalysisRequest(BaseModel):
    source_member: str
    target_member: Optional[str] = None
    target_role: Optional[str] = None
    min_frequency: Optional[float] = 0.3

class CareerPathRequest(BaseModel):
    source_member: str
    target_member: Optional[str] = None
    target_role: Optional[str] = None
    min_frequency: Optional[float] = 0.3
    min_total_score: Optional[float] = 0.02
```

---

## 開発ワークフロー

### フロントエンド開発

1. **新しいページの追加**

```typescript
// src/pages/NewPage.tsx
import React from 'react';

const NewPage: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold">New Page</h1>
      {/* コンテンツ */}
    </div>
  );
};

export default NewPage;
```

2. **ルーティング** (`App.tsx`で設定)

3. **コンポーネントの再利用**
   - 共通コンポーネントは `components/` に配置
   - Propsで動作をカスタマイズ

### バックエンド開発

1. **新しいAPIエンドポイントの追加**

```python
# backend/api/new_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/new", tags=["new"])

class NewRequest(BaseModel):
    param: str

@router.post("/endpoint")
async def new_endpoint(request: NewRequest):
    # 処理
    return {"result": "success"}
```

2. **ルーターの登録** (`main.py`):

```python
from api.new_api import router as new_router
app.include_router(new_router)
```

---

## デプロイ

### フロントエンドのビルド

```bash
cd frontend
npm run build
```

ビルド成果物は `dist/` ディレクトリに生成されます。

### バックエンドの本番起動

```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker対応

プロジェクトには `Dockerfile` と `docker-compose.yml` を追加することで、コンテナ化が可能です。

---

## トラブルシューティング

### フロントエンドが起動しない

**症状**: `npm run dev` でエラー

**対処法**:
```bash
# node_modulesを削除して再インストール
rm -rf node_modules package-lock.json
npm install
```

### APIリクエストが失敗する

**症状**: ネットワークエラー、CORS エラー

**対処法**:
1. バックエンドが起動しているか確認
2. `backend/main.py` のCORS設定を確認:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 型エラー

**症状**: TypeScriptコンパイルエラー

**対処法**:
- `tsconfig.json` の設定を確認
- 型定義を追加
- `any` 型の使用を最小限に

---

## パフォーマンス最適化

### フロントエンド

1. **コード分割**
   - React.lazy で動的インポート
   - ルートベースでの分割

2. **メモ化**
   - `React.memo` で再レンダリング抑制
   - `useMemo`, `useCallback` の活用

3. **バンドルサイズの削減**
   - Tree shaking
   - 不要なライブラリの削除

### バックエンド

1. **非同期処理**
   - `async`/`await` の活用
   - 並列処理の最適化

2. **キャッシング**
   - モデルの再計算を避ける
   - セッション管理

3. **データベース接続プーリング** (将来の拡張)

---

## 関連ドキュメント

- [Streamlitアプリガイド](STREAMLIT_APPS.md)
- [REST API リファレンス](API_REFERENCE.md)
- [アーキテクチャドキュメント](ARCHITECTURE.md)
- [機械学習モデル参考資料](ML_MODELS_REFERENCE.md)

---

## 今後の拡張

### 実装予定の機能

1. **認証・認可**
   - ユーザーログイン
   - ロールベースアクセス制御

2. **リアルタイム通知**
   - WebSocket対応
   - プッシュ通知

3. **データベース統合**
   - PostgreSQL/MySQL対応
   - データ永続化

4. **テスト自動化**
   - Jest (Frontend)
   - Pytest (Backend)
   - E2Eテスト

5. **CI/CDパイプライン**
   - GitHub Actions
   - 自動デプロイ
