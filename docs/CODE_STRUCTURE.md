# コード構造ガイド

このドキュメントでは、CareerNavigatorプロジェクトのコード構造と各モジュールの役割について説明します。

## プロジェクト構造

```
CareerNavigator/
├── skillnote_recommendation/       # メインパッケージ（バックエンドロジック）
│   ├── core/                       # コアビジネスロジック
│   ├── ml/                         # 機械学習モジュール
│   ├── graph/                      # グラフベース推薦・因果推論モジュール
│   ├── utils/                      # ユーティリティモジュール
│   └── scripts/                    # 実行スクリプト
│
├── frontend/                       # WebUI フロントエンド (React)
│   ├── src/
│   │   ├── components/             # UIコンポーネント
│   │   ├── pages/                  # ページコンポーネント
│   │   └── types/                  # TypeScript型定義
│   └── package.json
│
├── backend/                        # WebUI バックエンド (FastAPI)
│   ├── api/                        # APIエンドポイント
│   └── main.py                     # アプリケーションエントリーポイント
│
├── pages/                          # Streamlitページ
│   ├── 1_Causal_Recommendation.py      # 因果推論推薦
│   ├── 2_Employee_Career_Dashboard.py  # 従業員キャリアダッシュボード
│   ├── 3_Organizational_Skill_Map.py   # 組織スキルマップ
│
├── streamlit_app.py                # Streamlitメインアプリ（データ読み込み）
├── tests/                          # テストコード
└── docs/                           # ドキュメント
```

## モジュール詳細

### skillnote_recommendation/core/

基盤コンポーネントとデータ処理の実装。

#### 主要モジュール:

- **config_v2.py**: 設定管理（新設計）
  - 環境別設定 (dev/staging/prod)
  - 不変データクラスによる型安全性

- **models.py**: データモデル定義
  - Member, Competence, Recommendation
  - Pydanticモデルによるバリデーション

- **data_loader.py**: データ読み込み
  - CSVファイルの読み込みと検証
  - ディレクトリスキャン

- **evaluator.py**: 推薦システム評価
  - 精度評価 (Precision, Recall, NDCG)
  - 多様性評価

### skillnote_recommendation/ml/

機械学習ベースの推薦システム。

#### 主要モジュール:

- **matrix_factorization.py**: 行列分解モデル (NMF)
- **ml_recommender.py**: ML推薦システム統合
- **diversity.py**: 多様性再ランキング (MMR, Category, Type)

### skillnote_recommendation/graph/

グラフベース推薦と因果推論。

#### 主要モジュール:

- **knowledge_graph.py**: 知識グラフ構築
- **hybrid_recommender.py**: ハイブリッド推薦 (RWR + NMF + Content)
- **career_path.py**: キャリアパス分析

#### causal/ (因果推論モジュール)

- **causal_graph_recommender.py**: 因果グラフ推薦エンジン
  - LiNGAMによる因果探索
  - 3軸スコアリング (Readiness, Probability, Utility)
- **dependency_analyzer.py**: スキル依存関係分析
- **causal_career_path.py**: 因果パス生成

### backend/ (WebUI API)

FastAPIによるREST API実装。

- **main.py**: FastAPIアプリケーション設定
- **api/career_dashboard.py**: キャリアダッシュボード用API
- **api/role_based_dashboard.py**: 役職ベース分析API

### frontend/ (WebUI Frontend)

React + TypeScript + Viteによるモダンフロントエンド。

- **src/pages/EmployeeCareerDashboard.tsx**: 従業員ダッシュボード
- **src/components/**: 再利用可能なUIコンポーネント (Charts, Tables)

## Streamlitアプリケーション構造

### streamlit_app.py - データ読み込み
**機能**: CSVファイルのアップロード、データ変換、品質チェック、セッション初期化

### pages/1_Causal_Recommendation.py
**機能**: 因果推論に基づくスキル推薦
- 因果グラフの可視化 (Pyvis)
- 3軸スコアによる推薦リスト
- インタラクティブなパラメータ調整

### pages/2_Employee_Career_Dashboard.py
**機能**: 従業員個人のキャリア分析
- スキル保有状況の可視化 (レーダーチャート)
- キャリアパスシミュレーション
- ギャップ分析

### pages/3_Organizational_Skill_Map.py
**機能**: 組織全体のスキル分析
- スキルヒートマップ
- 部門別スキル分布
- 不足スキル分析

## コーディング規約とテスト

- **型ヒント**: 全ての関数に型ヒントを付与 (mypy対応)
- **ドキュメント**: Googleスタイルのdocstring
- **テスト**: pytestによる単体テスト (tests/ディレクトリ)
- **フォーマッター**: black, isort

## 関連ドキュメント

- [ARCHITECTURE.md](ARCHITECTURE.md) - アーキテクチャ詳細
- [STREAMLIT_APPS.md](STREAMLIT_APPS.md) - Streamlitアプリ詳細ガイド
- [WEBUI_GUIDE.md](WEBUI_GUIDE.md) - WebUI開発ガイド
- [ML_MODELS_REFERENCE.md](ML_MODELS_REFERENCE.md) - MLモデル詳細
- [CAUSAL_RECOMMENDATION.md](CAUSAL_RECOMMENDATION.md) - 因果推論詳細
