# システムアーキテクチャ

CareerNavigator推薦システムのアーキテクチャドキュメント

**バージョン**: 2.0.0
**最終更新**: 2025-11-05

---

## 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [レイヤー構成](#レイヤー構成)
3. [モジュール構成](#モジュール構成)
4. [データフロー](#データフロー)
5. [主要コンポーネント](#主要コンポーネント)
6. [設計原則](#設計原則)

---

## アーキテクチャ概要

CareerNavigatorは**階層型アーキテクチャ**を採用し、関心の分離を実現しています。

```
┌─────────────────────────────────────────┐
│        Presentation Layer               │
│  (Streamlit UI / CLI / API)             │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│        Application Layer                │
│  (推薦システム / 評価システム)             │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│        Domain Layer                     │
│  (機械学習ベース推薦 / グラフベース推薦)   │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│        Infrastructure Layer             │
│  (データ読込 / ログ / エラー / Config)    │
└─────────────────────────────────────────┘
```

---

## レイヤー構成

### 1. Presentation Layer（プレゼンテーション層）

**責務**: ユーザーインターフェース、入出力制御

**コンポーネント**:
- `streamlit_app.py`: Streamlit Webアプリケーション
- Streamlit Pages: 各種分析・推薦ページ

**依存関係**: Application Layer

---

### 2. Application Layer（アプリケーション層）

**責務**: ユースケース実装、ビジネスロジックの調整

**コンポーネント**:
- `core/evaluator.py`: 評価システム
- `ml/ml_evaluation.py`: ML評価システム

**依存関係**: Domain Layer, Infrastructure Layer

---

### 3. Domain Layer（ドメイン層）

**責務**: コアビジネスロジック、推薦アルゴリズム

**コンポーネント**:

#### 機械学習ベース推薦
- `ml/matrix_factorization.py`: Matrix Factorizationモデル
- `ml/ml_recommender.py`: ML推薦エンジン
- `ml/diversity.py`: 多様性再ランキング
- `ml/hyperparameter_tuning.py`: ハイパーパラメータチューニング
- `core/reference_persons.py`: 参考人物検索

#### グラフベース推薦
- `graph/knowledge_graph.py`: 知識グラフ構築
- `graph/hybrid_recommender.py`: ハイブリッド推薦
- `graph/career_path.py`: キャリアパス分析

**依存関係**: Infrastructure Layer

---

### 4. Infrastructure Layer（インフラストラクチャ層）

**責務**: 技術的基盤、横断的関心事

**コンポーネント**:

#### データ管理
- `core/data_loader.py`: データ読み込み
- `core/data_transformer.py`: データ変換
- `ml/data_preprocessing.py`: ML用データ前処理

#### 設定・ログ・エラー
- `core/config.py` (旧) / `core/config_v2.py` (新): 設定管理
- `core/logging_config.py`: 構造化ロギング
- `core/errors.py`: エラー定義
- `core/retry.py`: リトライロジック

#### バリデーション・品質
- `core/schemas.py`: 入力バリデーション（Pydantic）
- `core/data_quality.py`: データ品質チェック
- `ml/model_serialization.py`: モデルシリアライゼーション

#### インターフェース定義
- `core/interfaces.py`: Protocol / ABC定義
- `core/models.py`: データモデル

**依存関係**: なし（最下層）

---

## モジュール構成

```
skillnote_recommendation/
├── core/                          # コアモジュール
│   ├── config.py                  # 旧Config（後方互換性）
│   ├── config_v2.py               # 新Config（不変設計）
│   ├── models.py                  # データモデル
│   ├── interfaces.py              # インターフェース定義
│   │
│   ├── data_loader.py             # データ読み込み
│   ├── data_transformer.py        # データ変換
│   ├── data_quality.py            # データ品質チェック
│   │
│   ├── logging_config.py          # 構造化ロギング
│   ├── errors.py                  # エラー定義
│   ├── retry.py                   # リトライロジック
│   ├── schemas.py                 # バリデーションスキーマ
│   │
│   ├── reference_persons.py       # 参考人物検索
│   └── evaluator.py               # 評価システム
│
├── ml/                            # MLモジュール
│   ├── matrix_factorization.py    # NMFモデル
│   ├── ml_recommender.py          # ML推薦エンジン
│   ├── diversity.py               # 多様性再ランキング
│   ├── hyperparameter_tuning.py   # ハイパーパラメータ最適化
│   ├── data_preprocessing.py      # データ前処理
│   ├── model_serialization.py     # モデル保存
│   ├── ml_evaluation.py           # ML評価
│   └── exceptions.py              # ML例外（非推奨）
│
├── graph/                         # グラフモジュール
│   ├── knowledge_graph.py         # ナレッジグラフ
│   └── ...
│
└── scripts/                       # スクリプト
    ├── convert_data.py            # データ変換
    ├── run_recommendation.py      # 推薦実行
    └── evaluate_recommendations.py # 評価実行
```

---

## データフロー

### 1. データ変換フロー

```
CSVファイル（data/）
    ↓ DataLoader
生データ（DataFrame）
    ↓ DataTransformer
変換データ（output/）
    ↓
- members_clean.csv
- competence_master.csv
- member_competence.csv
- skill_matrix.csv
- competence_similarity.csv
```

### 2. ML推薦フロー

```
メンバー×力量マトリックス
    ↓ DataPreprocessor（前処理）
前処理済みマトリックス
    ↓ MatrixFactorizationModel（NMF学習）
学習済みモデル（W, H行列）
    ↓ MLRecommender
予測スコア
    ↓ DiversityReranker（多様性再ランキング）
推薦結果（Recommendation[]）
```

### 3. 評価フロー

```
メンバー習得力量データ
    ↓ Evaluator.temporal_train_test_split
訓練データ / テストデータ
    ↓ Evaluator.evaluate_recommendations
評価メトリクス（Precision@K, Recall@K, NDCG@K, etc.）
    ↓ Evaluator.calculate_diversity_metrics
多様性メトリクス（Gini, Novelty, Coverage, etc.）
```

---

## 主要コンポーネント

### データローダー（DataLoader）

**責務**: CSVデータの読み込みと結合

**特徴**:
- ディレクトリ内の全CSVファイルを自動検出
- カラム構造の整合性検証
- エンコーディング自動判定

**インターフェース**:
```python
loader = DataLoader()
data = loader.load_all_data()
# → dict with keys: 'members', 'member_competence', 'competence_master', etc.
```

---

### ML推薦エンジン（MLRecommender）

**責務**: 機械学習ベースの推薦

**アルゴリズム**:
1. **Matrix Factorization（NMF）**: メンバー×力量マトリックスを潜在因子に分解
2. **予測**: 未習得力量のスコアを予測
3. **多様性再ランキング**: MMR/Category/Type/Hybridで多様化

**インターフェース**:
```python
recommender = MLRecommender.build(
    member_competence=df_member_competence,
    competence_master=df_competence_master,
    member_master=df_member_master
)
recommendations = recommender.recommend(
    member_code='M001',
    top_n=10,
    use_diversity=True,
    diversity_strategy='hybrid'
)
```

---

### 評価システム（Evaluator）

**責務**: 推薦システムの性能評価

**メトリクス**:
- **精度**: Precision@K, Recall@K, F1@K, NDCG@K
- **ランキング**: MRR, MAP
- **多様性**: Gini Index, Novelty, Coverage

**分割方法**:
- Temporal Split（時系列分割）
- Random User Split
- Leave-One-Out

**インターフェース**:
```python
evaluator = RecommendationEvaluator()
metrics = evaluator.evaluate_recommendations(
    train_data=train,
    test_data=test,
    competence_master=master,
    top_k=10
)
```

---

### Config管理（Config / Config V2）

**責務**: システム設定の管理

**新設計（v2.0.0）**:
- 不変（frozen）dataclass
- 環境分離（dev/staging/prod）
- 型安全性

**インターフェース**:
```python
# 新（推奨）
from skillnote_recommendation.core.config_v2 import Config
config = Config.from_env("dev")
params = config.mf  # MFParams（不変）

# 旧（後方互換性）
from skillnote_recommendation.core.config import Config
params = Config.MF_PARAMS  # dict（可変）
```

---

### エラーハンドリング（Errors）

**責務**: 統一的なエラー管理

**エラーコード体系**:
- D001-D099: データ関連
- M001-M099: モデル関連
- R001-R099: 推薦関連
- E001-E099: 外部サービス
- S001-S099: システム

**インターフェース**:
```python
from skillnote_recommendation.core.errors import ColdStartError

raise ColdStartError(
    member_code="M001",
    suggestion="Add member data or use content-based fallback"
)
```

---

### ロギング（Logging）

**責務**: 構造化ログ出力

**特徴**:
- JSON形式（本番環境）
- 人間可読形式（開発環境）
- トレースID対応（将来拡張）

**インターフェース**:
```python
from skillnote_recommendation.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info(
    "recommendation_generated",
    member_code="M001",
    top_n=10,
    diversity_score=0.85
)
```

---

## 設計原則

### 1. 関心の分離（Separation of Concerns）

各レイヤーは明確な責務を持ち、他のレイヤーに干渉しません。

### 2. 依存性の逆転（Dependency Inversion）

上位レイヤーは下位レイヤーに依存しますが、具体実装ではなくインターフェース（Protocol/ABC）に依存します。

```python
# Good: インターフェースに依存
def recommend(
    mf_model: MatrixFactorizationProtocol,
    reranker: DiversityRerankerProtocol
) -> list[Recommendation]:
    ...

# Bad: 具体実装に依存
def recommend(
    mf_model: MatrixFactorizationModel,
    reranker: DiversityReranker
) -> list[Recommendation]:
    ...
```

### 3. 単一責任の原則（Single Responsibility）

各クラス・モジュールは1つの責務のみを持ちます。

### 4. 開放閉鎖の原則（Open/Closed）

拡張に対しては開いており、修正に対しては閉じています。

### 5. 不変性（Immutability）

設定やモデルデータは不変（frozen）とし、並行処理の安全性を確保します。

### 6. 型安全性（Type Safety）

mypy strict modeを有効化し、静的型チェックを実施します。

### 7. テスト容易性（Testability）

依存性注入とインターフェースにより、モック/スタブを用いたテストが容易です。

---

## 今後の拡張

### Phase 4: テスト・運用（未実装）

1. **テストカバレッジ80%以上**
2. **CI/CDパイプライン構築**
3. **モニタリング・観測性**

### 将来的な機能拡張

1. **リアルタイム推薦**: ストリーミングデータ対応
2. **A/Bテスト基盤**: 推薦戦略の比較実験
3. **マルチテナント対応**: 複数組織での利用
4. **API化**: REST/GraphQL APIの提供

---

## 参考資料

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
