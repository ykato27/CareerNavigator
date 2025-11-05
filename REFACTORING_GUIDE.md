# リファクタリングガイド

## 概要

GAFAレベルのソフトウェアエンジニアリング標準に基づき、CareerNavigatorプロジェクトの大規模リファクタリングを実施しました。

**バージョン**: v1.2.0
**実施日**: 2025-10-24
**対象**: エンタープライズグレード機能の追加（基盤強化、アーキテクチャ改善、セキュリティ・品質向上）

---

## 主な改善内容

### Phase 1: 基盤強化

#### 1.1 構造化ロギング導入

**ファイル**: `skillnote_recommendation/core/logging_config.py`

**改善内容**:
- ❌ **旧**: 非構造化ログ（文字列ベース）
- ✅ **新**: 構造化ログ（JSON形式、検索・集計可能）

**メリット**:
- ログの検索・分析が容易
- トレースID対応（将来拡張）
- 本番環境でのデバッグ効率向上

**使用例**:
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

#### 1.2 エラーハンドリング体系化

**ファイル**: `skillnote_recommendation/core/errors.py`

**改善内容**:
- エラーコード体系（D001-D099: データ、M001-M099: モデル、等）
- リトライ可能性の判定
- 構造化されたエラーコンテキスト

**主な例外クラス**:
```python
DataNotFoundError          # D001: データが見つからない
DataInvalidError           # D002: データが不正
ModelNotTrainedError       # M001: モデルが未学習
ColdStartError             # R001: コールドスタート問題
InvalidParameterError      # R002: パラメータが不正
ConfigurationError         # S001: 設定エラー
```

**後方互換性**:
- `MLModelNotTrainedError` → `ModelNotTrainedError` (非推奨だが利用可能)

#### 1.3 リトライロジック

**ファイル**: `skillnote_recommendation/core/retry.py`

**改善内容**:
- 指数バックオフによる自動リトライ
- リトライ可能なエラーの自動判定

**使用例**:
```python
from skillnote_recommendation.core.retry import with_retry

@with_retry(max_attempts=3)
def fetch_external_data():
    # リトライ可能なエラーの場合、最大3回リトライ
    ...
```

---

### Phase 2: アーキテクチャ改善

#### 2.1 Config管理リファクタリング

**ファイル**: `skillnote_recommendation/core/config_v2.py`

**改善内容**:
- ❌ **旧**: グローバル可変状態（`class Config`）
- ✅ **新**: 不変（frozen）dataclass + 環境分離

**環境別設定**:
```python
from skillnote_recommendation.core.config_v2 import Config

# 開発環境
config_dev = Config.from_env("dev")

# 本番環境
config_prod = Config.from_env("prod")

# 環境変数から取得
config = Config.from_env()  # APP_ENV環境変数から取得
```

**メリット**:
- テスト容易性の向上
- 環境ごとの設定分離
- 型安全性
- 並行処理安全

**主な設定クラス**:
- `DirectoryConfig`: ディレクトリ設定
- `MFParams`: Matrix Factorizationパラメータ
- `OptunaParams`: ハイパーパラメータチューニング設定
- `EvaluationParams`: 評価設定
- `LoggingParams`: ログ設定

#### 2.2 インターフェース定義

**ファイル**: `skillnote_recommendation/core/interfaces.py`

**改善内容**:
- Protocol（構造的部分型）による柔軟な型定義
- ABC（抽象基底クラス）による契約定義
- 依存性注入の基盤

**主なインターフェース**:
```python
# Protocol（構造的部分型）
MatrixFactorizationProtocol
DiversityRerankerProtocol
ReferencePersonFinderProtocol
DataPreprocessorProtocol
EvaluatorProtocol

# ABC（抽象基底クラス）
BaseRecommender
BaseHyperparameterTuner
```

**メリット**:
- テスト時のモック作成が容易
- 実装の切り替えが容易
- インターフェースの明示化

---

### Phase 3: セキュリティ・品質

#### 3.1 モデル保存形式変更

**ファイル**: `skillnote_recommendation/ml/model_serialization.py`

**改善内容**:
- ❌ **旧**: pickle（セキュリティリスク、非人間可読）
- ✅ **新**: joblib（行列データ） + JSON（メタデータ）

**保存形式**:
```
model_name.json      # メタデータ（人間可読、監査可能）
model_name.joblib    # 行列データ（圧縮、安全）
```

**メリット**:
- セキュリティリスクの軽減
- バージョン管理
- 監査可能性
- 人間可読なメタデータ

**後方互換性**:
- `.pkl`形式のモデルも読み込み可能

**使用例**:
```python
from skillnote_recommendation.ml.model_serialization import ModelSerializer

# 保存
ModelSerializer.save_matrix_factorization_model(
    filepath="models/nmf_model",
    W=W, H=H,
    member_codes=member_codes,
    competence_codes=competence_codes,
    ...
)

# 読み込み
model_data = ModelSerializer.load_matrix_factorization_model("models/nmf_model")
```

#### 3.2 入力バリデーション強化

**ファイル**: `skillnote_recommendation/core/schemas.py`

**改善内容**:
- Pydanticによる型安全な入力検証
- 自動バリデーション
- 明示的なエラーメッセージ

**主なスキーマ**:
```python
PredictionRequest              # 予測リクエスト
TopKPredictionRequest          # Top-K推薦リクエスト
RecommendationRequest          # 推薦リクエスト
HyperparameterTuningRequest    # チューニングリクエスト
EvaluationRequest              # 評価リクエスト
DataQualityReport              # データ品質レポート
```

**使用例**:
```python
from skillnote_recommendation.core.schemas import RecommendationRequest
from pydantic import ValidationError

try:
    request = RecommendationRequest(
        member_code="M001",
        top_n=10,
        diversity_strategy="hybrid"
    )
except ValidationError as e:
    print(e.errors())
```

#### 3.3 データ品質チェック

**ファイル**: `skillnote_recommendation/core/data_quality.py`

**改善内容**:
- 欠損値・無限値検出
- データ分布検証
- スパース性チェック
- 負の値チェック（NMF要件）

**使用例**:
```python
from skillnote_recommendation.core.data_quality import DataQualityChecker

checker = DataQualityChecker()
report = checker.validate_skill_matrix(skill_matrix, strict=False)

if not report.is_valid:
    print("Errors:", report.errors)
    print("Warnings:", report.warnings)
    print("Statistics:", report.statistics)
```

**検証項目**:
- ✅ 形状チェック（最小サイズ）
- ✅ 欠損値チェック
- ✅ 無限値チェック
- ✅ 負の値チェック
- ✅ スパース性チェック
- ✅ データ型チェック
- ✅ 分布チェック（外れ値）

---

## 依存関係の追加

**requirements.txt** / **pyproject.toml**:
```
structlog>=23.0.0          # 構造化ロギング
pydantic>=2.0.0            # バリデーション
python-json-logger>=2.0.0  # JSONログ出力
tenacity>=8.0.0            # リトライロジック
joblib>=1.3.0              # モデル保存
```

---

## 型チェック設定の厳格化

**pyproject.toml**:
```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
no_implicit_optional = true
```

---

## マイグレーションガイド

### 既存コードの移行

#### 1. ロギングの移行

**旧**:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing member {member_code}")
```

**新**:
```python
from skillnote_recommendation.core.logging_config import get_logger
logger = get_logger(__name__)
logger.info("processing_member", member_code=member_code)
```

#### 2. 例外処理の移行

**旧**:
```python
from skillnote_recommendation.ml.exceptions import ColdStartError
```

**新**:
```python
from skillnote_recommendation.core.errors import ColdStartError
```

#### 3. Config使用の移行

**旧**:
```python
from skillnote_recommendation.core.config import Config
params = Config.MF_PARAMS
```

**新**:
```python
from skillnote_recommendation.core.config_v2 import Config
config = Config.from_env("dev")
params = config.mf
```

#### 4. モデル保存の移行

**旧**:
```python
model.save("model.pkl")  # pickle使用
```

**新**:
```python
from skillnote_recommendation.ml.model_serialization import ModelSerializer
ModelSerializer.save_matrix_factorization_model("model", ...)
```

---

## テスト戦略

### 新しいモジュールのテスト

```python
# tests/test_errors.py
def test_cold_start_error():
    error = ColdStartError("M001")
    assert error.code == ErrorCode.COLD_START
    assert error.retryable == False
    assert "M001" in str(error)

# tests/test_schemas.py
def test_recommendation_request_validation():
    # 正常系
    request = RecommendationRequest(
        member_code="M001",
        top_n=10
    )
    assert request.top_n == 10

    # 異常系
    with pytest.raises(ValidationError):
        RecommendationRequest(
            member_code="",  # 空文字列はエラー
            top_n=10
        )

# tests/test_data_quality.py
def test_validate_skill_matrix():
    checker = DataQualityChecker()

    # 正常なマトリクス
    matrix = pd.DataFrame(...)
    report = checker.validate_skill_matrix(matrix)
    assert report.is_valid

    # NaNを含むマトリクス
    matrix_with_nan = matrix.copy()
    matrix_with_nan.iloc[0, 0] = np.nan
    report = checker.validate_skill_matrix(matrix_with_nan)
    assert not report.is_valid
    assert "NaN" in report.errors[0]
```

---

## 今後の推奨事項

### Phase 4: テスト・運用（未実施）

1. **テストカバレッジ向上**
   - 目標: 80%以上
   - パラメトライズドテスト
   - プロパティベースドテスト（Hypothesis）

2. **CI/CDパイプライン構築**
   - GitHub Actions設定
   - 自動テスト実行
   - 静的解析（mypy, flake8）

3. **モニタリング・観測性**
   - メトリクス収集（Prometheus）
   - APMツール統合（DataDog）
   - アラート設定

---

## 参考資料

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [structlog Documentation](https://www.structlog.org/)
- [tenacity Documentation](https://tenacity.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Twelve-Factor App](https://12factor.net/)

---

## 変更履歴

| 日付 | バージョン | 概要 |
|------|-----------|------|
| 2025-10-24 | v1.2.0 | エンタープライズグレード機能の追加（ロギング、エラーハンドリング、Config管理、バリデーション、データ品質チェック等） |

---

## お問い合わせ

リファクタリングに関する質問や問題がある場合は、開発チームまでお問い合わせください。
