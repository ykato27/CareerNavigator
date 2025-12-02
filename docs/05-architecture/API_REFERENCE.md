# APIリファレンス v2.0.0

CareerNavigator v2.0.0 で追加された新しいAPIの使用方法

---

## 目次

1. [構造化ロギング](#構造化ロギング)
2. [エラーハンドリング](#エラーハンドリング)
3. [Config管理](#config管理)
4. [データバリデーション](#データバリデーション)
5. [データ品質チェック](#データ品質チェック)
6. [モデルシリアライゼーション](#モデルシリアライゼーション)
7. [リトライロジック](#リトライロジック)

---

## 構造化ロギング

### 概要

JSON形式の検索・集計可能なログを出力します。

### 基本的な使用

```python
from skillnote_recommendation.core.logging_config import get_logger, setup_structured_logging

# ロギング設定（アプリケーション起動時に1回だけ）
setup_structured_logging(
    log_level="INFO",
    enable_json=True,  # JSON形式（本番環境）
    enable_console=True
)

# ロガーの取得
logger = get_logger(__name__)

# 構造化ログの出力
logger.info(
    "user_login",  # イベント名
    user_id="U123",
    ip_address="192.168.1.1",
    login_method="oauth"
)

logger.error(
    "recommendation_failed",
    member_code="M001",
    error_code="R001",
    reason="Cold start problem"
)
```

### LoggerMixinの使用

```python
from skillnote_recommendation.core.logging_config import LoggerMixin

class MyRecommender(LoggerMixin):
    def recommend(self, member_code: str):
        self.logger.info(
            "recommendation_started",
            member_code=member_code
        )
        # ...
        self.logger.info(
            "recommendation_completed",
            member_code=member_code,
            recommendations_count=10
        )
```

### 環境別設定

```python
# 開発環境: 人間可読形式
setup_structured_logging(
    log_level="DEBUG",
    enable_json=False
)

# 本番環境: JSON形式
setup_structured_logging(
    log_level="INFO",
    enable_json=True
)
```

---

## エラーハンドリング

### 概要

エラーコード体系と構造化されたエラーコンテキストを提供します。

### エラー一覧

| エラーコード | クラス名 | 説明 | リトライ可能 |
|-------------|---------|------|-------------|
| D001 | DataNotFoundError | データが見つからない | No |
| D002 | DataInvalidError | データが不正 | No |
| M001 | ModelNotTrainedError | モデルが未学習 | No |
| R001 | ColdStartError | コールドスタート問題 | No |
| R002 | InvalidParameterError | パラメータが不正 | No |
| E001 | ExternalServiceError | 外部サービスエラー | Yes |
| S001 | ConfigurationError | 設定エラー | No |

### 基本的な使用

```python
from skillnote_recommendation.core.errors import (
    ColdStartError,
    ModelNotTrainedError,
    InvalidParameterError
)

# エラーの発生
if member_code not in training_data:
    raise ColdStartError(
        member_code=member_code,
        suggestion="Add member data or use content-based fallback"
    )

# エラーのキャッチ
try:
    recommendations = model.predict(member_code)
except ColdStartError as e:
    print(f"Error: {e}")
    print(f"Code: {e.code.value}")
    print(f"Context: {e.context}")
    print(f"Retryable: {e.retryable}")
```

### カスタムエラーの作成

```python
from skillnote_recommendation.core.errors import RecommendationError, ErrorCode

# カスタムエラーコードの追加（Enumを拡張）
# または既存のエラーコードを使用

raise RecommendationError(
    code=ErrorCode.RECOMMENDATION_FAILED,
    message="Diversity reranking failed",
    retryable=True,
    diversity_strategy="hybrid",
    candidate_count=0
)
```

### エラー情報の取得

```python
try:
    # ...
except RecommendationError as e:
    error_dict = e.to_dict()
    # {
    #     'error_code': 'R001',
    #     'message': '...',
    #     'retryable': False,
    #     'context': {...}
    # }
```

---

## Config管理

### 概要

不変（frozen）設計で環境分離に対応した設定管理です。

### 基本的な使用

```python
from skillnote_recommendation.core.config_v2 import Config

# 開発環境
config_dev = Config.from_env("dev")

# 本番環境
config_prod = Config.from_env("prod")

# 環境変数から取得（APP_ENV）
config = Config.from_env()  # デフォルト: "dev"

# デフォルト設定
config = Config.default()
```

### 設定へのアクセス

```python
# Matrix Factorizationパラメータ
mf_params = config.mf
print(mf_params.n_components)  # 20
print(mf_params.max_iter)      # 1000

# 評価パラメータ
eval_params = config.evaluation
print(eval_params.top_k)       # 10
print(eval_params.train_ratio) # 0.8

# ディレクトリ設定
data_dir = config.directories.data_dir
output_dir = config.directories.output_dir

# ログ設定
log_level = config.logging.level
enable_json = config.logging.enable_json
```

### パスの取得

```python
# 入力ディレクトリ
members_dir = config.get_input_dir("members")
# → Path("/path/to/data/members")

# 出力ファイル
output_path = config.get_output_path("skill_matrix")
# → Path("/path/to/output/skill_matrix.csv")
```

### ディレクトリの作成

```python
config.ensure_directories()
# data/ と output/ ディレクトリを作成
```

### カスタム設定の作成

```python
from skillnote_recommendation.core.config_v2 import (
    Config,
    MFParams,
    LoggingParams
)

# カスタムパラメータ
custom_mf = MFParams(
    n_components=30,
    max_iter=2000,
    random_state=123
)

custom_logging = LoggingParams(
    level="DEBUG",
    enable_json=False
)

# カスタム設定
config = Config(
    environment=Environment.DEVELOPMENT,
    directories=DirectoryConfig.from_project_root(Path(".")),
    mf=custom_mf,
    logging=custom_logging
)
```

---

## データバリデーション

### 概要

Pydanticベースの型安全な入力検証です。

### 推薦リクエストの検証

```python
from skillnote_recommendation.core.schemas import RecommendationRequest
from pydantic import ValidationError

# 正常なリクエスト
try:
    request = RecommendationRequest(
        member_code="M001",
        top_n=10,
        competence_type=["SKILL", "EDUCATION"],
        diversity_strategy="hybrid"
    )
    print(f"Valid request: {request}")
except ValidationError as e:
    print(f"Validation error: {e.errors()}")

# 異常なリクエスト（自動検証）
try:
    invalid_request = RecommendationRequest(
        member_code="",  # 空文字列 → エラー
        top_n=1000,      # 上限超過 → エラー
        diversity_strategy="invalid"  # 不正な値 → エラー
    )
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}, Error: {error['msg']}")
```

### Top-K予測リクエストの検証

```python
from skillnote_recommendation.core.schemas import TopKPredictionRequest

request = TopKPredictionRequest(
    member_code="M001",
    k=20,
    exclude_acquired=True,
    acquired_competences=["C001", "C002", "C003"]
)
```

### 評価リクエストの検証

```python
from skillnote_recommendation.core.schemas import EvaluationRequest

eval_request = EvaluationRequest(
    top_k=10,
    train_ratio=0.8,
    use_temporal_split=True,
    include_extended_metrics=True
)
```

### カスタムバリデータ

```python
from pydantic import BaseModel, Field, field_validator

class CustomRequest(BaseModel):
    value: int = Field(..., ge=0, le=100)

    @field_validator('value')
    @classmethod
    def validate_even_number(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError('Value must be even')
        return v
```

---

## データ品質チェック

### 概要

データの品質を包括的に検証し、レポートを生成します。

### スキルマトリックスの検証

```python
from skillnote_recommendation.core.data_quality import DataQualityChecker
import pandas as pd

checker = DataQualityChecker()

# スキルマトリックスの検証
skill_matrix = pd.DataFrame(...)  # メンバー×力量マトリックス
report = checker.validate_skill_matrix(
    skill_matrix,
    strict=False  # 警告を許容
)

# レポートの確認
if not report.is_valid:
    print("❌ Data quality issues found:")
    for error in report.errors:
        print(f"  - {error}")

if report.warnings:
    print("⚠️ Warnings:")
    for warning in report.warnings:
        print(f"  - {warning}")

# 統計情報
print(f"Sparsity: {report.statistics['sparsity']:.2%}")
print(f"Mean (non-zero): {report.statistics['mean']:.3f}")
print(f"Std (non-zero): {report.statistics['std']:.3f}")
```

### DataFrameの検証

```python
# 一般的なDataFrameの検証
report = checker.validate_dataframe(
    df=member_competence,
    required_columns=["メンバーコード", "力量コード", "正規化レベル"],
    name="Member Competence"
)

if not report.is_valid:
    print(f"Errors: {report.errors}")
```

### 厳格モード

```python
# 厳格モード: 警告もエラーとして扱う
report = checker.validate_skill_matrix(
    skill_matrix,
    strict=True
)
```

---

## モデルシリアライゼーション

### 概要

安全で監査可能なモデル保存・読み込み機能です。

### モデルの保存

```python
from skillnote_recommendation.ml.model_serialization import ModelSerializer

# Matrix Factorizationモデルの保存
ModelSerializer.save_matrix_factorization_model(
    filepath="models/nmf_model",  # 拡張子なし
    W=W,  # メンバー因子行列
    H=H,  # 力量因子行列
    member_codes=member_codes,
    competence_codes=competence_codes,
    member_index=member_index,
    competence_index=competence_index,
    params={'n_components': 20, 'max_iter': 1000},
    reconstruction_err=0.123,
    n_iter=150
)

# 以下のファイルが生成される：
# - models/nmf_model.json     # メタデータ（人間可読）
# - models/nmf_model.joblib   # 行列データ（圧縮）
```

### モデルの読み込み

```python
# モデルの読み込み
model_data = ModelSerializer.load_matrix_factorization_model(
    filepath="models/nmf_model"
)

# データの取得
W = model_data['W']
H = model_data['H']
member_codes = model_data['member_codes']
params = model_data['params']
version = model_data['version']

print(f"Model version: {version}")
print(f"Created at: {model_data['created_at']}")
print(f"Reconstruction error: {model_data['metrics']['reconstruction_error']}")
```

### 互換性検証

```python
saved_params = model_data['params']
current_params = {'n_components': 20, 'init': 'nndsvda'}

is_compatible, warnings = ModelSerializer.validate_model_compatibility(
    saved_params,
    current_params
)

if not is_compatible:
    print("⚠️ Model compatibility warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

---

## リトライロジック

### 概要

指数バックオフによる自動リトライ機能です。

### デコレータの使用

```python
from skillnote_recommendation.core.retry import with_retry

@with_retry(max_attempts=3, min_wait_seconds=1, max_wait_seconds=10)
def fetch_external_data():
    # リトライ可能なエラーの場合、自動的にリトライ
    # (RecommendationError でretryable=Trueのエラー)
    ...
```

### 特定例外でのリトライ

```python
from skillnote_recommendation.core.retry import with_retry_on_exception
from skillnote_recommendation.core.errors import ExternalServiceError, NetworkError

@with_retry_on_exception(
    (ExternalServiceError, NetworkError),
    max_attempts=3
)
def call_external_api():
    # ExternalServiceError または NetworkError の場合のみリトライ
    ...
```

### リトライ可能なエラーの判定

```python
from skillnote_recommendation.core.retry import is_retryable_error
from skillnote_recommendation.core.errors import ColdStartError, ExternalServiceError

error1 = ColdStartError("M001")
error2 = ExternalServiceError("api", "Timeout")

print(is_retryable_error(error1))  # False
print(is_retryable_error(error2))  # True
```

---

## 使用例

### フルスタック例: ML推薦システム

```python
from skillnote_recommendation.core.config_v2 import Config
from skillnote_recommendation.core.logging_config import setup_structured_logging, get_logger
from skillnote_recommendation.core.data_quality import DataQualityChecker
from skillnote_recommendation.core.schemas import RecommendationRequest
from skillnote_recommendation.core.errors import ColdStartError
from skillnote_recommendation.ml import MLRecommender
from pydantic import ValidationError

# 1. 設定とロギングのセットアップ
config = Config.from_env("prod")
setup_structured_logging(
    log_level=config.logging.level,
    enable_json=config.logging.enable_json
)
logger = get_logger(__name__)

# 2. データ品質チェック
checker = DataQualityChecker()
report = checker.validate_skill_matrix(skill_matrix)
if not report.is_valid:
    logger.error("data_quality_failed", errors=report.errors)
    raise DataInvalidError("Data quality check failed", quality_issues=report.errors)

# 3. 入力バリデーション
try:
    request = RecommendationRequest(
        member_code="M001",
        top_n=10,
        diversity_strategy="hybrid"
    )
except ValidationError as e:
    logger.error("validation_failed", errors=e.errors())
    raise

# 4. 推薦生成
try:
    recommender = MLRecommender.build(...)
    recommendations = recommender.recommend(
        member_code=request.member_code,
        top_n=request.top_n,
        diversity_strategy=request.diversity_strategy
    )

    logger.info(
        "recommendation_success",
        member_code=request.member_code,
        recommendations_count=len(recommendations)
    )

except ColdStartError as e:
    logger.warning("cold_start_detected", member_code=e.context['member_code'])
    # フォールバック処理
```

---

## 参考資料

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [structlog Documentation](https://www.structlog.org/)
- [tenacity Documentation](https://tenacity.readthedocs.io/)
- [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
