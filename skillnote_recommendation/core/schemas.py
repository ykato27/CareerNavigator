"""
入力バリデーションスキーマ

Pydanticを使用した型安全な入力検証
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


# =============================================================================
# 型エイリアス
# =============================================================================

DiversityStrategy = Literal["mmr", "category", "type", "hybrid"]
CompetenceType = Literal["SKILL", "EDUCATION", "LICENSE"]


# =============================================================================
# Matrix Factorization関連スキーマ
# =============================================================================

class PredictionRequest(BaseModel):
    """予測リクエスト"""
    model_config = ConfigDict(frozen=True)

    member_code: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[A-Za-z0-9_-]+$',
        description="メンバーコード（英数字、アンダースコア、ハイフンのみ）"
    )
    competence_codes: list[str] | None = Field(
        None,
        max_length=1000,
        description="予測対象の力量コードリスト"
    )

    @field_validator('member_code')
    @classmethod
    def validate_member_code_no_whitespace(cls, v: str) -> str:
        """メンバーコードに空白がないことを検証"""
        if v.strip() != v:
            raise ValueError('Member code cannot have leading/trailing whitespace')
        return v

    @field_validator('competence_codes')
    @classmethod
    def validate_competence_codes_unique(cls, v: list[str] | None) -> list[str] | None:
        """力量コードが重複していないことを検証"""
        if v is not None and len(v) != len(set(v)):
            raise ValueError('Competence codes must be unique')
        return v


class TopKPredictionRequest(BaseModel):
    """Top-K推薦リクエスト"""
    model_config = ConfigDict(frozen=True)

    member_code: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[A-Za-z0-9_-]+$'
    )
    k: int = Field(
        10,
        ge=1,
        le=100,
        description="推薦数（1-100）"
    )
    exclude_acquired: bool = Field(
        True,
        description="既習得力量を除外するか"
    )
    acquired_competences: list[str] | None = Field(
        None,
        description="既習得力量のリスト"
    )

    @field_validator('member_code')
    @classmethod
    def validate_member_code(cls, v: str) -> str:
        if v.strip() != v:
            raise ValueError('Member code cannot have leading/trailing whitespace')
        return v


# =============================================================================
# Recommender関連スキーマ
# =============================================================================

class RecommendationRequest(BaseModel):
    """推薦リクエスト"""
    model_config = ConfigDict(frozen=True)

    member_code: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[A-Za-z0-9_-]+$'
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description="推薦数"
    )
    competence_type: list[CompetenceType] | CompetenceType | None = Field(
        None,
        description="力量タイプフィルタ"
    )
    category_filter: str | None = Field(
        None,
        max_length=200,
        description="カテゴリフィルタ"
    )
    use_diversity: bool = Field(
        True,
        description="多様性を考慮するか"
    )
    diversity_strategy: DiversityStrategy = Field(
        "hybrid",
        description="多様性戦略"
    )

    @field_validator('member_code')
    @classmethod
    def validate_member_code(cls, v: str) -> str:
        if v.strip() != v:
            raise ValueError('Member code cannot have leading/trailing whitespace')
        return v

    @field_validator('competence_type')
    @classmethod
    def normalize_competence_type(
        cls,
        v: list[CompetenceType] | CompetenceType | None
    ) -> list[CompetenceType] | None:
        """力量タイプを正規化（文字列の場合はリストに変換）"""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]  # type: ignore
        return v


# =============================================================================
# Hyperparameter Tuning関連スキーマ
# =============================================================================

class HyperparameterTuningRequest(BaseModel):
    """ハイパーパラメータチューニングリクエスト"""
    model_config = ConfigDict(frozen=True)

    n_trials: int = Field(
        50,
        ge=1,
        le=1000,
        description="試行回数"
    )
    timeout: int = Field(
        600,
        ge=10,
        le=86400,
        description="タイムアウト（秒）"
    )
    use_cross_validation: bool = Field(
        True,
        description="交差検証を使用するか"
    )
    n_folds: int = Field(
        3,
        ge=2,
        le=10,
        description="交差検証の分割数"
    )
    random_state: int = Field(
        42,
        ge=0,
        description="乱数シード"
    )


# =============================================================================
# Evaluation関連スキーマ
# =============================================================================

class EvaluationRequest(BaseModel):
    """評価リクエスト"""
    model_config = ConfigDict(frozen=True)

    top_k: int = Field(
        10,
        ge=1,
        le=100,
        description="評価する推薦数"
    )
    train_ratio: float = Field(
        0.8,
        gt=0.0,
        lt=1.0,
        description="訓練データの割合"
    )
    use_temporal_split: bool = Field(
        True,
        description="時系列分割を使用するか"
    )
    include_extended_metrics: bool = Field(
        True,
        description="拡張メトリクスを計算するか"
    )
    include_diversity_metrics: bool = Field(
        True,
        description="多様性指標を計算するか"
    )


# =============================================================================
# Data Quality関連スキーマ
# =============================================================================

class DataQualityReport(BaseModel):
    """データ品質レポート"""
    model_config = ConfigDict(frozen=False)  # レポートは可変

    is_valid: bool = Field(..., description="データが有効か")
    errors: list[str] = Field(default_factory=list, description="エラーリスト")
    warnings: list[str] = Field(default_factory=list, description="警告リスト")
    statistics: dict[str, float] = Field(default_factory=dict, description="統計情報")

    def add_error(self, message: str) -> None:
        """エラーを追加"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """警告を追加"""
        self.warnings.append(message)

    def set_statistic(self, key: str, value: float) -> None:
        """統計情報を設定"""
        self.statistics[key] = value
