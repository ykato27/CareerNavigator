"""
統合設定管理システム - Pydantic-based Settings

このモジュールは以下の3つの設定システムを統合します:
- skillnote_recommendation/config.py (シンプルなdataclass)
- skillnote_recommendation/core/config.py (レガシーdict-based)
- skillnote_recommendation/core/config_v2.py (モダンfrozen dataclass)

主な特徴:
- Pydantic BaseSettingsによる型安全性
- 環境変数サポート (.env ファイル対応)
- 環境別設定 (dev/staging/prod)
- Immutableな設定
- バリデーション機能
- 後方互換性

使用例:
    >>> from skillnote_recommendation.settings import get_settings
    >>> settings = get_settings()  # APP_ENV環境変数から自動取得
    >>> settings = get_settings(env="prod")  # 明示的に指定
    >>> print(settings.paths.data_dir)
"""

import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==================== 環境定義 ====================


class Environment(str, Enum):
    """環境種別"""

    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


# ==================== パス設定 ====================


class PathSettings(BaseSettings):
    """パス設定"""

    model_config = SettingsConfigDict(
        env_prefix="PATH_",
        frozen=True,
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="プロジェクトルートディレクトリ",
    )
    data_dir: Optional[Path] = Field(
        default=None,
        description="データディレクトリ（Noneの場合は project_root/data）",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="出力ディレクトリ（Noneの場合は project_root/output）",
    )
    pages_dir: Optional[Path] = Field(
        default=None,
        description="ページディレクトリ（Noneの場合は project_root/pages）",
    )

    @model_validator(mode="after")
    def set_default_dirs(self):
        """デフォルトディレクトリを設定"""
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", self.project_root / "data")
        if self.output_dir is None:
            object.__setattr__(self, "output_dir", self.project_root / "output")
        if self.pages_dir is None:
            object.__setattr__(self, "pages_dir", self.project_root / "pages")
        return self

    def ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)


class InputDirectorySettings(BaseSettings):
    """入力ディレクトリ名設定"""

    model_config = SettingsConfigDict(
        env_prefix="INPUT_DIR_",
        frozen=True,
    )

    members: str = Field(default="members", description="メンバーディレクトリ名")
    acquired: str = Field(default="acquired", description="習得力量ディレクトリ名")
    skills: str = Field(default="skills", description="力量ディレクトリ名")
    education: str = Field(default="education", description="教育履歴ディレクトリ名")
    license: str = Field(default="license", description="資格ディレクトリ名")
    categories: str = Field(default="categories", description="カテゴリディレクトリ名")


class OutputFileSettings(BaseSettings):
    """出力ファイル名設定"""

    model_config = SettingsConfigDict(
        env_prefix="OUTPUT_FILE_",
        frozen=True,
    )

    members_clean: str = Field(default="members_clean.csv", description="クリーンなメンバーファイル名")
    competence_master: str = Field(default="competence_master.csv", description="力量マスターファイル名")
    member_competence: str = Field(default="member_competence.csv", description="メンバー×力量ファイル名")
    skill_matrix: str = Field(default="skill_matrix.csv", description="力量マトリクスファイル名")
    competence_similarity: str = Field(default="competence_similarity.csv", description="力量類似度ファイル名")


# ==================== モデル設定 ====================


class MatrixFactorizationSettings(BaseSettings):
    """Matrix Factorization モデル設定"""

    model_config = SettingsConfigDict(
        env_prefix="MF_",
        frozen=True,
    )

    # 基本パラメータ
    n_components: int = Field(default=20, ge=1, le=100, description="潜在因子の数")
    max_iter: int = Field(default=1000, ge=100, le=5000, description="最大イテレーション数")
    random_state: int = Field(default=42, description="乱数シード")

    # 収束パラメータ
    tol: float = Field(default=1e-5, gt=0, description="収束判定の閾値")

    # 初期化戦略
    init: Literal["nndsvda", "nndsvd", "random"] = Field(
        default="nndsvda",
        description="初期化方法",
    )

    # 正則化パラメータ
    alpha_w: float = Field(default=0.01, ge=0, le=1, description="メンバー因子行列の正則化")
    alpha_h: float = Field(default=0.01, ge=0, le=1, description="力量因子行列の正則化")
    l1_ratio: float = Field(default=0.5, ge=0, le=1, description="L1正則化の割合")

    # ソルバー
    solver: Literal["cd", "mu"] = Field(
        default="cd",
        description="ソルバー (cd=coordinate descent, mu=multiplicative update)",
    )

    # Confidence Weighting
    use_confidence_weighting: bool = Field(
        default=False,
        description="confidence weightingを使用するか",
    )
    confidence_alpha: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="confidence = 1 + alpha * rating",
    )


class GraphSettings(BaseSettings):
    """グラフモデル設定"""

    model_config = SettingsConfigDict(
        env_prefix="GRAPH_",
        frozen=True,
    )

    # LiNGAM
    lingam_prior_knowledge_threshold: float = Field(
        default=0.0,
        description="LiNGAM事前知識閾値",
    )

    # Graph embedding
    walk_length: int = Field(default=10, ge=5, le=50, description="ランダムウォーク長")
    num_walks: int = Field(default=80, ge=10, le=200, description="ランダムウォーク数")
    embedding_dim: int = Field(default=64, ge=16, le=256, description="埋め込み次元数")

    # Knowledge Graph
    member_similarity_threshold: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="メンバー類似度の閾値",
    )
    member_similarity_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="類似メンバー数",
    )


# ==================== 推薦設定 ====================


class RecommendationSettings(BaseSettings):
    """推薦パラメータ設定"""

    model_config = SettingsConfigDict(
        env_prefix="RECOMMENDATION_",
        frozen=True,
    )

    category_importance_weight: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="カテゴリ重要度の重み",
    )
    acquisition_ease_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="習得難易度の重み",
    )
    popularity_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="人気度の重み",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="類似度の閾値",
    )
    similarity_sample_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="類似度計算のサンプル数",
    )

    @field_validator("category_importance_weight", "acquisition_ease_weight", "popularity_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """重みの合計が妥当であることを検証（警告のみ）"""
        # Note: 個別のバリデーションでは合計チェックは困難なため、
        # model_validatorで実施することを推奨
        return v


class CareerPatternSettings(BaseSettings):
    """キャリアパターン別推薦設定"""

    model_config = SettingsConfigDict(
        env_prefix="CAREER_PATTERN_",
        frozen=True,
    )

    similar_career_threshold: float = Field(default=0.7, ge=0, le=1)
    different_career1_threshold: float = Field(default=0.4, ge=0, le=1)
    similar_career_top_k: int = Field(default=5, ge=1, le=20)
    different_career1_top_k: int = Field(default=5, ge=1, le=20)
    different_career2_top_k: int = Field(default=5, ge=1, le=20)
    similar_career_ref_persons: int = Field(default=5, ge=1, le=20)
    different_career1_ref_persons: int = Field(default=5, ge=1, le=20)
    different_career2_ref_persons: int = Field(default=5, ge=1, le=20)
    min_ref_persons: int = Field(default=1, ge=1)
    ref_person_selection: Literal["top_similar", "random"] = Field(default="top_similar")


# ==================== データ処理設定 ====================


class PreprocessingSettings(BaseSettings):
    """データ前処理設定"""

    model_config = SettingsConfigDict(
        env_prefix="PREPROCESSING_",
        frozen=True,
    )

    min_competences_per_member: int = Field(
        default=3,
        ge=1,
        description="メンバーが保有すべき最小力量数",
    )
    min_members_per_competence: int = Field(
        default=3,
        ge=1,
        description="力量を保有すべき最小メンバー数",
    )
    normalization_method: Optional[Literal["minmax", "standard", "l2"]] = Field(
        default="minmax",
        description="正規化方法",
    )
    enable_preprocessing: bool = Field(default=True, description="前処理を有効にするか")


class ValidationSettings(BaseSettings):
    """データ検証設定"""

    model_config = SettingsConfigDict(
        env_prefix="VALIDATION_",
        frozen=True,
    )

    min_competences_per_member: int = Field(default=1, ge=0)
    max_name_length: int = Field(default=100, ge=1)
    invalid_name_patterns: list[str] = Field(
        default_factory=lambda: ["削除", "テスト", "test"],
    )


# ==================== 最適化設定 ====================


class OptunaSettings(BaseSettings):
    """Optuna ハイパーパラメータチューニング設定"""

    model_config = SettingsConfigDict(
        env_prefix="OPTUNA_",
        frozen=True,
    )

    n_trials: int = Field(default=50, ge=1, le=500, description="試行回数")
    timeout: int = Field(default=600, ge=60, description="タイムアウト（秒）")
    n_jobs: int = Field(default=1, ge=-1, description="並列実行数")
    show_progress_bar: bool = Field(default=True)

    # 交差検証
    use_cross_validation: bool = Field(default=True)
    n_folds: int = Field(default=3, ge=2, le=10)
    use_time_series_split: bool = Field(default=True)
    test_size: float = Field(default=0.15, gt=0, lt=1)

    # Early stopping
    enable_early_stopping: bool = Field(default=True)
    early_stopping_patience: int = Field(default=5, ge=1)
    early_stopping_batch_size: int = Field(default=50, ge=10)

    # 探索空間
    n_components_min: int = Field(default=10, ge=1)
    n_components_max: int = Field(default=30, ge=1)
    alpha_min: float = Field(default=0.001, gt=0)
    alpha_max: float = Field(default=0.5, gt=0)
    l1_ratio_min: float = Field(default=0.0, ge=0, le=1)
    l1_ratio_max: float = Field(default=1.0, ge=0, le=1)
    max_iter_min: int = Field(default=500, ge=100)
    max_iter_max: int = Field(default=1500, ge=100)


# ==================== 評価設定 ====================


class EvaluationSettings(BaseSettings):
    """評価パラメータ設定"""

    model_config = SettingsConfigDict(
        env_prefix="EVALUATION_",
        frozen=True,
    )

    # 基本パラメータ
    top_k: int = Field(default=10, ge=1, le=100, description="推薦数")
    include_extended_metrics: bool = Field(default=True, description="拡張メトリクスを含めるか")
    include_diversity_metrics: bool = Field(default=True, description="多様性指標を含めるか")

    # Hold-out評価
    train_ratio: float = Field(default=0.8, gt=0, lt=1, description="訓練データの割合")
    use_temporal_split: bool = Field(default=True, description="時系列分割を使用するか")

    # 交差検証
    n_folds: int = Field(default=5, ge=2, le=10, description="交差検証の分割数")
    cv_use_temporal: bool = Field(default=True, description="時系列交差検証を使用するか")

    # Leave-One-Out評価
    loo_max_users: Optional[int] = Field(default=None, description="評価する最大ユーザー数")

    # その他
    min_test_items: int = Field(default=1, ge=1, description="テストデータに必要な最小力量数")
    calculate_gini: bool = Field(default=True)
    calculate_novelty: bool = Field(default=True)
    detailed_report: bool = Field(default=True)
    export_results: bool = Field(default=True)


# ==================== 可視化設定 ====================


class VisualizationSettings(BaseSettings):
    """可視化パラメータ設定"""

    model_config = SettingsConfigDict(
        env_prefix="VISUALIZATION_",
        frozen=True,
    )

    heatmap_height: int = Field(default=500, ge=100)
    scatter_plot_height: int = Field(default=500, ge=100)
    max_members_to_show: int = Field(default=10, ge=1)
    max_competences_to_show: int = Field(default=10, ge=1)
    color_target_member: str = Field(default="#FF4B4B")
    color_reference_person: str = Field(default="#4B8BFF")
    color_other_member: str = Field(default="#CCCCCC")


# ==================== ログ・エンコーディング設定 ====================


class LoggingSettings(BaseSettings):
    """ログ設定"""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        frozen=True,
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    enable_json: bool = Field(default=False, description="JSON形式でログ出力")
    enable_console: bool = Field(default=True, description="コンソール出力")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")


class EncodingSettings(BaseSettings):
    """エンコーディング設定"""

    model_config = SettingsConfigDict(
        env_prefix="ENCODING_",
        frozen=True,
    )

    file_encoding: str = Field(default="utf-8")
    output_encoding: str = Field(default="utf-8-sig")


# ==================== アプリケーション設定 ====================


class ApplicationSettings(BaseSettings):
    """アプリケーション設定"""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        frozen=True,
    )

    title: str = Field(default="CareerNavigator")
    version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)


# ==================== メイン設定 ====================


class Settings(BaseSettings):
    """
    統合設定クラス

    全ての設定を統合し、環境変数・.envファイルからの読み込みをサポート

    環境変数の優先順位:
    1. システム環境変数
    2. .env ファイル
    3. デフォルト値

    使用例:
        >>> settings = Settings()
        >>> settings = Settings(_env_file=".env.prod")
        >>> print(settings.mf.n_components)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # 例: MF__N_COMPONENTS=20
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # 環境
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="実行環境",
    )

    # パス
    paths: PathSettings = Field(default_factory=PathSettings)

    # 入出力
    input_dirs: InputDirectorySettings = Field(default_factory=InputDirectorySettings)
    output_files: OutputFileSettings = Field(default_factory=OutputFileSettings)

    # モデル
    mf: MatrixFactorizationSettings = Field(default_factory=MatrixFactorizationSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)

    # 推薦
    recommendation: RecommendationSettings = Field(default_factory=RecommendationSettings)
    career_pattern: CareerPatternSettings = Field(default_factory=CareerPatternSettings)

    # データ処理
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)

    # 最適化・評価
    optuna: OptunaSettings = Field(default_factory=OptunaSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    # 可視化
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)

    # その他
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    encoding: EncodingSettings = Field(default_factory=EncodingSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)

    @model_validator(mode="after")
    def configure_environment_specific_settings(self):
        """環境別の設定を適用"""
        if self.environment == Environment.PRODUCTION:
            # 本番環境ではJSON形式ログ、DEBUGは無効
            object.__setattr__(
                self,
                "logging",
                LoggingSettings(level="INFO", enable_json=True, enable_console=True),
            )
            object.__setattr__(
                self,
                "app",
                ApplicationSettings(title=self.app.title, version=self.app.version, debug=False),
            )
        elif self.environment == Environment.STAGING:
            # ステージング環境ではJSON形式ログ
            object.__setattr__(
                self,
                "logging",
                LoggingSettings(level="INFO", enable_json=True, enable_console=True),
            )
        else:  # DEVELOPMENT
            # 開発環境では人間可読形式、DEBUGレベル
            object.__setattr__(
                self,
                "logging",
                LoggingSettings(level="DEBUG", enable_json=False, enable_console=True),
            )

        return self

    def get_input_dir(self, dir_key: str) -> Path:
        """
        入力ディレクトリのパスを取得

        Args:
            dir_key: ディレクトリキー (members, acquired, skills, etc.)

        Returns:
            入力ディレクトリのパス
        """
        dir_name = getattr(self.input_dirs, dir_key)
        return self.paths.data_dir / dir_name

    def get_output_path(self, file_key: str) -> Path:
        """
        出力ファイルのパスを取得

        Args:
            file_key: ファイルキー (members_clean, competence_master, etc.)

        Returns:
            出力ファイルのパス
        """
        filename = getattr(self.output_files, file_key)
        return self.paths.output_dir / filename

    def ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        self.paths.ensure_directories()


# ==================== ファクトリー関数 ====================


_settings_cache: dict[str, Settings] = {}


def get_settings(
    env: Optional[str] = None,
    env_file: Optional[str] = None,
    force_reload: bool = False,
) -> Settings:
    """
    設定を取得（シングルトンパターン）

    Args:
        env: 環境名 (dev/staging/prod)。Noneの場合は環境変数 APP_ENV から取得
        env_file: .envファイルのパス。Noneの場合はデフォルトの .env を使用
        force_reload: キャッシュを無視して強制的に再読み込み

    Returns:
        Settings インスタンス

    使用例:
        >>> settings = get_settings()  # APP_ENV環境変数から取得
        >>> settings = get_settings(env="prod")  # 本番環境設定
        >>> settings = get_settings(env_file=".env.test")  # テスト環境ファイル
    """
    if env is None:
        env = os.getenv("APP_ENV", "dev")

    # キャッシュキー
    cache_key = f"{env}:{env_file or '.env'}"

    # キャッシュチェック
    if not force_reload and cache_key in _settings_cache:
        return _settings_cache[cache_key]

    # 環境の検証
    try:
        environment = Environment(env)
    except ValueError:
        raise ValueError(
            f"Invalid environment: {env}. Must be one of: {[e.value for e in Environment]}"
        )

    # 設定の作成
    if env_file:
        settings = Settings(environment=environment, _env_file=env_file)
    else:
        settings = Settings(environment=environment)

    # キャッシュに保存
    _settings_cache[cache_key] = settings

    return settings


def clear_settings_cache() -> None:
    """設定キャッシュをクリア（主にテスト用）"""
    global _settings_cache
    _settings_cache = {}
