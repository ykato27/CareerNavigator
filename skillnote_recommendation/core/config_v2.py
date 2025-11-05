"""
設定管理（V2）- GAFAレベルの設計

主な改善点:
- 不変（Immutable）設定
- 環境分離（dev/staging/prod）
- 型安全性
- テスト容易性
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Final


class Environment(Enum):
    """環境種別"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


@dataclass(frozen=True)
class DirectoryConfig:
    """ディレクトリ設定"""
    project_root: Path
    data_dir: Path
    output_dir: Path

    @classmethod
    def from_project_root(cls, root: Path) -> "DirectoryConfig":
        """プロジェクトルートから設定を作成"""
        return cls(
            project_root=root,
            data_dir=root / "data",
            output_dir=root / "output"
        )


@dataclass(frozen=True)
class InputDirectories:
    """入力ディレクトリ名"""
    members: str = "members"
    acquired: str = "acquired"
    skills: str = "skills"
    education: str = "education"
    license: str = "license"
    categories: str = "categories"


@dataclass(frozen=True)
class OutputFiles:
    """出力ファイル名"""
    members_clean: str = "members_clean.csv"
    competence_master: str = "competence_master.csv"
    member_competence: str = "member_competence.csv"
    skill_matrix: str = "skill_matrix.csv"
    competence_similarity: str = "competence_similarity.csv"


@dataclass(frozen=True)
class RecommendationParams:
    """推薦パラメータ"""
    category_importance_weight: float = 0.4
    acquisition_ease_weight: float = 0.3
    popularity_weight: float = 0.3
    similarity_threshold: float = 0.3
    similarity_sample_size: int = 100


@dataclass(frozen=True)
class MFParams:
    """Matrix Factorizationパラメータ"""
    n_components: int = 20
    max_iter: int = 1000
    random_state: int = 42
    tol: float = 1e-5
    init: Literal["nndsvda", "nndsvd", "random"] = "nndsvda"
    alpha_W: float = 0.01
    alpha_H: float = 0.01
    l1_ratio: float = 0.5
    solver: Literal["cd", "mu"] = "cd"
    use_confidence_weighting: bool = False
    confidence_alpha: float = 1.0


@dataclass(frozen=True)
class DataPreprocessingParams:
    """データ前処理パラメータ"""
    min_competences_per_member: int = 3
    min_members_per_competence: int = 3
    normalization_method: Literal["minmax", "standard", "l2"] | None = "minmax"
    enable_preprocessing: bool = True


@dataclass(frozen=True)
class OptunaParams:
    """Optunaハイパーパラメータチューニングパラメータ"""
    n_trials: int = 50
    timeout: int = 600
    n_jobs: int = 1
    show_progress_bar: bool = True
    use_cross_validation: bool = True
    n_folds: int = 3

    # 探索空間
    n_components_range: tuple[int, int] = (10, 30)
    alpha_range: tuple[float, float] = (0.001, 0.5)
    l1_ratio_range: tuple[float, float] = (0.0, 1.0)
    max_iter_range: tuple[int, int] = (500, 1500)


@dataclass(frozen=True)
class EvaluationParams:
    """評価パラメータ"""
    top_k: int = 10
    include_extended_metrics: bool = True
    include_diversity_metrics: bool = True
    train_ratio: float = 0.8
    use_temporal_split: bool = True
    n_folds: int = 5
    cv_use_temporal: bool = True
    loo_max_users: int | None = None
    min_test_items: int = 1
    calculate_gini: bool = True
    calculate_novelty: bool = True
    detailed_report: bool = True
    export_results: bool = True


@dataclass(frozen=True)
class LoggingParams:
    """ログ設定"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_json: bool = False  # dev環境ではfalse、prod環境ではtrue
    enable_console: bool = True
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'


@dataclass(frozen=True)
class EncodingConfig:
    """エンコーディング設定"""
    file_encoding: str = 'utf-8'
    output_encoding: str = 'utf-8-sig'


@dataclass(frozen=True)
class Config:
    """
    システム設定（不変）

    全ての設定は不変（frozen=True）であり、
    環境ごとに異なるインスタンスを作成する
    """
    environment: Environment
    directories: DirectoryConfig
    input_dirs: InputDirectories = field(default_factory=InputDirectories)
    output_files: OutputFiles = field(default_factory=OutputFiles)
    recommendation: RecommendationParams = field(default_factory=RecommendationParams)
    mf: MFParams = field(default_factory=MFParams)
    preprocessing: DataPreprocessingParams = field(default_factory=DataPreprocessingParams)
    optuna: OptunaParams = field(default_factory=OptunaParams)
    evaluation: EvaluationParams = field(default_factory=EvaluationParams)
    logging: LoggingParams = field(default_factory=LoggingParams)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)

    @classmethod
    def from_env(cls, env: str = "dev", project_root: Path | None = None) -> "Config":
        """
        環境名から設定を作成

        Args:
            env: 環境名（dev, staging, prod）
            project_root: プロジェクトルート（Noneの場合は自動検出）

        Returns:
            Config インスタンス
        """
        environment = Environment(env)

        if project_root is None:
            # 自動検出: このファイルから3階層上がプロジェクトルート
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent

        directories = DirectoryConfig.from_project_root(project_root)

        # 環境ごとの設定のカスタマイズ
        if environment == Environment.PRODUCTION:
            logging_params = LoggingParams(
                level="INFO",
                enable_json=True,  # 本番環境はJSON形式
                enable_console=True
            )
        elif environment == Environment.STAGING:
            logging_params = LoggingParams(
                level="INFO",
                enable_json=True,
                enable_console=True
            )
        else:  # DEVELOPMENT
            logging_params = LoggingParams(
                level="DEBUG",
                enable_json=False,  # 開発環境は人間可読形式
                enable_console=True
            )

        return cls(
            environment=environment,
            directories=directories,
            logging=logging_params
        )

    @classmethod
    def default(cls) -> "Config":
        """デフォルト設定（開発環境）を取得"""
        return cls.from_env("dev")

    def get_input_dir(self, dir_key: str) -> Path:
        """入力ディレクトリのパスを取得"""
        dir_name = getattr(self.input_dirs, dir_key)
        return self.directories.data_dir / dir_name

    def get_output_path(self, file_key: str) -> Path:
        """出力ファイルのパスを取得"""
        filename = getattr(self.output_files, file_key)
        return self.directories.output_dir / filename

    def ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        self.directories.data_dir.mkdir(parents=True, exist_ok=True)
        self.directories.output_dir.mkdir(parents=True, exist_ok=True)


# グローバルデフォルト設定（後方互換性のため）
# 新しいコードでは Config.from_env() または Config.default() を使用すること
DEFAULT_CONFIG: Final[Config] = Config.default()


def get_config(env: str | None = None) -> Config:
    """
    設定を取得

    Args:
        env: 環境名（Noneの場合は環境変数 APP_ENV から取得、デフォルトは "dev"）

    Returns:
        Config インスタンス
    """
    if env is None:
        env = os.getenv("APP_ENV", "dev")

    return Config.from_env(env)
