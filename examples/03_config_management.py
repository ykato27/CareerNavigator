"""
Config管理の使用例

v2.0.0で追加された不変設計のConfig管理のデモンストレーション
"""

from skillnote_recommendation.core.config_v2 import (
    Config,
    MFParams,
    OptunaParams,
    EvaluationParams,
    Environment
)
from pathlib import Path


def example_basic_config():
    """基本的なConfig使用"""
    # 開発環境の設定
    config = Config.from_env("dev")

    print("=== 開発環境設定 ===")
    print(f"Environment: {config.environment.value}")
    print(f"Data Directory: {config.directories.data_dir}")
    print(f"Output Directory: {config.directories.output_dir}")
    print(f"Log Level: {config.logging.level}")
    print(f"JSON Logging: {config.logging.enable_json}")

    # Matrix Factorizationパラメータ
    print(f"\n=== MF Parameters ===")
    print(f"N Components: {config.mf.n_components}")
    print(f"Max Iter: {config.mf.max_iter}")
    print(f"Random State: {config.mf.random_state}")


def example_environment_specific_config():
    """環境別の設定"""
    # 本番環境
    config_prod = Config.from_env("prod")
    print("\n=== 本番環境設定 ===")
    print(f"Log Level: {config_prod.logging.level}")
    print(f"JSON Logging: {config_prod.logging.enable_json}")

    # ステージング環境
    config_staging = Config.from_env("staging")
    print("\n=== ステージング環境設定 ===")
    print(f"Log Level: {config_staging.logging.level}")
    print(f"JSON Logging: {config_staging.logging.enable_json}")


def example_custom_config():
    """カスタム設定の作成"""
    from skillnote_recommendation.core.config_v2 import (
        DirectoryConfig,
        LoggingParams
    )

    # カスタムMFパラメータ
    custom_mf = MFParams(
        n_components=30,
        max_iter=2000,
        random_state=123,
        alpha_W=0.02,
        alpha_H=0.02
    )

    # カスタムロギング
    custom_logging = LoggingParams(
        level="DEBUG",
        enable_json=False,
        enable_console=True
    )

    # カスタム設定
    custom_config = Config(
        environment=Environment.DEVELOPMENT,
        directories=DirectoryConfig.from_project_root(Path(".")),
        mf=custom_mf,
        logging=custom_logging
    )

    print("\n=== カスタム設定 ===")
    print(f"MF Components: {custom_config.mf.n_components}")
    print(f"MF Max Iter: {custom_config.mf.max_iter}")
    print(f"Log Level: {custom_config.logging.level}")


def example_path_helpers():
    """パスヘルパーの使用"""
    config = Config.default()

    # 入力ディレクトリ
    members_dir = config.get_input_dir("members")
    print(f"\n=== Path Helpers ===")
    print(f"Members Dir: {members_dir}")

    # 出力ファイル
    skill_matrix_path = config.get_output_path("skill_matrix")
    print(f"Skill Matrix: {skill_matrix_path}")

    # ディレクトリ作成
    config.ensure_directories()
    print("✅ Directories ensured")


def example_immutability():
    """不変性のデモンストレーション"""
    config = Config.default()

    print("\n=== 不変性（Immutability） ===")
    print(f"Original N Components: {config.mf.n_components}")

    try:
        # これはエラーになる（frozen=True）
        config.mf.n_components = 50  # type: ignore
    except Exception as e:
        print(f"❌ Cannot modify (frozen): {type(e).__name__}")

    # 新しい設定を作成する必要がある
    new_mf = MFParams(
        n_components=50,
        max_iter=config.mf.max_iter,
        random_state=config.mf.random_state
    )
    print(f"✅ Create new config instead: {new_mf.n_components}")


if __name__ == "__main__":
    example_basic_config()
    example_environment_specific_config()
    example_custom_config()
    example_path_helpers()
    example_immutability()
