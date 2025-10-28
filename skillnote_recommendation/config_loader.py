"""
設定ファイル読み込みモジュール

config.yamlから設定を読み込み、シングルトンとして提供します。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """
    設定ファイル読み込みクラス（シングルトン）

    config.yamlを読み込み、設定値へのアクセスを提供します。
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """設定ファイルを読み込む"""
        # config.yamlのパスを取得
        config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        設定値を取得する

        Args:
            key_path: ドット区切りのキーパス（例: "nmf.n_components"）
            default: デフォルト値（キーが存在しない場合）

        Returns:
            設定値

        Examples:
            >>> config = ConfigLoader()
            >>> config.get("nmf.n_components")
            20
            >>> config.get("rwr.restart_prob")
            0.15
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        設定セクション全体を取得する

        Args:
            section: セクション名（例: "nmf", "rwr"）

        Returns:
            設定セクションの辞書

        Examples:
            >>> config = ConfigLoader()
            >>> config.get_section("nmf")
            {'n_components': 20, 'random_state': 42}
        """
        return self._config.get(section, {})

    def reload(self):
        """設定ファイルを再読み込みする"""
        self._config = None
        self._load_config()

    @property
    def config(self) -> Dict[str, Any]:
        """設定辞書全体を取得"""
        return self._config


# シングルトンインスタンスを作成
config = ConfigLoader()


# 便利な関数
def get_config(key_path: str, default: Any = None) -> Any:
    """
    設定値を取得する便利関数

    Args:
        key_path: ドット区切りのキーパス
        default: デフォルト値

    Returns:
        設定値
    """
    return config.get(key_path, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """
    設定セクションを取得する便利関数

    Args:
        section: セクション名

    Returns:
        設定セクションの辞書
    """
    return config.get_section(section)
