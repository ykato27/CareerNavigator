"""
モデルシリアライゼーション

GAFAレベルのモデル保存:
- pickle → joblib + JSON（安全性向上）
- バージョン管理
- メタデータ保存
- 監査可能性
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any
import joblib
import numpy as np
from numpy.typing import NDArray

from skillnote_recommendation.core.logging_config import LoggerMixin
from skillnote_recommendation.core.errors import ModelLoadError, ModelSaveError, ErrorCode


MODEL_VERSION = "2.0.0"  # モデル保存形式のバージョン


class ModelSerializer(LoggerMixin):
    """
    モデルシリアライザ

    モデルを安全かつ監査可能な形式で保存・読み込み
    """

    @staticmethod
    def save_matrix_factorization_model(
        filepath: str,
        W: NDArray[np.float64],
        H: NDArray[np.float64],
        member_codes: list[str],
        competence_codes: list[str],
        member_index: dict[str, int],
        competence_index: dict[str, int],
        params: dict[str, Any],
        reconstruction_err: float,
        n_iter: int,
    ) -> None:
        """
        Matrix Factorizationモデルを保存

        Args:
            filepath: 保存先ファイルパス（拡張子なし）
            W: メンバー因子行列
            H: 力量因子行列
            member_codes: メンバーコードリスト
            competence_codes: 力量コードリスト
            member_index: メンバーコード → インデックス
            competence_index: 力量コード → インデックス
            params: モデルパラメータ
            reconstruction_err: 再構成誤差
            n_iter: イテレーション数

        Raises:
            ModelSaveError: 保存に失敗した場合
        """
        try:
            base_path = Path(filepath)
            base_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. メタデータをJSON形式で保存（人間可読）
            metadata = {
                "model_type": "MatrixFactorization",
                "version": MODEL_VERSION,
                "created_at": datetime.utcnow().isoformat(),
                "params": params,
                "metrics": {
                    "reconstruction_error": float(reconstruction_err),
                    "n_iterations": int(n_iter),
                },
                "dimensions": {
                    "n_members": len(member_codes),
                    "n_competences": len(competence_codes),
                    "n_components": W.shape[1],
                },
                "member_codes": member_codes,
                "competence_codes": competence_codes,
            }

            metadata_path = base_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 2. 行列データをjoblibで保存（圧縮 + 安全）
            model_data = {
                "W": W,
                "H": H,
                "member_index": member_index,
                "competence_index": competence_index,
            }

            data_path = base_path.with_suffix(".joblib")
            joblib.dump(model_data, data_path, compress=3)

            logger = LoggerMixin().logger
            logger.info(
                "model_saved",
                filepath=str(filepath),
                metadata_path=str(metadata_path),
                data_path=str(data_path),
                model_version=MODEL_VERSION,
            )

        except Exception as e:
            raise ModelSaveError(filepath=filepath, reason=str(e)) from e

    @staticmethod
    def load_matrix_factorization_model(filepath: str) -> dict[str, Any]:
        """
        Matrix Factorizationモデルを読み込み

        Args:
            filepath: モデルファイルパス（拡張子なし）

        Returns:
            モデルデータの辞書

        Raises:
            ModelLoadError: 読み込みに失敗した場合
        """
        try:
            base_path = Path(filepath)

            # 1. メタデータを読み込み
            metadata_path = base_path.with_suffix(".json")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # バージョンチェック
            model_version = metadata.get("version", "1.0.0")
            if model_version.split(".")[0] != MODEL_VERSION.split(".")[0]:
                # メジャーバージョンが異なる場合は警告
                logger = LoggerMixin().logger
                logger.warning(
                    "model_version_mismatch",
                    saved_version=model_version,
                    current_version=MODEL_VERSION,
                )

            # 2. 行列データを読み込み
            data_path = base_path.with_suffix(".joblib")
            if not data_path.exists():
                # 後方互換性: .joblibがない場合は古い形式（.pkl）を試す
                data_path = base_path.with_suffix(".pkl")
                if data_path.exists():
                    import pickle

                    with open(data_path, "rb") as f:
                        model_data = pickle.load(f)
                else:
                    raise FileNotFoundError(f"Model data file not found: {data_path}")
            else:
                model_data = joblib.load(data_path)

            # 3. メタデータとモデルデータを統合
            result = {**metadata, **model_data}

            logger = LoggerMixin().logger
            logger.info(
                "model_loaded",
                filepath=str(filepath),
                model_version=model_version,
                n_members=metadata["dimensions"]["n_members"],
                n_competences=metadata["dimensions"]["n_competences"],
            )

            return result

        except Exception as e:
            raise ModelLoadError(filepath=filepath, reason=str(e)) from e

    @staticmethod
    def validate_model_compatibility(
        saved_params: dict[str, Any], current_params: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        保存されたモデルと現在のパラメータの互換性を検証

        Args:
            saved_params: 保存時のパラメータ
            current_params: 現在のパラメータ

        Returns:
            (互換性あり, 警告メッセージリスト)
        """
        warnings = []

        # 重要なパラメータの不一致を検出
        critical_params = ["n_components", "init", "solver"]

        for param in critical_params:
            if saved_params.get(param) != current_params.get(param):
                warnings.append(
                    f"Parameter mismatch: {param} "
                    f"(saved={saved_params.get(param)}, current={current_params.get(param)})"
                )

        is_compatible = len(warnings) == 0
        return is_compatible, warnings
