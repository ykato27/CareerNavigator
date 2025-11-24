"""
データ前処理モジュール

NMFモデルの学習前にスキルマトリックスを前処理し、
データ品質を向上させる
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import logging

logger = logging.getLogger(__name__)


class SkillMatrixPreprocessor:
    """スキルマトリックス前処理クラス"""

    def __init__(
        self,
        min_competences_per_member: int = 3,
        min_members_per_competence: int = 3,
        normalization_method: Optional[str] = "minmax",
    ):
        """
        初期化

        Args:
            min_competences_per_member: メンバーが保有すべき最小力量数
            min_members_per_competence: 力量を保有すべき最小メンバー数
            normalization_method: 正規化方法 ('minmax', 'standard', 'l2', None)
        """
        self.min_competences_per_member = min_competences_per_member
        self.min_members_per_competence = min_members_per_competence
        self.normalization_method = normalization_method

        # 統計情報を保存
        self.preprocessing_stats = {}

        # データリーケージ防止: scalerを保存
        self.scaler_ = None
        self.is_fitted_ = False

    def preprocess(
        self, skill_matrix: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        スキルマトリックスを前処理

        Args:
            skill_matrix: メンバー×力量マトリックス (index=メンバーコード, columns=力量コード)
            verbose: 詳細情報を出力するか

        Returns:
            (前処理済みマトリックス, 統計情報)
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("スキルマトリックス前処理開始")
            logger.info("=" * 60)

        original_shape = skill_matrix.shape
        stats = {
            "original_shape": original_shape,
            "original_members": original_shape[0],
            "original_competences": original_shape[1],
            "original_sparsity": self._calculate_sparsity(skill_matrix),
        }

        # ステップ1: 外れ値の除去
        filtered_matrix = self._remove_outliers(skill_matrix, verbose)
        stats["filtered_shape"] = filtered_matrix.shape
        stats["removed_members"] = original_shape[0] - filtered_matrix.shape[0]
        stats["removed_competences"] = original_shape[1] - filtered_matrix.shape[1]

        # ステップ2: 正規化
        normalized_matrix = self._normalize(filtered_matrix, verbose)
        stats["final_shape"] = normalized_matrix.shape
        stats["final_sparsity"] = self._calculate_sparsity(normalized_matrix)
        stats["normalization_method"] = self.normalization_method

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("前処理完了")
            logger.info("=" * 60)
            logger.info(f"元のサイズ: {stats['original_shape']}")
            logger.info(f"最終サイズ: {stats['final_shape']}")
            logger.info(f"除外メンバー数: {stats['removed_members']}")
            logger.info(f"除外力量数: {stats['removed_competences']}")
            logger.info(f"元のスパース性: {stats['original_sparsity']:.2f}%")
            logger.info(f"最終スパース性: {stats['final_sparsity']:.2f}%")
            logger.info(f"正規化方法: {stats['normalization_method']}")
            logger.info("=" * 60 + "\n")

        self.preprocessing_stats = stats
        return normalized_matrix, stats

    def _remove_outliers(self, skill_matrix: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        外れ値（極端に力量が少ないメンバー・保有者が少ない力量）を除去

        Args:
            skill_matrix: スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            フィルタリング済みマトリックス
        """
        if verbose:
            logger.info("\n--- 外れ値除去 ---")

        # メンバーごとの力量数をカウント（0より大きい値の数）
        member_competence_counts = (skill_matrix > 0).sum(axis=1)
        valid_members = member_competence_counts >= self.min_competences_per_member

        if verbose:
            logger.info(
                f"力量数が{self.min_competences_per_member}未満のメンバー: "
                f"{(~valid_members).sum()}名を除外"
            )

        # 力量ごとの保有者数をカウント
        competence_member_counts = (skill_matrix > 0).sum(axis=0)
        valid_competences = competence_member_counts >= self.min_members_per_competence

        if verbose:
            logger.info(
                f"保有者が{self.min_members_per_competence}名未満の力量: "
                f"{(~valid_competences).sum()}個を除外"
            )

        # フィルタリング
        filtered_matrix = skill_matrix.loc[valid_members, valid_competences]

        return filtered_matrix

    def _normalize(self, skill_matrix: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        スキルマトリックスを正規化

        Args:
            skill_matrix: スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            正規化済みマトリックス
        """
        if self.normalization_method is None:
            if verbose:
                logger.info("\n--- 正規化スキップ ---")
            return skill_matrix

        if verbose:
            logger.info(f"\n--- 正規化 ({self.normalization_method}) ---")

        if self.normalization_method == "minmax":
            # Min-Max正規化（0-1範囲）
            scaler = MinMaxScaler()
            normalized_values = scaler.fit_transform(skill_matrix.values)
            normalized_matrix = pd.DataFrame(
                normalized_values, index=skill_matrix.index, columns=skill_matrix.columns
            )
            if verbose:
                logger.info("Min-Max正規化を適用（範囲: 0-1）")

        elif self.normalization_method == "standard":
            # 標準化（平均0、分散1）
            scaler = StandardScaler()
            normalized_values = scaler.fit_transform(skill_matrix.values)
            # NMFは非負値が必要なので、負の値を0にクリップ
            normalized_values = np.clip(normalized_values, 0, None)
            normalized_matrix = pd.DataFrame(
                normalized_values, index=skill_matrix.index, columns=skill_matrix.columns
            )
            if verbose:
                logger.info("標準化を適用（負の値は0にクリップ）")

        elif self.normalization_method == "l2":
            # L2ノルム正規化（各行の二乗和=1）
            normalized_values = normalize(skill_matrix.values, norm="l2", axis=1)
            normalized_matrix = pd.DataFrame(
                normalized_values, index=skill_matrix.index, columns=skill_matrix.columns
            )
            if verbose:
                logger.info("L2ノルム正規化を適用（行ごと）")

        else:
            logger.warning(
                f"不明な正規化方法: {self.normalization_method}. " "正規化をスキップします。"
            )
            normalized_matrix = skill_matrix

        return normalized_matrix

    def _calculate_sparsity(self, matrix: pd.DataFrame) -> float:
        """
        スパース性（ゼロ要素の割合）を計算

        Args:
            matrix: マトリックス

        Returns:
            スパース性（0-100%）
        """
        total_elements = matrix.size
        zero_elements = (matrix == 0).sum().sum()
        sparsity = (zero_elements / total_elements) * 100
        return sparsity

    def get_statistics(self) -> dict:
        """
        前処理の統計情報を取得

        Returns:
            統計情報の辞書
        """
        return self.preprocessing_stats

    def fit_transform_train(
        self, skill_matrix: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        訓練データで前処理（データリーケージ防止）

        このメソッドは訓練データのみで統計量を計算し、正規化を適用します。
        テストデータには transform_test() を使用してください。

        Args:
            skill_matrix: 訓練用スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            (前処理済み訓練マトリックス, 統計情報)

        Raises:
            RuntimeError: 既にfitされている場合
        """
        if self.is_fitted_:
            raise RuntimeError(
                "Preprocessor is already fitted. "
                "Create a new instance or use transform_test() for test data."
            )

        if verbose:
            logger.info("=" * 60)
            logger.info("訓練データで前処理（データリーケージ防止モード）")
            logger.info("=" * 60)

        original_shape = skill_matrix.shape
        stats = {
            "original_shape": original_shape,
            "original_members": original_shape[0],
            "original_competences": original_shape[1],
            "original_sparsity": self._calculate_sparsity(skill_matrix),
        }

        # ステップ1: 外れ値の除去（訓練データのみ）
        filtered_matrix = self._remove_outliers(skill_matrix, verbose)
        stats["filtered_shape"] = filtered_matrix.shape
        stats["removed_members"] = original_shape[0] - filtered_matrix.shape[0]
        stats["removed_competences"] = original_shape[1] - filtered_matrix.shape[1]

        # ステップ2: 正規化（訓練データでfitして保存）
        normalized_matrix = self._fit_normalize(filtered_matrix, verbose)
        stats["final_shape"] = normalized_matrix.shape
        stats["final_sparsity"] = self._calculate_sparsity(normalized_matrix)
        stats["normalization_method"] = self.normalization_method

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("訓練データ前処理完了")
            logger.info("=" * 60)
            logger.info(f"元のサイズ: {stats['original_shape']}")
            logger.info(f"最終サイズ: {stats['final_shape']}")
            logger.info(f"除外メンバー数: {stats['removed_members']}")
            logger.info(f"除外力量数: {stats['removed_competences']}")
            logger.info(f"元のスパース性: {stats['original_sparsity']:.2f}%")
            logger.info(f"最終スパース性: {stats['final_sparsity']:.2f}%")
            logger.info(f"正規化方法: {stats['normalization_method']}")
            logger.info("✅ Scaler saved for test data transformation")
            logger.info("=" * 60 + "\n")

        self.preprocessing_stats = stats
        self.is_fitted_ = True
        return normalized_matrix, stats

    def transform_test(
        self, skill_matrix: pd.DataFrame, verbose: bool = True
    ) -> pd.DataFrame:
        """
        テストデータの前処理（訓練データの統計を使用）

        訓練データで fit_transform_train() を実行した後にのみ使用できます。
        訓練データで計算した統計量を使ってテストデータを変換します。

        Args:
            skill_matrix: テスト用スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            前処理済みテストマトリックス

        Raises:
            RuntimeError: fit_transform_train() が実行されていない場合
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Preprocessor is not fitted yet. "
                "Call fit_transform_train() on training data first."
            )

        if verbose:
            logger.info("=" * 60)
            logger.info("テストデータ変換（訓練統計を使用）")
            logger.info("=" * 60)

        # テストデータは外れ値除去なし（訓練データと同じ列構造を維持）
        # 正規化のみ適用（訓練データの統計を使用）
        normalized_matrix = self._apply_normalize(skill_matrix, verbose)

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("テストデータ変換完了")
            logger.info("=" * 60)
            logger.info(f"サイズ: {normalized_matrix.shape}")
            logger.info(f"スパース性: {self._calculate_sparsity(normalized_matrix):.2f}%")
            logger.info("✅ Test data transformed using training statistics")
            logger.info("=" * 60 + "\n")

        return normalized_matrix

    def _fit_normalize(self, skill_matrix: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        正規化をfitして適用（訓練データ用）

        Args:
            skill_matrix: スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            正規化済みマトリックス
        """
        if self.normalization_method is None:
            if verbose:
                logger.info("\n--- 正規化スキップ ---")
            return skill_matrix

        if verbose:
            logger.info(f"\n--- 正規化（fit & transform: {self.normalization_method}） ---")

        if self.normalization_method == "minmax":
            self.scaler_ = MinMaxScaler()
            normalized_values = self.scaler_.fit_transform(skill_matrix.values)
            if verbose:
                logger.info("✅ MinMaxScaler fitted on training data")

        elif self.normalization_method == "standard":
            self.scaler_ = StandardScaler()
            normalized_values = self.scaler_.fit_transform(skill_matrix.values)
            normalized_values = np.clip(normalized_values, 0, None)  # NMF用
            if verbose:
                logger.info("✅ StandardScaler fitted on training data")

        elif self.normalization_method == "l2":
            # L2正規化はstatelessなのでscaler不要
            normalized_values = normalize(skill_matrix.values, norm="l2", axis=1)
            if verbose:
                logger.info("✅ L2 normalization applied (stateless)")

        else:
            logger.warning(
                f"不明な正規化方法: {self.normalization_method}. 正規化をスキップします。"
            )
            return skill_matrix

        normalized_matrix = pd.DataFrame(
            normalized_values, index=skill_matrix.index, columns=skill_matrix.columns
        )

        return normalized_matrix

    def _apply_normalize(self, skill_matrix: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        保存されたscalerを使って正規化を適用（テストデータ用）

        Args:
            skill_matrix: スキルマトリックス
            verbose: 詳細情報を出力するか

        Returns:
            正規化済みマトリックス
        """
        if self.normalization_method is None:
            if verbose:
                logger.info("\n--- 正規化スキップ ---")
            return skill_matrix

        if verbose:
            logger.info(f"\n--- 正規化（transform only: {self.normalization_method}） ---")

        if self.normalization_method == "l2":
            # L2正規化はstatelessなので直接適用
            normalized_values = normalize(skill_matrix.values, norm="l2", axis=1)
            if verbose:
                logger.info("✅ L2 normalization applied")

        elif self.scaler_ is not None:
            # 訓練データで計算した統計を使用
            normalized_values = self.scaler_.transform(skill_matrix.values)

            if self.normalization_method == "standard":
                normalized_values = np.clip(normalized_values, 0, None)  # NMF用

            if verbose:
                logger.info(f"✅ {self.scaler_.__class__.__name__} transform applied (using training stats)")

        else:
            raise RuntimeError(
                "Scaler not found. This should not happen if fit_transform_train() was called."
            )

        normalized_matrix = pd.DataFrame(
            normalized_values, index=skill_matrix.index, columns=skill_matrix.columns
        )

        return normalized_matrix


def create_preprocessor_from_config(config) -> SkillMatrixPreprocessor:
    """
    Configから前処理器を作成

    Args:
        config: Configクラスインスタンス

    Returns:
        SkillMatrixPreprocessor
    """
    params = config.DATA_PREPROCESSING_PARAMS

    return SkillMatrixPreprocessor(
        min_competences_per_member=params["min_competences_per_member"],
        min_members_per_competence=params["min_members_per_competence"],
        normalization_method=params["normalization_method"],
    )
