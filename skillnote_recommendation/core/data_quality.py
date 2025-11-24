"""
データ品質チェック

GAFAレベルのデータバリデーション:
- 欠損値検出
- 無限値検出
- データ分布検証
- データドリフト検出（将来拡張用）
"""

import pandas as pd
import numpy as np
from typing import Any
from numpy.typing import NDArray

from skillnote_recommendation.core.schemas import DataQualityReport
from skillnote_recommendation.core.logging_config import LoggerMixin


class DataQualityChecker(LoggerMixin):
    """
    データ品質検証クラス

    様々なデータ品質チェックを実行し、レポートを生成する
    """

    def validate_skill_matrix(
        self, matrix: pd.DataFrame, strict: bool = False
    ) -> DataQualityReport:
        """
        スキルマトリックスを検証

        Args:
            matrix: スキルマトリックス（メンバー×力量）
            strict: 厳格モード（警告もエラーとして扱う）

        Returns:
            DataQualityReport
        """
        report = DataQualityReport(is_valid=True)

        self.logger.info("data_quality_check_started", matrix_shape=matrix.shape, strict=strict)

        # 1. 基本的な形状チェック
        self._check_shape(matrix, report)

        # 2. 欠損値チェック
        self._check_missing_values(matrix, report)

        # 3. 無限値チェック
        self._check_infinite_values(matrix, report)

        # 4. 負の値チェック（NMFは非負が必要）
        self._check_negative_values(matrix, report, strict)

        # 5. スパース性チェック
        self._check_sparsity(matrix, report, strict)

        # 6. データ型チェック
        self._check_dtypes(matrix, report)

        # 7. 分布チェック
        self._check_distribution(matrix, report, strict)

        # 統計情報を設定
        self._calculate_statistics(matrix, report)

        # strictモードの場合、警告もエラーとして扱う
        if strict and report.warnings:
            for warning in report.warnings:
                report.add_error(f"[STRICT] {warning}")

        self.logger.info(
            "data_quality_check_completed",
            is_valid=report.is_valid,
            errors_count=len(report.errors),
            warnings_count=len(report.warnings),
        )

        return report

    def _check_shape(self, matrix: pd.DataFrame, report: DataQualityReport) -> None:
        """形状チェック"""
        if matrix.shape[0] == 0:
            report.add_error("Matrix has zero rows (no members)")

        if matrix.shape[1] == 0:
            report.add_error("Matrix has zero columns (no competences)")

        if matrix.shape[0] < 10:
            report.add_warning(f"Matrix has only {matrix.shape[0]} members (very small dataset)")

        if matrix.shape[1] < 10:
            report.add_warning(
                f"Matrix has only {matrix.shape[1]} competences (very small dataset)"
            )

    def _check_missing_values(self, matrix: pd.DataFrame, report: DataQualityReport) -> None:
        """欠損値チェック"""
        nan_count = matrix.isna().sum().sum()

        if nan_count > 0:
            nan_ratio = nan_count / matrix.size
            report.add_error(f"Found {nan_count} NaN values ({nan_ratio:.2%} of matrix)")
            report.set_statistic("nan_count", float(nan_count))
            report.set_statistic("nan_ratio", nan_ratio)
        else:
            report.set_statistic("nan_count", 0.0)
            report.set_statistic("nan_ratio", 0.0)

    def _check_infinite_values(self, matrix: pd.DataFrame, report: DataQualityReport) -> None:
        """無限値チェック"""
        inf_count = np.isinf(matrix.values).sum()

        if inf_count > 0:
            inf_ratio = inf_count / matrix.size
            report.add_error(f"Found {inf_count} Inf values ({inf_ratio:.2%} of matrix)")
            report.set_statistic("inf_count", float(inf_count))
        else:
            report.set_statistic("inf_count", 0.0)

    def _check_negative_values(
        self, matrix: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        """負の値チェック"""
        negative_mask = matrix < 0
        neg_count = negative_mask.sum().sum()

        if neg_count > 0:
            neg_ratio = neg_count / matrix.size
            message = f"Found {neg_count} negative values ({neg_ratio:.2%}). NMF requires non-negative values."

            if strict:
                report.add_error(message)
            else:
                report.add_warning(message)

            report.set_statistic("negative_count", float(neg_count))
        else:
            report.set_statistic("negative_count", 0.0)

    def _check_sparsity(
        self, matrix: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        """スパース性チェック"""
        zero_count = (matrix == 0).sum().sum()
        sparsity = zero_count / matrix.size

        report.set_statistic("sparsity", sparsity)
        report.set_statistic("density", 1.0 - sparsity)

        if sparsity > 0.99:
            message = f"Matrix is {sparsity:.1%} sparse (extremely sparse)"
            if strict:
                report.add_error(message)
            else:
                report.add_warning(message)
        elif sparsity > 0.95:
            report.add_warning(f"Matrix is {sparsity:.1%} sparse (very sparse)")

    def _check_dtypes(self, matrix: pd.DataFrame, report: DataQualityReport) -> None:
        """データ型チェック"""
        non_numeric_cols = []

        for col in matrix.columns:
            if not pd.api.types.is_numeric_dtype(matrix[col]):
                non_numeric_cols.append(str(col))

        if non_numeric_cols:
            report.add_error(
                f"Found non-numeric columns: {', '.join(non_numeric_cols[:10])}"
                + (f" and {len(non_numeric_cols) - 10} more" if len(non_numeric_cols) > 10 else "")
            )

    def _check_distribution(
        self, matrix: pd.DataFrame, report: DataQualityReport, strict: bool
    ) -> None:
        """分布チェック"""
        # 非ゼロ値のみを対象に分布を確認
        non_zero_values = matrix.values[matrix.values > 0]

        if len(non_zero_values) == 0:
            report.add_error("Matrix contains no positive values")
            return

        mean_val = float(np.mean(non_zero_values))
        std_val = float(np.std(non_zero_values))
        min_val = float(np.min(non_zero_values))
        max_val = float(np.max(non_zero_values))

        # 標準偏差が0（全ての値が同じ）
        if std_val == 0:
            report.add_warning("All non-zero values are identical (std=0)")

        # 極端な外れ値チェック（平均 ± 10σ）
        if max_val > mean_val + 10 * std_val:
            report.add_warning(
                f"Detected extreme outliers (max={max_val:.2f} >> mean+10σ={mean_val + 10 * std_val:.2f})"
            )

        # 値の範囲が広すぎる
        if max_val / min_val > 1000:
            report.add_warning(f"Value range is very large (min={min_val:.2e}, max={max_val:.2e})")

    def _calculate_statistics(self, matrix: pd.DataFrame, report: DataQualityReport) -> None:
        """統計情報を計算"""
        values = matrix.values
        non_zero_values = values[values > 0]

        report.set_statistic("total_elements", float(matrix.size))
        report.set_statistic("n_rows", float(matrix.shape[0]))
        report.set_statistic("n_cols", float(matrix.shape[1]))

        if len(non_zero_values) > 0:
            report.set_statistic("mean", float(np.mean(non_zero_values)))
            report.set_statistic("std", float(np.std(non_zero_values)))
            report.set_statistic("min", float(np.min(non_zero_values)))
            report.set_statistic("max", float(np.max(non_zero_values)))
            report.set_statistic("median", float(np.median(non_zero_values)))
        else:
            report.set_statistic("mean", 0.0)
            report.set_statistic("std", 0.0)
            report.set_statistic("min", 0.0)
            report.set_statistic("max", 0.0)
            report.set_statistic("median", 0.0)

    def validate_dataframe(
        self, df: pd.DataFrame, required_columns: list[str], name: str = "DataFrame"
    ) -> DataQualityReport:
        """
        一般的なDataFrameを検証

        Args:
            df: 検証対象のDataFrame
            required_columns: 必須カラムのリスト
            name: データの名前（ログ用）

        Returns:
            DataQualityReport
        """
        report = DataQualityReport(is_valid=True)

        self.logger.info("dataframe_validation_started", name=name, shape=df.shape)

        # 空チェック
        if df.empty:
            report.add_error(f"{name} is empty")
            return report

        # 必須カラムチェック
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            report.add_error(f"{name} is missing required columns: {', '.join(missing_cols)}")

        # 重複行チェック
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            report.add_warning(f"{name} contains {duplicate_count} duplicate rows")

        # 基本統計
        report.set_statistic("row_count", float(len(df)))
        report.set_statistic("column_count", float(len(df.columns)))
        report.set_statistic("duplicate_count", float(duplicate_count))

        self.logger.info("dataframe_validation_completed", name=name, is_valid=report.is_valid)

        return report
