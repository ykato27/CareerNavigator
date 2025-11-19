"""
統一SEM推定器（Unified SEM Estimator）

semopyライブラリを使用した実装:
- 統一された目的関数（最尤推定）
- 明示的な共分散構造モデル
- 標準的な適合度指標（RMSEA, CFI, TLI, AIC, BIC）
- 測定誤差の明示的モデル化

理論的根拠:
- Bollen, K. A. (1989). Structural Equations with Latent Variables
- Kline, R. B. (2015). Principles and Practice of SEM
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import scipy.stats as stats

from skillnote_recommendation.config import config
from skillnote_recommendation.utils.logger import setup_logger

logger = setup_logger(__name__)

# Optional dependency
try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    semopy = None


@dataclass
class SEMParameter:
    """SEMパラメータ"""
    name: str
    value: float
    std_error: Optional[float] = None
    z_value: Optional[float] = None
    p_value: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    is_significant: bool = False


@dataclass
class SEMFitIndices:
    """SEM適合度指標"""
    # カイ二乗検定
    chi_square: float
    df: int
    p_value: float

    # 絶対適合度指標
    gfi: float  # Goodness of Fit Index
    agfi: float  # Adjusted GFI
    rmsea: float  # Root Mean Square Error of Approximation
    rmsea_ci_lower: float
    rmsea_ci_upper: float

    # 相対適合度指標
    nfi: float  # Normed Fit Index
    cfi: float  # Comparative Fit Index
    tli: float  # Tucker-Lewis Index

    # 情報量基準
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion

    # その他
    srmr: float  # Standardized Root Mean Square Residual
    n_obs: int
    n_params: int

    def is_good_fit(self) -> bool:
        """良好な適合か判定（厳格な基準）"""
        return (
            self.rmsea < 0.06 and
            self.cfi > 0.95 and
            self.tli > 0.95 and
            self.srmr < 0.06
        )

    def is_excellent_fit(self) -> bool:
        """優れた適合か判定（厳格な基準）"""
        return (
            self.rmsea < 0.05 and
            self.cfi > 0.97 and
            self.tli > 0.97 and
            self.srmr < 0.05
        )


@dataclass
class MeasurementModelSpec:
    """測定モデルの仕様"""
    latent_name: str
    observed_vars: List[str]
    reference_indicator: Optional[str] = None  # 参照指標（λ=1に固定）


@dataclass
class StructuralModelSpec:
    """構造モデルの仕様"""
    from_latent: str
    to_latent: str


class SEMConvergenceError(Exception):
    """SEM推定が収束しなかった場合の例外"""
    pass


class UnifiedSEMEstimator:
    """
    統一SEM推定器（semopyベース）

    semopyライブラリを使用して、測定モデルと構造モデルのパラメータを同時推定します。
    既存のインターフェースとの完全な後方互換性を維持しています。

    使用例:
    --------
    # モデル仕様の定義
    measurement_specs = [
        MeasurementModelSpec(
            latent_name="技術力",
            observed_vars=["Python", "SQL", "機械学習"]
        ),
        MeasurementModelSpec(
            latent_name="ビジネス力",
            observed_vars=["企画力", "交渉力", "プレゼン力"]
        )
    ]

    structural_specs = [
        StructuralModelSpec(from_latent="技術力", to_latent="総合力"),
        StructuralModelSpec(from_latent="ビジネス力", to_latent="総合力")
    ]

    # モデルの推定
    estimator = UnifiedSEMEstimator(measurement_specs, structural_specs)
    estimator.fit(data)

    # 結果の取得
    print(estimator.fit_indices_)
    print(estimator.parameters_)
    """

    def __init__(
        self,
        measurement_specs: List[MeasurementModelSpec],
        structural_specs: List[StructuralModelSpec],
        method: str = 'ML',
        confidence_level: float = 0.95,
    ):
        """
        初期化

        Parameters:
        -----------
        measurement_specs: List[MeasurementModelSpec]
            測定モデルの仕様
        structural_specs: List[StructuralModelSpec]
            構造モデルの仕様
        method: str
            推定方法（'ML': 最尤推定）
        confidence_level: float
            信頼区間の水準
        """
        if not SEMOPY_AVAILABLE:
            raise ImportError(
                "semopy is not installed. Please install it with: pip install semopy>=2.3.9"
            )

        self.measurement_specs = measurement_specs
        self.structural_specs = structural_specs
        self.method = method
        self.confidence_level = confidence_level

        # semopy用の属性
        self.model_desc = ""
        self.model = None

        # 既存の属性（後方互換性のため）
        self.parameters_ = {}
        self.fit_indices_ = None
        self.is_fitted = False

        # モデル仕様の検証
        self._validate_specs()

        logger.info(
            f"UnifiedSEMEstimator initialized with {len(measurement_specs)} measurement models "
            f"and {len(structural_specs)} structural paths (semopy backend)"
        )

    def _validate_specs(self):
        """モデル仕様の妥当性を検証"""
        # 測定モデルの検証
        if not self.measurement_specs:
            raise ValueError("測定モデルの仕様が空です")

        latent_names = set()
        for spec in self.measurement_specs:
            if not spec.observed_vars:
                raise ValueError(f"潜在変数 '{spec.latent_name}' に観測変数が指定されていません")
            if len(spec.observed_vars) < 2:
                logger.warning(
                    f"潜在変数 '{spec.latent_name}' の観測変数が2個未満です。"
                    "識別のため最低2個の観測変数が推奨されます。"
                )
            latent_names.add(spec.latent_name)

        # 構造モデルの検証
        for spec in self.structural_specs:
            if spec.from_latent not in latent_names:
                raise ValueError(f"構造モデルの '{spec.from_latent}' が測定モデルに定義されていません")
            if spec.to_latent not in latent_names:
                raise ValueError(f"構造モデルの '{spec.to_latent}' が測定モデルに定義されていません")

    def fit(self, data: pd.DataFrame):
        """
        モデルを推定

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ（各列が観測変数）

        Returns:
        --------
        self: UnifiedSEMEstimator
            推定済みのモデル
        """
        logger.info("Starting SEM estimation with semopy...")

        try:
            # 1. lavaan構文を生成
            self.model_desc = self._build_semopy_syntax()
            logger.debug(f"Model syntax:\n{self.model_desc}")

            # 2. semopyモデルを構築・推定
            self.model = semopy.Model(self.model_desc)
            self.model.fit(data, obj='MLW')  # Maximum Likelihood with Wishart

            # 3. 結果を既存の形式に変換
            self._extract_results(data)

            self.is_fitted = True
            logger.info("SEM estimation completed successfully")
            return self

        except Exception as e:
            logger.error(f"SEM estimation failed: {e}")
            raise SEMConvergenceError(f"モデル推定に失敗しました: {e}")

    def _build_semopy_syntax(self) -> str:
        """
        スペックからlavaan構文を生成

        Returns:
        --------
        str: lavaan形式のモデル記述
        """
        lines = []

        # 測定モデル: Latent =~ Obs1 + Obs2 + Obs3
        for spec in self.measurement_specs:
            obs_str = " + ".join(spec.observed_vars)
            lines.append(f"{spec.latent_name} =~ {obs_str}")

        # 構造モデル: Target ~ Source
        for spec in self.structural_specs:
            lines.append(f"{spec.to_latent} ~ {spec.from_latent}")

        return "\n".join(lines)

    def _extract_results(self, data: pd.DataFrame):
        """
        semopyの結果を既存の形式に変換

        Parameters:
        -----------
        data: pd.DataFrame
            観測データ（適合度計算に使用）
        """
        # パラメータ推定値を取得
        params_df = self.model.inspect()

        # SEMParameterオブジェクトに変換
        self.parameters_ = {}
        for _, row in params_df.iterrows():
            # パラメータ名を構築
            param_name = f"{row['lval']} {row['op']} {row['rval']}"

            param = SEMParameter(
                name=param_name,
                value=row['Estimate'],
                std_error=row.get('Std. Err', None),
                z_value=row.get('z-value', None),
                p_value=row.get('p-value', None),
            )

            # 有意性判定
            if param.p_value is not None:
                alpha = 1 - self.confidence_level
                param.is_significant = param.p_value < alpha

            # 信頼区間（正規分布を仮定）
            if param.std_error is not None and param.std_error > 0:
                z_critical = stats.norm.ppf(1 - alpha / 2)
                param.ci_lower = param.value - z_critical * param.std_error
                param.ci_upper = param.value + z_critical * param.std_error

            self.parameters_[param_name] = param

        # 適合度指標を取得
        try:
            stats_dict = self.model.calc_stats(data)

            self.fit_indices_ = SEMFitIndices(
                chi_square=stats_dict.get('chi2', 0.0),
                df=int(stats_dict.get('dof', 0)),
                p_value=stats_dict.get('chi2_pvalue', 0.0),
                gfi=0.0,  # semopyは提供しない
                agfi=0.0,  # semopyは提供しない
                rmsea=stats_dict.get('RMSEA', 0.0),
                rmsea_ci_lower=0.0,  # 将来的に計算可能
                rmsea_ci_upper=0.0,  # 将来的に計算可能
                nfi=0.0,  # semopyは提供しない
                cfi=stats_dict.get('CFI', 0.0),
                tli=stats_dict.get('TLI', 0.0),
                aic=stats_dict.get('AIC', 0.0),
                bic=stats_dict.get('BIC', 0.0),
                srmr=0.0,  # semopyは提供しない
                n_obs=len(data),
                n_params=len(params_df),
            )

            logger.info(
                f"Fit indices: RMSEA={self.fit_indices_.rmsea:.3f}, "
                f"CFI={self.fit_indices_.cfi:.3f}, TLI={self.fit_indices_.tli:.3f}"
            )

        except Exception as e:
            logger.warning(f"Failed to calculate fit indices: {e}")
            # フォールバック: 最小限の適合度指標
            self.fit_indices_ = SEMFitIndices(
                chi_square=0.0,
                df=0,
                p_value=1.0,
                gfi=0.0,
                agfi=0.0,
                rmsea=0.0,
                rmsea_ci_lower=0.0,
                rmsea_ci_upper=0.0,
                nfi=0.0,
                cfi=0.0,
                tli=0.0,
                aic=0.0,
                bic=0.0,
                srmr=0.0,
                n_obs=len(data),
                n_params=len(params_df),
            )

    def get_parameter(self, name: str) -> Optional[SEMParameter]:
        """
        パラメータを名前で取得

        Parameters:
        -----------
        name: str
            パラメータ名

        Returns:
        --------
        SEMParameter or None
        """
        return self.parameters_.get(name)

    def get_all_parameters(self) -> Dict[str, SEMParameter]:
        """
        すべてのパラメータを取得

        Returns:
        --------
        Dict[str, SEMParameter]
        """
        return self.parameters_

    def get_fit_indices(self) -> Optional[SEMFitIndices]:
        """
        適合度指標を取得

        Returns:
        --------
        SEMFitIndices or None
        """
        return self.fit_indices_

    def summary(self) -> str:
        """
        推定結果のサマリーを文字列で返す

        Returns:
        --------
        str: サマリー文字列
        """
        if not self.is_fitted:
            return "Model not fitted yet."

        lines = []
        lines.append("=" * 80)
        lines.append("SEM Estimation Results (semopy backend)")
        lines.append("=" * 80)
        lines.append("")

        # 適合度指標
        if self.fit_indices_:
            lines.append("Fit Indices:")
            lines.append(f"  Chi-square: {self.fit_indices_.chi_square:.3f} (df={self.fit_indices_.df}, p={self.fit_indices_.p_value:.3f})")
            lines.append(f"  RMSEA: {self.fit_indices_.rmsea:.3f}")
            lines.append(f"  CFI: {self.fit_indices_.cfi:.3f}")
            lines.append(f"  TLI: {self.fit_indices_.tli:.3f}")
            lines.append(f"  AIC: {self.fit_indices_.aic:.3f}")
            lines.append(f"  BIC: {self.fit_indices_.bic:.3f}")
            lines.append("")

        # パラメータ
        lines.append("Parameter Estimates:")
        lines.append(f"{'Parameter':<40} {'Estimate':>10} {'Std.Err':>10} {'z-value':>10} {'p-value':>10}")
        lines.append("-" * 80)

        for name, param in self.parameters_.items():
            sig_marker = "*" if param.is_significant else " "
            lines.append(
                f"{name:<40} {param.value:>10.3f} "
                f"{param.std_error if param.std_error else 0:>10.3f} "
                f"{param.z_value if param.z_value else 0:>10.3f} "
                f"{param.p_value if param.p_value else 1:>10.3f}{sig_marker}"
            )

        lines.append("")
        lines.append("Note: * indicates significance at the specified confidence level")
        lines.append("=" * 80)

        return "\n".join(lines)
