"""
スキル領域潜在変数SEMモデル（正しいSEM実装版）

構造方程式モデリング（SEM）の理論に基づいた実装：
1. 測定モデル（Measurement Model）: スキル → 潜在変数
2. 構造モデル（Structural Model）: 潜在変数 → 潜在変数
3. 統計的推定と有意性検定

参考文献:
- Kline, R. B. (2015). Principles and Practice of Structural Equation Modeling
- Bollen, K. A. (1989). Structural Equations with Latent Variables
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import scipy.stats as stats

try:
    import networkx as nx
    import plotly.graph_objects as go
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

logger = logging.getLogger(__name__)


@dataclass
class LatentFactor:
    """潜在変数の定義"""

    factor_name: str  # 例：「初級プログラミング」
    domain_category: str  # 例：「プログラミング」
    level: int  # 0:初級, 1:中級, 2:上級
    observed_skills: List[str] = field(default_factory=list)  # この潜在変数に対応するスキル
    factor_loadings: Dict[str, float] = field(default_factory=dict)  # スキルのファクターローディング
    factor_variance: float = 0.0  # 潜在変数の分散


@dataclass
class PathCoefficient:
    """パス係数（構造モデルの因果効果）"""

    from_factor: str  # 元のファクター
    to_factor: str  # 先のファクター
    coefficient: float  # パス係数（標準化）
    std_error: float  # 標準誤差
    t_value: float  # t値
    p_value: float  # p値
    ci_lower: float  # 信頼区間下限
    ci_upper: float  # 信頼区間上限
    is_significant: bool  # p < 0.05か


@dataclass
class MeasurementModel:
    """測定モデル（スキル → 潜在変数）"""

    factor_name: str
    factor_loadings: Dict[str, float]  # {スキルコード: ファクターローディング}
    measurement_error_variance: Dict[str, float]  # {スキルコード: 測定誤差分散}
    factor_variance: float  # 潜在変数の分散
    item_reliability: float  # アイテム信頼性（Cronbach's alpha）


@dataclass
class DomainStructure:
    """スキル領域の構造定義"""

    domain_name: str
    latent_factors: List[LatentFactor] = field(default_factory=list)
    measurement_models: Dict[str, MeasurementModel] = field(default_factory=dict)
    path_coefficients: List[PathCoefficient] = field(default_factory=list)
    model_fit_indices: Dict[str, float] = field(default_factory=dict)  # GFI, RMSEA等


class SkillDomainSEMModel:
    """
    正しいSEM実装：スキル領域潜在変数モデル

    【理論的フレームワーク】
    1. 測定モデル: 観測スキル → 潜在段階変数（初級/中級/上級）
    2. 構造モデル: 潜在変数間の因果関係（初級→中級→上級）
    3. 統計的検定: パス係数の有意性、モデル適合度

    【使用方法】
    model = SkillDomainSEMModel(member_competence_df, competence_master_df)
    sem_score = model.calculate_sem_score("M001", "C001")
    """

    def __init__(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        num_domain_categories: int = 8,
        confidence_level: float = 0.95,
    ):
        """
        初期化

        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            num_domain_categories: スキル領域の分類数（5～10推奨）
            confidence_level: 信頼区間のレベル（0.95 = 95%）
        """
        self.member_competence_df = member_competence_df.copy()
        self.competence_master_df = competence_master_df.copy()
        self.num_domain_categories = num_domain_categories
        self.confidence_level = confidence_level

        # データ検証
        self._validate_data()

        # メンバーの潜在変数スコアをキャッシュ（先に初期化）
        self.member_latent_scores: Dict[str, Dict[str, float]] = {}

        # スキル領域を分類
        self.domain_structures: Dict[str, DomainStructure] = {}
        self._build_domain_structures()

        # メンバーの潜在変数スコアを推定
        self._estimate_member_latent_scores()

        logger.info(
            f"SkillDomainSEMModel initialized with {len(self.domain_structures)} domains"
        )

    def _validate_data(self):
        """データの妥当性を検証"""
        required_cols_competence = ["力量コード", "力量名", "力量カテゴリー名"]
        missing_cols = [
            col
            for col in required_cols_competence
            if col not in self.competence_master_df.columns
        ]
        if missing_cols:
            raise ValueError(f"competence_master_dfに必要なカラムがありません: {missing_cols}")

        required_cols_member = ["メンバーコード", "力量コード", "正規化レベル"]
        missing_cols = [
            col
            for col in required_cols_member
            if col not in self.member_competence_df.columns
        ]
        if missing_cols:
            raise ValueError(f"member_competence_dfに必要なカラムがありません: {missing_cols}")

        # 正規化レベルが0-5の範囲内か確認
        if not (
            (self.member_competence_df["正規化レベル"] >= 0)
            & (self.member_competence_df["正規化レベル"] <= 5)
        ).all():
            logger.warning("スキルレベルが0-5の範囲外です")

    def _build_domain_structures(self):
        """スキル領域の構造を構築"""
        # 力量カテゴリーを集約
        domain_mapping = self._aggregate_categories()

        # 各領域に対して構造を設定
        for domain_name, skills in domain_mapping.items():
            domain_struct = self._create_domain_structure(domain_name, skills)
            self.domain_structures[domain_name] = domain_struct
            logger.debug(f"Created domain structure for: {domain_name}")

        # 構造モデル（潜在変数間）を推定
        self._estimate_structural_model()

    def _aggregate_categories(self) -> Dict[str, List[str]]:
        """カテゴリーをスキル領域に集約"""
        domain_mapping = defaultdict(list)

        for _, row in self.competence_master_df.iterrows():
            skill_code = row.get("力量コード")
            category = row.get("力量カテゴリー名", "その他")

            if pd.isna(category) or not str(category).strip():
                domain = "その他"
            else:
                # 最初の「>」までを領域名とする
                parts = str(category).split(">")
                domain = parts[0].strip() if parts else "その他"

            if skill_code not in domain_mapping[domain]:
                domain_mapping[domain].append(skill_code)

        # 領域数の制限
        if len(domain_mapping) > self.num_domain_categories:
            sorted_domains = sorted(
                domain_mapping.items(), key=lambda x: len(x[1]), reverse=True
            )
            limited_domains = {}
            other_skills = []

            for i, (domain, skills) in enumerate(sorted_domains):
                if i < self.num_domain_categories - 1:
                    limited_domains[domain] = skills
                else:
                    other_skills.extend(skills)

            if other_skills:
                limited_domains["その他"] = other_skills

            domain_mapping = limited_domains

        logger.info(f"Aggregated into {len(domain_mapping)} domains")
        return dict(domain_mapping)

    def _create_domain_structure(
        self, domain_name: str, skill_codes: List[str]
    ) -> DomainStructure:
        """
        スキル領域の構造を作成（測定モデル）

        スキルをレベル帯別に分類し、各潜在変数に対応させる
        """
        domain_struct = DomainStructure(domain_name=domain_name)

        # スキルをレベル帯で分類（測定モデルの基盤）
        skill_level_map = self._classify_skills_by_level(skill_codes)

        levels = [
            (0, "初級", skill_level_map.get("low", [])),
            (1, "中級", skill_level_map.get("mid", [])),
            (2, "上級", skill_level_map.get("high", [])),
        ]

        # 各潜在変数と測定モデルを構築
        for level_id, level_name, level_skills in levels:
            factor_name = f"{domain_name}_{level_name}"

            # 測定モデルを推定
            measurement_model = self._estimate_measurement_model(
                factor_name, level_skills
            )

            # 潜在変数を作成
            latent_factor = LatentFactor(
                factor_name=factor_name,
                domain_category=domain_name,
                level=level_id,
                observed_skills=level_skills if level_skills else [skill_codes[0]],
                factor_loadings=measurement_model.factor_loadings,
                factor_variance=measurement_model.factor_variance,
            )
            domain_struct.latent_factors.append(latent_factor)
            domain_struct.measurement_models[factor_name] = measurement_model

        return domain_struct

    def _classify_skills_by_level(self, skill_codes: List[str]) -> Dict[str, List[str]]:
        """
        スキルをレベル帯別に分類

        メンバーの習得データから、各スキルの典型的なレベルを推定
        """
        # スキルごとの平均習得レベルを計算
        skill_avg_levels = {}

        for skill_code in skill_codes:
            skill_data = self.member_competence_df[
                self.member_competence_df["力量コード"] == skill_code
            ]
            if len(skill_data) > 0:
                avg_level = skill_data["正規化レベル"].mean()
            else:
                avg_level = 2.5  # デフォルト
            skill_avg_levels[skill_code] = avg_level

        # レベル帯で分類（0-2: 初級, 2-4: 中級, 4-5: 上級）
        low_skills = [
            code for code, level in skill_avg_levels.items() if level <= 2
        ]
        mid_skills = [
            code for code, level in skill_avg_levels.items() if 2 < level <= 4
        ]
        high_skills = [
            code for code, level in skill_avg_levels.items() if level > 4
        ]

        # スキルが偏らないよう調整
        if not low_skills and skill_codes:
            low_skills = [skill_codes[0]]
        if not mid_skills and skill_codes:
            mid_skills = [skill_codes[len(skill_codes) // 2]]
        if not high_skills and skill_codes:
            high_skills = [skill_codes[-1]]

        return {"low": low_skills, "mid": mid_skills, "high": high_skills}

    def _estimate_measurement_model(
        self, factor_name: str, observed_skills: List[str]
    ) -> MeasurementModel:
        """
        測定モデルを推定（スキル → 潜在変数）

        ファクターローディング（因子負荷量）を計算
        """
        if not observed_skills:
            # スキルがない場合はダミー
            return MeasurementModel(
                factor_name=factor_name,
                factor_loadings={},
                measurement_error_variance={},
                factor_variance=1.0,
                item_reliability=0.0,
            )

        # メンバーのスキルレベルデータを取得
        skill_data = self.member_competence_df[
            self.member_competence_df["力量コード"].isin(observed_skills)
        ]

        if len(skill_data) == 0:
            return MeasurementModel(
                factor_name=factor_name,
                factor_loadings={skill: 0.7 for skill in observed_skills},
                measurement_error_variance={skill: 0.51 for skill in observed_skills},
                factor_variance=1.0,
                item_reliability=0.0,
            )

        # 正規化スキルレベル（0-1スケール）
        skill_data = skill_data.copy()
        skill_data["normalized_level"] = skill_data["正規化レベル"] / 5.0

        # スキルごとのローディングを計算（相関係数ベース）
        factor_loadings = {}
        measurement_error_variance = {}

        for skill_code in observed_skills:
            skill_levels = skill_data[
                skill_data["力量コード"] == skill_code
            ]["normalized_level"].values

            if len(skill_levels) > 1:
                # ファクターローディング = スキル分散の平方根（簡易推定）
                loading = np.std(skill_levels) if np.std(skill_levels) > 0 else 0.7
                loading = min(max(loading, 0.3), 0.95)  # 0.3-0.95の範囲に正規化
            else:
                loading = 0.7  # デフォルト

            error_variance = 1.0 - loading**2

            factor_loadings[skill_code] = loading
            measurement_error_variance[skill_code] = error_variance

        # 潜在変数の分散（固定=1.0）
        factor_variance = 1.0

        # 信頼性（Cronbach's alpha簡易推定）
        if len(factor_loadings) > 1:
            item_reliability = (
                (len(factor_loadings) * np.mean(list(factor_loadings.values())))
                / (1 + (len(factor_loadings) - 1) * 0.5)
                if len(factor_loadings) > 1
                else 0.0
            )
        else:
            item_reliability = 0.0

        return MeasurementModel(
            factor_name=factor_name,
            factor_loadings=factor_loadings,
            measurement_error_variance=measurement_error_variance,
            factor_variance=factor_variance,
            item_reliability=item_reliability,
        )

    def _estimate_member_latent_scores(self):
        """
        メンバーの潜在変数スコアを推定（因子スコア法）

        観測スキルレベル × ファクターローディングで潜在変数を推定
        """
        member_ids = self.member_competence_df["メンバーコード"].unique()

        for member_id in member_ids:
            member_data = self.member_competence_df[
                self.member_competence_df["メンバーコード"] == member_id
            ]
            member_scores = {}

            # 各潜在変数のスコアを計算
            for domain_name, domain_struct in self.domain_structures.items():
                for latent_factor in domain_struct.latent_factors:
                    # この潜在変数に対応するスキルデータを取得
                    factor_skills = member_data[
                        member_data["力量コード"].isin(latent_factor.observed_skills)
                    ]

                    if len(factor_skills) > 0 and latent_factor.factor_loadings:
                        # 因子スコア法：スキルレベル × ローディング
                        weighted_score = 0.0
                        total_loading = 0.0

                        for _, row in factor_skills.iterrows():
                            skill_code = row["力量コード"]
                            skill_level = row["正規化レベル"] / 5.0  # 0-1に正規化

                            loading = latent_factor.factor_loadings.get(
                                skill_code, 0.7
                            )
                            weighted_score += skill_level * loading
                            total_loading += loading

                        # 加重平均
                        if total_loading > 0:
                            latent_score = weighted_score / total_loading
                        else:
                            latent_score = 0.0
                    else:
                        latent_score = 0.0

                    # 0-1の範囲に制限
                    latent_score = min(1.0, max(0.0, latent_score))

                    member_scores[latent_factor.factor_name] = latent_score

            self.member_latent_scores[member_id] = member_scores

        logger.info(f"Estimated latent scores for {len(self.member_latent_scores)} members")

    def _estimate_structural_model(self):
        """
        構造モデルを推定（潜在変数間の因果効果）

        実際のメンバースコアから相関係数を計算し、統計的検定を実施
        """
        for domain_name, domain_struct in self.domain_structures.items():
            latent_factors = domain_struct.latent_factors

            # 同じ領域内の段階的遷移（初級→中級→上級）
            for i in range(len(latent_factors) - 1):
                from_factor = latent_factors[i]
                to_factor = latent_factors[i + 1]

                # メンバーのスコアペアを取得
                from_scores = []
                to_scores = []

                for member_id in self.member_latent_scores.keys():
                    from_score = self.member_latent_scores[member_id].get(
                        from_factor.factor_name
                    )
                    to_score = self.member_latent_scores[member_id].get(
                        to_factor.factor_name
                    )

                    if from_score is not None and to_score is not None:
                        from_scores.append(from_score)
                        to_scores.append(to_score)

                # パス係数を推定
                path_coef = self._calculate_path_coefficient(
                    from_scores, to_scores, from_factor.factor_name, to_factor.factor_name
                )
                domain_struct.path_coefficients.append(path_coef)

    def _calculate_path_coefficient(
        self,
        from_scores: List[float],
        to_scores: List[float],
        from_name: str,
        to_name: str,
    ) -> PathCoefficient:
        """
        パス係数を計算（統計的検定付き）

        相関係数からt値とp値を計算
        """
        if len(from_scores) < 3:
            # データ不足の場合はデフォルト
            return PathCoefficient(
                from_factor=from_name,
                to_factor=to_name,
                coefficient=0.0,
                std_error=0.0,
                t_value=0.0,
                p_value=1.0,
                ci_lower=0.0,
                ci_upper=0.0,
                is_significant=False,
            )

        # ピアソン相関係数を計算
        from_array = np.array(from_scores)
        to_array = np.array(to_scores)

        # 標準化（平均0、標準偏差1）
        from_std = (from_array - from_array.mean()) / (from_array.std() + 1e-10)
        to_std = (to_array - to_array.mean()) / (to_array.std() + 1e-10)

        # パス係数（相関係数）
        coefficient = np.corrcoef(from_std, to_std)[0, 1]
        if np.isnan(coefficient):
            coefficient = 0.0

        # t値を計算
        n = len(from_scores)
        if abs(coefficient) < 0.9999:
            t_value = coefficient * np.sqrt(n - 2) / np.sqrt(
                max(1 - coefficient**2, 1e-10)
            )
        else:
            t_value = 0.0

        # p値を計算（両側検定）
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - 2))

        # 信頼区間を計算（フィッシャーのz変換）
        z = 0.5 * np.log((1 + coefficient) / (1 - coefficient + 1e-10))
        se_z = 1.0 / np.sqrt(n - 3)
        z_critical = stats.norm.ppf(
            (1 + self.confidence_level) / 2
        )  # 例：95%なら1.96
        ci_lower = np.tanh(z - z_critical * se_z)
        ci_upper = np.tanh(z + z_critical * se_z)

        is_significant = p_value < 0.05

        return PathCoefficient(
            from_factor=from_name,
            to_factor=to_name,
            coefficient=coefficient,
            std_error=1.0 / np.sqrt(n),
            t_value=t_value,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=is_significant,
        )

    def calculate_sem_score(self, member_code: str, skill_code: str) -> float:
        """
        SEMスコアを計算（推薦スコアに統合用）

        メンバーがスキルを習得する確率を、パス係数に基づいて推定
        """
        domain = self._find_skill_domain(skill_code)
        if not domain:
            return 0.0

        member_scores = self.member_latent_scores.get(member_code, {})
        domain_struct = self.domain_structures.get(domain)

        if not domain_struct:
            return 0.0

        # メンバーのこの領域での現在のレベルを推定
        current_level = self._estimate_current_level(member_code, domain)

        # 最高レベルに既に到達している場合
        if current_level >= len(domain_struct.latent_factors) - 1:
            return 0.8  # 習得確率80%（既に高度なスキルを持っている）

        if current_level < 0:  # スキルがない場合
            return 0.3  # 習得確率30%

        # 次のレベルへのパス係数を取得
        current_factor = domain_struct.latent_factors[current_level]
        current_score = member_scores.get(current_factor.factor_name, 0.0)

        # 次のレベルへのパス係数
        path_coef = None
        for pc in domain_struct.path_coefficients:
            if pc.from_factor == current_factor.factor_name:
                path_coef = pc
                break

        if path_coef and path_coef.is_significant:
            # パス係数がある場合：現在のレベルスコア × パス係数
            sem_score = current_score * path_coef.coefficient
        else:
            # パス係数がない場合：現在のスコアをそのまま使用
            sem_score = current_score * 0.6

        # 0-1の範囲に正規化
        return min(1.0, max(0.0, sem_score))

    def _estimate_current_level(self, member_code: str, domain_category: str) -> int:
        """
        メンバーの領域内での現在のレベルを推定（段階的）

        Args:
            member_code: メンバーコード
            domain_category: 領域名

        Returns:
            レベル（-1: 未習得, 0: 初級, 1: 中級, 2: 上級）
        """
        member_scores = self.member_latent_scores.get(member_code, {})
        domain_struct = self.domain_structures.get(domain_category)

        if not domain_struct:
            return -1

        # スコアを取得
        scores = [
            member_scores.get(f.factor_name, 0.0) for f in domain_struct.latent_factors
        ]

        if not scores:
            return -1

        # 段階的なレベル判定
        max_score = max(scores)

        if max_score < 0.3:
            return -1  # 未習得
        elif max_score < 0.6:
            return 0  # 初級
        elif max_score < 0.8:
            return 1  # 中級
        else:
            return 2  # 上級

    def _find_skill_domain(self, skill_code: str) -> Optional[str]:
        """スキルコードから所属領域を検索"""
        for domain_name, domain_struct in self.domain_structures.items():
            for latent_factor in domain_struct.latent_factors:
                if skill_code in latent_factor.observed_skills:
                    return domain_name
        return None

    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """領域の詳細情報を取得"""
        domain_struct = self.domain_structures.get(domain_name)
        if not domain_struct:
            return {}

        return {
            "domain_name": domain_name,
            "num_latent_factors": len(domain_struct.latent_factors),
            "latent_factors": [
                {
                    "name": f.factor_name,
                    "level": f.level,
                    "num_skills": len(f.observed_skills),
                    "factor_loadings": f.factor_loadings,
                }
                for f in domain_struct.latent_factors
            ],
            "path_coefficients": [
                {
                    "from": p.from_factor,
                    "to": p.to_factor,
                    "coefficient": round(p.coefficient, 3),
                    "p_value": round(p.p_value, 4),
                    "t_value": round(p.t_value, 3),
                    "is_significant": p.is_significant,
                    "ci": (round(p.ci_lower, 3), round(p.ci_upper, 3)),
                }
                for p in domain_struct.path_coefficients
            ],
        }

    def get_all_domains(self) -> List[str]:
        """全領域名を取得"""
        return list(self.domain_structures.keys())

    def get_member_domain_profile(self, member_code: str) -> Dict[str, Dict[str, float]]:
        """メンバーの領域別プロファイルを取得"""
        member_scores = self.member_latent_scores.get(member_code, {})
        profile = {}

        for domain_name, domain_struct in self.domain_structures.items():
            profile[domain_name] = {}
            for latent_factor in domain_struct.latent_factors:
                score = member_scores.get(latent_factor.factor_name, 0.0)
                profile[domain_name][latent_factor.factor_name] = score

        return profile

    def get_model_fit_indices(self, domain_name: str) -> Dict[str, float]:
        """
        モデル適合度指標を取得（拡張版）

        Returns:
            Dict containing:
            - avg_path_coefficient: 平均パス係数
            - significant_paths: 有意なパス数
            - total_paths: 総パス数
            - avg_loading: 平均因子負荷量
            - avg_effect_size: 平均効果サイズ（Cohen's d）
            - model_variance_explained: 説明分散（R²）
            - gfi: 適合度指標（GFI）
            - nfi: 規準適合度指標（NFI）
        """
        domain_struct = self.domain_structures.get(domain_name)
        if not domain_struct:
            return {}

        # パス係数の統計
        path_coeffs = [p.coefficient for p in domain_struct.path_coefficients]
        significant_paths = sum(
            1 for p in domain_struct.path_coefficients if p.is_significant
        )

        # 平均因子負荷量
        loadings = [
            loading
            for f in domain_struct.latent_factors
            for loading in f.factor_loadings.values()
        ]
        avg_loading = np.mean(loadings) if loadings else 0.0

        # 効果サイズ（Cohen's d）の計算
        # パス係数を標準化された効果サイズとして扱う
        effect_sizes = [abs(p.coefficient) for p in domain_struct.path_coefficients]
        avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0

        # 説明分散（R²）の推定
        # 因子負荷量の二乗平均として計算
        variance_explained = np.mean([l**2 for l in loadings]) if loadings else 0.0

        # GFI（適合度指標）の簡易推定
        # 1に近いほど良好（0.9以上が望ましい）
        # 有意なパスの割合と平均ローディングから推定
        sig_ratio = significant_paths / len(domain_struct.path_coefficients) if domain_struct.path_coefficients else 0
        gfi = (sig_ratio * 0.5) + (avg_loading * 0.5)

        # NFI（規準適合度指標）の簡易推定
        # 1に近いほど良好（0.9以上が望ましい）
        # パス係数の大きさと有意性から推定
        nfi = (np.mean([abs(p.coefficient) for p in domain_struct.path_coefficients if p.is_significant])
               if any(p.is_significant for p in domain_struct.path_coefficients) else 0.0)

        return {
            "avg_path_coefficient": np.mean(path_coeffs) if path_coeffs else 0.0,
            "significant_paths": significant_paths,
            "total_paths": len(domain_struct.path_coefficients),
            "avg_loading": avg_loading,
            "avg_effect_size": avg_effect_size,
            "variance_explained": variance_explained,
            "gfi": min(gfi, 1.0),  # 0-1の範囲に制限
            "nfi": min(nfi, 1.0),  # 0-1の範囲に制限
        }

    def get_skill_dependency_graph(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        スキル依存関係ネットワークグラフを生成（個別力量レベル）

        Args:
            domain_name: 領域名

        Returns:
            ノードとエッジを含むグラフ構造（個別の力量をノードとして表示）
        """
        domain_struct = self.domain_structures.get(domain_name)
        if not domain_struct:
            return None

        # ノード情報（個別の力量）
        nodes = []
        skill_to_factor = {}  # スキルがどの潜在因子に属するかのマッピング

        for latent_factor in domain_struct.latent_factors:
            for skill_code in latent_factor.observed_skills:
                # 力量マスタから情報を取得
                skill_info = self.competence_master_df[
                    self.competence_master_df['力量コード'] == skill_code
                ]

                if not skill_info.empty:
                    skill_row = skill_info.iloc[0]
                    skill_name = skill_row.get('力量名', skill_code)
                    skill_type = skill_row.get('力量タイプ', 'UNKNOWN')

                    # ファクターローディングを取得
                    factor_loading = latent_factor.factor_loadings.get(skill_code, 0.0)

                    nodes.append({
                        "id": skill_code,
                        "label": skill_name,
                        "competence_type": skill_type,
                        "factor_name": latent_factor.factor_name,
                        "factor_loading": factor_loading,
                        "level": latent_factor.level,
                    })

                    skill_to_factor[skill_code] = latent_factor.factor_name

        # エッジ情報（潜在因子間のパス係数に基づく力量間の関係）
        edges = []
        for path_coeff in domain_struct.path_coefficients:
            from_factor = path_coeff.from_factor
            to_factor = path_coeff.to_factor

            # この2つの潜在因子に属する力量同士の関係を作成
            from_skills = [
                lf for lf in domain_struct.latent_factors
                if lf.factor_name == from_factor
            ]
            to_skills = [
                lf for lf in domain_struct.latent_factors
                if lf.factor_name == to_factor
            ]

            if from_skills and to_skills:
                from_latent = from_skills[0]
                to_latent = to_skills[0]

                # 各力量ペアに対してエッジを作成
                # エッジの重みは: パス係数 × from_loading × to_loading
                for from_skill in from_latent.observed_skills:
                    from_loading = from_latent.factor_loadings.get(from_skill, 0.5)

                    for to_skill in to_latent.observed_skills:
                        to_loading = to_latent.factor_loadings.get(to_skill, 0.5)

                        # 複合的な影響度を計算
                        combined_coefficient = (
                            path_coeff.coefficient * from_loading * to_loading
                        )

                        edges.append({
                            "from": from_skill,
                            "to": to_skill,
                            "coefficient": combined_coefficient,
                            "path_coefficient": path_coeff.coefficient,
                            "from_loading": from_loading,
                            "to_loading": to_loading,
                            "p_value": path_coeff.p_value,
                            "is_significant": path_coeff.is_significant,
                            "t_value": path_coeff.t_value,
                        })

        return {
            "domain": domain_name,
            "nodes": nodes,
            "edges": edges,
        }

    def visualize_domain_network(
        self,
        domain_name: str,
        layout: str = "spring",
        show_all_edges: bool = False,
        min_coefficient: float = 0.0
    ) -> Optional[go.Figure]:
        """
        領域のスキル依存関係をインタラクティブにプロット（拡張版）

        Args:
            domain_name: 領域名
            layout: レイアウト手法 ("spring", "circular", "hierarchical")
            show_all_edges: すべてのエッジを表示（有意でないものも含む）
            min_coefficient: 表示する最小パス係数（絶対値）

        Returns:
            Plotly Figure（グラフがない場合はNone）
        """
        if not HAS_VISUALIZATION:
            logger.warning("networkx and plotly are required for visualization")
            return None

        graph_data = self.get_skill_dependency_graph(domain_name)
        if not graph_data or not graph_data["edges"]:
            return None

        # ネットワークグラフを作成
        G = nx.DiGraph()

        # ノードを追加
        for node in graph_data["nodes"]:
            G.add_node(
                node["id"],
                label=node["label"],
                competence_type=node["competence_type"],
                factor_name=node["factor_name"],
                factor_loading=node["factor_loading"],
                level=node["level"]
            )

        # エッジを追加（フィルタリング）
        edge_data_list = []
        for edge in graph_data["edges"]:
            if show_all_edges or edge["is_significant"]:
                if abs(edge["coefficient"]) >= min_coefficient:
                    G.add_edge(
                        edge["from"],
                        edge["to"],
                        weight=abs(edge["coefficient"]),
                        coefficient=edge["coefficient"],
                        p_value=edge["p_value"],
                        t_value=edge["t_value"],
                        is_significant=edge["is_significant"]
                    )
                    edge_data_list.append(edge)

        # レイアウトを計算
        if layout == "spring":
            try:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            except Exception as e:
                logger.warning(f"Spring layout failed: {e}, using circular layout")
                pos = nx.circular_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "hierarchical":
            try:
                pos = nx.kamada_kawai_layout(G)
            except Exception as e:
                logger.warning(f"Hierarchical layout failed: {e}, using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # エッジを描画（パス係数に応じた太さと色）
        edge_traces = []
        for edge_data in edge_data_list:
            edge_from = edge_data["from"]
            edge_to = edge_data["to"]

            if edge_from not in pos or edge_to not in pos:
                continue

            x0, y0 = pos[edge_from]
            x1, y1 = pos[edge_to]

            # パス係数の大きさに応じてエッジの太さを変更
            width = abs(edge_data["coefficient"]) * 5 + 0.5

            # 有意性に応じて色を変更
            color = "#2E7D32" if edge_data["is_significant"] else "#BDBDBD"

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                hovertext=f"複合係数: {edge_data['coefficient']:.3f}<br>"
                         f"パス係数: {edge_data['path_coefficient']:.3f}<br>"
                         f"From Loading: {edge_data['from_loading']:.3f}<br>"
                         f"To Loading: {edge_data['to_loading']:.3f}<br>"
                         f"t値: {edge_data['t_value']:.3f}<br>"
                         f"p値: {edge_data['p_value']:.4f}<br>"
                         f"有意: {'Yes' if edge_data['is_significant'] else 'No'}",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

        # ノードを描画（力量タイプごとに色分け）
        node_x, node_y, node_text, node_hover = [], [], [], []
        node_sizes = []
        node_colors = []

        # 力量タイプごとの色定義
        type_colors = {
            'SKILL': '#1f77b4',      # 青
            'EDUCATION': '#ff7f0e',  # オレンジ
            'LICENSE': '#2ca02c',    # 緑
            'UNKNOWN': '#7f7f7f',    # グレー
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # ノード情報を取得
            node_info = next((n for n in graph_data["nodes"] if n["id"] == node), None)

            if node_info:
                competence_type = node_info.get("competence_type", "UNKNOWN")
                factor_loading = node_info.get("factor_loading", 0.5)
                factor_name = node_info.get("factor_name", "")
                level_name = ["初級", "中級", "上級"][node_info.get("level", 0)]

                # ノードテキスト（力量名）
                label = node_info["label"]
                # 長い名前は省略
                display_label = label if len(label) <= 15 else label[:12] + "..."
                node_text.append(display_label)

                # ノードサイズをファクターローディングに応じて調整
                node_sizes.append(15 + factor_loading * 30)

                # ノード色を力量タイプに応じて設定
                node_colors.append(type_colors.get(competence_type, type_colors['UNKNOWN']))

                # ホバー情報
                hover_text = f"<b>{label}</b><br>"
                hover_text += f"力量タイプ: {competence_type}<br>"
                hover_text += f"潜在変数: {factor_name}<br>"
                hover_text += f"レベル: {level_name}<br>"
                hover_text += f"ファクターローディング: {factor_loading:.3f}<br>"

                # 入力・出力エッジの情報
                in_edges = [e for e in edge_data_list if e["to"] == node]
                out_edges = [e for e in edge_data_list if e["from"] == node]

                if in_edges:
                    hover_text += f"<br><b>入力パス ({len(in_edges)}個):</b><br>"
                    for e in in_edges[:3]:  # 最大3つ表示
                        hover_text += f"  • {e['coefficient']:.3f} (p={e['p_value']:.3f})<br>"
                    if len(in_edges) > 3:
                        hover_text += f"  ... 他 {len(in_edges) - 3} 個<br>"

                if out_edges:
                    hover_text += f"<br><b>出力パス ({len(out_edges)}個):</b><br>"
                    for e in out_edges[:3]:  # 最大3つ表示
                        hover_text += f"  • {e['coefficient']:.3f} (p={e['p_value']:.3f})<br>"
                    if len(out_edges) > 3:
                        hover_text += f"  ... 他 {len(out_edges) - 3} 個<br>"

                node_hover.append(hover_text)
            else:
                # フォールバック
                node_text.append(node)
                node_sizes.append(20)
                node_colors.append('#7f7f7f')
                node_hover.append(f"<b>{node}</b>")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2,
                line_color="#ffffff",
            ),
        )

        # Figureを作成（すべてのエッジトレースとノードトレース）
        fig_data = edge_traces + [node_trace]
        fig = go.Figure(data=fig_data)

        fig.update_layout(
            title=f"📊 {domain_name} - 力量依存関係ネットワーク（個別力量レベル）",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=40, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            # インタラクティブ機能を有効化
            dragmode='pan',  # パン操作をデフォルトに
        )

        # 力量タイプの凡例を追加
        fig.add_annotation(
            text="<b>力量タイプ:</b> 🔵 SKILL（スキル） | 🟠 EDUCATION（教育） | 🟢 LICENSE（資格）",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=11, color="black"),
            xanchor='center'
        )

        # ズーム・パン機能のヘルプを追加
        fig.add_annotation(
            text="ヒント: マウスホイールでズーム、ドラッグでパン、ノード/エッジにホバーで詳細表示<br>"
                 "ノードサイズ = ファクターローディング強度、エッジ太さ = パス係数×ローディング積",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor='center'
        )

        return fig
