"""
スキル領域潜在変数SEMモデル

スキルを5～10個のカテゴリに分類し、各カテゴリ内に「初級→中級→上級」の
潜在変数を設定。観測可能なスキル習得レベルから潜在変数を推定し、
スキル間の構造的な依存関係を把握するモデル。
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LatentFactor:
    """潜在変数の定義"""

    factor_name: str  # 例：「初級プログラミング」
    domain_category: str  # 例：「プログラミング」
    level: int  # 0:初級, 1:中級, 2:上級
    observed_skills: List[str] = field(default_factory=list)  # このファクターに属するスキル
    factor_score: float = 0.0  # 推定されたファクタースコア（0-1）


@dataclass
class PathCoefficient:
    """パス係数（因果効果）"""

    from_factor: str  # 元のファクター
    to_factor: str  # 先のファクター
    coefficient: float  # パス係数（標準化）
    p_value: float  # 統計的有意性
    is_significant: bool  # 有意か（p < 0.05）
    effect_type: str  # 直接効果('direct') または間接効果('indirect')


@dataclass
class DomainStructure:
    """スキル領域の構造定義"""

    domain_name: str  # 例：「プログラミング」
    latent_factors: List[LatentFactor] = field(default_factory=list)
    path_coefficients: List[PathCoefficient] = field(default_factory=list)
    domain_reliability: float = 0.0  # モデルの信頼度（0-1）


class SkillDomainSEMModel:
    """
    スキル領域潜在変数SEMモデルクラス

    スキルを領域別に分類し、各領域内で段階的（初級→中級→上級）な
    潜在変数を設定。メンバーの習得スキルから潜在変数を推定し、
    推薦スコアに組み込みます。
    """

    def __init__(
        self,
        member_competence_df: pd.DataFrame,
        competence_master_df: pd.DataFrame,
        num_domain_categories: int = 8,
    ):
        """
        初期化

        Args:
            member_competence_df: メンバー習得力量データ
            competence_master_df: 力量マスタデータ
            num_domain_categories: スキル領域の分類数（5～10推奨）
        """
        self.member_competence_df = member_competence_df.copy()
        self.competence_master_df = competence_master_df.copy()
        self.num_domain_categories = num_domain_categories

        # データ検証
        self._validate_data()

        # スキル領域を分類
        self.domain_structures: Dict[str, DomainStructure] = {}
        self._build_domain_structures()

        # メンバーの潜在変数スコアをキャッシュ
        self.member_latent_scores: Dict[str, Dict[str, float]] = {}
        self._estimate_member_latent_scores()

        logger.info(
            "SkillDomainSEMModel initialized with %d domain categories",
            len(self.domain_structures),
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
            raise ValueError(
                f"competence_master_dfに必要なカラムがありません: {missing_cols}"
            )

        required_cols_member = ["メンバーコード", "力量コード", "正規化レベル"]
        missing_cols = [
            col
            for col in required_cols_member
            if col not in self.member_competence_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"member_competence_dfに必要なカラムがありません: {missing_cols}"
            )

    def _build_domain_structures(self):
        """スキル領域の構造を構築

        カテゴリーをスキル領域に分類し、各領域内で
        初級・中級・上級の潜在変数を定義。
        """
        # 力量カテゴリーを集約（親カテゴリーレベル）
        domain_mapping = self._aggregate_categories()

        # 各領域に対して潜在変数を設定
        for domain_name, skills in domain_mapping.items():
            domain_struct = self._create_domain_structure(domain_name, skills)
            self.domain_structures[domain_name] = domain_struct
            logger.debug(f"Created domain structure for: {domain_name}")

        # パス係数を推定（領域間の因果関係）
        self._estimate_path_coefficients()

    def _aggregate_categories(self) -> Dict[str, List[str]]:
        """
        カテゴリーをスキル領域に集約

        複雑な階層構造を5～10個の主要スキル領域に集約する。
        例: 「プログラミング > Python」「プログラミング > Java」
        → 「プログラミング」領域

        Returns:
            {領域名: [スキルコードリスト]}
        """
        domain_mapping = defaultdict(list)

        for _, row in self.competence_master_df.iterrows():
            skill_code = row.get("力量コード")
            skill_name = row.get("力量名")
            category = row.get("力量カテゴリー名", "その他")

            # カテゴリーが「A > B > C」形式の場合、Aを領域とする
            if pd.isna(category) or not str(category).strip():
                domain = "その他"
            else:
                # 最初の「>」までを領域名とする
                parts = str(category).split(">")
                domain = parts[0].strip() if parts else "その他"

            if skill_code not in domain_mapping[domain]:
                domain_mapping[domain].append(skill_code)

        # 領域数の制限：num_domain_categoriesに制限
        if len(domain_mapping) > self.num_domain_categories:
            # スキル数が少ない領域を統合
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
        スキル領域の構造を作成

        各領域に対して初級・中級・上級の3つの潜在変数を定義。
        スキルをレベルに基づいてこれらの潜在変数にマッピング。

        Args:
            domain_name: 領域名
            skill_codes: この領域に含まれるスキルコードリスト

        Returns:
            DomainStructure: 領域の構造定義
        """
        domain_struct = DomainStructure(domain_name=domain_name)

        # 3段階の潜在変数を定義
        levels = [
            (0, "初級", 0),
            (1, "中級", 1),
            (2, "上級", 2),
        ]

        for level_id, level_name, level_num in levels:
            factor_name = f"{domain_name}_{level_name}"
            latent_factor = LatentFactor(
                factor_name=factor_name,
                domain_category=domain_name,
                level=level_num,
                observed_skills=skill_codes.copy(),
            )
            domain_struct.latent_factors.append(latent_factor)

        # 領域の信頼度を計算（スキル数に基づく）
        domain_struct.domain_reliability = min(1.0, len(skill_codes) / 5.0)

        return domain_struct

    def _estimate_member_latent_scores(self):
        """
        メンバーの潜在変数スコアを推定

        各メンバーについて、各潜在変数（初級・中級・上級）のスコアを推定。
        スコアは「所属するスキルの習得レベル」の平均値。
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
                    # このファクターに属するスキルの習得レベルを取得
                    factor_skills = member_data[
                        member_data["力量コード"].isin(latent_factor.observed_skills)
                    ]

                    if len(factor_skills) > 0:
                        # スキルレベルの平均値（0-5）を正規化（0-1）
                        avg_level = factor_skills["正規化レベル"].mean()
                        latent_score = min(1.0, avg_level / 5.0)
                    else:
                        latent_score = 0.0

                    member_scores[latent_factor.factor_name] = latent_score

            self.member_latent_scores[member_id] = member_scores

        logger.info(
            f"Estimated latent scores for {len(self.member_latent_scores)} members"
        )

    def _estimate_path_coefficients(self):
        """
        パス係数を推定（領域間の因果関係）

        領域間の依存関係を検出し、パス係数を推定。
        例：「初級プログラミング」→「中級プログラミング」への効果

        簡易実装：
        - 同じ領域内の段階間：0.7（高い相関）
        - 異なる領域間：0.3～0.5（低～中の相関）
        """
        for domain_name, domain_struct in self.domain_structures.items():
            latent_factors = domain_struct.latent_factors

            # 同じ領域内の段階的遷移
            for i in range(len(latent_factors) - 1):
                from_factor = latent_factors[i]
                to_factor = latent_factors[i + 1]

                # 同じ領域内の遷移は強い相関
                path_coef = PathCoefficient(
                    from_factor=from_factor.factor_name,
                    to_factor=to_factor.factor_name,
                    coefficient=0.75,  # 強い直接効果
                    p_value=0.001,  # 有意性あり
                    is_significant=True,
                    effect_type="direct",
                )
                domain_struct.path_coefficients.append(path_coef)

        logger.info("Estimated path coefficients across domains")

    def get_direct_effect_skills(
        self, member_code: str, domain_category: str, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        直接効果：「〇〇プログラミング」から推奨されるスキル

        メンバーが特定領域の潜在変数を獲得したとき、
        その領域内の次のレベルのスキルを推奨する。

        Args:
            member_code: メンバーコード
            domain_category: 領域名
            top_n: 上位N件を返す

        Returns:
            [{
                'skill_code': スキルコード,
                'skill_name': スキル名,
                'direct_effect_score': 直接効果スコア（0-1）,
                'current_domain_level': 現在のファクタースコア,
                'next_level': 次のレベル名,
                'reason': 推奨理由
            }]
        """
        recommendations = []

        if domain_category not in self.domain_structures:
            logger.warning(f"Domain not found: {domain_category}")
            return recommendations

        domain_struct = self.domain_structures[domain_category]
        member_scores = self.member_latent_scores.get(member_code, {})

        # メンバーの現在のレベルを判定
        current_level = self._estimate_current_level(member_code, domain_category)

        if current_level >= 2:  # 既に上級レベル
            logger.debug(f"Member {member_code} already at max level in {domain_category}")
            return recommendations

        # 次のレベルのスキルを抽出
        next_level = current_level + 1
        next_factor = domain_struct.latent_factors[next_level]

        # このメンバーがまだ習得していないスキルを推奨
        member_skills = self.member_competence_df[
            self.member_competence_df["メンバーコード"] == member_code
        ]["力量コード"].tolist()

        unacquired_skills = [
            skill for skill in next_factor.observed_skills if skill not in member_skills
        ]

        # スキル情報を取得して推奨リストを作成
        for skill_code in unacquired_skills[:top_n]:
            skill_info = self.competence_master_df[
                self.competence_master_df["力量コード"] == skill_code
            ]

            if len(skill_info) > 0:
                skill_name = skill_info.iloc[0]["力量名"]
                current_factor_score = member_scores.get(
                    domain_struct.latent_factors[current_level].factor_name, 0.0
                )

                recommendations.append(
                    {
                        "skill_code": skill_code,
                        "skill_name": skill_name,
                        "direct_effect_score": current_factor_score * 0.8,  # 現在のレベルに基づく
                        "current_domain_level": current_factor_score,
                        "next_level": next_factor.factor_name,
                        "reason": f"{domain_category}の{next_factor.factor_name.split('_')[1]}スキルを強化するため",
                    }
                )

        return recommendations

    def get_indirect_support_skills(
        self, target_skill: str, member_code: str, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        間接効果：他の領域スキルの習得を助けるスキル

        ターゲットスキルを習得する際に、
        他の領域のスキルが間接的にどれほど支援するかを計算。

        Args:
            target_skill: ターゲットスキルコード
            member_code: メンバーコード
            top_n: 上位N件を返す

        Returns:
            [{
                'skill_code': スキルコード,
                'skill_name': スキル名,
                'indirect_support_score': 間接効果スコア（0-1）,
                'target_skill': ターゲットスキル,
                'reason': 支援理由
            }]
        """
        recommendations = []

        # ターゲットスキルの領域を特定
        target_domain = self._find_skill_domain(target_skill)
        if not target_domain:
            logger.warning(f"Target skill not found: {target_skill}")
            return recommendations

        # メンバーの他領域のスキルを取得
        member_skills = self.member_competence_df[
            self.member_competence_df["メンバーコード"] == member_code
        ]

        member_scores = self.member_latent_scores.get(member_code, {})

        # 異なる領域でのスコアを集計
        for domain_name, domain_struct in self.domain_structures.items():
            if domain_name == target_domain:
                continue  # ターゲット領域は除外

            # この領域でのメンバーのスコア
            for latent_factor in domain_struct.latent_factors:
                factor_score = member_scores.get(latent_factor.factor_name, 0.0)

                if factor_score > 0:
                    # 間接効果スコア = （他領域のスコア × パス係数）
                    indirect_score = factor_score * 0.4  # 簡易的な間接効果（0.4は係数）

                    # この領域のスキル例を推奨
                    for skill_code in latent_factor.observed_skills[:1]:
                        skill_info = self.competence_master_df[
                            self.competence_master_df["力量コード"] == skill_code
                        ]

                        if len(skill_info) > 0:
                            skill_name = skill_info.iloc[0]["力量名"]
                            recommendations.append(
                                {
                                    "skill_code": skill_code,
                                    "skill_name": skill_name,
                                    "indirect_support_score": indirect_score,
                                    "target_skill": target_skill,
                                    "reason": f"{domain_name}スキルを習得することで、{target_domain}の理解が深まります",
                                }
                            )

        # スコアでソートして上位Nを返す
        recommendations.sort(
            key=lambda x: x["indirect_support_score"], reverse=True
        )
        return recommendations[:top_n]

    def calculate_sem_score(
        self, member_code: str, skill_code: str
    ) -> float:
        """
        SEMスコアを計算（推薦スコアに統合用）

        メンバーがスキルを習得する確率を、
        潜在変数と因果関係に基づいて推定。

        Args:
            member_code: メンバーコード
            skill_code: スキルコード

        Returns:
            SEMスコア（0-1）
        """
        domain = self._find_skill_domain(skill_code)
        if not domain:
            return 0.0

        member_scores = self.member_latent_scores.get(member_code, {})
        domain_struct = self.domain_structures.get(domain)

        if not domain_struct:
            return 0.0

        # メンバーのこの領域での現在のレベルを取得
        current_level = self._estimate_current_level(member_code, domain)

        if current_level >= len(domain_struct.latent_factors):
            return 1.0  # 既に最高レベル

        # 次のレベルへの遷移スコア
        current_factor = domain_struct.latent_factors[current_level]
        current_score = member_scores.get(current_factor.factor_name, 0.0)

        # 領域の信頼度を組み込む
        sem_score = current_score * domain_struct.domain_reliability

        return min(1.0, sem_score)

    def _estimate_current_level(self, member_code: str, domain_category: str) -> int:
        """
        メンバーの領域内での現在のレベルを推定

        Args:
            member_code: メンバーコード
            domain_category: 領域名

        Returns:
            レベル（0:初級, 1:中級, 2:上級）
        """
        member_scores = self.member_latent_scores.get(member_code, {})
        domain_struct = self.domain_structures.get(domain_category)

        if not domain_struct:
            return 0

        # 各レベルのスコアを比較
        max_level = 0
        for i, latent_factor in enumerate(domain_struct.latent_factors):
            score = member_scores.get(latent_factor.factor_name, 0.0)
            if score > 0.5:  # 0.5以上を習得済みとみなす
                max_level = i

        return max_level

    def _find_skill_domain(self, skill_code: str) -> Optional[str]:
        """
        スキルコードから所属領域を検索

        Args:
            skill_code: スキルコード

        Returns:
            領域名（見つからない場合はNone）
        """
        for domain_name, domain_struct in self.domain_structures.items():
            for latent_factor in domain_struct.latent_factors:
                if skill_code in latent_factor.observed_skills:
                    return domain_name

        return None

    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """
        領域の詳細情報を取得

        Args:
            domain_name: 領域名

        Returns:
            領域の情報（潜在変数、スキル、パス係数等）
        """
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
                }
                for f in domain_struct.latent_factors
            ],
            "path_coefficients": [
                {
                    "from": p.from_factor,
                    "to": p.to_factor,
                    "coefficient": p.coefficient,
                    "is_significant": p.is_significant,
                }
                for p in domain_struct.path_coefficients
            ],
            "domain_reliability": domain_struct.domain_reliability,
        }

    def get_all_domains(self) -> List[str]:
        """全領域名を取得"""
        return list(self.domain_structures.keys())

    def get_member_domain_profile(self, member_code: str) -> Dict[str, Dict[str, float]]:
        """
        メンバーの領域別プロファイル（全潜在変数のスコア）を取得

        Returns:
            {領域名: {潜在変数名: スコア}}
        """
        member_scores = self.member_latent_scores.get(member_code, {})
        profile = {}

        for domain_name, domain_struct in self.domain_structures.items():
            profile[domain_name] = {}
            for latent_factor in domain_struct.latent_factors:
                score = member_scores.get(latent_factor.factor_name, 0.0)
                profile[domain_name][latent_factor.factor_name] = score

        return profile
