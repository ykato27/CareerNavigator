"""
スキル領域SEMモデル

スキル領域の階層構造（初級→中級→上級）に基づいたSEMモデルを構築し、
メンバーの潜在的なスキルレベルを推定、推薦に活用します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from skillnote_recommendation.ml.skill_domain_hierarchy import SkillDomainHierarchy
from skillnote_recommendation.ml.unified_sem_estimator import (
    UnifiedSEMEstimator,
    MeasurementModelSpec,
    StructuralModelSpec,
)

logger = logging.getLogger(__name__)


class SkillDomainSEMModel:
    """
    スキル領域の階層的SEMモデル

    各ドメイン（プログラミング、データベースなど）について、
    初級→中級→上級の潜在変数を推定し、スキル推薦に活用します。

    例:
        プログラミング領域:
            初級プログラミング（潜在変数） → [Python基礎, Java基礎, Git]
                ↓
            中級プログラミング（潜在変数） → [Web開発, API開発, テスト]
                ↓
            上級プログラミング（潜在変数） → [システム設計, アーキテクチャ]
    """

    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        domain_hierarchy: Optional[SkillDomainHierarchy] = None,
    ):
        """
        Args:
            member_competence: メンバー力量データ（メンバーコード、力量コード、正規化レベル）
            competence_master: 力量マスタ（力量コード、力量名）
            domain_hierarchy: スキル領域階層（Noneの場合は自動生成）
        """
        self.member_competence = member_competence
        self.competence_master = competence_master

        # ドメイン階層を構築
        if domain_hierarchy is None:
            self.domain_hierarchy = SkillDomainHierarchy(competence_master)
        else:
            self.domain_hierarchy = domain_hierarchy

        # SEMモデル（ドメインごと）
        self.sem_models: Dict[str, UnifiedSEMEstimator] = {}

        # 潜在変数スコア（メンバーごと）
        self.latent_scores: Dict[str, Dict[str, Dict[int, float]]] = {}

        self.is_fitted = False

    def fit(self, domains: Optional[List[str]] = None, min_competences_per_level: int = 3):
        """
        各ドメインのSEMモデルを学習

        Args:
            domains: 学習するドメインリスト（Noneの場合は全ドメイン）
            min_competences_per_level: 各レベルで最低限必要な力量数（デフォルト3）
        """
        if domains is None:
            domains = list(self.domain_hierarchy.hierarchy.keys())

        logger.info("=" * 60)
        logger.info("スキル領域SEMモデルの学習開始")
        logger.info("=" * 60)

        for domain in domains:
            logger.info(f"\n【{domain}】ドメインの学習中...")

            try:
                # ドメインのSEMモデルを構築・学習
                sem_model = self._fit_domain_sem(domain, min_competences_per_level)

                if sem_model is not None:
                    self.sem_models[domain] = sem_model
                    logger.info(f"✅ {domain}のSEMモデル学習完了")
                else:
                    logger.warning(f"⚠️ {domain}のSEMモデル学習スキップ（データ不足）")

            except Exception as e:
                logger.error(f"❌ {domain}のSEMモデル学習失敗: {e}")

        # 潜在変数スコアを推定
        self._estimate_latent_scores()

        self.is_fitted = True

        logger.info("\n" + "=" * 60)
        logger.info(f"✅ スキル領域SEMモデルの学習完了（{len(self.sem_models)}ドメイン）")
        logger.info("=" * 60)

    def _fit_domain_sem(
        self,
        domain: str,
        min_competences_per_level: int
    ) -> Optional[UnifiedSEMEstimator]:
        """
        特定のドメインのSEMモデルを構築・学習

        Args:
            domain: ドメイン名
            min_competences_per_level: 各レベルで最低限必要な力量数

        Returns:
            学習済みSEMモデル（データ不足の場合はNone）
        """
        # ドメインの力量を取得
        level_1_competences = self.domain_hierarchy.get_competences_by_level(domain, 1)
        level_2_competences = self.domain_hierarchy.get_competences_by_level(domain, 2)
        level_3_competences = self.domain_hierarchy.get_competences_by_level(domain, 3)

        logger.info(f"  Level 1: {len(level_1_competences)}個")
        logger.info(f"  Level 2: {len(level_2_competences)}個")
        logger.info(f"  Level 3: {len(level_3_competences)}個")

        # 各レベルで最低限の力量数があるか確認
        if (len(level_1_competences) < min_competences_per_level or
            len(level_2_competences) < min_competences_per_level):
            logger.warning(f"  ⚠️ データ不足: Level 1またはLevel 2の力量が{min_competences_per_level}個未満")
            return None

        # データを準備
        all_competences = level_1_competences + level_2_competences + level_3_competences

        # メンバー×力量のデータフレームを作成
        member_skill_data = self.member_competence[
            self.member_competence['力量コード'].isin(all_competences)
        ].copy()

        if len(member_skill_data) < 10:
            logger.warning(f"  ⚠️ データ不足: 習得記録が10件未満")
            return None

        # ピボットテーブル化（メンバー×力量）
        skill_matrix = member_skill_data.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0.0
        )

        # 力量コードから力量名にマッピング
        competence_name_map = dict(zip(
            self.competence_master['力量コード'],
            self.competence_master['力量名']
        ))

        # 測定モデルを定義
        measurement_models = []

        # Level 1: 初級
        if len(level_1_competences) >= 2:
            level_1_names = [competence_name_map.get(c, c) for c in level_1_competences if c in skill_matrix.columns]
            if len(level_1_names) >= 2:
                measurement_models.append(
                    MeasurementModelSpec(
                        latent_variable=f'{domain}_初級',
                        indicators=level_1_names,
                        reference_indicator=level_1_names[0]
                    )
                )

        # Level 2: 中級
        if len(level_2_competences) >= 2:
            level_2_names = [competence_name_map.get(c, c) for c in level_2_competences if c in skill_matrix.columns]
            if len(level_2_names) >= 2:
                measurement_models.append(
                    MeasurementModelSpec(
                        latent_variable=f'{domain}_中級',
                        indicators=level_2_names,
                        reference_indicator=level_2_names[0]
                    )
                )

        # Level 3: 上級
        if len(level_3_competences) >= 2:
            level_3_names = [competence_name_map.get(c, c) for c in level_3_competences if c in skill_matrix.columns]
            if len(level_3_names) >= 2:
                measurement_models.append(
                    MeasurementModelSpec(
                        latent_variable=f'{domain}_上級',
                        indicators=level_3_names,
                        reference_indicator=level_3_names[0]
                    )
                )

        if len(measurement_models) < 2:
            logger.warning(f"  ⚠️ 測定モデルが2個未満（{len(measurement_models)}個）")
            return None

        # 構造モデルを定義（初級→中級→上級の階層）
        structural_models = []

        if len(measurement_models) >= 2:
            structural_models.append(
                StructuralModelSpec(
                    from_variable=f'{domain}_初級',
                    to_variable=f'{domain}_中級'
                )
            )

        if len(measurement_models) >= 3:
            structural_models.append(
                StructuralModelSpec(
                    from_variable=f'{domain}_中級',
                    to_variable=f'{domain}_上級'
                )
            )

        # 力量名でデータフレームを作成
        renamed_columns = {code: competence_name_map.get(code, code) for code in skill_matrix.columns}
        skill_matrix_renamed = skill_matrix.rename(columns=renamed_columns)

        # SEMモデルを学習
        try:
            sem_model = UnifiedSEMEstimator(
                measurement_model=measurement_models,
                structural_model=structural_models
            )

            sem_model.fit(skill_matrix_renamed)

            logger.info(f"  ✅ SEMモデル学習完了（適合度: {sem_model.fit_info.get('gfi', 'N/A'):.3f if isinstance(sem_model.fit_info.get('gfi'), (int, float)) else 'N/A'}）")

            return sem_model

        except Exception as e:
            logger.error(f"  ❌ SEMモデル学習エラー: {e}")
            return None

    def _estimate_latent_scores(self):
        """
        各メンバーの潜在変数スコアを推定

        self.latent_scores = {
            'M001': {
                'プログラミング': {1: 0.75, 2: 0.45, 3: 0.15},
                'データベース': {1: 0.60, 2: 0.30, 3: 0.0},
            },
            ...
        }
        """
        logger.info("\n潜在変数スコアの推定中...")

        # メンバーごとに初期化
        member_codes = self.member_competence['メンバーコード'].unique()
        for member_code in member_codes:
            self.latent_scores[member_code] = {}

        # ドメインごとにスコアを推定
        for domain, sem_model in self.sem_models.items():
            if not sem_model.is_fitted:
                continue

            # メンバーの習得力量からスコアを推定
            for member_code in member_codes:
                member_skills = self.member_competence[
                    self.member_competence['メンバーコード'] == member_code
                ]

                # 各レベルのスコアを計算
                level_scores = {}

                for level in [1, 2, 3]:
                    level_competences = self.domain_hierarchy.get_competences_by_level(domain, level)

                    # メンバーが習得している力量を取得
                    acquired = member_skills[
                        member_skills['力量コード'].isin(level_competences)
                    ]

                    if len(acquired) > 0:
                        # 平均正規化レベルをスコアとする
                        level_scores[level] = acquired['正規化レベル'].mean()
                    else:
                        level_scores[level] = 0.0

                self.latent_scores[member_code][domain] = level_scores

        logger.info(f"✅ {len(member_codes)}名の潜在変数スコア推定完了")

    def get_member_latent_score(
        self,
        member_code: str,
        domain: str,
        level: int
    ) -> float:
        """
        メンバーの特定ドメイン・レベルの潜在変数スコアを取得

        Args:
            member_code: メンバーコード
            domain: ドメイン名
            level: レベル（1=初級, 2=中級, 3=上級）

        Returns:
            潜在変数スコア（0.0～1.0）
        """
        if member_code not in self.latent_scores:
            return 0.0

        if domain not in self.latent_scores[member_code]:
            return 0.0

        return self.latent_scores[member_code][domain].get(level, 0.0)

    def recommend_next_skills(
        self,
        member_code: str,
        top_n: int = 5,
        min_current_level_score: float = 0.6
    ) -> List[Dict]:
        """
        メンバーの次に習得すべきスキルを推薦

        推薦ロジック:
        1. 各ドメインで現在のレベルを判定（潜在変数スコア >= min_current_level_score）
        2. 次のレベルのスキルを推薦
        3. 推薦理由を生成

        Args:
            member_code: メンバーコード
            top_n: 推薦数
            min_current_level_score: 現在のレベルと判定する最小スコア

        Returns:
            [{
                'competence_code': str,
                'competence_name': str,
                'domain': str,
                'level': int,
                'score': float,
                'reason': str,
            }]
        """
        recommendations = []

        if member_code not in self.latent_scores:
            logger.warning(f"メンバー {member_code} の潜在変数スコアがありません")
            return []

        member_scores = self.latent_scores[member_code]

        # 既習得力量
        acquired_competences = set(
            self.member_competence[
                self.member_competence['メンバーコード'] == member_code
            ]['力量コード']
        )

        for domain, level_scores in member_scores.items():
            # 現在のレベルを判定
            current_level = 0

            for level in [1, 2, 3]:
                if level_scores.get(level, 0.0) >= min_current_level_score:
                    current_level = level

            # 次のレベルを推薦
            if current_level < 3:
                next_level = current_level + 1

                next_level_competences = self.domain_hierarchy.get_competences_by_level(
                    domain, next_level
                )

                for comp_code in next_level_competences:
                    # 既習得は除外
                    if comp_code in acquired_competences:
                        continue

                    comp_info = self.competence_master[
                        self.competence_master['力量コード'] == comp_code
                    ]

                    if len(comp_info) == 0:
                        continue

                    comp_name = comp_info.iloc[0]['力量名']

                    # 推薦スコア = 現在のレベルのスコア
                    rec_score = level_scores.get(current_level, 0.0)

                    # 推薦理由
                    level_name_map = {1: '初級', 2: '中級', 3: '上級'}
                    current_level_name = level_name_map.get(current_level, '基礎')
                    next_level_name = level_name_map.get(next_level, '応用')

                    reason = (
                        f"{domain}領域の{current_level_name}スキル（スコア{rec_score:.2f}）を習得済み。"
                        f"次のステップとして{next_level_name}スキル「{comp_name}」をおすすめします。"
                    )

                    recommendations.append({
                        'competence_code': comp_code,
                        'competence_name': comp_name,
                        'domain': domain,
                        'level': next_level,
                        'score': rec_score,
                        'reason': reason,
                    })

        # スコア降順でソート
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations[:top_n]

    def get_member_skill_profile(self, member_code: str) -> pd.DataFrame:
        """
        メンバーのスキルプロファイルを取得

        Returns:
            DataFrame with columns: [Domain, Level_1_Score, Level_2_Score, Level_3_Score]
        """
        if member_code not in self.latent_scores:
            return pd.DataFrame()

        member_scores = self.latent_scores[member_code]

        profile_data = []

        for domain, level_scores in member_scores.items():
            profile_data.append({
                'Domain': domain,
                'Level_1_Score': level_scores.get(1, 0.0),
                'Level_2_Score': level_scores.get(2, 0.0),
                'Level_3_Score': level_scores.get(3, 0.0),
            })

        return pd.DataFrame(profile_data)
