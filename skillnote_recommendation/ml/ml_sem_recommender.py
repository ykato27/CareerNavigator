"""
機械学習ベース推薦エンジン with SEM拡張

MLRecommenderにスキル領域潜在変数SEMモデルを統合し、
直接効果・間接効果のスコアを推薦スコアに追加します。
"""

import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel
from skillnote_recommendation.core.models import Recommendation

logger = logging.getLogger(__name__)


class MLSEMRecommender(MLRecommender):
    """
    SEMモデルを統合したML推薦エンジン

    既存のMLRecommenderの機能を拡張し、
    スキル領域潜在変数モデルを組み込んで、
    より説明可能な推薦を実現します。
    """

    def __init__(
        self,
        mf_model,
        competence_master: pd.DataFrame,
        member_competence: pd.DataFrame,
        member_master: pd.DataFrame,
        sem_model: Optional[SkillDomainSEMModel] = None,
        diversity_reranker=None,
        reference_person_finder=None,
        tuning_results: Optional[dict] = None,
        sem_weight: float = 0.2,
    ):
        """
        初期化

        Args:
            mf_model: MatrixFactorizationModel
            competence_master: 力量マスタ
            member_competence: メンバー習得力量
            member_master: メンバーマスタ
            sem_model: SkillDomainSEMModel（Noneの場合は自動作成）
            diversity_reranker: 多様性再ランキング
            reference_person_finder: 参考人物検索
            tuning_results: ハイパーパラメータチューニング結果
            sem_weight: SEM スコアの重み（0-1、他の方法の合計重みは1-sem_weight）
        """
        super().__init__(
            mf_model=mf_model,
            competence_master=competence_master,
            member_competence=member_competence,
            member_master=member_master,
            diversity_reranker=diversity_reranker,
            reference_person_finder=reference_person_finder,
            tuning_results=tuning_results,
        )

        self.sem_weight = sem_weight
        self.sem_model = sem_model or self._initialize_sem_model()

        logger.info(f"MLSEMRecommender initialized with SEM weight={sem_weight}")

    def _initialize_sem_model(self) -> SkillDomainSEMModel:
        """
        SEMモデルを初期化

        Returns:
            SkillDomainSEMModel: 初期化されたSEMモデル
        """
        logger.info("Initializing SkillDomainSEMModel...")
        sem_model = SkillDomainSEMModel(
            member_competence_df=self.member_competence,
            competence_master_df=self.competence_master,
            num_domain_categories=8,  # デフォルト: 8領域
        )
        logger.info(f"SEMModel initialized with {len(sem_model.get_all_domains())} domains")
        return sem_model

    @classmethod
    def build(
        cls,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_master: pd.DataFrame,
        use_preprocessing: bool = True,
        use_tuning: bool = False,
        n_components: Optional[int] = None,
        tuning_n_trials: Optional[int] = None,
        tuning_timeout: Optional[int] = None,
        tuning_search_space: Optional[Dict] = None,
        tuning_sampler: Optional[str] = None,
        tuning_random_state: Optional[int] = None,
        tuning_progress_callback: Optional[object] = None,
        use_sem: bool = True,
        sem_weight: float = 0.2,
        num_domain_categories: int = 8,
    ):
        """
        MLSEMRecommenderを構築

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ
            member_master: メンバーマスタ
            use_preprocessing: データ前処理を使用
            use_tuning: ハイパーパラメータチューニングを使用
            n_components: 潜在因子数
            tuning_n_trials: チューニング試行回数
            tuning_timeout: チューニングタイムアウト
            tuning_search_space: チューニング探索空間
            tuning_sampler: チューニングサンプラー
            tuning_random_state: チューニング乱数シード
            tuning_progress_callback: チューニング進捗コールバック
            use_sem: SEMモデルを使用
            sem_weight: SEMスコアの重み
            num_domain_categories: スキル領域の分類数

        Returns:
            MLSEMRecommender: 構築されたレコメンダー
        """
        # 親クラスのbuild()メソッドを呼び出してMLRecommenderを構築
        ml_recommender = MLRecommender.build(
            member_competence=member_competence,
            competence_master=competence_master,
            member_master=member_master,
            use_preprocessing=use_preprocessing,
            use_tuning=use_tuning,
            n_components=n_components,
            tuning_n_trials=tuning_n_trials,
            tuning_timeout=tuning_timeout,
            tuning_search_space=tuning_search_space,
            tuning_sampler=tuning_sampler,
            tuning_random_state=tuning_random_state,
            tuning_progress_callback=tuning_progress_callback,
        )

        # SEMモデルを初期化
        sem_model = None
        if use_sem:
            logger.info("Building SkillDomainSEMModel...")
            sem_model = SkillDomainSEMModel(
                member_competence_df=member_competence,
                competence_master_df=competence_master,
                num_domain_categories=num_domain_categories,
            )

        # MLSEMRecommenderを構築
        return cls(
            mf_model=ml_recommender.mf_model,
            competence_master=competence_master,
            member_competence=member_competence,
            member_master=member_master,
            sem_model=sem_model,
            diversity_reranker=ml_recommender.diversity_reranker,
            reference_person_finder=ml_recommender.reference_person_finder,
            tuning_results=ml_recommender.tuning_results,
            sem_weight=sem_weight,
        )

    def recommend(
        self,
        member_code: str,
        top_n: int = 10,
        exclude_categories: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
        use_sem: bool = True,
        return_explanation: bool = False,
    ) -> List[Recommendation]:
        """
        推薦を実施（SEM統合版）

        Args:
            member_code: メンバーコード
            top_n: 上位N件を返す
            exclude_categories: 除外するカテゴリーリスト
            exclude_types: 除外する力量タイプリスト（SKILL, EDUCATION, LICENSE）
            use_sem: SEMスコアを使用
            return_explanation: 説明情報を含める

        Returns:
            Recommendationオブジェクトのリスト
        """
        # 親クラスの推薦を取得
        base_recommendations = super().recommend(
            member_code=member_code,
            top_n=top_n,
            exclude_categories=exclude_categories,
            exclude_types=exclude_types,
        )

        if not use_sem or not self.sem_model:
            return base_recommendations

        # SEMスコアを追加して再スコアリング
        return self._apply_sem_scoring(
            member_code=member_code,
            base_recommendations=base_recommendations,
            top_n=top_n,
            return_explanation=return_explanation,
        )

    def _apply_sem_scoring(
        self,
        member_code: str,
        base_recommendations: List[Recommendation],
        top_n: int,
        return_explanation: bool = False,
    ) -> List[Recommendation]:
        """
        SEMスコアを適用して推薦を再スコアリング

        Args:
            member_code: メンバーコード
            base_recommendations: ベースの推薦リスト
            top_n: 上位N件
            return_explanation: 説明情報を含める

        Returns:
            SEMスコアで再スコアリングされた推薦リスト
        """
        updated_recommendations = []

        for rec in base_recommendations:
            # SEMスコアを計算
            sem_score = self.sem_model.calculate_sem_score(
                member_code=member_code,
                skill_code=rec.competence_code,
            )

            # スコアを統合：基本スコアを保持しつつSEMスコアを反映
            # 調整後のスコア = 基本スコア + (SEM スコア × SEM重み)
            original_score = rec.priority_score
            adjusted_score = original_score * (1 - self.sem_weight) + sem_score * self.sem_weight

            # 説明文にSEM情報を追加
            explanation = rec.reason
            if return_explanation:
                explanation = self._generate_sem_explanation(
                    member_code=member_code,
                    competence_code=rec.competence_code,
                    sem_score=sem_score,
                    base_explanation=rec.reason,
                )

            # 新しいRecommendationオブジェクトを作成
            updated_rec = Recommendation(
                competence_code=rec.competence_code,
                competence_name=rec.competence_name,
                priority_score=adjusted_score,
                reason=explanation,
                reference_persons=rec.reference_persons,
            )
            updated_recommendations.append(updated_rec)

        # スコアでソート
        updated_recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        return updated_recommendations[:top_n]

    def _generate_sem_explanation(
        self,
        member_code: str,
        competence_code: str,
        sem_score: float,
        base_explanation: str,
    ) -> str:
        """
        SEMモデルに基づいた説明文を生成

        Args:
            member_code: メンバーコード
            competence_code: スキルコード
            sem_score: SEMスコア
            base_explanation: ベースの説明文

        Returns:
            拡張された説明文
        """
        domain = self.sem_model._find_skill_domain(competence_code)
        if not domain:
            return base_explanation

        # 領域情報を取得
        domain_info = self.sem_model.get_domain_info(domain)
        member_profile = self.sem_model.get_member_domain_profile(member_code)

        domain_scores = member_profile.get(domain, {})
        current_domain_level = sum(domain_scores.values()) / len(
            domain_scores
        ) if domain_scores else 0

        # スキル情報を取得
        skill_info = self.competence_master[
            self.competence_master["力量コード"] == competence_code
        ]
        skill_name = skill_info.iloc[0]["力量名"] if len(skill_info) > 0 else competence_code

        # 説明文を生成
        explanation_parts = [base_explanation]

        if current_domain_level > 0:
            level_percentage = int(current_domain_level * 100)
            explanation_parts.append(
                f"\n【SEM分析】{domain}領域の習得度が{level_percentage}%のため、"
                f"次のステップとして{skill_name}の習得をお勧めします。"
            )
        else:
            explanation_parts.append(
                f"\n【SEM分析】{domain}領域の基礎を構築するために、"
                f"{skill_name}の習得をお勧めします。"
            )

        return "".join(explanation_parts)

    def get_direct_effect_recommendations(
        self,
        member_code: str,
        domain_category: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        直接効果に基づく推薦を取得

        特定領域で潜在変数を獲得したときの、
        その領域内の次レベルスキルの推薦。

        Args:
            member_code: メンバーコード
            domain_category: 領域名
            top_n: 上位N件

        Returns:
            推薦リスト
        """
        if not self.sem_model:
            return []

        return self.sem_model.get_direct_effect_skills(
            member_code=member_code,
            domain_category=domain_category,
            top_n=top_n,
        )

    def get_indirect_support_recommendations(
        self,
        member_code: str,
        target_skill: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        間接効果に基づく推薦を取得

        ターゲットスキル習得を支援する、
        他領域のスキル推薦。

        Args:
            member_code: メンバーコード
            target_skill: ターゲットスキルコード
            top_n: 上位N件

        Returns:
            推薦リスト
        """
        if not self.sem_model:
            return []

        return self.sem_model.get_indirect_support_skills(
            member_code=member_code,
            target_skill=target_skill,
            top_n=top_n,
        )

    def get_member_domain_profile(
        self, member_code: str
    ) -> Dict[str, Dict[str, float]]:
        """
        メンバーの領域別プロファイルを取得

        全領域の潜在変数スコアを取得。

        Args:
            member_code: メンバーコード

        Returns:
            {領域名: {潜在変数名: スコア}}
        """
        if not self.sem_model:
            return {}

        return self.sem_model.get_member_domain_profile(member_code)

    def get_all_domains(self) -> List[str]:
        """全スキル領域を取得"""
        if not self.sem_model:
            return []
        return self.sem_model.get_all_domains()
