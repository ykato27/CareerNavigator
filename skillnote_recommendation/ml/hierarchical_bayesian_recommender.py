"""
階層的ベイジアン推薦システム（統合モジュール）

3層すべて（L1ベイジアンネットワーク、L2条件付き確率、L3行列分解）を統合し、
最終的なスキル推薦を生成します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from skillnote_recommendation.ml.base_recommender import BaseRecommender
from skillnote_recommendation.ml.category_hierarchy_extractor import (
    CategoryHierarchyExtractor,
    CategoryHierarchy
)
from skillnote_recommendation.ml.category_network_learner import CategoryNetworkLearner
from skillnote_recommendation.ml.conditional_probability_learner import ConditionalProbabilityLearner
from skillnote_recommendation.ml.category_wise_mf import CategoryWiseMatrixFactorization

logger = logging.getLogger(__name__)


class HierarchicalBayesianRecommender(BaseRecommender):
    """
    階層的ベイジアン推薦システム
    
    3層アーキテクチャを統合してスキル推薦を生成：
    - Layer 1: 大カテゴリのベイジアンネットワーク
    - Layer 2: P(中カテゴリ | 大カテゴリ)の条件付き確率
    - Layer 3: カテゴリ別行列分解
    """
    
    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        category_csv_path: str,
        skill_csv_path: str,
        max_indegree: int = 3,
        n_components: int = 10
    ):
        """
        初期化
        
        Args:
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            category_csv_path: カテゴリマスタCSVのパス
            skill_csv_path: スキルマスタCSVのパス
            max_indegree: ベイジアンネットワークの最大入次数
            n_components: 行列分解の潜在因子数
        """
        super().__init__(member_competence, competence_master)
        
        self.category_csv_path = category_csv_path
        self.skill_csv_path = skill_csv_path
        self.max_indegree = max_indegree
        self.n_components = n_components
        
        # 各レイヤーのコンポーネント
        self.hierarchy_extractor: Optional[CategoryHierarchyExtractor] = None
        self.hierarchy: Optional[CategoryHierarchy] = None
        self.network_learner: Optional[CategoryNetworkLearner] = None
        self.prob_learner: Optional[ConditionalProbabilityLearner] = None
        self.mf_learner: Optional[CategoryWiseMatrixFactorization] = None
        
        # スコア統合の重み
        self.weight_l1 = 0.3
        self.weight_l2 = 0.3
        self.weight_l3 = 0.4
    
    def fit(self, min_members_per_skill: int = 5):
        """
        モデルを学習
        
        Args:
            min_members_per_skill: 学習に含めるスキルの最小保持人数
            
        Returns:
            self
        """
        logger.info("階層的ベイジアン推薦システムの学習を開始")
        
        # 1. カテゴリ階層を抽出
        logger.info("Phase 1: カテゴリ階層を抽出中")
        self.hierarchy_extractor = CategoryHierarchyExtractor(
            self.category_csv_path,
            self.skill_csv_path
        )
        self.hierarchy = self.hierarchy_extractor.extract_hierarchy()
        
        # 2. ユーザー×スキルマトリクスを準備
        user_skills = self._prepare_user_skill_matrix()
        
        # 3. Layer 1: ベイジアンネットワークを学習
        logger.info("Phase 2: Layer 1 ベイジアンネットワークを学習中")
        self.network_learner = CategoryNetworkLearner(
            max_indegree=self.max_indegree
        )
        try:
            self.network_learner.fit(
                user_skills,
                self.hierarchy,
                aggregation_method='max',
                n_bins=3
            )
            logger.info("Layer 1 学習完了")
        except Exception as e:
            logger.warning(f"Layer 1 学習エラー: {e}")
            self.network_learner = None
        
        # 4. Layer 2: 条件付き確率を学習
        logger.info("Phase 3: Layer 2 条件付き確率を学習中")
        self.prob_learner = ConditionalProbabilityLearner()
        self.prob_learner.learn_conditional_probs(self.hierarchy, user_skills)
        logger.info("Layer 2 学習完了")
        
        # 5. Layer 3: カテゴリ別行列分解を学習
        logger.info("Phase 4: Layer 3 カテゴリ別行列分解を学習中")
        self.mf_learner = CategoryWiseMatrixFactorization(
            n_components=self.n_components
        )
        self.mf_learner.fit_category_models(user_skills, self.hierarchy)
        logger.info("Layer 3 学習完了")
        
        logger.info("階層的ベイジアン推薦システムの学習完了")
        return self
    
    def _prepare_user_skill_matrix(self) -> pd.DataFrame:
        """
        ユーザー×スキルマトリクスを準備
        
        Returns:
            ユーザー×スキルのDataFrame
        """
        # member_competenceからSKILLタイプのみを抽出
        skill_data = self.member_competence[
            self.member_competence['力量タイプ  ###[competence_type]###'] == 'SKILL'
        ].copy()
        
        # ピボットしてユーザー×スキルマトリクスを作成
        user_skill_matrix = skill_data.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='レベル',
            fill_value=0
        )
        
        return user_skill_matrix
    
    def recommend(
        self,
        member_code: str,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        メンバーへのスキル推薦
        
        Args:
            member_code: メンバーID
            top_n: 推薦件数
            
        Returns:
            推薦結果のリスト
        """
        if self.hierarchy is None:
            raise ValueError("先にfit()を実行してください")
        
        logger.info(f"メンバー {member_code} への推薦を生成中")
        
        # ユーザーの保有スキルを取得
        user_skills = self._get_user_skills(member_code)
        
        # 候補スキルを取得（保有していないスキル）
        all_skills = set(self.hierarchy.skill_to_category.keys())
        owned_skills = set(user_skills.keys())
        candidate_skills = list(all_skills - owned_skills)
        
        # 各候補スキルのスコアを計算
        recommendations = []
        for skill_code in candidate_skills:
            score, explanation = self._calculate_skill_score(
                member_code,
                skill_code,
                user_skills
            )
            
            if score > 0:
                # スキル情報を取得
                skill_info = self._get_skill_info(skill_code)
                
                recommendations.append({
                    '力量コード': skill_code,
                    '力量名': skill_info.get('力量名', skill_code),
                    'スコア': score,
                    '説明': explanation,
                    'カテゴリ': skill_info.get('カテゴリ', '')
                })
        
        # スコアでソート
        recommendations.sort(key=lambda x: x['スコア'], reverse=True)
        
        logger.info(f"{len(recommendations)}件の候補から上位{top_n}件を推薦")
        
        return recommendations[:top_n]
    
    def _get_user_skills(self, member_code: str) -> Dict[str, float]:
        """
        ユーザーの保有スキルとレベルを取得
        
        Args:
            member_code: メンバーコード
            
        Returns:
            スキルコードとレベルの辞書
        """
        user_data = self.member_competence[
            (self.member_competence['メンバーコード'] == member_code) &
            (self.member_competence['力量タイプ  ###[competence_type]###'] == 'SKILL')
        ]
        
        return dict(zip(user_data['力量コード'], user_data['レベル']))
    
    def _calculate_skill_score(
        self,
        member_code: str,
        skill_code: str,
        user_skills: Dict[str, float]
    ) -> tuple[float, str]:
        """
        スキルの推薦スコアを計算
        
        Args:
            member_code: メンバーコード
            skill_code: スキルコード
            user_skills: ユーザーの保有スキル
            
        Returns:
            (スコア, 説明文)のタプル
        """
        # スキルのカテゴリを取得
        if skill_code not in self.hierarchy.skill_to_category:
            return 0.0, ""
        
        category_code = self.hierarchy.skill_to_category[skill_code]
        
        # L1, L2カテゴリを取得
        l1_code = self.hierarchy.get_l1_category(category_code)
        l2_code = self.hierarchy.get_l2_category(category_code)
        
        if not l1_code or not l2_code:
            return 0.0, ""
        
        # Layer 1: L1カテゴリの準備度
        l1_readiness = self._get_l1_readiness(l1_code, user_skills)
        
        # Layer 2: P(L2 | L1)
        l2_prob = self.prob_learner.get_conditional_prob(l1_code, l2_code)
        
        # Layer 3: スキルのMFスコア
        l3_score = self._get_l3_score(member_code, l2_code, skill_code)
        
        # スコアを統合（乗算的アプローチ）
        final_score = (
            (l1_readiness ** self.weight_l1) *
            (l2_prob ** self.weight_l2) *
            (l3_score ** self.weight_l3)
        )
        
        # 説明文を生成
        explanation = self._generate_explanation(
            l1_code,
            l2_code,
            l1_readiness,
            l2_prob,
            l3_score
        )
        
        return final_score, explanation
    
    def _get_l1_readiness(
        self,
        l1_code: str,
        user_skills: Dict[str, float]
    ) -> float:
        """
        L1カテゴリの準備度を取得
        
        Args:
            l1_code: L1カテゴリコード
            user_skills: ユーザーの保有スキル
            
        Returns:
            準備度スコア（0.0-1.0）
        """
        if self.network_learner is None or self.network_learner.model is None:
            # ベイジアンネットワークが利用できない場合は簡易計算
            return self._simple_l1_readiness(l1_code, user_skills)
        
        # ユーザーのカテゴリレベルを計算
        user_categories = {}
        for skill_code, level in user_skills.items():
            if skill_code in self.hierarchy.skill_to_category:
                cat = self.hierarchy.skill_to_category[skill_code]
                cat_l1 = self.hierarchy.get_l1_category(cat)
                if cat_l1:
                    user_categories[cat_l1] = max(
                        user_categories.get(cat_l1, 0),
                        level
                    )
        
        # 離散化（0-2のレベルに）
        user_categories_discrete = {
            k: min(int(v / 2), 2) for k, v in user_categories.items()
        }
        
        try:
            readiness = self.network_learner.predict_category_readiness(
                user_categories_discrete,
                l1_code
            )
            return readiness
        except Exception as e:
            logger.warning(f"L1準備度計算エラー: {e}")
            return self._simple_l1_readiness(l1_code, user_skills)
    
    def _simple_l1_readiness(
        self,
        l1_code: str,
        user_skills: Dict[str, float]
    ) -> float:
        """
        簡易的なL1準備度計算
        
        Args:
            l1_code: L1カテゴリコード
            user_skills: ユーザーの保有スキル
            
        Returns:
            準備度スコア（0.0-1.0）
        """
        # このL1カテゴリのスキルの平均レベルを計算
        l1_skills = []
        for skill_code, level in user_skills.items():
            if skill_code in self.hierarchy.skill_to_category:
                cat = self.hierarchy.skill_to_category[skill_code]
                if self.hierarchy.get_l1_category(cat) == l1_code:
                    l1_skills.append(level)
        
        if not l1_skills:
            return 0.5  # デフォルト値
        
        avg_level = np.mean(l1_skills)
        return min(avg_level / 5.0, 1.0)  # 0-5 → 0-1に正規化
    
    def _get_l3_score(
        self,
        member_code: str,
        l2_code: str,
        skill_code: str
    ) -> float:
        """
        L3（MF）スコアを取得
        
        Args:
            member_code: メンバーコード
            l2_code: L2カテゴリコード
            skill_code: スキルコード
            
        Returns:
            MFスコア（0.0-1.0）
        """
        if self.mf_learner is None:
            return 0.5  # デフォルト値
        
        try:
            user_skill_matrix = self._prepare_user_skill_matrix()
            scores = self.mf_learner.predict_skill_scores(
                member_code,
                l2_code,
                user_skill_matrix
            )
            
            if skill_code in scores.index:
                score = scores[skill_code]
                # 0-5のスコアを0-1に正規化
                return min(score / 5.0, 1.0)
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"L3スコア計算エラー: {e}")
            return 0.5
    
    def _generate_explanation(
        self,
        l1_code: str,
        l2_code: str,
        l1_readiness: float,
        l2_prob: float,
        l3_score: float
    ) -> str:
        """
        階層的説明文を生成
        
        Args:
            l1_code: L1カテゴリコード
            l2_code: L2カテゴリコード
            l1_readiness: L1準備度
            l2_prob: L2条件付き確率
            l3_score: L3スコア
            
        Returns:
            説明文
        """
        l1_name = self.hierarchy.category_names.get(l1_code, l1_code)
        l2_name = self.hierarchy.category_names.get(l2_code, l2_code)
        
        explanation = (
            f"{l1_name}の準備度{l1_readiness*100:.0f}%、"
            f"{l2_name}への適合度{l2_prob*100:.0f}%、"
            f"スキル推薦度{l3_score*100:.0f}%"
        )
        
        return explanation
    
    def _get_skill_info(self, skill_code: str) -> Dict[str, str]:
        """
        スキル情報を取得
        
        Args:
            skill_code: スキルコード
            
        Returns:
            スキル情報の辞書
        """
        skill_info = self.competence_master[
            self.competence_master['力量コード  ###[skill_code]###'] == skill_code
        ]
        
        if len(skill_info) > 0:
            row = skill_info.iloc[0]
            category_code = self.hierarchy.skill_to_category.get(skill_code, '')
            category_name = self.hierarchy.category_names.get(category_code, '')
            
            return {
                '力量名': row.get('力量名  ###[skill_name]###', skill_code),
                'カテゴリ': category_name
            }
        else:
            return {
                '力量名': skill_code,
                'カテゴリ': ''
            }
