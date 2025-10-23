"""
推薦エンジン

力量推薦のコアロジックを提供
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from skillnote_recommendation.core.models import Recommendation
from skillnote_recommendation.core.config import Config


class RecommendationEngine:
    """推薦エンジンクラス"""
    
    def __init__(self, 
                 df_members: pd.DataFrame,
                 df_competence_master: pd.DataFrame,
                 df_member_competence: pd.DataFrame,
                 df_similarity: pd.DataFrame,
                 category_importance_weight: float = None,
                 acquisition_ease_weight: float = None,
                 popularity_weight: float = None):
        """
        初期化
        
        Args:
            df_members: 会員マスタ
            df_competence_master: 力量マスタ
            df_member_competence: 会員習得力量データ
            df_similarity: 類似度データ
            category_importance_weight: カテゴリ重要度の重み
            acquisition_ease_weight: 習得容易性の重み
            popularity_weight: 人気度の重み
        """
        self.df_members = df_members
        self.df_competence_master = df_competence_master
        self.df_member_competence = df_member_competence
        self.df_similarity = df_similarity
        
        # 重み付けパラメータ
        params = Config.RECOMMENDATION_PARAMS
        self.category_importance_weight = category_importance_weight or params['category_importance_weight']
        self.acquisition_ease_weight = acquisition_ease_weight or params['acquisition_ease_weight']
        self.popularity_weight = popularity_weight or params['popularity_weight']
    
    def get_member_competences(self, member_code: str) -> pd.DataFrame:
        """
        会員の保有力量を取得
        
        Args:
            member_code: 会員コード
            
        Returns:
            保有力量データ
        """
        return self.df_member_competence[
            self.df_member_competence['メンバーコード'] == member_code
        ]
    
    def get_unacquired_competences(self, member_code: str, 
                                   competence_type: Optional[str] = None) -> pd.DataFrame:
        """
        未習得力量を取得
        
        Args:
            member_code: 会員コード
            competence_type: 力量タイプ（SKILL/EDUCATION/LICENSE）
            
        Returns:
            未習得力量データ
        """
        acquired = set(self.get_member_competences(member_code)['力量コード'].unique())
        
        all_competences = self.df_competence_master.copy()
        if competence_type:
            all_competences = all_competences[all_competences['力量タイプ'] == competence_type]
        
        return all_competences[~all_competences['力量コード'].isin(acquired)]
    
    def calculate_category_importance(self, competence_code: str, category: str) -> float:
        """
        カテゴリ重要度を計算
        
        Args:
            competence_code: 力量コード
            category: カテゴリ名
            
        Returns:
            カテゴリ重要度（0-10）
        """
        category_competences = self.df_member_competence[
            self.df_member_competence['力量カテゴリー名'] == category
        ]
        
        if len(category_competences) == 0:
            return 5.0
        
        target_count = len(category_competences[
            category_competences['力量コード'] == competence_code
        ]['メンバーコード'].unique())
        
        max_count = category_competences.groupby('力量コード')['メンバーコード'].nunique().max()
        
        if max_count == 0:
            return 5.0
        
        return (target_count / max_count) * 10
    
    def calculate_acquisition_ease(self, member_code: str, 
                                   target_competence_code: str) -> float:
        """
        習得容易性を計算
        
        Args:
            member_code: 会員コード
            target_competence_code: 対象力量コード
            
        Returns:
            習得容易性（0-10）
        """
        acquired = set(self.get_member_competences(member_code)['力量コード'].unique())
        
        similar_competences = self.df_similarity[
            (self.df_similarity['力量1'] == target_competence_code) |
            (self.df_similarity['力量2'] == target_competence_code)
        ]
        
        if len(similar_competences) == 0:
            return 3.0
        
        max_similarity = 0.0
        for _, row in similar_competences.iterrows():
            related_code = row['力量2'] if row['力量1'] == target_competence_code else row['力量1']
            
            if related_code in acquired:
                similarity = row['類似度']
                max_similarity = max(max_similarity, similarity)
        
        return 3.0 + (max_similarity * 7.0)
    
    def calculate_popularity(self, competence_code: str) -> float:
        """
        人気度を計算
        
        Args:
            competence_code: 力量コード
            
        Returns:
            人気度（0-10）
        """
        acquirer_count = len(self.df_member_competence[
            self.df_member_competence['力量コード'] == competence_code
        ]['メンバーコード'].unique())
        
        total_members = len(self.df_members)
        
        return (acquirer_count / total_members) * 10
    
    def calculate_priority_score(self, category_importance: float,
                                 acquisition_ease: float, popularity: float) -> float:
        """
        優先度スコアを計算
        
        Args:
            category_importance: カテゴリ重要度
            acquisition_ease: 習得容易性
            popularity: 人気度
            
        Returns:
            優先度スコア
        """
        return (
            category_importance * self.category_importance_weight +
            acquisition_ease * self.acquisition_ease_weight +
            popularity * self.popularity_weight
        )
    
    def generate_recommendation_reason(self, competence_name: str, competence_type: str,
                                      category: str, category_importance: float,
                                      acquisition_ease: float, popularity: float) -> str:
        """
        推薦理由を生成
        
        Args:
            competence_name: 力量名
            competence_type: 力量タイプ
            category: カテゴリ名
            category_importance: カテゴリ重要度
            acquisition_ease: 習得容易性
            popularity: 人気度
            
        Returns:
            推薦理由
        """
        reasons = []
        
        if category_importance >= 8:
            reasons.append(f"{category}において多くの技術者が習得している力量です")
        elif category_importance >= 5:
            reasons.append(f"{category}で推奨される力量です")
        
        if acquisition_ease >= 7:
            reasons.append("類似する力量を既に保有しているため習得しやすいです")
        elif acquisition_ease >= 5:
            reasons.append("関連する知識があれば取り組みやすい力量です")
        
        if popularity >= 0.5:
            reasons.append("多くの技術者が習得している基本的な力量です")
        
        if competence_type == 'SKILL':
            reasons.append("実務スキルとして習得を推奨します")
        elif competence_type == 'EDUCATION':
            reasons.append("教育研修を受講することで習得できます")
        elif competence_type == 'LICENSE':
            reasons.append("資格取得により専門性を証明できます")
        
        if not reasons:
            reasons.append("キャリア発展に有用な力量です")
        
        return "。".join(reasons) + "。"
    
    def recommend(self, member_code: str, competence_type: Optional[str] = None,
                 category_filter: Optional[str] = None, top_n: int = 10) -> List[Recommendation]:
        """
        力量を推薦
        
        Args:
            member_code: 会員コード
            competence_type: 力量タイプフィルタ（None/SKILL/EDUCATION/LICENSE）
            category_filter: カテゴリフィルタ（部分一致）
            top_n: 推薦件数
            
        Returns:
            推薦結果のリスト
        """
        unacquired = self.get_unacquired_competences(member_code, competence_type)
        
        if category_filter:
            unacquired = unacquired[
                unacquired['力量カテゴリー名'].str.contains(category_filter, na=False, case=False)
            ]
        
        if len(unacquired) == 0:
            return []
        
        recommendations = []
        
        for _, row in unacquired.iterrows():
            comp_code = row['力量コード']
            comp_name = row['力量名']
            comp_type = row['力量タイプ']
            category = row['力量カテゴリー名']
            
            # スコア計算
            cat_importance = self.calculate_category_importance(comp_code, category)
            ease = self.calculate_acquisition_ease(member_code, comp_code)
            popularity = self.calculate_popularity(comp_code)
            
            priority = self.calculate_priority_score(cat_importance, ease, popularity)
            
            reason = self.generate_recommendation_reason(
                comp_name, comp_type, category,
                cat_importance, ease, popularity
            )
            
            recommendation = Recommendation(
                competence_code=comp_code,
                competence_name=comp_name,
                competence_type=comp_type,
                category=category,
                priority_score=priority,
                category_importance=cat_importance,
                acquisition_ease=ease,
                popularity=popularity,
                reason=reason
            )
            
            recommendations.append(recommendation)
        
        # 優先度でソート
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return recommendations[:top_n]
