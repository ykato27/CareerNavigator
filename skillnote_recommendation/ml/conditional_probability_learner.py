"""
条件付き確率学習モジュール（Layer 2）

P(中カテゴリ | 大カテゴリ)の条件付き確率を学習します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConditionalProbabilityLearner:
    """
    条件付き確率学習クラス
    
    カテゴリ階層に基づいて、P(L2 | L1)の条件付き確率を学習します。
    """
    
    def __init__(self):
        """初期化"""
        self.conditional_probs: Dict[str, Dict[str, float]] = {}
        self.hierarchy = None
    
    def learn_conditional_probs(
        self,
        hierarchy,
        user_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        条件付き確率 P(L2 | L1) を学習
        
        Args:
            hierarchy: CategoryHierarchy オブジェクト
            user_data: ユーザー×スキルのDataFrame
            
        Returns:
            条件付き確率の辞書 {L1_code: {L2_code: probability}}
        """
        logger.info("条件付き確率 P(L2|L1) を学習中")

        self.hierarchy = hierarchy
        self.conditional_probs = {}

        logger.info(f"L1カテゴリ数: {len(hierarchy.level1_categories)}")
        logger.info(f"L2カテゴリ数: {len(hierarchy.level2_categories)}")
        logger.info(f"親子マッピング数: {len(hierarchy.children_mapping)}")

        # 各L1カテゴリについて処理
        for l1_code in hierarchy.level1_categories:
            l1_name = hierarchy.category_names.get(l1_code, l1_code)
            logger.debug(f"処理中のL1カテゴリ: {l1_name} ({l1_code})")

            # このL1の子カテゴリ（L2）を取得
            l2_children = []
            if l1_code in hierarchy.children_mapping:
                logger.debug(f"  {l1_code} の子カテゴリ: {hierarchy.children_mapping[l1_code]}")
                for child in hierarchy.children_mapping[l1_code]:
                    if child in hierarchy.level2_categories:
                        l2_children.append(child)
                        logger.debug(f"    L2子を追加: {hierarchy.category_names.get(child, child)}")
            else:
                logger.debug(f"  {l1_code} は children_mapping に存在しません")

            if not l2_children:
                logger.warning(f"L1カテゴリ {l1_name} にL2子カテゴリがありません")
                continue
            
            # L1とL2のスキルレベルを集約
            l1_levels = self._aggregate_category_level(l1_code, user_data, hierarchy)
            
            l2_probs = {}
            for l2_code in l2_children:
                l2_levels = self._aggregate_category_level(l2_code, user_data, hierarchy)
                
                # 条件付き確率を計算
                # P(L2 | L1) = P(L2とL1が両方高い) / P(L1が高い)
                prob = self._calculate_conditional_prob(l1_levels, l2_levels)
                l2_probs[l2_code] = prob
            
            # 正規化（合計が1になるように）
            total = sum(l2_probs.values())
            if total > 0:
                l2_probs = {k: v / total for k, v in l2_probs.items()}
            
            self.conditional_probs[l1_code] = l2_probs
            
            logger.debug(
                f"L1 {l1_name}: {len(l2_probs)}個のL2カテゴリの確率を学習"
            )
        
        logger.info(
            f"条件付き確率学習完了: {len(self.conditional_probs)}個のL1カテゴリ"
        )
        
        return self.conditional_probs
    
    def _aggregate_category_level(
        self,
        category_code: str,
        user_data: pd.DataFrame,
        hierarchy
    ) -> pd.Series:
        """
        カテゴリのスキルレベルを集約
        
        Args:
            category_code: カテゴリコード
            user_data: ユーザー×スキルのDataFrame
            hierarchy: CategoryHierarchy オブジェクト
            
        Returns:
            ユーザーごとのカテゴリレベル（Series）
        """
        # このカテゴリに属するすべてのスキルを取得
        skills_in_category = []
        
        # 直接このカテゴリに属するスキル
        if category_code in hierarchy.category_to_skills:
            skills_in_category.extend(hierarchy.category_to_skills[category_code])
        
        # 子カテゴリのスキルも含める
        if category_code in hierarchy.children_mapping:
            for child_code in hierarchy.children_mapping[category_code]:
                if child_code in hierarchy.category_to_skills:
                    skills_in_category.extend(hierarchy.category_to_skills[child_code])
                
                # 孫カテゴリのスキルも含める
                if child_code in hierarchy.children_mapping:
                    for grandchild_code in hierarchy.children_mapping[child_code]:
                        if grandchild_code in hierarchy.category_to_skills:
                            skills_in_category.extend(
                                hierarchy.category_to_skills[grandchild_code]
                            )
        
        # 重複を除去
        skills_in_category = list(set(skills_in_category))
        
        # user_dataに存在するスキルのみを使用
        available_skills = [s for s in skills_in_category if s in user_data.columns]
        
        if not available_skills:
            # スキルがない場合は0を返す
            return pd.Series(0, index=user_data.index)
        
        # 最大値で集約
        return user_data[available_skills].max(axis=1)
    
    def _calculate_conditional_prob(
        self,
        l1_levels: pd.Series,
        l2_levels: pd.Series
    ) -> float:
        """
        条件付き確率 P(L2が高い | L1が高い) を計算
        
        Args:
            l1_levels: L1カテゴリのレベル
            l2_levels: L2カテゴリのレベル
            
        Returns:
            条件付き確率
        """
        # 「高い」の閾値を設定（レベル3以上）
        threshold = 3
        
        # L1が高いユーザー
        l1_high = l1_levels >= threshold
        
        # L1が高いユーザーの数
        n_l1_high = l1_high.sum()
        
        if n_l1_high == 0:
            # L1が高いユーザーがいない場合は均等確率
            return 0.5
        
        # L1が高く、かつL2も高いユーザーの数
        n_both_high = (l1_high & (l2_levels >= threshold)).sum()
        
        # 条件付き確率
        prob = n_both_high / n_l1_high
        
        # スムージング（ゼロ確率を避ける）
        alpha = 0.1
        prob = (prob + alpha) / (1 + alpha * 2)
        
        return prob
    
    def get_conditional_prob(
        self,
        l1_code: str,
        l2_code: str
    ) -> float:
        """
        特定のL1→L2の条件付き確率を取得
        
        Args:
            l1_code: L1カテゴリコード
            l2_code: L2カテゴリコード
            
        Returns:
            条件付き確率
        """
        if l1_code not in self.conditional_probs:
            return 0.5  # デフォルト値
        
        if l2_code not in self.conditional_probs[l1_code]:
            return 0.5  # デフォルト値
        
        return self.conditional_probs[l1_code][l2_code]
    
    def get_l2_probabilities_given_l1(
        self,
        l1_code: str
    ) -> Dict[str, float]:
        """
        L1が与えられたときの、すべてのL2の確率を取得
        
        Args:
            l1_code: L1カテゴリコード
            
        Returns:
            L2カテゴリコードと確率の辞書
        """
        if l1_code not in self.conditional_probs:
            return {}
        
        return self.conditional_probs[l1_code].copy()
    
    def get_all_conditional_probs(self) -> Dict[str, Dict[str, float]]:
        """
        すべての条件付き確率を取得
        
        Returns:
            条件付き確率の辞書 {L1: {L2: prob}}
        """
        return self.conditional_probs.copy()
