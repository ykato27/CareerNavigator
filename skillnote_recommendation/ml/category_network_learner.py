"""
カテゴリネットワーク学習モジュール（Layer 1）

ベイジアンネットワークを使用して大カテゴリ（L1）レベルでの
ネットワーク構造を学習します。
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    BayesianNetwork = None
    logger.warning("pgmpyがインストールされていません。ベイジアンネットワーク機能は無効です。")


class CategoryNetworkLearner:
    """
    カテゴリネットワーク学習クラス
    
    大カテゴリ（L1）レベルでベイジアンネットワーク構造を学習し、
    カテゴリ間の依存関係をモデル化します。
    """
    
    def __init__(
        self,
        max_indegree: int = 3,
        scoring_method: str = 'bic'
    ):
        """
        初期化
        
        Args:
            max_indegree: ノードへの最大入次数（過学習防止）
            scoring_method: スコアリング手法（'bic', 'aic'など）
        """
        if not PGMPY_AVAILABLE:
            raise ImportError(
                "pgmpyがインストールされていません。"
                "pip install pgmpy でインストールしてください。"
            )
        
        self.max_indegree = max_indegree
        self.scoring_method = scoring_method
        self.model: Optional[BayesianNetwork] = None
        self.inference: Optional[VariableElimination] = None
    
    def aggregate_to_categories(
        self,
        user_skills: pd.DataFrame,
        hierarchy,
        aggregation_method: str = 'max'
    ) -> pd.DataFrame:
        """
        ユーザースキルデータをカテゴリレベルに集約
        
        Args:
            user_skills: ユーザー×スキルのDataFrame（行: ユーザー、列: スキルコード）
            hierarchy: CategoryHierarchy オブジェクト
            aggregation_method: 集約方法（'max', 'mean', 'mode'）
            
        Returns:
            ユーザー×L1カテゴリのDataFrame
        """
        logger.info(f"スキルデータをL1カテゴリに集約中（方法: {aggregation_method}）")
        
        # L1カテゴリごとにスキルを集約
        category_data = {}
        
        for l1_code in hierarchy.level1_categories:
            # このL1カテゴリに属するすべてのスキルを取得（子孫含む）
            from skillnote_recommendation.ml.category_hierarchy_extractor import CategoryHierarchyExtractor
            
            # 仮のextractorを作成（hierarchyから直接取得できるように修正が必要）
            # ここでは簡易的に実装
            skills_in_category = []
            
            # L1の子カテゴリを取得
            if l1_code in hierarchy.children_mapping:
                for l2_code in hierarchy.children_mapping[l1_code]:
                    # L2に直接属するスキル
                    if l2_code in hierarchy.category_to_skills:
                        skills_in_category.extend(hierarchy.category_to_skills[l2_code])
                    
                    # L2の子（L3）に属するスキル
                    if l2_code in hierarchy.children_mapping:
                        for l3_code in hierarchy.children_mapping[l2_code]:
                            if l3_code in hierarchy.category_to_skills:
                                skills_in_category.extend(hierarchy.category_to_skills[l3_code])
            
            # L1に直接属するスキルも含める
            if l1_code in hierarchy.category_to_skills:
                skills_in_category.extend(hierarchy.category_to_skills[l1_code])
            
            # 重複を除去
            skills_in_category = list(set(skills_in_category))
            
            if not skills_in_category:
                logger.warning(f"L1カテゴリ {l1_code} にスキルが見つかりません")
                continue
            
            # user_skillsに存在するスキルのみを使用
            available_skills = [s for s in skills_in_category if s in user_skills.columns]
            
            if not available_skills:
                logger.warning(f"L1カテゴリ {l1_code} の利用可能なスキルがありません")
                continue
            
            # 集約方法に応じて処理
            if aggregation_method == 'max':
                category_data[l1_code] = user_skills[available_skills].max(axis=1)
            elif aggregation_method == 'mean':
                category_data[l1_code] = user_skills[available_skills].mean(axis=1)
            elif aggregation_method == 'mode':
                # 最頻値を使用
                category_data[l1_code] = user_skills[available_skills].mode(axis=1)[0]
            else:
                raise ValueError(f"未知の集約方法: {aggregation_method}")
        
        result = pd.DataFrame(category_data, index=user_skills.index)
        
        logger.info(
            f"集約完了: {len(user_skills.columns)}スキル → "
            f"{len(result.columns)}L1カテゴリ"
        )
        
        return result
    
    def discretize_data(
        self,
        category_data: pd.DataFrame,
        n_bins: int = 3
    ) -> pd.DataFrame:
        """
        連続値データを離散化
        
        Args:
            category_data: カテゴリレベルのデータ
            n_bins: ビン数（離散化のレベル数）
            
        Returns:
            離散化されたDataFrame
        """
        logger.info(f"データを{n_bins}レベルに離散化中")
        
        discretized = category_data.copy()
        
        for col in discretized.columns:
            # すでに離散値の場合はスキップ
            unique_values = discretized[col].nunique()
            if unique_values <= n_bins:
                continue
            
            # 等頻度ビニング
            discretized[col] = pd.qcut(
                discretized[col],
                q=n_bins,
                labels=range(n_bins),
                duplicates='drop'
            ).astype(int)
        
        return discretized
    
    def learn_structure(
        self,
        category_data: pd.DataFrame
    ) -> BayesianNetwork:
        """
        ベイジアンネットワーク構造を学習
        
        Args:
            category_data: カテゴリレベルのデータ（離散化済み）
            
        Returns:
            学習されたBayesianNetwork
        """
        logger.info("ベイジアンネットワーク構造を学習中")
        
        # スコアリング関数を設定
        if self.scoring_method == 'bic':
            scoring_method = BicScore(category_data)
        else:
            raise ValueError(f"未知のスコアリング手法: {self.scoring_method}")
        
        # Hill Climb探索で構造学習
        hc = HillClimbSearch(category_data)
        
        logger.info(f"Hill Climb探索を実行（max_indegree={self.max_indegree}）")
        best_model = hc.estimate(
            scoring_method=scoring_method,
            max_indegree=self.max_indegree
        )
        
        logger.info(
            f"構造学習完了: {len(best_model.nodes())}ノード, "
            f"{len(best_model.edges())}エッジ"
        )
        
        self.model = best_model
        return best_model
    
    def fit_parameters(
        self,
        category_data: pd.DataFrame
    ):
        """
        ベイジアンネットワークのパラメータ（CPD）を学習
        
        Args:
            category_data: カテゴリレベルのデータ（離散化済み）
        """
        if self.model is None:
            raise ValueError("先にlearn_structure()を実行してください")
        
        logger.info("CPD（条件付き確率分布）を学習中")
        
        # 最尤推定でパラメータを学習
        self.model.fit(
            category_data,
            estimator=MaximumLikelihoodEstimator
        )
        
        # 推論エンジンを初期化
        self.inference = VariableElimination(self.model)
        
        logger.info("パラメータ学習完了")
    
    def fit(
        self,
        user_skills: pd.DataFrame,
        hierarchy,
        aggregation_method: str = 'max',
        n_bins: int = 3
    ) -> BayesianNetwork:
        """
        完全な学習パイプライン
        
        Args:
            user_skills: ユーザー×スキルのDataFrame
            hierarchy: CategoryHierarchy オブジェクト
            aggregation_method: 集約方法
            n_bins: 離散化のビン数
            
        Returns:
            学習されたBayesianNetwork
        """
        # 1. カテゴリレベルに集約
        category_data = self.aggregate_to_categories(
            user_skills,
            hierarchy,
            aggregation_method
        )
        
        # 2. 離散化
        category_data_discrete = self.discretize_data(category_data, n_bins)
        
        # 3. 構造学習
        self.learn_structure(category_data_discrete)
        
        # 4. パラメータ学習
        self.fit_parameters(category_data_discrete)
        
        return self.model
    
    def predict_category_readiness(
        self,
        user_categories: Dict[str, int],
        target_category: str
    ) -> float:
        """
        ユーザーの保有カテゴリから、ターゲットカテゴリの準備度を予測
        
        Args:
            user_categories: ユーザーが保有するカテゴリとそのレベル
            target_category: 予測対象のカテゴリ
            
        Returns:
            準備度スコア（0.0-1.0）
        """
        if self.inference is None:
            raise ValueError("先にfit()を実行してください")
        
        # ターゲットカテゴリが保有カテゴリに含まれる場合
        if target_category in user_categories:
            # 正規化されたスコアを返す
            return user_categories[target_category] / 2.0  # 0-2 → 0-1
        
        # ベイジアン推論で確率を計算
        try:
            # エビデンスを設定
            evidence = {k: v for k, v in user_categories.items() if k in self.model.nodes()}
            
            if not evidence:
                # エビデンスがない場合は事前確率を使用
                return 0.5
            
            # 推論実行
            result = self.inference.query(
                variables=[target_category],
                evidence=evidence
            )
            
            # 高レベル（レベル2）の確率を準備度として使用
            if hasattr(result, 'values'):
                # 最高レベルの確率を返す
                return float(result.values[-1])
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"推論エラー: {e}")
            return 0.5
    
    def get_network_info(self) -> Dict:
        """
        学習されたネットワークの情報を取得
        
        Returns:
            ネットワーク情報の辞書
        """
        if self.model is None:
            return {}
        
        return {
            'nodes': list(self.model.nodes()),
            'edges': list(self.model.edges()),
            'n_nodes': len(self.model.nodes()),
            'n_edges': len(self.model.edges()),
            'max_indegree': self.max_indegree
        }
