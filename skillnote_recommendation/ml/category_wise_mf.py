"""
カテゴリ別行列分解モジュール（Layer 3）

各中カテゴリ（L2）ごとに独立した行列分解モデルを学習し、
スキルレベルの推薦スコアを生成します。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
    MF_AVAILABLE = True
except ImportError:
    MF_AVAILABLE = False
    MatrixFactorizationModel = None
    logger.warning("MatrixFactorizationModelがインポートできません")


class CategoryWiseMatrixFactorization:
    """
    カテゴリ別行列分解クラス
    
    各L2カテゴリごとに独立したMatrix Factorizationモデルを学習し、
    スキルレベルの推薦スコアを生成します。
    """
    
    def __init__(
        self,
        n_components: int = 10,
        max_iter: int = 200,
        random_state: int = 42
    ):
        """
        初期化
        
        Args:
            n_components: 潜在因子の数
            max_iter: 最大反復回数
            random_state: 乱数シード
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.category_models: Dict[str, MatrixFactorizationModel] = {}
        self.hierarchy = None
    
    def fit_category_models(
        self,
        user_skills: pd.DataFrame,
        hierarchy
    ) -> Dict[str, MatrixFactorizationModel]:
        """
        各L2カテゴリごとにMFモデルを学習
        
        Args:
            user_skills: ユーザー×スキルのDataFrame
            hierarchy: CategoryHierarchy オブジェクト
            
        Returns:
            カテゴリコードとMFモデルの辞書
        """
        logger.info("カテゴリ別行列分解モデルを学習中")

        self.hierarchy = hierarchy
        self.category_models = {}

        logger.info(f"L2カテゴリ数: {len(hierarchy.level2_categories)}")
        logger.info(f"親子マッピング数: {len(hierarchy.children_mapping)}")

        # 各L2カテゴリについて処理
        for l2_code in hierarchy.level2_categories:
            l2_name = hierarchy.category_names.get(l2_code, l2_code)
            logger.debug(f"処理中のL2カテゴリ: {l2_name} ({l2_code})")

            # このL2カテゴリに属するスキルを取得
            skills_in_category = self._get_skills_in_l2_category(l2_code, hierarchy)
            logger.debug(f"  取得したスキル数: {len(skills_in_category)}")

            # デバッグ: L2の子カテゴリを表示
            if l2_code in hierarchy.children_mapping:
                l3_children = [c for c in hierarchy.children_mapping[l2_code]
                              if c in hierarchy.level3_categories]
                logger.debug(f"  L3子カテゴリ数: {len(l3_children)}")
                if len(l3_children) > 0:
                    logger.debug(f"  L3子カテゴリ例: {[hierarchy.category_names.get(c, c) for c in l3_children[:3]]}")
            else:
                logger.debug(f"  {l2_code} は children_mapping に存在しません")

            if len(skills_in_category) < 2:
                logger.warning(
                    f"L2カテゴリ {l2_name} のスキル数が不足（{len(skills_in_category)}個）"
                )
                continue
            
            # user_skillsに存在するスキルのみを使用
            available_skills = [s for s in skills_in_category if s in user_skills.columns]
            
            if len(available_skills) < 2:
                logger.debug(
                    f"L2カテゴリ {l2_name} の利用可能なスキル数が不足"
                )
                continue
            
            # このカテゴリのスキルデータを抽出
            category_data = user_skills[available_skills].copy()
            
            # スキルを持っているユーザーが少なすぎる場合はスキップ
            skill_counts = (category_data > 0).sum(axis=0)
            if skill_counts.max() < 3:
                logger.debug(
                    f"L2カテゴリ {l2_name} のユーザー数が不足"
                )
                continue
            
            # MFモデルを学習
            try:
                if MF_AVAILABLE and MatrixFactorizationModel is not None:
                    model = MatrixFactorizationModel(
                        n_components=min(self.n_components, len(available_skills) - 1),
                        max_iter=self.max_iter,
                        random_state=self.random_state
                    )
                    model.fit(category_data)
                    self.category_models[l2_code] = model
                    
                    logger.debug(
                        f"L2カテゴリ {l2_name}: {len(available_skills)}スキルで"
                        f"MFモデルを学習"
                    )
                else:
                    # MatrixFactorizationModelが利用できない場合は簡易実装
                    model = self._create_simple_model(category_data)
                    self.category_models[l2_code] = model
                    
            except Exception as e:
                logger.warning(f"L2カテゴリ {l2_name} のMF学習エラー: {e}")
                continue
        
        logger.info(
            f"カテゴリ別MF学習完了: {len(self.category_models)}個のL2カテゴリ"
        )
        
        return self.category_models
    
    def _get_skills_in_l2_category(
        self,
        l2_code: str,
        hierarchy
    ) -> List[str]:
        """
        L2カテゴリに属するすべてのスキルを取得
        
        Args:
            l2_code: L2カテゴリコード
            hierarchy: CategoryHierarchy オブジェクト
            
        Returns:
            スキルコードのリスト
        """
        skills = []
        
        # L2に直接属するスキル
        if l2_code in hierarchy.category_to_skills:
            skills.extend(hierarchy.category_to_skills[l2_code])
        
        # L2の子（L3）に属するスキル
        if l2_code in hierarchy.children_mapping:
            for l3_code in hierarchy.children_mapping[l2_code]:
                if l3_code in hierarchy.category_to_skills:
                    skills.extend(hierarchy.category_to_skills[l3_code])
        
        # 重複を除去
        return list(set(skills))
    
    def _create_simple_model(self, data: pd.DataFrame):
        """
        簡易的なモデルを作成（MatrixFactorizationModelが利用できない場合）
        
        Args:
            data: スキルデータ
            
        Returns:
            簡易モデル（辞書）
        """
        # 各スキルの平均値を計算
        skill_means = data.mean(axis=0)
        
        return {
            'type': 'simple',
            'skill_means': skill_means,
            'data': data
        }
    
    def predict_skill_scores(
        self,
        member_code: str,
        l2_category: str,
        user_skills: pd.DataFrame
    ) -> pd.Series:
        """
        特定のL2カテゴリ内のスキルスコアを予測
        
        Args:
            member_code: メンバーコード
            l2_category: L2カテゴリコード
            user_skills: ユーザー×スキルのDataFrame
            
        Returns:
            スキルコードとスコアのSeries
        """
        if l2_category not in self.category_models:
            # モデルがない場合は空のSeriesを返す
            return pd.Series(dtype=float)
        
        model = self.category_models[l2_category]
        
        # モデルのタイプに応じて予測
        if isinstance(model, dict) and model.get('type') == 'simple':
            # 簡易モデルの場合
            return model['skill_means']
        else:
            # MatrixFactorizationModelの場合
            try:
                if member_code in user_skills.index:
                    # メンバーの予測スコアを取得
                    scores = model.predict_for_member(member_code)
                    return scores
                else:
                    # メンバーが存在しない場合は平均スコアを返す
                    return pd.Series(dtype=float)
            except Exception as e:
                logger.warning(f"予測エラー: {e}")
                return pd.Series(dtype=float)
    
    def predict_all_skills(
        self,
        member_code: str,
        user_skills: pd.DataFrame
    ) -> pd.Series:
        """
        すべてのカテゴリのスキルスコアを予測
        
        Args:
            member_code: メンバーコード
            user_skills: ユーザー×スキルのDataFrame
            
        Returns:
            すべてのスキルのスコア
        """
        all_scores = {}
        
        for l2_code, model in self.category_models.items():
            scores = self.predict_skill_scores(member_code, l2_code, user_skills)
            all_scores.update(scores.to_dict())
        
        return pd.Series(all_scores)
    
    def get_model_info(self) -> Dict:
        """
        学習されたモデルの情報を取得
        
        Returns:
            モデル情報の辞書
        """
        info = {
            'n_categories': len(self.category_models),
            'categories': list(self.category_models.keys()),
            'n_components': self.n_components,
            'max_iter': self.max_iter
        }
        
        # 各カテゴリのスキル数を追加
        if self.hierarchy:
            category_skill_counts = {}
            for l2_code in self.category_models.keys():
                skills = self._get_skills_in_l2_category(l2_code, self.hierarchy)
                category_skill_counts[l2_code] = len(skills)
            info['category_skill_counts'] = category_skill_counts
        
        return info
