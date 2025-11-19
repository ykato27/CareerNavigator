"""
因果グラフ推薦モジュール

学習された因果構造（Causal Graph）に基づいて、
「なぜそのスキルが必要か（原因）」と「そのスキルが何に役立つか（結果）」
の両面から説得力のある推薦を行います。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from skillnote_recommendation.ml.causal_structure_learner import CausalStructureLearner
from skillnote_recommendation.ml.bayesian_network_recommender import BayesianNetworkRecommender
from skillnote_recommendation.config import config
from skillnote_recommendation.utils.logger import setup_logger

logger = setup_logger(__name__)

class CausalGraphRecommender:
    """
    因果グラフ活用型推薦クラス
    
    LiNGAMで学習した因果関係を用いて、以下の2つの観点からスコアリングを行います。
    1. Readiness (準備完了度): 既に持っているスキルから、そのスキルへの因果効果の合計
       「あなたはAを持っているので、Bを習得する準備ができています」
    2. Utility (有用性): そのスキルから、将来習得すべきスキルへの因果効果の合計
       「Bを習得すると、将来CやDの習得に役立ちます」
    """
    
    def __init__(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        learner_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            learner_params: CausalStructureLearnerのパラメータ
        """
        self.member_competence = member_competence
        self.competence_master = competence_master
        
        params = learner_params or {}
        # デフォルトパラメータをconfigから取得してマージすることも可能
        if 'random_state' not in params:
            params['random_state'] = config.model.RANDOM_STATE
            
        self.learner = CausalStructureLearner(**params)
        self.bn_recommender: Optional[BayesianNetworkRecommender] = None
        
        self.is_fitted = False
        self.skill_matrix_: Optional[pd.DataFrame] = None
        self.total_effects_: Optional[Dict[str, Dict[str, float]]] = None
        self.code_to_name: Dict[str, str] = {}
        self.name_to_code: Dict[str, str] = {}

    def fit(self, min_members_per_skill: int = 5) -> 'CausalGraphRecommender':
        """
        モデルを学習
        
        Args:
            min_members_per_skill: 学習に含めるスキルの最小保持人数
            
        Returns:
            self
        """
        logger.info("因果グラフ推薦モデルの学習開始")
        
        # 1. データ前処理: メンバー×スキルのマトリクス作成
        # 正規化レベルを使用
        skill_matrix = self.member_competence.pivot_table(
            index='メンバーコード',
            columns='力量コード',
            values='正規化レベル',
            fill_value=0.0
        )
        
        # 力量名へのマッピング
        self.code_to_name = dict(zip(
            self.competence_master['力量コード'],
            self.competence_master['力量名']
        ))
        self.name_to_code = {v: k for k, v in self.code_to_name.items()}
        
        # カラム名を力量名に変換（可読性のため）
        renamed_cols = {}
        for code in skill_matrix.columns:
            name = self.code_to_name.get(code, str(code))
            # 重複回避
            if name in renamed_cols.values():
                name = f"{name}_{code}" 
            renamed_cols[code] = name
            
        skill_matrix_renamed = skill_matrix.rename(columns=renamed_cols)
        
        # メンバー数が少なすぎるスキルを除外
        counts = (skill_matrix_renamed > 0).sum()
        valid_skills = counts[counts >= min_members_per_skill].index
        skill_matrix_filtered = skill_matrix_renamed[valid_skills]
        
        logger.info(f"スキル数: {len(skill_matrix.columns)} -> {len(valid_skills)} (フィルタ後)")
        
        self.skill_matrix_ = skill_matrix_filtered
        
        # 2. 因果構造学習
        self.learner.fit(skill_matrix_filtered)
        
        # 3. 総合効果の取得
        # 3. 総合効果の取得
        self.total_effects_ = self.learner.get_causal_effects()
        
        # 4. ベイジアンネットワークの学習
        try:
            adj_matrix = self.learner.get_adjacency_matrix()
            self.bn_recommender = BayesianNetworkRecommender(adj_matrix)
            # バイナリデータ（0/1）に変換して学習
            binary_data = (skill_matrix_filtered > 0).astype(int)
            self.bn_recommender.fit(binary_data)
        except Exception as e:
            logger.error(f"ベイジアンネットワークの学習に失敗しました: {e}")
            self.bn_recommender = None
        
        self.is_fitted = True
        logger.info("学習完了")
        
        return self

    def recommend(self, member_code: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        メンバーへのスキル推薦
        
        Args:
            member_code: メンバーID
            top_n: 推薦件数
            
        Returns:
            推薦結果のリスト
        """
        if not self.is_fitted:
            logger.warning("モデルが学習されていません")
            return []
            
        # メンバーの保有スキル取得
        if member_code not in self.skill_matrix_.index:
            logger.warning(f"メンバー {member_code} がデータに存在しません")
            return []
            
        member_skills = self.skill_matrix_.loc[member_code]
        owned_skills = member_skills[member_skills > 0].index.tolist()
        unowned_skills = member_skills[member_skills == 0].index.tolist()
        
        scores = []
        
        for target_skill in unowned_skills:
            # 1. Readiness Score: Owned -> Target
            readiness_score = 0.0
            readiness_reasons = []
            
            for owned in owned_skills:
                # owned が target に与える影響
                effect = self._get_effect(owned, target_skill)
                # 閾値を0.001に下げて、より弱い因果効果も捕捉
                if effect > 0.001:
                    readiness_score += effect
                    readiness_reasons.append((owned, effect))
            
            # 2. Utility Score: Target -> Unowned (Future)
            utility_score = 0.0
            utility_reasons = []
            
            for future in unowned_skills:
                if future == target_skill:
                    continue
                # target が future に与える影響
                effect = self._get_effect(target_skill, future)
                # 閾値を0.001に下げて、より弱い因果効果も捕捉
                if effect > 0.001:
                    utility_score += effect
                    utility_reasons.append((future, effect))
                    utility_score += effect
                    utility_reasons.append((future, effect))
            
            # 3. Bayesian Score: P(Target=1 | Owned)
            bayesian_score = 0.0
            if self.bn_recommender:
                try:
                    # ターゲットスキル名をコードから名前に変換する必要があるか確認
                    # skill_matrixのカラム名は既に名前に変換されている(fitメソッド内で)
                    # なので、target_skill (名前) をそのまま渡せばOK
                    bayesian_score = self.bn_recommender.predict_probability(owned_skills, target_skill)
                except Exception as e:
                    logger.warning(f"ベイジアン推論エラー ({target_skill}): {e}")
            
            # 総合スコア: Readinessを重視（0.9）してユーザー固有の推薦を強化
            # ベイジアン確率も考慮に入れる（Readinessの一部として解釈可能）
            # 新スコア = (Readiness * 0.7 + Bayesian * 0.3) * 0.9 + Utility * 0.1 くらい？
            # シンプルに: total = readiness * 0.6 + bayesian * 0.3 + utility * 0.1
            
            total_score = readiness_score * 0.6 + bayesian_score * 0.3 + utility_score * 0.1
            
            # Readiness Scoreが0のスキルは除外（ユーザー固有性を確保）
            # ただし、保有スキルが少ない場合は例外的に含める
            min_readiness = 0.0 if len(owned_skills) < 3 else 0.001
            
            if total_score > 0 and readiness_score >= min_readiness:
                scores.append({
                    'skill_name': target_skill,
                    'skill_code': self.name_to_code.get(target_skill, target_skill),
                    'total_score': total_score,
                    'readiness_score': readiness_score,
                    'utility_score': utility_score,
                    'bayesian_score': bayesian_score,
                    'readiness_reasons': sorted(readiness_reasons, key=lambda x: x[1], reverse=True),
                    'utility_reasons': sorted(utility_reasons, key=lambda x: x[1], reverse=True)
                })
        
        # ソート
        scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 上位N件を整形して返す
        results = []
        for item in scores[:top_n]:
            explanation = self._generate_explanation(item)
            results.append({
                'competence_code': item['skill_code'],
                'competence_name': item['skill_name'],
                'score': item['total_score'],
                'explanation': explanation,
                'details': item # 詳細データも含める
            })
            
        return results

    def _get_effect(self, cause: str, effect: str) -> float:
        """因果効果を取得 (存在しない場合は0)"""
        if self.total_effects_ is None:
            return 0.0

        return self.total_effects_.get(cause, {}).get(effect, 0.0)

    def _generate_explanation(self, item: Dict[str, Any]) -> str:
        """推薦の説明文を生成"""
        lines = []

        # Readiness (準備完了度)
        if item['readiness_reasons']:
            lines.append("【習得の準備ができています】")
            for skill_name, score in item['readiness_reasons'][:3]:  # 上位3つ
                lines.append(f"・{skill_name} の経験があるため (因果効果: {score:.3f})")

        # Bayesian Score
        if item['bayesian_score'] > 0:
            prob_pct = item['bayesian_score'] * 100
            lines.append(f"・同様のスキルセットを持つ方の {prob_pct:.1f}% がこのスキルを習得しています")

        # Utility (有用性)
        if item['utility_reasons']:
            lines.append("\n【このスキルが役立つ場面】")
            for skill_name, score in item['utility_reasons'][:3]:  # 上位3つ
                lines.append(f"・{skill_name} の習得に役立ちます (因果効果: {score:.3f})")

        if not lines:
            lines.append("・基礎スキルとして推奨されます。")

        return "\n".join(lines)
