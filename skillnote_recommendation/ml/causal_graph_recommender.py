"""
因果グラフ推薦モジュール

学習された因果構造（Causal Graph）に基づいて、
「なぜそのスキルが必要か（原因）」と「そのスキルが何に役立つか（結果）」
の両面から説得力のある推薦を行います。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from skillnote_recommendation.ml.causal_structure_learner import CausalStructureLearner

logger = logging.getLogger(__name__)

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
        learner_params: Optional[Dict] = None
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
        self.learner = CausalStructureLearner(**params)
        
        self.is_fitted = False
        self.skill_matrix_ = None
        self.total_effects_ = None

    def fit(self, min_members_per_skill: int = 5):
        """
        モデルを学習
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
        # 注意: 重複がある場合はコードを付与するなど対策が必要だが、ここでは単純化
        renamed_cols = {}
        for code in skill_matrix.columns:
            name = self.code_to_name.get(code, code)
            if name in renamed_cols.values():
                name = f"{name}_{code}" # 重複回避
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
        self.total_effects_ = self.learner.get_causal_effects()
        
        self.is_fitted = True
        logger.info("学習完了")
        
        return self

    def recommend(self, member_code: str, top_n: int = 10) -> List[Dict]:
        """
        メンバーへのスキル推薦
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
                if effect > 0.01:
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
                if effect > 0.01:
                    utility_score += effect
                    utility_reasons.append((future, effect))
            
            # 総合スコア (重み付けは調整可能)
            total_score = readiness_score * 0.6 + utility_score * 0.4
            
            if total_score > 0:
                scores.append({
                    'skill_name': target_skill,
                    'skill_code': self.name_to_code.get(target_skill, target_skill),
                    'total_score': total_score,
                    'readiness_score': readiness_score,
                    'utility_score': utility_score,
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

    def _generate_explanation(self, item: Dict) -> str:
        """推薦理由のテキスト生成"""
        skill_name = item['skill_name']
        readiness = item['readiness_reasons']
        utility = item['utility_reasons']
        
        lines = [f"【因果推論推薦】スコア: {item['total_score']:.2f}"]
        
        if readiness:
            top_cause, effect_val = readiness[0]
            lines.append(f"・準備: あなたの「{top_cause}」スキルが、このスキルの習得を強く後押しします (効果: {effect_val:.2f})。")
            
        if utility:
            top_result, effect_val = utility[0]
            lines.append(f"・将来: このスキルを学ぶと、将来「{top_result}」の習得がスムーズになります (効果: {effect_val:.2f})。")
            
        if not readiness and not utility:
            lines.append("・基礎スキルとして推奨されます。")
            
        return "\n".join(lines)
