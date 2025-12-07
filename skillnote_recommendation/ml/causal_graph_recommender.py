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
        learner_params: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            member_competence: メンバー力量データ
            competence_master: 力量マスタ
            learner_params: CausalStructureLearnerのパラメータ
            weights: スコア重み {'readiness': 0.6, 'utility': 0.4}（2軸スコアリング）
        """
        self.member_competence = member_competence
        self.competence_master = competence_master

        params = learner_params or {}
        # デフォルトパラメータをconfigから取得してマージすることも可能
        if 'random_state' not in params:
            params['random_state'] = config.model.RANDOM_STATE

        self.learner = CausalStructureLearner(**params)
        self.bn_recommender: Optional[BayesianNetworkRecommender] = None

        # 重みのデフォルト値（2軸: Readiness 60%, Utility 40%）
        self.weights = weights or {'readiness': 0.6, 'utility': 0.4, 'bayesian': 0.0}

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
        
        # 1. データ前処理: メンバー×スキルのマトリックス作成
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

        # 全スキルから未習得スキルへの正の因果効果の合計を事前計算（Utility正規化用）
        total_effects_to_unowned = 0.0
        for source_skill in self.skill_matrix_.columns:
            for future_skill in unowned_skills:
                effect = self._get_effect(source_skill, future_skill)
                if effect > 0.001:
                    total_effects_to_unowned += effect

        scores = []

        for target_skill in unowned_skills:
            # 1. Readiness Score: 正の因果効果を持つ因子スキルのうち、保有している割合
            # 正の因果効果のみを因子としてカウント（負の効果は除外）
            total_positive_effects = 0.0
            owned_positive_effects = 0.0
            readiness_reasons = []

            # target_skillへの正の因果効果を持つスキル（因子）を集計
            for skill in self.skill_matrix_.columns:
                effect = self._get_effect(skill, target_skill)
                # 正の因果効果のみを因子として扱う（閾値: 0.001）
                if effect > 0.001:
                    total_positive_effects += effect

                    # 保有スキルの場合
                    if skill in owned_skills:
                        owned_positive_effects += effect
                        readiness_reasons.append((skill, effect))

            # Readiness Score = 保有因子からの効果 / 全因子からの効果
            # 正の効果のみを対象にすることで、100%を超えることがなくなる
            if total_positive_effects > 0.001:
                readiness_score = owned_positive_effects / total_positive_effects
            else:
                # 因子がない場合
                readiness_score = 0.0
            
            # 2. Utility Score: ターゲットから未習得スキルへの効果 / 全スキルから未習得スキルへの効果
            # Readinessと対称的なロジック（割合ベース）
            target_effects_to_unowned = 0.0
            utility_reasons = []

            for future in unowned_skills:
                if future == target_skill:
                    continue
                # target が future に与える影響
                effect = self._get_effect(target_skill, future)
                # 正の因果効果のみ
                if effect > 0.001:
                    target_effects_to_unowned += effect
                    utility_reasons.append((future, effect))

            # Utility Score = このスキルから未習得への効果 / 全スキルから未習得への効果
            if total_effects_to_unowned > 0.001:
                utility_score = target_effects_to_unowned / total_effects_to_unowned
            else:
                utility_score = 0.0
            
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
            
            # Readiness Scoreが0のスキルは除外（ユーザー固有性を確保）
            # ただし、保有スキルが少ない場合は例外的に含める
            min_readiness = 0.0 if len(owned_skills) < 3 else 0.001

            # 仮の総合スコアでフィルタリング（正規化前）
            temp_total_score = (
                readiness_score * self.weights['readiness'] +
                bayesian_score * self.weights['bayesian'] +
                utility_score * self.weights['utility']
            )

            if temp_total_score > 0 and readiness_score >= min_readiness:
                scores.append({
                    'skill_name': target_skill,
                    'skill_code': self.name_to_code.get(target_skill, target_skill),
                    'total_score': 0.0,  # 正規化後に再計算
                    'readiness_score': readiness_score,
                    'utility_score': utility_score,
                    'bayesian_score': bayesian_score,
                    'readiness_reasons': sorted(readiness_reasons, key=lambda x: x[1], reverse=True),
                    'utility_reasons': sorted(utility_reasons, key=lambda x: x[1], reverse=True)
                })

        # スコアの正規化
        # Readiness, Utility, Bayesian は全て既に0〜1の範囲（割合ベース）
        if scores:
            for s in scores:
                # Readiness: 既に0〜1（保有因子効果 / 全因子効果）
                s['readiness_score_normalized'] = s['readiness_score']

                # Utility: 既に0〜1（ターゲット→未習得への効果 / 全スキル→未習得への効果）
                s['utility_score_normalized'] = s['utility_score']

                # Bayesian: 既に0〜1
                s['bayesian_score_normalized'] = s['bayesian_score']

                # 総合スコアを計算
                s['total_score'] = (
                    s['readiness_score_normalized'] * self.weights['readiness'] +
                    s['bayesian_score_normalized'] * self.weights['bayesian'] +
                    s['utility_score_normalized'] * self.weights['utility']
                )

        # ソート
        scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 上位N件を整形して返す
        results = []
        for item in scores[:top_n]:
            explanation = self._generate_explanation(item)
            
            # competence_masterからカテゴリを取得
            skill_code = item['skill_code']
            category = ''
            if not self.competence_master.empty:
                mask = self.competence_master['力量コード'] == skill_code
                if mask.any():
                    category = str(self.competence_master.loc[mask, '力量カテゴリー名'].iloc[0])
            
            results.append({
                'skill_code': skill_code,
                'skill_name': item['skill_name'],
                'category': category,
                'readiness_score': item['readiness_score_normalized'],
                'probability_score': item['bayesian_score_normalized'],
                'utility_score': item['utility_score_normalized'],
                'final_score': item['total_score'],
                'reason': explanation,
                'dependencies': []  # TODO: 依存関係を返す場合は実装
            })
            
        return results

    def get_effect(self, cause: str, effect: str) -> float:
        """
        因果効果を取得 (公開メソッド)
        
        Args:
            cause: 原因スキル名
            effect: 結果スキル名
            
        Returns:
            因果効果（存在しない場合は0.0）
        """
        if self.total_effects_ is None:
            return 0.0

        return self.total_effects_.get(cause, {}).get(effect, 0.0)
    
    def _get_effect(self, cause: str, effect: str) -> float:
        """因果効果を取得 (内部用、後方互換性のため残す)"""
        return self.get_effect(cause, effect)

    def get_score_for_skill(
        self,
        member_code: str,
        skill_code: str
    ) -> Dict[str, Any]:
        """
        特定スキルに対するCausalスコアを取得
        
        Args:
            member_code: メンバーコード
            skill_code: 力量コード
            
        Returns:
            {
                'readiness': float (0-1),  # 正規化済み準備完了度
                'bayesian': float (0-1),   # ベイジアン確率
                'utility': float (0-1),    # 正規化済み有用性
                'total_score': float (0-1), # 総合スコア
                'readiness_reasons': [(skill_name, effect), ...],
                'utility_reasons': [(skill_name, effect), ...]
            }
        """
        if not self.is_fitted:
            logger.warning("モデルが学習されていません")
            return {
                'readiness': 0.0,
                'bayesian': 0.0,
                'utility': 0.0,
                'total_score': 0.0,
                'readiness_reasons': [],
                'utility_reasons': []
            }
        
        # メンバーの保有スキル取得
        if member_code not in self.skill_matrix_.index:
            logger.warning(f"メンバー {member_code} がデータに存在しません")
            return {
                'readiness': 0.0,
                'bayesian': 0.0,
                'utility': 0.0,
                'total_score': 0.0,
                'readiness_reasons': [],
                'utility_reasons': []
            }
        
        # スキルコードからスキル名に変換
        skill_name = self.code_to_name.get(skill_code, skill_code)
        
        # 保有スキルと未習得スキルを取得
        member_skills = self.skill_matrix_.loc[member_code]
        owned_skills = member_skills[member_skills > 0].index.tolist()
        
        # 既に保有しているスキルの場合
        if skill_name in owned_skills:
            logger.debug(f"スキル {skill_name} は既に保有済み")
            return {
                'readiness': 1.0,
                'bayesian': 1.0,
                'utility': 1.0,
                'total_score': 1.0,
                'readiness_reasons': [("既に習得済み", 1.0)],
                'utility_reasons': []
            }
        
        # スキル名がskill_matrixに存在しない場合
        if skill_name not in self.skill_matrix_.columns:
            logger.warning(f"スキル {skill_name} がスキルマトリックスに存在しません")
            return {
                'readiness': 0.0,
                'bayesian': 0.0,
                'utility': 0.0,
                'total_score': 0.0,
                'readiness_reasons': [],
                'utility_reasons': []
            }
        
        # --- Causalスコア計算（新ロジック）---

        # 1. Readiness Score: 正の因果効果を持つ因子スキルのうち、保有している割合
        # 正の因果効果のみを因子としてカウント（負の効果は除外）
        total_positive_effects = 0.0
        owned_positive_effects = 0.0
        readiness_reasons = []

        # target_skillへの正の因果効果を持つスキル（因子）を集計
        for skill in self.skill_matrix_.columns:
            effect = self._get_effect(skill, skill_name)
            # 正の因果効果のみを因子として扱う（閾値: 0.001）
            if effect > 0.001:
                total_positive_effects += effect

                if skill in owned_skills:
                    owned_positive_effects += effect
                    readiness_reasons.append((skill, effect))

        # Readiness Score = 保有因子からの効果 / 全因子からの効果
        # 正の効果のみを対象にすることで、100%を超えることがなくなる
        if total_positive_effects > 0.001:
            readiness_score = owned_positive_effects / total_positive_effects
        else:
            readiness_score = 0.0
        
        # 2. Utility Score: ターゲットから未習得スキルへの効果 / 全スキルから未習得スキルへの効果
        unowned_skills = member_skills[member_skills == 0].index.tolist()

        # 全スキルから未習得スキルへの正の因果効果の合計を計算
        total_effects_to_unowned = 0.0
        for source_skill in self.skill_matrix_.columns:
            for future in unowned_skills:
                effect = self._get_effect(source_skill, future)
                if effect > 0.001:
                    total_effects_to_unowned += effect

        # このスキルから未習得スキルへの正の因果効果の合計を計算
        target_effects_to_unowned = 0.0
        utility_reasons = []
        for future in unowned_skills:
            if future == skill_name:
                continue
            effect = self._get_effect(skill_name, future)
            if effect > 0.001:
                target_effects_to_unowned += effect
                utility_reasons.append((future, effect))

        # Utility Score = このスキルから未習得への効果 / 全スキルから未習得への効果
        if total_effects_to_unowned > 0.001:
            utility_score = target_effects_to_unowned / total_effects_to_unowned
        else:
            utility_score = 0.0
        
        # 3. Bayesian Score: P(Target=1 | Owned)
        bayesian_score = 0.0
        if self.bn_recommender:
            try:
                bayesian_score = self.bn_recommender.predict_probability(owned_skills, skill_name)
            except Exception as e:
                logger.debug(f"ベイジアン推論エラー ({skill_name}): {e}")
        
        # スコアの正規化
        # Readiness, Utility, Bayesian は全て既に0〜1の範囲（割合ベース）
        readiness_normalized = readiness_score
        utility_normalized = utility_score
        bayesian_normalized = bayesian_score
        
        # 総合スコア
        total_score = (
            readiness_normalized * self.weights['readiness'] +
            bayesian_normalized * self.weights['bayesian'] +
            utility_normalized * self.weights['utility']
        )
        
        # 推薦理由をソート
        readiness_reasons.sort(key=lambda x: x[1], reverse=True)
        utility_reasons.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"スキル {skill_name}: total={total_score:.3f}, readiness={readiness_normalized:.3f}, bayesian={bayesian_normalized:.3f}, utility={utility_normalized:.3f}")
        
        return {
            'readiness': readiness_normalized,
            'bayesian': bayesian_normalized,
            'utility': utility_normalized,
            'total_score': total_score,
            'readiness_reasons': readiness_reasons[:5],  # 上位5件
            'utility_reasons': utility_reasons[:5]  # 上位5件
        }
    
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

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        推薦スコアの重みを設定

        Args:
            weights: {'readiness': float, 'bayesian': float, 'utility': float}
                     合計が1.0になることを推奨
        """
        # 合計が1.0でない場合は正規化
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"重みの合計が1.0ではありません ({total:.4f})。正規化します。")
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = weights

        logger.info(f"重みを更新しました: {self.weights}")

    def get_weights(self) -> Dict[str, float]:
        """
        現在の重みを取得

        Returns:
            {'readiness': float, 'bayesian': float, 'utility': float}
        """
        return self.weights.copy()

    def optimize_weights(
        self,
        n_trials: int = 50,
        n_jobs: int = -1,
        holdout_ratio: float = 0.2,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        ベイズ最適化により最適な重みを自動探索

        Args:
            n_trials: 試行回数
            n_jobs: 並列ジョブ数（-1で全コア使用）
            holdout_ratio: 評価用データの割合
            top_k: 評価時の推薦件数

        Returns:
            最適な重み
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。fit()を実行してください。")

        from skillnote_recommendation.ml.weight_optimizer import WeightOptimizer

        optimizer = WeightOptimizer(
            recommender=self,
            n_trials=n_trials,
            n_jobs=n_jobs
        )

        best_weights = optimizer.optimize(
            holdout_ratio=holdout_ratio,
            top_k=top_k
        )

        # 最適な重みを自動的に設定
        self.set_weights(best_weights)

        return best_weights

    def plot_readiness_utility_scatter(
        self,
        member_code: str,
        top_n: Optional[int] = None,
        highlight_top_n: int = 5,
        width: int = 800,
        height: int = 600
    ) -> 'go.Figure':
        """
        Readiness（準備度）× Utility（将来性）の散布図を作成

        横軸: Readiness（このスキルを習得する準備がどれだけできているか）
        縦軸: Utility（このスキルが将来どれだけ役立つか）

        Args:
            member_code: メンバーコード
            top_n: 表示するスキル数（Noneで全て）
            highlight_top_n: ハイライトする上位スキル数
            width: グラフの幅
            height: グラフの高さ

        Returns:
            Plotly Figure オブジェクト
        """
        import plotly.graph_objects as go

        if not self.is_fitted:
            logger.warning("モデルが学習されていません")
            fig = go.Figure()
            fig.update_layout(title="モデルが学習されていません")
            return fig

        # 推薦結果を取得（全スキル）
        all_recommendations = self.recommend(member_code, top_n=top_n or 1000)

        if not all_recommendations:
            fig = go.Figure()
            fig.update_layout(title="推薦結果がありません")
            return fig

        # データを抽出
        skill_names = [r['skill_name'] for r in all_recommendations]
        readiness_scores = [r['readiness_score'] * 100 for r in all_recommendations]  # %表示
        utility_scores = [r['utility_score'] * 100 for r in all_recommendations]  # %表示
        final_scores = [r['final_score'] for r in all_recommendations]

        # 上位N件をハイライト
        top_indices = set(range(min(highlight_top_n, len(all_recommendations))))

        colors = []
        sizes = []
        for i in range(len(all_recommendations)):
            if i in top_indices:
                colors.append('#E24A4A')  # 赤: 上位推薦
                sizes.append(15)
            else:
                colors.append('#4A90E2')  # 青: その他
                sizes.append(8)

        # 散布図を作成
        fig = go.Figure()

        # その他のスキル（先に描画）
        other_indices = [i for i in range(len(skill_names)) if i not in top_indices]
        if other_indices:
            fig.add_trace(go.Scatter(
                x=[readiness_scores[i] for i in other_indices],
                y=[utility_scores[i] for i in other_indices],
                mode='markers',
                name='その他のスキル',
                marker=dict(
                    size=8,
                    color='#4A90E2',
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=[skill_names[i] for i in other_indices],
                customdata=[[final_scores[i]] for i in other_indices],
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    '準備度: %{x:.1f}%<br>'
                    '将来性: %{y:.1f}%<br>'
                    '総合スコア: %{customdata[0]:.3f}<br>'
                    '<extra></extra>'
                )
            ))

        # 上位スキル（後に描画して前面に）
        top_indices_list = list(top_indices)
        if top_indices_list:
            fig.add_trace(go.Scatter(
                x=[readiness_scores[i] for i in top_indices_list],
                y=[utility_scores[i] for i in top_indices_list],
                mode='markers+text',
                name=f'上位{highlight_top_n}推薦',
                marker=dict(
                    size=15,
                    color='#E24A4A',
                    line=dict(width=2, color='white')
                ),
                text=[skill_names[i] for i in top_indices_list],
                textposition='top center',
                textfont=dict(size=10),
                customdata=[[final_scores[i]] for i in top_indices_list],
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    '準備度: %{x:.1f}%<br>'
                    '将来性: %{y:.1f}%<br>'
                    '総合スコア: %{customdata[0]:.3f}<br>'
                    '<extra></extra>'
                )
            ))

        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=f'スキル推薦マップ: Readiness × Utility<br>'
                     f'<sub>メンバー: {member_code} | 総スキル数: {len(skill_names)}</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Readiness（準備度）%',
                range=[-5, 105],
                gridcolor='lightgray',
                zerolinecolor='gray'
            ),
            yaxis=dict(
                title='Utility（将来性）%',
                range=[-5, 105],
                gridcolor='lightgray',
                zerolinecolor='gray'
            ),
            width=width,
            height=height,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            ),
            plot_bgcolor='white',
            # 四象限の説明を追加
            annotations=[
                dict(
                    x=90, y=90,
                    text='🎯 最優先<br>(準備OK・将来性高)',
                    showarrow=False,
                    font=dict(size=10, color='green'),
                    bgcolor='rgba(144, 238, 144, 0.3)'
                ),
                dict(
                    x=10, y=90,
                    text='📚 基盤構築が必要<br>(準備不足・将来性高)',
                    showarrow=False,
                    font=dict(size=10, color='orange'),
                    bgcolor='rgba(255, 200, 100, 0.3)'
                ),
                dict(
                    x=90, y=10,
                    text='✅ すぐ習得可能<br>(準備OK・将来性低)',
                    showarrow=False,
                    font=dict(size=10, color='blue'),
                    bgcolor='rgba(173, 216, 230, 0.3)'
                ),
                dict(
                    x=10, y=10,
                    text='⏸️ 後回し<br>(準備不足・将来性低)',
                    showarrow=False,
                    font=dict(size=10, color='gray'),
                    bgcolor='rgba(200, 200, 200, 0.3)'
                )
            ]
        )

        return fig
