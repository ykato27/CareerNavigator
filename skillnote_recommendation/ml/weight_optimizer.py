"""
重み最適化モジュール

ベイズ最適化を用いて、因果推論推薦の重み（Readiness, Bayesian, Utility）を
データ駆動で自動調整します。並列処理により高速化を実現します。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import optuna
from joblib import Parallel, delayed
from sklearn.metrics import ndcg_score
import logging

from skillnote_recommendation.utils.logger import setup_logger

logger = setup_logger(__name__)


class WeightOptimizer:
    """
    推薦システムの重みパラメータを最適化するクラス

    ベイズ最適化（Optuna）と並列評価により、効率的にハイパーパラメータを探索します。
    """

    def __init__(
        self,
        recommender: Any,
        n_trials: int = 50,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Args:
            recommender: CausalGraphRecommenderインスタンス（学習済み）
            n_trials: 試行回数
            n_jobs: 並列ジョブ数（-1で全コア使用）
            random_state: 乱数シード
        """
        self.recommender = recommender
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.best_weights_: Optional[Dict[str, float]] = None
        self.best_score_: Optional[float] = None
        self.study_: Optional[optuna.Study] = None

    def optimize(
        self,
        holdout_ratio: float = 0.2,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        重みを最適化

        Args:
            holdout_ratio: 評価用にholdoutするスキルの割合
            top_k: 評価時の推薦件数

        Returns:
            最適な重み {'readiness': 0.xx, 'bayesian': 0.xx, 'utility': 0.xx}
        """
        logger.info("=" * 60)
        logger.info("重み最適化開始")
        logger.info(f"試行回数: {self.n_trials}, 並列数: {self.n_jobs}")
        logger.info("=" * 60)

        # 評価用データの準備
        train_data, test_data = self._prepare_evaluation_data(holdout_ratio)

        if not test_data:
            logger.warning("評価用データが不足しています。デフォルト重みを返します。")
            return {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}

        # Optunaで最適化
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # 目的関数を定義
        def objective(trial: optuna.Trial) -> float:
            # 重みをサンプリング（合計が1になるように制約）
            readiness_w = trial.suggest_float('readiness', 0.0, 1.0)
            bayesian_w = trial.suggest_float('bayesian', 0.0, 1.0 - readiness_w)
            utility_w = 1.0 - readiness_w - bayesian_w

            weights = {
                'readiness': readiness_w,
                'bayesian': bayesian_w,
                'utility': utility_w
            }

            # 並列評価
            score = self._evaluate_weights_parallel(weights, train_data, test_data, top_k)

            return score

        # 最適化実行
        self.study_.optimize(objective, n_trials=self.n_trials, n_jobs=1)

        # 結果の取得
        best_params = self.study_.best_params
        self.best_weights_ = {
            'readiness': best_params['readiness'],
            'bayesian': best_params['bayesian'],
            'utility': 1.0 - best_params['readiness'] - best_params['bayesian']
        }
        self.best_score_ = self.study_.best_value

        logger.info("=" * 60)
        logger.info(f"最適化完了! ベストスコア (NDCG@{top_k}): {self.best_score_:.4f}")
        logger.info(f"最適な重み: {self.best_weights_}")
        logger.info("=" * 60)

        return self.best_weights_

    def _prepare_evaluation_data(
        self,
        holdout_ratio: float
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        各メンバーの保有スキルをtrain/testに分割

        Returns:
            (train_data, test_data)
            train_data: {member_code: [skills_for_recommendation]}
            test_data: {member_code: [holdout_skills_for_evaluation]}
        """
        train_data = {}
        test_data = {}

        skill_matrix = self.recommender.skill_matrix_

        for member_code in skill_matrix.index:
            member_skills = skill_matrix.loc[member_code]
            owned_skills = member_skills[member_skills > 0].index.tolist()

            # 最低3つ以上のスキルを持っている場合のみ評価対象
            if len(owned_skills) < 3:
                continue

            # ランダムにholdout
            np.random.seed(self.random_state + hash(member_code) % 10000)
            n_holdout = max(1, int(len(owned_skills) * holdout_ratio))

            holdout_indices = np.random.choice(
                len(owned_skills),
                size=n_holdout,
                replace=False
            )

            holdout_skills = [owned_skills[i] for i in holdout_indices]
            train_skills = [s for i, s in enumerate(owned_skills) if i not in holdout_indices]

            train_data[member_code] = train_skills
            test_data[member_code] = holdout_skills

        logger.info(f"評価対象メンバー数: {len(test_data)}")

        return train_data, test_data

    def _evaluate_weights_parallel(
        self,
        weights: Dict[str, float],
        train_data: Dict[str, List[str]],
        test_data: Dict[str, List[str]],
        top_k: int
    ) -> float:
        """
        指定された重みで推薦精度を評価（並列処理）

        Returns:
            平均NDCG@K
        """
        member_codes = list(test_data.keys())

        # 並列評価
        scores = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._evaluate_single_member)(
                member_code,
                train_data[member_code],
                test_data[member_code],
                weights,
                top_k
            )
            for member_code in member_codes
        )

        # 有効なスコアのみで平均
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            return 0.0

        return np.mean(valid_scores)

    def _evaluate_single_member(
        self,
        member_code: str,
        train_skills: List[str],
        test_skills: List[str],
        weights: Dict[str, float],
        top_k: int
    ) -> Optional[float]:
        """
        単一メンバーに対する推薦精度を評価

        Returns:
            NDCG@K or None
        """
        try:
            # 推薦スコアを計算（内部で重みを適用）
            recommendations = self._recommend_with_weights(
                member_code,
                train_skills,
                weights,
                top_k=top_k * 2  # 余裕を持たせる
            )

            if not recommendations:
                return None

            # 推薦されたスキル名リスト
            rec_skills = [r['skill_name'] for r in recommendations]
            rec_scores = [r['total_score'] for r in recommendations]

            # Ground truth: test_skillsが1、それ以外が0
            true_relevance = [1 if skill in test_skills else 0 for skill in rec_skills]

            # 少なくとも1つは正解がないとNDCGが計算できない
            if sum(true_relevance) == 0:
                return None

            # NDCG@K計算
            # sklearn.metrics.ndcg_scoreは2次元配列を期待
            true_relevance_array = np.array([true_relevance[:top_k]])
            pred_scores_array = np.array([rec_scores[:top_k]])

            ndcg = ndcg_score(true_relevance_array, pred_scores_array)

            return ndcg

        except Exception as e:
            logger.debug(f"メンバー {member_code} の評価に失敗: {e}")
            return None

    def _recommend_with_weights(
        self,
        member_code: str,
        owned_skills: List[str],
        weights: Dict[str, float],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        指定された重みで推薦を実行

        （CausalGraphRecommenderのrecommendメソッドをベースに、
        　重みをパラメータ化したバージョン）
        """
        skill_matrix = self.recommender.skill_matrix_

        if member_code not in skill_matrix.index:
            return []

        # 未保有スキルのリスト
        member_skills = skill_matrix.loc[member_code]
        all_skills = member_skills.index.tolist()
        unowned_skills = [s for s in all_skills if s not in owned_skills]

        scores = []

        for target_skill in unowned_skills:
            # 1. Readiness Score
            readiness_score = 0.0
            for owned in owned_skills:
                effect = self.recommender._get_effect(owned, target_skill)
                if effect > 0.001:
                    readiness_score += effect

            # 2. Utility Score
            utility_score = 0.0
            for future in unowned_skills:
                if future == target_skill:
                    continue
                effect = self.recommender._get_effect(target_skill, future)
                if effect > 0.001:
                    utility_score += effect

            # 3. Bayesian Score
            bayesian_score = 0.0
            if self.recommender.bn_recommender:
                try:
                    bayesian_score = self.recommender.bn_recommender.predict_probability(
                        owned_skills, target_skill
                    )
                except:
                    pass

            scores.append({
                'skill_name': target_skill,
                'readiness_score': readiness_score,
                'utility_score': utility_score,
                'bayesian_score': bayesian_score
            })

        # 正規化
        if scores:
            max_readiness = max(s['readiness_score'] for s in scores)
            max_utility = max(s['utility_score'] for s in scores)

            for s in scores:
                s['readiness_score_normalized'] = (
                    s['readiness_score'] / max_readiness if max_readiness > 0 else 0.0
                )
                s['utility_score_normalized'] = (
                    s['utility_score'] / max_utility if max_utility > 0 else 0.0
                )
                s['bayesian_score_normalized'] = s['bayesian_score']

                # 指定された重みで総合スコアを計算
                s['total_score'] = (
                    s['readiness_score_normalized'] * weights['readiness'] +
                    s['bayesian_score_normalized'] * weights['bayesian'] +
                    s['utility_score_normalized'] * weights['utility']
                )

        # ソート
        scores.sort(key=lambda x: x['total_score'], reverse=True)

        return scores[:top_k]

    def get_optimization_history(self) -> pd.DataFrame:
        """
        最適化の履歴を取得

        Returns:
            DataFrame with columns: trial_number, readiness, bayesian, utility, score
        """
        if self.study_ is None:
            return pd.DataFrame()

        trials = self.study_.trials

        data = []
        for trial in trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = trial.params
                data.append({
                    'trial_number': trial.number,
                    'readiness': params['readiness'],
                    'bayesian': params['bayesian'],
                    'utility': 1.0 - params['readiness'] - params['bayesian'],
                    'score': trial.value
                })

        return pd.DataFrame(data)
