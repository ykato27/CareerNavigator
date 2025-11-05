"""
ML推薦システム（Matrix Factorization）専用の評価モジュール

NMFモデルに特化した評価機能を提供
- 再構成誤差の分析
- Hold-out評価
- 交差検証
- 各種推薦品質メトリクス
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import KFold

from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.core.evaluator import RecommendationEvaluator

logger = logging.getLogger(__name__)


class MLRecommendationEvaluator:
    """ML推薦システム専用の評価クラス"""

    def __init__(self):
        """初期化"""
        self.base_evaluator = RecommendationEvaluator()

    def evaluate_with_holdout(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_master: pd.DataFrame,
        train_ratio: float = 0.8,
        top_k: int = 10,
        use_temporal_split: bool = True,
        random_state: int = 42,
        **mf_params
    ) -> Dict[str, float]:
        """
        Hold-out評価を実行

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ
            member_master: メンバーマスタ
            train_ratio: 訓練データの割合
            top_k: 推薦する上位K件
            use_temporal_split: 時系列分割を使用するか（Falseの場合はランダム分割）
            random_state: 乱数シード
            **mf_params: MatrixFactorizationModelへのパラメータ

        Returns:
            評価メトリクスの辞書
        """
        logger.info("\n" + "=" * 80)
        logger.info("ML推薦システム Hold-out評価")
        logger.info("=" * 80)

        # データ分割
        if use_temporal_split and '取得日' in member_competence.columns:
            logger.info("時系列分割を使用")
            train_data, test_data = self.base_evaluator.temporal_train_test_split(
                member_competence,
                train_ratio=train_ratio
            )
        else:
            logger.info("ランダム分割を使用")
            train_data, test_data = self.base_evaluator.random_user_split(
                member_competence,
                train_ratio=train_ratio,
                random_state=random_state
            )

        logger.info(f"訓練データ: {len(train_data)}件")
        logger.info(f"テストデータ: {len(test_data)}件")

        # MLRecommenderを学習
        logger.info("\nモデル学習中...")
        ml_recommender = MLRecommender.build(
            member_competence=train_data,
            competence_master=competence_master,
            member_master=member_master,
            use_preprocessing=True,
            use_tuning=False
        )

        # 評価実行
        logger.info("\n評価実行中...")
        metrics = self._evaluate_ml_recommender(
            ml_recommender=ml_recommender,
            train_data=train_data,
            test_data=test_data,
            competence_master=competence_master,
            top_k=top_k
        )

        # モデル固有の情報を追加
        metrics['reconstruction_error'] = ml_recommender.mf_model.get_reconstruction_error()
        metrics['n_components'] = ml_recommender.mf_model.n_components
        metrics['n_iterations'] = ml_recommender.mf_model.model.n_iter_

        return metrics

    def evaluate_with_cross_validation(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_master: pd.DataFrame,
        n_folds: int = 5,
        top_k: int = 10,
        use_temporal: bool = True,
        random_state: int = 42,
        **mf_params
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        交差検証による評価

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ
            member_master: メンバーマスタ
            n_folds: 分割数
            top_k: 推薦する上位K件
            use_temporal: 時系列分割を使用するか
            random_state: 乱数シード
            **mf_params: MatrixFactorizationModelへのパラメータ

        Returns:
            (平均メトリクス, 各foldのメトリクスリスト)
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"ML推薦システム {n_folds}-Fold 交差検証")
        logger.info("=" * 80)

        fold_metrics = []

        if use_temporal and '取得日' in member_competence.columns:
            # 時系列交差検証
            df = member_competence.copy()
            df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')
            df = df[df['取得日_dt'].notna()].sort_values('取得日_dt')
            df = df.drop(columns=['取得日_dt'])

            total_size = len(df)
            fold_size = total_size // (n_folds + 1)

            for i in range(n_folds):
                logger.info(f"\n--- Fold {i+1}/{n_folds} ---")
                train_end = (i + 1) * fold_size
                test_start = train_end
                test_end = test_start + fold_size

                train_data = df.iloc[:train_end]
                test_data = df.iloc[test_start:test_end]

                if len(train_data) == 0 or len(test_data) == 0:
                    continue

                # モデル学習と評価
                metrics = self._train_and_evaluate_fold(
                    train_data, test_data, competence_master,
                    member_master, top_k, i + 1
                )
                fold_metrics.append(metrics)
        else:
            # ランダムK-Fold
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

            for i, (train_idx, test_idx) in enumerate(kfold.split(member_competence)):
                logger.info(f"\n--- Fold {i+1}/{n_folds} ---")
                train_data = member_competence.iloc[train_idx]
                test_data = member_competence.iloc[test_idx]

                metrics = self._train_and_evaluate_fold(
                    train_data, test_data, competence_master,
                    member_master, top_k, i + 1
                )
                fold_metrics.append(metrics)

        # 平均メトリクスを計算
        avg_metrics = self._average_metrics(fold_metrics)
        avg_metrics['n_folds'] = len(fold_metrics)

        logger.info("\n" + "=" * 80)
        logger.info("交差検証完了")
        logger.info("=" * 80)

        return avg_metrics, fold_metrics

    def _train_and_evaluate_fold(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_master: pd.DataFrame,
        top_k: int,
        fold_num: int
    ) -> Dict[str, float]:
        """
        1つのfoldでモデルを学習して評価

        Args:
            train_data: 訓練データ
            test_data: テストデータ
            competence_master: 力量マスタ
            member_master: メンバーマスタ
            top_k: 推薦数
            fold_num: Fold番号

        Returns:
            評価メトリクス
        """
        # モデル学習
        ml_recommender = MLRecommender.build(
            member_competence=train_data,
            competence_master=competence_master,
            member_master=member_master,
            use_preprocessing=True,
            use_tuning=False
        )

        # 評価
        metrics = self._evaluate_ml_recommender(
            ml_recommender=ml_recommender,
            train_data=train_data,
            test_data=test_data,
            competence_master=competence_master,
            top_k=top_k
        )

        metrics['fold'] = fold_num
        metrics['train_size'] = len(train_data)
        metrics['test_size'] = len(test_data)
        metrics['reconstruction_error'] = ml_recommender.mf_model.get_reconstruction_error()

        logger.info(f"Fold {fold_num} 完了: Precision@{top_k}={metrics.get(f'precision@{top_k}', 0):.4f}")

        return metrics

    def _evaluate_ml_recommender(
        self,
        ml_recommender: MLRecommender,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int
    ) -> Dict[str, float]:
        """
        MLRecommenderを評価

        Args:
            ml_recommender: 学習済みMLRecommender
            train_data: 訓練データ
            test_data: テストデータ
            competence_master: 力量マスタ
            top_k: 推薦数

        Returns:
            評価メトリクス
        """
        # 評価対象メンバー
        member_sample = test_data['メンバーコード'].unique().tolist()

        # メトリクス集計用
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        f1_scores = []
        mrr_scores = []
        ap_scores = []
        hit_count = 0
        total_members = 0

        for member_code in member_sample:
            # テストデータでの習得力量（正解データ）
            actual_acquired = test_data[
                test_data['メンバーコード'] == member_code
            ]['力量コード'].unique().tolist()

            if len(actual_acquired) == 0:
                continue

            total_members += 1

            try:
                # 推薦生成
                recommendations = ml_recommender.recommend(
                    member_code=member_code,
                    top_n=top_k,
                    use_diversity=False  # 純粋な精度評価のため多様性オフ
                )

                recommended_codes = [rec.competence_code for rec in recommendations]

                if len(recommended_codes) == 0:
                    precision_scores.append(0.0)
                    recall_scores.append(0.0)
                    ndcg_scores.append(0.0)
                    f1_scores.append(0.0)
                    mrr_scores.append(0.0)
                    ap_scores.append(0.0)
                    continue

                # メトリクス計算
                hits = len(set(recommended_codes) & set(actual_acquired))
                precision = hits / len(recommended_codes)
                recall = hits / len(actual_acquired)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                # NDCG, MRR, AP
                ndcg = self.base_evaluator._calculate_ndcg(recommended_codes, actual_acquired, top_k)
                mrr = self.base_evaluator._calculate_mrr(recommended_codes, actual_acquired)
                ap = self.base_evaluator._calculate_average_precision(recommended_codes, actual_acquired)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)
                ap_scores.append(ap)

                if hits > 0:
                    hit_count += 1

            except Exception as e:
                logger.warning(f"メンバー {member_code} の評価でエラー: {e}")
                continue

        # 平均を計算
        if total_members == 0:
            return {
                f'precision@{top_k}': 0.0,
                f'recall@{top_k}': 0.0,
                f'ndcg@{top_k}': 0.0,
                f'f1@{top_k}': 0.0,
                'hit_rate': 0.0,
                'mrr': 0.0,
                f'map@{top_k}': 0.0,
                'evaluated_members': 0
            }

        return {
            f'precision@{top_k}': np.mean(precision_scores),
            f'recall@{top_k}': np.mean(recall_scores),
            f'ndcg@{top_k}': np.mean(ndcg_scores),
            f'f1@{top_k}': np.mean(f1_scores),
            'hit_rate': hit_count / total_members,
            'mrr': np.mean(mrr_scores),
            f'map@{top_k}': np.mean(ap_scores),
            'evaluated_members': total_members
        }

    def _average_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        複数のメトリクスの平均を計算

        Args:
            metrics_list: メトリクスのリスト

        Returns:
            平均メトリクス
        """
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key in ['fold', 'train_size', 'test_size']:
                continue
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)

        return avg_metrics

    def print_ml_evaluation_results(
        self,
        metrics: Dict[str, float],
        fold_metrics: Optional[List[Dict[str, float]]] = None
    ):
        """
        ML推薦システムの評価結果を表示

        Args:
            metrics: 評価メトリクス
            fold_metrics: 各foldのメトリクス（交差検証の場合）
        """
        logger.info("\n" + "=" * 80)
        logger.info("ML推薦システム 評価結果")
        logger.info("=" * 80)

        # モデル情報
        if 'n_components' in metrics:
            logger.info("\n【モデル情報】")
            logger.info(f"  潜在因子数:       {metrics['n_components']}")
            logger.info(f"  イテレーション数: {metrics.get('n_iterations', 'N/A')}")
            logger.info(f"  再構成誤差:       {metrics.get('reconstruction_error', 0):.6f}")

        # 評価メトリクス
        self.base_evaluator.print_evaluation_results(metrics, detailed=True)

        # 交差検証の結果
        if fold_metrics and len(fold_metrics) > 1:
            logger.info("\n【交差検証の詳細】")
            logger.info(f"  分割数: {len(fold_metrics)}")

            # K値を取得
            k = None
            for key in metrics.keys():
                if key.startswith('precision@'):
                    k = key.split('@')[1]
                    break

            if k:
                logger.info(f"\n  Precision@{k} (各Fold):")
                for i, m in enumerate(fold_metrics):
                    logger.info(f"    Fold {i+1}: {m.get(f'precision@{k}', 0):.4f}")
                logger.info(f"  平均: {metrics.get(f'precision@{k}', 0):.4f} " +
                           f"(±{metrics.get(f'precision@{k}_std', 0):.4f})")

        logger.info("\n" + "=" * 80)
