"""
推薦システム評価

時系列分割による評価とメトリクス計算を提供
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine


logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """推薦システム評価クラス"""

    def __init__(self, engine: RecommendationEngine = None):
        """
        初期化

        Args:
            engine: 推薦エンジン（オプション）
        """
        self.engine = engine

    def temporal_train_test_split(
        self,
        member_competence: pd.DataFrame,
        split_date: str = None,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        時系列による学習・評価データ分割

        Args:
            member_competence: メンバー習得力量データ（取得日カラム必須）
            split_date: 分割日（YYYY-MM-DD形式、Noneの場合はtrain_ratioで自動計算）
            train_ratio: 学習データ割合（split_dateがNoneの場合に使用）

        Returns:
            (学習データ, 評価データ)のタプル

        Raises:
            ValueError: 取得日カラムがない場合
        """
        # 取得日カラムの確認
        if '取得日' not in member_competence.columns:
            raise ValueError("メンバー習得力量データに '取得日' カラムが必要です")

        # 取得日をdatetime型に変換
        df = member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')

        # 欠損値を除外
        df = df[df['取得日_dt'].notna()].copy()

        if len(df) == 0:
            raise ValueError("有効な取得日を持つデータがありません")

        # 分割日の決定
        if split_date is None:
            # データを時系列でソートし、train_ratio位置で分割
            sorted_dates = df['取得日_dt'].sort_values()
            split_idx = int(len(sorted_dates) * train_ratio)
            split_datetime = sorted_dates.iloc[split_idx]
        else:
            split_datetime = pd.to_datetime(split_date)

        # 学習データと評価データに分割
        train_data = df[df['取得日_dt'] < split_datetime].copy()
        test_data = df[df['取得日_dt'] >= split_datetime].copy()

        # 一時カラムを削除
        train_data = train_data.drop(columns=['取得日_dt'])
        test_data = test_data.drop(columns=['取得日_dt'])

        return train_data, test_data

    def evaluate_recommendations(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        member_sample: Optional[List[str]] = None,
        similarity_data: pd.DataFrame = None,
        include_extended_metrics: bool = True
    ) -> Dict[str, float]:
        """
        推薦結果を評価

        拡張版：MRR, F1, MAP などの追加メトリクスをサポート

        Args:
            train_data: 学習データ（過去の習得力量）
            test_data: 評価データ（将来の習得力量）
            competence_master: 力量マスタ
            top_k: 推薦する上位K件
            member_sample: 評価対象メンバーリスト（Noneの場合は全メンバー）
            similarity_data: 類似度データ（Noneの場合は空のDataFrame）
            include_extended_metrics: 拡張メトリクス（MRR, F1, MAP）を計算するか

        Returns:
            評価メトリクスの辞書
        """
        # 評価対象メンバーの決定
        if member_sample is None:
            # テストデータで習得記録があるメンバー
            member_sample = test_data['メンバーコード'].unique().tolist()

        # エンジンの準備
        if self.engine is None:
            # 類似度データの準備
            if similarity_data is None:
                similarity_data = pd.DataFrame(columns=['力量1', '力量2', '類似度'])

            # メンバーマスタの準備（メンバーコードのみ）
            members_data = pd.DataFrame({
                'メンバーコード': train_data['メンバーコード'].unique()
            })

            # 推薦エンジンを作成
            engine = RecommendationEngine(
                df_members=members_data,
                df_competence_master=competence_master,
                df_member_competence=train_data,
                df_similarity=similarity_data
            )
        else:
            engine = self.engine

        # メトリクス集計用
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        f1_scores = []
        mrr_scores = []
        ap_scores = []  # Average Precision for MAP
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

            # 学習データを使って推薦を生成
            recommendations = engine.recommend(
                member_code=member_code,
                top_n=top_k
            )

            # 推薦された力量コード
            recommended_codes = [rec.competence_code for rec in recommendations]

            if len(recommended_codes) == 0:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                ndcg_scores.append(0.0)
                f1_scores.append(0.0)
                mrr_scores.append(0.0)
                ap_scores.append(0.0)
                continue

            # Precision@K: 推薦のうち実際に習得した割合
            hits = len(set(recommended_codes) & set(actual_acquired))
            precision = hits / len(recommended_codes) if len(recommended_codes) > 0 else 0.0
            precision_scores.append(precision)

            # Recall@K: 実際に習得したもののうち推薦に含まれていた割合
            recall = hits / len(actual_acquired) if len(actual_acquired) > 0 else 0.0
            recall_scores.append(recall)

            # F1-Score: PrecisionとRecallの調和平均
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)

            # NDCG@K: ランキングの質を評価
            ndcg = self._calculate_ndcg(recommended_codes, actual_acquired, top_k)
            ndcg_scores.append(ndcg)

            # Hit: 少なくとも1つ正解があったか
            if hits > 0:
                hit_count += 1

            # 拡張メトリクス
            if include_extended_metrics:
                # MRR (Mean Reciprocal Rank): 最初のヒットの逆順位
                mrr = self._calculate_mrr(recommended_codes, actual_acquired)
                mrr_scores.append(mrr)

                # Average Precision (for MAP)
                ap = self._calculate_average_precision(recommended_codes, actual_acquired)
                ap_scores.append(ap)

        # メトリクスの平均を計算
        if total_members == 0:
            base_metrics = {
                f'precision@{top_k}': 0.0,
                f'recall@{top_k}': 0.0,
                f'ndcg@{top_k}': 0.0,
                f'f1@{top_k}': 0.0,
                'hit_rate': 0.0,
                'evaluated_members': 0
            }
            if include_extended_metrics:
                base_metrics.update({
                    'mrr': 0.0,
                    f'map@{top_k}': 0.0,
                })
            return base_metrics

        base_metrics = {
            f'precision@{top_k}': np.mean(precision_scores) if precision_scores else 0.0,
            f'recall@{top_k}': np.mean(recall_scores) if recall_scores else 0.0,
            f'ndcg@{top_k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            f'f1@{top_k}': np.mean(f1_scores) if f1_scores else 0.0,
            'hit_rate': hit_count / total_members if total_members > 0 else 0.0,
            'evaluated_members': total_members
        }

        if include_extended_metrics:
            base_metrics.update({
                'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
                f'map@{top_k}': np.mean(ap_scores) if ap_scores else 0.0,
            })

        return base_metrics

    def _calculate_ndcg(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k: int
    ) -> float:
        """
        NDCG@K（Normalized Discounted Cumulative Gain）を計算

        Args:
            recommended_items: 推薦アイテムリスト（ランキング順）
            relevant_items: 関連アイテムリスト（正解）
            k: 上位K件

        Returns:
            NDCG@K スコア（0.0-1.0）
        """
        # DCG（Discounted Cumulative Gain）を計算
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                # 順位が高いほど重要（対数で割引）
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

        # IDCG（Ideal DCG）を計算
        # 全ての関連アイテムが上位にランクされた場合のDCG
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_mrr(
        self,
        recommended_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        MRR (Mean Reciprocal Rank) を計算

        最初のヒットの順位の逆数を返す

        Args:
            recommended_items: 推薦アイテムリスト（ランキング順）
            relevant_items: 関連アイテムリスト（正解）

        Returns:
            MRR スコア（0.0-1.0）
        """
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                # 最初のヒットの順位の逆数
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_average_precision(
        self,
        recommended_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        Average Precision (AP) を計算

        各関連アイテムがランク付けされた位置での精度の平均

        Args:
            recommended_items: 推薦アイテムリスト（ランキング順）
            relevant_items: 関連アイテムリスト（正解）

        Returns:
            AP スコア（0.0-1.0）
        """
        if len(relevant_items) == 0:
            return 0.0

        hits = 0
        precision_sum = 0.0

        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                hits += 1
                # このポジションまでの精度
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i

        if hits == 0:
            return 0.0

        return precision_sum / len(relevant_items)

    def cross_validate_temporal(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        n_splits: int = 5,
        top_k: int = 10
    ) -> List[Dict[str, float]]:
        """
        時系列クロスバリデーション

        データを時系列で複数のfoldに分割し、各foldで評価を実行

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ
            n_splits: 分割数
            top_k: 推薦する上位K件

        Returns:
            各foldの評価メトリクスリスト
        """
        # 取得日でソート
        df = member_competence.copy()
        df['取得日_dt'] = pd.to_datetime(df['取得日'], errors='coerce')
        df = df[df['取得日_dt'].notna()].sort_values('取得日_dt')
        df = df.drop(columns=['取得日_dt'])

        # データサイズ
        total_size = len(df)
        fold_size = total_size // (n_splits + 1)

        results = []

        for i in range(n_splits):
            # 累積的に学習データを増やし、次のfoldをテストデータとする
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size

            train_data = df.iloc[:train_end]
            test_data = df.iloc[test_start:test_end]

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # 評価実行
            metrics = self.evaluate_recommendations(
                train_data=train_data,
                test_data=test_data,
                competence_master=competence_master,
                top_k=top_k
            )

            metrics['fold'] = i + 1
            metrics['train_size'] = len(train_data)
            metrics['test_size'] = len(test_data)

            results.append(metrics)

        return results

    def print_evaluation_results(self, metrics: Dict[str, float], detailed: bool = True):
        """
        評価結果を表示（拡張版）

        Args:
            metrics: 評価メトリクスの辞書
            detailed: 詳細表示するか
        """
        logger.info("\n" + "=" * 80)
        logger.info("推薦システム評価結果")
        logger.info("=" * 80)

        logger.info("\n評価対象メンバー数: %d名", metrics.get('evaluated_members', 0))

        # K値を取得
        k = None
        for key in metrics.keys():
            if key.startswith('precision@'):
                k = key.split('@')[1]
                break

        if k:
            logger.info("\n【Top-%s 推薦の精度評価】", k)
            logger.info(
                "  Precision@%s:  %.4f  (推薦のうち正解の割合)",
                k,
                metrics.get(f'precision@{k}', 0.0),
            )
            logger.info(
                "  Recall@%s:     %.4f  (正解のうち推薦された割合)",
                k,
                metrics.get(f'recall@{k}', 0.0),
            )
            logger.info(
                "  F1@%s:         %.4f  (PrecisionとRecallの調和平均)",
                k,
                metrics.get(f'f1@{k}', 0.0),
            )
            logger.info(
                "  NDCG@%s:       %.4f  (ランキング品質)",
                k,
                metrics.get(f'ndcg@{k}', 0.0),
            )
            logger.info(
                "  Hit Rate:       %.4f  (少なくとも1つ正解があった割合)",
                metrics.get('hit_rate', 0.0),
            )

        # 拡張メトリクス
        if detailed:
            if 'mrr' in metrics:
                logger.info("\n【ランキング評価】")
                logger.info(
                    "  MRR:            %.4f  (最初のヒットの平均逆順位)",
                    metrics.get('mrr', 0.0)
                )
            if f'map@{k}' in metrics:
                logger.info(
                    "  MAP@%s:         %.4f  (Mean Average Precision)",
                    k,
                    metrics.get(f'map@{k}', 0.0)
                )

            # 多様性・カバレッジ指標
            if 'catalog_coverage' in metrics or 'coverage' in metrics:
                logger.info("\n【多様性・カバレッジ評価】")
                coverage = metrics.get('catalog_coverage', metrics.get('coverage', 0.0))
                logger.info(
                    "  Catalog Coverage: %.4f  (推薦に含まれた力量の割合)",
                    coverage
                )
                logger.info(
                    "  Unique Items:     %d個  (推薦された力量の種類)",
                    metrics.get('total_unique_recommended', 0)
                )

            if 'avg_category_diversity' in metrics:
                logger.info(
                    "  Category Div.:    %.4f  (カテゴリの多様性)",
                    metrics.get('avg_category_diversity', 0.0)
                )
                logger.info(
                    "  Type Diversity:   %.4f  (タイプの多様性)",
                    metrics.get('avg_type_diversity', 0.0)
                )

            # 高度なメトリクス
            if 'gini_index' in metrics:
                logger.info("\n【推薦の偏り評価】")
                logger.info(
                    "  Gini Index:       %.4f  (0=均等, 1=偏り大)",
                    metrics.get('gini_index', 0.0)
                )

            if 'avg_novelty' in metrics:
                logger.info(
                    "  Novelty:          %.4f  (新奇性、高いほど人気度低)",
                    metrics.get('avg_novelty', 0.0)
                )

        logger.info("\n" + "=" * 80)

    def export_evaluation_results(
        self,
        metrics: Dict[str, float],
        output_path: str
    ):
        """
        評価結果をCSVに出力

        Args:
            metrics: 評価メトリクスの辞書
            output_path: 出力ファイルパス
        """
        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info("\n評価結果を出力: %s", output_path)

    def calculate_diversity_metrics(
        self,
        recommendations_list: List[List],
        competence_master: pd.DataFrame,
        member_competence: Optional[pd.DataFrame] = None,
        include_advanced_metrics: bool = True
    ) -> Dict[str, float]:
        """
        推薦結果の多様性指標を計算（拡張版）

        追加機能：Gini Index, Novelty, User Coverage

        Args:
            recommendations_list: メンバーごとの推薦結果リスト（各要素は推薦オブジェクトのリスト）
            competence_master: 力量マスタ
            member_competence: メンバー習得力量データ（Novelty計算用、オプション）
            include_advanced_metrics: 高度なメトリクス（Gini, Novelty）を計算するか

        Returns:
            多様性指標の辞書
        """
        if not recommendations_list or len(recommendations_list) == 0:
            base_metrics = {
                'avg_category_diversity': 0.0,
                'avg_type_diversity': 0.0,
                'avg_unique_categories': 0.0,
                'avg_unique_types': 0.0,
                'catalog_coverage': 0.0,
                'total_unique_recommended': 0
            }
            if include_advanced_metrics:
                base_metrics.update({
                    'gini_index': 0.0,
                    'avg_novelty': 0.0,
                })
            return base_metrics

        category_diversities = []
        type_diversities = []
        unique_categories_list = []
        unique_types_list = []
        all_recommended_competences = set()
        competence_recommendation_counts = defaultdict(int)  # Gini Index用

        for recommendations in recommendations_list:
            if len(recommendations) == 0:
                continue

            # 推薦された力量のカテゴリとタイプを集計
            categories = set()
            types = set()

            for rec in recommendations:
                all_recommended_competences.add(rec.competence_code)
                competence_recommendation_counts[rec.competence_code] += 1

                # カテゴリとタイプを取得
                categories.add(rec.category if rec.category else 'Unknown')
                types.add(rec.competence_type)

            # カテゴリ多様性：ユニークなカテゴリ数 / 推薦数
            category_diversity = len(categories) / len(recommendations)
            category_diversities.append(category_diversity)
            unique_categories_list.append(len(categories))

            # タイプ多様性：ユニークなタイプ数 / 推薦数
            type_diversity = len(types) / len(recommendations)
            type_diversities.append(type_diversity)
            unique_types_list.append(len(types))

        # Catalog Coverage：推薦に含まれた力量の割合
        total_competences = len(competence_master)
        catalog_coverage = len(all_recommended_competences) / total_competences if total_competences > 0 else 0.0

        base_metrics = {
            'avg_category_diversity': np.mean(category_diversities) if category_diversities else 0.0,
            'avg_type_diversity': np.mean(type_diversities) if type_diversities else 0.0,
            'avg_unique_categories': np.mean(unique_categories_list) if unique_categories_list else 0.0,
            'avg_unique_types': np.mean(unique_types_list) if unique_types_list else 0.0,
            'catalog_coverage': catalog_coverage,
            'total_unique_recommended': len(all_recommended_competences),
            'total_users': len(recommendations_list)
        }

        # 高度なメトリクス
        if include_advanced_metrics:
            # Gini Index: 推薦の偏り度（0=完全に均等、1=完全に偏っている）
            gini_index = self._calculate_gini_index(competence_recommendation_counts)
            base_metrics['gini_index'] = gini_index

            # Novelty: 人気度が低いアイテムをどれだけ推薦しているか
            if member_competence is not None:
                avg_novelty = self._calculate_novelty(
                    recommendations_list,
                    member_competence
                )
                base_metrics['avg_novelty'] = avg_novelty

        return base_metrics

    def _calculate_gini_index(
        self,
        recommendation_counts: Dict[str, int]
    ) -> float:
        """
        Gini Index（ジニ係数）を計算

        推薦の偏り度を測定。0=完全に均等、1=完全に偏っている

        Args:
            recommendation_counts: 力量コード → 推薦された回数のマッピング

        Returns:
            Gini Index（0.0-1.0）
        """
        if not recommendation_counts:
            return 0.0

        # カウントを昇順にソート
        counts = sorted(recommendation_counts.values())
        n = len(counts)
        total = sum(counts)

        if total == 0:
            return 0.0

        # Gini係数の計算
        cumsum = 0
        gini_sum = 0

        for i, count in enumerate(counts):
            cumsum += count
            gini_sum += (2 * (i + 1) - n - 1) * count

        gini = gini_sum / (n * total)

        return gini

    def _calculate_novelty(
        self,
        recommendations_list: List[List],
        member_competence: pd.DataFrame
    ) -> float:
        """
        Novelty（新奇性）を計算

        人気度が低い（保有者が少ない）力量をどれだけ推薦しているか

        Args:
            recommendations_list: メンバーごとの推薦結果リスト
            member_competence: メンバー習得力量データ

        Returns:
            平均Noveltyスコア（高いほど新奇性が高い）
        """
        # 各力量の人気度（保有者数）を計算
        competence_popularity = member_competence['力量コード'].value_counts().to_dict()
        total_users = member_competence['メンバーコード'].nunique()

        novelty_scores = []

        for recommendations in recommendations_list:
            if len(recommendations) == 0:
                continue

            # 各推薦アイテムのnoveltyを計算（-log(popularity)）
            rec_novelties = []
            for rec in recommendations:
                popularity = competence_popularity.get(rec.competence_code, 0) / total_users
                # 人気度が低いほど新奇性が高い
                if popularity > 0:
                    novelty = -np.log2(popularity)
                else:
                    novelty = np.log2(total_users)  # 最大novelty
                rec_novelties.append(novelty)

            # このユーザーへの推薦の平均novelty
            if rec_novelties:
                novelty_scores.append(np.mean(rec_novelties))

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def evaluate_with_diversity(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        member_sample: Optional[List[str]] = None,
        similarity_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        推薦結果を評価（多様性指標込み）

        Args:
            train_data: 学習データ（過去の習得力量）
            test_data: 評価データ（将来の習得力量）
            competence_master: 力量マスタ
            top_k: 推薦する上位K件
            member_sample: 評価対象メンバーリスト（Noneの場合は全メンバー）
            similarity_data: 類似度データ（Noneの場合は空のDataFrame）

        Returns:
            評価メトリクス + 多様性指標の辞書
        """
        # 基本的な評価メトリクスを計算
        base_metrics = self.evaluate_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=competence_master,
            top_k=top_k,
            member_sample=member_sample,
            similarity_data=similarity_data
        )

        # 多様性計算のために推薦結果を再生成
        if member_sample is None:
            member_sample = test_data['メンバーコード'].unique().tolist()

        # エンジンの準備
        if self.engine is None:
            from skillnote_recommendation.core.recommendation_engine import RecommendationEngine

            if similarity_data is None:
                similarity_data = pd.DataFrame(columns=['力量1', '力量2', '類似度'])

            members_data = pd.DataFrame({
                'メンバーコード': train_data['メンバーコード'].unique()
            })

            engine = RecommendationEngine(
                df_members=members_data,
                df_competence_master=competence_master,
                df_member_competence=train_data,
                df_similarity=similarity_data
            )
        else:
            engine = self.engine

        # 各メンバーの推薦結果を収集
        recommendations_list = []
        for member_code in member_sample:
            actual_acquired = test_data[
                test_data['メンバーコード'] == member_code
            ]['力量コード'].unique().tolist()

            if len(actual_acquired) == 0:
                continue

            recommendations = engine.recommend(
                member_code=member_code,
                top_n=top_k
            )

            if len(recommendations) > 0:
                recommendations_list.append(recommendations)

        # 多様性指標を計算（member_competenceも渡してNovelty計算を有効化）
        diversity_metrics = self.calculate_diversity_metrics(
            recommendations_list,
            competence_master,
            member_competence=train_data,
            include_advanced_metrics=True
        )

        # 統合
        combined_metrics = {**base_metrics, **diversity_metrics}

        return combined_metrics

    def random_user_split(
        self,
        member_competence: pd.DataFrame,
        train_ratio: float = 0.8,
        min_test_items: int = 1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        ユーザーごとにランダムに力量を分割（User-based random split）

        各ユーザーの保有力量を訓練データとテストデータにランダムに分割

        Args:
            member_competence: メンバー習得力量データ
            train_ratio: 訓練データの割合（デフォルト: 0.8）
            min_test_items: テストデータに必要な最小力量数（デフォルト: 1）
            random_state: 乱数シード

        Returns:
            (学習データ, 評価データ)のタプル
        """
        np.random.seed(random_state)

        train_list = []
        test_list = []

        # ユーザーごとに処理
        for member_code in member_competence['メンバーコード'].unique():
            member_data = member_competence[
                member_competence['メンバーコード'] == member_code
            ].copy()

            n_items = len(member_data)

            # 最小テスト件数を確保できない場合はスキップ
            if n_items < min_test_items + 1:
                train_list.append(member_data)
                continue

            # ランダムシャッフル
            member_data = member_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # 分割
            n_train = max(1, int(n_items * train_ratio))
            train_data = member_data.iloc[:n_train]
            test_data = member_data.iloc[n_train:]

            # テストデータが最小件数以上あることを確認
            if len(test_data) >= min_test_items:
                train_list.append(train_data)
                test_list.append(test_data)
            else:
                # テストデータが不足する場合は全て訓練データへ
                train_list.append(member_data)

        train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

        return train_df, test_df

    def leave_one_out_split(
        self,
        member_competence: pd.DataFrame,
        random_state: int = 42
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Leave-One-Out分割

        各ユーザーの最後に取得した1つの力量をテストデータとし、
        残りを訓練データとする（時系列考慮版）

        Args:
            member_competence: メンバー習得力量データ（取得日カラム必要）
            random_state: 乱数シード（取得日がない場合のランダム選択用）

        Returns:
            (学習データ, 評価データ)のタプルのリスト
        """
        np.random.seed(random_state)

        splits = []

        # 取得日があるかチェック
        has_date = '取得日' in member_competence.columns

        # ユーザーごとに処理
        for member_code in member_competence['メンバーコード'].unique():
            member_data = member_competence[
                member_competence['メンバーコード'] == member_code
            ].copy()

            if len(member_data) < 2:
                # 力量が1つしかない場合はスキップ
                continue

            if has_date:
                # 取得日でソート
                member_data['取得日_dt'] = pd.to_datetime(member_data['取得日'], errors='coerce')
                member_data = member_data.sort_values('取得日_dt')
                member_data = member_data.drop(columns=['取得日_dt'])
            else:
                # 取得日がない場合はランダムシャッフル
                member_data = member_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # 最後の1つをテストデータ、残りを訓練データ
            train_data = member_data.iloc[:-1]
            test_data = member_data.iloc[[-1]]

            splits.append((train_data, test_data))

        return splits

    def evaluate_leave_one_out(
        self,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        max_users: Optional[int] = None,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Leave-One-Out評価を実行

        Args:
            member_competence: メンバー習得力量データ
            competence_master: 力量マスタ
            top_k: 推薦する上位K件
            max_users: 評価する最大ユーザー数（Noneの場合は全ユーザー）
            random_state: 乱数シード

        Returns:
            評価メトリクスの辞書
        """
        splits = self.leave_one_out_split(member_competence, random_state=random_state)

        if max_users is not None and len(splits) > max_users:
            # ランダムサンプリング
            np.random.seed(random_state)
            sampled_indices = np.random.choice(len(splits), max_users, replace=False)
            splits = [splits[i] for i in sampled_indices]

        # 全splitで評価を実行し、平均を取る
        all_metrics = []

        for train_data, test_data in splits:
            # 他のユーザーの訓練データも含める
            # （このユーザーのテストデータ以外の全データ）
            member_code = test_data['メンバーコード'].iloc[0]

            # 全ユーザーの訓練データを使用
            full_train = member_competence[
                ~((member_competence['メンバーコード'] == member_code) &
                  (member_competence.index.isin(test_data.index)))
            ]

            # 評価
            metrics = self.evaluate_recommendations(
                train_data=full_train,
                test_data=test_data,
                competence_master=competence_master,
                top_k=top_k,
                member_sample=[member_code]
            )

            all_metrics.append(metrics)

        # 平均を計算
        if not all_metrics:
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

        # 各メトリクスの平均
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'evaluated_members':
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = np.mean(values) if values else 0.0

        avg_metrics['evaluated_members'] = len(all_metrics)
        avg_metrics['total_splits'] = len(splits)

        return avg_metrics
