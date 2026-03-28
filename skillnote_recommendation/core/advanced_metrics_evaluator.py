"""
Advanced Metrics Evaluator

GAFAレベルの包括的な評価システム:
- バイアス評価（Popularity bias, Position bias, Filter bubble）
- 多様性・カバレッジ評価（ILD, Serendipity, Personalization, Calibration）
- 包括的評価レポート

問題意識:
- オフライン評価のみ（Precision, Recall, NDCG）ではビジネスメトリクスとの相関が不明
- バイアス評価がないと、人気スキル過剰推薦やフィルターバブルに陥る
- 多様性が不十分だとユーザー満足度が低下

Reference:
- Hu et al. (2008): Collaborative Filtering for Implicit Feedback Datasets
- Vargas & Castells (2011): Rank and Relevance in Novelty and Diversity Metrics
- Abdollahpouri et al. (2019): Managing Popularity Bias in Recommender Systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import warnings

from .evaluator import RecommendationEvaluator


@dataclass
class ComprehensiveEvaluationResult:
    """包括的評価結果

    Attributes:
        accuracy_metrics: 精度評価（Precision, Recall, NDCG等）
        bias_metrics: バイアス評価（Popularity, Position, Filter bubble）
        diversity_metrics: 多様性評価（ILD, Serendipity, Personalization）
        coverage_metrics: カバレッジ評価（Coverage, Catalog Coverage）
        calibration_metrics: キャリブレーション評価（KL divergence）
        overall_score: 総合スコア（重み付き平均）
    """
    accuracy_metrics: Dict[str, float]
    bias_metrics: Dict[str, float]
    diversity_metrics: Dict[str, float]
    coverage_metrics: Dict[str, float]
    calibration_metrics: Dict[str, float]
    overall_score: float


class AdvancedMetricsEvaluator(RecommendationEvaluator):
    """高度な評価メトリクスを提供するクラス

    RecommendationEvaluatorを拡張し、以下の機能を追加:
    1. バイアス評価（Popularity, Position, Filter bubble）
    2. 多様性評価（ILD, Serendipity, Personalization）
    3. キャリブレーション評価（KL divergence）
    4. 包括的評価レポート

    Usage:
        >>> evaluator = AdvancedMetricsEvaluator()
        >>> result = evaluator.comprehensive_evaluation(
        ...     train_data=train_df,
        ...     test_data=test_df,
        ...     competence_master=master_df,
        ...     top_k=10
        ... )
        >>> print(result.overall_score)
        >>> evaluator.print_comprehensive_report(result)
    """

    def evaluate_popularity_bias(
        self,
        recommendations_list: List[List],
        member_competence: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Popularity Bias（人気スキル過剰推薦）を評価

        人気スキルばかり推薦していないかを評価。
        健全な推薦システムはロングテールスキルも推薦すべき。

        Args:
            recommendations_list: メンバーごとの推薦結果リスト
            member_competence: メンバー習得力量データ（人気度計算用）

        Returns:
            {
                'popularity_bias': float,  # 推薦スキルの平均人気度（0-1、高いほどバイアス大）
                'tail_ratio': float,  # ロングテール推薦率（0-1、高いほど健全）
                'gini_coefficient': float,  # 推薦分布のGini係数（0-1、高いほど偏り大）
            }

        Reference:
            Abdollahpouri et al. (2019): Managing Popularity Bias in Recommender Systems
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return {
                'popularity_bias': 0.0,
                'tail_ratio': 0.0,
                'gini_coefficient': 0.0,
            }

        # 各スキルの人気度を計算（保有者数）
        competence_counts = member_competence['力量コード'].value_counts()
        total_members = member_competence['メンバーコード'].nunique()

        # 人気度を正規化（0-1）
        competence_popularity = {
            code: count / total_members
            for code, count in competence_counts.items()
        }

        # ロングテール定義（人気度が中央値以下）
        popularity_values = list(competence_popularity.values())
        median_popularity = np.median(popularity_values) if popularity_values else 0.0

        # 推薦された各スキルの人気度を集計
        recommended_popularities = []
        tail_count = 0
        total_recommendations = 0

        for recommendations in recommendations_list:
            for rec in recommendations:
                popularity = competence_popularity.get(rec.competence_code, 0.0)
                recommended_popularities.append(popularity)
                total_recommendations += 1

                # ロングテールスキルか判定
                if popularity <= median_popularity:
                    tail_count += 1

        # Popularity Bias: 推薦スキルの平均人気度（高いほどバイアス大）
        avg_popularity = np.mean(recommended_popularities) if recommended_popularities else 0.0

        # Tail Ratio: ロングテール推薦率（高いほど健全）
        tail_ratio = tail_count / total_recommendations if total_recommendations > 0 else 0.0

        # Gini Coefficient: 推薦分布の偏り
        # 既存のGini計算を流用
        recommendation_counts = Counter([rec.competence_code for recs in recommendations_list for rec in recs])
        gini = self._calculate_gini_index(recommendation_counts)

        return {
            'popularity_bias': avg_popularity,
            'tail_ratio': tail_ratio,
            'gini_coefficient': gini,
        }

    def evaluate_position_bias(
        self,
        recommendations_list: List[List],
        actual_acquired_list: List[List[str]],
    ) -> Dict[str, float]:
        """
        Position Bias（ランキング位置バイアス）を評価

        ランキング上位ほどクリックされる傾向（Position bias）を補正せずに
        評価すると、実際のユーザー行動を過大評価する。

        Args:
            recommendations_list: メンバーごとの推薦結果リスト
            actual_acquired_list: メンバーごとの実際の習得力量リスト

        Returns:
            {
                'position_bias_score': float,  # 位置バイアススコア（0-1、高いほどバイアス大）
                'top3_hit_ratio': float,  # Top-3ヒット率
                'bottom_half_hit_ratio': float,  # 下半分ヒット率
                'bias_ratio': float,  # Top-3 / Bottom-half の比率（高いほどバイアス大）
            }

        Reference:
            Joachims et al. (2017): Unbiased Learning-to-Rank with Biased Feedback
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return {
                'position_bias_score': 0.0,
                'top3_hit_ratio': 0.0,
                'bottom_half_hit_ratio': 0.0,
                'bias_ratio': 0.0,
            }

        top3_hits = 0
        bottom_half_hits = 0
        top3_total = 0
        bottom_half_total = 0

        for recommendations, actual_acquired in zip(recommendations_list, actual_acquired_list):
            if len(recommendations) == 0:
                continue

            actual_set = set(actual_acquired)
            k = len(recommendations)
            half_k = k // 2

            # Top-3のヒット
            for i, rec in enumerate(recommendations[:3]):
                top3_total += 1
                if rec.competence_code in actual_set:
                    top3_hits += 1

            # 下半分のヒット
            for i, rec in enumerate(recommendations[half_k:]):
                bottom_half_total += 1
                if rec.competence_code in actual_set:
                    bottom_half_hits += 1

        # Top-3ヒット率
        top3_hit_ratio = top3_hits / top3_total if top3_total > 0 else 0.0

        # 下半分ヒット率
        bottom_half_hit_ratio = bottom_half_hits / bottom_half_total if bottom_half_total > 0 else 0.0

        # バイアス比率（Top-3 / Bottom-half）
        # 理想的には1.0に近い（位置バイアスなし）
        # 大きいほど位置バイアスが強い
        if bottom_half_hit_ratio > 0:
            bias_ratio = top3_hit_ratio / bottom_half_hit_ratio
        else:
            bias_ratio = 0.0

        # 位置バイアススコア（0-1、高いほどバイアス大）
        # bias_ratioを正規化（1.0が理想、大きいほどバイアス大）
        if bias_ratio >= 1.0:
            position_bias_score = min(1.0, (bias_ratio - 1.0) / 4.0)  # 5倍以上で1.0
        else:
            position_bias_score = 0.0

        return {
            'position_bias_score': position_bias_score,
            'top3_hit_ratio': top3_hit_ratio,
            'bottom_half_hit_ratio': bottom_half_hit_ratio,
            'bias_ratio': bias_ratio,
        }

    def evaluate_filter_bubble(
        self,
        train_data: pd.DataFrame,
        recommendations_list: List[List],
        competence_master: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Filter Bubble（フィルターバブル）を評価

        既存スキルと似たものばかり推薦していないかを評価。
        健全な推薦システムは新しいカテゴリーのスキルも提案すべき。

        Args:
            train_data: 訓練データ（既習得スキル）
            recommendations_list: メンバーごとの推薦結果リスト
            competence_master: 力量マスタ（カテゴリー情報）

        Returns:
            {
                'filter_bubble_score': float,  # フィルターバブルスコア（0-1、高いほど問題）
                'avg_category_overlap': float,  # カテゴリー重複率（0-1）
                'avg_type_overlap': float,  # タイプ重複率（0-1）
                'new_category_ratio': float,  # 新カテゴリー推薦率（0-1、高いほど健全）
            }

        Reference:
            Pariser (2011): The Filter Bubble
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return {
                'filter_bubble_score': 0.0,
                'avg_category_overlap': 0.0,
                'avg_type_overlap': 0.0,
                'new_category_ratio': 0.0,
            }

        # 力量マスタからカテゴリー・タイプ情報を取得
        competence_info = {}
        for _, row in competence_master.iterrows():
            competence_info[row['力量コード']] = {
                'category': row.get('カテゴリー', 'Unknown'),
                'type': row.get('力量タイプ', 'Unknown'),
            }

        category_overlaps = []
        type_overlaps = []
        new_category_counts = []

        # メンバーごとに分析
        member_codes = train_data['メンバーコード'].unique()
        for member_code, recommendations in zip(member_codes, recommendations_list):
            if len(recommendations) == 0:
                continue

            # このメンバーの既習得スキルのカテゴリー・タイプ
            member_train = train_data[train_data['メンバーコード'] == member_code]
            acquired_categories = set()
            acquired_types = set()

            for comp_code in member_train['力量コード'].unique():
                info = competence_info.get(comp_code, {})
                acquired_categories.add(info.get('category', 'Unknown'))
                acquired_types.add(info.get('type', 'Unknown'))

            # 推薦スキルのカテゴリー・タイプ
            recommended_categories = set()
            recommended_types = set()

            for rec in recommendations:
                info = competence_info.get(rec.competence_code, {})
                recommended_categories.add(info.get('category', 'Unknown'))
                recommended_types.add(info.get('type', 'Unknown'))

            # 重複率を計算
            category_overlap = 0.0
            if len(recommended_categories) > 0:
                category_overlap = len(acquired_categories & recommended_categories) / len(recommended_categories)

            type_overlap = 0.0
            if len(recommended_types) > 0:
                type_overlap = len(acquired_types & recommended_types) / len(recommended_types)

            category_overlaps.append(category_overlap)
            type_overlaps.append(type_overlap)

            # 新カテゴリー推薦数
            new_categories = recommended_categories - acquired_categories
            new_category_ratio = len(new_categories) / len(recommended_categories) if len(recommended_categories) > 0 else 0.0
            new_category_counts.append(new_category_ratio)

        # 平均を計算
        avg_category_overlap = np.mean(category_overlaps) if category_overlaps else 0.0
        avg_type_overlap = np.mean(type_overlaps) if type_overlaps else 0.0
        new_category_ratio = np.mean(new_category_counts) if new_category_counts else 0.0

        # Filter Bubble Score: 重複率が高いほどバブル（問題）
        filter_bubble_score = (avg_category_overlap + avg_type_overlap) / 2.0

        return {
            'filter_bubble_score': filter_bubble_score,
            'avg_category_overlap': avg_category_overlap,
            'avg_type_overlap': avg_type_overlap,
            'new_category_ratio': new_category_ratio,
        }

    def calculate_intra_list_diversity(
        self,
        recommendations_list: List[List],
        competence_master: pd.DataFrame,
    ) -> float:
        """
        Intra-List Diversity (ILD)を計算

        リスト内の推薦スキル同士の多様性を測定。
        高いほど推薦が多様（ユーザー満足度向上）。

        Args:
            recommendations_list: メンバーごとの推薦結果リスト
            competence_master: 力量マスタ（カテゴリー情報）

        Returns:
            ILDスコア（0-1、高いほど多様）

        Reference:
            Vargas & Castells (2011): Rank and Relevance in Novelty and Diversity Metrics
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return 0.0

        # 力量マスタからカテゴリー情報を取得
        competence_categories = {}
        for _, row in competence_master.iterrows():
            competence_categories[row['力量コード']] = row.get('カテゴリー', 'Unknown')

        ild_scores = []

        for recommendations in recommendations_list:
            if len(recommendations) <= 1:
                continue

            # リスト内の各ペアの非類似度を計算
            dissimilarities = []
            n = len(recommendations)

            for i in range(n):
                for j in range(i + 1, n):
                    code_i = recommendations[i].competence_code
                    code_j = recommendations[j].competence_code

                    category_i = competence_categories.get(code_i, 'Unknown')
                    category_j = competence_categories.get(code_j, 'Unknown')

                    # 異なるカテゴリー = 非類似度1.0、同じ = 0.0
                    dissimilarity = 1.0 if category_i != category_j else 0.0
                    dissimilarities.append(dissimilarity)

            # 平均非類似度 = ILD
            if dissimilarities:
                ild = np.mean(dissimilarities)
                ild_scores.append(ild)

        return np.mean(ild_scores) if ild_scores else 0.0

    def calculate_serendipity(
        self,
        recommendations_list: List[List],
        train_data: pd.DataFrame,
        member_competence: pd.DataFrame,
        competence_master: pd.DataFrame,
    ) -> float:
        """
        Serendipity（意外性）を計算

        意外でありながら有用な推薦（セレンディピティ）を測定。
        - 人気度が低い（意外）
        - 既習得スキルと異なるカテゴリー（意外）
        - かつ実際に習得された（有用）

        Args:
            recommendations_list: メンバーごとの推薦結果リスト
            train_data: 訓練データ
            member_competence: 全メンバー習得データ（人気度計算用）
            competence_master: 力量マスタ

        Returns:
            Serendipityスコア（0-1、高いほど意外性が高い）

        Reference:
            Ge et al. (2010): Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return 0.0

        # 人気度を計算
        competence_counts = member_competence['力量コード'].value_counts()
        total_members = member_competence['メンバーコード'].nunique()
        competence_popularity = {
            code: count / total_members
            for code, count in competence_counts.items()
        }

        # カテゴリー情報
        competence_categories = {}
        for _, row in competence_master.iterrows():
            competence_categories[row['力量コード']] = row.get('カテゴリー', 'Unknown')

        serendipity_scores = []

        # メンバーごとに分析
        member_codes = train_data['メンバーコード'].unique()
        for member_code, recommendations in zip(member_codes, recommendations_list):
            if len(recommendations) == 0:
                continue

            # このメンバーの既習得カテゴリー
            member_train = train_data[train_data['メンバーコード'] == member_code]
            acquired_categories = set()
            for comp_code in member_train['力量コード'].unique():
                acquired_categories.add(competence_categories.get(comp_code, 'Unknown'))

            # 各推薦のセレンディピティを計算
            for rec in recommendations:
                popularity = competence_popularity.get(rec.competence_code, 0.0)
                category = competence_categories.get(rec.competence_code, 'Unknown')

                # 意外性 = 低人気度 × 異カテゴリー
                # 人気度が低いほど意外（1 - popularity）
                unexpectedness = 1.0 - popularity

                # 異なるカテゴリーなら意外性加算
                if category not in acquired_categories:
                    unexpectedness *= 1.5  # ブースト

                serendipity_scores.append(min(1.0, unexpectedness))

        return np.mean(serendipity_scores) if serendipity_scores else 0.0

    def calculate_personalization(
        self,
        recommendations_list: List[List],
    ) -> float:
        """
        Personalization（個人化度）を計算

        ユーザー間の推薦の差異を測定。
        高いほど個人化されている（同じ推薦を全員に出していない）。

        Args:
            recommendations_list: メンバーごとの推薦結果リスト

        Returns:
            Personalizationスコア（0-1、高いほど個人化されている）

        Formula:
            personalization = 1 - avg_jaccard_similarity(recommendations)

        Reference:
            Adomavicius & Tuzhilin (2005): Toward the Next Generation of Recommender Systems
        """
        if not recommendations_list or len(recommendations_list) < 2:
            return 0.0

        # 各ユーザーの推薦をセットに変換
        recommendation_sets = []
        for recommendations in recommendations_list:
            rec_set = set([rec.competence_code for rec in recommendations])
            recommendation_sets.append(rec_set)

        # 全ペアのJaccard類似度を計算
        jaccard_similarities = []
        n = len(recommendation_sets)

        for i in range(n):
            for j in range(i + 1, n):
                set_i = recommendation_sets[i]
                set_j = recommendation_sets[j]

                if len(set_i) == 0 and len(set_j) == 0:
                    continue

                # Jaccard類似度
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)

                if union > 0:
                    jaccard = intersection / union
                    jaccard_similarities.append(jaccard)

        # Personalization = 1 - avg(Jaccard)
        avg_jaccard = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
        personalization = 1.0 - avg_jaccard

        return personalization

    def calculate_calibration(
        self,
        train_data: pd.DataFrame,
        recommendations_list: List[List],
        competence_master: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calibration（キャリブレーション）を計算

        推薦分布 vs ユーザーの興味分布のKLダイバージェンスを測定。
        低いほどユーザーの興味を忠実に反映している。

        Args:
            train_data: 訓練データ（ユーザーの興味を推定）
            recommendations_list: メンバーごとの推薦結果リスト
            competence_master: 力量マスタ（カテゴリー情報）

        Returns:
            {
                'kl_divergence': float,  # KLダイバージェンス（低いほど良い）
                'category_calibration': float,  # カテゴリーレベルのキャリブレーション
            }

        Reference:
            Steck (2018): Calibrated Recommendations
        """
        if not recommendations_list or len(recommendations_list) == 0:
            return {
                'kl_divergence': 0.0,
                'category_calibration': 0.0,
            }

        # カテゴリー情報
        competence_categories = {}
        for _, row in competence_master.iterrows():
            competence_categories[row['力量コード']] = row.get('カテゴリー', 'Unknown')

        kl_divergences = []
        category_calibrations = []

        # メンバーごとに分析
        member_codes = train_data['メンバーコード'].unique()
        for member_code, recommendations in zip(member_codes, recommendations_list):
            if len(recommendations) == 0:
                continue

            # ユーザーの興味分布（訓練データのカテゴリー分布）
            member_train = train_data[train_data['メンバーコード'] == member_code]
            user_category_counts = Counter()

            for comp_code in member_train['力量コード'].unique():
                category = competence_categories.get(comp_code, 'Unknown')
                user_category_counts[category] += 1

            # 推薦分布（推薦のカテゴリー分布）
            rec_category_counts = Counter()
            for rec in recommendations:
                category = competence_categories.get(rec.competence_code, 'Unknown')
                rec_category_counts[category] += 1

            # 全カテゴリーを取得
            all_categories = set(user_category_counts.keys()) | set(rec_category_counts.keys())

            if len(all_categories) == 0:
                continue

            # 確率分布に変換
            user_total = sum(user_category_counts.values())
            rec_total = sum(rec_category_counts.values())

            user_probs = []
            rec_probs = []

            for category in all_categories:
                user_prob = user_category_counts[category] / user_total if user_total > 0 else 0.0
                rec_prob = rec_category_counts[category] / rec_total if rec_total > 0 else 0.0

                user_probs.append(user_prob)
                rec_probs.append(rec_prob)

            # KLダイバージェンスを計算: KL(P || Q) = Σ p(x) log(p(x) / q(x))
            kl_div = 0.0
            for p, q in zip(user_probs, rec_probs):
                if p > 0 and q > 0:
                    kl_div += p * np.log(p / q)
                elif p > 0 and q == 0:
                    # q=0の場合は非常に大きなペナルティ
                    kl_div += p * 10.0

            kl_divergences.append(kl_div)

            # Category Calibration: ユーザー分布と推薦分布の差の絶対値
            category_calib = sum(abs(p - q) for p, q in zip(user_probs, rec_probs)) / 2.0
            category_calibrations.append(category_calib)

        return {
            'kl_divergence': np.mean(kl_divergences) if kl_divergences else 0.0,
            'category_calibration': np.mean(category_calibrations) if category_calibrations else 0.0,
        }

    def comprehensive_evaluation(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        top_k: int = 10,
        member_sample: Optional[List[str]] = None,
        include_all_metrics: bool = True,
    ) -> ComprehensiveEvaluationResult:
        """
        包括的な評価を実行

        精度、バイアス、多様性、カバレッジ、キャリブレーションを一度に評価。

        Args:
            train_data: 訓練データ
            test_data: テストデータ
            competence_master: 力量マスタ
            top_k: Top-K推薦
            member_sample: 評価対象メンバー（Noneで全員）
            include_all_metrics: 全メトリクスを計算するか（高コスト）

        Returns:
            ComprehensiveEvaluationResult

        Example:
            >>> evaluator = AdvancedMetricsEvaluator()
            >>> result = evaluator.comprehensive_evaluation(
            ...     train_data=train_df,
            ...     test_data=test_df,
            ...     competence_master=master_df,
            ...     top_k=10
            ... )
            >>> print(f"Overall Score: {result.overall_score:.3f}")
            >>> evaluator.print_comprehensive_report(result)
        """
        print(f"\n{'='*80}")
        print(f"包括的評価開始（GAFAレベル）")
        print(f"{'='*80}\n")

        # 1. 精度評価（既存のメトリクス）
        print("[1/5] 精度評価（Precision, Recall, NDCG, MRR, MAP）...")
        accuracy_metrics = self.evaluate_with_diversity(
            train_data=train_data,
            test_data=test_data,
            competence_master=competence_master,
            top_k=top_k,
            member_sample=member_sample,
        )

        # 推薦結果を収集（以降のメトリクス計算用）
        if member_sample is None:
            member_sample = test_data['メンバーコード'].unique().tolist()

        recommendations_list, actual_acquired_list = self._collect_recommendations(
            train_data=train_data,
            test_data=test_data,
            competence_master=competence_master,
            member_sample=member_sample,
            top_k=top_k,
        )

        # 2. バイアス評価
        print("[2/5] バイアス評価（Popularity, Position, Filter bubble）...")
        bias_metrics = {}

        # Popularity Bias
        popularity_bias = self.evaluate_popularity_bias(
            recommendations_list=recommendations_list,
            member_competence=train_data,
        )
        bias_metrics.update(popularity_bias)

        # Position Bias
        position_bias = self.evaluate_position_bias(
            recommendations_list=recommendations_list,
            actual_acquired_list=actual_acquired_list,
        )
        bias_metrics.update(position_bias)

        # Filter Bubble
        filter_bubble = self.evaluate_filter_bubble(
            train_data=train_data,
            recommendations_list=recommendations_list,
            competence_master=competence_master,
        )
        bias_metrics.update(filter_bubble)

        # 3. 多様性評価
        print("[3/5] 多様性評価（ILD, Serendipity, Personalization）...")
        diversity_metrics = {}

        # ILD
        ild = self.calculate_intra_list_diversity(
            recommendations_list=recommendations_list,
            competence_master=competence_master,
        )
        diversity_metrics['intra_list_diversity'] = ild

        # Serendipity
        serendipity = self.calculate_serendipity(
            recommendations_list=recommendations_list,
            train_data=train_data,
            member_competence=train_data,
            competence_master=competence_master,
        )
        diversity_metrics['serendipity'] = serendipity

        # Personalization
        personalization = self.calculate_personalization(
            recommendations_list=recommendations_list,
        )
        diversity_metrics['personalization'] = personalization

        # 4. カバレッジ評価（既にaccuracy_metricsに含まれている）
        print("[4/5] カバレッジ評価...")
        coverage_metrics = {
            'catalog_coverage': accuracy_metrics.get('catalog_coverage', 0.0),
            'total_unique_recommended': accuracy_metrics.get('total_unique_recommended', 0),
        }

        # 5. キャリブレーション評価
        print("[5/5] キャリブレーション評価（KL divergence）...")
        calibration_metrics = self.calculate_calibration(
            train_data=train_data,
            recommendations_list=recommendations_list,
            competence_master=competence_master,
        )

        # 総合スコアを計算（重み付き平均）
        overall_score = self._calculate_overall_score(
            accuracy_metrics=accuracy_metrics,
            bias_metrics=bias_metrics,
            diversity_metrics=diversity_metrics,
            coverage_metrics=coverage_metrics,
            calibration_metrics=calibration_metrics,
            top_k=top_k,
        )

        result = ComprehensiveEvaluationResult(
            accuracy_metrics=accuracy_metrics,
            bias_metrics=bias_metrics,
            diversity_metrics=diversity_metrics,
            coverage_metrics=coverage_metrics,
            calibration_metrics=calibration_metrics,
            overall_score=overall_score,
        )

        print(f"\n{'='*80}")
        print(f"包括的評価完了")
        print(f"{'='*80}\n")

        return result

    def _collect_recommendations(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        competence_master: pd.DataFrame,
        member_sample: List[str],
        top_k: int,
    ) -> Tuple[List[List], List[List[str]]]:
        """推薦結果と正解データを収集"""
        # MLレコメンダーの準備
        if self.recommender is None:
            from ..ml.ml_recommender import MLRecommender

            member_codes = train_data['メンバーコード'].unique()
            members_data = pd.DataFrame({
                'メンバーコード': member_codes,
                'メンバー名': [f'メンバー{code}' for code in member_codes],
                '役職': ['未設定'] * len(member_codes),
                '職能等級': ['未設定'] * len(member_codes),
            })

            n_members = len(train_data['メンバーコード'].unique())
            n_competences = len(train_data['力量コード'].unique())
            safe_n_components = min(20, n_members, n_competences)

            recommender = MLRecommender.build(
                member_competence=train_data,
                competence_master=competence_master,
                member_master=members_data,
                use_preprocessing=False,
                use_tuning=False,
                n_components=safe_n_components,
            )
        else:
            recommender = self.recommender

        recommendations_list = []
        actual_acquired_list = []

        for member_code in member_sample:
            # 正解データ
            actual_acquired = test_data[test_data['メンバーコード'] == member_code]['力量コード'].unique().tolist()

            if len(actual_acquired) == 0:
                continue

            try:
                # 推薦生成
                recommendations = recommender.recommend(
                    member_code=member_code,
                    top_n=top_k,
                    use_diversity=False
                )

                if len(recommendations) > 0:
                    recommendations_list.append(recommendations)
                    actual_acquired_list.append(actual_acquired)

            except Exception:
                continue

        return recommendations_list, actual_acquired_list

    def _calculate_overall_score(
        self,
        accuracy_metrics: Dict[str, float],
        bias_metrics: Dict[str, float],
        diversity_metrics: Dict[str, float],
        coverage_metrics: Dict[str, float],
        calibration_metrics: Dict[str, float],
        top_k: int,
    ) -> float:
        """
        総合スコアを計算（重み付き平均）

        重み:
        - Accuracy: 40%
        - Diversity: 20%
        - Coverage: 15%
        - Bias (inverse): 15%
        - Calibration (inverse): 10%
        """
        # Accuracy (0-1)
        accuracy_score = (
            accuracy_metrics.get(f'precision@{top_k}', 0.0) * 0.3
            + accuracy_metrics.get(f'recall@{top_k}', 0.0) * 0.3
            + accuracy_metrics.get(f'ndcg@{top_k}', 0.0) * 0.4
        )

        # Diversity (0-1)
        diversity_score = (
            diversity_metrics.get('intra_list_diversity', 0.0) * 0.4
            + diversity_metrics.get('serendipity', 0.0) * 0.3
            + diversity_metrics.get('personalization', 0.0) * 0.3
        )

        # Coverage (0-1)
        coverage_score = coverage_metrics.get('catalog_coverage', 0.0)

        # Bias (inverse, 0-1)
        # 低いほど良いので反転
        bias_score = 1.0 - (
            bias_metrics.get('popularity_bias', 0.0) * 0.3
            + bias_metrics.get('position_bias_score', 0.0) * 0.3
            + bias_metrics.get('filter_bubble_score', 0.0) * 0.4
        )

        # Calibration (inverse, 0-1)
        # KL divergenceは低いほど良い（0-10程度を想定）
        kl_div = calibration_metrics.get('kl_divergence', 0.0)
        calibration_score = max(0.0, 1.0 - kl_div / 10.0)

        # 総合スコア
        overall = (
            accuracy_score * 0.40
            + diversity_score * 0.20
            + coverage_score * 0.15
            + bias_score * 0.15
            + calibration_score * 0.10
        )

        return overall

    def print_comprehensive_report(self, result: ComprehensiveEvaluationResult, top_k: int = 10):
        """
        包括的評価レポートを表示

        Args:
            result: ComprehensiveEvaluationResult
            top_k: Top-K（表示用）
        """
        print(f"\n{'='*80}")
        print(f"📊 包括的評価レポート（GAFAレベル）")
        print(f"{'='*80}\n")

        # 総合スコア
        print(f"🎯 総合スコア: {result.overall_score:.3f} / 1.000")
        print(f"{'='*80}\n")

        # 1. 精度評価
        print(f"【1. 精度評価（Accuracy Metrics）】")
        print(f"  Precision@{top_k}:  {result.accuracy_metrics.get(f'precision@{top_k}', 0.0):.4f}")
        print(f"  Recall@{top_k}:     {result.accuracy_metrics.get(f'recall@{top_k}', 0.0):.4f}")
        print(f"  NDCG@{top_k}:       {result.accuracy_metrics.get(f'ndcg@{top_k}', 0.0):.4f}")
        print(f"  F1@{top_k}:         {result.accuracy_metrics.get(f'f1@{top_k}', 0.0):.4f}")
        print(f"  Hit Rate:       {result.accuracy_metrics.get('hit_rate', 0.0):.4f}\n")

        # 2. バイアス評価
        print(f"【2. バイアス評価（Bias Metrics）】")
        print(f"  Popularity Bias:  {result.bias_metrics.get('popularity_bias', 0.0):.4f}  (低いほど健全)")
        print(f"  Tail Ratio:       {result.bias_metrics.get('tail_ratio', 0.0):.4f}  (高いほど健全)")
        print(f"  Position Bias:    {result.bias_metrics.get('position_bias_score', 0.0):.4f}  (低いほど健全)")
        print(f"  Filter Bubble:    {result.bias_metrics.get('filter_bubble_score', 0.0):.4f}  (低いほど健全)\n")

        # 3. 多様性評価
        print(f"【3. 多様性評価（Diversity Metrics）】")
        print(f"  ILD:              {result.diversity_metrics.get('intra_list_diversity', 0.0):.4f}  (リスト内多様性)")
        print(f"  Serendipity:      {result.diversity_metrics.get('serendipity', 0.0):.4f}  (意外性)")
        print(f"  Personalization:  {result.diversity_metrics.get('personalization', 0.0):.4f}  (個人化度)\n")

        # 4. カバレッジ評価
        print(f"【4. カバレッジ評価（Coverage Metrics）】")
        print(f"  Catalog Coverage: {result.coverage_metrics.get('catalog_coverage', 0.0):.4f}  (推薦に含まれた力量の割合)")
        print(f"  Unique Items:     {result.coverage_metrics.get('total_unique_recommended', 0)}個\n")

        # 5. キャリブレーション評価
        print(f"【5. キャリブレーション評価（Calibration Metrics）】")
        print(f"  KL Divergence:    {result.calibration_metrics.get('kl_divergence', 0.0):.4f}  (低いほど良い)")
        print(f"  Category Calib.:  {result.calibration_metrics.get('category_calibration', 0.0):.4f}  (低いほど良い)\n")

        print(f"{'='*80}")

        # 診断メッセージ
        self._print_diagnostic_messages(result)

    def _print_diagnostic_messages(self, result: ComprehensiveEvaluationResult):
        """診断メッセージを表示"""
        print(f"\n💡 診断・推奨事項:\n")

        issues = []
        recommendations = []

        # Popularity Bias
        if result.bias_metrics.get('popularity_bias', 0.0) > 0.7:
            issues.append("⚠️ 人気スキルを過剰に推薦しています（Popularity Bias）")
            recommendations.append("   → ロングテールスキルの重み付けを増やしてください")

        # Position Bias
        if result.bias_metrics.get('position_bias_score', 0.0) > 0.5:
            issues.append("⚠️ ランキング上位に偏った評価になっています（Position Bias）")
            recommendations.append("   → 位置バイアス補正を実装してください")

        # Filter Bubble
        if result.bias_metrics.get('filter_bubble_score', 0.0) > 0.7:
            issues.append("⚠️ 既存スキルと似たものばかり推薦しています（Filter Bubble）")
            recommendations.append("   → 新カテゴリーの推薦重みを増やしてください")

        # ILD
        if result.diversity_metrics.get('intra_list_diversity', 0.0) < 0.3:
            issues.append("⚠️ リスト内の多様性が低いです（ILD）")
            recommendations.append("   → 多様性を考慮した推薦アルゴリズムを導入してください")

        # Personalization
        if result.diversity_metrics.get('personalization', 0.0) < 0.3:
            issues.append("⚠️ ユーザー間で推薦が類似しています（Personalization）")
            recommendations.append("   → パーソナライゼーション強度を上げてください")

        # Coverage
        if result.coverage_metrics.get('catalog_coverage', 0.0) < 0.2:
            issues.append("⚠️ カタログカバレッジが低いです")
            recommendations.append("   → ロングテールアイテムを推薦するよう調整してください")

        if issues:
            for issue in issues:
                print(issue)
            print()
            for rec in recommendations:
                print(rec)
        else:
            print("✅ すべてのメトリクスが健全な範囲内です")

        print()
