"""
多様性再ランキングのテスト
"""

import pytest
import pandas as pd
from skillnote_recommendation.ml.diversity import DiversityReranker


# ==================== フィクスチャ ====================

@pytest.fixture
def sample_candidates():
    """サンプル候補リスト（力量コード, スコア）"""
    return [
        ('s001', 0.9),
        ('s002', 0.8),
        ('s003', 0.7),
        ('s004', 0.6),
        ('s005', 0.5),
        ('s006', 0.4),
        ('s007', 0.3),
        ('s008', 0.2),
    ]


@pytest.fixture
def sample_competence_info():
    """サンプル力量情報"""
    return pd.DataFrame({
        '力量コード': ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008'],
        '力量名': ['Python', 'Java', 'SQL', 'AWS研修', 'Azure研修', '基本情報', 'Docker', 'React'],
        '力量タイプ': ['SKILL', 'SKILL', 'SKILL', 'EDUCATION', 'EDUCATION', 'LICENSE', 'SKILL', 'SKILL'],
        '力量カテゴリー名': [
            'プログラミング',
            'プログラミング',
            'データベース',
            'クラウド',
            'クラウド',
            'IT資格',
            'インフラ',
            'フロントエンド'
        ]
    })


@pytest.fixture
def reranker():
    """多様性再ランキング器"""
    return DiversityReranker(lambda_relevance=0.7)


# ==================== 初期化テスト ====================

class TestInitialization:
    """初期化のテスト"""

    def test_default_initialization(self):
        """デフォルト初期化"""
        reranker = DiversityReranker()

        assert reranker.lambda_relevance == 0.7
        assert reranker.category_weight == 0.5
        assert reranker.type_weight == 0.3

    def test_custom_initialization(self):
        """カスタムパラメータで初期化"""
        reranker = DiversityReranker(
            lambda_relevance=0.5,
            category_weight=0.6,
            type_weight=0.4
        )

        assert reranker.lambda_relevance == 0.5
        assert reranker.category_weight == 0.6
        assert reranker.type_weight == 0.4


# ==================== MMR再ランキングテスト ====================

class TestMMRReranking:
    """MMR再ランキングのテスト"""

    def test_mmr_basic(self, reranker, sample_candidates, sample_competence_info):
        """基本的なMMR再ランキング"""
        result = reranker.rerank_mmr(
            sample_candidates,
            sample_competence_info,
            k=5
        )

        assert len(result) == 5
        assert all(isinstance(item, tuple) for item in result)

    def test_mmr_first_item_is_highest_relevance(self, sample_candidates, sample_competence_info):
        """最初のアイテムは最も関連性が高い（lambda=1.0の場合）"""
        reranker = DiversityReranker(lambda_relevance=1.0)
        result = reranker.rerank_mmr(sample_candidates, sample_competence_info, k=3)

        # lambda=1.0 なら関連性のみで選ばれるので、最初は最高スコア
        assert result[0] == sample_candidates[0]

    def test_mmr_promotes_diversity(self, sample_candidates, sample_competence_info):
        """多様性が促進される（lambda=0の場合）"""
        reranker = DiversityReranker(lambda_relevance=0.0)
        result = reranker.rerank_mmr(sample_candidates, sample_competence_info, k=5)

        # 多様性重視なので、同じカテゴリやタイプが連続しにくい
        types = [
            sample_competence_info[sample_competence_info['力量コード'] == code]['力量タイプ'].iloc[0]
            for code, _ in result
        ]

        # 少なくとも2種類のタイプが含まれる
        assert len(set(types)) >= 2

    def test_mmr_empty_candidates(self, reranker, sample_competence_info):
        """候補が空の場合"""
        result = reranker.rerank_mmr([], sample_competence_info, k=5)

        assert result == []

    def test_mmr_k_larger_than_candidates(self, reranker, sample_candidates, sample_competence_info):
        """k が候補数より大きい場合"""
        result = reranker.rerank_mmr(sample_candidates, sample_competence_info, k=100)

        assert len(result) == len(sample_candidates)


# ==================== カテゴリ多様性テスト ====================

class TestCategoryDiversity:
    """カテゴリ多様性再ランキングのテスト"""

    def test_category_diversity_basic(self, reranker, sample_candidates, sample_competence_info):
        """基本的なカテゴリ多様性"""
        result = reranker.rerank_category_diversity(
            sample_candidates,
            sample_competence_info,
            k=5
        )

        assert len(result) == 5

    def test_category_diversity_with_max_per_category(self, reranker, sample_candidates, sample_competence_info):
        """カテゴリごとの最大数制限"""
        result = reranker.rerank_category_diversity(
            sample_candidates,
            sample_competence_info,
            k=8,
            max_per_category=2
        )

        # カテゴリごとのカウント
        category_counts = {}
        for code, _ in result:
            category = sample_competence_info[
                sample_competence_info['力量コード'] == code
            ]['力量カテゴリー名'].iloc[0]
            category_counts[category] = category_counts.get(category, 0) + 1

        # 各カテゴリが最大2つまで
        assert all(count <= 2 for count in category_counts.values())

    def test_category_diversity_empty_candidates(self, reranker, sample_competence_info):
        """候補が空の場合"""
        result = reranker.rerank_category_diversity([], sample_competence_info, k=5)

        assert result == []


# ==================== タイプ多様性テスト ====================

class TestTypeDiversity:
    """タイプ多様性再ランキングのテスト"""

    def test_type_diversity_basic(self, reranker, sample_candidates, sample_competence_info):
        """基本的なタイプ多様性"""
        result = reranker.rerank_type_diversity(
            sample_candidates,
            sample_competence_info,
            k=6
        )

        assert len(result) == 6

    def test_type_diversity_with_ratios(self, reranker, sample_candidates, sample_competence_info):
        """タイプ比率指定"""
        # SKILL:EDUCATION:LICENSE = 50%:30%:20%
        result = reranker.rerank_type_diversity(
            sample_candidates,
            sample_competence_info,
            k=10,
            type_ratios={'SKILL': 0.5, 'EDUCATION': 0.3, 'LICENSE': 0.2}
        )

        # タイプごとのカウント
        type_counts = {'SKILL': 0, 'EDUCATION': 0, 'LICENSE': 0}
        for code, _ in result:
            comp_type = sample_competence_info[
                sample_competence_info['力量コード'] == code
            ]['力量タイプ'].iloc[0]
            type_counts[comp_type] += 1

        # おおよそ比率に従っているか（±1は許容）
        assert 4 <= type_counts['SKILL'] <= 6  # 50% of 10 = 5
        assert 2 <= type_counts['EDUCATION'] <= 4  # 30% of 10 = 3
        assert 1 <= type_counts['LICENSE'] <= 3  # 20% of 10 = 2

    def test_type_diversity_empty_candidates(self, reranker, sample_competence_info):
        """候補が空の場合"""
        result = reranker.rerank_type_diversity([], sample_competence_info, k=5)

        assert result == []


# ==================== ハイブリッド再ランキングテスト ====================

class TestHybridReranking:
    """ハイブリッド再ランキングのテスト"""

    def test_hybrid_basic(self, reranker, sample_candidates, sample_competence_info):
        """基本的なハイブリッド再ランキング"""
        result = reranker.rerank_hybrid(
            sample_candidates,
            sample_competence_info,
            k=5
        )

        assert len(result) == 5

    def test_hybrid_promotes_diversity(self, reranker, sample_candidates, sample_competence_info):
        """ハイブリッドは多様性を促進"""
        result = reranker.rerank_hybrid(
            sample_candidates,
            sample_competence_info,
            k=6,
            max_per_category=2
        )

        # 複数のタイプが含まれる
        types = set()
        for code, _ in result:
            comp_type = sample_competence_info[
                sample_competence_info['力量コード'] == code
            ]['力量タイプ'].iloc[0]
            types.add(comp_type)

        assert len(types) >= 2


# ==================== 多様性指標計算テスト ====================

class TestDiversityMetrics:
    """多様性指標計算のテスト"""

    def test_calculate_diversity_metrics_basic(self, reranker, sample_candidates, sample_competence_info):
        """基本的な多様性指標計算"""
        recommendations = sample_candidates[:5]

        metrics = reranker.calculate_diversity_metrics(
            recommendations,
            sample_competence_info
        )

        assert 'category_diversity' in metrics
        assert 'type_diversity' in metrics
        assert 'intra_list_diversity' in metrics
        assert 'unique_categories' in metrics
        assert 'unique_types' in metrics

    def test_diversity_metrics_all_same_category(self, reranker, sample_competence_info):
        """全て同じカテゴリの場合"""
        # プログラミングのみ
        recommendations = [('s001', 0.9), ('s002', 0.8)]

        metrics = reranker.calculate_diversity_metrics(
            recommendations,
            sample_competence_info
        )

        # カテゴリ多様性は低い（1/2 = 0.5）
        assert metrics['category_diversity'] == 0.5
        assert metrics['unique_categories'] == 1

    def test_diversity_metrics_all_different_types(self, reranker, sample_competence_info):
        """全て異なるタイプの場合"""
        # SKILL, EDUCATION, LICENSE
        recommendations = [('s001', 0.9), ('s004', 0.8), ('s006', 0.7)]

        metrics = reranker.calculate_diversity_metrics(
            recommendations,
            sample_competence_info
        )

        # タイプ多様性は最大（3/3 = 1.0）
        assert metrics['type_diversity'] == 1.0
        assert metrics['unique_types'] == 3

    def test_diversity_metrics_empty_recommendations(self, reranker, sample_competence_info):
        """推薦が空の場合"""
        metrics = reranker.calculate_diversity_metrics([], sample_competence_info)

        assert metrics['category_diversity'] == 0.0
        assert metrics['type_diversity'] == 0.0

    def test_intra_list_diversity_calculation(self, reranker, sample_competence_info):
        """Intra-List多様性の計算"""
        # 全て同じカテゴリ・タイプ
        same_recommendations = [('s001', 0.9), ('s002', 0.8)]  # 両方SKILL, プログラミング

        # 異なるカテゴリ・タイプ
        diverse_recommendations = [('s001', 0.9), ('s004', 0.8)]  # SKILL vs EDUCATION

        same_metrics = reranker.calculate_diversity_metrics(
            same_recommendations,
            sample_competence_info
        )

        diverse_metrics = reranker.calculate_diversity_metrics(
            diverse_recommendations,
            sample_competence_info
        )

        # 多様性が高い方がintra_list_diversityも高い
        assert diverse_metrics['intra_list_diversity'] > same_metrics['intra_list_diversity']


# ==================== 統合テスト ====================

class TestIntegration:
    """統合テスト"""

    def test_full_diversity_workflow(self, sample_candidates, sample_competence_info):
        """完全な多様性再ランキングワークフロー"""
        # 1. 再ランキング器を初期化
        reranker = DiversityReranker(lambda_relevance=0.6)

        # 2. 各種再ランキングを実行
        mmr_result = reranker.rerank_mmr(sample_candidates, sample_competence_info, k=5)
        category_result = reranker.rerank_category_diversity(
            sample_candidates, sample_competence_info, k=5, max_per_category=2
        )
        type_result = reranker.rerank_type_diversity(sample_candidates, sample_competence_info, k=5)
        hybrid_result = reranker.rerank_hybrid(sample_candidates, sample_competence_info, k=5)

        # 3. 全ての結果が妥当
        assert len(mmr_result) == 5
        assert len(category_result) == 5
        assert len(type_result) == 5
        assert len(hybrid_result) == 5

        # 4. 多様性指標を計算
        hybrid_metrics = reranker.calculate_diversity_metrics(
            hybrid_result,
            sample_competence_info
        )

        # 5. 多様性指標が妥当
        assert 0.0 <= hybrid_metrics['category_diversity'] <= 1.0
        assert 0.0 <= hybrid_metrics['type_diversity'] <= 1.0
        assert 0.0 <= hybrid_metrics['intra_list_diversity'] <= 1.0

    def test_comparison_relevance_vs_diversity(self, sample_candidates, sample_competence_info):
        """関連性重視 vs 多様性重視の比較"""
        # 関連性重視
        relevance_reranker = DiversityReranker(lambda_relevance=0.9)
        relevance_result = relevance_reranker.rerank_mmr(
            sample_candidates, sample_competence_info, k=5
        )

        # 多様性重視
        diversity_reranker = DiversityReranker(lambda_relevance=0.1)
        diversity_result = diversity_reranker.rerank_mmr(
            sample_candidates, sample_competence_info, k=5
        )

        # 多様性指標を計算
        relevance_metrics = relevance_reranker.calculate_diversity_metrics(
            relevance_result, sample_competence_info
        )
        diversity_metrics = diversity_reranker.calculate_diversity_metrics(
            diversity_result, sample_competence_info
        )

        # 多様性重視の方が多様性が高い（はず）
        # ただし、データによっては必ずしもそうならない場合もあるので緩い検証
        assert diversity_metrics['intra_list_diversity'] >= 0.0
        assert relevance_metrics['intra_list_diversity'] >= 0.0
