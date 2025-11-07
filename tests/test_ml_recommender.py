"""
MLRecommenderのテスト
"""

import pytest
import pandas as pd
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.ml.ml_recommender import MLRecommender


# ==================== フィクスチャ ====================


@pytest.fixture
def sample_skill_matrix():
    """サンプルメンバー×力量マトリクス"""
    return pd.DataFrame(
        {
            "s001": [3, 0, 2, 0, 1],
            "s002": [0, 4, 0, 3, 0],
            "s003": [2, 0, 5, 0, 2],
            "s004": [0, 3, 0, 4, 0],
            "s005": [1, 0, 2, 0, 3],
        },
        index=["m001", "m002", "m003", "m004", "m005"],
    )


@pytest.fixture
def sample_competence_master():
    """サンプル力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["s001", "s002", "s003", "s004", "s005"],
            "力量名": ["Python", "Java", "SQL", "AWS研修", "Docker"],
            "力量タイプ": ["SKILL", "SKILL", "SKILL", "EDUCATION", "SKILL"],
            "力量カテゴリー名": [
                "プログラミング",
                "プログラミング",
                "データベース",
                "クラウド",
                "インフラ",
            ],
        }
    )


@pytest.fixture
def sample_member_competence():
    """サンプルメンバー習得力量データ"""
    data = []
    for member_idx, member_code in enumerate(["m001", "m002", "m003", "m004", "m005"]):
        for comp_idx, comp_code in enumerate(["s001", "s002", "s003", "s004", "s005"]):
            level = [3, 0, 2, 0, 1][comp_idx] if member_idx == 0 else 0
            if level > 0:
                data.append(
                    {"メンバーコード": member_code, "力量コード": comp_code, "正規化レベル": level}
                )

    return pd.DataFrame(data)


@pytest.fixture
def sample_member_master():
    """サンプルメンバーマスタ"""
    return pd.DataFrame(
        {
            "メンバーコード": ["m001", "m002", "m003", "m004", "m005"],
            "氏名": ["山田太郎", "佐藤花子", "鈴木一郎", "田中次郎", "高橋三郎"],
            "役職": ["エンジニア", "マネージャー", "エンジニア", "エンジニア", "リーダー"],
            "職能等級": ["3等級", "4等級", "2等級", "3等級", "3等級"],
        }
    )


@pytest.fixture
def ml_recommender(
    sample_skill_matrix, sample_competence_master, sample_member_competence, sample_member_master
):
    """MLベース推薦エンジン"""
    # Matrix Factorizationモデルを学習
    mf_model = MatrixFactorizationModel(n_components=2, random_state=42)
    mf_model.fit(sample_skill_matrix)

    # MLRecommenderを作成
    return MLRecommender(
        mf_model=mf_model,
        competence_master=sample_competence_master,
        member_competence=sample_member_competence,
        member_master=sample_member_master,
    )


# ==================== 基本推薦テスト ====================


class TestBasicRecommendation:
    """基本的な推薦のテスト"""

    def test_recommend_basic(self, ml_recommender):
        """基本的な推薦"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=3)

        assert len(recommendations) <= 3
        assert all(hasattr(rec, "competence_code") for rec in recommendations)
        assert all(hasattr(rec, "priority_score") for rec in recommendations)

    def test_recommend_excludes_acquired(self, ml_recommender):
        """既習得力量を除外"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=5)

        # m001が既に習得している力量（s001, s003, s005）は推薦されない
        recommended_codes = [rec.competence_code for rec in recommendations]
        assert "s001" not in recommended_codes
        assert "s003" not in recommended_codes
        assert "s005" not in recommended_codes

    def test_recommend_with_type_filter(self, ml_recommender):
        """タイプフィルタ付き推薦"""
        recommendations = ml_recommender.recommend(
            member_code="m001", top_n=3, competence_type="EDUCATION"
        )

        # 全てEDUCATION
        assert all(rec.competence_type == "EDUCATION" for rec in recommendations)

    def test_recommend_with_category_filter(self, ml_recommender):
        """カテゴリフィルタ付き推薦"""
        recommendations = ml_recommender.recommend(
            member_code="m001", top_n=3, category_filter="プログラミング"
        )

        # 全てプログラミングカテゴリ
        assert all("プログラミング" in rec.category for rec in recommendations)


# ==================== 多様性再ランキングテスト ====================


class TestDiversityReranking:
    """多様性再ランキングのテスト"""

    def test_recommend_with_diversity(self, ml_recommender):
        """多様性考慮の推薦"""
        recommendations = ml_recommender.recommend(
            member_code="m001", top_n=3, use_diversity=True, diversity_strategy="hybrid"
        )

        assert len(recommendations) <= 3

    def test_recommend_without_diversity(self, ml_recommender):
        """多様性なし（精度重視）の推薦"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=3, use_diversity=False)

        assert len(recommendations) <= 3

    def test_different_diversity_strategies(self, ml_recommender):
        """異なる多様性戦略"""
        strategies = ["mmr", "category", "type", "hybrid"]

        for strategy in strategies:
            recommendations = ml_recommender.recommend(
                member_code="m001", top_n=3, use_diversity=True, diversity_strategy=strategy
            )

            assert len(recommendations) <= 3


# ==================== 多様性指標テスト ====================


class TestDiversityMetrics:
    """多様性指標のテスト"""

    def test_calculate_diversity_metrics(self, ml_recommender):
        """多様性指標の計算"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=3, use_diversity=True)

        metrics = ml_recommender.calculate_diversity_metrics(recommendations)

        assert "category_diversity" in metrics
        assert "type_diversity" in metrics
        assert "intra_list_diversity" in metrics

    def test_diversity_metrics_range(self, ml_recommender):
        """多様性指標の値域"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=3)
        metrics = ml_recommender.calculate_diversity_metrics(recommendations)

        assert 0.0 <= metrics["category_diversity"] <= 1.0
        assert 0.0 <= metrics["type_diversity"] <= 1.0
        assert 0.0 <= metrics["intra_list_diversity"] <= 1.0


# ==================== 推薦理由テスト ====================


class TestRecommendationReason:
    """推薦理由のテスト"""

    def test_recommendation_has_reason(self, ml_recommender):
        """推薦に理由が含まれる"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=1)

        if len(recommendations) > 0:
            assert recommendations[0].reason is not None
            assert len(recommendations[0].reason) > 0

    def test_reason_mentions_diversity(self, ml_recommender):
        """多様性戦略が理由に含まれる"""
        recommendations = ml_recommender.recommend(
            member_code="m001", top_n=1, use_diversity=True, diversity_strategy="hybrid"
        )

        if len(recommendations) > 0:
            reason = recommendations[0].reason
            # 何らかの多様性に関する言及がある
            assert "バランス" in reason or "多様性" in reason or "スキル" in reason


# ==================== エッジケーステスト ====================


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_recommend_all_acquired(self, ml_recommender, sample_skill_matrix):
        """全て習得済みのメンバー"""
        # m003は全て習得済みと仮定（実際はテストデータによる）
        recommendations = ml_recommender.recommend(member_code="m003", top_n=3)

        # 推薦数は少ない可能性がある
        assert isinstance(recommendations, list)

    def test_recommend_top_n_zero(self, ml_recommender):
        """top_n=0の場合"""
        recommendations = ml_recommender.recommend(member_code="m001", top_n=0)

        assert len(recommendations) == 0


# ==================== 統合テスト ====================


class TestIntegration:
    """統合テスト"""

    def test_full_ml_recommendation_workflow(self, ml_recommender):
        """完全なML推薦ワークフロー"""
        # 1. 基本推薦
        basic_recs = ml_recommender.recommend(member_code="m001", top_n=5, use_diversity=False)

        # 2. 多様性考慮推薦
        diverse_recs = ml_recommender.recommend(
            member_code="m001", top_n=5, use_diversity=True, diversity_strategy="hybrid"
        )

        # 3. 多様性指標計算
        basic_metrics = ml_recommender.calculate_diversity_metrics(basic_recs)
        diverse_metrics = ml_recommender.calculate_diversity_metrics(diverse_recs)

        # 4. 両方の推薦が成功
        assert len(basic_recs) > 0
        assert len(diverse_recs) > 0

        # 5. 多様性指標が計算される
        assert basic_metrics["category_diversity"] >= 0.0
        assert diverse_metrics["category_diversity"] >= 0.0
