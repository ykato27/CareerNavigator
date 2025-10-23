"""
RecommendationEngineクラスのテスト

推薦エンジンのコアロジック（スコア計算、推薦実行）をテスト
"""

import pytest
import pandas as pd
from skillnote_recommendation.core.recommendation_engine import RecommendationEngine
from skillnote_recommendation.core.models import Recommendation


# ==================== フィクスチャ ====================

@pytest.fixture
def sample_engine(sample_members, sample_competence_master,
                  sample_member_competence, sample_similarity):
    """推薦エンジンのフィクスチャ"""
    return RecommendationEngine(
        df_members=sample_members,
        df_competence_master=sample_competence_master,
        df_member_competence=sample_member_competence,
        df_similarity=sample_similarity
    )


# ==================== 初期化テスト ====================

class TestEngineInitialization:
    """推薦エンジン初期化のテスト"""

    def test_engine_initialization(self, sample_members, sample_competence_master,
                                   sample_member_competence, sample_similarity):
        """必要なDataFrameで正常に初期化できる"""
        engine = RecommendationEngine(
            df_members=sample_members,
            df_competence_master=sample_competence_master,
            df_member_competence=sample_member_competence,
            df_similarity=sample_similarity
        )

        assert engine is not None
        assert engine.df_members is not None
        assert engine.df_competence_master is not None
        assert engine.df_member_competence is not None
        assert engine.df_similarity is not None

    def test_engine_initialization_with_custom_weights(self, sample_members,
                                                       sample_competence_master,
                                                       sample_member_competence,
                                                       sample_similarity):
        """カスタム重みで初期化できる"""
        engine = RecommendationEngine(
            df_members=sample_members,
            df_competence_master=sample_competence_master,
            df_member_competence=sample_member_competence,
            df_similarity=sample_similarity,
            category_importance_weight=0.5,
            acquisition_ease_weight=0.3,
            popularity_weight=0.2
        )

        assert engine.category_importance_weight == 0.5
        assert engine.acquisition_ease_weight == 0.3
        assert engine.popularity_weight == 0.2


# ==================== 会員力量取得テスト ====================

class TestGetMemberCompetences:
    """会員保有力量取得のテスト"""

    def test_get_member_competences(self, sample_engine):
        """会員の保有力量が取得できる"""
        # m001は3つの力量を保有
        competences = sample_engine.get_member_competences('m001')

        assert len(competences) == 3
        assert 'm001' in competences['メンバーコード'].values

    def test_get_member_competences_empty(self, sample_engine):
        """力量未保有会員で空データを返す"""
        competences = sample_engine.get_member_competences('m999')

        assert len(competences) == 0

    def test_get_member_competences_multiple(self, sample_engine):
        """複数の力量を保有する会員"""
        competences = sample_engine.get_member_competences('m001')

        # 力量コードを確認
        codes = competences['力量コード'].tolist()
        assert 's001' in codes
        assert 's002' in codes


# ==================== 未習得力量取得テスト ====================

class TestGetUnacquiredCompetences:
    """未習得力量取得のテスト"""

    def test_get_unacquired_competences(self, sample_engine):
        """未習得力量が取得できる"""
        # m001はs001, s002, e001を保有
        unacquired = sample_engine.get_unacquired_competences('m001')

        # s003, l001などが未習得
        assert len(unacquired) > 0
        acquired_codes = ['s001', 's002', 'e001']
        for code in acquired_codes:
            assert code not in unacquired['力量コード'].values

    def test_unacquired_with_type_filter_skill(self, sample_engine):
        """SKILLタイプでフィルタ"""
        unacquired = sample_engine.get_unacquired_competences('m001', competence_type='SKILL')

        # 全てSKILLタイプであること
        assert all(unacquired['力量タイプ'] == 'SKILL')

    def test_unacquired_with_type_filter_education(self, sample_engine):
        """EDUCATIONタイプでフィルタ"""
        unacquired = sample_engine.get_unacquired_competences('m002', competence_type='EDUCATION')

        # 全てEDUCATIONタイプであること
        if len(unacquired) > 0:
            assert all(unacquired['力量タイプ'] == 'EDUCATION')

    def test_unacquired_with_type_filter_license(self, sample_engine):
        """LICENSEタイプでフィルタ"""
        unacquired = sample_engine.get_unacquired_competences('m001', competence_type='LICENSE')

        # 全てLICENSEタイプであること
        if len(unacquired) > 0:
            assert all(unacquired['力量タイプ'] == 'LICENSE')


# ==================== カテゴリ重要度計算テスト ====================

class TestCalculateCategoryImportance:
    """カテゴリ重要度計算のテスト"""

    def test_calculate_category_importance(self, sample_engine):
        """カテゴリ重要度が0-10の範囲内"""
        importance = sample_engine.calculate_category_importance('s001', '技術 > プログラミング')

        assert 0 <= importance <= 10

    def test_category_importance_popular_skill(self, sample_engine):
        """人気のある力量は高いカテゴリ重要度"""
        # s001 (Python)は複数人が習得
        importance = sample_engine.calculate_category_importance('s001', '技術 > プログラミング')

        assert importance > 0

    def test_category_importance_empty_category(self, sample_engine):
        """存在しないカテゴリはデフォルト値5.0"""
        importance = sample_engine.calculate_category_importance('s999', '存在しないカテゴリ')

        assert importance == 5.0

    def test_category_importance_edge_case(self, sample_engine):
        """エッジケース: カテゴリ内習得者ゼロ"""
        # 新しいカテゴリで習得者ゼロのケース
        importance = sample_engine.calculate_category_importance('s999', '技術 > プログラミング')

        # 習得者ゼロの場合は0になるはず
        assert importance == 0.0


# ==================== 習得容易性計算テスト ====================

class TestCalculateAcquisitionEase:
    """習得容易性計算のテスト"""

    def test_calculate_acquisition_ease(self, sample_engine):
        """習得容易性が0-10の範囲内"""
        ease = sample_engine.calculate_acquisition_ease('m001', 's003')

        assert 0 <= ease <= 10

    def test_acquisition_ease_with_similar_skill(self, sample_engine):
        """類似力量を保有している場合は高スコア"""
        # sample_similarityにs001-s003の類似度0.42がある
        # m001はs001を保有
        ease = sample_engine.calculate_acquisition_ease('m001', 's003')

        # 類似力量があるので3.0より高いはず
        assert ease > 3.0

    def test_acquisition_ease_no_similar(self, sample_engine):
        """類似力量なしの場合はデフォルト値3.0"""
        # s999は類似度データに存在しない
        ease = sample_engine.calculate_acquisition_ease('m001', 's999')

        assert ease == 3.0

    def test_acquisition_ease_multiple_similar(self, sample_engine):
        """複数の類似力量がある場合は最大値を使用"""
        # 複数の類似力量がある場合
        ease = sample_engine.calculate_acquisition_ease('m001', 's003')

        # 最大類似度に基づくスコア
        assert ease >= 3.0


# ==================== 人気度計算テスト ====================

class TestCalculatePopularity:
    """人気度計算のテスト"""

    def test_calculate_popularity(self, sample_engine):
        """人気度が0-10の範囲内"""
        popularity = sample_engine.calculate_popularity('s001')

        assert 0 <= popularity <= 10

    def test_popularity_common_skill(self, sample_engine):
        """多くの人が習得している力量は高人気度"""
        # s001は3人が習得（m001, m002, m004）
        popularity = sample_engine.calculate_popularity('s001')

        # 3人/5人 = 0.6 → 6.0
        expected = (3 / 5) * 10
        assert abs(popularity - expected) < 0.1

    def test_popularity_rare_skill(self, sample_engine):
        """習得者が少ない力量は低人気度"""
        # s004は1人のみ（m005）
        popularity = sample_engine.calculate_popularity('s004')

        # 1人/5人 = 0.2 → 2.0
        expected = (1 / 5) * 10
        assert abs(popularity - expected) < 0.1

    def test_popularity_zero_acquirers(self, sample_engine):
        """習得者ゼロの力量は人気度0.0"""
        popularity = sample_engine.calculate_popularity('s999')

        assert popularity == 0.0


# ==================== 優先度スコア計算テスト ====================

class TestCalculatePriorityScore:
    """優先度スコア計算のテスト"""

    def test_calculate_priority_score(self, sample_engine):
        """優先度スコア計算式の検証"""
        # デフォルト重み: category=0.4, ease=0.3, popularity=0.3
        cat_importance = 8.0
        ease = 6.0
        popularity = 5.0

        expected = 8.0 * 0.4 + 6.0 * 0.3 + 5.0 * 0.3
        result = sample_engine.calculate_priority_score(cat_importance, ease, popularity)

        assert abs(result - expected) < 0.01

    def test_priority_score_range(self, sample_engine):
        """優先度スコアが0-10の範囲内"""
        score = sample_engine.calculate_priority_score(10.0, 10.0, 10.0)
        assert score == 10.0

        score = sample_engine.calculate_priority_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_priority_score_weights(self, sample_engine):
        """重みの合計が1.0であること"""
        total_weight = (
            sample_engine.category_importance_weight +
            sample_engine.acquisition_ease_weight +
            sample_engine.popularity_weight
        )

        assert abs(total_weight - 1.0) < 0.01


# ==================== 推薦理由生成テスト ====================

class TestGenerateRecommendationReason:
    """推薦理由生成のテスト"""

    def test_generate_recommendation_reason(self, sample_engine):
        """推薦理由が文字列で返される"""
        reason = sample_engine.generate_recommendation_reason(
            'Python', 'SKILL', '技術 > プログラミング',
            8.0, 7.0, 6.0
        )

        assert isinstance(reason, str)
        assert len(reason) > 0
        assert reason.endswith('。')

    def test_reason_high_category_importance(self, sample_engine):
        """高カテゴリ重要度時の理由文言"""
        reason = sample_engine.generate_recommendation_reason(
            'Python', 'SKILL', '技術 > プログラミング',
            9.0, 5.0, 5.0
        )

        assert '多くの技術者が習得' in reason

    def test_reason_high_acquisition_ease(self, sample_engine):
        """高習得容易性時の理由文言"""
        reason = sample_engine.generate_recommendation_reason(
            'Python', 'SKILL', '技術 > プログラミング',
            5.0, 8.0, 5.0
        )

        assert '類似する力量' in reason or '習得しやすい' in reason

    def test_reason_by_competence_type_skill(self, sample_engine):
        """SKILLタイプの理由文言"""
        reason = sample_engine.generate_recommendation_reason(
            'Python', 'SKILL', '技術 > プログラミング',
            5.0, 5.0, 5.0
        )

        assert 'スキル' in reason

    def test_reason_by_competence_type_education(self, sample_engine):
        """EDUCATIONタイプの理由文言"""
        reason = sample_engine.generate_recommendation_reason(
            'AWS研修', 'EDUCATION', '技術 > クラウド',
            5.0, 5.0, 5.0
        )

        assert '教育' in reason or '研修' in reason

    def test_reason_by_competence_type_license(self, sample_engine):
        """LICENSEタイプの理由文言"""
        reason = sample_engine.generate_recommendation_reason(
            '基本情報技術者', 'LICENSE', '資格 > IT資格',
            5.0, 5.0, 5.0
        )

        assert '資格' in reason


# ==================== 推薦実行テスト ====================

class TestRecommend:
    """推薦実行のテスト"""

    def test_recommend_basic(self, sample_engine):
        """基本的な推薦実行"""
        recommendations = sample_engine.recommend('m001', top_n=5)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        assert all(isinstance(rec, Recommendation) for rec in recommendations)

    def test_recommend_top_n(self, sample_engine):
        """top_nパラメータの動作"""
        recommendations = sample_engine.recommend('m001', top_n=3)

        assert len(recommendations) <= 3

    def test_recommend_sorted_by_priority(self, sample_engine):
        """推薦結果が優先度降順でソートされる"""
        recommendations = sample_engine.recommend('m001', top_n=10)

        if len(recommendations) > 1:
            scores = [rec.priority_score for rec in recommendations]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_with_type_filter_skill(self, sample_engine):
        """SKILLタイプでフィルタ"""
        recommendations = sample_engine.recommend('m001', competence_type='SKILL', top_n=5)

        assert all(rec.competence_type == 'SKILL' for rec in recommendations)

    def test_recommend_with_type_filter_education(self, sample_engine):
        """EDUCATIONタイプでフィルタ"""
        recommendations = sample_engine.recommend('m001', competence_type='EDUCATION', top_n=5)

        assert all(rec.competence_type == 'EDUCATION' for rec in recommendations)

    def test_recommend_with_type_filter_license(self, sample_engine):
        """LICENSEタイプでフィルタ"""
        recommendations = sample_engine.recommend('m001', competence_type='LICENSE', top_n=5)

        assert all(rec.competence_type == 'LICENSE' for rec in recommendations)

    def test_recommend_with_category_filter(self, sample_engine):
        """カテゴリフィルタ適用"""
        recommendations = sample_engine.recommend('m001', category_filter='プログラミング', top_n=5)

        # カテゴリフィルタが適用される
        assert all('プログラミング' in rec.category for rec in recommendations)

    def test_recommend_no_results(self, sample_engine):
        """推薦可能力量なし時に空リスト"""
        # 存在しないカテゴリでフィルタ
        recommendations = sample_engine.recommend('m001', category_filter='存在しないカテゴリ', top_n=10)

        assert recommendations == []

    def test_recommend_all_acquired(self):
        """全力量を習得済みの会員は空リスト"""
        # 全ての力量を習得している会員データを作成
        df_members = pd.DataFrame({'メンバーコード': ['m001']})
        df_competence_master = pd.DataFrame({
            '力量コード': ['s001'],
            '力量名': ['Python'],
            '力量タイプ': ['SKILL'],
            '力量カテゴリー名': ['技術 > プログラミング']
        })
        df_member_competence = pd.DataFrame({
            'メンバーコード': ['m001'],
            '力量コード': ['s001'],
            '正規化レベル': [3],
            '力量タイプ': ['SKILL'],
            '力量カテゴリー名': ['技術 > プログラミング']
        })
        df_similarity = pd.DataFrame()

        engine = RecommendationEngine(
            df_members, df_competence_master, df_member_competence, df_similarity
        )

        recommendations = engine.recommend('m001', top_n=10)

        assert recommendations == []


# ==================== Recommendationオブジェクトテスト ====================

class TestRecommendationObject:
    """推薦結果オブジェクトのテスト"""

    def test_recommendation_object_structure(self, sample_engine):
        """Recommendationオブジェクトの構造確認"""
        recommendations = sample_engine.recommend('m001', top_n=1)

        if len(recommendations) > 0:
            rec = recommendations[0]
            assert hasattr(rec, 'competence_code')
            assert hasattr(rec, 'competence_name')
            assert hasattr(rec, 'competence_type')
            assert hasattr(rec, 'category')
            assert hasattr(rec, 'priority_score')
            assert hasattr(rec, 'category_importance')
            assert hasattr(rec, 'acquisition_ease')
            assert hasattr(rec, 'popularity')
            assert hasattr(rec, 'reason')

    def test_recommendation_scores_valid(self, sample_engine):
        """推薦結果のスコアが妥当な範囲"""
        recommendations = sample_engine.recommend('m001', top_n=5)

        for rec in recommendations:
            assert 0 <= rec.priority_score <= 10
            assert 0 <= rec.category_importance <= 10
            assert 0 <= rec.acquisition_ease <= 10
            assert 0 <= rec.popularity <= 10
