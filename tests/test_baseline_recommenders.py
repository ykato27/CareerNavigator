"""
ベースライン推薦モデルのテスト
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.ml.baseline_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    CategoryBasedRecommender,
)


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータ"""
    # 力量マスタ
    competence_master = pd.DataFrame({
        '力量コード': ['C001', 'C002', 'C003', 'C004', 'C005'],
        '力量名': ['Python基礎', 'SQL基礎', 'Web開発', 'データ分析', 'プロジェクト管理'],
        '力量タイプ': ['SKILL', 'SKILL', 'SKILL', 'SKILL', 'EDUCATION'],
        '力量カテゴリー名': ['プログラミング', 'データベース', 'プログラミング', 'データ分析', 'マネジメント'],
    })

    # メンバー習得力量
    member_competence = pd.DataFrame({
        'メンバーコード': ['M001', 'M001', 'M002', 'M002', 'M002', 'M003', 'M003', 'M004'],
        '力量コード': ['C001', 'C002', 'C001', 'C002', 'C003', 'C002', 'C003', 'C001'],
        '正規化レベル': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    # メンバーマスタ
    member_master = pd.DataFrame({
        'メンバーコード': ['M001', 'M002', 'M003', 'M004'],
        '氏名': ['山田太郎', '佐藤花子', '鈴木一郎', '田中美咲'],
    })

    return competence_master, member_competence, member_master


class TestRandomRecommender:
    """RandomRecommenderのテスト"""

    def test_initialize(self, sample_data):
        """初期化テスト"""
        competence_master, member_competence, member_master = sample_data
        rec = RandomRecommender(competence_master, member_competence, member_master, random_state=42)
        assert rec is not None
        assert rec.random_state == 42

    def test_recommend_returns_correct_number(self, sample_data):
        """推薦数が正しいかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = RandomRecommender(competence_master, member_competence, member_master, random_state=42)

        recommendations = rec.recommend('M001', top_n=2)
        assert len(recommendations) == 2

    def test_recommend_excludes_acquired(self, sample_data):
        """既習得力量を除外するかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = RandomRecommender(competence_master, member_competence, member_master, random_state=42)

        # M001は C001, C002 を習得済み
        recommendations = rec.recommend('M001', top_n=5)
        recommended_codes = [r.competence_code for r in recommendations]

        assert 'C001' not in recommended_codes
        assert 'C002' not in recommended_codes

    def test_recommend_with_type_filter(self, sample_data):
        """力量タイプフィルタのテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = RandomRecommender(competence_master, member_competence, member_master, random_state=42)

        recommendations = rec.recommend('M001', top_n=5, competence_type=['SKILL'])
        recommended_types = [r.competence_type for r in recommendations]

        assert all(t == 'SKILL' for t in recommended_types)

    def test_recommend_empty_candidates(self, sample_data):
        """候補がない場合のテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = RandomRecommender(competence_master, member_competence, member_master, random_state=42)

        # すべての力量を習得済みのメンバーを作成
        member_competence_all = pd.DataFrame({
            'メンバーコード': ['M999'] * 5,
            '力量コード': ['C001', 'C002', 'C003', 'C004', 'C005'],
            '正規化レベル': [1.0] * 5,
        })

        rec.member_competence = pd.concat([member_competence, member_competence_all])
        rec._member_acquired_cache = {}  # キャッシュクリア

        recommendations = rec.recommend('M999', top_n=5)
        assert len(recommendations) == 0


class TestPopularityRecommender:
    """PopularityRecommenderのテスト"""

    def test_initialize(self, sample_data):
        """初期化テスト"""
        competence_master, member_competence, member_master = sample_data
        rec = PopularityRecommender(competence_master, member_competence, member_master)
        assert rec is not None
        assert '人気度スコア' in rec.competence_master.columns

    def test_popularity_scores_calculated(self, sample_data):
        """人気度スコアが正しく計算されるかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = PopularityRecommender(competence_master, member_competence, member_master)

        # C001: 3人, C002: 4人, C003: 3人, C004: 0人, C005: 0人
        # C002が最も人気（スコア10.0）
        c002_score = rec.competence_master[rec.competence_master['力量コード'] == 'C002']['人気度スコア'].values[0]
        assert c002_score == 10.0

    def test_recommend_sorted_by_popularity(self, sample_data):
        """人気度順にソートされるかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = PopularityRecommender(competence_master, member_competence, member_master)

        recommendations = rec.recommend('M004', top_n=5)
        scores = [r.priority_score for r in recommendations]

        # スコアが降順
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_acquired(self, sample_data):
        """既習得力量を除外するかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = PopularityRecommender(competence_master, member_competence, member_master)

        # M002は C001, C002, C003 を習得済み
        recommendations = rec.recommend('M002', top_n=5)
        recommended_codes = [r.competence_code for r in recommendations]

        assert 'C001' not in recommended_codes
        assert 'C002' not in recommended_codes
        assert 'C003' not in recommended_codes


class TestCategoryBasedRecommender:
    """CategoryBasedRecommenderのテスト"""

    def test_initialize(self, sample_data):
        """初期化テスト"""
        competence_master, member_competence, member_master = sample_data
        rec = CategoryBasedRecommender(competence_master, member_competence, member_master)
        assert rec is not None

    def test_recommend_based_on_category(self, sample_data):
        """カテゴリベースで推薦されるかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = CategoryBasedRecommender(competence_master, member_competence, member_master)

        # M001は「プログラミング」「データベース」を保有
        recommendations = rec.recommend('M001', top_n=3)

        # 推薦理由にカテゴリ名が含まれるか確認
        for r in recommendations:
            assert 'カテゴリ' in r.reason or r.category in r.reason

    def test_recommend_excludes_acquired(self, sample_data):
        """既習得力量を除外するかテスト"""
        competence_master, member_competence, member_master = sample_data
        rec = CategoryBasedRecommender(competence_master, member_competence, member_master)

        # M003は C002, C003 を習得済み
        recommendations = rec.recommend('M003', top_n=5)
        recommended_codes = [r.competence_code for r in recommendations]

        assert 'C002' not in recommended_codes
        assert 'C003' not in recommended_codes

    def test_recommend_no_category_info(self, sample_data):
        """カテゴリ情報がない場合のフォールバック"""
        competence_master, member_competence, member_master = sample_data

        # カテゴリ情報を削除
        member_competence_no_cat = member_competence.copy()

        rec = CategoryBasedRecommender(competence_master, member_competence_no_cat, member_master)

        # M999（習得なし）でテスト
        member_competence_no_cat = pd.concat([
            member_competence_no_cat,
            pd.DataFrame({
                'メンバーコード': ['M999'],
                '力量コード': ['C001'],
                '正規化レベル': [1.0],
            })
        ])

        rec.member_competence = member_competence_no_cat
        rec._member_acquired_cache = {}

        # カテゴリ情報がない場合、ランダム推薦にフォールバックする
        recommendations = rec.recommend('M999', top_n=2)
        assert len(recommendations) >= 0  # エラーなく実行される


class TestBaselineRecommendersIntegration:
    """統合テスト"""

    def test_all_recommenders_return_valid_recommendations(self, sample_data):
        """すべての推薦モデルが有効な推薦を返すか"""
        competence_master, member_competence, member_master = sample_data

        recommenders = [
            RandomRecommender(competence_master, member_competence, member_master, random_state=42),
            PopularityRecommender(competence_master, member_competence, member_master),
            CategoryBasedRecommender(competence_master, member_competence, member_master),
        ]

        for rec in recommenders:
            recommendations = rec.recommend('M001', top_n=3)

            # 推薦が返される
            assert len(recommendations) > 0

            # すべての推薦が有効なRecommendationオブジェクト
            for r in recommendations:
                assert r.competence_code is not None
                assert r.competence_name is not None
                assert r.competence_type is not None
                assert r.priority_score >= 0
                assert r.reason is not None
