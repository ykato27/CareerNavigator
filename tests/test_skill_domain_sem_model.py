"""
SkillDomainSEMModelのテスト

スキル領域SEMモデルの学習と推薦機能のテスト
"""

import pytest
import pandas as pd
import numpy as np

from skillnote_recommendation.ml.skill_domain_hierarchy import SkillDomainHierarchy
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータ"""
    np.random.seed(42)

    # 力量マスタ
    competence_master = pd.DataFrame({
        '力量コード': [
            'C001', 'C002', 'C003',  # プログラミング Level 1
            'C004', 'C005', 'C006',  # プログラミング Level 2
            'C007', 'C008',          # プログラミング Level 3
            'C009', 'C010', 'C011',  # データベース Level 1
            'C012', 'C013',          # データベース Level 2
        ],
        '力量名': [
            'Python基礎', 'Java基礎', 'Git',
            'Web開発', 'API開発', 'テスト',
            'システム設計', 'アーキテクチャ',
            'SQL基礎', 'データベース基礎', 'SELECT',
            'JOIN', 'インデックス',
        ],
    })

    # メンバー力量データ
    member_competence_data = []

    # M001: プログラミング Level 1を高スコアで習得
    for comp in ['C001', 'C002', 'C003']:
        member_competence_data.append({
            'メンバーコード': 'M001',
            '力量コード': comp,
            '正規化レベル': np.random.uniform(0.8, 1.0),
        })

    # M002: プログラミング Level 1 + Level 2を習得
    for comp in ['C001', 'C002', 'C003', 'C004', 'C005']:
        member_competence_data.append({
            'メンバーコード': 'M002',
            '力量コード': comp,
            '正規化レベル': np.random.uniform(0.7, 0.9),
        })

    # M003: データベース Level 1を習得
    for comp in ['C009', 'C010', 'C011']:
        member_competence_data.append({
            'メンバーコード': 'M003',
            '力量コード': comp,
            '正規化レベル': np.random.uniform(0.6, 0.8),
        })

    # M004: 広く浅く習得
    for comp in ['C001', 'C004', 'C009', 'C012']:
        member_competence_data.append({
            'メンバーコード': 'M004',
            '力量コード': comp,
            '正規化レベル': np.random.uniform(0.4, 0.6),
        })

    # 追加メンバー（データ量を増やす）
    for i in range(5, 21):
        n_skills = np.random.randint(3, 8)
        selected_comps = np.random.choice(
            competence_master['力量コード'].tolist(),
            n_skills,
            replace=False
        )
        for comp in selected_comps:
            member_competence_data.append({
                'メンバーコード': f'M{i:03d}',
                '力量コード': comp,
                '正規化レベル': np.random.uniform(0.3, 0.9),
            })

    member_competence = pd.DataFrame(member_competence_data)

    return competence_master, member_competence


class TestSkillDomainSEMModelInit:
    """初期化のテスト"""

    def test_initialization(self, sample_data):
        """初期化が正常に行われるかテスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        assert model is not None
        assert model.domain_hierarchy is not None
        assert not model.is_fitted

    def test_custom_domain_hierarchy(self, sample_data):
        """カスタムドメイン階層を使用できるかテスト"""
        competence_master, member_competence = sample_data

        custom_hierarchy = SkillDomainHierarchy(competence_master)

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
            domain_hierarchy=custom_hierarchy,
        )

        assert model.domain_hierarchy == custom_hierarchy


class TestSEMModelFitting:
    """SEMモデル学習のテスト"""

    def test_fit_basic(self, sample_data):
        """基本的な学習テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        # 学習実行（エラーなく完了）
        model.fit(min_competences_per_level=3)

        assert model.is_fitted
        assert len(model.sem_models) > 0

    def test_fit_specific_domains(self, sample_data):
        """特定ドメインのみ学習するテスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        # プログラミングのみ学習
        model.fit(domains=['プログラミング'], min_competences_per_level=3)

        assert model.is_fitted
        assert 'プログラミング' in model.sem_models

    def test_fit_with_insufficient_data(self):
        """データ不足の場合のテスト"""
        # 少量データ
        competence_master = pd.DataFrame({
            '力量コード': ['C001', 'C002'],
            '力量名': ['Python基礎', 'Web開発'],
        })

        member_competence = pd.DataFrame({
            'メンバーコード': ['M001'],
            '力量コード': ['C001'],
            '正規化レベル': [0.8],
        })

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        # データ不足でもエラーなく完了（ただしモデルは学習されない）
        model.fit(min_competences_per_level=3)

        assert model.is_fitted
        # SEMモデルは学習されていない（データ不足）
        assert len(model.sem_models) == 0


class TestLatentScoreEstimation:
    """潜在変数スコア推定のテスト"""

    def test_latent_scores_estimated(self, sample_data):
        """潜在変数スコアが推定されるかテスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # 潜在変数スコアが推定されている
        assert len(model.latent_scores) > 0

        # M001のスコアが存在
        assert 'M001' in model.latent_scores

    def test_get_member_latent_score(self, sample_data):
        """メンバーの潜在変数スコア取得テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # M001のプログラミング Level 1スコアを取得
        score = model.get_member_latent_score('M001', 'プログラミング', 1)

        # スコアが0.0～1.0の範囲
        assert 0.0 <= score <= 1.0

        # M001はプログラミング Level 1を高スコアで習得しているはず
        assert score > 0.7

    def test_latent_score_for_nonexistent_member(self, sample_data):
        """存在しないメンバーのスコア取得テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # 存在しないメンバー → 0.0
        score = model.get_member_latent_score('M999', 'プログラミング', 1)
        assert score == 0.0


class TestRecommendation:
    """推薦機能のテスト"""

    def test_recommend_next_skills(self, sample_data):
        """次のスキル推薦テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # M001の推薦（プログラミング Level 1を高スコアで習得済み）
        recommendations = model.recommend_next_skills(
            member_code='M001',
            top_n=5,
            min_current_level_score=0.6
        )

        # 推薦が生成される
        assert len(recommendations) > 0

        # 推薦結果の構造
        for rec in recommendations:
            assert 'competence_code' in rec
            assert 'competence_name' in rec
            assert 'domain' in rec
            assert 'level' in rec
            assert 'score' in rec
            assert 'reason' in rec

    def test_recommend_excludes_acquired(self, sample_data):
        """既習得スキルを除外するかテスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        recommendations = model.recommend_next_skills(
            member_code='M001',
            top_n=10,
            min_current_level_score=0.6
        )

        recommended_codes = [r['competence_code'] for r in recommendations]

        # M001が既に習得している力量は推薦されない
        assert 'C001' not in recommended_codes  # Python基礎
        assert 'C002' not in recommended_codes  # Java基礎
        assert 'C003' not in recommended_codes  # Git

    def test_recommend_higher_level_skills(self, sample_data):
        """より高いレベルのスキルが推薦されるかテスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # M001の推薦（Level 1を高スコアで習得 → Level 2を推薦）
        recommendations = model.recommend_next_skills(
            member_code='M001',
            top_n=5,
            min_current_level_score=0.7
        )

        if len(recommendations) > 0:
            # 推薦されるスキルのレベルが2以上
            levels = [r['level'] for r in recommendations]
            assert any(level >= 2 for level in levels)

    def test_recommend_with_high_threshold(self, sample_data):
        """高い閾値での推薦テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # 閾値を高く設定（0.9）
        recommendations_high = model.recommend_next_skills(
            member_code='M001',
            top_n=10,
            min_current_level_score=0.9
        )

        # 閾値を低く設定（0.5）
        recommendations_low = model.recommend_next_skills(
            member_code='M001',
            top_n=10,
            min_current_level_score=0.5
        )

        # 閾値が高いほど推薦数が減る（または同じ）
        assert len(recommendations_high) <= len(recommendations_low)


class TestSkillProfile:
    """スキルプロファイル取得のテスト"""

    def test_get_member_skill_profile(self, sample_data):
        """メンバーのスキルプロファイル取得テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # M001のプロファイルを取得
        profile_df = model.get_member_skill_profile('M001')

        # DataFrameが返される
        assert isinstance(profile_df, pd.DataFrame)

        # 必須カラムが存在
        if len(profile_df) > 0:
            assert 'Domain' in profile_df.columns
            assert 'Level_1_Score' in profile_df.columns
            assert 'Level_2_Score' in profile_df.columns
            assert 'Level_3_Score' in profile_df.columns

            # スコアが0.0～1.0の範囲
            assert (profile_df['Level_1_Score'] >= 0.0).all()
            assert (profile_df['Level_1_Score'] <= 1.0).all()

    def test_profile_for_nonexistent_member(self, sample_data):
        """存在しないメンバーのプロファイル取得テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        model.fit(min_competences_per_level=3)

        # 存在しないメンバー → 空のDataFrame
        profile_df = model.get_member_skill_profile('M999')
        assert isinstance(profile_df, pd.DataFrame)
        assert len(profile_df) == 0


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_recommend_before_fitting(self, sample_data):
        """学習前の推薦テスト"""
        competence_master, member_competence = sample_data

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        # 学習前に推薦を実行
        recommendations = model.recommend_next_skills(
            member_code='M001',
            top_n=5
        )

        # 空リストが返される（エラーにはならない）
        assert len(recommendations) == 0

    def test_fit_with_single_member(self):
        """単一メンバーでの学習テスト"""
        competence_master = pd.DataFrame({
            '力量コード': ['C001', 'C002', 'C003', 'C004'],
            '力量名': ['Python基礎', 'Java基礎', 'Git', 'Web開発'],
        })

        member_competence = pd.DataFrame({
            'メンバーコード': ['M001', 'M001', 'M001'],
            '力量コード': ['C001', 'C002', 'C003'],
            '正規化レベル': [0.8, 0.7, 0.9],
        })

        model = SkillDomainSEMModel(
            member_competence=member_competence,
            competence_master=competence_master,
        )

        # エラーなく実行される（ただしSEMモデルは学習されない可能性）
        model.fit(min_competences_per_level=2)

        assert model.is_fitted
