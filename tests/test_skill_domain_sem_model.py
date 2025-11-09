"""
スキル領域潜在変数SEMモデルのテスト
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.ml.skill_domain_sem_model import (
    SkillDomainSEMModel,
    LatentFactor,
    DomainStructure,
)


@pytest.fixture
def sample_member_competence():
    """サンプルメンバー習得力量データ"""
    return pd.DataFrame({
        "メンバーコード": ["M001", "M001", "M001", "M002", "M002", "M002", "M003", "M003"],
        "力量コード": ["C001", "C002", "C003", "C001", "C004", "C005", "C002", "C006"],
        "正規化レベル": [4, 3, 5, 2, 4, 2, 3, 4],
        "取得日": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2023-01-15",
            "2023-02-15",
            "2023-03-15",
            "2023-01-10",
            "2023-02-10",
        ],
    })


@pytest.fixture
def sample_competence_master():
    """サンプル力量マスタ"""
    return pd.DataFrame({
        "力量コード": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "力量名": [
            "Python基礎",
            "Java基礎",
            "Webアプリ開発",
            "SQL基礎",
            "データベース設計",
            "Git",
        ],
        "力量タイプ": ["SKILL"] * 6,
        "力量カテゴリー名": [
            "プログラミング > Python",
            "プログラミング > Java",
            "プログラミング > Webアプリ",
            "データベース > SQL",
            "データベース > DB設計",
            "プログラミング > Git",
        ],
    })


class TestSkillDomainSEMModel:
    """SkillDomainSEMModelのテスト"""

    def test_initialization(self, sample_member_competence, sample_competence_master):
        """モデルの初期化テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        assert model is not None
        assert len(model.domain_structures) > 0
        assert len(model.member_latent_scores) == 3  # M001, M002, M003

    def test_domain_aggregation(self, sample_member_competence, sample_competence_master):
        """領域集約のテスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        domains = model.get_all_domains()
        assert "プログラミング" in domains
        assert "データベース" in domains

    def test_latent_factor_creation(
        self, sample_member_competence, sample_competence_master
    ):
        """潜在変数の作成テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        for domain_struct in model.domain_structures.values():
            # 各領域は3つの潜在変数を持つ（初級、中級、上級）
            assert len(domain_struct.latent_factors) == 3
            assert domain_struct.latent_factors[0].level == 0  # 初級
            assert domain_struct.latent_factors[1].level == 1  # 中級
            assert domain_struct.latent_factors[2].level == 2  # 上級

    def test_member_latent_scores(
        self, sample_member_competence, sample_competence_master
    ):
        """メンバーの潜在変数スコア推定テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # M001のスコアを確認
        m001_scores = model.member_latent_scores.get("M001", {})
        assert len(m001_scores) > 0
        # スコアは0-1の範囲
        for score in m001_scores.values():
            assert 0 <= score <= 1

    def test_get_direct_effect_skills(
        self, sample_member_competence, sample_competence_master
    ):
        """直接効果スキル推薦テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # M001がプログラミング領域で直接効果スキルを取得
        recs = model.get_direct_effect_skills(
            member_code="M001",
            domain_category="プログラミング",
            top_n=3,
        )

        # 推薦が返される（M001はレベル1まで到達している可能性）
        if recs:
            for rec in recs:
                assert "skill_code" in rec
                assert "skill_name" in rec
                assert "direct_effect_score" in rec
                assert 0 <= rec["direct_effect_score"] <= 1

    def test_get_indirect_support_skills(
        self, sample_member_competence, sample_competence_master
    ):
        """間接効果スキル推薦テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        recs = model.get_indirect_support_skills(
            member_code="M001",
            target_skill="C001",  # Python基礎
            top_n=3,
        )

        # 推薦が返される可能性
        for rec in recs:
            assert "skill_code" in rec
            assert "skill_name" in rec
            assert "indirect_support_score" in rec
            assert 0 <= rec["indirect_support_score"] <= 1

    def test_calculate_sem_score(
        self, sample_member_competence, sample_competence_master
    ):
        """SEMスコア計算テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # M001がPythonスキルを習得する確率
        sem_score = model.calculate_sem_score(
            member_code="M001",
            skill_code="C001",
        )

        assert isinstance(sem_score, float)
        assert 0 <= sem_score <= 1

    def test_get_member_domain_profile(
        self, sample_member_competence, sample_competence_master
    ):
        """メンバー領域プロファイル取得テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        profile = model.get_member_domain_profile("M001")

        assert isinstance(profile, dict)
        assert len(profile) > 0
        # 各領域に複数の潜在変数がある
        for domain_name, domain_scores in profile.items():
            assert isinstance(domain_scores, dict)
            assert len(domain_scores) == 3  # 初級、中級、上級

    def test_domain_info(self, sample_member_competence, sample_competence_master):
        """領域情報取得テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        domain_info = model.get_domain_info("プログラミング")

        assert "domain_name" in domain_info
        assert "num_latent_factors" in domain_info
        assert "latent_factors" in domain_info
        assert "path_coefficients" in domain_info
        assert "domain_reliability" in domain_info

    def test_find_skill_domain(self, sample_member_competence, sample_competence_master):
        """スキル領域検索テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # Python基礎（C001）の領域を検索
        domain = model._find_skill_domain("C001")
        assert domain == "プログラミング"

        # SQL基礎（C004）の領域を検索
        domain = model._find_skill_domain("C004")
        assert domain == "データベース"

        # 存在しないスキル
        domain = model._find_skill_domain("C999")
        assert domain is None

    def test_estimate_current_level(
        self, sample_member_competence, sample_competence_master
    ):
        """現在レベル推定テスト"""
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # M001はプログラミングスキルが高い（レベル3-5）
        level = model._estimate_current_level("M001", "プログラミング")
        assert 0 <= level <= 2  # 0-2の範囲

    def test_num_domain_categories_limit(
        self, sample_member_competence, sample_competence_master
    ):
        """領域数制限のテスト"""
        # 領域数を2に制限
        model = SkillDomainSEMModel(
            member_competence_df=sample_member_competence,
            competence_master_df=sample_competence_master,
            num_domain_categories=2,
        )

        domains = model.get_all_domains()
        assert len(domains) <= 2

    def test_empty_member_competence(self, sample_competence_master):
        """空のメンバー習得力量データのテスト"""
        empty_df = pd.DataFrame({
            "メンバーコード": [],
            "力量コード": [],
            "正規化レベル": [],
            "取得日": [],
        })

        model = SkillDomainSEMModel(
            member_competence_df=empty_df,
            competence_master_df=sample_competence_master,
            num_domain_categories=5,
        )

        # 空のメンバースコアになるはず
        assert len(model.member_latent_scores) == 0
