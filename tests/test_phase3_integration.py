"""
Phase 3 統合テスト - SEMモデルのUI統合検証

Streamlitアプリケーションに統合されたSEMモデルの基本機能をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel
from skillnote_recommendation.ml.ml_sem_recommender import MLSEMRecommender
from skillnote_recommendation.ml.ml_recommender import MLRecommender


@pytest.fixture
def sample_data():
    """テスト用サンプルデータ"""
    member_competence = pd.DataFrame({
        "メンバーコード": ["M001", "M001", "M001", "M002", "M002", "M002", "M003", "M003"],
        "力量コード": ["C001", "C002", "C003", "C001", "C004", "C005", "C002", "C006"],
        "正規化レベル": [4, 3, 5, 2, 4, 2, 3, 4],
        "取得日": [
            "2023-01-01", "2023-02-01", "2023-03-01",
            "2023-01-15", "2023-02-15", "2023-03-15",
            "2023-01-10", "2023-02-10",
        ],
    })

    competence_master = pd.DataFrame({
        "力量コード": ["C001", "C002", "C003", "C004", "C005", "C006"],
        "力量名": [
            "Python基礎", "Java基礎", "Webアプリ開発",
            "SQL基礎", "データベース設計", "Git"
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

    member_master = pd.DataFrame({
        "メンバーコード": ["M001", "M002", "M003"],
        "メンバー名": ["太郎", "花子", "次郎"],
        "職種": ["エンジニア", "エンジニア", "マネージャー"],
        "等級": ["主任", "一般", "課長"],
    })

    return {
        "member_competence": member_competence,
        "competence_master": competence_master,
        "member_master": member_master,
    }


class TestSEMUIIntegration:
    """UI統合機能のテスト"""

    def test_sem_model_initialization(self, sample_data):
        """SEMモデルが正常に初期化される"""
        sem_model = SkillDomainSEMModel(
            member_competence_df=sample_data["member_competence"],
            competence_master_df=sample_data["competence_master"],
            num_domain_categories=5,
        )

        assert sem_model is not None
        assert len(sem_model.domain_structures) > 0
        assert len(sem_model.member_latent_scores) > 0

    def test_ml_sem_recommender_build(self, sample_data):
        """MLSEMRecommenderが正常にビルドされる"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_preprocessing=True,
            use_tuning=False,
            use_sem=True,
            sem_weight=0.2,
            num_domain_categories=8,
        )

        assert recommender is not None
        assert hasattr(recommender, 'sem_model')
        assert recommender.sem_model is not None
        assert recommender.sem_weight == 0.2

    def test_ml_recommender_without_sem(self, sample_data):
        """SEMなしのMLRecommenderもビルドされる（後方互換性）"""
        recommender = MLRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_preprocessing=True,
            use_tuning=False,
        )

        assert recommender is not None
        # SEMなしの場合はsem_modelメンバーがない
        assert not hasattr(recommender, 'sem_model')

    def test_sem_recommender_recommend(self, sample_data):
        """SEMを使用した推薦が実行される"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_sem=True,
            sem_weight=0.2,
        )

        # 推薦を実行
        recommendations = recommender.recommend(
            member_code="M001",
            top_n=5,
            use_sem=True,
            return_explanation=True,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # 説明文にSEM情報が含まれていることを確認
        assert any("SEM" in rec.reason or "領域" in rec.reason for rec in recommendations)

    def test_sem_recommender_disable_sem(self, sample_data):
        """SEMが有効でもuse_sem=Falseで無効化できる"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_sem=True,
            sem_weight=0.2,
        )

        # use_sem=Falseで実行
        recommendations = recommender.recommend(
            member_code="M001",
            top_n=5,
            use_sem=False,  # SEMを無効化
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_sem_weight_variations(self, sample_data):
        """異なるSEM重みで推薦が実行される"""
        weights = [0.05, 0.2, 0.3]

        for weight in weights:
            recommender = MLSEMRecommender.build(
                member_competence=sample_data["member_competence"],
                competence_master=sample_data["competence_master"],
                member_master=sample_data["member_master"],
                use_sem=True,
                sem_weight=weight,
            )

            recommendations = recommender.recommend(
                member_code="M001",
                top_n=5,
                use_sem=True,
            )

            assert len(recommendations) > 0
            assert recommender.sem_weight == weight

    def test_member_profile_display(self, sample_data):
        """メンバープロフィールが取得可能（UI表示用）"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_sem=True,
        )

        # メンバープロフィールを取得
        profile = recommender.sem_model.get_member_domain_profile("M001")

        assert isinstance(profile, dict)
        assert len(profile) > 0
        # 各領域にスコアが含まれていることを確認
        for domain_name, domain_scores in profile.items():
            assert isinstance(domain_scores, dict)
            assert len(domain_scores) > 0
            # スコアが0-1の範囲内か確認
            for score in domain_scores.values():
                assert 0 <= score <= 1

    def test_domain_info_display(self, sample_data):
        """領域情報が取得可能（UI表示用）"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_sem=True,
        )

        domains = recommender.sem_model.get_all_domains()
        assert len(domains) > 0

        for domain in domains:
            info = recommender.sem_model.get_domain_info(domain)
            assert info is not None
            assert 'domain_name' in info
            assert 'latent_factors' in info
            assert 'path_coefficients' in info

            # パス係数の形式を確認
            for path in info['path_coefficients']:
                assert 'coefficient' in path
                assert 'p_value' in path
                assert 'is_significant' in path
                assert 'ci' in path

    def test_sem_score_calculation(self, sample_data):
        """SEMスコア計算が機能する"""
        recommender = MLSEMRecommender.build(
            member_competence=sample_data["member_competence"],
            competence_master=sample_data["competence_master"],
            member_master=sample_data["member_master"],
            use_sem=True,
        )

        # 複数のスキルに対してSEMスコアを計算
        for skill_code in ["C001", "C002", "C003"]:
            sem_score = recommender.sem_model.calculate_sem_score("M001", skill_code)
            assert isinstance(sem_score, float)
            assert 0 <= sem_score <= 1

    def test_num_domain_categories_effect(self, sample_data):
        """スキル領域数の設定が機能する"""
        categories_list = [5, 8, 10]

        for num_categories in categories_list:
            recommender = MLSEMRecommender.build(
                member_competence=sample_data["member_competence"],
                competence_master=sample_data["competence_master"],
                member_master=sample_data["member_master"],
                use_sem=True,
                num_domain_categories=num_categories,
            )

            domains = recommender.sem_model.get_all_domains()
            # 領域数が指定範囲以内か確認
            assert len(domains) <= num_categories


class TestUIComponentsExist:
    """UI コンポーネントの存在確認"""

    def test_training_page_has_sem_option(self):
        """pages/2_Model_Training.pyにSEMオプションが存在"""
        with open("pages/2_Model_Training.py", "r", encoding="utf-8") as f:
            content = f.read()
            assert "use_sem" in content
            assert "sem_weight" in content
            assert "num_domain_categories" in content
            assert "MLSEMRecommender" in content
            assert "SEM（スキル依存性分析）" in content

    def test_inference_page_has_sem_analysis(self):
        """pages/4_Inference.pyにSEM分析表示が存在"""
        with open("pages/4_Inference.py", "r", encoding="utf-8") as f:
            content = f.read()
            assert "sem_model" in content
            assert "get_member_domain_profile" in content
            assert "get_domain_info" in content
            assert "path_coefficients" in content
            assert "SEM分析（スキル依存性分析）" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
