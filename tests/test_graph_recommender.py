"""
SkillTransitionGraphRecommenderのテスト
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from skillnote_recommendation.ml.graph_recommender import SkillTransitionGraphRecommender
from skillnote_recommendation.ml.base_recommender import Recommendation


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータ"""
    # メンバー習得力量データ
    base_date = datetime(2024, 1, 1)
    member_competence = pd.DataFrame(
        {
            "メンバーコード": ["M001"] * 5 + ["M002"] * 5 + ["M003"] * 4,
            "力量コード": [
                # M001: Python → Pandas → 機械学習 → TensorFlow → PyTorch
                "SKILL001",
                "SKILL002",
                "SKILL003",
                "SKILL004",
                "SKILL005",
                # M002: Python → Pandas → 機械学習 → Scikit-learn
                "SKILL001",
                "SKILL002",
                "SKILL003",
                "SKILL006",
                "SKILL007",
                # M003: Python → Django → REST API → Docker
                "SKILL001",
                "SKILL008",
                "SKILL009",
                "SKILL010",
            ],
            "正規化レベル": [3] * 14,
            "取得日": [
                # M001
                (base_date + timedelta(days=0)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=30)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=90)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=150)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=200)).strftime("%Y/%m/%d"),
                # M002
                (base_date + timedelta(days=0)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=25)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=85)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=140)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=180)).strftime("%Y/%m/%d"),
                # M003
                (base_date + timedelta(days=0)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=40)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=100)).strftime("%Y/%m/%d"),
                (base_date + timedelta(days=160)).strftime("%Y/%m/%d"),
            ],
        }
    )

    # 力量マスタ
    competence_master = pd.DataFrame(
        {
            "力量コード": [f"SKILL{i:03d}" for i in range(1, 11)],
            "力量名": [
                "Python基礎",
                "Pandas",
                "機械学習基礎",
                "TensorFlow",
                "PyTorch",
                "Scikit-learn",
                "データ分析",
                "Django",
                "REST API",
                "Docker",
            ],
            "力量カテゴリー名": ["プログラミング"] * 10,
        }
    )

    return member_competence, competence_master


class TestSkillTransitionGraphRecommender:
    """SkillTransitionGraphRecommenderのテストクラス"""

    def test_initialization(self):
        """初期化のテスト"""
        recommender = SkillTransitionGraphRecommender(time_window_days=180, min_transition_count=2)

        assert recommender.name == "SkillTransitionGraph"
        assert recommender.interpretability_score == 4
        assert not recommender.is_fitted

    def test_fit(self, sample_data):
        """学習のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(
            time_window_days=200, min_transition_count=1  # テストデータが少ないので1
        )

        recommender.fit(member_competence, competence_master)

        assert recommender.is_fitted
        assert recommender.graph is not None
        assert recommender.graph.number_of_nodes() > 0
        assert recommender.graph.number_of_edges() > 0

    def test_fit_without_date_column(self, sample_data):
        """取得日なしでの学習（エラーケース）"""
        member_competence, competence_master = sample_data
        member_competence_no_date = member_competence.drop(columns=["取得日"])

        recommender = SkillTransitionGraphRecommender()

        with pytest.raises(ValueError, match="取得日"):
            recommender.fit(member_competence_no_date, competence_master)

    def test_graph_construction(self, sample_data):
        """グラフ構築のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        # Python → Pandas の遷移が存在するはず
        assert recommender.graph.has_edge("SKILL001", "SKILL002")

        # エッジデータの確認
        edge_data = recommender.graph["SKILL001"]["SKILL002"]
        assert "weight" in edge_data
        assert edge_data["weight"] >= 1  # 少なくとも1人は遷移

    def test_recommend(self, sample_data):
        """推薦のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        # M001に推薦
        # M001は Python → Pandas → 機械学習 → TensorFlow → PyTorch を習得済み
        recommendations = recommender.recommend("M001", n=5)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

        if recommendations:
            rec = recommendations[0]
            assert isinstance(rec, Recommendation)
            assert hasattr(rec, "skill_code")
            assert hasattr(rec, "skill_name")
            assert hasattr(rec, "score")
            assert hasattr(rec, "explanation")
            assert hasattr(rec, "confidence")

    def test_recommend_exclude_acquired(self, sample_data):
        """習得済みスキルを除外した推薦のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        user_skills = recommender.get_user_skills("M001")
        recommendations = recommender.recommend("M001", n=10, exclude_acquired=True)

        # 推薦されたスキルが習得済みスキルに含まれていないことを確認
        for rec in recommendations:
            assert rec.skill_code not in user_skills

    def test_explain(self, sample_data):
        """推薦理由説明のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        # M003は Python → Django を習得
        # REST API の推薦理由を取得
        explanation = recommender.explain("M003", "SKILL009")

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_get_learning_path(self, sample_data):
        """学習パス取得のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        # Python → 機械学習 のパスを取得
        path = recommender.get_learning_path("SKILL001", "SKILL003")

        if path:
            assert isinstance(path, list)
            assert path[0] == "SKILL001"
            assert path[-1] == "SKILL003"
            assert len(path) >= 2

    def test_get_graph_statistics(self, sample_data):
        """グラフ統計情報取得のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender(time_window_days=200, min_transition_count=1)

        recommender.fit(member_competence, competence_master)

        stats = recommender.get_graph_statistics()

        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "density" in stats
        assert "avg_in_degree" in stats
        assert "avg_out_degree" in stats

        assert stats["num_nodes"] > 0
        assert stats["num_edges"] > 0

    def test_get_user_skills(self, sample_data):
        """ユーザースキル取得のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender()
        recommender.member_competence = member_competence

        user_skills = recommender.get_user_skills("M001")

        assert isinstance(user_skills, list)
        assert len(user_skills) == 5  # M001は5つのスキルを習得
        assert "SKILL001" in user_skills  # Python基礎

    def test_get_skill_name(self, sample_data):
        """スキル名取得のテスト"""
        member_competence, competence_master = sample_data

        recommender = SkillTransitionGraphRecommender()
        recommender.competence_master = competence_master

        skill_name = recommender.get_skill_name("SKILL001")
        assert skill_name == "Python基礎"

        # 存在しないスキル
        unknown_name = recommender.get_skill_name("UNKNOWN")
        assert unknown_name == "UNKNOWN"

    def test_interpretability_info(self):
        """解釈性情報取得のテスト"""
        recommender = SkillTransitionGraphRecommender()

        info = recommender.get_interpretability_info()

        assert "score" in info
        assert "level" in info
        assert "model_name" in info
        assert info["score"] == 4
        assert info["model_name"] == "SkillTransitionGraph"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
