"""
特徴量エンジニアリングのテスト

FeatureEngineerクラスの各機能をテスト
- エンコーディング機能
- 時系列パターン抽出
- 共起関係計算
- カテゴリ埋め込み
- 親和性計算
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skillnote_recommendation.ml.feature_engineering import FeatureEngineer


# ==================== フィクスチャ ====================


@pytest.fixture
def extended_member_master():
    """拡張メンバーマスタ（role, gradeを含む）"""
    return pd.DataFrame(
        {
            "member_code": ["m001", "m002", "m003", "m004", "m005"],
            "name": ["田中太郎", "鈴木花子", "佐藤次郎", "高橋美咲", "伊藤健一"],
            "role": ["Engineer", "Manager", "Engineer", "Engineer", "Lead"],
            "grade": ["G3", "G4", "G2", "G3", "G3"],
        }
    )


@pytest.fixture
def extended_competence_master():
    """拡張力量マスタ（category, typeを含む）"""
    return pd.DataFrame(
        {
            "competence_code": ["s001", "s002", "s003", "s004", "s005", "s006"],
            "name": ["Python", "SQL", "JavaScript", "Docker", "Git", "AWS"],
            "competence_type": ["SKILL", "SKILL", "SKILL", "SKILL", "SKILL", "EDUCATION"],
            "category": [
                "Programming",
                "Database",
                "Programming",
                "Infrastructure",
                "Infrastructure",
                "Cloud",
            ],
        }
    )


@pytest.fixture
def extended_member_competence():
    """拡張メンバー習得力量データ（acquired_dateを含む）"""
    base_date = datetime(2024, 1, 1)
    return pd.DataFrame(
        {
            "member_code": [
                "m001",
                "m001",
                "m001",
                "m002",
                "m002",
                "m003",
                "m003",
                "m003",
                "m004",
                "m005",
                "m005",
            ],
            "competence_code": [
                "s001",
                "s002",
                "s003",
                "s001",
                "s002",
                "s002",
                "s003",
                "s004",
                "s001",
                "s004",
                "s005",
            ],
            "level": [3, 4, 2, 2, 3, 5, 3, 4, 4, 3, 4],
            "acquired_date": [
                base_date,
                base_date + timedelta(days=30),
                base_date + timedelta(days=60),
                base_date + timedelta(days=10),
                base_date + timedelta(days=40),
                base_date + timedelta(days=20),
                base_date + timedelta(days=50),
                base_date + timedelta(days=80),
                base_date + timedelta(days=5),
                base_date + timedelta(days=15),
                base_date + timedelta(days=25),
            ],
        }
    )


@pytest.fixture
def category_hierarchy():
    """カテゴリ階層構造"""
    return {
        "Technology": ["Programming", "Database", "Infrastructure"],
        "Programming": ["Python", "JavaScript"],
        "Infrastructure": ["Docker", "Git"],
    }


@pytest.fixture
def feature_engineer(
    extended_member_master, extended_competence_master, extended_member_competence, category_hierarchy
):
    """FeatureEngineerインスタンス"""
    return FeatureEngineer(
        member_master=extended_member_master,
        competence_master=extended_competence_master,
        member_competence=extended_member_competence,
        category_hierarchy=category_hierarchy,
    )


@pytest.fixture
def minimal_member_master():
    """最小限のメンバーマスタ（roleとgradeなし）"""
    return pd.DataFrame(
        {
            "member_code": ["m001", "m002"],
            "name": ["田中太郎", "鈴木花子"],
        }
    )


@pytest.fixture
def minimal_competence_master():
    """最小限の力量マスタ"""
    return pd.DataFrame(
        {
            "competence_code": ["s001", "s002"],
            "name": ["Python", "SQL"],
        }
    )


@pytest.fixture
def minimal_member_competence():
    """最小限のメンバー習得力量データ（acquired_dateなし）"""
    return pd.DataFrame(
        {
            "member_code": ["m001", "m002"],
            "competence_code": ["s001", "s002"],
            "level": [3, 4],
        }
    )


# ==================== 初期化テスト ====================


class TestInitialization:
    """FeatureEngineerの初期化テスト"""

    def test_initialization_with_full_data(self, feature_engineer):
        """完全なデータでの初期化"""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, "role_encoder")
        assert hasattr(feature_engineer, "grade_encoder")
        assert hasattr(feature_engineer, "category_encoder")
        assert hasattr(feature_engineer, "type_encoder")

    def test_initialization_with_minimal_data(
        self, minimal_member_master, minimal_competence_master, minimal_member_competence
    ):
        """最小限のデータでの初期化"""
        fe = FeatureEngineer(
            member_master=minimal_member_master,
            competence_master=minimal_competence_master,
            member_competence=minimal_member_competence,
        )
        assert fe is not None

    def test_category_hierarchy_stored(self, feature_engineer, category_hierarchy):
        """カテゴリ階層が正しく保存される"""
        assert feature_engineer.category_hierarchy == category_hierarchy

    def test_competence_cooccurrence_computed(self, feature_engineer):
        """共起関係が計算される"""
        assert isinstance(feature_engineer.competence_cooccurrence, dict)
        assert len(feature_engineer.competence_cooccurrence) > 0

    def test_category_embeddings_computed(self, feature_engineer):
        """カテゴリ埋め込みが計算される"""
        assert isinstance(feature_engineer.category_embeddings, dict)


# ==================== エンコーディングテスト ====================


class TestEncoding:
    """エンコーディング機能のテスト"""

    def test_encode_member_attributes_basic(self, feature_engineer):
        """メンバー属性の基本エンコーディング"""
        encoded = feature_engineer.encode_member_attributes("m001")
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0
        # ワンホットエンコーディングなので、0と1のみ
        assert all(x in [0, 1, 0.0, 1.0] for x in encoded)

    def test_encode_member_attributes_unknown_member(self, feature_engineer):
        """未知のメンバーのエンコーディング"""
        encoded = feature_engineer.encode_member_attributes("m999")
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0

    def test_encode_member_attributes_shape(self, feature_engineer):
        """エンコードされたベクトルの形状が一貫している"""
        encoded1 = feature_engineer.encode_member_attributes("m001")
        encoded2 = feature_engineer.encode_member_attributes("m002")
        assert encoded1.shape == encoded2.shape

    def test_encode_competence_attributes_basic(self, feature_engineer):
        """力量属性の基本エンコーディング"""
        encoded = feature_engineer.encode_competence_attributes("s001")
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0
        assert all(x in [0, 1, 0.0, 1.0] for x in encoded)

    def test_encode_competence_attributes_unknown_competence(self, feature_engineer):
        """未知の力量のエンコーディング"""
        encoded = feature_engineer.encode_competence_attributes("s999")
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0

    def test_encode_competence_attributes_shape(self, feature_engineer):
        """エンコードされたベクトルの形状が一貫している"""
        encoded1 = feature_engineer.encode_competence_attributes("s001")
        encoded2 = feature_engineer.encode_competence_attributes("s002")
        assert encoded1.shape == encoded2.shape


# ==================== 時系列パターン抽出テスト ====================


class TestTemporalPatterns:
    """時系列パターン抽出のテスト"""

    def test_extract_temporal_patterns_basic(self, feature_engineer):
        """基本的な時系列パターン抽出"""
        patterns = feature_engineer.extract_temporal_patterns("m001")
        assert isinstance(patterns, dict)
        assert "acquisition_rate" in patterns
        assert "recent_activity" in patterns
        assert "skill_variety" in patterns
        assert "category_focus" in patterns
        assert "learning_velocity" in patterns

    def test_temporal_patterns_value_types(self, feature_engineer):
        """時系列パターンの値の型が正しい"""
        patterns = feature_engineer.extract_temporal_patterns("m001")
        for key, value in patterns.items():
            assert isinstance(value, (float, int))

    def test_temporal_patterns_value_ranges(self, feature_engineer):
        """時系列パターンの値の範囲が妥当"""
        patterns = feature_engineer.extract_temporal_patterns("m001")
        # 負の値がないことを確認
        assert patterns["acquisition_rate"] >= 0
        assert patterns["recent_activity"] >= 0
        assert patterns["learning_velocity"] >= 0
        # 多様性と集中度は0-1の範囲
        assert 0 <= patterns["skill_variety"] <= 1
        assert 0 <= patterns["category_focus"] <= 1

    def test_temporal_patterns_unknown_member(self, feature_engineer):
        """未知のメンバーの時系列パターン"""
        patterns = feature_engineer.extract_temporal_patterns("m999")
        assert patterns["acquisition_rate"] == 0.0
        assert patterns["recent_activity"] == 0.0
        assert patterns["skill_variety"] == 0.0
        assert patterns["category_focus"] == 0.0
        assert patterns["learning_velocity"] == 0.0

    def test_temporal_patterns_multiple_members(self, feature_engineer):
        """複数メンバーで一貫した結果が得られる"""
        patterns1 = feature_engineer.extract_temporal_patterns("m001")
        patterns2 = feature_engineer.extract_temporal_patterns("m002")
        assert patterns1.keys() == patterns2.keys()


# ==================== 共起関係テスト ====================


class TestCooccurrence:
    """力量共起関係のテスト"""

    def test_competence_cooccurrence_structure(self, feature_engineer):
        """共起関係の構造が正しい"""
        cooccurrence = feature_engineer.competence_cooccurrence
        assert isinstance(cooccurrence, dict)
        for comp_code, related in cooccurrence.items():
            assert isinstance(related, dict)
            for other_comp, score in related.items():
                assert isinstance(score, float)
                assert 0 <= score <= 1

    def test_get_competence_cooccurrence_score_basic(self, feature_engineer):
        """共起スコアの基本取得"""
        score = feature_engineer.get_competence_cooccurrence_score("s001", "s002")
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_get_competence_cooccurrence_score_symmetric(self, feature_engineer):
        """共起スコアが対称である"""
        score1 = feature_engineer.get_competence_cooccurrence_score("s001", "s002")
        score2 = feature_engineer.get_competence_cooccurrence_score("s002", "s001")
        assert score1 == score2

    def test_get_competence_cooccurrence_score_unknown(self, feature_engineer):
        """未知の力量の共起スコア"""
        score = feature_engineer.get_competence_cooccurrence_score("s999", "s998")
        assert score == 0.0

    def test_get_related_competences_basic(self, feature_engineer):
        """関連力量の基本取得"""
        related = feature_engineer.get_related_competences("s001", top_k=5)
        assert isinstance(related, list)
        assert len(related) <= 5
        for comp_code, score in related:
            assert isinstance(comp_code, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_get_related_competences_sorted(self, feature_engineer):
        """関連力量がスコア順にソートされている"""
        related = feature_engineer.get_related_competences("s001", top_k=10)
        scores = [score for _, score in related]
        assert scores == sorted(scores, reverse=True)

    def test_get_related_competences_limit(self, feature_engineer):
        """top_kパラメータが正しく機能する"""
        related_3 = feature_engineer.get_related_competences("s001", top_k=3)
        related_5 = feature_engineer.get_related_competences("s001", top_k=5)
        assert len(related_3) <= 3
        assert len(related_5) <= 5


# ==================== カテゴリ埋め込みテスト ====================


class TestCategoryEmbeddings:
    """カテゴリ埋め込みのテスト"""

    def test_category_embeddings_structure(self, feature_engineer):
        """カテゴリ埋め込みの構造が正しい"""
        embeddings = feature_engineer.category_embeddings
        assert isinstance(embeddings, dict)
        for category, embedding in embeddings.items():
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32

    def test_get_category_embedding_basic(self, feature_engineer):
        """カテゴリ埋め込みの基本取得"""
        embedding = feature_engineer.get_category_embedding("Programming")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 16  # デフォルトの埋め込み次元数

    def test_get_category_embedding_unknown(self, feature_engineer):
        """未知のカテゴリの埋め込み"""
        embedding = feature_engineer.get_category_embedding("Unknown")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 16
        assert np.all(embedding == 0)

    def test_category_embedding_consistency(self, feature_engineer):
        """同じカテゴリの埋め込みが一貫している"""
        embedding1 = feature_engineer.get_category_embedding("Programming")
        embedding2 = feature_engineer.get_category_embedding("Programming")
        assert np.array_equal(embedding1, embedding2)


# ==================== 親和性計算テスト ====================


class TestAffinityComputation:
    """親和性計算のテスト"""

    def test_compute_member_competence_affinity_basic(self, feature_engineer):
        """基本的な親和性計算"""
        affinity = feature_engineer.compute_member_competence_affinity("m001", "s004")
        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_affinity_for_acquired_competence(self, feature_engineer):
        """習得済み力量との親和性"""
        # m001はs001を習得済み
        affinity = feature_engineer.compute_member_competence_affinity("m001", "s001")
        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_affinity_for_unacquired_competence(self, feature_engineer):
        """未習得力量との親和性"""
        # m001はs004を未習得
        affinity = feature_engineer.compute_member_competence_affinity("m001", "s004")
        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_affinity_unknown_member(self, feature_engineer):
        """未知のメンバーの親和性"""
        affinity = feature_engineer.compute_member_competence_affinity("m999", "s001")
        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_affinity_unknown_competence(self, feature_engineer):
        """未知の力量との親和性"""
        affinity = feature_engineer.compute_member_competence_affinity("m001", "s999")
        assert isinstance(affinity, float)
        assert 0 <= affinity <= 1

    def test_affinity_values_bounded(self, feature_engineer):
        """親和性スコアが0-1の範囲内"""
        members = ["m001", "m002", "m003", "m004", "m005"]
        competences = ["s001", "s002", "s003", "s004", "s005"]
        for member in members:
            for comp in competences:
                affinity = feature_engineer.compute_member_competence_affinity(member, comp)
                assert 0 <= affinity <= 1


# ==================== バッチ処理テスト ====================


class TestBatchProcessing:
    """バッチ処理のテスト"""

    def test_compute_batch_affinity_basic(self, feature_engineer):
        """基本的なバッチ親和性計算"""
        competences = ["s001", "s002", "s003"]
        affinities = feature_engineer.compute_batch_affinity("m001", competences)
        assert isinstance(affinities, dict)
        assert len(affinities) == 3
        assert all(comp in affinities for comp in competences)

    def test_compute_batch_affinity_values(self, feature_engineer):
        """バッチ親和性計算の値が正しい"""
        competences = ["s001", "s002", "s003"]
        affinities = feature_engineer.compute_batch_affinity("m001", competences)
        for comp, affinity in affinities.items():
            assert isinstance(affinity, float)
            assert 0 <= affinity <= 1

    def test_compute_batch_affinity_consistency(self, feature_engineer):
        """バッチ計算と単一計算の一貫性"""
        competences = ["s001", "s002"]
        batch_affinities = feature_engineer.compute_batch_affinity("m001", competences)
        single_affinity_1 = feature_engineer.compute_member_competence_affinity("m001", "s001")
        single_affinity_2 = feature_engineer.compute_member_competence_affinity("m001", "s002")

        assert batch_affinities["s001"] == single_affinity_1
        assert batch_affinities["s002"] == single_affinity_2

    def test_compute_batch_affinity_empty_list(self, feature_engineer):
        """空のリストでのバッチ計算"""
        affinities = feature_engineer.compute_batch_affinity("m001", [])
        assert affinities == {}


# ==================== エッジケーステスト ====================


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_member_competence(
        self, extended_member_master, extended_competence_master
    ):
        """メンバー習得力量が空の場合"""
        empty_mc = pd.DataFrame(columns=["member_code", "competence_code", "level"])
        fe = FeatureEngineer(
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_competence=empty_mc,
        )
        patterns = fe.extract_temporal_patterns("m001")
        assert all(value == 0.0 for value in patterns.values())

    def test_single_competence_member(
        self, extended_member_master, extended_competence_master
    ):
        """1つだけ力量を持つメンバー"""
        single_mc = pd.DataFrame(
            {
                "member_code": ["m001"],
                "competence_code": ["s001"],
                "level": [3],
                "acquired_date": [datetime(2024, 1, 1)],
            }
        )
        fe = FeatureEngineer(
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_competence=single_mc,
        )
        patterns = fe.extract_temporal_patterns("m001")
        assert isinstance(patterns, dict)

    def test_japanese_column_names(self, sample_members, sample_competence_master):
        """日本語カラム名のサポート"""
        # sample_membersとsample_competence_masterは日本語カラム名を使用
        japanese_mc = pd.DataFrame(
            {
                "メンバーコード": ["m001", "m002"],
                "力量コード": ["s001", "s002"],
                "レベル": [3, 4],
            }
        )
        fe = FeatureEngineer(
            member_master=sample_members,
            competence_master=sample_competence_master,
            member_competence=japanese_mc,
        )
        assert fe is not None

    def test_mixed_column_names(
        self, extended_member_master, extended_competence_master
    ):
        """英語と日本語が混在したカラム名"""
        mixed_mc = pd.DataFrame(
            {
                "member_code": ["m001", "m002"],  # 英語
                "力量コード": ["s001", "s002"],  # 日本語
                "level": [3, 4],  # 英語
            }
        )
        fe = FeatureEngineer(
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_competence=mixed_mc,
        )
        # カラム名マッピングが正しく機能することを確認
        assert hasattr(fe, "member_code_col")
        assert hasattr(fe, "competence_code_col")


# ==================== 統合テスト ====================


class TestIntegration:
    """統合テスト"""

    def test_full_workflow(self, feature_engineer):
        """完全なワークフローのテスト"""
        # 1. メンバーのエンコーディング
        member_encoded = feature_engineer.encode_member_attributes("m001")
        assert len(member_encoded) > 0

        # 2. 力量のエンコーディング
        comp_encoded = feature_engineer.encode_competence_attributes("s001")
        assert len(comp_encoded) > 0

        # 3. 時系列パターン抽出
        patterns = feature_engineer.extract_temporal_patterns("m001")
        assert len(patterns) == 5

        # 4. 関連力量取得
        related = feature_engineer.get_related_competences("s001", top_k=3)
        assert len(related) <= 3

        # 5. 親和性計算
        affinity = feature_engineer.compute_member_competence_affinity("m001", "s004")
        assert 0 <= affinity <= 1

    def test_recommendation_workflow(self, feature_engineer):
        """推薦ワークフローのテスト"""
        member_code = "m001"
        candidate_competences = ["s004", "s005", "s006"]

        # バッチで親和性を計算
        affinities = feature_engineer.compute_batch_affinity(member_code, candidate_competences)

        # 親和性順にソート
        sorted_comps = sorted(affinities.items(), key=lambda x: x[1], reverse=True)

        assert len(sorted_comps) == 3
        assert all(0 <= affinity <= 1 for _, affinity in sorted_comps)
        # ソート順が正しいことを確認
        assert sorted_comps[0][1] >= sorted_comps[1][1] >= sorted_comps[2][1]
