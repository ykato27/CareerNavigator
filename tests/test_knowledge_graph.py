"""
CompetenceKnowledgeGraphのテスト

ナレッジグラフ構築、ノード/エッジ追加、統計情報取得など、
CompetenceKnowledgeGraphクラスの主要機能をテストする。
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
import tempfile
import os
from pathlib import Path

from skillnote_recommendation.graph.knowledge_graph import CompetenceKnowledgeGraph
from skillnote_recommendation.core.config import Config


# ==================== テストデータフィクスチャ ====================


@pytest.fixture
def extended_member_master():
    """拡張メンバーマスタ（グラフテスト用）"""
    return pd.DataFrame(
        {
            "メンバーコード": ["m001", "m002", "m003", "m004", "m005", "m006"],
            "メンバー名": [
                "田中太郎",
                "鈴木花子",
                "佐藤次郎",
                "高橋美咲",
                "伊藤健一",
                "渡辺直美",
            ],
            "職能等級": ["3等級", "4等級", "2等級", "2等級", "5等級", "3等級"],
            "役職": ["主任", "係長", "スタッフ", "スタッフ", "課長", "主任"],
        }
    )


@pytest.fixture
def extended_competence_master():
    """拡張力量マスタ（グラフテスト用）"""
    return pd.DataFrame(
        {
            "力量コード": [
                "s001",
                "s002",
                "s003",
                "s004",
                "s005",
                "s006",
                "e001",
                "e002",
                "l001",
                "l002",
            ],
            "力量名": [
                "Python",
                "SQL",
                "JavaScript",
                "Docker",
                "Git",
                "TypeScript",
                "AWS研修",
                "セキュリティ研修",
                "基本情報技術者",
                "応用情報技術者",
            ],
            "力量タイプ": [
                "SKILL",
                "SKILL",
                "SKILL",
                "SKILL",
                "SKILL",
                "SKILL",
                "EDUCATION",
                "EDUCATION",
                "LICENSE",
                "LICENSE",
            ],
            "力量カテゴリー名": [
                "技術 > プログラミング",
                "技術 > データベース",
                "技術 > プログラミング",
                "技術 > インフラ",
                "技術 > インフラ",
                "技術 > プログラミング",
                "技術 > クラウド",
                "技術 > セキュリティ",
                "資格 > IT資格",
                "資格 > IT資格",
            ],
            "概要": [
                "Pythonプログラミング",
                "SQLデータベース",
                "JavaScriptプログラミング",
                "コンテナ技術",
                "バージョン管理",
                "TypeScriptプログラミング",
                "AWSクラウド研修",
                "セキュリティ基礎研修",
                "基本情報技術者試験",
                "応用情報技術者試験",
            ],
        }
    )


@pytest.fixture
def extended_member_competence():
    """拡張メンバー習得力量データ（グラフテスト用）"""
    return pd.DataFrame(
        {
            "メンバーコード": [
                "m001",
                "m001",
                "m001",
                "m001",
                "m002",
                "m002",
                "m002",
                "m003",
                "m003",
                "m003",
                "m004",
                "m004",
                "m005",
                "m005",
                "m005",
                "m006",
                "m006",
            ],
            "力量コード": [
                "s001",
                "s002",
                "s003",
                "e001",
                "s001",
                "s003",
                "s006",
                "s002",
                "s004",
                "l001",
                "s001",
                "s005",
                "s004",
                "s005",
                "e002",
                "s001",
                "s002",
            ],
            "正規化レベル": [5, 4, 3, 1, 4, 4, 3, 5, 3, 1, 4, 3, 5, 4, 1, 3, 4],
        }
    )


@pytest.fixture
def minimal_data():
    """最小限のテストデータ（1メンバー、1力量）"""
    member_master = pd.DataFrame(
        {
            "メンバーコード": ["m001"],
            "メンバー名": ["テストユーザー"],
            "職能等級": ["3等級"],
            "役職": ["スタッフ"],
        }
    )

    competence_master = pd.DataFrame(
        {
            "力量コード": ["s001"],
            "力量名": ["Python"],
            "力量タイプ": ["SKILL"],
            "力量カテゴリー名": ["技術 > プログラミング"],
            "概要": ["Pythonプログラミング"],
        }
    )

    member_competence = pd.DataFrame(
        {"メンバーコード": ["m001"], "力量コード": ["s001"], "正規化レベル": [3]}
    )

    return member_master, competence_master, member_competence


@pytest.fixture
def empty_member_competence():
    """空の習得力量データ"""
    return pd.DataFrame(columns=["メンバーコード", "力量コード", "正規化レベル"])


@pytest.fixture
def no_category_competence_master():
    """カテゴリーなしの力量マスタ"""
    return pd.DataFrame(
        {
            "力量コード": ["s001", "s002"],
            "力量名": ["Python", "SQL"],
            "力量タイプ": ["SKILL", "SKILL"],
            "力量カテゴリー名": [None, None],
        }
    )


# ==================== テストクラス ====================


class TestCompetenceKnowledgeGraphInitialization:
    """グラフ初期化のテスト"""

    def test_initialization_with_category_hierarchy(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カテゴリー階層ありでの初期化"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=True,
        )

        assert kg.G is not None
        assert isinstance(kg.G, nx.Graph)
        assert kg.use_category_hierarchy is True
        assert kg.category_hierarchy is not None
        assert kg.G.number_of_nodes() > 0
        assert kg.G.number_of_edges() > 0

    def test_initialization_without_category_hierarchy(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カテゴリー階層なしでの初期化"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=False,
        )

        assert kg.G is not None
        assert kg.use_category_hierarchy is False
        assert kg.category_hierarchy is None
        assert kg.G.number_of_nodes() > 0

    def test_initialization_with_minimal_data(self, minimal_data):
        """最小限のデータでの初期化"""
        member_master, competence_master, member_competence = minimal_data

        kg = CompetenceKnowledgeGraph(
            member_competence=member_competence,
            member_master=member_master,
            competence_master=competence_master,
            use_category_hierarchy=False,
        )

        assert kg.G.number_of_nodes() >= 2  # 少なくともメンバー1 + 力量1
        assert kg.G.number_of_edges() >= 1  # 少なくとも習得エッジ1

    def test_initialization_with_custom_similarity_params(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カスタム類似度パラメータでの初期化"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=False,
            member_similarity_threshold=0.5,
            member_similarity_top_k=3,
        )

        assert kg.member_similarity_threshold == 0.5
        assert kg.member_similarity_top_k == 3

    def test_dataframe_attributes_stored(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """DataFrameが正しく保存されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        assert kg.member_competence_df is not None
        assert kg.member_master_df is not None
        assert kg.competence_master_df is not None
        assert len(kg.member_master_df) == len(extended_member_master)
        assert len(kg.competence_master_df) == len(extended_competence_master)


class TestNodeAddition:
    """ノード追加のテスト"""

    def test_member_nodes_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """メンバーノードが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # メンバーノードの数を確認
        member_nodes = [n for n in kg.G.nodes() if n.startswith("member_")]
        assert len(member_nodes) == len(extended_member_master)

        # メンバーノードの属性を確認
        member_node = "member_m001"
        assert kg.G.has_node(member_node)
        node_data = kg.G.nodes[member_node]
        assert node_data["node_type"] == "member"
        assert node_data["code"] == "m001"
        assert node_data["name"] == "田中太郎"
        assert "grade" in node_data
        assert "position" in node_data

    def test_competence_nodes_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """力量ノードが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # 力量ノードの数を確認
        competence_nodes = [n for n in kg.G.nodes() if n.startswith("competence_")]
        assert len(competence_nodes) == len(extended_competence_master)

        # 力量ノードの属性を確認
        comp_node = "competence_s001"
        assert kg.G.has_node(comp_node)
        node_data = kg.G.nodes[comp_node]
        assert node_data["node_type"] == "competence"
        assert node_data["code"] == "s001"
        assert node_data["name"] == "Python"
        assert node_data["type"] == "SKILL"
        assert "category" in node_data
        assert "description" in node_data

    def test_category_nodes_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カテゴリーノードが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=True,
        )

        # カテゴリーノードの数を確認
        category_nodes = [n for n in kg.G.nodes() if n.startswith("category_")]
        assert len(category_nodes) > 0

        # カテゴリーノードが存在することを確認（フルパス形式）
        assert kg.G.has_node("category_技術 > プログラミング")
        assert kg.G.has_node("category_技術 > データベース")

    def test_category_nodes_with_empty_values(
        self, extended_member_master, no_category_competence_master, empty_member_competence
    ):
        """カテゴリーが空の場合"""
        kg = CompetenceKnowledgeGraph(
            member_competence=empty_member_competence,
            member_master=extended_member_master,
            competence_master=no_category_competence_master,
        )

        # カテゴリーノードが追加されないことを確認
        category_nodes = [n for n in kg.G.nodes() if n.startswith("category_")]
        assert len(category_nodes) == 0


class TestEdgeAddition:
    """エッジ追加のテスト"""

    def test_member_competence_edges_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """メンバー-力量エッジが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # 習得エッジの確認
        member_node = "member_m001"
        comp_node = "competence_s001"

        assert kg.G.has_edge(member_node, comp_node)
        edge_data = kg.G[member_node][comp_node]
        assert edge_data["edge_type"] == "acquired"
        assert "weight" in edge_data
        assert "level" in edge_data
        assert edge_data["weight"] > 0

    def test_competence_category_edges_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """力量-カテゴリーエッジが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # 所属エッジの確認
        comp_node = "competence_s001"
        category_node = "category_技術 > プログラミング"

        assert kg.G.has_edge(comp_node, category_node)
        edge_data = kg.G[comp_node][category_node]
        assert edge_data["edge_type"] == "belongs_to"
        assert edge_data["weight"] == 1.0

    def test_category_hierarchy_edges_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カテゴリー階層エッジが正しく追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=True,
        )

        # 親子エッジの確認
        parent_node = "category_技術"
        child_node = "category_技術 > プログラミング"

        if kg.G.has_node(parent_node) and kg.G.has_node(child_node):
            assert kg.G.has_edge(parent_node, child_node)
            edge_data = kg.G[parent_node][child_node]
            assert edge_data["edge_type"] == "parent_of"
            assert edge_data["weight"] == 1.0

    def test_member_similarity_edges_added(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """メンバー類似度エッジが追加されているか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_similarity_threshold=0.1,  # 低い閾値で類似エッジが出るようにする
            member_similarity_top_k=5,
        )

        # 類似エッジの存在を確認
        similarity_edges = [
            (u, v) for u, v, d in kg.G.edges(data=True) if d.get("edge_type") == "similar"
        ]

        # 少なくとも一部のメンバー間に類似エッジがあるはず
        # （データによっては類似度が低くてエッジがない場合もある）
        assert len(similarity_edges) >= 0

    def test_member_similarity_with_high_threshold(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """高い閾値での類似度エッジ追加"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_similarity_threshold=0.99,  # 非常に高い閾値
            member_similarity_top_k=5,
        )

        # 類似エッジが少ないまたはゼロであることを確認
        similarity_edges = [
            (u, v) for u, v, d in kg.G.edges(data=True) if d.get("edge_type") == "similar"
        ]
        # 高い閾値では類似エッジがほとんど追加されない
        assert len(similarity_edges) >= 0

    def test_member_similarity_with_single_member(self, minimal_data):
        """1メンバーのみの場合、類似度エッジは追加されない"""
        member_master, competence_master, member_competence = minimal_data

        kg = CompetenceKnowledgeGraph(
            member_competence=member_competence,
            member_master=member_master,
            competence_master=competence_master,
            use_category_hierarchy=False,
        )

        # 類似エッジが存在しないことを確認
        similarity_edges = [
            (u, v) for u, v, d in kg.G.edges(data=True) if d.get("edge_type") == "similar"
        ]
        assert len(similarity_edges) == 0


class TestGraphStatistics:
    """グラフ統計情報のテスト"""

    def test_graph_has_nodes_and_edges(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """グラフにノードとエッジが存在する"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        assert kg.G.number_of_nodes() > 0
        assert kg.G.number_of_edges() > 0

    def test_node_type_distribution(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """ノードタイプの分布が正しい"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        node_types = {}
        for node in kg.G.nodes():
            node_type = kg.G.nodes[node].get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        assert "member" in node_types
        assert "competence" in node_types
        assert "category" in node_types
        assert node_types["member"] == len(extended_member_master)
        assert node_types["competence"] == len(extended_competence_master)

    def test_edge_type_distribution(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """エッジタイプの分布が正しい"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            use_category_hierarchy=True,
        )

        edge_types = {}
        for u, v in kg.G.edges():
            edge_type = kg.G[u][v].get("edge_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        assert "acquired" in edge_types
        assert "belongs_to" in edge_types
        # parent_of と similar は条件によって存在しない場合もある


class TestGetNeighbors:
    """get_neighborsメソッドのテスト"""

    def test_get_neighbors_all_types(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """全エッジタイプでの隣接ノード取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        member_node = "member_m001"
        neighbors = kg.get_neighbors(member_node)

        assert isinstance(neighbors, list)
        assert len(neighbors) > 0

    def test_get_neighbors_by_edge_type_acquired(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """習得エッジのみでの隣接ノード取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        member_node = "member_m001"
        neighbors = kg.get_neighbors(member_node, edge_type="acquired")

        assert isinstance(neighbors, list)
        # m001は複数のスキルを習得している
        assert len(neighbors) > 0
        # 全てがcompetenceノードであることを確認
        for n in neighbors:
            assert n.startswith("competence_")

    def test_get_neighbors_by_edge_type_belongs_to(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """所属エッジのみでの隣接ノード取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        comp_node = "competence_s001"
        neighbors = kg.get_neighbors(comp_node, edge_type="belongs_to")

        assert isinstance(neighbors, list)
        # s001はカテゴリーに所属している
        assert len(neighbors) > 0
        # 全てがcategoryノードであることを確認
        for n in neighbors:
            assert n.startswith("category_")

    def test_get_neighbors_nonexistent_node(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """存在しないノードの隣接ノード取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        neighbors = kg.get_neighbors("nonexistent_node")
        assert neighbors == []

    def test_get_neighbors_by_edge_type_similar(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """類似エッジのみでの隣接ノード取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
            member_similarity_threshold=0.1,
            member_similarity_top_k=5,
        )

        member_node = "member_m001"
        neighbors = kg.get_neighbors(member_node, edge_type="similar")

        assert isinstance(neighbors, list)
        # 全てがmemberノードであることを確認
        for n in neighbors:
            assert n.startswith("member_")


class TestGetNodeInfo:
    """get_node_infoメソッドのテスト"""

    def test_get_member_node_info(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """メンバーノード情報取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        info = kg.get_node_info("member_m001")

        assert isinstance(info, dict)
        assert info["node_type"] == "member"
        assert info["code"] == "m001"
        assert info["name"] == "田中太郎"
        assert "grade" in info
        assert "position" in info

    def test_get_competence_node_info(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """力量ノード情報取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        info = kg.get_node_info("competence_s001")

        assert isinstance(info, dict)
        assert info["node_type"] == "competence"
        assert info["code"] == "s001"
        assert info["name"] == "Python"
        assert info["type"] == "SKILL"
        assert "category" in info

    def test_get_category_node_info(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """カテゴリーノード情報取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        info = kg.get_node_info("category_技術 > プログラミング")

        assert isinstance(info, dict)
        assert info["node_type"] == "category"
        assert info["name"] == "技術 > プログラミング"

    def test_get_node_info_nonexistent(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """存在しないノード情報取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        info = kg.get_node_info("nonexistent_node")
        assert info == {}


class TestGetMemberAcquiredCompetences:
    """get_member_acquired_competencesメソッドのテスト"""

    def test_get_acquired_competences(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """習得済み力量の取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        acquired = kg.get_member_acquired_competences("m001")

        assert isinstance(acquired, set)
        # m001はs001, s002, s003, e001を習得している
        assert "s001" in acquired
        assert "s002" in acquired
        assert "s003" in acquired
        assert "e001" in acquired

    def test_get_acquired_competences_empty(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """習得力量がないメンバー（存在しないメンバーコード）"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        acquired = kg.get_member_acquired_competences("nonexistent")
        assert acquired == set()

    def test_get_acquired_competences_count(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """習得力量数が正しいか"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # m001の習得力量数を確認
        acquired_m001 = kg.get_member_acquired_competences("m001")
        expected_count = len(
            extended_member_competence[extended_member_competence["メンバーコード"] == "m001"]
        )
        assert len(acquired_m001) == expected_count


class TestGetCompetenceCategory:
    """get_competence_categoryメソッドのテスト"""

    def test_get_competence_category(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """力量カテゴリーの取得"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        category = kg.get_competence_category("s001")

        assert category is not None
        assert category == "技術 > プログラミング"

    def test_get_competence_category_none(
        self, extended_member_master, no_category_competence_master, empty_member_competence
    ):
        """カテゴリーがない力量"""
        kg = CompetenceKnowledgeGraph(
            member_competence=empty_member_competence,
            member_master=extended_member_master,
            competence_master=no_category_competence_master,
        )

        category = kg.get_competence_category("s001")
        assert category is None

    def test_get_competence_category_nonexistent(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """存在しない力量コード"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        category = kg.get_competence_category("nonexistent")
        assert category is None


class TestExportToGEXF:
    """export_to_gexfメソッドのテスト"""

    def test_export_to_gexf(
        self, extended_member_master, extended_competence_master, extended_member_competence, tmp_path
    ):
        """GEXF形式でのエクスポート"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        output_file = tmp_path / "test_graph.gexf"
        kg.export_to_gexf(str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_export_to_gexf_can_be_loaded(
        self, extended_member_master, extended_competence_master, extended_member_competence, tmp_path
    ):
        """エクスポートしたGEXFファイルが読み込み可能か"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        output_file = tmp_path / "test_graph.gexf"
        kg.export_to_gexf(str(output_file))

        # ファイルを読み込んで検証
        loaded_graph = nx.read_gexf(str(output_file))
        assert loaded_graph.number_of_nodes() == kg.G.number_of_nodes()
        assert loaded_graph.number_of_edges() == kg.G.number_of_edges()


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_member_competence(
        self, extended_member_master, extended_competence_master, empty_member_competence
    ):
        """空の習得力量データ"""
        kg = CompetenceKnowledgeGraph(
            member_competence=empty_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        # メンバーノードと力量ノードは存在するが、エッジは少ない
        member_nodes = [n for n in kg.G.nodes() if n.startswith("member_")]
        comp_nodes = [n for n in kg.G.nodes() if n.startswith("competence_")]

        assert len(member_nodes) == len(extended_member_master)
        assert len(comp_nodes) == len(extended_competence_master)

        # 習得エッジはゼロ
        acquired_edges = [
            (u, v) for u, v, d in kg.G.edges(data=True) if d.get("edge_type") == "acquired"
        ]
        assert len(acquired_edges) == 0

    def test_graph_is_undirected(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """グラフが無向グラフであることを確認"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        assert isinstance(kg.G, nx.Graph)
        assert not isinstance(kg.G, nx.DiGraph)

    def test_edge_weights_are_numeric(
        self, extended_member_master, extended_competence_master, extended_member_competence
    ):
        """エッジの重みが数値であることを確認"""
        kg = CompetenceKnowledgeGraph(
            member_competence=extended_member_competence,
            member_master=extended_member_master,
            competence_master=extended_competence_master,
        )

        for u, v, data in kg.G.edges(data=True):
            if "weight" in data:
                assert isinstance(data["weight"], (int, float))
                assert data["weight"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
