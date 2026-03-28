"""
Tests for skillnote_recommendation.utils.visualization

可視化ユーティリティ関数のテスト。
matplotlib/plotlyのモックを使用して実際の描画は回避する。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import plotly.graph_objects as go

from skillnote_recommendation.utils.visualization import (
    create_member_positioning_data,
    create_positioning_plot,
    create_positioning_plot_with_patterns,
    prepare_positioning_display_dataframe,
    create_dependency_graph,
    _calculate_node_positions,
    _create_arrow,
    create_learning_path_timeline,
    create_skill_transition_graph,
    create_graph_statistics_chart,
)


# ==================== フィクスチャ ====================


@pytest.fixture
def mock_mf_model():
    """モックMatrixFactorizationModel"""
    model = Mock()
    model.n_components = 2
    model.member_index = {"m001": 0, "m002": 1, "m003": 2, "m004": 3, "m005": 4}
    # W行列（メンバー x 潜在因子）
    model.W = np.array([
        [0.5, 0.8],
        [0.3, 0.6],
        [0.7, 0.4],
        [0.2, 0.9],
        [0.6, 0.3],
    ])
    return model


@pytest.fixture
def position_df():
    """ポジショニングデータ"""
    return pd.DataFrame({
        "メンバーコード": ["m001", "m002", "m003", "m004", "m005"],
        "メンバー名": ["田中太郎", "鈴木花子", "佐藤次郎", "高橋美咲", "伊藤健一"],
        "総合スキルレベル": [10.5, 8.3, 12.0, 6.5, 15.2],
        "保有力量数": [5, 4, 6, 3, 7],
        "平均レベル": [2.1, 2.075, 2.0, 2.167, 2.171],
        "潜在因子1": [0.5, 0.3, 0.7, 0.2, 0.6],
        "潜在因子2": [0.8, 0.6, 0.4, 0.9, 0.3],
    })


@pytest.fixture
def graph_data():
    """グラフデータ"""
    return {
        "nodes": [
            {"id": "s001", "label": "Python", "type": "SKILL"},
            {"id": "s002", "label": "SQL", "type": "SKILL"},
            {"id": "s003", "label": "JavaScript", "type": "SKILL"},
            {"id": "e001", "label": "AWS研修", "type": "EDUCATION"},
        ],
        "edges": [
            {"source": "s001", "target": "s002", "strength": "強", "evidence": "5人", "time_gap_days": 30},
            {"source": "s002", "target": "s003", "strength": "中", "evidence": "3人", "time_gap_days": 45},
            {"source": "s001", "target": "e001", "strength": "弱", "evidence": "2人", "time_gap_days": 60},
        ],
    }


@pytest.fixture
def mock_learning_path():
    """モックLearningPath"""
    learning_path = Mock()
    learning_path.competence_name = "Python"
    learning_path.recommended_prerequisites = [
        {"skill_name": "基礎プログラミング", "average_time_gap_days": 90, "confidence": 0.8},
        {"skill_name": "アルゴリズム", "average_time_gap_days": 60, "confidence": 0.7},
        {"skill_name": "データ構造", "average_time_gap_days": 30, "confidence": 0.9},
    ]
    return learning_path


@pytest.fixture
def mock_graph_recommender():
    """モックSkillTransitionGraphRecommender"""
    import networkx as nx

    recommender = Mock()

    # グラフ作成
    G = nx.DiGraph()
    G.add_edge("s001", "s002", weight=5)
    G.add_edge("s002", "s003", weight=3)
    G.add_edge("s001", "s003", weight=2)
    recommender.graph = G

    # メソッドをモック
    recommender.get_user_skills = Mock(return_value=["s001"])
    recommender.get_skill_name = Mock(side_effect=lambda x: {
        "s001": "Python",
        "s002": "SQL",
        "s003": "JavaScript",
    }.get(x, "Unknown"))

    # 統計情報
    recommender.get_graph_statistics = Mock(return_value={
        "num_nodes": 3,
        "num_edges": 3,
        "avg_degree": 2.0,
        "top_target_skills": [
            ("Python", 5),
            ("SQL", 3),
            ("JavaScript", 2),
        ],
    })

    return recommender


# ==================== create_member_positioning_data テスト ====================


def test_create_member_positioning_data_success(sample_member_competence, sample_members, mock_mf_model):
    """正常系: メンバーポジショニングデータを正しく作成"""
    result = create_member_positioning_data(
        sample_member_competence, sample_members, mock_mf_model
    )

    # 基本検証
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "メンバーコード" in result.columns
    assert "メンバー名" in result.columns
    assert "総合スキルレベル" in result.columns
    assert "保有力量数" in result.columns
    assert "平均レベル" in result.columns
    assert "潜在因子1" in result.columns
    assert "潜在因子2" in result.columns


def test_create_member_positioning_data_no_competence(sample_members, mock_mf_model):
    """エッジケース: 力量データが空の場合"""
    empty_competence = pd.DataFrame(columns=["メンバーコード", "力量コード", "正規化レベル"])

    result = create_member_positioning_data(
        empty_competence, sample_members, mock_mf_model
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_create_member_positioning_data_member_not_in_model(sample_member_competence, sample_members):
    """エッジケース: メンバーがモデルに存在しない場合"""
    model = Mock()
    model.n_components = 2
    model.member_index = {}  # 空のインデックス
    model.W = np.array([])

    result = create_member_positioning_data(
        sample_member_competence, sample_members, model
    )

    # 潜在因子は0になるはず
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        assert all(result["潜在因子1"] == 0)
        assert all(result["潜在因子2"] == 0)


def test_create_member_positioning_data_single_component(sample_member_competence, sample_members):
    """エッジケース: 潜在因子が1つだけの場合"""
    model = Mock()
    model.n_components = 1
    model.member_index = {"m001": 0, "m002": 1, "m003": 2}
    model.W = np.array([[0.5], [0.3], [0.7]])

    result = create_member_positioning_data(
        sample_member_competence, sample_members, model
    )

    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        # 潜在因子2は0になるはず
        assert "潜在因子2" in result.columns


# ==================== create_positioning_plot テスト ====================


def test_create_positioning_plot_success(position_df):
    """正常系: ポジショニングプロットを正しく作成"""
    fig = create_positioning_plot(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002", "m003"],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="テストプロット",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.layout.title.text == "テストプロット"
    assert fig.layout.xaxis.title.text == "総合スキルレベル"
    assert fig.layout.yaxis.title.text == "保有力量数"


def test_create_positioning_plot_no_reference(position_df):
    """エッジケース: 参考人物が空の場合"""
    fig = create_positioning_plot(
        position_df,
        target_member_code="m001",
        reference_person_codes=[],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="テストプロット",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_positioning_plot_custom_height(position_df):
    """正常系: カスタム高さ設定"""
    custom_height = 800
    fig = create_positioning_plot(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002"],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="テストプロット",
        height=custom_height,
    )

    assert fig.layout.height == custom_height


def test_create_positioning_plot_empty_dataframe():
    """エッジケース: 空のDataFrame"""
    empty_df = pd.DataFrame(columns=[
        "メンバーコード", "メンバー名", "総合スキルレベル", "保有力量数"
    ])

    fig = create_positioning_plot(
        empty_df,
        target_member_code="m001",
        reference_person_codes=[],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="空データ",
    )

    assert isinstance(fig, go.Figure)


# ==================== create_positioning_plot_with_patterns テスト ====================


def test_create_positioning_plot_with_patterns_success(position_df):
    """正常系: パターン付きプロットを正しく作成"""
    fig = create_positioning_plot_with_patterns(
        position_df,
        target_member_code="m001",
        similar_career_codes=["m002"],
        different_career1_codes=["m003"],
        different_career2_codes=["m004"],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="キャリアパターン",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_positioning_plot_with_patterns_no_different_career(position_df):
    """エッジケース: 異なるキャリアが空の場合"""
    fig = create_positioning_plot_with_patterns(
        position_df,
        target_member_code="m001",
        similar_career_codes=["m002", "m003"],
        different_career1_codes=[],
        different_career2_codes=[],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="類似キャリアのみ",
    )

    assert isinstance(fig, go.Figure)


# ==================== prepare_positioning_display_dataframe テスト ====================


def test_prepare_positioning_display_dataframe_success(position_df):
    """正常系: 表示用DataFrameを正しく準備"""
    result = prepare_positioning_display_dataframe(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002", "m003"],
    )

    assert isinstance(result, pd.DataFrame)
    assert "タイプ" in result.columns
    assert "メンバー名" in result.columns
    assert result.iloc[0]["タイプ"] == "あなた"
    assert result.iloc[0]["メンバーコード"] == "m001"


def test_prepare_positioning_display_dataframe_sorting(position_df):
    """正常系: ソート順序を検証"""
    result = prepare_positioning_display_dataframe(
        position_df,
        target_member_code="m005",
        reference_person_codes=["m003", "m001"],
    )

    # 先頭はターゲット、次は参考人物、最後はその他
    types = result["タイプ"].unique()
    assert types[0] == "あなた"


def test_prepare_positioning_display_dataframe_column_order(position_df):
    """正常系: カラム順序を検証"""
    result = prepare_positioning_display_dataframe(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002"],
    )

    expected_cols = [
        "タイプ", "メンバー名", "メンバーコード", "総合スキルレベル",
        "保有力量数", "平均レベル", "潜在因子1", "潜在因子2"
    ]
    assert list(result.columns) == expected_cols


# ==================== create_dependency_graph テスト ====================


def test_create_dependency_graph_success(graph_data):
    """正常系: 依存関係グラフを正しく作成"""
    fig = create_dependency_graph(graph_data)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_dependency_graph_with_highlight(graph_data):
    """正常系: ハイライト付きグラフ"""
    fig = create_dependency_graph(
        graph_data,
        highlight_competence="s001",
    )

    assert isinstance(fig, go.Figure)


def test_create_dependency_graph_empty_nodes():
    """エッジケース: ノードが空の場合"""
    empty_data = {"nodes": [], "edges": []}

    fig = create_dependency_graph(empty_data)

    assert isinstance(fig, go.Figure)
    assert "依存関係データがありません" in fig.layout.title.text


def test_create_dependency_graph_custom_size(graph_data):
    """正常系: カスタムサイズ"""
    width = 1200
    height = 900
    fig = create_dependency_graph(
        graph_data,
        width=width,
        height=height,
    )

    assert fig.layout.width == width
    assert fig.layout.height == height


def test_create_dependency_graph_hierarchical_layout(graph_data):
    """正常系: 階層的レイアウト"""
    fig = create_dependency_graph(
        graph_data,
        layout_type="hierarchical",
    )

    assert isinstance(fig, go.Figure)


# ==================== _calculate_node_positions テスト ====================


def test_calculate_node_positions_small_graph():
    """正常系: 小さいグラフ（10ノード以下）"""
    nodes = [
        {"id": "n1", "label": "Node 1"},
        {"id": "n2", "label": "Node 2"},
        {"id": "n3", "label": "Node 3"},
    ]
    edges = [
        {"source": "n1", "target": "n2"},
        {"source": "n2", "target": "n3"},
    ]

    positions = _calculate_node_positions(nodes, edges)

    assert isinstance(positions, dict)
    assert len(positions) == 3
    assert all(isinstance(pos, tuple) for pos in positions.values())
    assert all(len(pos) == 2 for pos in positions.values())


def test_calculate_node_positions_large_graph():
    """正常系: 大きいグラフ（10ノード超）"""
    nodes = [{"id": f"n{i}", "label": f"Node {i}"} for i in range(15)]
    edges = [{"source": f"n{i}", "target": f"n{i+1}"} for i in range(14)]

    positions = _calculate_node_positions(nodes, edges)

    assert isinstance(positions, dict)
    assert len(positions) == 15


def test_calculate_node_positions_empty():
    """エッジケース: 空のノード"""
    positions = _calculate_node_positions([], [])

    assert isinstance(positions, dict)
    assert len(positions) == 0


# ==================== _create_arrow テスト ====================


def test_create_arrow_success():
    """正常系: 矢印を正しく作成"""
    arrow = _create_arrow(0, 0, 100, 100, "red")

    assert isinstance(arrow, go.Scatter)
    assert arrow.mode == "lines"


def test_create_arrow_zero_length():
    """エッジケース: 長さ0の矢印"""
    arrow = _create_arrow(50, 50, 50, 50, "blue")

    assert isinstance(arrow, go.Scatter)
    assert len(arrow.x) == 0


def test_create_arrow_vertical():
    """正常系: 垂直の矢印"""
    arrow = _create_arrow(50, 0, 50, 100, "green")

    assert isinstance(arrow, go.Scatter)
    assert len(arrow.x) == 3


def test_create_arrow_horizontal():
    """正常系: 水平の矢印"""
    arrow = _create_arrow(0, 50, 100, 50, "orange")

    assert isinstance(arrow, go.Scatter)
    assert len(arrow.x) == 3


# ==================== create_learning_path_timeline テスト ====================


def test_create_learning_path_timeline_success(mock_learning_path):
    """正常系: 学習パスタイムラインを正しく作成"""
    fig = create_learning_path_timeline(mock_learning_path)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_learning_path_timeline_no_prerequisites():
    """エッジケース: 前提スキルがない場合"""
    learning_path = Mock()
    learning_path.competence_name = "Python"
    learning_path.recommended_prerequisites = []

    fig = create_learning_path_timeline(learning_path)

    assert isinstance(fig, go.Figure)
    assert "前提スキルはありません" in fig.layout.title.text


def test_create_learning_path_timeline_custom_size(mock_learning_path):
    """正常系: カスタムサイズ"""
    width = 1200
    height = 600
    fig = create_learning_path_timeline(
        mock_learning_path,
        width=width,
        height=height,
    )

    assert fig.layout.width == width
    assert fig.layout.height == height


# ==================== create_skill_transition_graph テスト ====================


def test_create_skill_transition_graph_success(mock_graph_recommender):
    """正常系: スキル遷移グラフを正しく作成"""
    fig = create_skill_transition_graph(
        mock_graph_recommender,
        member_code="m001",
        recommended_skill="s003",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_skill_transition_graph_no_path(mock_graph_recommender):
    """エッジケース: パスが見つからない場合"""
    mock_graph_recommender.get_user_skills = Mock(return_value=["s999"])

    fig = create_skill_transition_graph(
        mock_graph_recommender,
        member_code="m001",
        recommended_skill="s003",
    )

    assert isinstance(fig, go.Figure)


def test_create_skill_transition_graph_custom_params(mock_graph_recommender):
    """正常系: カスタムパラメータ"""
    fig = create_skill_transition_graph(
        mock_graph_recommender,
        member_code="m001",
        recommended_skill="s003",
        width=800,
        height=300,
        max_paths=2,
        max_path_length=5,
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.width == 800
    assert fig.layout.height == 300


# ==================== create_graph_statistics_chart テスト ====================


def test_create_graph_statistics_chart_degree_distribution(mock_graph_recommender):
    """正常系: 次数分布チャート"""
    fig = create_graph_statistics_chart(
        mock_graph_recommender,
        chart_type="degree_distribution",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_create_graph_statistics_chart_top_skills(mock_graph_recommender):
    """正常系: トップスキルチャート"""
    fig = create_graph_statistics_chart(
        mock_graph_recommender,
        chart_type="top_skills",
    )

    assert isinstance(fig, go.Figure)


def test_create_graph_statistics_chart_default(mock_graph_recommender):
    """正常系: デフォルトチャート"""
    fig = create_graph_statistics_chart(
        mock_graph_recommender,
        chart_type="unknown",
    )

    assert isinstance(fig, go.Figure)


# ==================== 統合テスト ====================


def test_full_visualization_pipeline(sample_member_competence, sample_members, mock_mf_model):
    """統合テスト: 完全な可視化パイプライン"""
    # 1. ポジショニングデータ作成
    position_df = create_member_positioning_data(
        sample_member_competence, sample_members, mock_mf_model
    )
    assert len(position_df) > 0

    # 2. プロット作成
    fig1 = create_positioning_plot(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002"],
        x_col="総合スキルレベル",
        y_col="保有力量数",
        title="統合テスト",
    )
    assert isinstance(fig1, go.Figure)

    # 3. 表示用DataFrame準備
    display_df = prepare_positioning_display_dataframe(
        position_df,
        target_member_code="m001",
        reference_person_codes=["m002"],
    )
    assert len(display_df) > 0


def test_graph_visualization_pipeline(graph_data):
    """統合テスト: グラフ可視化パイプライン"""
    # 1. 依存関係グラフ作成
    fig = create_dependency_graph(graph_data)
    assert isinstance(fig, go.Figure)

    # 2. ノード位置計算
    positions = _calculate_node_positions(
        graph_data["nodes"],
        graph_data["edges"],
    )
    assert len(positions) > 0

    # 3. 矢印作成
    arrow = _create_arrow(0, 0, 100, 100, "red")
    assert isinstance(arrow, go.Scatter)
