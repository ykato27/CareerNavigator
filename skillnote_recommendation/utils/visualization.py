"""
Visualization utilities for the Career Navigator application.

This module provides functions for creating interactive visualizations
of member positioning, skill distributions, and model analysis results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional

from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.core.config import Config


# =========================================================
# Configuration-based Constants
# =========================================================

# Get colors from configuration
COLOR_TARGET_MEMBER = Config.VISUALIZATION_PARAMS["color_target_member"]
COLOR_REFERENCE_PERSON = Config.VISUALIZATION_PARAMS["color_reference_person"]
COLOR_OTHER_MEMBER = Config.VISUALIZATION_PARAMS["color_other_member"]

# Marker sizes
MARKER_SIZE_TARGET = 20
MARKER_SIZE_REFERENCE = 15
MARKER_SIZE_OTHER = 8


# =========================================================
# Member Positioning Functions
# =========================================================


def create_member_positioning_data(
    member_competence: pd.DataFrame, member_master: pd.DataFrame, mf_model: MatrixFactorizationModel
) -> pd.DataFrame:
    """
    Create positioning data for all members in the dataset.

    This function calculates various metrics for each member including:
    - Total skill level (sum of all normalized competence levels)
    - Number of competences
    - Average skill level
    - Latent factors from the NMF model

    Args:
        member_competence: DataFrame containing member-competence relationships
                          with columns: メンバーコード, 力量コード, 正規化レベル
        member_master: DataFrame containing member information
                      with columns: メンバーコード, メンバー名
        mf_model: Trained MatrixFactorizationModel instance

    Returns:
        DataFrame with columns:
        - メンバーコード: Member code
        - メンバー名: Member name
        - 総合スキルレベル: Total skill level
        - 保有力量数: Number of competences
        - 平均レベル: Average skill level
        - 潜在因子1: First latent factor (if available)
        - 潜在因子2: Second latent factor (if available)

    Example:
        >>> position_df = create_member_positioning_data(
        ...     member_comp_df, member_df, trained_model
        ... )
        >>> print(position_df.head())
    """
    data = []

    for member_code in member_master["メンバーコード"]:
        # Get member's competence data
        member_comp = member_competence[member_competence["メンバーコード"] == member_code]

        if len(member_comp) == 0:
            continue

        # Get member name
        member_name = member_master[member_master["メンバーコード"] == member_code][
            "メンバー名"
        ].values[0]

        # Calculate skill metrics
        total_level = member_comp["正規化レベル"].sum()
        competence_count = len(member_comp)
        avg_level = member_comp["正規化レベル"].mean()

        # Get latent factors from NMF model
        latent_factor_1 = 0
        latent_factor_2 = 0
        if member_code in mf_model.member_index:
            member_idx = mf_model.member_index[member_code]
            if mf_model.n_components > 0:
                latent_factor_1 = mf_model.W[member_idx, 0]
            if mf_model.n_components > 1:
                latent_factor_2 = mf_model.W[member_idx, 1]

        data.append(
            {
                "メンバーコード": member_code,
                "メンバー名": member_name,
                "総合スキルレベル": total_level,
                "保有力量数": competence_count,
                "平均レベル": avg_level,
                "潜在因子1": latent_factor_1,
                "潜在因子2": latent_factor_2,
            }
        )

    return pd.DataFrame(data)


def create_positioning_plot(
    position_df: pd.DataFrame,
    target_member_code: str,
    reference_person_codes: List[str],
    x_col: str,
    y_col: str,
    title: str,
    height: int = 500,
) -> go.Figure:
    """
    Create an interactive scatter plot showing member positioning.

    Members are color-coded as:
    - Target member (red, large marker)
    - Reference persons (blue, medium marker)
    - Other members (gray, small marker)

    Args:
        position_df: DataFrame with member positioning data
        target_member_code: Code of the target member to highlight
        reference_person_codes: List of reference person codes to highlight
        x_col: Column name to use for X axis
        y_col: Column name to use for Y axis
        title: Plot title
        height: Plot height in pixels (default: 500)

    Returns:
        Plotly Figure object with the scatter plot

    Example:
        >>> fig = create_positioning_plot(
        ...     position_df,
        ...     "M001",
        ...     ["M002", "M003"],
        ...     "総合スキルレベル",
        ...     "保有力量数",
        ...     "スキルレベル vs 保有力量数"
        ... )
        >>> st.plotly_chart(fig)
    """
    # Classify member types
    df = position_df.copy()
    df["メンバータイプ"] = "その他"
    df.loc[df["メンバーコード"] == target_member_code, "メンバータイプ"] = "あなた"
    df.loc[df["メンバーコード"].isin(reference_person_codes), "メンバータイプ"] = "参考人物"

    # Map colors and sizes
    color_map = {
        "あなた": COLOR_TARGET_MEMBER,
        "参考人物": COLOR_REFERENCE_PERSON,
        "その他": COLOR_OTHER_MEMBER,
    }

    size_map = {
        "あなた": MARKER_SIZE_TARGET,
        "参考人物": MARKER_SIZE_REFERENCE,
        "その他": MARKER_SIZE_OTHER,
    }

    df["color"] = df["メンバータイプ"].map(color_map)
    df["size"] = df["メンバータイプ"].map(size_map)

    # Adjust plot order (others -> reference -> target)
    df["plot_order"] = df["メンバータイプ"].map({"その他": 1, "参考人物": 2, "あなた": 3})
    df = df.sort_values("plot_order")

    # Create scatter plot
    fig = go.Figure()

    for member_type in ["その他", "参考人物", "あなた"]:
        df_subset = df[df["メンバータイプ"] == member_type]

        if len(df_subset) == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_subset[x_col],
                y=df_subset[y_col],
                mode="markers",
                name=member_type,
                marker=dict(
                    size=df_subset["size"],
                    color=df_subset["color"],
                    line=dict(width=1, color="white"),
                ),
                text=df_subset["メンバー名"],
                customdata=df_subset["メンバーコード"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "コード: %{customdata}<br>"
                    + f"{x_col}: %{{x:.1f}}<br>"
                    + f"{y_col}: %{{y:.2f}}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
        height=height,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_positioning_plot_with_patterns(
    position_df: pd.DataFrame,
    target_member_code: str,
    similar_career_codes: List[str],
    different_career1_codes: List[str],
    different_career2_codes: List[str],
    x_col: str,
    y_col: str,
    title: str,
    height: int = 500,
) -> go.Figure:
    """
    Create an interactive scatter plot with career pattern-based coloring.

    Members are color-coded as:
    - Target member (red, large marker)
    - Similar career (blue, medium marker)
    - Different career 1 (green, medium marker)
    - Different career 2 (orange, medium marker)
    - Other members (gray, small marker)

    Args:
        position_df: DataFrame with member positioning data
        target_member_code: Code of the target member to highlight
        similar_career_codes: List of similar career person codes
        different_career1_codes: List of different career 1 person codes
        different_career2_codes: List of different career 2 person codes
        x_col: Column name to use for X axis
        y_col: Column name to use for Y axis
        title: Plot title
        height: Plot height in pixels (default: 500)

    Returns:
        Plotly Figure object with the scatter plot
    """
    # Classify member types
    df = position_df.copy()
    df["メンバータイプ"] = "その他"
    df.loc[df["メンバーコード"] == target_member_code, "メンバータイプ"] = "あなた"
    df.loc[df["メンバーコード"].isin(similar_career_codes), "メンバータイプ"] = "💼 類似キャリア"
    df.loc[df["メンバーコード"].isin(different_career1_codes), "メンバータイプ"] = (
        "🌟 異なるキャリア1"
    )
    df.loc[df["メンバーコード"].isin(different_career2_codes), "メンバータイプ"] = (
        "🚀 異なるキャリア2"
    )

    # Map colors and sizes
    color_map = {
        "あなた": COLOR_TARGET_MEMBER,  # 赤
        "💼 類似キャリア": "#4B8BFF",  # 青
        "🌟 異なるキャリア1": "#4CAF50",  # 緑
        "🚀 異なるキャリア2": "#FF9800",  # オレンジ
        "その他": COLOR_OTHER_MEMBER,  # グレー
    }

    size_map = {
        "あなた": MARKER_SIZE_TARGET,
        "💼 類似キャリア": MARKER_SIZE_REFERENCE,
        "🌟 異なるキャリア1": MARKER_SIZE_REFERENCE,
        "🚀 異なるキャリア2": MARKER_SIZE_REFERENCE,
        "その他": MARKER_SIZE_OTHER,
    }

    df["color"] = df["メンバータイプ"].map(color_map)
    df["size"] = df["メンバータイプ"].map(size_map)

    # Adjust plot order (others -> different2 -> different1 -> similar -> target)
    df["plot_order"] = df["メンバータイプ"].map(
        {
            "その他": 1,
            "🚀 異なるキャリア2": 2,
            "🌟 異なるキャリア1": 3,
            "💼 類似キャリア": 4,
            "あなた": 5,
        }
    )
    df = df.sort_values("plot_order")

    # Create scatter plot
    fig = go.Figure()

    for member_type in [
        "その他",
        "🚀 異なるキャリア2",
        "🌟 異なるキャリア1",
        "💼 類似キャリア",
        "あなた",
    ]:
        df_subset = df[df["メンバータイプ"] == member_type]

        if len(df_subset) == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_subset[x_col],
                y=df_subset[y_col],
                mode="markers",
                name=member_type,
                marker=dict(
                    size=df_subset["size"],
                    color=df_subset["color"],
                    line=dict(width=1, color="white"),
                ),
                text=df_subset["メンバー名"],
                customdata=df_subset["メンバーコード"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "コード: %{customdata}<br>"
                    + f"{x_col}: %{{x:.1f}}<br>"
                    + f"{y_col}: %{{y:.2f}}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
        height=height,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def prepare_positioning_display_dataframe(
    position_df: pd.DataFrame, target_member_code: str, reference_person_codes: List[str]
) -> pd.DataFrame:
    """
    Prepare a DataFrame for display with member type classification.

    Args:
        position_df: DataFrame with member positioning data
        target_member_code: Code of the target member
        reference_person_codes: List of reference person codes

    Returns:
        DataFrame sorted by member type (target -> reference -> others)
        with an added "タイプ" column

    Example:
        >>> display_df = prepare_positioning_display_dataframe(
        ...     position_df, "M001", ["M002", "M003"]
        ... )
    """
    df = position_df.copy()

    # Add member type classification
    df["タイプ"] = "その他"
    df.loc[df["メンバーコード"] == target_member_code, "タイプ"] = "あなた"
    df.loc[df["メンバーコード"].isin(reference_person_codes), "タイプ"] = "参考人物"

    # Sort by type and skill level
    df["sort_order"] = df["タイプ"].map({"あなた": 0, "参考人物": 1, "その他": 2})
    df = df.sort_values(["sort_order", "総合スキルレベル"], ascending=[True, False])
    df = df.drop(columns=["sort_order"])

    # Reorder columns for better display
    cols = [
        "タイプ",
        "メンバー名",
        "メンバーコード",
        "総合スキルレベル",
        "保有力量数",
        "平均レベル",
        "潜在因子1",
        "潜在因子2",
    ]
    df = df[cols]

    return df


# =========================================================
# Skill Dependency Graph Visualization
# =========================================================


def create_dependency_graph(
    graph_data: dict,
    highlight_competence: Optional[str] = None,
    layout_type: str = "force",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    スキル依存関係グラフを作成

    Args:
        graph_data: ノードとエッジを含む辞書
        highlight_competence: ハイライトする力量コード
        layout_type: レイアウトタイプ ('force', 'hierarchical')
        width: グラフの幅
        height: グラフの高さ

    Returns:
        Plotly Figure
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        # 空のグラフを返す
        fig = go.Figure()
        fig.update_layout(title="依存関係データがありません", width=width, height=height)
        return fig

    # ノード位置の計算（簡易的な力学ベースレイアウト）
    node_positions = _calculate_node_positions(nodes, edges, layout_type)

    # エッジのトレースを作成
    edge_traces = []
    for edge in edges:
        source = edge["source"]
        target = edge["target"]

        if source not in node_positions or target not in node_positions:
            continue

        x0, y0 = node_positions[source]
        x1, y1 = node_positions[target]

        # 依存強度に応じて色を変える
        strength = edge.get("strength", "なし")
        if strength == "強":
            color = "rgba(255, 0, 0, 0.6)"
            width_val = 3
        elif strength == "中":
            color = "rgba(255, 165, 0, 0.5)"
            width_val = 2
        elif strength == "弱":
            color = "rgba(100, 100, 100, 0.4)"
            width_val = 1
        else:
            color = "rgba(200, 200, 200, 0.3)"
            width_val = 1

        # エッジ（矢印付き）
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width_val, color=color),
            hoverinfo="text",
            hovertext=f"{edge.get('evidence', '')}<br>平均学習間隔: {edge.get('time_gap_days', 0)}日",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

        # 矢印を追加
        arrow_trace = _create_arrow(x0, y0, x1, y1, color)
        edge_traces.append(arrow_trace)

    # ノードのトレースを作成
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in nodes:
        node_id = node["id"]
        if node_id not in node_positions:
            continue

        x, y = node_positions[node_id]
        node_x.append(x)
        node_y.append(y)

        label = node["label"]
        node_type = node.get("type", "")
        node_text.append(f"{label}<br>({node_type})")

        # ハイライト
        if highlight_competence and node_id == highlight_competence:
            node_colors.append("red")
            node_sizes.append(20)
        else:
            # タイプに応じて色分け
            if node_type == "SKILL":
                node_colors.append("lightblue")
            elif node_type == "EDUCATION":
                node_colors.append("lightgreen")
            elif node_type == "LICENSE":
                node_colors.append("lightyellow")
            else:
                node_colors.append("lightgray")
            node_sizes.append(15)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[node["label"] for node in nodes if node["id"] in node_positions],
        textposition="top center",
        textfont=dict(size=10),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        showlegend=False,
    )

    # グラフを作成
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title={
            "text": "スキル依存関係グラフ<br><sub>矢印の向きが学習順序を示します（赤=強い依存関係、橙=中程度、灰=弱い）</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=False,
        hovermode="closest",
        width=width,
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
    )

    return fig


def _calculate_node_positions(
    nodes: List[dict], edges: List[dict], layout_type: str = "force"
) -> dict:
    """
    ノード位置を計算（簡易的な力学ベースレイアウト）

    Args:
        nodes: ノードリスト
        edges: エッジリスト
        layout_type: レイアウトタイプ

    Returns:
        {node_id: (x, y)} の辞書
    """
    import random
    import math

    # ノード数が少ない場合は円形配置
    if len(nodes) <= 10:
        positions = {}
        radius = 100
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node["id"]] = (x, y)
        return positions

    # 初期位置をランダムに配置
    positions = {}
    for node in nodes:
        positions[node["id"]] = (random.uniform(-100, 100), random.uniform(-100, 100))

    # 簡易的な力学シミュレーション（Spring-Electric model）
    iterations = 50
    k = 50  # 理想的なばね長
    c = 0.1  # クーロン力定数

    for _ in range(iterations):
        forces = {node["id"]: [0, 0] for node in nodes}

        # ばね力（エッジで結ばれたノード間）
        for edge in edges:
            source = edge["source"]
            target = edge["target"]

            if source not in positions or target not in positions:
                continue

            x1, y1 = positions[source]
            x2, y2 = positions[target]

            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx**2 + dy**2) + 0.01

            # フックの法則
            force = (distance - k) / distance

            forces[source][0] += force * dx
            forces[source][1] += force * dy
            forces[target][0] -= force * dx
            forces[target][1] -= force * dy

        # 反発力（全ノード間）
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                id1 = node1["id"]
                id2 = node2["id"]

                if id1 not in positions or id2 not in positions:
                    continue

                x1, y1 = positions[id1]
                x2, y2 = positions[id2]

                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx**2 + dy**2) + 0.01

                # クーロンの法則
                force = c / (distance**2)

                forces[id1][0] -= force * dx / distance
                forces[id1][1] -= force * dy / distance
                forces[id2][0] += force * dx / distance
                forces[id2][1] += force * dy / distance

        # 位置を更新
        for node in nodes:
            node_id = node["id"]
            if node_id in positions:
                x, y = positions[node_id]
                fx, fy = forces[node_id]
                # ダンピング
                damping = 0.9
                positions[node_id] = (x + fx * damping, y + fy * damping)

    return positions


def _create_arrow(x0: float, y0: float, x1: float, y1: float, color: str) -> go.Scatter:
    """
    矢印アノテーションを作成

    Args:
        x0, y0: 始点
        x1, y1: 終点
        color: 色

    Returns:
        矢印のScatterトレース
    """
    import math

    # 矢印の先端を計算
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx**2 + dy**2)

    if length < 0.01:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False)

    # 矢印の位置（エッジの80%の位置）
    arrow_pos = 0.8
    ax = x0 + dx * arrow_pos
    ay = y0 + dy * arrow_pos

    # 矢印の向き
    angle = math.atan2(dy, dx)

    # 矢印の大きさ
    arrow_length = 5
    arrow_angle = math.pi / 6  # 30度

    # 矢印の両端
    x_left = ax - arrow_length * math.cos(angle + arrow_angle)
    y_left = ay - arrow_length * math.sin(angle + arrow_angle)
    x_right = ax - arrow_length * math.cos(angle - arrow_angle)
    y_right = ay - arrow_length * math.sin(angle - arrow_angle)

    arrow_trace = go.Scatter(
        x=[x_left, ax, x_right],
        y=[y_left, ay, y_right],
        mode="lines",
        line=dict(width=2, color=color),
        hoverinfo="skip",
        showlegend=False,
    )

    return arrow_trace


def create_learning_path_timeline(
    learning_path: "LearningPath", width: int = 1000, height: int = 400
) -> go.Figure:
    """
    学習パスのタイムライン可視化

    Args:
        learning_path: LearningPathオブジェクト
        width: グラフの幅
        height: グラフの高さ

    Returns:
        Plotly Figure
    """
    prerequisites = learning_path.recommended_prerequisites

    if not prerequisites:
        fig = go.Figure()
        fig.update_layout(
            title=f"{learning_path.competence_name}には前提スキルはありません",
            width=width,
            height=height,
        )
        return fig

    # 前提スキルを時間順に並べる
    sorted_prereqs = sorted(
        prerequisites, key=lambda x: x.get("average_time_gap_days", 0), reverse=True
    )

    # タイムラインデータを作成
    y_positions = list(range(len(sorted_prereqs)))
    skill_names = [p["skill_name"] for p in sorted_prereqs]
    time_gaps = [p.get("average_time_gap_days", 0) for p in sorted_prereqs]
    confidences = [p.get("confidence", 0) for p in sorted_prereqs]

    # バーチャート
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=skill_names,
            x=time_gaps,
            orientation="h",
            marker=dict(
                color=confidences,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="信頼度"),
            ),
            text=[f"{gap}日前" for gap in time_gaps],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>平均学習間隔: %{x}日<extra></extra>",
        )
    )

    # ターゲットスキルを追加
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[learning_path.competence_name],
            mode="markers+text",
            marker=dict(size=20, color="red", symbol="star"),
            text=["目標スキル"],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"<b>{learning_path.competence_name}</b><br>（習得目標）<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{learning_path.competence_name} への学習パス",
        xaxis_title="平均学習間隔（日）",
        yaxis_title="スキル",
        width=width,
        height=height,
        hovermode="closest",
    )

    return fig


# =========================================================
# Graph-based Recommendation Visualization
# =========================================================


def create_skill_transition_graph(
    graph_recommender,
    member_code: str,
    recommended_skill: str,
    width: int = 900,
    height: int = 400,
    max_paths: int = 1,
    max_path_length: int = 10,
    show_all_intermediate: bool = True,
) -> go.Figure:
    """
    スキル遷移グラフの可視化（シンプルなパス表示）

    Args:
        graph_recommender: SkillTransitionGraphRecommenderインスタンス
        member_code: メンバーコード
        recommended_skill: 推薦されたスキルコード
        width: 図の幅
        height: 図の高さ
        max_paths: 表示する最大パス数（デフォルト: 1 = 最短パスのみ）
        max_path_length: 表示する最大パス長（デフォルト: 10）
        show_all_intermediate: すべての中間ノードを表示するか（デフォルト: True）

    Returns:
        plotly.graph_objects.Figure
    """
    import networkx as nx

    graph = graph_recommender.graph
    user_skills = graph_recommender.get_user_skills(member_code)

    # 指定された最大長以下の最短パスを見つける
    best_path = None
    best_length = float("inf")

    for user_skill in user_skills:
        try:
            if user_skill in graph and recommended_skill in graph:
                path = nx.shortest_path(graph, user_skill, recommended_skill)
                # 最大パス長を超えない範囲で最短を選ぶ
                if len(path) <= max_path_length and len(path) < best_length:
                    best_path = path
                    best_length = len(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    # パスが見つからない、または最大長を超える場合の処理
    if not best_path:
        # 直接接続を探す
        for user_skill in user_skills:
            if graph.has_edge(user_skill, recommended_skill):
                best_path = [user_skill, recommended_skill]
                break

        # それでも見つからない場合は直接推薦として表示
        if not best_path:
            best_path = [user_skills[0] if user_skills else "Unknown", recommended_skill]

    # 階層的レイアウト（左から右へ）
    pos = {}
    for i, node in enumerate(best_path):
        pos[node] = (i * 2, 0)

    # エッジの描画（パスに沿った矢印）
    edge_traces = []
    annotations = []  # 矢印用

    for i in range(len(best_path) - 1):
        source = best_path[i]
        target = best_path[i + 1]
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # 遷移情報を取得
        weight = 1
        if graph.has_edge(source, target):
            weight = graph[source][target].get("weight", 1)

        # エッジライン
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=3, color="#4A90E2"),
            hoverinfo="text",
            hovertext=f"{graph_recommender.get_skill_name(source)} → "
            f"{graph_recommender.get_skill_name(target)}<br>"
            f"遷移人数: {weight}人",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

        # 矢印アノテーション
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="#4A90E2",
                standoff=20,
            )
        )

    # ノードの描画
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

    for i, node in enumerate(best_path):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        skill_name = graph_recommender.get_skill_name(node)

        # ステップ番号を追加
        if i == 0:
            node_text.append(f"【START】<br>{skill_name}")
        elif i == len(best_path) - 1:
            node_text.append(f"【GOAL】<br>{skill_name}")
        else:
            node_text.append(f"Step {i}<br>{skill_name}")

        # 色分け
        if node in user_skills:
            node_color.append("#4A90E2")  # 青: 習得済み
            node_size.append(40)
        elif node == recommended_skill:
            node_color.append("#E24A4A")  # 赤: 推薦スキル
            node_size.append(45)
        else:
            node_color.append("#95A5A6")  # グレー: 中間スキル
            node_size.append(35)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=12, family="Arial, sans-serif"),
        marker=dict(
            size=node_size, color=node_color, line=dict(width=3, color="white"), symbol="circle"
        ),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
    )

    # レイアウト
    fig = go.Figure(data=edge_traces + [node_trace])

    # パス情報をテキストで表示
    path_text = " → ".join([graph_recommender.get_skill_name(n) for n in best_path])

    fig.update_layout(
        title=dict(
            text=f"推奨学習パス（{len(best_path)}ステップ）<br><sub>{path_text}</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        showlegend=False,
        hovermode="closest",
        width=width,
        height=height,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, (len(best_path) - 1) * 2 + 0.5],
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
        plot_bgcolor="#F8F9FA",
        annotations=annotations,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    return fig


def create_graph_statistics_chart(
    graph_recommender, chart_type: str = "degree_distribution"
) -> go.Figure:
    """
    グラフ統計情報の可視化

    Args:
        graph_recommender: SkillTransitionGraphRecommenderインスタンス
        chart_type: チャートタイプ ('degree_distribution', 'top_skills')

    Returns:
        plotly.graph_objects.Figure
    """
    stats = graph_recommender.get_graph_statistics()
    graph = graph_recommender.graph

    if chart_type == "degree_distribution":
        # 次数分布
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=in_degrees, name="入次数（学ばれる回数）", opacity=0.7, marker_color="#4A90E2"
            )
        )

        fig.add_trace(
            go.Histogram(
                x=out_degrees, name="出次数（次に学ぶ回数）", opacity=0.7, marker_color="#E24A4A"
            )
        )

        fig.update_layout(
            title="スキル遷移の次数分布",
            xaxis_title="次数",
            yaxis_title="スキル数",
            barmode="overlay",
            hovermode="x",
        )

    elif chart_type == "top_skills":
        # トップスキルの表示
        fig = go.Figure()

        if "top_target_skills" in stats:
            skills = [s[0] for s in stats["top_target_skills"]]
            degrees = [s[1] for s in stats["top_target_skills"]]

            fig.add_trace(
                go.Bar(
                    y=skills,
                    x=degrees,
                    orientation="h",
                    name="最も学ばれるスキル",
                    marker_color="#4A90E2",
                )
            )

        fig.update_layout(
            title="最も学ばれるスキル Top 5",
            xaxis_title="遷移回数",
            yaxis_title="スキル",
            height=400,
        )

    else:
        # デフォルト: 基本統計
        fig = go.Figure()

    return fig
