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


# =========================================================
# Color Scheme Constants
# =========================================================

COLOR_TARGET_MEMBER = "#FF4B4B"  # Red for target member
COLOR_REFERENCE_PERSON = "#4B8BFF"  # Blue for reference persons
COLOR_OTHER_MEMBER = "#CCCCCC"  # Gray for others

MARKER_SIZE_TARGET = 20
MARKER_SIZE_REFERENCE = 15
MARKER_SIZE_OTHER = 8


# =========================================================
# Member Positioning Functions
# =========================================================

def create_member_positioning_data(
    member_competence: pd.DataFrame,
    member_master: pd.DataFrame,
    mf_model: MatrixFactorizationModel
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
        member_comp = member_competence[
            member_competence["メンバーコード"] == member_code
        ]

        if len(member_comp) == 0:
            continue

        # Get member name
        member_name = member_master[
            member_master["メンバーコード"] == member_code
        ]["メンバー名"].values[0]

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

        data.append({
            "メンバーコード": member_code,
            "メンバー名": member_name,
            "総合スキルレベル": total_level,
            "保有力量数": competence_count,
            "平均レベル": avg_level,
            "潜在因子1": latent_factor_1,
            "潜在因子2": latent_factor_2
        })

    return pd.DataFrame(data)


def create_positioning_plot(
    position_df: pd.DataFrame,
    target_member_code: str,
    reference_person_codes: List[str],
    x_col: str,
    y_col: str,
    title: str,
    height: int = 500
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
    df.loc[
        df["メンバーコード"].isin(reference_person_codes),
        "メンバータイプ"
    ] = "参考人物"

    # Map colors and sizes
    color_map = {
        "あなた": COLOR_TARGET_MEMBER,
        "参考人物": COLOR_REFERENCE_PERSON,
        "その他": COLOR_OTHER_MEMBER
    }

    size_map = {
        "あなた": MARKER_SIZE_TARGET,
        "参考人物": MARKER_SIZE_REFERENCE,
        "その他": MARKER_SIZE_OTHER
    }

    df["color"] = df["メンバータイプ"].map(color_map)
    df["size"] = df["メンバータイプ"].map(size_map)

    # Adjust plot order (others -> reference -> target)
    df["plot_order"] = df["メンバータイプ"].map({
        "その他": 1,
        "参考人物": 2,
        "あなた": 3
    })
    df = df.sort_values("plot_order")

    # Create scatter plot
    fig = go.Figure()

    for member_type in ["その他", "参考人物", "あなた"]:
        df_subset = df[df["メンバータイプ"] == member_type]

        if len(df_subset) == 0:
            continue

        fig.add_trace(go.Scatter(
            x=df_subset[x_col],
            y=df_subset[y_col],
            mode="markers",
            name=member_type,
            marker=dict(
                size=df_subset["size"],
                color=df_subset["color"],
                line=dict(width=1, color="white")
            ),
            text=df_subset["メンバー名"],
            customdata=df_subset["メンバーコード"],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "コード: %{customdata}<br>" +
                f"{x_col}: %{{x:.1f}}<br>" +
                f"{y_col}: %{{y:.2f}}<br>" +
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
        height=height,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def prepare_positioning_display_dataframe(
    position_df: pd.DataFrame,
    target_member_code: str,
    reference_person_codes: List[str]
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
    df["sort_order"] = df["タイプ"].map({
        "あなた": 0,
        "参考人物": 1,
        "その他": 2
    })
    df = df.sort_values(
        ["sort_order", "総合スキルレベル"],
        ascending=[True, False]
    )
    df = df.drop(columns=["sort_order"])

    # Reorder columns for better display
    cols = [
        "タイプ", "メンバー名", "メンバーコード", "総合スキルレベル",
        "保有力量数", "平均レベル", "潜在因子1", "潜在因子2"
    ]
    df = df[cols]

    return df
