"""
高度な組織分析ダッシュボードコンポーネント

データサイエンティスト視点での戦略的人材分析機能を提供

Updated: 2025-11-22
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple
from scipy import stats


def extract_category_hierarchy(category_name: str, level: int = 1) -> str:
    """
    カテゴリ名から指定階層までを抽出（フルパス保持）

    Args:
        category_name: カテゴリ名（例: "技術 > プログラミング > Python"）
        level: 抽出する階層レベル（1=第一階層、2=第二階層、3=第三階層）

    Returns:
        指定階層までのカテゴリ名（例: "技術 > プログラミング"）
    """
    if pd.isna(category_name):
        return "未分類"

    parts = str(category_name).split(">")

    if level >= len(parts):
        return category_name

    return " > ".join([p.strip() for p in parts[:level]])


def render_org_skill_summary(
    gap_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    percentile_used: float = 0.0
) -> None:
    """
    組織スキル概要サマリーを表示

    Args:
        gap_df: スキルギャップデータ
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        percentile_used: 使用したパーセンタイル（0.0=全体平均、0.5=中央値、0.75=75%タイル）
    """
    st.markdown("### 📊 組織スキル状況サマリー")

    col1, col2, col3, col4 = st.columns(4)

    # 総スキル数
    total_skills = len(competence_master_df)

    # 組織平均保有率
    avg_coverage = gap_df["現在保有率"].mean() if "現在保有率" in gap_df.columns else 0

    # クリティカルギャップ（目標保有率との差が30%以上）
    if "保有率ギャップ率" in gap_df.columns:
        critical_gaps = len(gap_df[gap_df["保有率ギャップ率"] > 0.3])
    else:
        critical_gaps = 0

    # 総メンバー数
    total_members = len(members_df)

    with col1:
        st.metric("総スキル数", f"{total_skills}件")

    with col2:
        st.metric("組織平均保有率", f"{avg_coverage*100:.1f}%")

    with col3:
        st.metric("クリティカルギャップ", f"{critical_gaps}件",
                 delta=f"{(critical_gaps/total_skills*100):.1f}%",
                 delta_color="inverse")

    with col4:
        st.metric("総メンバー数", f"{total_members}名")

    st.markdown("---")


def render_category_skill_distribution(
    gap_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    hierarchy_level: int = 1
) -> None:
    """
    カテゴリ別スキル分布を表示

    Args:
        gap_df: スキルギャップデータ
        competence_master_df: 力量マスタ
        hierarchy_level: 階層レベル（1=第一階層、2=第二階層）
    """
    st.markdown("### 📈 カテゴリ別スキル保有状況")

    # カテゴリ情報をマージ
    if "力量コード" in gap_df.columns and "力量コード" in competence_master_df.columns:
        merged = gap_df.merge(
            competence_master_df[["力量コード", "力量タイプ"]],
            on="力量コード",
            how="left"
        )
    else:
        st.warning("⚠️ カテゴリ情報が不足しているため、この分析をスキップします")
        return

    # 階層抽出
    merged["カテゴリ"] = merged["力量タイプ"].apply(
        lambda x: extract_category_hierarchy(x, hierarchy_level)
    )

    # カテゴリごとの集計
    category_stats = merged.groupby("カテゴリ").agg({
        "現在保有率": "mean",
        "目標保有率": "mean",
        "保有率ギャップ率": "mean",
        "力量コード": "count"
    }).reset_index()

    category_stats.columns = ["カテゴリ", "平均現在保有率", "平均目標保有率", "平均ギャップ率", "スキル数"]

    # ソート
    category_stats = category_stats.sort_values("平均ギャップ率", ascending=False)

    # 可視化
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="現在保有率",
        x=category_stats["カテゴリ"],
        y=category_stats["平均現在保有率"] * 100,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name="目標保有率",
        x=category_stats["カテゴリ"],
        y=category_stats["平均目標保有率"] * 100,
        marker_color='orange'
    ))

    fig.update_layout(
        title="カテゴリ別スキル保有率（現在 vs 目標）",
        xaxis_title="カテゴリ",
        yaxis_title="保有率 (%)",
        barmode='group',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # データテーブル
    st.dataframe(
        category_stats.style.format({
            "平均現在保有率": "{:.1%}",
            "平均目標保有率": "{:.1%}",
            "平均ギャップ率": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )


def render_risk_priority_matrix(
    gap_df: pd.DataFrame,
    top_n: int = 20
) -> None:
    """
    リスク優先度マトリックスを表示

    Args:
        gap_df: スキルギャップデータ
        top_n: 表示する上位スキル数
    """
    st.markdown("### 🎯 リスク優先度マトリックス")

    st.markdown("""
    **保有率ギャップ** と **目標保有率** の2軸でリスクを評価します。
    - 右上: 高い目標 × 大きなギャップ = **最優先対応**
    - 左上: 高い目標 × 小さなギャップ = 維持・強化
    - 右下: 低い目標 × 大きなギャップ = 中優先度
    """)

    # 上位スキルを選択
    if "保有率ギャップ率" in gap_df.columns:
        top_gaps = gap_df.nlargest(top_n, "保有率ギャップ率")
    else:
        st.warning("⚠️ 保有率ギャップ率カラムが見つかりません")
        return

    # バブルチャート
    fig = px.scatter(
        top_gaps,
        x="保有率ギャップ率",
        y="目標保有率",
        size="現在保有率",
        color="保有率ギャップ率",
        hover_name="力量名",
        hover_data={
            "現在保有率": ":.1%",
            "目標保有率": ":.1%",
            "保有率ギャップ率": ":.1%"
        },
        title=f"リスク優先度マトリックス（上位{top_n}スキル）",
        labels={
            "保有率ギャップ率": "保有率ギャップ (%)",
            "目標保有率": "目標保有率 (%)"
        },
        color_continuous_scale="Reds"
    )

    # 四分位線を追加
    median_gap = top_gaps["保有率ギャップ率"].median()
    median_target = top_gaps["目標保有率"].median()

    fig.add_hline(y=median_target, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_gap, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_skill_coverage_by_role(
    member_competence_df: pd.DataFrame,
    members_df: pd.DataFrame,
    role_column: str = "役職"
) -> None:
    """
    役職・職種別のスキルカバレッジを表示

    Args:
        member_competence_df: メンバー習得力量データ
        members_df: メンバーマスタ
        role_column: 役職または職種のカラム名
    """
    st.markdown(f"### 👥 {role_column}別スキルカバレッジ分析")

    # メンバー情報とマージ
    if "メンバーコード" in member_competence_df.columns and "メンバーコード" in members_df.columns:
        merged = member_competence_df.merge(
            members_df[["メンバーコード", role_column]],
            on="メンバーコード",
            how="left"
        )
    else:
        st.warning(f"⚠️ {role_column}別分析をスキップします（メンバーコードが不足）")
        return

    # 役職が欠損している行を除外
    merged = merged[merged[role_column].notna()]

    if len(merged) == 0:
        st.warning(f"⚠️ {role_column}情報が見つかりません")
        return

    # メンバーごとのスキル保有数を計算
    member_skill_counts = merged.groupby(["メンバーコード", role_column]).size().reset_index(name="スキル保有数")

    # 役職ごとの統計
    role_stats = member_skill_counts.groupby(role_column).agg({
        "スキル保有数": ["mean", "median", "min", "max", "std"]
    }).reset_index()

    role_stats.columns = [role_column, "平均", "中央値", "最小", "最大", "標準偏差"]
    role_stats = role_stats.sort_values("平均", ascending=False)

    # ボックスプロット
    fig = px.box(
        member_skill_counts,
        x=role_column,
        y="スキル保有数",
        title=f"{role_column}別スキル保有数分布",
        labels={role_column: role_column, "スキル保有数": "スキル保有数"},
        points="all"
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 統計テーブル
    st.markdown("#### 📊 統計サマリー")
    st.dataframe(
        role_stats.style.format({
            "平均": "{:.1f}",
            "中央値": "{:.1f}",
            "最小": "{:.0f}",
            "最大": "{:.0f}",
            "標準偏差": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


def render_succession_planning_advanced(
    member_competence_df: pd.DataFrame,
    members_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    target_role: str,
    role_column: str = "役職"
) -> None:
    """
    高度な後継者計画分析を表示

    Args:
        member_competence_df: メンバー習得力量データ
        members_df: メンバーマスタ
        competence_master_df: 力量マスタ
        target_role: 対象役職
        role_column: 役職カラム名
    """
    st.markdown(f"### 👔 後継者計画: {target_role}")

    # 現在の役職保持者のスキルセット分析
    current_holders = members_df[members_df[role_column] == target_role]

    if len(current_holders) == 0:
        st.warning(f"⚠️ {target_role}の現在の保持者が見つかりません")
        return

    st.markdown(f"**現在の{target_role}保持者**: {len(current_holders)}名")

    # 現在の保持者のスキルセット
    holder_codes = current_holders["メンバーコード"].tolist()
    holder_skills = member_competence_df[
        member_competence_df["メンバーコード"].isin(holder_codes)
    ]

    # 役職に必要なスキルセット（現在保持者が80%以上持っているスキル）
    required_skills = holder_skills.groupby("力量コード").size()
    threshold = len(holder_codes) * 0.8
    required_skill_codes = required_skills[required_skills >= threshold].index.tolist()

    st.markdown(f"**必須スキル数**: {len(required_skill_codes)}件（現保持者の80%以上が保有）")

    # 後継者候補の評価
    all_members = members_df[members_df[role_column] != target_role]["メンバーコード"].tolist()

    candidate_scores = []
    for member_code in all_members:
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ]["力量コード"].tolist()

        # 必須スキルのうち保有している割合
        match_count = len(set(member_skills) & set(required_skill_codes))
        coverage = match_count / len(required_skill_codes) if len(required_skill_codes) > 0 else 0

        member_info = members_df[members_df["メンバーコード"] == member_code].iloc[0]

        candidate_scores.append({
            "メンバーコード": member_code,
            "メンバー名": member_info.get("メンバー名", ""),
            "現在の役職": member_info.get(role_column, ""),
            "必須スキルカバー率": coverage,
            "保有スキル数": len(member_skills),
            "不足スキル数": len(required_skill_codes) - match_count
        })

    candidate_df = pd.DataFrame(candidate_scores)
    candidate_df = candidate_df.sort_values("必須スキルカバー率", ascending=False).head(10)

    # 準備度による分類
    def classify_readiness(coverage):
        if coverage >= 0.9:
            return "🟢 即戦力（Ready Now）"
        elif coverage >= 0.7:
            return "🟡 短期育成（1-2年）"
        elif coverage >= 0.5:
            return "🟠 中期育成（2-3年）"
        else:
            return "🔴 長期育成（3年以上）"

    candidate_df["準備度"] = candidate_df["必須スキルカバー率"].apply(classify_readiness)

    # 可視化
    fig = px.bar(
        candidate_df.head(10),
        x="メンバー名",
        y="必須スキルカバー率",
        color="準備度",
        title=f"{target_role}後継者候補トップ10",
        labels={"必須スキルカバー率": "必須スキルカバー率 (%)"},
        hover_data=["現在の役職", "保有スキル数", "不足スキル数"]
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 候補者リスト
    st.markdown("#### 📋 候補者詳細")
    st.dataframe(
        candidate_df.style.format({
            "必須スキルカバー率": "{:.1%}",
            "保有スキル数": "{:.0f}",
            "不足スキル数": "{:.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )


def render_team_composition_analysis(
    member_competence_df: pd.DataFrame,
    members_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    team_column: str = "職種"
) -> None:
    """
    チーム構成分析（スキルの相補性・冗長性）

    Args:
        member_competence_df: メンバー習得力量データ
        members_df: メンバーマスタ
        competence_master_df: 力量マスタ
        team_column: チーム分類カラム（職種、部署など）
    """
    st.markdown(f"### 🤝 チーム構成分析（{team_column}別）")

    st.markdown("""
    **分析視点**:
    - **カバレッジ**: チームが保有するユニークスキル数
    - **冗長性**: 複数メンバーが保有するスキルの割合（リスク分散）
    - **専門性**: 1名のみが保有するスキルの割合（属人化リスク）
    """)

    # メンバー情報とマージ
    if "メンバーコード" not in member_competence_df.columns or "メンバーコード" not in members_df.columns:
        st.warning("⚠️ メンバーコードが不足しているため、この分析をスキップします")
        return

    merged = member_competence_df.merge(
        members_df[["メンバーコード", team_column]],
        on="メンバーコード",
        how="left"
    )

    merged = merged[merged[team_column].notna()]

    if len(merged) == 0:
        st.warning(f"⚠️ {team_column}情報が見つかりません")
        return

    # チームごとの分析
    team_stats = []

    for team_name, team_data in merged.groupby(team_column):
        member_count = team_data["メンバーコード"].nunique()
        unique_skills = team_data["力量コード"].nunique()

        # スキルごとの保有メンバー数
        skill_member_counts = team_data.groupby("力量コード")["メンバーコード"].nunique()

        # 冗長性（2名以上が保有）
        redundant_skills = (skill_member_counts >= 2).sum()
        redundancy_rate = redundant_skills / unique_skills if unique_skills > 0 else 0

        # 属人化リスク（1名のみが保有）
        single_person_skills = (skill_member_counts == 1).sum()
        risk_rate = single_person_skills / unique_skills if unique_skills > 0 else 0

        team_stats.append({
            team_column: team_name,
            "メンバー数": member_count,
            "ユニークスキル数": unique_skills,
            "1人あたりスキル数": unique_skills / member_count if member_count > 0 else 0,
            "冗長性率": redundancy_rate,
            "属人化リスク率": risk_rate
        })

    team_stats_df = pd.DataFrame(team_stats)
    team_stats_df = team_stats_df.sort_values("ユニークスキル数", ascending=False)

    # 可視化
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="冗長性率（複数保有）",
        x=team_stats_df[team_column],
        y=team_stats_df["冗長性率"] * 100,
        marker_color='lightgreen'
    ))

    fig.add_trace(go.Bar(
        name="属人化リスク率（1名のみ）",
        x=team_stats_df[team_column],
        y=team_stats_df["属人化リスク率"] * 100,
        marker_color='salmon'
    ))

    fig.update_layout(
        title=f"{team_column}別スキル冗長性 vs 属人化リスク",
        xaxis_title=team_column,
        yaxis_title="割合 (%)",
        barmode='group',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # データテーブル
    st.markdown("#### 📊 チーム構成統計")
    st.dataframe(
        team_stats_df.style.format({
            "メンバー数": "{:.0f}",
            "ユニークスキル数": "{:.0f}",
            "1人あたりスキル数": "{:.1f}",
            "冗長性率": "{:.1%}",
            "属人化リスク率": "{:.1%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 推奨事項
    st.markdown("#### 💡 推奨アクション")

    high_risk_teams = team_stats_df[team_stats_df["属人化リスク率"] > 0.5]
    low_redundancy_teams = team_stats_df[team_stats_df["冗長性率"] < 0.3]

    if len(high_risk_teams) > 0:
        st.warning(f"⚠️ **高い属人化リスク**: {', '.join(high_risk_teams[team_column].tolist())} でスキルの属人化が進んでいます。クロストレーニングを推奨します。")

    if len(low_redundancy_teams) > 0:
        st.warning(f"⚠️ **低い冗長性**: {', '.join(low_redundancy_teams[team_column].tolist())} でスキルのバックアップ体制が不足しています。")


def render_competency_based_org_design(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame
) -> None:
    """
    コンピテンシーベースの組織設計分析

    Args:
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
    """
    st.markdown("### 🏢 コンピテンシーベース組織設計")

    st.markdown("""
    **データドリブン組織設計アプローチ**:
    スキルの共起パターンから自然なチーム編成を発見します。
    """)

    # スキル共起マトリックスの作成
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # メンバー×スキルマトリックス
    member_skill_matrix = member_competence_df.pivot_table(
        index="メンバーコード",
        columns="力量コード",
        values="レベル" if "レベル" in member_competence_df.columns else "メンバーコード",
        aggfunc="count",
        fill_value=0
    )

    # クラスタリング（メンバーを類似スキルセットでグループ化）
    n_clusters = min(5, len(member_skill_matrix) // 3)  # 最大5クラスタ、最小3名/クラスタ

    if n_clusters < 2:
        st.warning("⚠️ クラスタリングには最低6名以上のメンバーが必要です")
        return

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(member_skill_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    # クラスタ情報を追加
    cluster_assignment = pd.DataFrame({
        "メンバーコード": member_skill_matrix.index,
        "クラスター": clusters
    })

    # メンバー情報とマージ
    cluster_members = cluster_assignment.merge(
        members_df[["メンバーコード", "メンバー名", "職種", "役職"]],
        on="メンバーコード",
        how="left"
    )

    # クラスタごとの代表的なスキルを特定
    st.markdown("#### 🎯 スキルベース人材クラスター")

    for cluster_id in range(n_clusters):
        cluster_member_codes = cluster_assignment[
            cluster_assignment["クラスター"] == cluster_id
        ]["メンバーコード"].tolist()

        cluster_skills = member_competence_df[
            member_competence_df["メンバーコード"].isin(cluster_member_codes)
        ]

        # クラスタの代表的なスキル（保有率が高いスキル）
        skill_coverage = cluster_skills.groupby("力量コード").size() / len(cluster_member_codes)
        top_skills = skill_coverage.nlargest(5)

        # スキル名を取得
        skill_names = competence_master_df[
            competence_master_df["力量コード"].isin(top_skills.index)
        ].set_index("力量コード")["力量名"].to_dict()

        with st.expander(f"🔷 クラスター {cluster_id + 1} ({len(cluster_member_codes)}名)", expanded=(cluster_id == 0)):
            st.markdown(f"**代表的なスキル**:")
            for skill_code in top_skills.index:
                skill_name = skill_names.get(skill_code, skill_code)
                coverage_pct = top_skills[skill_code] * 100
                st.markdown(f"- {skill_name} ({coverage_pct:.0f}%)")

            st.markdown(f"**メンバー**:")
            cluster_member_list = cluster_members[cluster_members["クラスター"] == cluster_id]
            st.dataframe(
                cluster_member_list[["メンバー名", "職種", "役職"]],
                use_container_width=True,
                hide_index=True
            )


def render_enhanced_skill_gap_analysis(
    gap_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    percentile_used: float = 0.0
) -> None:
    """
    高度なスキルギャップ分析（データサイエンス視点）

    データサイエンティスト兼人事スペシャリストの視点で、
    戦略的な意思決定に役立つ包括的なスキルギャップ分析を提供。

    Args:
        gap_df: スキルギャップデータ（力量コード、力量名、現在保有率、目標保有率、ギャップなど）
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        percentile_used: 使用したパーセンタイル（0.0=全体平均、0.5=中央値、0.75=75%タイル）
    """

    st.markdown("---")
    st.markdown("---")
    st.markdown("## 🎓 高度なスキルギャップ分析")
    st.markdown("**データサイエンス × 人事スペシャリスト視点での戦略的分析**")

    st.markdown("""
    このセクションでは、組織のスキルギャップを多角的に分析し、
    データドリブンな人材育成戦略を策定するための洞察を提供します。
    """)

    # ============================================
    # 1. エグゼクティブサマリー
    # ============================================
    st.markdown("### 📊 エグゼクティブサマリー")

    col1, col2, col3, col4, col5 = st.columns(5)

    # KPI計算
    total_gaps = len(gap_df)

    # クリティカルギャップ（ギャップ率30%以上）
    if "保有率ギャップ率" in gap_df.columns:
        critical_gaps = len(gap_df[gap_df["保有率ギャップ率"] > 0.3])
        medium_gaps = len(gap_df[(gap_df["保有率ギャップ率"] > 0.1) & (gap_df["保有率ギャップ率"] <= 0.3)])
        avg_gap_rate = gap_df["保有率ギャップ率"].mean()
    else:
        critical_gaps = 0
        medium_gaps = 0
        avg_gap_rate = 0

    # 育成必要人数の推定（ギャップ率 × メンバー数）
    total_training_needs = 0
    if "保有率ギャップ率" in gap_df.columns:
        total_training_needs = (gap_df["保有率ギャップ率"] * len(members_df)).sum()

    with col1:
        st.metric("総スキルギャップ数", f"{total_gaps}件")

    with col2:
        st.metric(
            "🔴 クリティカル",
            f"{critical_gaps}件",
            delta=f"{(critical_gaps/total_gaps*100):.0f}%",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "🟡 中程度",
            f"{medium_gaps}件",
            delta=f"{(medium_gaps/total_gaps*100):.0f}%",
            delta_color="off"
        )

    with col4:
        st.metric(
            "平均ギャップ率",
            f"{avg_gap_rate*100:.1f}%"
        )

    with col5:
        st.metric(
            "育成必要人数",
            f"{int(total_training_needs)}人"
        )

    st.markdown("---")

    # ============================================
    # 2. 多次元優先度分析
    # ============================================
    st.markdown("### 🎯 多次元優先度分析")

    st.markdown("""
    **3つの重要指標でスキルギャップを評価**:
    - **ビジネスインパクト** (1-10): ビジネス成果への影響度
    - **緊急性** (1-10): 対応の緊急度
    - **習得難易度** (1-10): スキル習得の難しさ（低いほど習得しやすい）
    """)

    # ギャップ分析データフレームをコピー
    gap_analysis_df = gap_df.copy()

    # ビジネスインパクト推定（目標保有率の高さで代用）
    if "目標保有率" in gap_analysis_df.columns:
        gap_analysis_df["ビジネスインパクト"] = gap_analysis_df["目標保有率"] * 10
    else:
        gap_analysis_df["ビジネスインパクト"] = 5.0  # デフォルト

    # 緊急性推定（ギャップ率の大きさで代用）
    if "保有率ギャップ率" in gap_analysis_df.columns:
        gap_analysis_df["緊急性"] = gap_analysis_df["保有率ギャップ率"] * 10
    else:
        gap_analysis_df["緊急性"] = 5.0  # デフォルト

    # 習得難易度推定（現在保有率の低さで代用 - 保有率が低い = 難しい）
    if "現在保有率" in gap_analysis_df.columns:
        gap_analysis_df["習得難易度"] = (1 - gap_analysis_df["現在保有率"]) * 10
    else:
        gap_analysis_df["習得難易度"] = 5.0  # デフォルト

    # 総合優先度スコア（重み付き平均: ビジネスインパクト40%, 緊急性40%, 習得難易度の逆数20%）
    gap_analysis_df["優先度スコア"] = (
        gap_analysis_df["ビジネスインパクト"] * 0.4 +
        gap_analysis_df["緊急性"] * 0.4 +
        (100 - gap_analysis_df["習得難易度"] * 10) * 0.2  # 難易度が低いほど高スコア
    )

    # 優先度カテゴリ分類
    def categorize_priority(row):
        if row["優先度スコア"] >= 70:
            return "🔴 最優先（Strategic Focus）"
        elif row["優先度スコア"] >= 50:
            return "🟠 高優先度（High Priority）"
        elif row["優先度スコア"] >= 30:
            return "🟡 中優先度（Medium Priority）"
        else:
            return "🟢 低優先度（Low Priority）"

    gap_analysis_df["優先度カテゴリ"] = gap_analysis_df.apply(categorize_priority, axis=1)

    # バブルチャート作成
    fig = px.scatter(
        gap_analysis_df.head(50),  # 上位50スキルを表示
        x="ビジネスインパクト",
        y="緊急性",
        size="習得難易度",
        color="優先度カテゴリ",
        hover_name="力量名",
        hover_data={
            "ビジネスインパクト": ":.1f",
            "緊急性": ":.1f",
            "習得難易度": ":.1f",
            "優先度スコア": ":.1f"
        },
        title="スキルギャップ優先度マトリックス（3次元分析）",
        labels={
            "ビジネスインパクト": "ビジネスインパクト",
            "緊急性": "緊急性"
        },
        color_discrete_map={
            "🔴 最優先（Strategic Focus）": "red",
            "🟠 高優先度（High Priority）": "orange",
            "🟡 中優先度（Medium Priority）": "yellow",
            "🟢 低優先度（Low Priority）": "green"
        }
    )

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 優先度カテゴリ別の分布
    priority_counts = gap_analysis_df["優先度カテゴリ"].value_counts()

    fig_pie = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title="優先度カテゴリ分布",
        color=priority_counts.index,
        color_discrete_map={
            "🔴 最優先（Strategic Focus）": "red",
            "🟠 高優先度（High Priority）": "orange",
            "🟡 中優先度（Medium Priority）": "yellow",
            "🟢 低優先度（Low Priority）": "green"
        }
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ============================================
    # 3. ROI（投資対効果）分析
    # ============================================
    st.markdown("### 💰 ROI（投資対効果）分析")

    st.markdown("""
    **スキル育成のROIを定量評価**:
    各スキルへの投資コストとビジネス価値を推定し、優先順位付けをサポートします。
    """)

    # ROIパラメータ設定
    roi_col1, roi_col2, roi_col3 = st.columns(3)

    with roi_col1:
        training_cost_per_skill = st.number_input(
            "1人あたり育成コスト（万円/スキル）",
            min_value=1,
            max_value=500,
            value=50,
            step=10,
            help="1つのスキルを1人に習得させるのにかかる平均コスト"
        )

    with roi_col2:
        months_per_level = st.number_input(
            "習得難易度1あたりの期間（月）",
            min_value=0.5,
            max_value=12.0,
            value=2.0,
            step=0.5,
            help="習得難易度1ポイントあたりの習得期間の目安"
        )

    with roi_col3:
        business_value_multiplier = st.number_input(
            "ビジネス価値倍率",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="育成コストに対するビジネス価値の倍率"
        )

    # ROI計算
    roi_df = gap_analysis_df.copy()

    # 必要な育成人数
    roi_df["育成必要人数"] = (roi_df["保有率ギャップ率"] * len(members_df)).round(0).astype(int)

    # 総投資コスト（万円）
    roi_df["総投資コスト"] = roi_df["育成必要人数"] * training_cost_per_skill

    # 習得期間（月）
    roi_df["推定習得期間"] = (roi_df["習得難易度"] * months_per_level).round(1)

    # ビジネス価値（万円）- ビジネスインパクトに基づく
    roi_df["推定ビジネス価値"] = (
        roi_df["ビジネスインパクト"] *
        roi_df["育成必要人数"] *
        training_cost_per_skill *
        business_value_multiplier
    )

    # ROI = (ビジネス価値 - 投資コスト) / 投資コスト × 100
    roi_df["ROI率"] = (
        (roi_df["推定ビジネス価値"] - roi_df["総投資コスト"]) /
        roi_df["総投資コスト"] * 100
    ).round(1)

    # ROI上位10スキル
    top_roi_skills = roi_df.nlargest(10, "ROI率")

    fig_roi = px.bar(
        top_roi_skills,
        x="力量名",
        y="ROI率",
        color="優先度カテゴリ",
        title="ROI上位10スキル（投資対効果が高いスキル）",
        labels={"ROI率": "ROI率 (%)"},
        hover_data=["総投資コスト", "推定ビジネス価値", "推定習得期間"],
        color_discrete_map={
            "🔴 最優先（Strategic Focus）": "red",
            "🟠 高優先度（High Priority）": "orange",
            "🟡 中優先度（Medium Priority）": "yellow",
            "🟢 低優先度（Low Priority）": "green"
        }
    )

    fig_roi.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_roi, use_container_width=True)

    # ROIサマリー統計
    roi_summary_col1, roi_summary_col2, roi_summary_col3 = st.columns(3)

    with roi_summary_col1:
        total_investment = roi_df["総投資コスト"].sum()
        st.metric("総投資コスト", f"¥{total_investment:,.0f}万円")

    with roi_summary_col2:
        total_value = roi_df["推定ビジネス価値"].sum()
        st.metric("総ビジネス価値", f"¥{total_value:,.0f}万円")

    with roi_summary_col3:
        overall_roi = ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0
        st.metric("全体ROI", f"{overall_roi:.1f}%")

    st.markdown("---")

    # ============================================
    # 4. スキルギャップクラスタリング
    # ============================================
    st.markdown("### 🔬 スキルギャップクラスタリング（機械学習分析）")

    st.markdown("""
    **K-meansクラスタリングによるパターン発見**:
    類似した特性を持つスキルギャップをグループ化し、効率的な育成戦略を提案します。
    """)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # 特徴量: ビジネスインパクト、緊急性、習得難易度
    cluster_features = gap_analysis_df[[
        "ビジネスインパクト", "緊急性", "習得難易度"
    ]].fillna(0)

    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_features)

    # K-meansクラスタリング（4クラスター）
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gap_analysis_df["クラスター"] = kmeans.fit_predict(features_scaled)

    # クラスター特性の分析
    cluster_characteristics = []

    for cluster_id in range(n_clusters):
        cluster_data = gap_analysis_df[gap_analysis_df["クラスター"] == cluster_id]

        avg_impact = cluster_data["ビジネスインパクト"].mean()
        avg_urgency = cluster_data["緊急性"].mean()
        avg_difficulty = cluster_data["習得難易度"].mean()

        # クラスター特性に基づいてラベル付け
        if avg_impact >= 7 and avg_urgency >= 7:
            label = "🔴 戦略的最重要クラスター"
        elif avg_impact >= 6 and avg_difficulty <= 5:
            label = "🟢 クイックウィンクラスター"
        elif avg_urgency >= 7:
            label = "🟠 緊急対応クラスター"
        else:
            label = "🟡 中長期育成クラスター"

        cluster_characteristics.append({
            "クラスター": f"クラスター {cluster_id}",
            "ラベル": label,
            "スキル数": len(cluster_data),
            "平均ビジネスインパクト": f"{avg_impact:.2f}",
            "平均緊急性": f"{avg_urgency:.2f}",
            "平均習得難易度": f"{avg_difficulty:.2f}",
            "推奨アプローチ": _get_cluster_recommendation(avg_impact, avg_urgency, avg_difficulty)
        })

        # ラベルを更新
        gap_analysis_df.loc[gap_analysis_df["クラスター"] == cluster_id, "クラスターラベル"] = label

    # クラスター特性表示
    cluster_df = pd.DataFrame(cluster_characteristics)

    st.markdown("##### 📋 スキルギャップクラスター分析結果")
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    # 3D散布図（インタラクティブ）
    fig_3d = px.scatter_3d(
        gap_analysis_df.head(100),
        x="ビジネスインパクト",
        y="緊急性",
        z="習得難易度",
        color="クラスターラベル",
        hover_name="力量名",
        title="スキルギャップ 3D クラスター可視化",
        labels={
            "ビジネスインパクト": "ビジネスインパクト",
            "緊急性": "緊急性",
            "習得難易度": "習得難易度"
        }
    )

    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

    # クラスターラベルをroi_dfにコピー（エクスポート用）
    roi_df["クラスター"] = gap_analysis_df["クラスター"]
    roi_df["クラスターラベル"] = gap_analysis_df["クラスターラベル"]

    st.markdown("---")

    # ============================================
    # 5. アクションプラン生成
    # ============================================
    st.markdown("#### 📝 データドリブン・アクションプラン")

    st.markdown("""
    **HR戦略への落とし込み**: 分析結果を実行可能なアクションに変換
    """)

    # 最優先スキルTOP5の詳細アクションプラン
    top_priority_skills = roi_df.nlargest(5, "優先度スコア")

    for idx, (_, skill) in enumerate(top_priority_skills.iterrows(), 1):
        with st.expander(f"🎯 アクションプラン {idx}: {skill['力量名']}", expanded=(idx == 1)):
            st.markdown(f"**優先度**: {skill['優先度カテゴリ']} （スコア: {skill['優先度スコア']:.1f}/100）")

            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:
                st.metric("現在保有率", f"{skill['現在保有率']*100:.1f}%")
                st.metric("目標保有率", f"{skill['目標保有率']*100:.1f}%")

            with action_col2:
                st.metric("ギャップ率", f"{skill['保有率ギャップ率']*100:.1f}%")
                st.metric("育成必要人数", f"{int(skill['育成必要人数'])}人")

            with action_col3:
                st.metric("推定コスト", f"¥{skill['総投資コスト']:,.0f}万円")
                st.metric("推定ROI", f"{skill['ROI率']:.1f}%")

            # アクション推奨事項
            st.markdown("**推奨アクション**:")
            recommendations = _generate_action_recommendations(
                skill['優先度スコア'],
                skill['習得難易度'],
                skill['育成必要人数']
            )

            for rec in recommendations:
                st.markdown(f"- {rec}")

            # タイムライン
            st.markdown("**実行タイムライン**:")
            timeline = _generate_timeline(skill['推定習得期間'])

            for phase, description in timeline.items():
                st.markdown(f"- **{phase}**: {description}")

    st.markdown("---")

    # ============================================
    # 6. ポートフォリオ最適化
    # ============================================
    st.markdown("#### 📊 スキルポートフォリオ最適化")

    st.markdown("""
    **組織全体のスキル投資バランスを最適化**:
    現在の投資分布と理想的な分布を比較し、リソース配分を最適化します。
    """)

    # 優先度カテゴリ別の投資配分
    portfolio_df = roi_df.groupby("優先度カテゴリ").agg({
        "総投資コスト": "sum",
        "推定ビジネス価値": "sum",
        "力量コード": "count"
    }).reset_index()

    portfolio_df.columns = ["優先度カテゴリ", "総投資コスト", "総ビジネス価値", "スキル数"]

    # 現在の投資割合
    portfolio_df["現在の投資割合"] = portfolio_df["総投資コスト"] / portfolio_df["総投資コスト"].sum()

    # 理想的な投資割合（優先度に応じて）
    ideal_allocation = {
        "🔴 最優先（Strategic Focus）": 0.50,
        "🟠 高優先度（High Priority）": 0.30,
        "🟡 中優先度（Medium Priority）": 0.15,
        "🟢 低優先度（Low Priority）": 0.05
    }

    portfolio_df["理想的な投資割合"] = portfolio_df["優先度カテゴリ"].map(ideal_allocation).fillna(0)

    # 比較チャート
    fig_portfolio = go.Figure()

    fig_portfolio.add_trace(go.Bar(
        name="現在の投資割合",
        x=portfolio_df["優先度カテゴリ"],
        y=portfolio_df["現在の投資割合"] * 100,
        marker_color='lightblue'
    ))

    fig_portfolio.add_trace(go.Bar(
        name="理想的な投資割合",
        x=portfolio_df["優先度カテゴリ"],
        y=portfolio_df["理想的な投資割合"] * 100,
        marker_color='orange'
    ))

    fig_portfolio.update_layout(
        title="スキル投資ポートフォリオ: 現在 vs 理想",
        xaxis_title="優先度カテゴリ",
        yaxis_title="投資割合 (%)",
        barmode='group',
        height=500
    )

    st.plotly_chart(fig_portfolio, use_container_width=True)

    # ポートフォリオ最適化の推奨事項
    st.markdown("##### 💡 ポートフォリオ最適化の推奨事項")

    for _, row in portfolio_df.iterrows():
        category = row["優先度カテゴリ"]
        current_pct = row["現在の投資割合"] * 100
        ideal_pct = row["理想的な投資割合"] * 100
        diff = current_pct - ideal_pct

        if abs(diff) > 10:  # 10%以上の差がある場合に推奨
            if diff > 0:
                st.info(f"**{category}**: 現在の投資が{diff:.1f}%過剰です。他の優先度へのリソース移動を検討してください。")
            else:
                st.warning(f"**{category}**: 現在の投資が{abs(diff):.1f}%不足しています。追加投資を検討してください。")

    st.markdown("---")

    # ============================================
    # 7. データエクスポート
    # ============================================
    st.markdown("### 💾 高度な分析結果エクスポート")

    st.markdown("""
    **包括的な分析結果をエクスポート**:
    全ての分析指標を含む詳細レポートをダウンロードできます。
    """)

    # エクスポート用データフレーム（全指標を含む）
    export_df = roi_df[[
        "力量名", "現在保有率", "目標保有率", "保有率ギャップ率",
        "ビジネスインパクト", "緊急性", "習得難易度", "優先度スコア", "優先度カテゴリ",
        "育成必要人数", "総投資コスト", "ROI率", "推定習得期間", "クラスターラベル"
    ]].copy()

    # CSV出力
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 詳細分析結果をCSV出力",
        data=csv_data,
        file_name="enhanced_skill_gap_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("✅ 高度なスキルギャップ分析が完了しました。")


def _get_cluster_recommendation(avg_impact: float, avg_urgency: float, avg_difficulty: float) -> str:
    """
    クラスター特性に基づいた推奨アプローチを生成

    Args:
        avg_impact: 平均ビジネスインパクト
        avg_urgency: 平均緊急性
        avg_difficulty: 平均習得難易度

    Returns:
        推奨アプローチの説明
    """
    if avg_impact >= 7 and avg_urgency >= 7:
        return "最優先で集中投資。外部研修・専門家招聘を検討"
    elif avg_impact >= 6 and avg_difficulty <= 5:
        return "短期集中育成プログラム。社内トレーナー活用"
    elif avg_urgency >= 7:
        return "緊急対応チーム編成。即戦力の外部採用も検討"
    elif avg_difficulty >= 7:
        return "長期育成計画。段階的なスキルアップパスを設計"
    else:
        return "OJT中心の育成。既存メンバーからの伝承"


def _generate_action_recommendations(priority_score: float, difficulty: float, training_needs: int) -> List[str]:
    """
    アクション推奨事項を生成

    Args:
        priority_score: 優先度スコア
        difficulty: 習得難易度
        training_needs: 育成必要人数

    Returns:
        推奨アクションのリスト
    """
    recommendations = []

    # 優先度に基づく推奨
    if priority_score >= 70:
        recommendations.append("🔴 **最優先対応**: 経営層の承認を得て即座に育成プログラムを開始")
        recommendations.append("専任トレーナーの配置または外部専門家の招聘を検討")
    elif priority_score >= 50:
        recommendations.append("🟠 **高優先度**: 次四半期の育成計画に組み込み")

    # 難易度に基づく推奨
    if difficulty >= 7:
        recommendations.append("📚 **高難度スキル**: 段階的な育成パス（基礎→応用→実践）を設計")
        recommendations.append("メンタリング制度の活用または外部研修の導入")
    elif difficulty <= 3:
        recommendations.append("⚡ **習得容易**: 短期集中研修（1-2週間）で効果的に習得可能")

    # 育成人数に基づく推奨
    if training_needs >= 10:
        recommendations.append(f"👥 **大規模育成**: {training_needs}名の育成が必要 - 集合研修やeラーニングの活用を推奨")
    elif training_needs >= 5:
        recommendations.append(f"👥 **中規模育成**: {training_needs}名の育成 - グループ研修が効率的")
    else:
        recommendations.append(f"👤 **少人数育成**: {training_needs}名 - マンツーマン指導やOJTが効果的")

    return recommendations


def _generate_timeline(duration_months: float) -> Dict[str, str]:
    """
    実行タイムラインを生成

    Args:
        duration_months: 推定習得期間（月）

    Returns:
        フェーズごとのタイムライン辞書
    """
    timeline = {}

    timeline["第1フェーズ（1-2週間）"] = "対象者選定、ベースライン評価、育成計画策定"

    if duration <= 3:
        timeline["第2フェーズ（1ヶ月）"] = "集中トレーニング実施"
        timeline["第3フェーズ（2-3ヶ月）"] = "実践・OJT、スキル定着確認"
    elif duration <= 6:
        timeline["第2フェーズ（1-3ヶ月）"] = "基礎トレーニング実施"
        timeline["第3フェーズ（4-6ヶ月）"] = "実践・OJT、中間評価"
        timeline["第4フェーズ（6ヶ月以降）"] = "スキル定着、最終評価"
    else:
        timeline["第2フェーズ（1-4ヶ月）"] = "基礎理論習得"
        timeline["第3フェーズ（5-8ヶ月）"] = "実践演習・プロジェクト適用"
        timeline["第4フェーズ（9-12ヶ月）"] = "実務適用・メンタリング"
        timeline["第5フェーズ（12ヶ月以降）"] = "マスタリー達成、後進育成"

    return timeline


def render_enhanced_skill_matrix_analysis(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    filters: Dict = {}
) -> None:
    """
    高度な人材スキルマトリックス分析（データサイエンス視点）

    データサイエンティスト兼人事スペシャリストの視点で、
    人材のスキル保有状況を多角的に分析し、戦略的な人材配置や
    育成計画に役立つ洞察を提供。

    Args:
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        filters: フィルタ条件の辞書
    """
    import io

    st.markdown("---")
    st.markdown("---")
    st.markdown("## 👥 高度な人材スキルマトリックス分析")
    st.markdown("**データサイエンス × 人事スペシャリスト視点での人材スキル分析**")

    st.markdown("""
    このセクションでは、組織の人材スキル保有状況を多角的に分析し、
    戦略的な人材配置や育成計画のための洞察を提供します。
    """)

    # ============================================
    # 1. エグゼクティブダッシュボード
    # ============================================
    st.markdown("### 📊 人材スキル総合ダッシュボード")

    # メンバー×スキルマトリックス作成
    member_skill_matrix = member_competence_df.pivot_table(
        index="メンバーコード",
        columns="力量コード",
        values="レベル" if "レベル" in member_competence_df.columns else "メンバーコード",
        aggfunc="max" if "レベル" in member_competence_df.columns else "count",
        fill_value=0
    )

    # KPI計算
    total_members = len(member_skill_matrix)
    total_skills = len(competence_master_df)
    total_skill_instances = (member_skill_matrix > 0).sum().sum()
    avg_skills_per_member = total_skill_instances / total_members if total_members > 0 else 0
    skill_coverage_rate = (member_skill_matrix > 0).any(axis=0).sum() / total_skills if total_skills > 0 else 0

    # スキル多様性（ジニ係数的指標）
    skills_per_member = (member_skill_matrix > 0).sum(axis=1)
    skill_diversity = skills_per_member.std() / skills_per_member.mean() if skills_per_member.mean() > 0 else 0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.metric("総メンバー数", f"{total_members}名")

    with kpi_col2:
        st.metric("総スキル数", f"{total_skills}件")

    with kpi_col3:
        st.metric("1人あたり平均スキル数", f"{avg_skills_per_member:.1f}件")

    with kpi_col4:
        st.metric("組織スキルカバー率", f"{skill_coverage_rate*100:.1f}%")

    with kpi_col5:
        st.metric("スキル分布の多様性", f"{skill_diversity:.2f}",
                 help="低いほど均一、高いほどバラツキが大きい")

    st.markdown("---")

    # ============================================
    # 2. スキル分布ヒートマップ
    # ============================================
    st.markdown("### 🔥 人材スキル分布ヒートマップ")

    st.markdown("""
    **組織全体のスキル保有パターンを可視化**:
    メンバー×スキルの保有状況をヒートマップで表示し、
    スキルの偏りや充足状況を一目で把握できます。
    """)

    # フィルタリング
    filtered_members = members_df.copy()
    if "職種" in filters and filters["職種"]:
        filtered_members = filtered_members[filtered_members["職種"].isin(filters["職種"])]
    if "役職" in filters and filters["役職"]:
        filtered_members = filtered_members[filtered_members["役職"].isin(filters["役職"])]
    if "等級" in filters and filters["等級"]:
        filtered_members = filtered_members[filtered_members["職能・等級"].isin(filters["等級"])]

    filtered_member_codes = filtered_members["メンバーコード"].tolist()

    # フィルタ後のマトリックス
    filtered_matrix = member_skill_matrix.loc[
        member_skill_matrix.index.isin(filtered_member_codes)
    ]

    # 上位30スキル（保有率が高い順）
    top_skills = (filtered_matrix > 0).sum(axis=0).nlargest(30)
    top_skill_codes = top_skills.index.tolist()

    # スキル名を取得
    skill_names_dict = competence_master_df.set_index("力量コード")["力量名"].to_dict()

    heatmap_matrix = filtered_matrix[top_skill_codes].head(50)  # 上位50メンバー

    # メンバー名を取得
    member_names_dict = members_df.set_index("メンバーコード")["メンバー名"].to_dict()
    heatmap_matrix.index = heatmap_matrix.index.map(lambda x: member_names_dict.get(x, x))
    heatmap_matrix.columns = heatmap_matrix.columns.map(lambda x: skill_names_dict.get(x, x))

    # ヒートマップ作成
    fig_heatmap = px.imshow(
        heatmap_matrix,
        labels=dict(x="スキル", y="メンバー", color="レベル"),
        title=f"人材スキルマトリックス（上位50メンバー × 保有率上位30スキル）",
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig_heatmap.update_layout(height=800)
    fig_heatmap.update_xaxes(tickangle=-45)

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # ============================================
    # 3. 人材スキル多様性分析（T型人材分析）
    # ============================================
    st.markdown("### 🎯 人材スキル多様性分析（T型人材評価）")

    st.markdown("""
    **T型人材の識別と評価**:
    - **広さ（Breadth）**: 保有スキル数（水平バー）
    - **深さ（Depth）**: 高レベルスキルの割合（垂直バー）
    - T型人材は広さと深さの両方を兼ね備えた人材
    """)

    # メンバーごとの分析
    member_analysis = []

    for member_code in filtered_member_codes:
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ]

        skill_count = len(member_skills)

        if "レベル" in member_skills.columns:
            # レベルが3以上を「深い」と定義
            deep_skills = len(member_skills[pd.to_numeric(member_skills["レベル"], errors='coerce') >= 3])
            depth_ratio = deep_skills / skill_count if skill_count > 0 else 0
        else:
            depth_ratio = 0

        member_info = members_df[members_df["メンバーコード"] == member_code].iloc[0]

        # T型人材スコア（広さ × 深さ）
        t_shape_score = skill_count * depth_ratio

        member_analysis.append({
            "メンバーコード": member_code,
            "メンバー名": member_info.get("メンバー名", ""),
            "職種": member_info.get("職種", ""),
            "役職": member_info.get("役職", ""),
            "スキル数（広さ）": skill_count,
            "深さ比率": depth_ratio,
            "T型スコア": t_shape_score
        })

    member_analysis_df = pd.DataFrame(member_analysis)
    member_analysis_df = member_analysis_df.sort_values("T型スコア", ascending=False)

    # T型人材分類
    def classify_t_shape(row):
        if row["スキル数（広さ）"] >= member_analysis_df["スキル数（広さ）"].quantile(0.75) and \
           row["深さ比率"] >= member_analysis_df["深さ比率"].quantile(0.75):
            return "🌟 T型人材（広く深い）"
        elif row["スキル数（広さ）"] >= member_analysis_df["スキル数（広さ）"].quantile(0.75):
            return "📏 I型人材（広く浅い）"
        elif row["深さ比率"] >= member_analysis_df["深さ比率"].quantile(0.75):
            return "📌 専門特化型（狭く深い）"
        else:
            return "🔰 育成中"

    member_analysis_df["人材タイプ"] = member_analysis_df.apply(classify_t_shape, axis=1)

    # 散布図（広さ vs 深さ）
    fig_t_shape = px.scatter(
        member_analysis_df,
        x="スキル数（広さ）",
        y="深さ比率",
        size="T型スコア",
        color="人材タイプ",
        hover_name="メンバー名",
        hover_data=["職種", "役職"],
        title="T型人材マッピング（広さ × 深さ）",
        labels={
            "スキル数（広さ）": "スキル数（広さ）",
            "深さ比率": "深さ比率（高レベルスキル割合）"
        }
    )

    # 四分位線を追加
    median_breadth = member_analysis_df["スキル数（広さ）"].median()
    median_depth = member_analysis_df["深さ比率"].median()

    fig_t_shape.add_hline(y=median_depth, line_dash="dash", line_color="gray", opacity=0.5)
    fig_t_shape.add_vline(x=median_breadth, line_dash="dash", line_color="gray", opacity=0.5)

    fig_t_shape.update_layout(height=600)
    st.plotly_chart(fig_t_shape, use_container_width=True)

    # T型人材ランキング
    st.markdown("#### 🏆 T型人材トップ10")
    st.dataframe(
        member_analysis_df.head(10)[["メンバー名", "職種", "役職", "スキル数（広さ）", "深さ比率", "T型スコア", "人材タイプ"]].style.format({
            "スキル数（広さ）": "{:.0f}",
            "深さ比率": "{:.1%}",
            "T型スコア": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 人材タイプ分布
    type_distribution = member_analysis_df["人材タイプ"].value_counts()

    fig_type_pie = px.pie(
        values=type_distribution.values,
        names=type_distribution.index,
        title="人材タイプ分布",
        hole=0.4
    )

    st.plotly_chart(fig_type_pie, use_container_width=True)

    st.markdown("---")

    # ============================================
    # 4. スキル共起ネットワーク分析
    # ============================================
    st.markdown("### 🕸️ スキル共起ネットワーク分析")

    st.markdown("""
    **スキルの関連性パターンを発見**:
    よく一緒に保有されるスキルの組み合わせを分析し、
    効果的なスキルセット形成のヒントを提供します。
    """)

    # スキル共起マトリックス（上位20スキルに絞る）
    top_20_skills = (member_skill_matrix > 0).sum(axis=0).nlargest(20).index.tolist()

    cooccurrence_matrix = np.zeros((len(top_20_skills), len(top_20_skills)))

    for i, skill1 in enumerate(top_20_skills):
        for j, skill2 in enumerate(top_20_skills):
            if i != j:
                # 両方のスキルを持っているメンバー数
                cooccurrence = ((member_skill_matrix[skill1] > 0) & (member_skill_matrix[skill2] > 0)).sum()
                cooccurrence_matrix[i, j] = cooccurrence

    # スキル名に変換
    top_20_skill_names = [skill_names_dict.get(code, code) for code in top_20_skills]

    # ヒートマップ作成
    fig_cooccurrence = px.imshow(
        cooccurrence_matrix,
        labels=dict(x="スキル", y="スキル", color="共起人数"),
        x=top_20_skill_names,
        y=top_20_skill_names,
        title="スキル共起マトリックス（保有率上位20スキル）",
        color_continuous_scale="Greens"
    )

    fig_cooccurrence.update_layout(height=700)
    fig_cooccurrence.update_xaxes(tickangle=-45)
    fig_cooccurrence.update_yaxes(tickangle=0)

    st.plotly_chart(fig_cooccurrence, use_container_width=True)

    # 強い共起関係（共起人数が多い組み合わせトップ10）
    cooccurrence_pairs = []
    for i in range(len(top_20_skills)):
        for j in range(i+1, len(top_20_skills)):
            cooccurrence_pairs.append({
                "スキル1": top_20_skill_names[i],
                "スキル2": top_20_skill_names[j],
                "共起人数": int(cooccurrence_matrix[i, j])
            })

    cooccurrence_df = pd.DataFrame(cooccurrence_pairs)
    cooccurrence_df = cooccurrence_df.sort_values("共起人数", ascending=False).head(10)

    st.markdown("#### 🔗 強い関連性を持つスキルペア（トップ10）")
    st.dataframe(cooccurrence_df, use_container_width=True, hide_index=True)

    st.info("💡 **活用ヒント**: これらのスキルペアは一緒に育成することで相乗効果が期待できます。")

    st.markdown("---")

    # ============================================
    # 5. 人材セグメンテーション（クラスタリング）
    # ============================================
    st.markdown("### 🔬 人材セグメンテーション（機械学習分析）")

    st.markdown("""
    **K-meansクラスタリングによる人材グループ化**:
    スキルパターンの類似性に基づいてメンバーをグループ化し、
    効果的なチーム編成や育成施策の立案を支援します。
    """)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    # 標準化
    scaler = StandardScaler()
    member_features_scaled = scaler.fit_transform(member_skill_matrix)

    # クラスタ数の決定（メンバー数に応じて）
    n_member_clusters = min(5, max(3, total_members // 10))

    if total_members < 10:
        st.warning("⚠️ クラスタリングには最低10名以上のメンバーが必要です")
    else:
        kmeans_members = KMeans(n_clusters=n_member_clusters, random_state=42, n_init=10)
        member_clusters = kmeans_members.fit_predict(member_features_scaled)

        # クラスタ情報を追加
        cluster_assignment_df = pd.DataFrame({
            "メンバーコード": member_skill_matrix.index,
            "クラスター": member_clusters
        })

        # メンバー情報とマージ
        cluster_members_df = cluster_assignment_df.merge(
            members_df[["メンバーコード", "メンバー名", "職種", "役職"]],
            on="メンバーコード",
            how="left"
        )

        # クラスタごとの代表的なスキルを特定
        st.markdown("#### 🎯 人材クラスター分析結果")

        for cluster_id in range(n_member_clusters):
            cluster_member_codes = cluster_assignment_df[
                cluster_assignment_df["クラスター"] == cluster_id
            ]["メンバーコード"].tolist()

            cluster_skills_df = member_competence_df[
                member_competence_df["メンバーコード"].isin(cluster_member_codes)
            ]

            # クラスタの代表的なスキル（保有率が高いスキル）
            skill_coverage = cluster_skills_df.groupby("力量コード").size() / len(cluster_member_codes)
            top_cluster_skills = skill_coverage.nlargest(5)

            # スキル名を取得
            cluster_skill_names = competence_master_df[
                competence_master_df["力量コード"].isin(top_cluster_skills.index)
            ].set_index("力量コード")["力量名"].to_dict()

            with st.expander(f"🔷 クラスター {cluster_id + 1} ({len(cluster_member_codes)}名)", expanded=(cluster_id == 0)):
                st.markdown(f"**代表的なスキルセット**:")
                for skill_code in top_cluster_skills.index:
                    skill_name = cluster_skill_names.get(skill_code, skill_code)
                    coverage_pct = top_cluster_skills[skill_code] * 100
                    st.markdown(f"- {skill_name} ({coverage_pct:.0f}%)")

                st.markdown(f"**所属メンバー**:")
                cluster_member_list = cluster_members_df[cluster_members_df["クラスター"] == cluster_id]
                st.dataframe(
                    cluster_member_list[["メンバー名", "職種", "役職"]],
                    use_container_width=True,
                    hide_index=True
                )

                # クラスタの特徴
                st.markdown("**クラスターの特徴**:")
                avg_skills = member_analysis_df[
                    member_analysis_df["メンバーコード"].isin(cluster_member_codes)
                ]["スキル数（広さ）"].mean()
                st.markdown(f"- 平均スキル保有数: {avg_skills:.1f}件")

        st.markdown("---")

    # ============================================
    # 6. 高度なデータエクスポート
    # ============================================
    st.markdown("### 💾 高度な分析結果エクスポート")

    st.markdown("""
    **包括的な人材スキル分析結果をエクスポート**:
    T型人材評価、スキル多様性指標など全ての分析結果をダウンロードできます。
    """)

    # エクスポート用データフレーム
    export_member_df = member_analysis_df.copy()

    # CSV出力
    csv_buffer = io.StringIO()
    export_member_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 人材スキル分析結果をCSV出力",
        data=csv_data,
        file_name="enhanced_skill_matrix_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("✅ 高度な人材スキルマトリックス分析が完了しました。")


# 必要なモジュールインポート
import io
