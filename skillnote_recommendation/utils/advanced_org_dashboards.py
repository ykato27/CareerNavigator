"""
高度な組織分析ダッシュボードコンポーネント

データサイエンティスト視点での戦略的人材分析機能を提供
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

    parts = str(category_name).split(" > ")
    if level > len(parts):
        return " > ".join(parts)

    return " > ".join(parts[:level])


def format_category_for_display(category_path: str) -> str:
    """
    カテゴリパスを階層的に表示するためにフォーマット

    Args:
        category_path: カテゴリパス（例: "技術 > プログラミング > Python"）

    Returns:
        階層に応じてインデント付きの最終階層名
    """
    if pd.isna(category_path) or category_path == "未分類":
        return "未分類"

    parts = str(category_path).split(" > ")
    level = len(parts)

    # 階層に応じてインデントを追加
    indent = "  " * (level - 1)
    return indent + parts[-1]  # 最後の階層のみ表示


def render_hierarchical_category_heatmap(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    group_by: str = "職種"
) -> None:
    """
    ①カテゴリ×職種の階層的ヒートマップ

    カテゴリを階層的に選択でき、平均値/中央値を選択可能

    Args:
        member_competence_df: メンバー力量マトリクス
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        group_by: グループ化する軸（"職種", "役職"等）
    """
    st.markdown(f"### 📊 カテゴリ別 × {group_by}別 スキル分析")

    # コントロールパネル
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        hierarchy_level = st.selectbox(
            "カテゴリ階層",
            options=[1, 2, 3],
            format_func=lambda x: ["第一階層", "第二階層", "第三階層"][x-1],
            help="カテゴリの詳細度を選択します"
        )

    with col2:
        aggregation_method = st.selectbox(
            "集計方法",
            options=["mean", "median"],
            format_func=lambda x: "平均値" if x == "mean" else "中央値"
        )

    with col3:
        show_count = st.checkbox("人数も表示", value=False)

    # データ準備
    # 力量マスタにカテゴリ階層を追加
    competence_master_df = competence_master_df.copy()
    if "力量カテゴリー名" in competence_master_df.columns:
        competence_master_df["カテゴリ階層"] = competence_master_df["力量カテゴリー名"].apply(
            lambda x: extract_category_hierarchy(x, hierarchy_level)
        )
    else:
        st.warning("力量カテゴリー名がマスタに含まれていません")
        return

    # メンバー力量にカテゴリ階層を結合
    merged_df = member_competence_df.merge(
        competence_master_df[["力量コード", "カテゴリ階層"]],
        on="力量コード",
        how="left"
    )

    # メンバー情報を結合
    if group_by not in members_df.columns:
        st.warning(f"{group_by}情報がメンバーマスタに含まれていません")
        return

    merged_df = merged_df.merge(
        members_df[["メンバーコード", group_by]],
        on="メンバーコード",
        how="left"
    )

    # カテゴリ選択（複数選択可能）
    available_categories = sorted(merged_df["カテゴリ階層"].dropna().unique())

    selected_categories = st.multiselect(
        "表示するカテゴリを選択",
        options=available_categories,
        default=available_categories[:10] if len(available_categories) > 10 else available_categories,
        help="分析対象のカテゴリを選択してください（デフォルトは上位10件）"
    )

    if not selected_categories:
        st.info("カテゴリを選択してください")
        return

    # 選択されたカテゴリのデータのみ抽出
    filtered_df = merged_df[merged_df["カテゴリ階層"].isin(selected_categories)]

    # 保有量カラムの検出
    level_col = None
    for col in ["保有量", "力量レベル", "レベル"]:
        if col in filtered_df.columns:
            level_col = col
            break

    if level_col is None:
        st.warning("保有量またはレベル情報が見つかりません")
        return

    # 保有量を数値型に変換（エラー回避）
    filtered_df[level_col] = pd.to_numeric(filtered_df[level_col], errors='coerce')

    # NaNを除外
    filtered_df = filtered_df.dropna(subset=[level_col])

    if len(filtered_df) == 0:
        st.warning("有効な保有量データがありません")
        return

    # 集計
    if aggregation_method == "mean":
        pivot_df = filtered_df.groupby(["カテゴリ階層", group_by])[level_col].mean().unstack(fill_value=0)
    else:
        pivot_df = filtered_df.groupby(["カテゴリ階層", group_by])[level_col].median().unstack(fill_value=0)

    # カテゴリ階層でソート（階層的な順序を保持）
    pivot_df = pivot_df.sort_index()

    # インデックスを階層的な表示にフォーマット
    formatted_index = [format_category_for_display(cat) for cat in pivot_df.index]
    pivot_df_display = pivot_df.copy()
    pivot_df_display.index = formatted_index

    # ヒートマップ描画（職種を横軸、カテゴリを縦軸に配置）
    fig = px.imshow(
        pivot_df_display,
        labels=dict(x=group_by, y="カテゴリ", color="保有量" if aggregation_method == "mean" else "保有量"),
        aspect="auto",
        color_continuous_scale="Greens",  # 薄い緑→濃い緑のグラデーション
        text_auto=".2f"
    )

    fig.update_layout(
        height=max(400, len(pivot_df) * 50),
        title=f"カテゴリ別 × {group_by}別 スキル保有状況（{aggregation_method == 'mean' and '平均' or '中央値'}）",
        font=dict(size=11),
        xaxis=dict(
            side='top',  # x軸ラベルを上に配置
            tickangle=-45  # ラベルを斜めに表示
        ),
        yaxis=dict(
            tickfont=dict(family="Courier New, monospace")  # 等幅フォントでインデントを正しく表示
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 人数表示（オプション）
    if show_count:
        st.markdown("#### 👥 グループ別人数")
        count_df = filtered_df.groupby(["カテゴリ階層", group_by]).size().unstack(fill_value=0)
        st.dataframe(count_df, use_container_width=True)

    # データエクスポート
    csv = pivot_df.to_csv(index=True).encode('utf-8-sig')
    st.download_button(
        label=f"📥 {group_by}別データをダウンロード (CSV)",
        data=csv,
        file_name=f"category_{group_by}_analysis.csv",
        mime="text/csv"
    )


def render_job_role_skill_heatmap(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame
) -> None:
    """
    ②職種×役職別スキル集計ダッシュボード

    Args:
        member_competence_df: メンバー力量マトリクス
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
    """
    st.markdown("### 🎯 職種 × 役職別 スキル集計")

    # コントロールパネル
    col1, col2, col3 = st.columns(3)

    with col1:
        aggregation_method = st.selectbox(
            "集計方法",
            options=["mean", "median"],
            format_func=lambda x: "平均値" if x == "mean" else "中央値",
            key="job_role_agg"
        )

    # 職種と役職の選択
    if "職種" not in members_df.columns or "役職" not in members_df.columns:
        st.warning("職種または役職情報がメンバーマスタに含まれていません")
        return

    with col2:
        available_jobs = sorted(members_df["職種"].dropna().unique())
        selected_jobs = st.multiselect(
            "表示する職種",
            options=available_jobs,
            default=available_jobs,
            key="job_select"
        )

    with col3:
        available_roles = sorted(members_df["役職"].dropna().unique())
        selected_roles = st.multiselect(
            "表示する役職",
            options=available_roles,
            default=available_roles,
            key="role_select"
        )

    if not selected_jobs or not selected_roles:
        st.info("職種と役職を選択してください")
        return

    # データ準備
    filtered_members = members_df[
        (members_df["職種"].isin(selected_jobs)) &
        (members_df["役職"].isin(selected_roles))
    ]

    # メンバー力量にメンバー情報を結合
    merged_df = member_competence_df.merge(
        filtered_members[["メンバーコード", "職種", "役職"]],
        on="メンバーコード",
        how="inner"
    )

    # 保有量カラムの検出
    level_col = None
    for col in ["保有量", "力量レベル", "レベル"]:
        if col in merged_df.columns:
            level_col = col
            break

    if level_col is None:
        st.warning("保有量またはレベル情報が見つかりません")
        return

    # 保有量を数値型に変換（エラー回避）
    merged_df[level_col] = pd.to_numeric(merged_df[level_col], errors='coerce')

    # NaNを除外
    merged_df = merged_df.dropna(subset=[level_col])

    if len(merged_df) == 0:
        st.warning("有効な保有量データがありません")
        return

    # 職種×役職でクロス集計
    if aggregation_method == "mean":
        pivot_df = merged_df.groupby(["職種", "役職"])[level_col].mean().unstack(fill_value=0)
    else:
        pivot_df = merged_df.groupby(["職種", "役職"])[level_col].median().unstack(fill_value=0)

    # ヒートマップ描画（役職を横軸、職種を縦軸に配置）
    fig = px.imshow(
        pivot_df,
        labels=dict(x="役職", y="職種", color="保有量"),
        aspect="auto",
        color_continuous_scale="Greens",  # 薄い緑→濃い緑のグラデーション
        text_auto=".2f"
    )

    fig.update_layout(
        height=max(400, len(pivot_df) * 60),
        title=f"職種 × 役職別 平均スキル保有状況（{aggregation_method == 'mean' and '平均' or '中央値'}）",
        font=dict(size=12),
        xaxis=dict(
            side='top',  # x軸ラベルを上に配置
            tickangle=-45  # ラベルを斜めに表示
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # 統計サマリー
    st.markdown("#### 📈 統計サマリー")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("最高スキル保有", f"{pivot_df.max().max():.2f}")
    with col2:
        st.metric("最低スキル保有", f"{pivot_df.min().min():.2f}")
    with col3:
        st.metric("全体平均", f"{pivot_df.mean().mean():.2f}")

    # 人数マトリクス
    st.markdown("#### 👥 職種 × 役職別 人数")
    count_df = merged_df.groupby(["職種", "役職"]).size().unstack(fill_value=0)
    st.dataframe(count_df, use_container_width=True)


def render_skill_portfolio_analysis(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame
) -> None:
    """
    ③スキルポートフォリオ分析ダッシュボード

    組織のスキル多様性、集中度、バランスを分析
    """
    st.markdown("### 💼 スキルポートフォリオ分析")
    st.markdown("""
    **目的**: 組織のスキル保有のバランスを評価し、リスクを特定します
    - 🔴 **高リスク**: 特定スキルに依存（保有者が少ない）
    - 🟡 **中リスク**: スキル偏在あり
    - 🟢 **低リスク**: バランスの取れたポートフォリオ
    """)

    # スキル保有者数の分布分析
    skill_holders = member_competence_df.groupby("力量コード").size().reset_index(name="保有者数")

    # 力量マスタから必要なカラムを取得（存在チェック）
    master_cols = ["力量コード", "力量名"]
    if "力量カテゴリー名" in competence_master_df.columns:
        master_cols.append("力量カテゴリー名")

    skill_holders = skill_holders.merge(
        competence_master_df[master_cols],
        on="力量コード",
        how="left"
    )

    # リスク分類
    total_members = len(members_df)
    skill_holders["保有率"] = skill_holders["保有者数"] / total_members

    def classify_risk(holder_count):
        if holder_count == 1:
            return "🔴 高リスク（1名のみ）"
        elif holder_count <= 3:
            return "🟠 中高リスク（2-3名）"
        elif holder_count <= 5:
            return "🟡 中リスク（4-5名）"
        else:
            return "🟢 低リスク（6名以上）"

    skill_holders["リスクレベル"] = skill_holders["保有者数"].apply(classify_risk)

    # リスク分布
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 スキルリスク分布")
        risk_dist = skill_holders["リスクレベル"].value_counts().reset_index()
        risk_dist.columns = ["リスクレベル", "スキル数"]

        fig = px.pie(
            risk_dist,
            values="スキル数",
            names="リスクレベル",
            title="スキル保有リスク分布",
            color_discrete_sequence=["#d62728", "#ff7f0e", "#ffbb78", "#2ca02c"]  # 赤→オレンジ→黄→緑
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 保有者数分布")
        fig = px.histogram(
            skill_holders,
            x="保有者数",
            nbins=20,
            title="スキル保有者数のヒストグラム",
            labels={"保有者数": "保有者数", "count": "スキル数"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # 高リスクスキル一覧
    st.markdown("#### ⚠️ 高リスクスキル（保有者3名以下）")
    high_risk_skills = skill_holders[skill_holders["保有者数"] <= 3].sort_values("保有者数")

    if len(high_risk_skills) > 0:
        # 表示するカラムを動的に決定
        display_cols = ["力量名"]
        if "力量カテゴリー名" in high_risk_skills.columns:
            display_cols.append("力量カテゴリー名")
        display_cols.extend(["保有者数", "保有率", "リスクレベル"])

        st.dataframe(
            high_risk_skills[display_cols],
            use_container_width=True,
            height=300
        )

        st.warning(f"⚠️ {len(high_risk_skills)}件のスキルが高リスク状態です。優先的に育成計画を立案することを推奨します。")
    else:
        st.success("✅ 高リスクスキルはありません")

    # スキルカテゴリ別集中度分析
    if "力量カテゴリー名" in skill_holders.columns:
        st.markdown("#### 📂 カテゴリ別スキル集中度")

        category_summary = skill_holders.groupby("力量カテゴリー名").agg({
            "保有者数": ["mean", "min", "max", "std"]
        }).reset_index()
        category_summary.columns = ["カテゴリ", "平均保有者数", "最小保有者数", "最大保有者数", "標準偏差"]
        category_summary["変動係数 (CV)"] = category_summary["標準偏差"] / category_summary["平均保有者数"]
        category_summary = category_summary.sort_values("変動係数 (CV)", ascending=False)

        st.dataframe(category_summary, use_container_width=True)
        st.caption("💡 変動係数(CV)が高いカテゴリは、スキル間の保有者数のばらつきが大きく、リスクが高い可能性があります")


def render_talent_risk_dashboard(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame
) -> None:
    """
    ④人材リスク分析ダッシュボード

    キーパーソンリスク、スキル依存度を分析
    """
    st.markdown("### 🚨 人材リスク分析")
    st.markdown("""
    **分析目的**: 特定メンバーへのスキル集中リスクを特定し、組織の脆弱性を可視化
    """)

    # メンバー別スキル保有数
    member_skill_counts = member_competence_df.groupby("メンバーコード").size().reset_index(name="保有スキル数")

    # メンバー情報の結合（カラム存在チェック）
    member_cols = ["メンバーコード"]
    optional_cols = {"メンバー名": "メンバーコード", "職種": None, "役職": None}  # フォールバック値

    for col, fallback in optional_cols.items():
        if col in members_df.columns:
            member_cols.append(col)
        elif fallback:
            # メンバー名がない場合はメンバーコードで代用
            if col == "メンバー名":
                members_df = members_df.copy()
                members_df["メンバー名"] = members_df["メンバーコード"]
                member_cols.append("メンバー名")

    member_skill_counts = member_skill_counts.merge(
        members_df[member_cols],
        on="メンバーコード",
        how="left"
    )

    # メンバー名カラムが存在しない場合の対応
    if "メンバー名" not in member_skill_counts.columns:
        member_skill_counts["メンバー名"] = member_skill_counts["メンバーコード"]

    # 上位スキル保有者
    st.markdown("#### 🌟 トップスキル保有者（組織のキーパーソン）")
    top_members = member_skill_counts.nlargest(10, "保有スキル数")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            top_members,
            x="メンバー名",
            y="保有スキル数",
            color="職種",
            title="トップ10スキル保有者",
            text="保有スキル数"
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("平均スキル数", f"{member_skill_counts['保有スキル数'].mean():.1f}")
        st.metric("中央値", f"{member_skill_counts['保有スキル数'].median():.0f}")
        st.metric("最大値", f"{member_skill_counts['保有スキル数'].max()}")

        # パレート分析（上位20%が何%のスキルを保有しているか）
        top_20_pct_count = int(len(member_skill_counts) * 0.2)
        top_20_pct_skills = member_skill_counts.nlargest(top_20_pct_count, "保有スキル数")["保有スキル数"].sum()
        total_skills = member_skill_counts["保有スキル数"].sum()
        pareto_ratio = (top_20_pct_skills / total_skills) * 100

        st.metric(
            "パレート比率",
            f"{pareto_ratio:.1f}%",
            help="上位20%のメンバーが保有するスキルの割合"
        )

    # ユニークスキル分析（そのメンバーしか持っていないスキル）
    st.markdown("#### 🎯 ユニークスキル保有者（離職リスク高）")

    skill_holder_counts = member_competence_df.groupby("力量コード")["メンバーコード"].nunique().reset_index(name="保有者数")
    unique_skills = skill_holder_counts[skill_holder_counts["保有者数"] == 1]["力量コード"].tolist()

    if unique_skills:
        unique_skill_holders = member_competence_df[
            member_competence_df["力量コード"].isin(unique_skills)
        ].merge(
            members_df[member_cols],  # 既に構築したmember_colsを使用
            on="メンバーコード",
            how="left"
        ).merge(
            competence_master_df[["力量コード", "力量名"]],
            on="力量コード",
            how="left"
        )

        # メンバー名カラムが存在しない場合の対応
        if "メンバー名" not in unique_skill_holders.columns:
            unique_skill_holders["メンバー名"] = unique_skill_holders["メンバーコード"]

        # groupbyのカラムを動的に構築
        groupby_cols = ["メンバー名"]
        if "職種" in unique_skill_holders.columns:
            groupby_cols.append("職種")
        if "役職" in unique_skill_holders.columns:
            groupby_cols.append("役職")

        unique_summary = unique_skill_holders.groupby(groupby_cols).agg({
            "力量コード": "count"
        }).reset_index()

        # カラム名を設定
        new_cols = groupby_cols + ["ユニークスキル数"]
        unique_summary.columns = new_cols
        unique_summary = unique_summary.sort_values("ユニークスキル数", ascending=False)

        st.dataframe(unique_summary, use_container_width=True, height=300)

        st.error(f"⚠️ {len(unique_summary)}名のメンバーが組織で唯一のスキルを保有しています。これらのメンバーの離職は組織に重大な影響を与えます。")

        # 詳細表示
        with st.expander("ユニークスキル詳細を表示"):
            # 表示カラムを動的に構築
            detail_cols = ["メンバー名"]
            if "職種" in unique_skill_holders.columns:
                detail_cols.append("職種")
            if "力量名" in unique_skill_holders.columns:
                detail_cols.append("力量名")

            st.dataframe(
                unique_skill_holders[detail_cols],
                use_container_width=True
            )
    else:
        st.success("✅ 全てのスキルが複数名で共有されています")

    # スキル分布の偏り分析
    st.markdown("#### 📊 スキル分布の不均衡度")

    # ジニ係数の計算（スキル保有の不平等度）
    skill_counts_sorted = member_skill_counts["保有スキル数"].sort_values().values
    n = len(skill_counts_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * skill_counts_sorted)) / (n * np.sum(skill_counts_sorted)) - (n + 1) / n

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ジニ係数", f"{gini:.3f}", help="0に近いほど均等、1に近いほど不均等")
    with col2:
        skewness = stats.skew(member_skill_counts["保有スキル数"])
        st.metric("歪度", f"{skewness:.2f}", help="正の値は一部のメンバーにスキルが集中")
    with col3:
        cv = member_skill_counts["保有スキル数"].std() / member_skill_counts["保有スキル数"].mean()
        st.metric("変動係数", f"{cv:.2f}", help="スキル保有数のばらつき度")

    if gini > 0.4:
        st.warning("⚠️ スキルが一部のメンバーに集中しています。組織全体でのスキル共有・育成を推奨します。")
    elif gini < 0.2:
        st.success("✅ スキルが均等に分散しています。")
    else:
        st.info("ℹ️ スキル分布は標準的です。")


def render_benchmark_dashboard(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame
) -> None:
    """
    ⑤組織ベンチマーキング＆競合比較ダッシュボード

    業界標準や理想状態と比較
    """
    st.markdown("### 📊 組織ベンチマーキング")
    st.markdown("""
    **分析目的**: 組織のスキル成熟度を評価し、改善領域を特定
    """)

    # 基本統計
    total_members = len(members_df)
    total_skills_available = len(competence_master_df)
    total_skill_acquisitions = len(member_competence_df)
    avg_skills_per_member = total_skill_acquisitions / total_members if total_members > 0 else 0
    coverage_rate = (member_competence_df["力量コード"].nunique() / total_skills_available) * 100

    # 各指標を安全に計算
    try:
        diversity_index = calculate_diversity_index(member_competence_df)
    except Exception as e:
        diversity_index = 0.0
        st.warning(f"スキル多様性指数の計算中にエラーが発生しました: {e}")

    try:
        t_shaped_ratio = calculate_t_shaped_ratio(member_competence_df, competence_master_df)
    except Exception as e:
        t_shaped_ratio = 0.0
        st.warning(f"T字型人材比率の計算中にエラーが発生しました: {e}")

    # ベンチマークデータ（業界標準値 - 仮想データ）
    # 実際のプロダクションでは外部APIや設定ファイルから取得
    benchmark_data = {
        "現在の組織": {
            "平均スキル数/人": avg_skills_per_member,
            "スキルカバレッジ率": coverage_rate,
            "スキル多様性指数": diversity_index,
            "T字型人材比率": t_shaped_ratio
        },
        "業界平均": {
            "平均スキル数/人": 8.5,
            "スキルカバレッジ率": 65.0,
            "スキル多様性指数": 0.75,
            "T字型人材比率": 35.0
        },
        "トップ企業": {
            "平均スキル数/人": 12.0,
            "スキルカバレッジ率": 85.0,
            "スキル多様性指数": 0.85,
            "T字型人材比率": 50.0
        }
    }

    df_benchmark = pd.DataFrame(benchmark_data).T

    # レーダーチャート
    st.markdown("#### 🎯 総合スコア比較")

    categories = list(df_benchmark.columns)

    fig = go.Figure()

    for org_name in df_benchmark.index:
        fig.add_trace(go.Scatterpolar(
            r=df_benchmark.loc[org_name].values,
            theta=categories,
            fill='toself',
            name=org_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
        title="組織スキル成熟度ベンチマーク"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 詳細比較テーブル
    st.markdown("#### 📋 詳細比較")

    comparison_df = df_benchmark.copy()
    comparison_df["vs 業界平均"] = ((comparison_df.loc["現在の組織"] / comparison_df.loc["業界平均"] - 1) * 100).round(1)
    comparison_df["vs トップ企業"] = ((comparison_df.loc["現在の組織"] / comparison_df.loc["トップ企業"] - 1) * 100).round(1)

    st.dataframe(comparison_df.T, use_container_width=True)

    # 改善推奨アクション
    st.markdown("#### 💡 改善推奨アクション")

    actions = []

    if avg_skills_per_member < 8.5:
        actions.append("📚 **スキル育成プログラムの強化**: 平均スキル数が業界平均を下回っています")

    if coverage_rate < 65:
        actions.append("🎯 **スキルカバレッジの拡大**: 組織として保有すべきスキルの範囲を広げましょう")

    diversity = calculate_diversity_index(member_competence_df)
    if diversity < 0.75:
        actions.append("🌈 **スキル多様性の向上**: 特定スキルへの偏りを是正しましょう")

    t_shaped = calculate_t_shaped_ratio(member_competence_df, competence_master_df)
    if t_shaped < 35:
        actions.append("🔰 **T字型人材の育成**: 専門性と幅広い知識を持つ人材を増やしましょう")

    if actions:
        for action in actions:
            st.markdown(f"- {action}")
    else:
        st.success("✅ 全ての指標で業界平均以上を達成しています！")


def calculate_diversity_index(member_competence_df: pd.DataFrame) -> float:
    """
    スキル多様性指数を計算（Shannon Entropy）
    """
    skill_counts = member_competence_df["力量コード"].value_counts()
    proportions = skill_counts / skill_counts.sum()
    entropy = -np.sum(proportions * np.log(proportions + 1e-10))
    max_entropy = np.log(len(skill_counts))
    return (entropy / max_entropy) * 100 if max_entropy > 0 else 0


def calculate_t_shaped_ratio(member_competence_df: pd.DataFrame, competence_master_df: pd.DataFrame) -> float:
    """
    T字型人材比率を計算

    T字型 = 1つ以上の深い専門性（レベル4以上） + 幅広い知識（3カテゴリ以上）
    """
    # 保有量カラムの検出
    level_col = None
    for col in ["保有量", "力量レベル", "レベル"]:
        if col in member_competence_df.columns:
            level_col = col
            break

    if level_col is None:
        return 0.0

    # 力量カテゴリー名カラムが存在しない場合は計算不可
    if "力量カテゴリー名" not in competence_master_df.columns:
        return 0.0

    # カテゴリ情報を結合
    merged = member_competence_df.merge(
        competence_master_df[["力量コード", "力量カテゴリー名"]],
        on="力量コード",
        how="left"
    )

    # 保有量を数値型に変換
    merged[level_col] = pd.to_numeric(merged[level_col], errors='coerce')

    t_shaped_count = 0
    total_members = merged["メンバーコード"].nunique()

    for member_code in merged["メンバーコード"].unique():
        member_data = merged[merged["メンバーコード"] == member_code]

        # 深い専門性チェック（レベル4以上のスキルがあるか）
        has_deep_skill = False
        try:
            valid_levels = member_data[level_col].dropna()
            if len(valid_levels) > 0 and (valid_levels >= 4).any():
                has_deep_skill = True
        except:
            pass

        # 幅広い知識チェック（3カテゴリ以上）
        category_count = member_data["力量カテゴリー名"].nunique()
        has_broad_knowledge = category_count >= 3

        if has_deep_skill and has_broad_knowledge:
            t_shaped_count += 1

    return (t_shaped_count / total_members * 100) if total_members > 0 else 0.0
