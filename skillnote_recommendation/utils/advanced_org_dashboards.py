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
        階層に応じた表示形式
    """
    if pd.isna(category_path) or category_path == "未分類":
        return "未分類"

    parts = str(category_path).split(" > ")
    level = len(parts)

    if level == 1:
        # 第一階層: そのまま表示
        return parts[0]
    elif level == 2:
        # 第二階層: "第一階層─第二階層" の形式
        return f"{parts[0]}─{parts[1]}"
    else:
        # 第三階層: "第一階層─第二階層─第三階層" の形式、インデント追加
        indent = "    "
        return indent + "└" + parts[-1]


def format_hierarchical_index(category_paths: List[str], hierarchy_level: int) -> List[str]:
    """
    カテゴリパスのリストを階層的に表示するためにフォーマット（セル結合風）

    Args:
        category_paths: カテゴリパスのリスト
        hierarchy_level: 選択されている階層レベル (1, 2, 3)

    Returns:
        フォーマット済みのラベルリスト
    """
    if hierarchy_level == 1:
        # 第一階層: シンプルに表示
        return [cat.split(" > ")[0] if " > " in cat else cat for cat in category_paths]

    elif hierarchy_level == 2:
        # 第二階層: セル結合風に表示（罫線なし、スペースのみ）
        formatted = []
        prev_parent = None
        parent_group = []

        for cat_path in category_paths:
            parts = cat_path.split(" > ")
            if len(parts) >= 2:
                parent = parts[0]
                child = parts[1]

                if parent != prev_parent:
                    parent_group.append((cat_path, parent, child, True))  # グループの最初
                    prev_parent = parent
                else:
                    parent_group.append((cat_path, parent, child, False))  # グループの続き
            else:
                parent_group.append((cat_path, cat_path, "", True))

        # 各グループ内で最初・中間・最後を判定
        i = 0
        while i < len(parent_group):
            cat_path, parent, child, is_first = parent_group[i]

            # 同じ親を持つ要素の数をカウント
            group_size = 1
            j = i + 1
            while j < len(parent_group) and parent_group[j][1] == parent:
                group_size += 1
                j += 1

            # グループ内の位置に応じてフォーマット（罫線なし）
            for k in range(group_size):
                _, _, child, _ = parent_group[i + k]
                if k == 0:
                    # グループの最初: 親名を表示
                    formatted.append(f"{parent}　{child}")
                else:
                    # グループの2行目以降: 親名と同じ長さの全角スペースでインデント
                    formatted.append(f"{'　' * len(parent)}　{child}")

            i += group_size

        return formatted

    else:  # hierarchy_level == 3
        # 第三階層: 第一階層 > 第二階層 > 第三階層 を表示
        # 第二階層ごとにグループ化して視覚的に分かりやすく表示
        formatted = []

        # まず、カテゴリを解析してグループ化
        category_groups = {}  # key: (parent, child), value: list of grandchildren

        for cat_path in category_paths:
            parts = cat_path.split(" > ")
            if len(parts) >= 3:
                parent = parts[0]
                child = parts[1]
                grandchild = parts[2]
                key = (parent, child)
                if key not in category_groups:
                    category_groups[key] = []
                category_groups[key].append(grandchild)
            elif len(parts) == 2:
                parent = parts[0]
                child = parts[1]
                key = (parent, child)
                if key not in category_groups:
                    category_groups[key] = []
            else:
                # 第一階層のみの場合
                key = (cat_path, "")
                if key not in category_groups:
                    category_groups[key] = []

        # ソートされた順序でフォーマット
        for parent, child in sorted(category_groups.keys()):
            grandchildren = category_groups[(parent, child)]
            parent_child_label = f"{parent}　{child}" if child else parent

            if grandchildren:
                # 第三階層がある場合
                for idx, grandchild in enumerate(grandchildren):
                    if idx == 0:
                        # 最初の行: 親-子-孫を全て表示
                        formatted.append(f"{parent_child_label}　{grandchild}")
                    else:
                        # 2行目以降: 親-子と同じ長さの全角スペースでインデント
                        formatted.append(f"{'　' * len(parent_child_label)}　{grandchild}")
            else:
                # 第三階層がない（第二階層まで）場合
                formatted.append(parent_child_label)

        return formatted


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

    # インデックスを階層的な表示にフォーマット（セル結合風）
    formatted_index = format_hierarchical_index(list(pivot_df.index), hierarchy_level)
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

    # リスクレベルの順序を定義（高リスク→低リスクの順）
    risk_order = [
        "🔴 高リスク（1名のみ）",
        "🟠 中高リスク（2-3名）",
        "🟡 中リスク（4-5名）",
        "🟢 低リスク（6名以上）"
    ]

    # リスクレベルをカテゴリカル型に変換（順序付き）
    skill_holders["リスクレベル"] = pd.Categorical(
        skill_holders["リスクレベル"],
        categories=risk_order,
        ordered=True
    )

    # リスク分布
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 スキルリスク分布")
        risk_dist = skill_holders["リスクレベル"].value_counts().reset_index()
        risk_dist.columns = ["リスクレベル", "スキル数"]

        # リスクレベルの順序に従ってソート
        risk_dist["リスクレベル"] = pd.Categorical(
            risk_dist["リスクレベル"],
            categories=risk_order,
            ordered=True
        )
        risk_dist = risk_dist.sort_values("リスクレベル")

        # 色のマッピング（順序に対応）
        color_map = {
            "🔴 高リスク（1名のみ）": "#d62728",      # 赤
            "🟠 中高リスク（2-3名）": "#ff7f0e",      # オレンジ
            "🟡 中リスク（4-5名）": "#ffbb78",        # 黄
            "🟢 低リスク（6名以上）": "#2ca02c"       # 緑
        }

        fig = px.pie(
            risk_dist,
            values="スキル数",
            names="リスクレベル",
            title="スキル保有リスク分布",
            color="リスクレベル",
            color_discrete_map=color_map
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

    # 説明を改善
    st.info("""
    **📌 この分析の目的**
    特定メンバーへのスキル集中リスクを特定し、組織の脆弱性を可視化します。
    - キーパーソンの識別（スキル保有数が多いメンバー）
    - ユニークスキル保有者の特定（そのメンバーしか持っていないスキル）
    - スキル分布の偏り分析
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

    # パレート分析を先に計算
    top_20_pct_count = max(1, int(len(member_skill_counts) * 0.2))
    top_20_pct_skills = member_skill_counts.nlargest(top_20_pct_count, "保有スキル数")["保有スキル数"].sum()
    total_skills = member_skill_counts["保有スキル数"].sum()
    pareto_ratio = (top_20_pct_skills / total_skills) * 100 if total_skills > 0 else 0

    # サマリーメトリクスを上部に表示
    st.markdown("---")
    st.markdown("#### 📊 組織全体のスキル保有状況")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="平均スキル数/人",
            value=f"{member_skill_counts['保有スキル数'].mean():.1f}",
            help="1人あたりの平均スキル保有数"
        )

    with metric_col2:
        st.metric(
            label="中央値",
            value=f"{member_skill_counts['保有スキル数'].median():.0f}",
            help="スキル保有数の中央値"
        )

    with metric_col3:
        st.metric(
            label="最大スキル数",
            value=f"{member_skill_counts['保有スキル数'].max()}",
            help="最もスキルを多く保有しているメンバーのスキル数"
        )

    with metric_col4:
        alert_icon = "🔴" if pareto_ratio > 50 else "🟡" if pareto_ratio > 40 else "🟢"
        st.metric(
            label="パレート比率",
            value=f"{alert_icon} {pareto_ratio:.1f}%",
            help="上位20%のメンバーが保有するスキルの割合（高いほど集中リスクあり）"
        )

    # 上位スキル保有者
    st.markdown("---")
    st.markdown("#### 🌟 トップスキル保有者（キーパーソン）")

    top_members = member_skill_counts.nlargest(10, "保有スキル数")

    # グラフを改善
    fig = px.bar(
        top_members,
        y="メンバー名",  # 横棒グラフに変更
        x="保有スキル数",
        color="保有スキル数",
        color_continuous_scale="Blues",
        text="保有スキル数",
        orientation='h'  # 横向き
    )

    fig.update_traces(
        texttemplate='%{text}件',
        textposition='outside',
        textfont_size=12
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="保有スキル数",
        yaxis_title="",
        yaxis={'categoryorder':'total ascending'},  # 値の昇順でソート
        font=dict(size=11),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    if pareto_ratio > 50:
        st.warning(f"⚠️ 上位20%のメンバーが全体の{pareto_ratio:.1f}%のスキルを保有しています。特定メンバーへの依存度が高い状態です。")
    elif pareto_ratio > 40:
        st.info(f"💡 上位20%のメンバーが全体の{pareto_ratio:.1f}%のスキルを保有しています。やや集中傾向があります。")
    else:
        st.success(f"✅ スキルが比較的分散されています（上位20%で{pareto_ratio:.1f}%）。")

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

    # ベンチマークデータに関する注意書き
    st.info("""
    ℹ️ **ベンチマークデータについて**

    「業界平均」と「トップ企業」の数値は**参考値**として表示しています。
    実際の業界データや自社の目標値に置き換えることで、より正確な分析が可能になります。

    **現在の参考値：**
    - 業界平均: 平均スキル数 8.5件/人、カバレッジ率 65%
    - トップ企業: 平均スキル数 12.0件/人、カバレッジ率 85%
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

    # ベンチマークデータ（参考値 - サンプルデータ）
    # NOTE: 実際の運用では、以下の方法でカスタマイズ可能：
    # 1. 設定ファイル（YAML/JSON）から読み込み
    # 2. 外部ベンチマークAPIから取得
    # 3. UI上で編集可能なサイドバー入力欄を追加
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

    # マージ後に力量カテゴリー名が存在しない場合は計算不可
    if "力量カテゴリー名" not in merged.columns:
        return 0.0

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
        try:
            category_count = member_data["力量カテゴリー名"].dropna().nunique()
            has_broad_knowledge = category_count >= 3
        except:
            has_broad_knowledge = False

        if has_deep_skill and has_broad_knowledge:
            t_shaped_count += 1

    return (t_shaped_count / total_members * 100) if total_members > 0 else 0.0


def render_enhanced_skill_gap_analysis(
    gap_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    percentile_used: float = 0.2
) -> None:
    """
    データサイエンティスト兼人事スペシャリスト視点での高度なスキルギャップ分析

    Args:
        gap_df: ギャップDataFrame
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタデータ
        members_df: メンバーマスタ
        percentile_used: 使用したパーセンタイル
    """

    st.markdown("### 🎯 高度なスキルギャップ分析")

    # 分析概要説明
    st.info("""
    📌 **データサイエンス × HR戦略の統合分析**

    この分析では、単なるギャップの特定にとどまらず、以下の高度な視点で組織のスキル開発戦略を支援します：
    - **多次元スキル優先度分析**: ビジネスインパクト、習得難易度、緊急性を総合評価
    - **スキル開発ROI推定**: 投資対効果を可視化し、予算配分を最適化
    - **パターン認識**: 機械学習的アプローチでスキルギャップのクラスター分析
    - **予測モデリング**: スキル習得タイムラインと組織成熟度の将来予測
    """)

    st.markdown("---")

    # ============================================
    # 1. エグゼクティブサマリー（KPIダッシュボード）
    # ============================================
    st.markdown("#### 📊 エグゼクティブサマリー")

    total_gaps = len(gap_df)
    critical_gaps = len(gap_df[gap_df["保有率ギャップ率"] >= 0.5])
    medium_gaps = len(gap_df[(gap_df["保有率ギャップ率"] >= 0.3) & (gap_df["保有率ギャップ率"] < 0.5)])
    avg_gap_rate = gap_df["保有率ギャップ率"].mean()
    total_training_need = gap_df["保有率ギャップ"].sum() * len(members_df)

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric(
            label="総スキルギャップ数",
            value=f"{total_gaps}件",
            help="目標と現状の差があるスキルの総数"
        )

    with metric_col2:
        st.metric(
            label="🔴 重大ギャップ",
            value=f"{critical_gaps}件",
            delta=f"{critical_gaps/total_gaps*100:.1f}%" if total_gaps > 0 else "0%",
            delta_color="inverse",
            help="ギャップ率50%以上の緊急対応が必要なスキル"
        )

    with metric_col3:
        st.metric(
            label="🟡 中程度ギャップ",
            value=f"{medium_gaps}件",
            help="ギャップ率30-50%の計画的対応が必要なスキル"
        )

    with metric_col4:
        st.metric(
            label="平均ギャップ率",
            value=f"{avg_gap_rate*100:.1f}%",
            delta=f"{(avg_gap_rate - 0.3)*100:.1f}%" if avg_gap_rate > 0 else "0%",
            delta_color="inverse",
            help="全スキルの平均ギャップ率（30%未満が健全）"
        )

    with metric_col5:
        st.metric(
            label="推定育成人数",
            value=f"{int(total_training_need):,}人",
            help="ギャップを埋めるために必要な延べ育成人数"
        )

    st.markdown("---")

    # ============================================
    # 2. 多次元スキル優先度マトリクス
    # ============================================
    st.markdown("#### 🎯 多次元スキル優先度分析（優先度マトリクス）")

    st.markdown("""
    **分析手法**: 各スキルを3つの軸で評価し、投資優先度を科学的に判定
    - **X軸（ビジネスインパクト）**: 目標保有率が高いほど、組織戦略上重要
    - **Y軸（緊急性）**: ギャップ率が大きいほど、即座の対応が必要
    - **バブルサイズ（習得難易度）**: レベルギャップが大きいほど、育成に時間とコストがかかる
    """)

    # 優先度スコア計算
    gap_analysis_df = gap_df.copy()

    # ビジネスインパクト: 目標保有率（0-100に正規化）
    gap_analysis_df["ビジネスインパクト"] = gap_analysis_df["目標保有率"] * 100

    # 緊急性: ギャップ率（0-100に正規化）
    gap_analysis_df["緊急性"] = gap_analysis_df["保有率ギャップ率"] * 100

    # 習得難易度: レベルギャップ（絶対値を使用、0-5スケール）
    gap_analysis_df["習得難易度"] = gap_analysis_df["レベルギャップ"].abs()

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
            "習得難易度": ":.2f",
            "優先度スコア": ":.1f",
            "現在保有率": ":.1%",
            "目標保有率": ":.1%"
        },
        title="スキル投資優先度マトリクス（バブルチャート）",
        color_discrete_map={
            "🔴 最優先（Strategic Focus）": "#d62728",
            "🟠 高優先度（High Priority）": "#ff7f0e",
            "🟡 中優先度（Medium Priority）": "#ffbb78",
            "🟢 低優先度（Low Priority）": "#2ca02c"
        }
    )

    fig.update_layout(
        height=600,
        xaxis_title="ビジネスインパクト（目標保有率）",
        yaxis_title="緊急性（ギャップ率）",
        showlegend=True
    )

    # 右上の象限を強調（高インパクト×高緊急性）
    fig.add_shape(
        type="rect",
        x0=60, y0=60, x1=100, y1=100,
        line=dict(color="red", width=2, dash="dash"),
        fillcolor="rgba(255,0,0,0.1)"
    )

    fig.add_annotation(
        x=80, y=95,
        text="<b>戦略的最優先エリア</b>",
        showarrow=False,
        font=dict(size=12, color="red")
    )

    st.plotly_chart(fig, use_container_width=True)

    # 優先度カテゴリ別サマリー
    priority_summary = gap_analysis_df["優先度カテゴリ"].value_counts().reset_index()
    priority_summary.columns = ["優先度カテゴリ", "スキル数"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**優先度分布**")
        st.dataframe(priority_summary, use_container_width=True, hide_index=True)

    with col2:
        # 円グラフ
        fig_pie = px.pie(
            priority_summary,
            values="スキル数",
            names="優先度カテゴリ",
            title="優先度カテゴリ別分布",
            color="優先度カテゴリ",
            color_discrete_map={
                "🔴 最優先（Strategic Focus）": "#d62728",
                "🟠 高優先度（High Priority）": "#ff7f0e",
                "🟡 中優先度（Medium Priority）": "#ffbb78",
                "🟢 低優先度（Low Priority）": "#2ca02c"
            }
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ============================================
    # 3. スキル開発ROI推定
    # ============================================
    st.markdown("#### 💰 スキル開発ROI推定（投資対効果分析）")

    st.markdown("""
    **分析目的**: 限られた予算と時間をどのスキル開発に投資すべきかを定量的に判断

    **前提条件**（カスタマイズ可能）:
    - 1スキル習得の平均コスト: 研修費 + 時間コスト
    - スキルレベルによる習得期間の違い
    - ビジネスインパクトによる価値の重み付け
    """)

    # ROI計算パラメータ（UIで調整可能）
    col1, col2, col3 = st.columns(3)

    with col1:
        training_cost_per_skill = st.number_input(
            "1スキル習得コスト（万円）",
            min_value=1,
            max_value=100,
            value=20,
            step=5,
            help="研修費、教材費、時間コストを含む"
        )

    with col2:
        months_per_level = st.number_input(
            "レベル1習得に必要な月数",
            min_value=1,
            max_value=12,
            value=3,
            step=1,
            help="平均的なスキル習得期間"
        )

    with col3:
        business_value_multiplier = st.number_input(
            "ビジネス価値係数",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="スキル習得による組織への価値貢献度"
        )

    # ROI計算
    roi_df = gap_analysis_df.copy()

    # 必要な育成人数
    roi_df["育成必要人数"] = (roi_df["保有率ギャップ"] * len(members_df)).round(0).astype(int)

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

    # ROI上位10スキルを表示
    roi_top = roi_df.nlargest(10, "ROI率")[[
        "力量名", "育成必要人数", "総投資コスト", "推定ビジネス価値",
        "ROI率", "推定習得期間", "優先度カテゴリ"
    ]].copy()

    st.markdown("##### 🏆 ROI上位10スキル（最も投資効果が高いスキル）")

    # スタイリング
    def highlight_roi(row):
        colors = [''] * len(row)
        roi_idx = row.index.get_loc("ROI率")

        if row["ROI率"] >= 200:
            colors[roi_idx] = 'background-color: #d4edda; font-weight: bold'
        elif row["ROI率"] >= 100:
            colors[roi_idx] = 'background-color: #fff3cd'

        return colors

    styled_roi = roi_top.style.apply(highlight_roi, axis=1).format({
        "総投資コスト": "{:,.0f}万円",
        "推定ビジネス価値": "{:,.0f}万円",
        "ROI率": "{:.1f}%",
        "推定習得期間": "{:.1f}ヶ月"
    })

    st.dataframe(styled_roi, use_container_width=True, hide_index=True)

    st.caption("🟢 緑背景: 高ROI（200%以上） | 🟡 黄背景: 中ROI（100%以上）")

    # ROI可視化
    fig_roi = px.bar(
        roi_top,
        x="ROI率",
        y="力量名",
        color="優先度カテゴリ",
        orientation='h',
        title="ROI上位スキルランキング",
        labels={"ROI率": "ROI率 (%)", "力量名": ""},
        color_discrete_map={
            "🔴 最優先（Strategic Focus）": "#d62728",
            "🟠 高優先度（High Priority）": "#ff7f0e",
            "🟡 中優先度（Medium Priority）": "#ffbb78",
            "🟢 低優先度（Low Priority）": "#2ca02c"
        }
    )

    fig_roi.update_layout(
        height=400,
        yaxis={'categoryorder':'total ascending'}
    )

    st.plotly_chart(fig_roi, use_container_width=True)

    # 投資シミュレーション
    st.markdown("##### 💡 投資シミュレーション")

    total_investment = roi_df["総投資コスト"].sum()
    total_value = roi_df["推定ビジネス価値"].sum()
    overall_roi = ((total_value - total_investment) / total_investment * 100) if total_investment > 0 else 0

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        st.metric(
            "全ギャップ解消の総投資額",
            f"{total_investment:,.0f}万円",
            help="全スキルギャップを埋めるために必要な総コスト"
        )

    with sim_col2:
        st.metric(
            "推定総ビジネス価値",
            f"{total_value:,.0f}万円",
            help="全スキルギャップを解消した場合の組織価値向上"
        )

    with sim_col3:
        st.metric(
            "全体ROI",
            f"{overall_roi:.1f}%",
            delta=f"{overall_roi - 100:.1f}%" if overall_roi > 0 else "0%",
            help="全体的な投資対効果"
        )

    st.markdown("---")

    # ============================================
    # 4. スキルギャップのパターン認識（クラスター分析）
    # ============================================
    st.markdown("#### 🔬 スキルギャップのパターン認識")

    st.markdown("""
    **分析手法**: K-meansクラスタリングにより、類似したギャップパターンを持つスキルをグループ化

    これにより、個別スキルではなく「スキルグループ」単位での戦略的育成プログラムを設計できます。
    """)

    # クラスタリング用データ準備
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

    # クラスターラベル付け
    cluster_labels = {
        0: "🎯 戦略的重要スキル群",
        1: "⚡ 緊急対応スキル群",
        2: "📚 基礎育成スキル群",
        3: "🔄 長期育成スキル群"
    }

    # クラスターの特性を分析して適切にラベル付け
    cluster_characteristics = []
    for cluster_id in range(n_clusters):
        cluster_data = gap_analysis_df[gap_analysis_df["クラスター"] == cluster_id]
        avg_impact = cluster_data["ビジネスインパクト"].mean()
        avg_urgency = cluster_data["緊急性"].mean()
        avg_difficulty = cluster_data["習得難易度"].mean()

        # 特性に基づいてラベルを決定
        if avg_impact > 60 and avg_urgency > 60:
            label = "🎯 戦略的重要スキル群"
        elif avg_urgency > 60:
            label = "⚡ 緊急対応スキル群"
        elif avg_difficulty < 2:
            label = "📚 基礎育成スキル群"
        else:
            label = "🔄 長期育成スキル群"

        cluster_characteristics.append({
            "クラスター": label,
            "スキル数": len(cluster_data),
            "平均ビジネスインパクト": f"{avg_impact:.1f}",
            "平均緊急性": f"{avg_urgency:.1f}",
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
                st.metric("推定投資額", f"{skill['総投資コスト']:.0f}万円")
                st.metric("ROI", f"{skill['ROI率']:.1f}%")

            st.markdown("---")

            # 具体的アクション
            st.markdown("##### 📌 推奨アクション")

            actions = _generate_action_recommendations(skill, members_df)

            for action in actions:
                st.markdown(f"- {action}")

            st.markdown("---")

            # タイムライン
            st.markdown("##### ⏱️ 実施タイムライン")

            timeline = _generate_timeline(skill)

            for phase, desc in timeline.items():
                st.markdown(f"**{phase}**: {desc}")

    st.markdown("---")

    # ============================================
    # 6. スキルポートフォリオ最適化提案
    # ============================================
    st.markdown("#### 🎨 スキルポートフォリオ最適化提案")

    st.markdown("""
    **組織全体の視点**: 個別スキルではなく、組織のスキルポートフォリオ全体を最適化
    """)

    # 現在のポートフォリオ状態
    current_strategic = len(gap_analysis_df[gap_analysis_df["優先度カテゴリ"] == "🔴 最優先（Strategic Focus）"])
    current_high = len(gap_analysis_df[gap_analysis_df["優先度カテゴリ"] == "🟠 高優先度（High Priority）"])
    current_medium = len(gap_analysis_df[gap_analysis_df["優先度カテゴリ"] == "🟡 中優先度（Medium Priority）"])
    current_low = len(gap_analysis_df[gap_analysis_df["優先度カテゴリ"] == "🟢 低優先度（Low Priority）"])

    # 理想的な配分（ベンチマーク）
    ideal_strategic = int(total_gaps * 0.2)
    ideal_high = int(total_gaps * 0.3)
    ideal_medium = int(total_gaps * 0.3)
    ideal_low = int(total_gaps * 0.2)

    portfolio_comparison = pd.DataFrame({
        "優先度カテゴリ": [
            "🔴 最優先",
            "🟠 高優先度",
            "🟡 中優先度",
            "🟢 低優先度"
        ],
        "現状": [current_strategic, current_high, current_medium, current_low],
        "理想": [ideal_strategic, ideal_high, ideal_medium, ideal_low],
        "差分": [
            current_strategic - ideal_strategic,
            current_high - ideal_high,
            current_medium - ideal_medium,
            current_low - ideal_low
        ]
    })

    # 比較グラフ
    fig_portfolio = go.Figure()

    fig_portfolio.add_trace(go.Bar(
        name="現状",
        x=portfolio_comparison["優先度カテゴリ"],
        y=portfolio_comparison["現状"],
        marker_color='lightblue'
    ))

    fig_portfolio.add_trace(go.Bar(
        name="理想（ベンチマーク）",
        x=portfolio_comparison["優先度カテゴリ"],
        y=portfolio_comparison["理想"],
        marker_color='lightgreen'
    ))

    fig_portfolio.update_layout(
        title="スキルポートフォリオ: 現状 vs 理想配分",
        xaxis_title="",
        yaxis_title="スキル数",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig_portfolio, use_container_width=True)

    # 改善提案
    st.markdown("##### 💡 ポートフォリオ最適化の提案")

    if current_strategic > ideal_strategic:
        st.warning(
            f"⚠️ **最優先スキルが多すぎます** ({current_strategic - ideal_strategic}件超過)\n\n"
            "一度に多くのスキルを最優先にすると、リソースが分散します。"
            "最も重要な20%に絞り込み、段階的に取り組むことを推奨します。"
        )
    elif current_strategic < ideal_strategic:
        st.info(
            f"ℹ️ **最優先スキルの明確化が必要** ({ideal_strategic - current_strategic}件不足)\n\n"
            "組織戦略上、最優先で取り組むべきスキルを明確に定義することで、投資効果が向上します。"
        )
    else:
        st.success("✅ 最優先スキルの数は適切です")

    # 総合推奨事項
    st.markdown("##### 🌟 総合推奨事項")

    st.markdown(f"""
    **データに基づく戦略的提言**:

    1. **即座に着手すべきスキル**:
       - {top_priority_skills.iloc[0]['力量名']}を筆頭に、最優先スキル{current_strategic}件に集中投資
       - 推定投資額: {top_priority_skills.head(5)['総投資コスト'].sum():,.0f}万円
       - 期待ROI: {top_priority_skills.head(5)['ROI率'].mean():.1f}%

    2. **6ヶ月以内の目標**:
       - 最優先スキルの平均保有率を現状から20%改善
       - クリティカルギャップ（ギャップ率50%以上）を{critical_gaps}件から半減

    3. **1年後の目標**:
       - 平均スキルギャップ率を{avg_gap_rate*100:.1f}%から20%未満に削減
       - 上位{int(percentile_used*100)}%メンバーのスキルセットを組織全体の標準に

    4. **投資配分の推奨**:
       - 最優先スキル: 予算の50%
       - 高優先度スキル: 予算の30%
       - 中優先度スキル: 予算の15%
       - 低優先度スキル: 予算の5%（機会学習）
    """)

    # データエクスポート
    st.markdown("---")
    st.markdown("### 💾 分析結果のエクスポート")

    export_df = roi_df[[
        "力量名", "現在保有率", "目標保有率", "保有率ギャップ率",
        "ビジネスインパクト", "緊急性", "習得難易度", "優先度スコア", "優先度カテゴリ",
        "育成必要人数", "総投資コスト", "ROI率", "推定習得期間", "クラスターラベル"
    ]].copy()

    csv = export_df.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 詳細分析結果をCSVでダウンロード",
        data=csv,
        file_name="enhanced_skill_gap_analysis.csv",
        mime="text/csv"
    )


def _get_cluster_recommendation(impact: float, urgency: float, difficulty: float) -> str:
    """クラスターごとの推奨アプローチを生成"""
    if impact > 60 and urgency > 60:
        return "集中投資・即時実行プログラム"
    elif urgency > 60:
        return "短期集中ブートキャンプ形式"
    elif difficulty < 2:
        return "eラーニング・自己学習支援"
    else:
        return "中長期OJT・メンター制度"


def _generate_action_recommendations(skill: pd.Series, members_df: pd.DataFrame) -> List[str]:
    """スキルごとの具体的アクション推奨を生成"""
    actions = []

    gap_rate = skill["保有率ギャップ率"]
    training_need = int(skill["育成必要人数"])

    # 育成方法の推奨
    if skill["習得難易度"] < 2:
        actions.append(f"📚 **育成方法**: eラーニングプラットフォームで自己学習プログラムを提供（コスト効率◎）")
    elif skill["習得難易度"] < 3.5:
        actions.append(f"🎓 **育成方法**: 社内研修プログラムを実施（期間: 1-3ヶ月）")
    else:
        actions.append(f"👨‍🏫 **育成方法**: 外部専門研修 + 社内メンター制度の併用（期間: 3-6ヶ月）")

    # 人数規模に応じた実施方法
    if training_need <= 5:
        actions.append(f"👥 **実施規模**: 少人数（{training_need}名）- 個別カスタマイズ型育成")
    elif training_need <= 15:
        actions.append(f"👥 **実施規模**: 中規模（{training_need}名）- グループ研修形式")
    else:
        actions.append(f"👥 **実施規模**: 大規模（{training_need}名）- 複数回に分けたローリング研修")

    # 採用も検討すべきか
    if gap_rate > 0.7:
        actions.append(f"💼 **追加施策**: ギャップが大きいため、外部採用も並行検討を推奨")

    # 社内エキスパート活用
    if skill["現在保有率"] > 0.1:
        actions.append(f"🌟 **社内リソース活用**: 既存保有者をメンター/トレーナーとして活用")

    return actions


def _generate_timeline(skill: pd.Series) -> Dict[str, str]:
    """スキル習得のタイムライン生成"""
    duration = skill["推定習得期間"]

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
