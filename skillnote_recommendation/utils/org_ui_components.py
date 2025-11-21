"""
組織向けUIコンポーネント

組織レベルの分析結果を可視化するためのStreamlitコンポーネント
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, List
import io


def render_skill_heatmap(
    skill_matrix_df: pd.DataFrame,
    title: str = "組織スキルマップ",
    max_skills: int = 50
) -> None:
    """
    スキル保有状況のヒートマップを描画
    
    Args:
        skill_matrix_df: メンバー × スキルのマトリクス（0/1またはレベル値）
        title: グラフタイトル
        max_skills: 表示する最大スキル数（多すぎると重くなるため）
    """
    # スキル数が多い場合は上位のみ表示
    if len(skill_matrix_df.columns) > max_skills:
        # 保有者数が多いスキルを優先
        skill_counts = skill_matrix_df.sum(axis=0).sort_values(ascending=False)
        top_skills = skill_counts.head(max_skills).index.tolist()
        display_df = skill_matrix_df[top_skills]
        st.warning(f"⚠️ スキル数が多いため、保有者数上位{max_skills}件を表示しています")
    else:
        display_df = skill_matrix_df
    
    # Plotlyヒートマップ
    fig = px.imshow(
        display_df.T,  # 転置（スキルを縦軸、メンバーを横軸）
        labels=dict(x="メンバー", y="スキル", color="レベル"),
        aspect="auto",
        color_continuous_scale="Blues",
        title=title
    )
    
    fig.update_layout(
        height=max(400, len(display_df.columns) * 15),
        xaxis_showticklabels=False,  # メンバー名は非表示（多すぎるため）
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_skill_distribution_chart(
    distribution_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "スキル分布"
) -> None:
    """
    スキル分布の棒グラフを描画
    
    Args:
        distribution_df: 分布データ
        x_col: X軸のカラム名
        y_col: Y軸のカラム名
        title: グラフタイトル
    """
    fig = px.bar(
        distribution_df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        color=y_col,
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_gap_ranking_table(
    gap_df: pd.DataFrame,
    top_n: int = 10
) -> None:
    """
    ギャップランキングテーブルを表示
    
    Args:
        gap_df: ギャップDataFrame
        top_n: 表示する上位件数
    """
    # 上位N件を抽出
    display_df = gap_df.head(top_n).copy()
    
    # パーセント表示に変換
    display_df["現在保有率"] = (display_df["現在保有率"] * 100).round(1).astype(str) + "%"
    display_df["目標保有率"] = (display_df["目標保有率"] * 100).round(1).astype(str) + "%"
    display_df["保有率ギャップ"] = (display_df["保有率ギャップ"] * 100).round(1).astype(str) + "%"
    display_df["保有率ギャップ率"] = (display_df["保有率ギャップ率"] * 100).round(1).astype(str) + "%"
    
    # クリティカルスキル（ギャップ率50%以上）をハイライト
    def highlight_critical(row):
        gap_rate = float(row["保有率ギャップ率"].replace("%", ""))
        if gap_rate >= 50:
            return ['background-color: #ffe6e6'] * len(row)
        elif gap_rate >= 30:
            return ['background-color: #fff3e6'] * len(row)
        else:
            return [''] * len(row)
    
    # 表示カラムを選択
    display_columns = [
        "力量名", "現在保有率", "目標保有率", "保有率ギャップ", "保有率ギャップ率"
    ]
    
    styled_df = display_df[display_columns].style.apply(highlight_critical, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # 凡例
    st.caption("🔴 赤背景: クリティカルスキル（ギャップ率50%以上） | 🟡 黄背景: 重要スキル（ギャップ率30%以上）")


def render_skill_matrix_table(
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    filters: Dict = {}
) -> pd.DataFrame:
    """
    フィルタリング可能なスキルマトリクステーブルを表示
    
    Args:
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        filters: フィルタ条件の辞書
        
    Returns:
        フィルタリング後のDataFrame
    """
    import re
    
    # カラム名クリーニング関数
    def clean_col_name(name):
        """カラム名から ###[...]### を削除"""
        return re.sub(r'\s*###\[.*?\]###', '', str(name)).strip()
    
    # メンバーマスタのカラム名をクリーニング
    members_df_clean = members_df.copy()
    members_df_clean.columns = [clean_col_name(col) for col in members_df_clean.columns]
    
    # 力量マスタのカラム名もクリーニング
    competence_master_clean = competence_master_df.copy()
    competence_master_clean.columns = [clean_col_name(col) for col in competence_master_clean.columns]
    
    # メンバー習得力量データのカラム名もクリーニング
    member_competence_clean = member_competence_df.copy()
    member_competence_clean.columns = [clean_col_name(col) for col in member_competence_clean.columns]
    
    # マージ前に、member_competence_cleanから力量名と力量タイプを削除（力量マスタから取得するため）
    cols_to_remove = []
    if "力量名" in member_competence_clean.columns:
        cols_to_remove.append("力量名")
    if "力量タイプ" in member_competence_clean.columns:
        cols_to_remove.append("力量タイプ")
    if cols_to_remove:
        member_competence_clean = member_competence_clean.drop(columns=cols_to_remove, errors='ignore')
    
    # メンバーマスタとマージ（必要なカラムのみ選択）
    member_columns = ["メンバーコード"]
    for col in ["メンバー名", "職種", "役職", "職能・等級"]:
        if col in members_df_clean.columns:
            member_columns.append(col)
    
    if "メンバーコード" in member_competence_clean.columns and "メンバーコード" in members_df_clean.columns:
        merged_df = member_competence_clean.merge(
            members_df_clean[member_columns],
            on="メンバーコード",
            how="left"
        )
    else:
        st.error("⚠️ メンバーコードカラムが見つかりません")
        return pd.DataFrame()
    
    # 力量マスタとマージ（必要なカラムのみ選択）
    if "力量コード" not in competence_master_clean.columns:
        st.error("⚠️ 力量マスタに力量コードカラムが見つかりません")
        return pd.DataFrame()
    
    comp_columns = ["力量コード"]
    if "力量名" in competence_master_clean.columns:
        comp_columns.append("力量名")
    if "力量タイプ" in competence_master_clean.columns:
        comp_columns.append("力量タイプ")
    
    if "力量コード" in merged_df.columns:
        merged_df = merged_df.merge(
            competence_master_clean[comp_columns],
            on="力量コード",
            how="left"
        )
    else:
        st.error("⚠️ マージ後のデータに力量コードカラムが見つかりません")
        return pd.DataFrame()
    
    # フィルタリング
    filtered_df = merged_df.copy()
    
    if "職種" in filters and filters["職種"]:
        if "職種" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["職種"].isin(filters["職種"])]
    
    if "役職" in filters and filters["役職"]:
        if "役職" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["役職"].isin(filters["役職"])]
    
    if "等級" in filters and filters["等級"]:
        if "職能・等級" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["職能・等級"].isin(filters["等級"])]
    
    if "カテゴリ" in filters and filters["カテゴリ"]:
        if "力量タイプ" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["力量タイプ"].isin(filters["カテゴリ"])]
    
    if "最小レベル" in filters and filters["最小レベル"]:
        if "レベル" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["レベル"] >= filters["最小レベル"]]
    
    # データが空の場合のチェック
    if len(filtered_df) == 0:
        st.warning("⚠️ フィルタ条件に一致するデータがありません")
        return pd.DataFrame()
    
    # ピボットテーブルのindexカラムを動的に構築
    index_cols = []
    if "メンバーコード" in filtered_df.columns:
        index_cols.append("メンバーコード")
    
    # 存在するカラムだけを追加
    optional_cols = ["メンバー名", "職種", "役職", "職能・等級"]
    for col in optional_cols:
        if col in filtered_df.columns:
            index_cols.append(col)
    
    if len(index_cols) == 0:
        st.error("⚠️ ピボットテーブルのindexに使用できるカラムがありません")
        return pd.DataFrame()
    
    # columnsに使用するカラムを確認
    if "力量名" not in filtered_df.columns:
        st.error("⚠️ 力量名カラムが見つかりません。")
        # 力量名がない場合は力量コードを使用
        if "力量コード" in filtered_df.columns:
            st.warning("力量名の代わりに力量コードを使用します")
            column_name = "力量コード"
        else:
            st.error("力量コードも見つかりません")
            return pd.DataFrame()
    else:
        column_name = "力量名"
    
    # ピボットテーブル化（メンバー × スキル）
    if "レベル" in filtered_df.columns:
        # レベル情報を数値に変換
        filtered_df["レベル_数値"] = pd.to_numeric(filtered_df["レベル"], errors='coerce').fillna(0)
        
        pivot_df = filtered_df.pivot_table(
            index=index_cols,
            columns=column_name,
            values="レベル_数値",
            fill_value=0,
            aggfunc="max"
        ).reset_index()
    else:
        # レベル情報がない場合は保有/未保有（1/0）
        filtered_df["保有"] = 1
        pivot_df = filtered_df.pivot_table(
            index=index_cols,
            columns=column_name,
            values="保有",
            fill_value=0,
            aggfunc="max"
        ).reset_index()
    
    # カラム名をフラット化
    pivot_df.columns.name = None
    
    st.dataframe(pivot_df, use_container_width=True, height=500)
    
    return pivot_df


def render_export_buttons(
    dataframe: pd.DataFrame,
    filename_prefix: str = "export"
) -> None:
    """
    CSV/Excelエクスポートボタンを表示
    
    Args:
        dataframe: エクスポートするDataFrame
        filename_prefix: ファイル名のプレフィックス
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV エクスポート（UTF-8 BOM付き）
        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 CSV出力",
            data=csv_data,
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel エクスポート
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Excel出力",
            data=excel_data,
            file_name=f"{filename_prefix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


def render_metric_cards_row(metrics: List[Dict]) -> None:
    """
    メトリクスカードを横並びで表示
    
    Args:
        metrics: メトリクス情報のリスト
                 [{"label": "ラベル", "value": "値", "delta": "変化量（オプション）"}]
    """
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta")
            )


def render_cross_tab_heatmap(
    cross_tab_df: pd.DataFrame,
    title: str = "クロス集計ヒートマップ"
) -> None:
    """
    クロス集計のヒートマップを描画
    
    Args:
        cross_tab_df: クロス集計DataFrame（ピボットテーブル形式）
        title: グラフタイトル
    """
    fig = px.imshow(
        cross_tab_df,
        labels=dict(x=cross_tab_df.columns.name or "項目2", 
                   y=cross_tab_df.index.name or "項目1", 
                   color="値"),
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title=title,
        text_auto=True
    )
    
    fig.update_layout(
        height=max(300, len(cross_tab_df) * 40),
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
