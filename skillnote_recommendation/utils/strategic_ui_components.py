"""
戦略的配置向けUIコンポーネント

後継者計画と組織シミュレーションの可視化
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import io


def render_succession_candidate_table(
    candidates_df: pd.DataFrame,
    top_n: int = 10
) -> None:
    """
    後継者候補ランキングテーブルを表示
    
    Args:
        candidates_df: 候補者DataFrame
        top_n: 表示する上位件数
    """
    display_df = candidates_df.head(top_n).copy()
    
    # 準備度スコアをパーセント表示に
    display_df["準備度スコア"] = (display_df["準備度スコア"] * 100).round(1).astype(str) + "%"
    display_df["スキルマッチ度"] = (display_df["スキルマッチ度"] * 100).round(1).astype(str) + "%"
    
    # 表示カラムを選択
    display_columns = [
        "メンバー名", "現在の役職", "現在の等級", 
        "準備度スコア", "スキルマッチ度", "保有スキル数", "不足スキル数"
    ]
    
    # スタイリング
    def highlight_readiness(row):
        readiness = float(row["準備度スコア"].replace("%", ""))
        if readiness >= 70:
            return ['background-color: #d4edda'] * len(row)  # 緑
        elif readiness >= 50:
            return ['background-color: #fff3cd'] * len(row)  # 黄
        else:
            return [''] * len(row)
    
    styled_df = display_df[display_columns].style.apply(highlight_readiness, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.caption("🟢 緑背景: 高準備度（70%以上） | 🟡 黄背景: 中準備度（50%以上）")


def render_readiness_gauge(
    readiness_score: float,
    member_name: str
) -> None:
    """
    準備度ゲージを表示
    
    Args:
        readiness_score: 準備度スコア（0.0-1.0）
        member_name: メンバー名
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=readiness_score * 100,
        title={'text': f"{member_name}の準備度"},
        delta={'reference': 70, 'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


def render_skill_gap_comparison(
    target_skills: List[str],
    candidate_skills: List[str],
    max_display: int = 15
) -> None:
    """
    スキルギャップ比較を表示
    
    Args:
        target_skills: 目標スキルリスト
        candidate_skills: 候補者スキルリスト
        max_display: 最大表示スキル数
    """
    # スキルの状態を判定
    matched = [skill for skill in target_skills if skill in candidate_skills]
    missing = [skill for skill in target_skills if skill not in candidate_skills]
    
    # 表示用データ作成
    skill_data = []
    for skill in matched[:max_display]:
        skill_data.append({"スキル": skill, "状態": "保有", "値": 1})
    for skill in missing[:max_display]:
        skill_data.append({"スキル": skill, "状態": "不足", "値": -1})
    
    if len(skill_data) == 0:
        st.info("スキルデータがありません")
        return
    
    df = pd.DataFrame(skill_data)
    
    # 横棒グラフ
    fig = px.bar(
        df,
        x="値",
        y="スキル",
        color="状態",
        orientation='h',
        title="スキルギャップ比較",
        color_discrete_map={"保有": "green", "不足": "red"},
        labels={"値": "", "スキル": ""}
    )
    
    fig.update_layout(
        height=max(300, len(skill_data) * 25),
        showlegend=True,
        xaxis={'visible': False}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_transfer_simulator_ui(
    members_df: pd.DataFrame,
    group_column: str = "職種"
) -> Dict:
    """
    異動シミュレーターUIを表示
    
    Args:
        members_df: メンバーマスタ
        group_column: グループカラム名
        
    Returns:
        異動設定の辞書
    """
    st.markdown("### 異動設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # メンバー選択
        member_options = members_df["メンバー名"].tolist() if "メンバー名" in members_df.columns else members_df["メンバーコード"].tolist()
        selected_member_name = st.selectbox("異動するメンバー", options=member_options)
        
        # メンバーコード取得
        if "メンバー名" in members_df.columns:
            member_code = members_df[members_df["メンバー名"] == selected_member_name]["メンバーコード"].iloc[0]
        else:
            member_code = selected_member_name
    
    # 現在のグループを取得
    if group_column in members_df.columns:
        current_group = members_df[members_df["メンバーコード"] == member_code][group_column].iloc[0]
    else:
        current_group = "不明"
    
    with col2:
        st.text_input("異動元", value=current_group, disabled=True)
    
    with col3:
        # 異動先選択
        group_options = members_df[group_column].dropna().unique().tolist() if group_column in members_df.columns else []
        to_group = st.selectbox("異動先", options=group_options)
    
    return {
        "member_code": member_code,
        "member_name": selected_member_name,
        "from_group": current_group,
        "to_group": to_group
    }


def render_before_after_comparison(
    comparison_df: pd.DataFrame
) -> None:
    """
    前後比較テーブルを表示
    
    Args:
        comparison_df: 比較DataFrame
    """
    st.markdown("### 📊 前後比較")
    
    # 変化量に応じて色付け
    def highlight_changes(row):
        colors = [''] * len(row)
        
        # メンバー数変化
        if "メンバー数_変化" in row.index:
            if row["メンバー数_変化"] > 0:
                idx = row.index.get_loc("メンバー数_変化")
                colors[idx] = 'background-color: lightblue'
            elif row["メンバー数_変化"] < 0:
                idx = row.index.get_loc("メンバー数_変化")
                colors[idx] = 'background-color: lightcoral'
        
        # 平均スキル数変化
        if "平均スキル数/人_変化" in row.index:
            if row["平均スキル数/人_変化"] > 0:
                idx = row.index.get_loc("平均スキル数/人_変化")
                colors[idx] = 'background-color: lightgreen'
            elif row["平均スキル数/人_変化"] < 0:
                idx = row.index.get_loc("平均スキル数/人_変化")
                colors[idx] = 'background-color: lightcoral'
        
        return colors
    
    styled_df = comparison_df.style.apply(highlight_changes, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.caption("🔵 青: メンバー増加 | 🟢 緑: スキル向上 | 🔴 赤: 減少/悪化")


def render_skill_distribution_comparison(
    current_summary: pd.DataFrame,
    simulated_summary: pd.DataFrame
) -> None:
    """
    スキル分布の前後比較グラフ
    
    Args:
        current_summary: 現在のサマリー
        simulated_summary: シミュレーション後のサマリー
    """
    # データを結合
    current_summary["状態"] = "現在"
    simulated_summary["状態"] = "シミュレーション後"
    
    combined = pd.concat([current_summary, simulated_summary])
    
    # グループ化されたバーチャート
    fig = px.bar(
        combined,
        x="グループ",
        y="平均スキル数/人",
        color="状態",
        barmode="group",
        title="職種別平均スキル数の比較",
        labels={"平均スキル数/人": "平均スキル数/人"}
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)


def render_candidate_detail_card(
    candidate: pd.Series,
    target_profile: pd.DataFrame,
    competence_master_df: pd.DataFrame
) -> None:
    """
    候補者詳細カードを表示
    
    Args:
        candidate: 候補者のSeries
        target_profile: 目標スキルプロファイル
        competence_master_df: 力量マスタ
    """
    with st.container():
        st.markdown(f"### {candidate['メンバー名']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("準備度スコア", f"{candidate['準備度スコア']*100:.1f}%")
        
        with col2:
            st.metric("スキルマッチ度", f"{candidate['スキルマッチ度']*100:.1f}%")
        
        with col3:
            st.metric("不足スキル", f"{candidate['不足スキル数']}個")
        
        # スキルギャップ表示
        if "総合スコア詳細" in candidate:
            detail = candidate["総合スコア詳細"]
            if "matched_skill_codes" in detail and "missing_skill_codes" in detail:
                # 力量名を取得
                matched_names = competence_master_df[
                    competence_master_df["力量コード"].isin(detail["matched_skill_codes"])
                ]["力量名"].tolist()
                
                missing_names = competence_master_df[
                    competence_master_df["力量コード"].isin(detail["missing_skill_codes"])
                ]["力量名"].tolist()
                
                render_skill_gap_comparison(
                    matched_names + missing_names,
                    matched_names,
                    max_display=10
                )
