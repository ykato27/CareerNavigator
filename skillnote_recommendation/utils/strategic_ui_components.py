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


def render_skill_coverage_matrix(
    candidates_df: pd.DataFrame,
    target_profile: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    max_candidates: int = 10,
    max_skills: int = 15
) -> None:
    """
    スキルカバレッジマトリクス（ヒートマップ）を表示
    
    Args:
        candidates_df: 候補者DataFrame
        target_profile: 目標スキルプロファイル
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        max_candidates: 表示する最大候補者数
        max_skills: 表示する最大スキル数
    """
    import numpy as np
    
    if len(candidates_df) == 0:
        st.info("候補者がいません")
        return
    
    # 上位候補者とトップスキルを選択
    top_candidates = candidates_df.head(max_candidates)
    top_skills = target_profile.head(max_skills)["力量コード"].tolist()
    
    # スキル名を取得
    skill_names = competence_master_df[
        competence_master_df["力量コード"].isin(top_skills)
    ].set_index("力量コード")["力量名"].to_dict()
    
    # マトリクスデータを作成
    matrix_data = []
    candidate_names = []
    
    for idx, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
        # 候補者名にランク番号を追加してユニークに
        candidate_names.append(f"{idx}. {candidate['メンバー名']}")
        member_code = candidate["メンバーコード"]
        
        # このメンバーのスキルを取得
        member_skills = member_competence_df[
            member_competence_df["メンバーコード"] == member_code
        ]
        
        row_data = []
        for skill_code in top_skills:
            # スキルを保有しているか確認
            has_skill = skill_code in member_skills["力量コード"].values
            
            if has_skill:
                # レベル情報があれば使用（1-5など）
                level_data = member_skills[member_skills["力量コード"] == skill_code]["レベル"]
                if len(level_data) > 0:
                    level = pd.to_numeric(level_data.iloc[0], errors='coerce')
                    if pd.notna(level):
                        row_data.append(level)
                    else:
                        row_data.append(1)  # レベル情報なしだが保有
                else:
                    row_data.append(1)
            else:
                row_data.append(0)  # 未保有
        
        matrix_data.append(row_data)
    
    # DataFrameに変換（スキル名を短く）
    matrix_df = pd.DataFrame(
        matrix_data,
        index=candidate_names,
        columns=[skill_names.get(sc, sc[:8])[:15] + "..." if len(skill_names.get(sc, sc[:8])) > 15 else skill_names.get(sc, sc[:8]) for sc in top_skills]
    )
    
    # ヒートマップ作成
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale=[
            [0, 'rgb(255,200,200)'],      # 赤: 未保有
            [0.2, 'rgb(255,255,200)'],    # 黄: レベル1
            [0.5, 'rgb(200,255,200)'],    # 薄緑: レベル2-3
            [1, 'rgb(100,200,100)']       # 緑: レベル4-5
        ],
        text=matrix_df.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='候補者: %{y}<br>スキル: %{x}<br>レベル: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="スキルカバレッジマトリクス",
        xaxis_title="必須スキル",
        yaxis_title="候補者",
        height=max(400, len(top_candidates) * 40),
        xaxis={
            'side': 'top',
            'tickangle': -45,  # ラベルを斜めに
            'tickfont': {'size': 9}  # フォントサイズを小さく
        },
        yaxis={
            'tickfont': {'size': 10}
        },
        margin=dict(l=100, r=20, t=150, b=20)  # 上部マージンを広げる
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 赤: 未保有 | 🟡 黄: レベル1 | 🟢 緑: レベル2以上")


def render_candidate_comparison_dashboard(
    candidates_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame
) -> None:
    """
    候補者比較ダッシュボードを表示
    
    Args:
        candidates_df: 候補者DataFrame
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
    """
    st.markdown("### 🔍 候補者比較分析")
    
    # 候補者選択UI
    st.markdown("#### 比較する候補者を選択（2-4人）")
    
    # 選択可能な候補者リスト
    candidate_options = {}
    for idx, row in candidates_df.head(10).iterrows():
        label = f"{row['メンバー名']} (準備度: {row['準備度スコア']*100:.1f}%)"
        candidate_options[label] = row["メンバーコード"]
    
    selected_labels = st.multiselect(
        "候補者を選択",
        options=list(candidate_options.keys()),
        max_selections=4,
        key="compare_candidates"
    )
    
    # デバッグ情報を表示
    if len(candidate_options) < 2:
        st.warning(f"⚠️ 比較可能な候補者が{len(candidate_options)}人しかいません。フィルタ条件を緩和してください。")
        return
    
    if len(selected_labels) < 2:
        st.info("👆 比較するには2人以上の候補者を選択してください")
        return
    
    # 選択された候補者のコード
    selected_codes = [candidate_options[label] for label in selected_labels]
    
    # 選択された候補者のデータを取得
    selected_candidates = candidates_df[
        candidates_df["メンバーコード"].isin(selected_codes)
    ]
    
    st.markdown("---")
    
    # メトリクス比較カード
    st.markdown("#### 📊 メトリクス比較")
    cols = st.columns(len(selected_candidates))
    
    for col, (_, candidate) in zip(cols, selected_candidates.iterrows()):
        with col:
            with st.container():
                st.markdown(f"**{candidate['メンバー名']}**")
                st.metric("準備度", f"{candidate['準備度スコア']*100:.1f}%")
                st.metric("スキルマッチ", f"{candidate['スキルマッチ度']*100:.1f}%")
                st.metric("保有スキル", f"{candidate['保有スキル数']}個")
                st.metric("不足スキル", f"{candidate['不足スキル数']}個")
    
    st.markdown("---")
    
    # バーチャート比較
    st.markdown("#### 📈 総合スコア比較")
    
    comparison_data = []
    for _, candidate in selected_candidates.iterrows():
        comparison_data.append({
            "候補者": candidate["メンバー名"],
            "準備度スコア": candidate["準備度スコア"] * 100,
            "スキルマッチ度": candidate["スキルマッチ度"] * 100
        })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_melted = comp_df.melt(id_vars=["候補者"], var_name="指標", value_name="スコア (%)")
    
    fig = px.bar(
        comp_melted,
        x="候補者",
        y="スコア (%)",
        color="指標",
        barmode="group",
        title="候補者別スコア比較"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # レーダーチャート（スキルカテゴリ別）
    st.markdown("#### 🎯 スキルカテゴリ別強み分析")
    
    # 力量タイプカラムの存在チェック
    if "力量タイプ" not in competence_master_df.columns:
        st.info("💡 力量マスタに「力量タイプ」カラムがないため、カテゴリ別分析をスキップします")
    else:
        # 力量タイプ別のスキル数を集計
        radar_data = []
        
        for _, candidate in selected_candidates.iterrows():
            member_code = candidate["メンバーコード"]
            member_skills = member_competence_df[
                member_competence_df["メンバーコード"] == member_code
            ]

            # 力量タイプ別にカウント
            skill_by_type = member_skills.merge(
                competence_master_df[["力量コード", "力量タイプ"]],
                on="力量コード",
                how="left"
            )

            # マージ後に力量タイプカラムが存在するか確認
            if "力量タイプ" in skill_by_type.columns and not skill_by_type.empty:
                type_counts = skill_by_type["力量タイプ"].value_counts().to_dict()
            else:
                type_counts = {}

            radar_data.append({
                "候補者": candidate["メンバー名"],
                **type_counts
            })
    
        if radar_data:
            radar_df = pd.DataFrame(radar_data).fillna(0)
            
            # レーダーチャート作成
            categories = [col for col in radar_df.columns if col != "候補者"]
            
            fig = go.Figure()
            
            for _, row in radar_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=row["候補者"]
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="スキルカテゴリ別保有数",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # 差分ハイライト
    st.markdown("---")
    st.markdown("#### 💡 主な違い")
    
    # 準備度が最も高い候補者
    best_readiness = selected_candidates.loc[selected_candidates["準備度スコア"].idxmax()]
    st.success(f"🏆 **最高準備度**: {best_readiness['メンバー名']} ({best_readiness['準備度スコア']*100:.1f}%)")
    
    # スキルマッチ度が最も高い候補者
    best_match = selected_candidates.loc[selected_candidates["スキルマッチ度"].idxmax()]
    st.info(f"🎯 **最高スキルマッチ**: {best_match['メンバー名']} ({best_match['スキルマッチ度']*100:.1f}%)")
    
    # 不足スキルが最も少ない候補者
    least_gap = selected_candidates.loc[selected_candidates["不足スキル数"].idxmin()]
    st.success(f"✨ **最少ギャップ**: {least_gap['メンバー名']} (不足{least_gap['不足スキル数']}個)")


def render_development_roadmap(
    roadmap_df: pd.DataFrame,
    candidate_name: str
) -> None:
    """
    育成ロードマップを可視化
    
    Args:
        roadmap_df: 育成ロードマップDataFrame
        candidate_name: 候補者名
    """
    if roadmap_df.empty:
        st.info(f"{candidate_name}さんは既に全ての必須スキルを保有しています！")
        return
    
    st.markdown(f"### 📚 {candidate_name}さんの育成ロードマップ")
    
    # サマリーカード
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総不足スキル数", f"{len(roadmap_df)}個")
    
    with col2:
        high_priority = len(roadmap_df[roadmap_df["優先度"] == "High"])
        st.metric("高優先度", f"{high_priority}個", delta="重要" if high_priority > 0 else None)
    
    with col3:
        total_months = roadmap_df["推定習得期間（月）"].sum()
        st.metric("総推定期間", f"{total_months}ヶ月")
    
    with col4:
        avg_months = int(roadmap_df["推定習得期間（月）"].mean())
        st.metric("平均習得期間", f"{avg_months}ヶ月/スキル")
    
    st.markdown("---")
    
    # タイムライン表示
    st.markdown("#### 📅 マイルストーン別スキル習得計画")
    
    # マイルストーン別にグループ化
    milestone_order = ["3ヶ月後", "6ヶ月後", "1年後", "1年以降"]
    
    for milestone in milestone_order:
        milestone_skills = roadmap_df[roadmap_df["マイルストーン"] == milestone]
        
        if len(milestone_skills) > 0:
            with st.expander(f"🎯 {milestone} ({len(milestone_skills)}スキル)", expanded=(milestone == "3ヶ月後")):
                # 優先度別に色分け
                for priority in ["High", "Medium", "Low"]:
                    priority_skills = milestone_skills[milestone_skills["優先度"] == priority]
                    
                    if len(priority_skills) > 0:
                        if priority == "High":
                            st.markdown(f"##### 🔴 高優先度 ({len(priority_skills)}スキル)")
                        elif priority == "Medium":
                            st.markdown(f"##### 🟡 中優先度 ({len(priority_skills)}スキル)")
                        else:
                            st.markdown(f"##### 🔵 低優先度 ({len(priority_skills)}スキル)")
                        
                        # スキルリスト表示
                        for _, skill in priority_skills.head(10).iterrows():
                            st.markdown(
                                f"- **{skill['力量名']}** "
                                f"(現在Lv.{skill['現在レベル']} → 目標Lv.{skill['目標レベル']}, "
                                f"保有率{skill['保有率']}%)"
                            )
    
    st.markdown("---")
    
    # ガントチャート風の可視化
    st.markdown("#### 📊 スキル習得タイムライン")
    
    # データ準備
    roadmap_display = roadmap_df.head(20).copy()  # 上位20スキル
    roadmap_display["開始月"] = 0
    
    # マイルストーンを数値に変換
    milestone_to_month = {
        "3ヶ月後": 3,
        "6ヶ月後": 6,
        "1年後": 12,
        "1年以降": 18
    }
    
    roadmap_display["終了月"] = roadmap_display["マイルストーン"].map(milestone_to_month)
    
    # 優先度順に開始月を調整（高優先度は早く開始）
    cumulative_month = 0
    for idx, row in roadmap_display.iterrows():
        roadmap_display.loc[idx, "開始月"] = min(cumulative_month, row["終了月"] - 1)
        if row["優先度"] == "High":
            cumulative_month += 0  # 並行して進める
        else:
            cumulative_month += 1
    
    # 横棒グラフ
    fig = go.Figure()
    
    # 優先度別に色分け
    priority_colors = {"High": "red", "Medium": "orange", "Low": "lightblue"}
    
    for priority in ["High", "Medium", "Low"]:
        priority_data = roadmap_display[roadmap_display["優先度"] == priority]
        
        for _, skill in priority_data.iterrows():
            fig.add_trace(go.Bar(
                y=[skill["力量名"][:30]],  # 名前を30文字に切る
                x=[skill["終了月"] - skill["開始月"]],
                base=skill["開始月"],
                orientation='h',
                name=priority,
                marker=dict(color=priority_colors[priority]),
                showlegend=True if skill.name == priority_data.index[0] else False,
                hovertemplate=f"<b>{skill['力量名']}</b><br>" +
                             f"優先度: {priority}<br>" +
                             f"期間: {skill['開始月']}ヶ月目 - {skill['終了月']}ヶ月目<br>" +
                             f"習得期間: {skill['推定習得期間（月）']}ヶ月<extra></extra>"
            ))
    
    fig.update_layout(
        title="スキル習得計画（上位20スキル）",
        xaxis_title="月数",
        yaxis_title="",
        height=max(400, len(roadmap_display) * 25),
        barmode='overlay',
        showlegend=True,
        legend=dict(title="優先度")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 詳細テーブル
    st.markdown("#### 📋 スキル習得計画の詳細")
    
    display_roadmap = roadmap_df[[
        "力量名", "力量タイプ", "現在レベル", "目標レベル", 
        "優先度", "推定習得期間（月）", "マイルストーン", "保有率"
    ]].copy()
    
    # スタイリング
    def highlight_priority(row):
        if row["優先度"] == "High":
            return ['background-color: #ffcccc'] * len(row)
        elif row["優先度"] == "Medium":
            return ['background-color: #fff4cc'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = display_roadmap.style.apply(highlight_priority, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    st.caption("🔴 赤背景: 高優先度 | 🟡 黄背景: 中優先度")


def render_candidate_detail_expanded(
    candidate: pd.Series,
    target_profile: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    members_df: pd.DataFrame,
    planner
) -> None:
    """
    候補者の詳細情報を拡張表示（ドリルダウン）
    
    Args:
        candidate: 候補者のSeries
        target_profile: 目標スキルプロファイル
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        members_df: メンバーマスタ
        planner: SuccessionPlannerインスタンス
    """
    st.markdown(f"## 👤 {candidate['メンバー名']}さんの詳細分析")
    
    # 基本情報カード
    st.markdown("### 📋 基本情報")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("準備度スコア", f"{candidate['準備度スコア']*100:.1f}%")
    
    with col2:
        st.metric("スキルマッチ度", f"{candidate['スキルマッチ度']*100:.1f}%")
    
    with col3:
        st.metric("現在の役職", candidate.get("現在の役職", "不明"))
    
    with col4:
        st.metric("現在の等級", candidate.get("現在の等級", "不明"))
    
    st.markdown("---")
    
    # 強み分析
    st.markdown("### 💪 強み分析")
    
    try:
        strengths = planner.analyze_candidate_strengths(
            candidate,
            member_competence_df,
            competence_master_df,
            members_df
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("総スキル数", f"{strengths['総スキル数']}個")
            st.caption(f"全メンバー中、上位{100 - strengths['スキル数百分位']:.1f}%")
        
        with col2:
            st.metric("最強カテゴリ", strengths['最強カテゴリ'])
            st.caption(f"{strengths['最強カテゴリスキル数']}スキル保有")
        
        with col3:
            if strengths['カテゴリ別内訳']:
                top_3_categories = sorted(
                    strengths['カテゴリ別内訳'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                st.markdown("**カテゴリ別TOP3**")
                for cat, count in top_3_categories:
                    st.markdown(f"- {cat}: {count}個")
    
    except Exception as e:
        st.warning(f"強み分析でエラーが発生しました: {e}")
    
    st.markdown("---")
    
    # 育成ロードマップ
    try:
        roadmap_df = planner.generate_development_roadmap(
            candidate,
            target_profile,
            competence_master_df,
            member_competence_df
        )
        
        render_development_roadmap(roadmap_df, candidate['メンバー名'])
        
    except Exception as e:
        st.error(f"育成ロードマップの生成でエラーが発生しました: {e}")
        st.exception(e)


def render_whatif_simulation(
    candidates_df: pd.DataFrame,
    members_df: pd.DataFrame,
    member_competence_df: pd.DataFrame,
    competence_master_df: pd.DataFrame,
    planner
) -> None:
    """
    What-Ifシミュレーションを表示
    
    Args:
        candidates_df: 候補者DataFrame
        members_df: メンバーマスタ
        member_competence_df: メンバー習得力量データ
        competence_master_df: 力量マスタ
        planner: SuccessionPlannerインスタンス
    """
    st.markdown("### 🔮 What-If シミュレーション")
    st.markdown("候補者を選択すると、その人を後継者にした場合の組織への影響を分析します")
    
    # 候補者選択
    candidate_options = {}
    for idx, row in candidates_df.head(5).iterrows():
        label = f"{row['メンバー名']} (準備度: {row['準備度スコア']*100:.1f}%)"
        candidate_options[label] = row["メンバーコード"]
    
    selected_label = st.selectbox(
        "シミュレーションする候補者",
        options=list(candidate_options.keys()),
        key="whatif_candidate_select"
    )
    
    if st.button("🚀 影響をシミュレーション", type="primary", key="run_whatif"):
        selected_member_code = candidate_options[selected_label]
        selected_candidate = candidates_df[
            candidates_df["メンバーコード"] == selected_member_code
        ].iloc[0]
        
        with st.spinner("組織への影響を分析中..."):
            try:
                impact_result = planner.simulate_succession_impact(
                    selected_candidate,
                    members_df,
                    member_competence_df,
                    competence_master_df,
                    position_column="役職"
                )
                
                # セッションステートに保存
                st.session_state.whatif_impact = impact_result
                st.success("✅ シミュレーション完了！")
                
            except Exception as e:
                st.error(f"シミュレーション中にエラーが発生しました: {e}")
                st.exception(e)
    
    # 結果表示
    if "whatif_impact" in st.session_state and st.session_state.whatif_impact is not None:
        impact = st.session_state.whatif_impact
        
        st.markdown("---")
        st.markdown(f"#### 📊 {impact['候補者名']}さんを選択した場合の影響分析")
        
        # サマリーカード
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("空くポジション", impact['現在のポジション'])
        
        with col2:
            cascade_count = impact['連鎖的な影響']['後継者候補数']
            st.metric("連鎖的な後継者候補", f"{cascade_count}人")
        
        with col3:
            skill_count = impact['移動するスキル']['総スキル数']
            st.metric("移動するスキル", f"{skill_count}個")
        
        st.markdown("---")
        
        # 連鎖分析の詳細
        st.markdown("#### 🔗 連鎖的な影響")
        
        cascade_info = impact['連鎖的な影響']
        
        if cascade_info['後継者候補数'] > 0:
            st.info(
                f"💡 {impact['現在のポジション']}が空くため、"
                f"さらに{cascade_info['後継者候補数']}人の候補者が見つかりました"
            )
            
            # 連鎖候補リスト
            if cascade_info['連鎖候補詳細'] is not None and not cascade_info['連鎖候補詳細'].empty:
                st.markdown(f"**{impact['現在のポジション']}の後継者候補TOP3:**")
                
                for i, name in enumerate(cascade_info['後継者候補'][:3], 1):
                    st.markdown(f"{i}. {name}")
                
                # 詳細テーブル
                with st.expander("連鎖候補の詳細を表示"):
                    cascade_df = cascade_info['連鎖候補詳細'][[
                        "メンバー名", "準備度スコア", "スキルマッチ度", "保有スキル数"
                    ]].copy()
                    cascade_df["準備度スコア"] = (cascade_df["準備度スコア"] * 100).round(1).astype(str) + "%"
                    cascade_df["スキルマッチ度"] = (cascade_df["スキルマッチ度"] * 100).round(1).astype(str) + "%"
                    
                    st.dataframe(cascade_df, use_container_width=True)
        else:
            st.warning(f"⚠️ {impact['現在のポジション']}の後継者候補が見つかりませんでした")
        
        st.markdown("---")
        
        # スキル移動の可視化
        st.markdown("#### 📦 移動するスキル")
        
        skill_types = impact['移動するスキル']['スキルタイプ別']
        
        if skill_types:
            # 円グラフ
            fig = go.Figure(data=[go.Pie(
                labels=list(skill_types.keys()),
                values=list(skill_types.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title=f"{impact['候補者名']}さんが持つスキルタイプの内訳",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 組織バランスへの影響
        st.markdown("---")
        st.markdown("#### ⚖️ 組織バランスへの影響")
        
        try:
            # 現在のバランススコア
            current_balance = planner.calculate_organization_balance_score(
                members_df,
                member_competence_df,
                group_column="職種"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "現在の組織バランススコア",
                    f"{current_balance['バランススコア']}/100"
                )
                st.caption(f"スキル分散: {current_balance['スキル分散']}")
            
            with col2:
                # 簡易的な予測（実際の異動をシミュレート）
                st.info("✨ 異動後の詳細な影響分析は組織シミュレーションタブで実行できます")
        
        except Exception as e:
            st.warning(f"バランススコア計算でエラー: {e}")


def render_scenario_management() -> None:
    """
    シナリオ保存・比較機能を表示
    """
    st.markdown("### 📂 シナリオ管理")
    st.markdown("複数の後継者プランを保存して比較できます")
    
    # セッションステートの初期化
    if "succession_scenarios" not in st.session_state:
        st.session_state.succession_scenarios = {}
    
    # シナリオ保存
    st.markdown("#### 💾 現在のプランを保存")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scenario_name = st.text_input(
            "シナリオ名",
            placeholder="例: 安定重視プラン",
            key="scenario_name_input"
        )
    
    with col2:
        if st.button("保存", key="save_scenario"):
            if scenario_name and "succession_candidates" in st.session_state:
                # 現在の状態を保存
                st.session_state.succession_scenarios[scenario_name] = {
                    "候補者": st.session_state.succession_candidates.copy(),
                    "対象役職": st.session_state.get("target_position", ""),
                    "保存日時": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success(f"✅ シナリオ「{scenario_name}」を保存しました")
            else:
                st.warning("シナリオ名を入力し、候補者を検索してから保存してください")
    
    # 保存されたシナリオ一覧
    if st.session_state.succession_scenarios:
        st.markdown("---")
        st.markdown("#### 📋 保存されたシナリオ")
        
        scenario_list = []
        for name, data in st.session_state.succession_scenarios.items():
            scenario_list.append({
                "シナリオ名": name,
                "対象役職": data["対象役職"],
                "候補者数": len(data["候補者"]),
                "保存日時": data["保存日時"]
            })
        
        scenario_df = pd.DataFrame(scenario_list)
        st.dataframe(scenario_df, use_container_width=True)
        
        # シナリオ比較
        if len(st.session_state.succession_scenarios) >= 2:
            st.markdown("---")
            st.markdown("#### ⚖️ シナリオ比較")
            
            scenario_names = list(st.session_state.succession_scenarios.keys())
            
            col1, col2 = st.columns(2)
            
            with col1:
                scenario_a = st.selectbox("シナリオA", options=scenario_names, key="scenario_a")
            
            with col2:
                scenario_b = st.selectbox("シナリオB", options=[s for s in scenario_names if s != scenario_a], key="scenario_b")
            
            if st.button("🔄 比較する", key="compare_scenarios"):
                data_a = st.session_state.succession_scenarios[scenario_a]
                data_b = st.session_state.succession_scenarios[scenario_b]
                
                st.markdown(f"##### {scenario_a} vs {scenario_b}")
                
                # TOP3候補者の比較
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{scenario_a} のTOP3**")
                    for i, (_, row) in enumerate(data_a["候補者"].head(3).iterrows(), 1):
                        st.markdown(f"{i}. {row['メンバー名']} (準備度: {row['準備度スコア']*100:.1f}%)")
                
                with col2:
                    st.markdown(f"**{scenario_b} のTOP3**")
                    for i, (_, row) in enumerate(data_b["候補者"].head(3).iterrows(), 1):
                        st.markdown(f"{i}. {row['メンバー名']} (準備度: {row['準備度スコア']*100:.1f}%)")
                
                # メリット・デメリット
                st.markdown("##### 💡 特徴分析")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{scenario_a})**")
                    avg_readiness_a = data_a["候補者"]["準備度スコア"].mean()
                    st.markdown(f"- 平均準備度: {avg_readiness_a*100:.1f}%")
                    st.markdown(f"- 候補者数: {len(data_a['候補者'])}人")
                
                with col2:
                    st.markdown(f"**{scenario_b})**")
                    avg_readiness_b = data_b["候補者"]["準備度スコア"].mean()
                    st.markdown(f"- 平均準備度: {avg_readiness_b*100:.1f}%")
                    st.markdown(f"- 候補者数: {len(data_b['候補者'])}人")
        
        # シナリオ削除
        st.markdown("---")
        st.markdown("#### 🗑️ シナリオ削除")
        
        delete_scenario = st.selectbox(
            "削除するシナリオ",
            options=list(st.session_state.succession_scenarios.keys()),
            key="delete_scenario_select"
        )
        
        if st.button("削除", key="delete_scenario_btn", type="secondary"):
            del st.session_state.succession_scenarios[delete_scenario]
            st.success(f"✅ シナリオ「{delete_scenario}」を削除しました")
            st.rerun()
    else:
        st.info("まだシナリオが保存されていません")
