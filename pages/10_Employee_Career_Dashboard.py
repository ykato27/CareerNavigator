"""
CareerNavigator - 従業員向けキャリアダッシュボード (MVP)

従業員が自分のキャリアパスを明確に理解できるダッシュボード
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

from skillnote_recommendation.graph import CompetenceKnowledgeGraph
from skillnote_recommendation.graph.career_path import (
    CareerGapAnalyzer,
    LearningPathGenerator,
)
from skillnote_recommendation.graph.career_path_visualizer import (
    CareerPathVisualizer,
    format_career_path_summary,
)
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - キャリアダッシュボード",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply enterprise UI styles
apply_enterprise_styles()

# ページヘッダー
render_page_header(
    title="🎯 従業員向けキャリアダッシュボード",
    icon="🚀",
    description="あなたのキャリアパスを可視化し、次のステップを明確にします"
)


# =========================================================
# データ読み込みチェック
# =========================================================
if "transformed_data" not in st.session_state or "knowledge_graph" not in st.session_state:
    st.warning("⚠️ データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
    st.stop()

transformed_data = st.session_state.transformed_data
knowledge_graph = st.session_state.knowledge_graph

# 必要なデータを取得
competence_master = transformed_data["competence_master"]
member_competence = transformed_data["member_competence"]
members_clean = transformed_data["members_clean"]


# =========================================================
# メンバー選択
# =========================================================
st.subheader("👤 メンバー選択")

member_codes = sorted(member_competence["メンバーコード"].unique())

# メンバー名を取得する関数
def get_member_name(code):
    member_info = members_clean[members_clean['メンバーコード'] == code]
    if len(member_info) > 0:
        return f"{code} - {member_info.iloc[0]['メンバー名']}"
    return code

col1, col2 = st.columns([2, 1])

with col1:
    selected_member = st.selectbox(
        "分析対象メンバー（あなた）",
        options=member_codes,
        format_func=get_member_name,
        key="source_member"
    )

with col2:
    # 現在の保有スキル数を表示
    if selected_member:
        current_skills = member_competence[
            member_competence["メンバーコード"] == selected_member
        ]
        st.metric("現在の保有スキル", f"{len(current_skills)}件")


# =========================================================
# キャリア目標の選択方式
# =========================================================
st.markdown("---")
st.subheader("🎯 キャリア目標の設定")

target_selection_mode = st.radio(
    "目標設定方法を選択",
    options=["ロールモデルから選ぶ", "職種・役職から選ぶ"],
    horizontal=True
)

target_configs = []  # 複数の目標を格納

if target_selection_mode == "ロールモデルから選ぶ":
    st.markdown("#### 目指すロールモデル")
    
    # 複数選択可能
    num_targets = st.number_input(
        "比較する目標数",
        min_value=1,
        max_value=3,
        value=1,
        help="最大3つまで比較できます"
    )
    
    for i in range(int(num_targets)):
        with st.expander(f"目標 {i+1}", expanded=(i == 0)):
            target_member = st.selectbox(
                "ロールモデルを選択",
                options=[m for m in member_codes if m != selected_member],
                format_func=get_member_name,
                key=f"target_member_{i}"
            )
            
            if target_member:
                target_info = members_clean[members_clean['メンバーコード'] == target_member]
                if len(target_info) > 0:
                    target_role = target_info.iloc[0].get('役職', '未設定')
                    st.info(f"**役職**: {target_role}")
                
                target_skills = member_competence[
                    member_competence["メンバーコード"] == target_member
                ]
                st.metric("保有スキル数", f"{len(target_skills)}件")
                
                target_configs.append({
                    "mode": "member",
                    "target_member": target_member,
                    "label": f"{get_member_name(target_member)}"
                })

else:  # 職種・役職から選ぶ
    st.markdown("#### 目指す職種・役職")
    
    # 役職一覧を取得
    if '役職' in members_clean.columns:
        roles = sorted(members_clean['役職'].dropna().unique())
        
        num_targets = st.number_input(
            "比較する目標数",
            min_value=1,
            max_value=3,
            value=1,
            help="最大3つまで比較できます"
        )
        
        for i in range(int(num_targets)):
            with st.expander(f"目標 {i+1}", expanded=(i == 0)):
                target_role = st.selectbox(
                    "目標役職を選択",
                    options=roles,
                    key=f"target_role_{i}"
                )
                
                if target_role:
                    # その役職の代表メンバーを選択（スキル数が多い人）
                    role_members = members_clean[members_clean['役職'] == target_role]['メンバーコード'].tolist()
                    
                    # 各メンバーのスキル数をカウント
                    skill_counts = {}
                    for rm in role_members:
                        if rm != selected_member:
                            skill_count = len(member_competence[
                                member_competence["メンバーコード"] == rm
                            ])
                            skill_counts[rm] = skill_count
                    
                    if skill_counts:
                        # スキル数が多い順にソート
                        top_member = max(skill_counts, key=skill_counts.get)
                        
                        st.info(
                            f"**代表メンバー**: {get_member_name(top_member)} "
                            f"({skill_counts[top_member]}スキル保有)"
                        )
                        
                        target_configs.append({
                            "mode": "role",
                            "target_member": top_member,
                            "target_role": target_role,
                            "label": f"{target_role}（代表: {get_member_name(top_member)}）"
                        })
                    else:
                        st.warning(f"⚠️ 役職「{target_role}」に該当する他のメンバーが見つかりません")
    else:
        st.error("❌ メンバーマスタに「役職」列が存在しません")


# =========================================================
# キャリアパス分析と可視化
# =========================================================
if target_configs and selected_member:
    st.markdown("---")
    st.subheader("🗺️ キャリアロードマップ")
    
    # 分析器を初期化
    gap_analyzer = CareerGapAnalyzer(
        knowledge_graph=knowledge_graph,
        member_competence_df=member_competence,
        competence_master_df=competence_master
    )
    
    path_generator = LearningPathGenerator(
        knowledge_graph=knowledge_graph
    )
    
    visualizer = CareerPathVisualizer()
    
    # タブで複数パスを表示
    if len(target_configs) > 1:
        tabs = st.tabs([config["label"] for config in target_configs])
    else:
        tabs = [st.container()]
    
    for idx, (tab, config) in enumerate(zip(tabs, target_configs)):
        with tab:
            target_member = config["target_member"]
            
            with st.spinner(f"キャリアパス分析中... ({config['label']})"):
                try:
                    # ギャップ分析
                    gap_result = gap_analyzer.analyze_gap(
                        source_member_code=selected_member,
                        target_member_code=target_member
                    )
                    
                    # 学習パス生成
                    career_path = path_generator.generate_learning_path(
                        gap_analysis=gap_result,
                        max_per_phase=5
                    )
                    
                    # サマリー表示
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    
                    with col_sum1:
                        st.metric(
                            "ギャップスキル数",
                            len(career_path.missing_competences),
                            help="目標達成に必要なスキル数"
                        )
                    
                    with col_sum2:
                        st.metric(
                            "到達度",
                            f"{career_path.estimated_completion_rate * 100:.0f}%",
                            help="現在の進捗率"
                        )
                    
                    with col_sum3:
                        # 推定学習期間（簡易計算: 1スキル = 2週間）
                        estimated_weeks = len(career_path.missing_competences) * 2
                        estimated_months = estimated_weeks / 4
                        st.metric(
                            "推定期間",
                            f"約{estimated_months:.1f}ヶ月",
                            help="全スキル習得にかかる推定期間"
                        )
                    
                    # プログレスバー
                    st.progress(career_path.estimated_completion_rate)
                    
                    # ロードマップ可視化
                    st.markdown("#### 📊 学習ロードマップ")
                    
                    target_name = config["label"]
                    roadmap_fig = visualizer.create_roadmap(career_path, target_name)
                    st.plotly_chart(roadmap_fig, use_container_width=True)
                    
                    # 到達度ゲージ
                    col_gauge1, col_gauge2 = st.columns(2)
                    
                    with col_gauge1:
                        st.markdown("#### 🎯 到達度")
                        gauge_fig = visualizer.create_progress_gauge(
                            career_path.estimated_completion_rate
                        )
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    with col_gauge2:
                        st.markdown("#### 📂 カテゴリー内訳")
                        category_fig = visualizer.create_category_breakdown(career_path)
                        st.plotly_chart(category_fig, use_container_width=True)
                    
                    # 詳細な学習パス
                    st.markdown("---")
                    st.markdown("#### 📝 段階別学習パス")
                    
                    phase_tabs = st.tabs([
                        f"🌱 Phase 1: 基礎固め ({len(career_path.phase_1_competences)})",
                        f"🌿 Phase 2: 専門性構築 ({len(career_path.phase_2_competences)})",
                        f"🌳 Phase 3: エキスパート ({len(career_path.phase_3_competences)})"
                    ])
                    
                    phases = [
                        career_path.phase_1_competences,
                        career_path.phase_2_competences,
                        career_path.phase_3_competences
                    ]
                    
                    for phase_tab, phase_comps in zip(phase_tabs, phases):
                        with phase_tab:
                            if len(phase_comps) > 0:
                                df_data = []
                                for comp in phase_comps:
                                    df_data.append({
                                        "力量名": comp.competence_name,
                                        "カテゴリー": comp.category,
                                        "重要度": f"{comp.importance_score:.2f}",
                                        "習得容易性": f"{comp.ease_score:.2f}",
                                        "優先度スコア": f"{comp.priority_score:.2f}"
                                    })
                                
                                df_phase = pd.DataFrame(df_data)
                                st.dataframe(df_phase, use_container_width=True)
                                
                                # CSVダウンロード
                                csv_data = df_phase.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label=f"📥 Phase {phase_tabs.index(phase_tab) + 1} をダウンロード",
                                    data=csv_data,
                                    file_name=f"learning_path_phase{phase_tabs.index(phase_tab) + 1}_{selected_member}.csv",
                                    mime="text/csv",
                                    key=f"download_phase_{idx}_{phase_tabs.index(phase_tab)}"
                                )
                            else:
                                st.info("このフェーズで習得すべきスキルはありません")
                    
                    # サマリーテキスト
                    with st.expander("📄 キャリアパスサマリー（テキスト形式）"):
                        summary_text = format_career_path_summary(career_path, target_name)
                        st.markdown(summary_text)
                
                except Exception as e:
                    st.error(f"❌ キャリアパス分析エラー: {e}")
                    import traceback
                    with st.expander("🔍 詳細なエラー情報"):
                        st.code(traceback.format_exc())


# =========================================================
# アクションプラン
# =========================================================
if target_configs and selected_member:
    st.markdown("---")
    st.subheader("🎬 次のアクション")
    
    st.markdown("""
    ### あなたの次のステップ
    
    #### 🔹 今週のアクション
    1. **Phase 1の最初の3つのスキルを確認**
       - 基礎スキルから順に習得することで、効率的にキャリアアップできます
       
    2. **学習リソースを探す**
       - 社内研修プログラムを確認
       - オンライン教材（Udemy、Courseraなど）を検索
       
    3. **上司・メンターに相談**
       - このキャリアパスを共有し、サポートを依頼
    
    #### 🔹 今月の目標
    - Phase 1のスキルを **少なくとも1つ** 習得
    - 進捗を記録し、このダッシュボードで確認
    
    #### 🔹 3ヶ月後の目標
    - Phase 1を **80%以上** 完了
    - Phase 2のスキル習得を開始
    """)
    
    # リマインダー設定（プレースホルダー）
    st.info("""
    💡 **ヒント**: このキャリアパスをPDFで保存し、定期的に進捗を確認しましょう。
    週次or月次で上司と1on1ミーティングを設定し、進捗を共有することをお勧めします。
    """)


# =========================================================
# フッター
# =========================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <small>
        💡 このダッシュボードは因果グラフ推薦システムとギャップ分析に基づいています。<br>
        定期的にデータを更新し、最新のキャリアパスを確認してください。
    </small>
</div>
""", unsafe_allow_html=True)
