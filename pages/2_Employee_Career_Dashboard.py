"""
CareerNavigator - 従業員向けキャリアダッシュボード (MVP)

従業員が自分のキャリアパスを明確に理解できるダッシュボード
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional

from skillnote_recommendation.graph import CompetenceKnowledgeGraph
from skillnote_recommendation.graph.career_path import (
    CareerGapAnalyzer,
    LearningPathGenerator,
)
from skillnote_recommendation.graph.causal_career_path import (
    CausalFilteredLearningPath,
    DependencyAnalyzer,
    SmartRoadmapVisualizer,
)
from skillnote_recommendation.graph.career_path_visualizer import (
    CareerPathVisualizer,
    format_career_path_summary,
)
from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
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

# 機能概要説明
with st.expander("ℹ️ このダッシュボードでできること", expanded=True):
    st.markdown("""
    このダッシュボードでは、あなたの現在のスキル状況を分析し、キャリア目標に向けた最適な学習パスを提案します。
    
    ### 🌟 主な機能
    
    1. **現状分析 (Current Status)**
       - あなたが現在保有しているスキルとレベルを可視化します。
       - 強みや専門性を一目で把握できます。
       
    2. **目標設定 (Goal Setting)**
       - 目指したい「ロールモデル（先輩社員）」や「職種」を設定できます。
       - 目標と現状のギャップ（不足スキル）を自動分析します。
       
    3. **AIスキル推薦 (Causal Recommendation)**
       - 因果推論AIが、あなたの保有スキルに基づいて「次に学ぶべきスキル」を提案します。
       - 「Aを学ぶとBが習得しやすくなる」という因果関係を考慮し、効率的な学習順序を導き出します。
       
    4. **学習ロードマップ (Smart Roadmap)**
       - 推奨スキルをどのような順序で学ぶべきか、ガントチャート形式で表示します。
       - スキル間の依存関係を考慮した、無理のない計画を立案できます。
    """)


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
# サイドバー: モード選択
# =========================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 表示モード")
    
    display_mode = st.radio(
        "モードを選択",
        options=["通常モード", "キャリア比較モード"],
        help="通常モード: 1つの目標に集中\nキャリア比較モード: 2つの目標をタブで比較"
    )


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
    
    # モードに応じて目標数を決定
    if display_mode == "キャリア比較モード":
        num_targets = 2
        st.info("💡 比較モードでは2つの目標を設定します")
    else:
        num_targets = 1
    
    
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
        
        # モードに応じて目標数を決定
        if display_mode == "キャリア比較モード":
            num_targets = 2
            st.info("💡 比較モードでは2つの目標を設定します")
        else:
            num_targets = 1
        
        
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
# Causal Recommenderの初期化
# =========================================================
if "causal_recommender" not in st.session_state:
    with st.spinner("🧠 因果グラフモデルを読み込み中..."):
        try:
            # Causal Recommenderを事前に学習しておく想定
            import pickle
            from pathlib import Path
            
            model_path = Path("models/causal_recommender.pkl")
            
            if model_path.exists():
                with open(model_path, "rb") as f:
                    causal_recommender = pickle.load(f)
                st.session_state.causal_recommender = causal_recommender
                st.success("✅ 学習済みCausal Recommenderを読み込みました")
            else:
                # モデルがない場合は新規学習
                st.warning("⚠️ 学習済みモデルが見つかりません。新規学習します...")
                causal_recommender = CausalGraphRecommender(
                    member_competence=member_competence,
                    competence_master=competence_master
                )
                causal_recommender.fit()
                st.session_state.causal_recommender = causal_recommender
                
                # モデルを保存
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump(causal_recommender, f)
                st.success("✅ Causal Recommenderを学習し、保存しました")
        except Exception as e:
            st.error(f"❌ Causal Recommenderの初期化エラー: {e}")
            st.stop()

causal_recommender = st.session_state.causal_recommender


# =========================================================
# 推薦閾値の調整UI
# =========================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Causal推薦設定")
    
    st.markdown("#### 📊 スコアフィルタリング")
    
    min_total_score = st.slider(
        "総合スコア閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,  # 0.05 → 0.01に変更
        help="この値以上のCausalスコアを持つスキルのみ推薦",
        key="min_total_score"
    )
    
    min_readiness = st.slider(
        "準備完了度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,  # 0.05 → 0.01に変更
        help="準備ができているスキルを優先",
        key="min_readiness"
    )
    
    st.markdown("#### 🔗 依存関係設定")
    
    min_effect_threshold = st.slider(
        "依存関係の閾値",
        min_value=0.0,
        max_value=0.5,
        value=0.03,
        step=0.01,
        help="スキル間の依存関係と見なす最小因果効果",
        key="min_effect_threshold"
    )
    
    st.markdown("---")
    
    # フィルタリング条件を表示
    st.info(f"""
    **現在の設定**:
    - 総合スコア ≥ {min_total_score:.2f}
    - 準備完了度 ≥ {min_readiness:.2f}
    - 依存関係 ≥ {min_effect_threshold:.2f}
    
    💡 スライダーを動かすと自動的に再描画されます
    """)


# =========================================================
# キャリアパス分析と可視化（Causal統合版）
# =========================================================
if target_configs and selected_member:
    st.markdown("---")
    st.subheader("🗺️ Causal統合キャリアロードマップ")
    
    # 分析器を初期化
    gap_analyzer = CareerGapAnalyzer(
        knowledge_graph=knowledge_graph,
        member_competence_df=member_competence,
        competence_master_df=competence_master
    )
    
    smart_visualizer = SmartRoadmapVisualizer()
    
    # タブで複数パスを表示
    if len(target_configs) > 1:
        tabs = st.tabs([config["label"] for config in target_configs])
    else:
        tabs = [st.container()]
    
    for idx, (tab, config) in enumerate(zip(tabs, target_configs)):
        with tab:
            target_member = config["target_member"]
            
            # Causal統合の分析器を初期化（ループ内で毎回作成して最新の値を使用）
            causal_path_generator = CausalFilteredLearningPath(
                causal_recommender=causal_recommender,
                min_total_score=min_total_score,  # 最新のスライダー値を使用
                min_readiness_score=min_readiness
            )
            
            dependency_analyzer = DependencyAnalyzer(
                causal_recommender=causal_recommender,
                min_effect_threshold=min_effect_threshold  # 最新のスライダー値を使用
            )
            
            with st.spinner(f"キャリアパス分析中... ({config['label']})"):
                try:
                    # ギャップ分析
                    gap_result = gap_analyzer.analyze_gap(
                        source_member_code=selected_member,
                        target_member_code=target_member
                    )
                    
                    # Causalフィルタリング
                    recommended_skills = causal_path_generator.generate_filtered_path(
                        gap_analysis=gap_result,
                        member_code=selected_member
                    )
                    
                    # 依存関係の抽出
                    dependencies = dependency_analyzer.extract_dependencies(
                        competences=recommended_skills,
                        competence_master=competence_master
                    )
                    
                    # サマリー表示
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric(
                            "ギャップスキル（全体）",
                            len(gap_result["missing_competences"]),
                            help="ギャップ分析で抽出されたスキル総数"
                        )
                    
                    with col_sum2:
                        st.metric(
                            "推薦スキル数",
                            len(recommended_skills),
                            help="Causalフィルタリング後の推薦スキル数"
                        )
                    
                    with col_sum3:
                        avg_score = sum(s.total_score for s in recommended_skills) / len(recommended_skills) if recommended_skills else 0
                        st.metric(
                            "平均スコア",
                            f"{avg_score:.2f}",
                            help="推薦スキルの平均Causalスコア"
                        )
                    
                    with col_sum4:
                        # 推定学習期間（依存関係考慮）
                        total_deps = sum(len(d["prerequisites"]) for d in dependencies.values())
                        estimated_weeks = len(recommended_skills) * 2 + total_deps
                        estimated_months = estimated_weeks / 4
                        st.metric(
                            "推定期間",
                            f"約{estimated_months:.1f}ヶ月",
                            help="依存関係を考慮した推定期間"
                        )
                    
                    # Causal統合ロードマップ可視化
                    if recommended_skills:
                        st.markdown("#### 📊 Causal統合学習ロードマップ")
                        st.info("""
                        🧠 **Causal統合の特徴**:
                        - 因果グラフに基づくスキル推薦
                        - 依存関係を考慮した直列・並列配置
                        - 準備完了度と有用性を両面から評価
                        """)
                        
                        # デバッグ情報：依存関係統計
                        total_deps = sum(len(d["prerequisites"]) for d in dependencies.values())
                        total_enables = sum(len(d["enables"]) for d in dependencies.values())
                        
                        col_debug1, col_debug2, col_debug3 = st.columns(3)
                        with col_debug1:
                            st.metric("検出された依存関係", f"{total_deps}件", help="前提スキルの総数")
                        with col_debug2:
                            st.metric("有効化関係", f"{total_enables}件", help="このスキルが役立つ関係の総数")
                        with col_debug3:
                            avg_deps = total_deps / len(dependencies) if dependencies else 0
                            st.metric("平均前提数", f"{avg_deps:.1f}", help="1スキルあたりの前提スキル数")
                        
                        if total_deps == 0:
                            st.warning("⚠️ 依存関係が検出されませんでした。依存関係の閾値を下げてみてください。")
                        
                        target_name = config["label"]
                        roadmap_fig = smart_visualizer.create_dependency_based_roadmap(
                            competences=recommended_skills,
                            dependencies=dependencies,
                            target_member_name=target_name
                        )
                        st.plotly_chart(roadmap_fig, use_container_width=True, key=f"causal_roadmap_{idx}")
                    else:
                        st.warning("⚠️ 推薦スキルが見つかりませんでした。閾値を下げてみてください。")
                    
                    # 推薦スキルの詳細リスト
                    if recommended_skills:
                        st.markdown("---")
                        st.markdown("#### 📝 推薦スキル詳細（Causalスコア順）")
                        
                        df_data = []
                        for comp in recommended_skills:
                            # 依存関係情報を取得
                            deps = dependencies.get(comp.competence_code, {})
                            prereq_count = len(deps.get("prerequisites", []))
                            enables_count = len(deps.get("enables", []))
                            
                            df_data.append({
                                "力量名": comp.competence_name,
                                "カテゴリー": comp.category,
                                "🎯 総合スコア": f"{comp.total_score:.3f}",
                                "✅ 準備完了度": f"{comp.readiness_score:.3f}",
                                "📊 確率": f"{comp.bayesian_score:.3f}",
                                "🚀 有用性": f"{comp.utility_score:.3f}",
                                "📌 前提": prereq_count,
                                "➡️ 次へ": enables_count,
                            })
                        
                        df_skills = pd.DataFrame(df_data)
                        st.dataframe(df_skills, use_container_width=True)
                        
                        # CSVダウンロード
                        csv_data = df_skills.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 推薦スキルをダウンロード",
                            data=csv_data,
                            file_name=f"causal_recommended_skills_{selected_member}.csv",
                            mime="text/csv",
                            key=f"download_causal_skills_{idx}"
                        )
                        
                        # 推薦理由の詳細表示
                        with st.expander("🔍 推薦理由の詳細"):
                            for i, comp in enumerate(recommended_skills[:5]):  # 上位5件
                                st.markdown(f"### {i+1}. {comp.competence_name}")
                                
                                col_reason1, col_reason2 = st.columns(2)
                                
                                with col_reason1:
                                    st.markdown("**✅ 準備ができています:**")
                                    if comp.readiness_reasons:
                                        for skill_name, effect in comp.readiness_reasons[:3]:
                                            st.markdown(f"- {skill_name} (因果効果: {effect:.3f})")
                                    else:
                                        st.markdown("- 基礎スキルとして推奨")
                                
                                with col_reason2:
                                    st.markdown("**🚀 役立つ場面:**")
                                    if comp.utility_reasons:
                                        for skill_name, effect in comp.utility_reasons[:3]:
                                            st.markdown(f"- {skill_name}の習得に役立つ (効果: {effect:.3f})")
                                    else:
                                        st.markdown("- 汎用スキル")
                                
                                st.markdown("---")
                        
                        # 関連因果グラフ
                        st.markdown("---")
                        st.markdown("#### 🔗 関連因果グラフ")
                        st.caption("推薦スキルを中心とした因果関係を可視化")
                        
                        if recommended_skills:
                            # スキル選択
                            skill_options = [
                                f"{i+1}. {comp.competence_name} (スコア: {comp.total_score:.2f})" 
                                for i, comp in enumerate(recommended_skills)
                            ]
                            
                            selected_skill_idx = st.selectbox(
                                "グラフを表示するスキルを選択",
                                range(len(recommended_skills)),
                                format_func=lambda x: skill_options[x],
                                key=f"skill_graph_select_{idx}"
                            )
                            
                            # グラフ表示設定
                            col_g1, col_g2, col_g3 = st.columns(3)
                            
                            with col_g1:
                                graph_threshold = st.slider(
                                    "表示閾値",
                                    0.01, 1.0, 0.05, 0.01,
                                    key=f"graph_threshold_{idx}",
                                    help="この値以上の因果係数を持つエッジのみ表示"
                                )
                            
                            with col_g2:
                                show_negative_graph = st.checkbox(
                                    "負の因果も表示",
                                    value=False,
                                    key=f"show_negative_{idx}",
                                    help="赤線（負の因果関係）も表示する"
                                )
                            
                            with col_g3:
                                graph_height = st.select_slider(
                                    "グラフの高さ",
                                    options=["小", "中", "大"],
                                    value="中",
                                    key=f"graph_height_{idx}"
                                )
                            
                            height_map = {"小": "400px", "中": "600px", "大": "800px"}
                            
                            try:
                                # 選択されたスキル
                                selected_skill = recommended_skills[selected_skill_idx]
                                center_node = selected_skill.competence_name
                                
                                # Visualizer作成
                                adj_matrix = causal_recommender.learner.get_adjacency_matrix()
                                visualizer = CausalGraphVisualizer(adj_matrix)
                                
                                # 保有スキル情報を取得
                                member_skills_codes = member_competence[
                                    member_competence["メンバーコード"] == selected_member
                                ]["力量コード"].tolist()
                                
                                # コード → 名前変換
                                code_to_name = causal_recommender.code_to_name
                                member_skill_names = [code_to_name.get(c, c) for c in member_skills_codes]
                                
                                # エゴネットワークを生成
                                html_path = visualizer.visualize_ego_network_pyvis(
                                    center_node=center_node,
                                    radius=1,
                                    threshold=graph_threshold,
                                    show_negative=show_negative_graph,
                                    member_skills=member_skill_names,
                                    output_path=f"ego_network_dashboard_{idx}.html",
                                    height=height_map[graph_height]
                                )
                                
                                # HTMLファイルを読み込んで表示
                                with open(html_path, 'r', encoding='utf-8') as f:
                                    source_code = f.read()
                                
                                components.html(source_code, height=int(height_map[graph_height].replace("px", "")), scrolling=False)
                                
                                # 凡例を表示
                                st.caption(f"💡 **{center_node}** を中心とした因果関係（拡大・移動可能）")
                                st.caption(
                                    "🟦 **青**: 推奨スキル（中心） | "
                                    "🟩 **緑**: あなたの保有スキル | "
                                    "⬜ **白**: 将来取得可能なスキル"
                                )
                                
                            except Exception as graph_error:
                                st.warning(f"⚠️ 因果グラフの表示に失敗しました: {graph_error}")
                
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
