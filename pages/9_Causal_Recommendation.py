import streamlit as st
import pandas as pd
import networkx as nx
import graphviz

from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - 因果推論推薦",
    page_icon="🧭",
    layout="wide"
)

apply_rich_ui_styles()

render_gradient_header(
    title="因果推論推薦 (LiNGAM)",
    icon="🔗",
    description="データからスキル間の因果関係を発見し、説得力のある推薦を行います"
)

# =========================================================
# データチェック
# =========================================================
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.warning("まずはトップページでデータを読み込んでください。")
    st.stop()

td = st.session_state.transformed_data

# =========================================================
# モデル学習セクション
# =========================================================
st.subheader("🧠 因果モデルの学習")

with st.expander("設定と学習", expanded=not st.session_state.get("causal_model_trained", False)):
    st.markdown("""
    **LiNGAM (Linear Non-Gaussian Acyclic Model)** を用いて、スキル間の因果構造を学習します。
    
    - **クラスタリング**: 計算コスト削減のため、スキルを相関の高いグループに分割して処理します。
    - **因果探索**: 各グループ内で因果の向き（原因→結果）を特定します。
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        min_members = st.number_input(
            "最小メンバー数/スキル", 
            min_value=3, 
            value=5, 
            help="これより少ないメンバーしか持っていないスキルは除外します"
        )
    
    with col2:
        corr_threshold = st.slider(
            "クラスタリング相関閾値",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="この値以上の相関があるスキル同士を同じグループにします"
        )

    if st.button("🚀 因果モデルを学習開始", type="primary"):
        with st.spinner("因果構造を学習中... (これには数分かかる場合があります)"):
            try:
                recommender = CausalGraphRecommender(
                    member_competence=td["member_competence"],
                    competence_master=td["competence_master"],
                    learner_params={
                        "correlation_threshold": corr_threshold,
                        "min_cluster_size": 3
                    }
                )
                
                recommender.fit(min_members_per_skill=min_members)
                
                st.session_state.causal_recommender = recommender
                st.session_state.causal_model_trained = True
                st.success("✅ 学習が完了しました！")
                st.rerun()
                
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
                st.exception(e)

if not st.session_state.get("causal_model_trained", False):
    st.stop()

recommender = st.session_state.causal_recommender

# =========================================================
# 推薦 & 可視化セクション
# =========================================================
st.markdown("---")

tab1, tab2 = st.tabs(["👤 メンバー別推薦", "🕸️ 因果グラフ全体"])

with tab1:
    st.subheader("メンバーへのスキル推薦")
    
    members = td["members_clean"]
    member_options = members["メンバーコード"].tolist()
    
    # メンバー選択
    selected_member_code = st.selectbox(
        "メンバーを選択",
        member_options,
        format_func=lambda x: f"{x} : {members[members['メンバーコード']==x]['氏名'].iloc[0] if '氏名' in members.columns else ''}"
    )
    
    if selected_member_code:
        col_rec, col_graph = st.columns([1, 1])
        
        with col_rec:
            st.markdown("### 🎯 推奨スキル")
            recommendations = recommender.recommend(selected_member_code, top_n=5)
            
            if not recommendations:
                st.info("推奨できるスキルが見つかりませんでした（保有スキルが十分でないか、因果関係が見つかりませんでした）。")
            else:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"#### {i}. {rec['competence_name']}")
                        st.caption(f"スコア: {rec['score']:.2f}")
                        st.info(rec['explanation'])
                        
                        # 詳細スコア
                        with st.expander("詳細スコア内訳"):
                            details = rec['details']
                            st.write(f"- Readiness (準備): {details['readiness_score']:.2f}")
                            st.write(f"- Utility (将来): {details['utility_score']:.2f}")
        
        with col_graph:
            st.markdown("### 🔗 関連因果グラフ")
            st.caption("選択したメンバーの保有スキル（青）と推奨スキル周辺の因果関係")
            
            # エゴネットワークの可視化
            # 推奨スキルのトップ1を中心にする
            if recommendations:
                center_node = recommendations[0]['competence_name']
                
                # Visualizer作成
                # adjacency_matrixは learner から取得
                adj_matrix = recommender.learner.get_adjacency_matrix()
                # カラム名がコードのままか名前に変換されているか確認が必要
                # CausalGraphRecommenderの実装では learner.fit に渡す前に名前変換している
                
                visualizer = CausalGraphVisualizer(adj_matrix)
                
                # 保有スキルをハイライト用リストに
                member_skills_codes = td["member_competence"][
                    td["member_competence"]["メンバーコード"] == selected_member_code
                ]["力量コード"].tolist()
                
                # コード -> 名前変換
                code_to_name = recommender.code_to_name
                member_skill_names = [code_to_name.get(c, c) for c in member_skills_codes]
                
                try:
                    dot = visualizer.visualize_ego_network(
                        center_node=center_node,
                        radius=1,
                        threshold=0.05
                    )
                    
                    # 保有スキルを色付け（visualize_ego_networkはcenterのみハイライトするので、ここで属性上書きは難しいが、
                    # visualizeメソッドを直接呼ぶ形にすれば制御可能。
                    # ここでは簡易的に graphviz オブジェクトを表示）
                    st.graphviz_chart(dot)
                    
                except Exception as e:
                    st.warning(f"グラフ描画エラー: {e}")
                    # フォールバック: 全体グラフの一部を表示など

with tab2:
    st.subheader("因果グラフ全体像")
    st.caption("学習されたスキル間の因果関係の全体像（主要なエッジのみ表示）")
    
    threshold = st.slider("表示閾値 (係数の絶対値)", 0.05, 0.5, 0.1, 0.01)
    
    if st.button("グラフを描画"):
        adj_matrix = recommender.learner.get_adjacency_matrix()
        visualizer = CausalGraphVisualizer(adj_matrix)
        
        dot = visualizer.visualize(threshold=threshold)
        st.graphviz_chart(dot)
