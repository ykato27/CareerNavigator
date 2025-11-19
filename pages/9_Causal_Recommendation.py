import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import graphviz

from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.graph.causal_graph_visualizer import CausalGraphVisualizer
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - 因果推論推薦",
    page_icon="🧭",
    layout="wide"
)

apply_enterprise_styles()

render_page_header(
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

# 学習データのサマリー情報を表示
st.info(f"📊 学習済みモデル: メンバー数 {len(recommender.skill_matrix_.index)}人、スキル数 {len(recommender.skill_matrix_.columns)}個")

# =========================================================
# 推薦 & 可視化セクション
# =========================================================
st.markdown("---")

tab1, tab2 = st.tabs(["👤 メンバー別推薦", "🕸️ 因果グラフ全体"])

with tab1:
    st.subheader("メンバーへのスキル推薦")

    members = td["members_clean"]

    # 推薦可能なメンバーのみを選択肢として表示
    # (skill_matrix_に存在するメンバーコードのみ)
    available_members = recommender.skill_matrix_.index.tolist()
    member_options = [m for m in members["メンバーコード"].tolist() if m in available_members]

    if not member_options:
        st.warning("推薦可能なメンバーが見つかりません。学習データを確認してください。")
        st.stop()

    # メンバー選択
    selected_member_code = st.selectbox(
        "メンバーを選択",
        member_options,
        format_func=lambda x: f"{x} : {members[members['メンバーコード']==x]['氏名'].iloc[0] if '氏名' in members.columns else ''}"
    )

    if selected_member_code:
        st.markdown("### 🎯 推奨スキル（優先順位順）")
        
        # スコアの説明
        with st.expander("📖 スコアの見方", expanded=False):
            st.markdown("""
            推奨スコアは以下の2つの要素から計算されます:
            
            - **Readiness（準備度）**: 現在の保有スキルが、推奨スキルの習得をどれだけサポートするか
              - 高いほど、今すぐ学習を始めやすいスキル
              - 保有スキルから推奨スキルへの因果関係の強さで評価
            
            - **Utility（将来性）**: 推奨スキルを習得することで、将来的にどれだけ多くのスキル習得が可能になるか
              - 高いほど、キャリアの選択肢を広げるスキル
              - 推奨スキルから他のスキルへの因果関係の強さで評価
            
            **総合スコア** = Readiness × 0.6 + Utility × 0.4
            """)
        
        recommendations = recommender.recommend(selected_member_code, top_n=10)

        if not recommendations:
            # メンバーの保有スキル数を表示
            member_skills = recommender.skill_matrix_.loc[selected_member_code]
            owned_count = (member_skills > 0).sum()
            st.warning(f"💡 推奨できるスキルが見つかりませんでした。")
            st.info(f"現在の保有スキル数: {owned_count}個\n\n以下の可能性があります：\n- 既にほとんどのスキルを習得済み\n- 保有スキルと他のスキルの間に明確な因果関係が見つからなかった")
        else:
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    st.markdown(f"#### {i}. {rec['competence_name']}")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.metric("総合スコア", f"{rec['score']:.2f}")
                    with col2:
                        details = rec['details']
                        st.metric("準備度", f"{details['readiness_score']:.2f}")
                    with col3:
                        st.metric("将来性", f"{details['utility_score']:.2f}")
                    
                    
                    st.info(rec['explanation'])
                    
                    # 詳細な理由を表示
                    with st.expander("📋 詳細な推薦理由"):
                        details = rec['details']
                        
                        st.markdown("**🟢 準備度（Readiness）**: なぜこのスキルが推奨されるか")
                        if details['readiness_reasons']:
                            st.markdown("あなたの以下の保有スキルが、このスキルの習得を後押しします:")
                            for skill, effect in details['readiness_reasons'][:5]:
                                st.write(f"- **{skill}** → 因果効果: {effect:.3f}")
                        else:
                            st.write("保有スキルからの直接的な因果関係は検出されませんでした。")
                        
                        st.markdown("**🔵 将来性（Utility）**: このスキルを習得すると何ができるか")
                        if details['utility_reasons']:
                            st.markdown("このスキルを習得すると、以下のスキル習得がスムーズになります:")
                            for skill, effect in details['utility_reasons'][:5]:
                                st.write(f"- **{skill}** ← 因果効果: {effect:.3f}")
                        else:
                            st.write("将来のスキルへの直接的な因果関係は検出されませんでした。")
                    
                    st.markdown("---")
        
        # グラフ表示用の推奨スキル選択
        st.markdown("### 🔗 関連因果グラフ")
        st.caption("選択した推奨スキルを中心とした因果関係")
        
        # 推奨スキルから選択（上位10個まで）
        skill_options = [f"{i+1}. {rec['competence_name']} (スコア: {rec['score']:.2f})" 
                        for i, rec in enumerate(recommendations[:10])]
        selected_skill_idx = st.selectbox(
            "グラフを表示する推奨スキルを選択",
            range(min(10, len(recommendations))),
            format_func=lambda x: skill_options[x],
            help="上位10個の推奨スキルから選択できます。"
        )

        # 表示設定
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            graph_threshold = st.slider(
                "表示閾値",
                0.01, 1.0, 0.05, 0.01,
                key="ego_threshold",
                help="この値以上の因果係数を持つエッジのみ表示"
            )
        with col_g2:
            physics_enabled = st.checkbox(
                "物理演算",
                value=True,
                key="ego_physics",
                help="ノードの自動配置（重い場合はOFF推奨）"
            )
        with col_g3:
            show_negative_ego = st.checkbox(
                "負の因果も表示",
                value=False,
                key="ego_show_negative",
                help="赤線（負の因果関係）も表示する"
            )

        # エゴネットワークの可視化
        if recommendations:
            center_node = recommendations[selected_skill_idx]['competence_name']

            # Visualizer作成
            adj_matrix = recommender.learner.get_adjacency_matrix()
            visualizer = CausalGraphVisualizer(adj_matrix)

            # 保有スキルをハイライト用リストに
            member_skills_codes = td["member_competence"][
                td["member_competence"]["メンバーコード"] == selected_member_code
            ]["力量コード"].tolist()

            # コード -> 名前変換
            code_to_name = recommender.code_to_name
            member_skill_names = [code_to_name.get(c, c) for c in member_skills_codes]

            try:
                # エゴネットワークをインタラクティブに表示
                html_path = visualizer.visualize_ego_network_pyvis(
                    center_node=center_node,
                    radius=1,
                    threshold=graph_threshold,
                    show_negative=show_negative_ego,
                    member_skills=member_skill_names,
                    output_path="ego_network.html",
                    height="600px"
                )
                
                # HTMLファイルを読み込んで表示
                with open(html_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=600, scrolling=False)
                
                # 凡例を表示
                st.caption(f"💡 **{center_node}** を中心とした因果関係（拡大・移動可能）")
                st.caption(
                    "🟦 **青**: 推奨スキル（中心） | "
                    "🟩 **緑**: あなたの保有スキル（なぜ推奨されるか） | "
                    "⬜ **白**: 将来取得可能なスキル"
                )
            except Exception as e:
                st.error(f"グラフを描画できませんでした: {e}")

with tab2:
    st.subheader("因果グラフ全体像（インタラクティブ）")
    st.caption("学習されたスキル間の因果関係の全体像")

    # 表示設定パネル
    st.info(
        "📊 **因果関係の表示について**\n\n"
        "- **黒線（正の因果）**: スキルAを習得すると、スキルBの習得が促進される関係\n"
        "- **赤線（負の因果）**: スキルAを習得すると、スキルBの習得が抑制される関係（競合・代替関係など）\n\n"
        "デフォルトでは正の因果関係のみを表示します。"
    )
    
    st.warning(
        "⚠️ **パフォーマンスに関する注意**\n\n"
        "グラフのノード数やエッジ数が多いと、ブラウザが重くなったりクラッシュする可能性があります。\n\n"
        "**推奨設定**: 表示ノード数 10-20個、表示閾値 0.3以上から開始してください。"
    )
    
    col1, col2, col3 = st.columns(3)

    with col1:
        display_mode = st.selectbox(
            "表示モード",
            ["全体（主要ノード）", "全体（全ノード）"],
            help="全ノード表示は非常に重くなります。主要ノードモードを推奨します。"
        )

    with col2:
        threshold = st.slider(
            "表示閾値（高いほど軽量）",
            0.05, 1.0, 0.3, 0.01,
            key="global_threshold",
            help="この値以上の因果係数を持つエッジのみ表示。高い値ほど表示されるエッジが少なくなり軽量になります。"
        )

    with col3:
        top_n = st.slider(
            "表示ノード数",
            5, 100, 20, 5,
            key="global_top_n",
            help="次数中心性が高い上位Nノードを表示。少ない数から始めることを推奨します。"
        ) if display_mode == "全体（主要ノード）" else 1000

    
    # 負の因果関係の表示オプション
    show_negative = st.checkbox(
        "負の因果関係も表示する（赤線）",
        value=False,
        help="チェックを入れると、負の因果関係（抑制関係）も表示されます。グラフが複雑になる可能性があります。"
    )

    if st.button("🎨 インタラクティブグラフを描画", type="primary"):
        with st.spinner("グラフを生成中..."):
            try:
                adj_matrix = recommender.learner.get_adjacency_matrix()
                visualizer = CausalGraphVisualizer(adj_matrix)

                html_path = visualizer.visualize_interactive(
                    output_path="causal_graph_interactive.html",
                    threshold=threshold,
                    top_n=top_n,
                    height="800px",
                    width="100%"
                )

                # HTMLファイルを読み込んで表示
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                components.html(html_content, height=820, scrolling=True)

                st.success(f"✅ {top_n}個のノード（次数中心性上位）を表示しました")
                st.caption("💡 ノードをドラッグ・ズーム・クリックして操作できます")

            except Exception as e:
                st.error(f"グラフ描画エラー: {e}")
                st.exception(e)

    # フォールバック: 静的グラフ表示
    with st.expander("📊 静的グラフを表示（軽量版）"):
        if st.button("静的グラフを描画"):
            adj_matrix = recommender.learner.get_adjacency_matrix()
            visualizer = CausalGraphVisualizer(adj_matrix)

            dot = visualizer.visualize(threshold=threshold)
            st.graphviz_chart(dot)
