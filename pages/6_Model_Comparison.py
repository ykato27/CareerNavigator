"""
モデル比較分析ページ

Graph-based vs NMF推薦モデルの比較分析を行います。
推薦結果、解釈性、性能を可視化します。
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

from skillnote_recommendation.ml.graph_recommender import SkillTransitionGraphRecommender
from skillnote_recommendation.utils.visualization import (
    create_skill_transition_graph,
    create_graph_statistics_chart
)
from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded
)


def create_comparison_table(graph_recs, nmf_recs, member_code):
    """推薦結果の比較テーブルを作成"""
    data = []

    max_len = max(len(graph_recs), len(nmf_recs))

    for i in range(max_len):
        row = {'順位': i + 1}

        # Graph-based
        if i < len(graph_recs):
            gr = graph_recs[i]
            row['Graph推薦'] = gr.skill_name
            row['Graphスコア'] = f"{gr.score:.2f}"
            row['Graph信頼度'] = f"{gr.confidence:.0%}"
        else:
            row['Graph推薦'] = '-'
            row['Graphスコア'] = '-'
            row['Graph信頼度'] = '-'

        # NMF
        if i < len(nmf_recs):
            nr = nmf_recs[i]
            row['NMF推薦'] = nr['skill_name']
            row['NMFスコア'] = f"{nr['predicted_score']:.2f}"
            row['NMF信頼度'] = f"{nr.get('confidence', 0):.0%}"
        else:
            row['NMF推薦'] = '-'
            row['NMFスコア'] = '-'
            row['NMF信頼度'] = '-'

        data.append(row)

    return pd.DataFrame(data)


def create_interpretability_radar(graph_info, nmf_info=None):
    """解釈性のレーダーチャート"""
    categories = ['解釈性', '推薦精度', '計算速度', 'Cold-start対応', '新規性']

    # Graph-based scores (主観的評価)
    graph_scores = [4, 3, 3, 2, 4]  # 解釈性とグラフ構造の新規性が高い

    # NMF scores
    nmf_scores = [2, 4, 5, 2, 3]  # 精度と速度が高いが解釈性低い

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=graph_scores + [graph_scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Graph-based',
        line_color='#4A90E2'
    ))

    if nmf_info:
        fig.add_trace(go.Scatterpolar(
            r=nmf_scores + [nmf_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='NMF',
            line_color='#E24A4A'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title='モデル特性比較',
        height=500
    )

    return fig


def main():
    st.set_page_config(
        page_title="モデル比較分析 - CareerNavigator",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 推薦モデル比較分析")
    st.markdown("""
    このページでは、**Graph-based推薦**と**NMF推薦**の2つのモデルを比較分析します。

    - 🕸️ **Graph-based**: スキル遷移パターンから学習パスを推薦（高解釈性）
    - 🧮 **NMF**: 行列分解による潜在因子ベース推薦（高精度）
    """)

    st.markdown("---")

    # =========================================================
    # 前提条件チェック
    # =========================================================

    check_data_loaded()

    # =========================================================
    # データ準備
    # =========================================================

    td = st.session_state.transformed_data
    member_competence = td["member_competence"]
    competence_master = td["competence_master"]

    # 取得日カラムの存在チェック
    if '取得日' not in member_competence.columns:
        st.error("❌ Graph-based推薦には「取得日」データが必要です")
        st.info("""
        **対処方法:**
        1. CSVファイルに取得日カラムを追加してください
        2. データを再アップロードしてください
        """)
        st.stop()

    # =========================================================
    # モデル設定
    # =========================================================

    st.sidebar.header("⚙️ モデル設定")

    # Graph-based設定
    st.sidebar.subheader("🕸️ Graph-based")
    time_window = st.sidebar.slider(
        "遷移期間（日数）",
        min_value=30,
        max_value=365,
        value=180,
        step=30,
        help="この期間内のスキル遷移を分析"
    )

    min_transitions = st.sidebar.slider(
        "最小遷移人数",
        min_value=1,
        max_value=10,
        value=2,
        help="この人数以上の遷移のみ使用"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 グラフ可視化設定")

    max_path_length = st.sidebar.slider(
        "最大パス長",
        min_value=2,
        max_value=20,
        value=10,
        help="グラフで表示する最大パス長（中間ノード数）"
    )

    # =========================================================
    # モデル学習
    # =========================================================

    st.header("📊 モデル学習")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🕸️ Graph-basedモデルを学習", type="primary"):
            # プログレスバーと経過時間の表示用プレースホルダー
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            start_time = time.time()

            try:
                # Step 1: 初期化 (10%)
                status_text.text("🔧 モデルを初期化中...")
                progress_bar.progress(10)
                elapsed = time.time() - start_time
                time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")

                graph_recommender = SkillTransitionGraphRecommender(
                    time_window_days=time_window,
                    min_transition_count=min_transitions
                )
                time.sleep(0.1)  # UI更新のための短い待機

                # Step 2: グラフ構築開始 (20%)
                status_text.text("🕸️ スキル遷移グラフを構築中...")
                progress_bar.progress(20)
                elapsed = time.time() - start_time
                time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")

                # グラフ構築のみを実行（fitの前半部分）
                # 実際のfit処理
                graph_recommender.member_competence = member_competence.copy()
                graph_recommender.competence_master = competence_master.copy()
                time.sleep(0.1)  # UI更新のための短い待機

                # Step 3: グラフ構築中 (40%)
                status_text.text("📊 学習遷移パターンを抽出中...")
                progress_bar.progress(40)
                elapsed = time.time() - start_time
                time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")

                graph_recommender.graph = graph_recommender._build_transition_graph()

                # Step 4: グラフ構築完了 (60%)
                elapsed = time.time() - start_time
                status_text.text(f"✅ グラフ構築完了 ({graph_recommender.graph.number_of_nodes()}ノード, {graph_recommender.graph.number_of_edges()}エッジ)")
                progress_bar.progress(60)
                time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")
                time.sleep(0.2)  # ユーザーが結果を確認できるように

                # Step 5: Node2Vec学習 (80%)
                if graph_recommender.graph.number_of_nodes() > 1:
                    status_text.text("🧮 Node2Vec埋め込みを学習中（ランダムウォーク生成）...")
                    progress_bar.progress(70)
                    elapsed = time.time() - start_time
                    time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")
                    time.sleep(0.1)

                    # Node2Vec学習開始
                    status_text.text("🧮 Node2Vec埋め込みを学習中（モデル学習）...")
                    progress_bar.progress(80)
                    elapsed = time.time() - start_time
                    time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")

                    graph_recommender._train_node2vec()

                    # Node2Vec完了
                    elapsed = time.time() - start_time
                    status_text.text("✅ Node2Vec埋め込み学習完了")
                    progress_bar.progress(95)
                    time_text.text(f"⏱️ 経過時間: {elapsed:.1f}秒")
                    time.sleep(0.2)
                else:
                    status_text.warning("⚠️ グラフのノード数が不足。Node2Vecをスキップ")
                    progress_bar.progress(95)

                # Step 6: 完了処理 (100%)
                graph_recommender.is_fitted = True
                graph_recommender.metadata = {
                    'num_nodes': graph_recommender.graph.number_of_nodes(),
                    'num_edges': graph_recommender.graph.number_of_edges(),
                    'time_window_days': graph_recommender.time_window_days,
                    'min_transition_count': graph_recommender.min_transition_count,
                    'has_embeddings': graph_recommender.node2vec_model is not None
                }

                status_text.text("✅ 学習完了！")
                progress_bar.progress(100)

                train_time = time.time() - start_time
                time_text.text(f"⏱️ 総学習時間: {train_time:.2f}秒")

                # セッションステートに保存
                st.session_state['graph_recommender'] = graph_recommender
                st.session_state['graph_train_time'] = train_time

                # 成功メッセージ
                st.success(f"🎉 学習完了！ (所要時間: {train_time:.2f}秒)")

                # グラフ統計
                stats = graph_recommender.get_graph_statistics()

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("グラフノード数", f"{stats['num_nodes']:,}")
                with col_stat2:
                    st.metric("グラフエッジ数", f"{stats['num_edges']:,}")
                with col_stat3:
                    st.metric("グラフ密度", f"{stats['density']:.4f}")

            except Exception as e:
                status_text.text("❌ 学習エラー")
                progress_bar.progress(0)
                elapsed = time.time() - start_time
                time_text.text(f"⏱️ 経過時間: {elapsed:.2f}秒")

                st.error(f"❌ 学習エラー: {e}")
                st.exception(e)

    with col2:
        # NMFモデルの状態確認
        has_ml_recommender = 'ml_recommender' in st.session_state and st.session_state['ml_recommender'] is not None
        has_engine = 'recommendation_engine' in st.session_state and st.session_state['recommendation_engine'] is not None

        if has_ml_recommender and has_engine:
            st.success("✅ NMFモデルは既に学習済みです")

            col_nmf1, col_nmf2 = st.columns(2)
            with col_nmf1:
                st.metric("学習データ数", f"{len(member_competence):,}件")
            with col_nmf2:
                # モデル情報があれば表示
                if hasattr(st.session_state['ml_recommender'], 'n_components'):
                    st.metric("潜在因子数", st.session_state['ml_recommender'].n_components)

        elif has_ml_recommender and not has_engine:
            st.warning("⚠️ NMFモデルは学習済みですが、RecommendationEngineが未初期化です")
            st.info("👉 「推薦実行」ページでモデルを初期化してください")

        else:
            st.warning("⚠️ NMFモデルが学習されていません")
            st.info("""
            **NMFモデルを学習するには:**

            1. サイドバーから「モデル学習」ページに移動
            2. 「モデル学習を開始」ボタンをクリック
            3. 学習完了後、このページに戻る

            Graph-basedモデルのみでも分析可能です。
            """)

    # =========================================================
    # 推薦結果の比較
    # =========================================================

    if 'graph_recommender' in st.session_state or st.session_state.get('ml_recommender'):
        st.markdown("---")
        st.header("🎯 推薦結果の比較")

        # メンバー選択
        members = member_competence['メンバーコード'].unique()
        target_member = st.selectbox(
            "分析対象メンバーを選択",
            options=members,
            help="推薦を生成するメンバーを選択してください"
        )

        top_n = st.slider("推薦件数", min_value=5, max_value=20, value=10)

        if st.button("🔍 推薦を実行", type="primary"):

            col1, col2 = st.columns(2)

            # Graph-based推薦
            with col1:
                st.subheader("🕸️ Graph-based推薦")

                if 'graph_recommender' in st.session_state:
                    try:
                        graph_rec = st.session_state['graph_recommender']
                        graph_recs = graph_rec.recommend(target_member, n=top_n)

                        if graph_recs:
                            for rec in graph_recs:
                                with st.expander(f"#{rec.rank} {rec.skill_name} (スコア: {rec.score:.2f})"):
                                    st.markdown(f"**信頼度:** {rec.confidence:.0%}")
                                    st.markdown("**推薦理由:**")
                                    st.markdown(rec.explanation)

                                    # 学習パスの可視化
                                    try:
                                        fig = create_skill_transition_graph(
                                            graph_rec,
                                            target_member,
                                            rec.skill_code,
                                            max_path_length=max_path_length
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"グラフ表示エラー: {e}")

                            st.session_state['graph_recs'] = graph_recs
                        else:
                            st.info("推薦結果がありません")

                    except Exception as e:
                        st.error(f"Graph-based推薦エラー: {e}")
                        st.exception(e)
                else:
                    st.warning("Graph-basedモデルを先に学習してください")

            # NMF推薦
            with col2:
                st.subheader("🧮 NMF推薦")

                # セッションステートの詳細チェック
                has_ml_recommender = 'ml_recommender' in st.session_state and st.session_state['ml_recommender'] is not None
                has_engine = 'recommendation_engine' in st.session_state and st.session_state['recommendation_engine'] is not None

                if has_ml_recommender and has_engine:
                    try:
                        engine = st.session_state['recommendation_engine']

                        # NMF推薦を実行
                        nmf_recs = engine.recommend_for_member(target_member, top_n=top_n)

                        if nmf_recs:
                            for i, rec in enumerate(nmf_recs, 1):
                                with st.expander(f"#{i} {rec['skill_name']} (スコア: {rec['predicted_score']:.2f})"):
                                    st.markdown(f"**信頼度:** {rec.get('confidence', 0.5):.0%}")
                                    st.markdown("**推薦理由:**")
                                    reason = rec.get('reason', '行列分解による推薦')
                                    st.markdown(reason)

                                    # 類似メンバー情報があれば表示
                                    if 'similar_members' in rec:
                                        st.markdown("**類似メンバー:**")
                                        st.write(rec['similar_members'][:3])

                            st.session_state['nmf_recs'] = nmf_recs
                        else:
                            st.info("推薦結果がありません")

                    except Exception as e:
                        st.error(f"❌ NMF推薦エラー: {e}")
                        st.exception(e)

                elif has_ml_recommender and not has_engine:
                    # MLモデルはあるがEngineがない
                    st.warning("⚠️ RecommendationEngineが初期化されていません")
                    st.info("👉 サイドバーから「推薦実行」ページに移動して、モデルを初期化してください")

                    if st.button("📝 手動でEngineを初期化", key="init_engine"):
                        try:
                            from skillnote_recommendation.core.recommendation_engine import RecommendationEngine

                            with st.spinner("RecommendationEngineを初期化中..."):
                                engine = RecommendationEngine(
                                    st.session_state['ml_recommender'],
                                    member_competence,
                                    competence_master
                                )
                                st.session_state['recommendation_engine'] = engine
                                st.success("✅ 初期化完了！ページをリフレッシュしてください")
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ 初期化エラー: {e}")
                            st.exception(e)

                else:
                    # MLモデルもない
                    st.warning("⚠️ NMFモデルが学習されていません")
                    st.info("""
                    **NMFモデルを使用するには:**

                    1. サイドバーから「モデル学習」ページに移動
                    2. 「モデル学習を開始」ボタンをクリック
                    3. 学習完了後、このページに戻ってきてください

                    または、Graph-basedモデルのみで比較分析を行うこともできます。
                    """)

    # =========================================================
    # 比較分析
    # =========================================================

    if 'graph_recs' in st.session_state and 'nmf_recs' in st.session_state:
        st.markdown("---")
        st.header("📊 比較分析")

        tab1, tab2, tab3 = st.tabs(["📋 推薦結果比較", "🎯 解釈性分析", "📈 グラフ統計"])

        with tab1:
            st.subheader("推薦結果の比較テーブル")

            comparison_df = create_comparison_table(
                st.session_state['graph_recs'],
                st.session_state['nmf_recs'],
                target_member
            )

            st.dataframe(comparison_df, use_container_width=True, height=400)

            # 重複分析
            graph_skills = {rec.skill_code for rec in st.session_state['graph_recs']}
            nmf_skills = {rec['skill_code'] for rec in st.session_state['nmf_recs']}
            overlap = graph_skills & nmf_skills

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Graph-based推薦", len(graph_skills))
            with col2:
                st.metric("NMF推薦", len(nmf_skills))
            with col3:
                st.metric("共通推薦", len(overlap))

            if overlap:
                st.success(f"✅ {len(overlap)}個のスキルが両モデルで推薦されました")
                st.write("共通推薦スキル:", [
                    st.session_state['graph_recommender'].get_skill_name(s)
                    for s in overlap
                ])

        with tab2:
            st.subheader("モデル特性の比較")

            graph_info = st.session_state['graph_recommender'].get_interpretability_info()

            fig = create_interpretability_radar(graph_info, nmf_info=True)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🕸️ Graph-based")
                st.markdown(f"**解釈性スコア:** {graph_info['score']}/5")
                st.markdown(f"**特徴:** {graph_info['level']}")
                st.markdown("**強み:**")
                st.markdown("- 学習パスが直感的")
                st.markdown("- 遷移理由が明確")
                st.markdown("- 可視化が強力")

            with col2:
                st.markdown("### 🧮 NMF")
                st.markdown("**解釈性スコア:** 2/5")
                st.markdown("**特徴:** 低い - 推薦理由の説明が難しい")
                st.markdown("**強み:**")
                st.markdown("- 予測精度が高い")
                st.markdown("- 計算が高速")
                st.markdown("- スケーラブル")

        with tab3:
            if 'graph_recommender' in st.session_state:
                st.subheader("グラフ統計情報")

                graph_rec = st.session_state['graph_recommender']
                stats = graph_rec.get_graph_statistics()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ノード数", stats['num_nodes'])
                with col2:
                    st.metric("エッジ数", stats['num_edges'])
                with col3:
                    st.metric("グラフ密度", f"{stats['density']:.4f}")

                # 次数分布
                st.markdown("### 次数分布")
                fig = create_graph_statistics_chart(graph_rec, 'degree_distribution')
                st.plotly_chart(fig, use_container_width=True)

                # トップスキル
                st.markdown("### 最も学ばれるスキル")
                fig = create_graph_statistics_chart(graph_rec, 'top_skills')
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
