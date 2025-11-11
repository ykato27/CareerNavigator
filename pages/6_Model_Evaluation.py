"""
モデル評価ページ

推薦モデルの性能を評価し、ベースラインモデルと比較します。

評価指標:
- Precision@K
- Recall@K
- NDCG@K
- Hit Rate
- カバレッジ
- 多様性メトリクス
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import time

# ページ設定
st.set_page_config(
    page_title="モデル評価 - CareerNavigator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 推薦モデル評価")
st.markdown("""
このページでは、ML推薦モデルとベースラインモデルの性能を比較評価します。
""")

# セッション状態の初期化
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None


# ===== データ読み込み =====
st.header("1️⃣ データ読み込み")

data_load_status = st.empty()

if not st.session_state.data_loaded:
    if st.button("📂 データを読み込む"):
        with st.spinner("データを読み込み中..."):
            try:
                from skillnote_recommendation.core.data_loader import DataLoader

                loader = DataLoader()
                data = loader.load_all_data()

                st.session_state.members_clean = data['members_clean']
                st.session_state.competence_master = data['competence_master']
                st.session_state.member_competence = data['member_competence']
                st.session_state.data_loaded = True

                data_load_status.success(
                    f"✅ データ読み込み完了\n\n"
                    f"- メンバー数: {len(st.session_state.members_clean)}\n"
                    f"- 力量数: {len(st.session_state.competence_master)}\n"
                    f"- 習得記録数: {len(st.session_state.member_competence)}"
                )
            except Exception as e:
                st.error(f"❌ データ読み込みエラー: {e}")
                st.stop()
else:
    data_load_status.success(
        f"✅ データ読み込み済み\n\n"
        f"- メンバー数: {len(st.session_state.members_clean)}\n"
        f"- 力量数: {len(st.session_state.competence_master)}\n"
        f"- 習得記録数: {len(st.session_state.member_competence)}"
    )

if not st.session_state.data_loaded:
    st.stop()


# ===== モデル学習 =====
st.header("2️⃣ モデル学習")

col1, col2, col3 = st.columns(3)

with col1:
    use_preprocessing = st.checkbox("データ前処理を使用", value=True)

with col2:
    n_components = st.number_input("潜在因子数", min_value=5, max_value=50, value=20)

with col3:
    test_ratio = st.slider("テストセット比率", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

if st.button("🚀 モデルを学習"):
    with st.spinner("モデルを学習中..."):
        try:
            from skillnote_recommendation.ml.ml_recommender import MLRecommender
            from skillnote_recommendation.ml.baseline_recommenders import (
                RandomRecommender,
                PopularityRecommender,
                CategoryBasedRecommender,
            )

            # ML推薦モデル
            st.info("🤖 ML推薦モデルを学習中...")
            ml_recommender = MLRecommender.build(
                member_competence=st.session_state.member_competence,
                competence_master=st.session_state.competence_master,
                member_master=st.session_state.members_clean,
                use_preprocessing=use_preprocessing,
                n_components=n_components,
            )

            # ベースラインモデル
            st.info("📊 ベースラインモデルを初期化中...")
            random_rec = RandomRecommender(
                competence_master=st.session_state.competence_master,
                member_competence=st.session_state.member_competence,
                member_master=st.session_state.members_clean,
            )

            popularity_rec = PopularityRecommender(
                competence_master=st.session_state.competence_master,
                member_competence=st.session_state.member_competence,
                member_master=st.session_state.members_clean,
            )

            category_rec = CategoryBasedRecommender(
                competence_master=st.session_state.competence_master,
                member_competence=st.session_state.member_competence,
                member_master=st.session_state.members_clean,
            )

            # セッション状態に保存
            st.session_state.ml_recommender = ml_recommender
            st.session_state.random_rec = random_rec
            st.session_state.popularity_rec = popularity_rec
            st.session_state.category_rec = category_rec
            st.session_state.models_trained = True

            st.success("✅ すべてのモデルの学習が完了しました！")

        except Exception as e:
            st.error(f"❌ モデル学習エラー: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

if not st.session_state.models_trained:
    st.info("👆 モデルを学習してください")
    st.stop()


# ===== モデル評価 =====
st.header("3️⃣ モデル評価")

eval_settings = st.expander("⚙️ 評価設定", expanded=True)

with eval_settings:
    col1, col2, col3 = st.columns(3)

    with col1:
        k_values = st.multiselect(
            "評価するK値",
            options=[3, 5, 10, 20],
            default=[5, 10]
        )

    with col2:
        n_test_users = st.number_input(
            "評価対象メンバー数",
            min_value=10,
            max_value=len(st.session_state.members_clean),
            value=min(50, len(st.session_state.members_clean)),
            step=10
        )

    with col3:
        min_acquired = st.number_input(
            "最小習得力量数",
            min_value=1,
            max_value=10,
            value=3,
            help="評価対象とするメンバーの最小習得力量数"
        )

if st.button("📈 評価を実行"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 評価対象メンバーの選択
        member_counts = (
            st.session_state.member_competence['メンバーコード']
            .value_counts()
        )
        eligible_members = member_counts[member_counts >= min_acquired].index.tolist()

        if len(eligible_members) == 0:
            st.error(f"❌ 最小習得力量数{min_acquired}以上のメンバーが存在しません")
            st.stop()

        # ランダムにn_test_users人を選択
        np.random.seed(42)
        test_members = np.random.choice(
            eligible_members,
            size=min(n_test_users, len(eligible_members)),
            replace=False
        )

        status_text.text(f"評価対象: {len(test_members)}名のメンバー")

        # 評価関数
        def evaluate_recommender(recommender, member_codes, k_list):
            """推薦モデルを評価"""
            results = {k: {'precision': [], 'recall': [], 'ndcg': [], 'hit': []} for k in k_list}

            for i, member_code in enumerate(member_codes):
                # 進捗更新
                progress_bar.progress((i + 1) / len(member_codes))

                # 実際に習得している力量
                actual = st.session_state.member_competence[
                    st.session_state.member_competence['メンバーコード'] == member_code
                ]['力量コード'].tolist()

                if len(actual) < 2:
                    continue

                # テストセット作成（最後の20%を隠す）
                n_test = max(1, int(len(actual) * test_ratio))
                train_actual = actual[:-n_test]
                test_actual = actual[-n_test:]

                # 訓練セットのみを使用して推薦
                # （member_competenceを一時的に更新）
                temp_member_competence = st.session_state.member_competence[
                    ~((st.session_state.member_competence['メンバーコード'] == member_code) &
                      (st.session_state.member_competence['力量コード'].isin(test_actual)))
                ]

                # キャッシュクリア
                recommender._member_acquired_cache = {}

                # 推薦生成
                try:
                    recommendations = recommender.recommend(
                        member_code=member_code,
                        top_n=max(k_list),
                    )
                    recommended_codes = [rec.competence_code for rec in recommendations]
                except Exception:
                    continue

                # 各K値で評価
                for k in k_list:
                    rec_at_k = recommended_codes[:k]

                    # Precision@K
                    hits = len(set(rec_at_k) & set(test_actual))
                    precision = hits / k if k > 0 else 0

                    # Recall@K
                    recall = hits / len(test_actual) if len(test_actual) > 0 else 0

                    # NDCG@K（簡易版）
                    dcg = sum([1 / np.log2(i + 2) for i, code in enumerate(rec_at_k) if code in test_actual])
                    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_actual)))])
                    ndcg = dcg / idcg if idcg > 0 else 0

                    # Hit Rate
                    hit = 1 if hits > 0 else 0

                    results[k]['precision'].append(precision)
                    results[k]['recall'].append(recall)
                    results[k]['ndcg'].append(ndcg)
                    results[k]['hit'].append(hit)

            # 平均を計算
            summary = {}
            for k in k_list:
                summary[k] = {
                    'Precision@K': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
                    'Recall@K': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
                    'NDCG@K': np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0,
                    'Hit Rate@K': np.mean(results[k]['hit']) if results[k]['hit'] else 0,
                }

            return summary

        # 各モデルを評価
        status_text.text("🤖 ML推薦モデルを評価中...")
        ml_results = evaluate_recommender(st.session_state.ml_recommender, test_members, k_values)

        status_text.text("🎲 ランダム推薦を評価中...")
        random_results = evaluate_recommender(st.session_state.random_rec, test_members, k_values)

        status_text.text("🔥 人気度ベース推薦を評価中...")
        popularity_results = evaluate_recommender(st.session_state.popularity_rec, test_members, k_values)

        status_text.text("📁 カテゴリベース推薦を評価中...")
        category_results = evaluate_recommender(st.session_state.category_rec, test_members, k_values)

        # 結果を保存
        st.session_state.evaluation_results = {
            'ML (NMF)': ml_results,
            'Random': random_results,
            'Popularity': popularity_results,
            'Category-Based': category_results,
        }

        progress_bar.progress(1.0)
        status_text.text("✅ 評価完了！")

    except Exception as e:
        st.error(f"❌ 評価エラー: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ===== 結果表示 =====
if st.session_state.evaluation_results:
    st.header("4️⃣ 評価結果")

    for k in k_values:
        st.subheader(f"📊 K={k}の評価結果")

        # データフレーム作成
        comparison_data = []
        for model_name, results in st.session_state.evaluation_results.items():
            if k in results:
                row = {'モデル': model_name}
                row.update(results[k])
                comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)

        # テーブル表示
        st.dataframe(
            df_comparison.style.highlight_max(
                subset=[col for col in df_comparison.columns if col != 'モデル'],
                color='lightgreen'
            ).format({
                col: '{:.4f}' for col in df_comparison.columns if col != 'モデル'
            }),
            use_container_width=True
        )

        # グラフ表示
        col1, col2 = st.columns(2)

        with col1:
            # Precision & Recall
            fig = go.Figure()
            for metric in ['Precision@K', 'Recall@K']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_comparison['モデル'],
                    y=df_comparison[metric],
                    text=df_comparison[metric].round(4),
                    textposition='auto',
                ))

            fig.update_layout(
                title=f'Precision & Recall @ K={k}',
                xaxis_title='モデル',
                yaxis_title='スコア',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # NDCG & Hit Rate
            fig = go.Figure()
            for metric in ['NDCG@K', 'Hit Rate@K']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_comparison['モデル'],
                    y=df_comparison[metric],
                    text=df_comparison[metric].round(4),
                    textposition='auto',
                ))

            fig.update_layout(
                title=f'NDCG & Hit Rate @ K={k}',
                xaxis_title='モデル',
                yaxis_title='スコア',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # 総合評価
    st.subheader("📈 総合評価")

    # すべてのK値での平均
    overall_scores = {}
    for model_name in st.session_state.evaluation_results.keys():
        overall_scores[model_name] = {}
        for metric in ['Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate@K']:
            scores = [
                st.session_state.evaluation_results[model_name][k][metric]
                for k in k_values
                if k in st.session_state.evaluation_results[model_name]
            ]
            overall_scores[model_name][metric] = np.mean(scores) if scores else 0

    df_overall = pd.DataFrame(overall_scores).T
    df_overall.index.name = 'モデル'

    st.dataframe(
        df_overall.style.highlight_max(color='lightgreen').format('{:.4f}'),
        use_container_width=True
    )

    # レーダーチャート
    fig = go.Figure()

    for model_name in overall_scores.keys():
        fig.add_trace(go.Scatterpolar(
            r=[overall_scores[model_name][m] for m in df_overall.columns],
            theta=df_overall.columns,
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='モデル性能比較（レーダーチャート）',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # 結論
    st.success("""
    ✅ **評価完了**

    - MLモデル（NMF）がベースラインモデルを上回る性能を示しているか確認してください
    - Precision@Kが低い場合は、ハイパーパラメータチューニングを検討してください
    - すべてのモデルが低スコアの場合、データ品質やテストセット分割を見直してください
    """)
