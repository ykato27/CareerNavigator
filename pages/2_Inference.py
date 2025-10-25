"""
キャリア推薦システム - 推論ページ

このページでは、学習済みMLモデルを使用して、メンバーへの力量推薦を実行し、
推薦結果の詳細と参考人物の可視化を提供します。

主な機能:
- メンバー選択と推論設定
- 力量推薦の実行
- 推薦理由と参考人物の表示
- メンバーポジショニングマップの可視化
- 推薦結果のCSVダウンロード
"""

from io import StringIO
from typing import List

import streamlit as st
import pandas as pd

from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    check_model_trained,
    display_error_details,
)
from skillnote_recommendation.utils.visualization import (
    create_member_positioning_data,
    create_positioning_plot,
    prepare_positioning_display_dataframe,
)
from skillnote_recommendation.core.models import Recommendation


# =========================================================
# ヘルパー関数
# =========================================================

def convert_hybrid_to_recommendation(hybrid_rec) -> Recommendation:
    """
    HybridRecommendationを標準のRecommendationオブジェクトに変換

    Args:
        hybrid_rec: HybridRecommendationオブジェクト

    Returns:
        Recommendationオブジェクト
    """
    return Recommendation(
        competence_code=hybrid_rec.competence_code,
        competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code),
        competence_type=hybrid_rec.competence_info.get('力量タイプ', 'UNKNOWN'),
        category=hybrid_rec.competence_info.get('カテゴリー', ''),
        priority_score=hybrid_rec.score,
        category_importance=0.5,  # デフォルト値
        acquisition_ease=0.5,  # デフォルト値
        popularity=0.5,  # デフォルト値
        reason=', '.join(hybrid_rec.reasons) if hybrid_rec.reasons else 'グラフベース推薦',
        reference_persons=[]
    )


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="キャリア推薦システム - 推論",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 推論実行")
st.markdown("**ステップ3**: 学習済みMLモデルを使用して、メンバーへの力量推薦を実行します。")


# =========================================================
# 前提条件チェック
# =========================================================

check_data_loaded()
check_model_trained()


# =========================================================
# データ準備
# =========================================================

td = st.session_state.transformed_data
members_df = td["members_clean"]
recommender = st.session_state.ml_recommender
mf_model = recommender.mf_model


# =========================================================
# ヘルパー関数
# =========================================================

def convert_recommendations_to_dataframe(recommendations) -> pd.DataFrame:
    """
    Recommendationオブジェクトのリストを表示用/ダウンロード用のDataFrameに変換する。

    Args:
        recommendations: Recommendationオブジェクトのリスト

    Returns:
        推薦結果のDataFrame（順位列付き）
    """
    if not recommendations:
        return pd.DataFrame()

    rows = []
    for rank, rec in enumerate(recommendations, start=1):
        rec_dict = rec.to_dict()
        rec_dict["順位"] = rank
        rows.append(rec_dict)

    # 順位を先頭列にする
    df = pd.DataFrame(rows)
    cols = ["順位"] + [c for c in df.columns if c != "順位"]
    return df[cols]


def get_reference_person_codes(recommendations) -> List[str]:
    """
    推薦結果から参考人物のコードリストを抽出する。

    Args:
        recommendations: Recommendationオブジェクトのリスト

    Returns:
        ユニークな参考人物コードのリスト
    """
    reference_codes = []
    for rec in recommendations:
        if rec.reference_persons:
            for ref_person in rec.reference_persons:
                if ref_person.member_code not in reference_codes:
                    reference_codes.append(ref_person.member_code)
    return reference_codes


def display_reference_person(ref_person):
    """
    参考人物の情報を表示する。

    Args:
        ref_person: ReferencePersonオブジェクト
    """
    # 参考タイプのアイコンとラベル
    if ref_person.reference_type == "similar_career":
        st.markdown("#### 🤝 類似キャリア")
    elif ref_person.reference_type == "role_model":
        st.markdown("#### ⭐ ロールモデル")
    else:
        st.markdown("#### 🌟 異なるキャリアパス")

    st.markdown(f"**{ref_person.member_name}さん**")
    st.markdown(ref_person.reason)

    # 差分分析を表示
    st.markdown("**📊 力量の比較**")
    st.metric("共通力量", f"{len(ref_person.common_competences)}個")
    st.metric("参考力量", f"{len(ref_person.unique_competences)}個")
    st.metric("類似度", f"{int(ref_person.similarity_score * 100)}%")


def display_recommendation_details(rec, idx: int):
    """
    推薦結果の詳細を展開可能なセクションで表示する。

    Args:
        rec: Recommendationオブジェクト
        idx: 推薦順位
    """
    with st.expander(
        f"🎯 推薦 {idx}: {rec.competence_name} (優先度: {rec.priority_score:.1f})"
    ):
        # 推薦理由
        st.markdown("### 📋 推薦理由")
        st.markdown(rec.reason)

        # 参考人物
        if rec.reference_persons:
            st.markdown("---")
            st.markdown("### 👥 参考になる人物")

            cols = st.columns(len(rec.reference_persons))
            for col_idx, ref_person in enumerate(rec.reference_persons):
                with cols[col_idx]:
                    display_reference_person(ref_person)


def display_positioning_maps(
    position_df: pd.DataFrame,
    target_code: str,
    reference_codes: List[str]
):
    """
    メンバーポジショニングマップを複数のタブで表示する。

    Args:
        position_df: メンバー位置データ
        target_code: 対象メンバーコード
        reference_codes: 参考人物コードのリスト
    """
    st.markdown("---")
    st.subheader("🗺️ メンバーポジショニングマップ")
    st.markdown(
        "あなたと参考人物が、全メンバーの中でどの位置にいるかを可視化します。\n"
        "**赤色**があなた、**青色**が参考人物、**灰色**がその他のメンバーです。"
    )

    # タブを作成
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 スキルレベル vs 保有力量数",
        "📈 平均レベル vs 保有力量数",
        "🔮 潜在因子マップ",
        "📋 データテーブル"
    ])

    with tab1:
        st.markdown("### 総合スキルレベル vs 保有力量数")
        st.markdown(
            "**X軸**: 総合スキルレベル（全保有力量の正規化レベルの合計）\n\n"
            "**Y軸**: 保有力量数\n\n"
            "右上に行くほど、多くの力量を高いレベルで保有していることを示します。"
        )
        fig1 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "総合スキルレベル", "保有力量数",
            "総合スキルレベル vs 保有力量数"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("### 平均レベル vs 保有力量数")
        st.markdown(
            "**X軸**: 保有力量数（スキルの幅）\n\n"
            "**Y軸**: 平均レベル（スキルの深さ）\n\n"
            "右上に行くほど、幅広い力量を深く習得していることを示します。"
        )
        fig2 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "保有力量数", "平均レベル",
            "スキルの幅 vs 深さ"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 潜在因子マップ（NMF空間）")
        st.markdown(
            "**X軸**: 潜在因子1（第1スキルパターン）\n\n"
            "**Y軸**: 潜在因子2（第2スキルパターン）\n\n"
            "NMFモデルが学習したスキルパターンの空間で、メンバーを配置します。\n"
            "近くにいる人は似たスキルパターンを持っています。"
        )
        fig3 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "潜在因子1", "潜在因子2",
            "潜在因子空間でのメンバー分布"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### 全メンバーのデータ")
        display_df = prepare_positioning_display_dataframe(
            position_df, target_code, reference_codes
        )
        st.dataframe(display_df, use_container_width=True, height=400)


# =========================================================
# メンバー選択UI
# =========================================================

st.subheader("👤 推論対象メンバーの選択")

# 学習データに存在するメンバーのみをフィルタ（コールドスタート問題を回避）
trained_member_codes = set(mf_model.member_codes)
available_members = members_df[
    members_df["メンバーコード"].isin(trained_member_codes)
]

if len(available_members) == 0:
    st.error("❌ 推論可能なメンバーが存在しません。")
    st.stop()

st.info(
    f"📊 推論可能なメンバー数: {len(available_members)} / {len(members_df)} 名\n\n"
    f"💡 **コールドスタート問題の回避**: 学習データに含まれるメンバーのみが選択可能です。\n"
    f"保有力量が未登録のメンバーは、データ登録後にモデルを再学習してください。"
)

# メンバー選択プルダウン
member_options = dict(
    zip(available_members["メンバーコード"], available_members["メンバー名"])
)

selected_member_code = st.selectbox(
    "推論対象メンバーを選択してください",
    options=list(member_options.keys()),
    format_func=lambda x: f"{member_options[x]} ({x})"
)


# =========================================================
# 推論設定UI
# =========================================================

st.subheader("⚙️ 推論設定")

col1, col2, col3 = st.columns(3)

with col1:
    top_n = st.slider(
        "推薦数",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

with col2:
    st.markdown("**力量タイプフィルタ**")
    selected_types = st.multiselect(
        "推薦する力量タイプを選択してください",
        options=["SKILL", "EDUCATION", "LICENSE"],
        default=["SKILL", "EDUCATION", "LICENSE"],
        help="複数選択可能。例: スキルのみ、スキルと教育、など"
    )

    # 空リストの場合はNoneに変換（全てを推薦）
    competence_type = selected_types if selected_types else None

    # 選択が空の場合の警告
    if not selected_types:
        st.warning("⚠️ 力量タイプが選択されていません。全てのタイプから推薦します。")
    else:
        # 選択されたタイプを確認表示
        st.caption(f"選択中: {', '.join(selected_types)}")

with col3:
    diversity_strategy = st.selectbox(
        "多様性戦略",
        options=["hybrid", "mmr", "category", "type"],
        index=0,
        help="推薦結果の多様性を確保する戦略を選択"
    )

# =========================================================
# グラフベース推薦設定
# =========================================================

st.markdown("---")
st.subheader("🔗 グラフベース推薦（実験的機能）")

use_graph_recommendation = st.checkbox(
    "グラフベース推薦を使用する",
    value=False,
    help="Random Walk with Restart (RWR) とNMFを組み合わせたハイブリッド推薦を使用します。推薦パスも可視化されます。"
)

# デフォルト値を設定
rwr_weight = 0.5
show_paths = True

if use_graph_recommendation:
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        rwr_weight = st.slider(
            "グラフベーススコアの重み",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.5 = グラフとNMFを同等に評価、1.0 = グラフのみ、0.0 = NMFのみ"
        )

    with col_g2:
        show_paths = st.checkbox(
            "推薦パスを表示",
            value=True,
            help="推薦理由をグラフで可視化します"
        )

    st.info("💡 グラフベース推薦は、メンバー間の類似性やカテゴリー構造を活用して、より説明可能な推薦を提供します。")


# =========================================================
# 推論実行
# =========================================================

st.subheader("🚀 推論実行")

if st.button("推薦を実行", type="primary"):
    with st.spinner("推薦を生成中..."):
        try:
            # グラフベース推薦を使用する場合
            if use_graph_recommendation:
                from skillnote_recommendation.graph import HybridGraphRecommender

                # HybridGraphRecommenderを初期化
                if 'knowledge_graph' not in st.session_state:
                    st.error("❌ Knowledge Graphが初期化されていません。データ読み込みページで再度データを読み込んでください。")
                    st.stop()

                hybrid_recommender = HybridGraphRecommender(
                    knowledge_graph=st.session_state.knowledge_graph,
                    ml_recommender=recommender,
                    rwr_weight=rwr_weight
                )

                # ハイブリッド推薦を実行
                hybrid_recs = hybrid_recommender.recommend(
                    member_code=selected_member_code,
                    top_n=top_n,
                    competence_type=competence_type,
                    category_filter=None,
                    use_diversity=True
                )

                # HybridRecommendationを標準のRecommendationに変換
                recs = [convert_hybrid_to_recommendation(hr) for hr in hybrid_recs]

                # グラフ推薦情報をセッションに保存
                st.session_state.graph_recommendations = hybrid_recs
                st.session_state.using_graph = True

            else:
                # 通常のNMF推薦を実行
                recs = recommender.recommend(
                    member_code=selected_member_code,
                    top_n=top_n,
                    competence_type=competence_type,
                    category_filter=None,
                    use_diversity=True,
                    diversity_strategy=diversity_strategy
                )
                st.session_state.using_graph = False

            # セッション状態に保存
            if not recs:
                st.warning("⚠️ 推薦できる力量がありません。")

                # 診断情報を表示
                st.info("### 💡 推薦が空になった理由:")

                # 選択された力量タイプを表示
                if competence_type:
                    type_str = "、".join(competence_type) if isinstance(competence_type, list) else competence_type
                    st.write(f"**選択された力量タイプ**: {type_str}")
                else:
                    st.write("**選択された力量タイプ**: 全て")

                # 保有力量の情報を表示
                member_comp = td["member_competence"][
                    td["member_competence"]["メンバーコード"] == selected_member_code
                ]
                acquired_count = len(member_comp)
                st.write(f"**既習得力量数**: {acquired_count}個")

                # タイプ別の保有力量数を表示
                if len(member_comp) > 0:
                    comp_master = td["competence_master"]
                    acquired_codes = member_comp["力量コード"].unique()
                    acquired_info = comp_master[comp_master["力量コード"].isin(acquired_codes)]

                    type_counts = acquired_info["力量タイプ"].value_counts().to_dict()
                    st.write("**タイプ別保有力量数**:")
                    for comp_type, count in type_counts.items():
                        st.write(f"  - {comp_type}: {count}個")

                # 改善案を提示
                st.markdown("### 🔧 改善案:")
                suggestions = []

                if competence_type and len(competence_type) < 3:
                    suggestions.append("- **力量タイプを追加**: 他の力量タイプも選択してみてください")

                if acquired_count > 50:
                    suggestions.append("- **すでに多くの力量を習得**: 新しい分野への挑戦も検討してみてください")

                suggestions.append("- **推薦数を増やす**: スライダーで推薦数を増やしてみてください")
                suggestions.append("- **多様性戦略を変更**: 異なる多様性戦略を試してみてください")

                for suggestion in suggestions:
                    st.write(suggestion)

                st.session_state.last_recommendations_df = None
                st.session_state.last_recommendations = None
                st.session_state.last_target_member_code = None
            else:
                df_result = convert_recommendations_to_dataframe(recs)
                st.session_state.last_recommendations_df = df_result
                st.session_state.last_recommendations = recs
                st.session_state.last_target_member_code = selected_member_code

                st.success(f"{len(df_result)}件の推薦が生成されました。")

                # 推薦結果の詳細表示
                for idx, rec in enumerate(recs, 1):
                    display_recommendation_details(rec, idx)

                # グラフベース推薦の場合、パス可視化を表示
                if use_graph_recommendation and show_paths and st.session_state.get('using_graph'):
                    st.markdown("---")
                    st.markdown("### 🔗 推薦パスの可視化")
                    st.info("グラフ構造に基づく推薦パスを表示しています。ノードをホバーすると詳細が表示されます。")

                    from skillnote_recommendation.graph import RecommendationPathVisualizer

                    visualizer = RecommendationPathVisualizer()
                    graph_recs = st.session_state.get('graph_recommendations', [])

                    # 詳細説明ジェネレーターを初期化
                    from skillnote_recommendation.graph.visualization_utils import (
                        ExplanationGenerator,
                        format_explanation_for_display,
                        export_figure_as_html
                    )

                    category_hierarchy = st.session_state.knowledge_graph.category_hierarchy if st.session_state.get('knowledge_graph') else None
                    explainer = ExplanationGenerator(category_hierarchy=category_hierarchy)

                    # 上位3件のみ可視化
                    for idx, hybrid_rec in enumerate(graph_recs[:3], 1):
                        if hybrid_rec.paths:
                            with st.expander(f"📈 推薦 {idx}: {hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code)}", expanded=(idx==1)):
                                # スコア情報を表示
                                col_s1, col_s2, col_s3 = st.columns(3)
                                with col_s1:
                                    st.metric("総合スコア", f"{hybrid_rec.score:.3f}")
                                with col_s2:
                                    st.metric("グラフスコア", f"{hybrid_rec.rwr_score:.3f}")
                                with col_s3:
                                    st.metric("NMFスコア", f"{hybrid_rec.nmf_score:.3f}")

                                # 詳細説明を生成
                                explanation = explainer.generate_detailed_explanation(
                                    paths=hybrid_rec.paths,
                                    rwr_score=hybrid_rec.rwr_score,
                                    nmf_score=hybrid_rec.nmf_score,
                                    competence_info=hybrid_rec.competence_info
                                )

                                # タブで表示
                                tab1, tab2 = st.tabs(["📊 グラフ可視化", "📝 詳細説明"])

                                with tab1:
                                    # パスを可視化
                                    member_name = members_df[
                                        members_df["メンバーコード"] == selected_member_code
                                    ]["メンバー名"].iloc[0]

                                    fig = visualizer.visualize_recommendation_path(
                                        paths=hybrid_rec.paths,
                                        target_member_name=member_name,
                                        target_competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    # エクスポートボタン
                                    col_e1, col_e2 = st.columns(2)
                                    with col_e1:
                                        if st.button(f"📥 HTMLとしてエクスポート", key=f"export_html_{idx}"):
                                            try:
                                                filename = f"recommendation_path_{hybrid_rec.competence_code}.html"
                                                filepath = export_figure_as_html(fig, filename)
                                                st.success(f"✅ エクスポート完了: {filepath}")
                                            except Exception as e:
                                                st.error(f"エクスポートエラー: {str(e)}")

                                with tab2:
                                    # 詳細説明を表示
                                    formatted_explanation = format_explanation_for_display(explanation)
                                    st.markdown(formatted_explanation)

                # テーブル表示（ダウンロード用）
                st.markdown("---")
                st.markdown("### 📊 推薦結果一覧")
                st.dataframe(df_result, use_container_width=True)

        except Exception as e:
            # エラー処理
            from skillnote_recommendation.ml.exceptions import (
                ColdStartError,
                MLModelNotTrainedError
            )

            if isinstance(e, ColdStartError):
                st.error("❌ コールドスタート問題が発生しました")
                st.warning(
                    f"**メンバーコード `{e.member_code}` の保有力量が登録されていないため、"
                    f"ML推薦ができません。**\n\n"
                    f"**原因:**\n"
                    f"- このメンバーの力量データがMLモデルの学習データに含まれていません。\n\n"
                    f"**対処方法:**\n"
                    f"1. このメンバーの力量データ（保有力量）を登録してください\n"
                    f"2. データ登録後、「モデル学習」ページで再学習してください\n"
                    f"3. 再学習後、再度推薦を実行してください"
                )
            elif isinstance(e, MLModelNotTrainedError):
                st.error("❌ MLモデルが学習されていません")
                st.info(
                    "「モデル学習」ページでMLモデルを学習してから、"
                    "推薦を実行してください。"
                )
            else:
                display_error_details(e, "推薦処理中")


# =========================================================
# 推薦結果のダウンロード & 可視化
# =========================================================

if st.session_state.get("last_recommendations_df") is not None:
    st.markdown("---")
    st.subheader("💾 推薦結果のダウンロード")

    csv_buffer = StringIO()
    st.session_state.last_recommendations_df.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 推薦結果をCSVでダウンロード",
        data=csv_buffer.getvalue(),
        file_name="recommendations.csv",
        mime="text/csv"
    )

    # メンバーポジショニングマップ
    if st.session_state.get("last_recommendations") is not None:
        # 参考人物のコードを収集
        reference_codes = get_reference_person_codes(
            st.session_state.last_recommendations
        )

        # ポジショニングデータを作成
        position_df = create_member_positioning_data(
            td["member_competence"],
            td["members_clean"],
            mf_model
        )

        # 可視化を表示
        display_positioning_maps(
            position_df,
            st.session_state.last_target_member_code,
            reference_codes
        )
