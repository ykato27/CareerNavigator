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


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="キャリア推薦システム - 推論",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 推論実行")
st.markdown("**ステップ3**: 学習済みMLモデルを使用して、会員への力量推薦を実行します。")


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

st.subheader("👤 推論対象会員の選択")

# 学習データに存在する会員のみをフィルタ（コールドスタート問題を回避）
trained_member_codes = set(mf_model.member_codes)
available_members = members_df[
    members_df["メンバーコード"].isin(trained_member_codes)
]

if len(available_members) == 0:
    st.error("❌ 推論可能な会員が存在しません。")
    st.stop()

st.info(
    f"📊 推論可能な会員数: {len(available_members)} / {len(members_df)} 名\n\n"
    f"💡 **コールドスタート問題の回避**: 学習データに含まれる会員のみが選択可能です。\n"
    f"保有力量が未登録の会員は、データ登録後にモデルを再学習してください。"
)

# メンバー選択プルダウン
member_options = dict(
    zip(available_members["メンバーコード"], available_members["メンバー名"])
)

selected_member_code = st.selectbox(
    "推論対象会員を選択してください",
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

with col3:
    diversity_strategy = st.selectbox(
        "多様性戦略",
        options=["hybrid", "mmr", "category", "type"],
        index=0,
        help="推薦結果の多様性を確保する戦略を選択"
    )


# =========================================================
# 推論実行
# =========================================================

st.subheader("🚀 推論実行")

if st.button("推薦を実行", type="primary"):
    with st.spinner("推薦を生成中..."):
        try:
            # 推薦を実行
            recs = recommender.recommend(
                member_code=selected_member_code,
                top_n=top_n,
                competence_type=competence_type,
                category_filter=None,
                use_diversity=True,
                diversity_strategy=diversity_strategy
            )

            # セッション状態に保存
            if not recs:
                st.warning("推薦できる力量がありません。")
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
                    f"**会員コード `{e.member_code}` の保有力量が登録されていないため、"
                    f"ML推薦ができません。**\n\n"
                    f"**原因:**\n"
                    f"- この会員の力量データがMLモデルの学習データに含まれていません。\n\n"
                    f"**対処方法:**\n"
                    f"1. この会員の力量データ（保有力量）を登録してください\n"
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
