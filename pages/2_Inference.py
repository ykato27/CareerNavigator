"""
キャリア推薦システム - 推論
"""

from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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
# データ読み込みチェック
# =========================================================
if not st.session_state.get("data_loaded", False):
    st.warning("⚠️ まずデータを読み込んでください。")
    st.info("👉 サイドバーから「データ読み込み」ページに戻ってCSVファイルをアップロードしてください。")
    st.stop()


# =========================================================
# モデル学習チェック
# =========================================================
if not st.session_state.get("model_trained", False):
    st.warning("⚠️ まずMLモデルを学習してください。")
    st.info("👉 サイドバーから「モデル学習」ページに移動して、MLモデルを学習してください。")
    st.stop()


# =========================================================
# 補助関数
# =========================================================
def convert_recommendations_to_dataframe(recommendations) -> pd.DataFrame:
    """
    Recommendationオブジェクトのリストを表示用/ダウンロード用のDataFrameに変換する
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
    df = df[cols]
    return df


def create_member_positioning_data(member_competence, member_master, mf_model):
    """
    全メンバーの位置データを作成

    Returns:
        DataFrame with columns: メンバーコード, メンバー名, 総合スキルレベル,
        保有力量数, 平均レベル, 潜在因子1, 潜在因子2
    """
    data = []

    for member_code in member_master["メンバーコード"]:
        # 保有力量データを取得
        member_comp = member_competence[member_competence["メンバーコード"] == member_code]

        if len(member_comp) == 0:
            continue

        # メンバー名
        member_name = member_master[member_master["メンバーコード"] == member_code]["メンバー名"].values[0]

        # スキルメトリクス
        total_level = member_comp["正規化レベル"].sum()
        competence_count = len(member_comp)
        avg_level = member_comp["正規化レベル"].mean()

        # 潜在因子（NMFモデルから）
        latent_factor_1 = 0
        latent_factor_2 = 0
        if member_code in mf_model.member_index:
            member_idx = mf_model.member_index[member_code]
            latent_factor_1 = mf_model.W[member_idx, 0] if mf_model.n_components > 0 else 0
            latent_factor_2 = mf_model.W[member_idx, 1] if mf_model.n_components > 1 else 0

        data.append({
            "メンバーコード": member_code,
            "メンバー名": member_name,
            "総合スキルレベル": total_level,
            "保有力量数": competence_count,
            "平均レベル": avg_level,
            "潜在因子1": latent_factor_1,
            "潜在因子2": latent_factor_2
        })

    return pd.DataFrame(data)


def create_positioning_plot(position_df, target_member_code, reference_person_codes,
                            x_col, y_col, title):
    """
    メンバーポジショニングプロットを作成

    Args:
        position_df: メンバー位置データ
        target_member_code: 対象メンバーコード
        reference_person_codes: 参考人物のコードリスト
        x_col: X軸に使用する列名
        y_col: Y軸に使用する列名
        title: グラフタイトル
    """
    # メンバータイプを分類
    df = position_df.copy()
    df["メンバータイプ"] = "その他"
    df.loc[df["メンバーコード"] == target_member_code, "メンバータイプ"] = "あなた"
    df.loc[df["メンバーコード"].isin(reference_person_codes), "メンバータイプ"] = "参考人物"

    # 色とサイズのマッピング
    color_map = {
        "あなた": "#FF4B4B",
        "参考人物": "#4B8BFF",
        "その他": "#CCCCCC"
    }

    size_map = {
        "あなた": 20,
        "参考人物": 15,
        "その他": 8
    }

    df["color"] = df["メンバータイプ"].map(color_map)
    df["size"] = df["メンバータイプ"].map(size_map)

    # プロット順序を調整（あなた→参考人物→その他の順で描画）
    df["plot_order"] = df["メンバータイプ"].map({"その他": 1, "参考人物": 2, "あなた": 3})
    df = df.sort_values("plot_order")

    # Plotlyで散布図を作成
    fig = go.Figure()

    for member_type in ["その他", "参考人物", "あなた"]:
        df_subset = df[df["メンバータイプ"] == member_type]

        fig.add_trace(go.Scatter(
            x=df_subset[x_col],
            y=df_subset[y_col],
            mode="markers",
            name=member_type,
            marker=dict(
                size=df_subset["size"],
                color=df_subset["color"],
                line=dict(width=1, color="white")
            ),
            text=df_subset["メンバー名"],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{x_col}: %{{x:.1f}}<br>" +
                f"{y_col}: %{{y:.2f}}<br>" +
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


# =========================================================
# 推論対象会員の選択
# =========================================================
st.subheader("👤 推論対象会員の選択")

td = st.session_state.transformed_data
members_df = td["members_clean"]
recommender = st.session_state.ml_recommender
mf_model = recommender.mf_model

# 学習データに存在する会員のみをフィルタ（コールドスタート問題を回避）
trained_member_codes = set(mf_model.member_index)
available_members = members_df[members_df["メンバーコード"].isin(trained_member_codes)]

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
# 推論設定
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
    competence_type = st.selectbox(
        "力量タイプフィルタ",
        options=["全て", "SKILL", "EDUCATION", "LICENSE"]
    )
    if competence_type == "全て":
        competence_type = None

with col3:
    diversity_strategy = st.selectbox(
        "多様性戦略",
        options=["hybrid", "mmr", "category", "type"],
        index=0
    )


# =========================================================
# 推論実行
# =========================================================
st.subheader("🚀 推論実行")

if st.button("推薦を実行", type="primary"):
    with st.spinner("推薦を生成中..."):
        try:
            recs = recommender.recommend(
                member_code=selected_member_code,
                top_n=top_n,
                competence_type=competence_type,
                category_filter=None,
                use_diversity=True,
                diversity_strategy=diversity_strategy
            )

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
                    with st.expander(f"🎯 推薦 {idx}: {rec.competence_name} (優先度: {rec.priority_score:.1f})"):
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

                # テーブル表示（ダウンロード用）
                st.markdown("---")
                st.markdown("### 📊 推薦結果一覧")
                st.dataframe(df_result, use_container_width=True)

        except Exception as e:
            import traceback

            # ColdStartErrorを個別に処理
            from skillnote_recommendation.ml.exceptions import ColdStartError, MLModelNotTrainedError

            if isinstance(e, ColdStartError):
                st.error(f"❌ コールドスタート問題が発生しました")
                st.warning(
                    f"**会員コード `{e.member_code}` の保有力量が登録されていないため、ML推薦ができません。**\n\n"
                    f"**原因:**\n"
                    f"- この会員の力量データがMLモデルの学習データに含まれていません。\n\n"
                    f"**対処方法:**\n"
                    f"1. この会員の力量データ（習得済み力量）を登録してください\n"
                    f"2. データ登録後、「モデル学習」ページで再学習してください\n"
                    f"3. 再学習後、再度推薦を実行してください"
                )
            elif isinstance(e, MLModelNotTrainedError):
                st.error("❌ MLモデルが学習されていません")
                st.info("「モデル学習」ページでMLモデルを学習してから、推薦を実行してください。")
            else:
                st.error(f"❌ 推薦処理中にエラーが発生しました: {type(e).__name__}: {e}")

                # 詳細なトレースバックを表示
                with st.expander("🔍 詳細なエラー情報を表示"):
                    st.code(traceback.format_exc())

                    st.markdown("### デバッグ情報")
                    st.write("**エラー型:**", type(e).__name__)
                    st.write("**エラーメッセージ:**", str(e))

                    # データ構造の検証
                    if st.session_state.transformed_data:
                        td = st.session_state.transformed_data
                        st.write("**transformed_data のキー:**", list(td.keys()))

                        if "competence_master" in td:
                            comp_master = td["competence_master"]
                            st.write("**competence_master のカラム:**", list(comp_master.columns))
                            st.write("**competence_master のサンプル:**")
                            st.dataframe(comp_master.head(3))

                        if "member_competence" in td:
                            member_comp = td["member_competence"]
                            st.write("**member_competence のカラム:**", list(member_comp.columns))

                    st.info("💡 このエラー情報をスクリーンショットして開発者に共有してください。")


# =========================================================
# 推薦結果ダウンロード
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

    # =========================================================
    # メンバーポジショニングマップ
    # =========================================================
    if st.session_state.get("last_recommendations") is not None:
        st.markdown("---")
        st.subheader("🗺️ メンバーポジショニングマップ")
        st.markdown(
            "あなたと参考人物が、全メンバーの中でどの位置にいるかを可視化します。\n"
            "**赤色**があなた、**青色**が参考人物、**灰色**がその他のメンバーです。"
        )

        # 参考人物のコードを収集
        reference_person_codes = []
        for rec in st.session_state.last_recommendations:
            if rec.reference_persons:
                for ref_person in rec.reference_persons:
                    if ref_person.member_code not in reference_person_codes:
                        reference_person_codes.append(ref_person.member_code)

        # ポジショニングデータを作成
        position_df = create_member_positioning_data(
            td["member_competence"],
            td["members_clean"],
            recommender.mf_model
        )

        target_code = st.session_state.last_target_member_code

        # 複数のビューを表示
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
                position_df, target_code, reference_person_codes,
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
                position_df, target_code, reference_person_codes,
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
                position_df, target_code, reference_person_codes,
                "潜在因子1", "潜在因子2",
                "潜在因子空間でのメンバー分布"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            st.markdown("### 全メンバーのデータ")
            # あなたと参考人物を強調表示
            display_df = position_df.copy()
            display_df["タイプ"] = "その他"
            display_df.loc[display_df["メンバーコード"] == target_code, "タイプ"] = "あなた"
            display_df.loc[display_df["メンバーコード"].isin(reference_person_codes), "タイプ"] = "参考人物"

            # タイプでソート（あなた→参考人物→その他）
            display_df["sort_order"] = display_df["タイプ"].map({"あなた": 0, "参考人物": 1, "その他": 2})
            display_df = display_df.sort_values(["sort_order", "総合スキルレベル"], ascending=[True, False])
            display_df = display_df.drop(columns=["sort_order"])

            # 列の順序を調整
            cols = ["タイプ", "メンバー名", "総合スキルレベル", "保有力量数", "平均レベル", "潜在因子1", "潜在因子2"]
            display_df = display_df[cols]

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
