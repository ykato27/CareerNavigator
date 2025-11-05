"""
キャリア推薦システム - モデル学習と分析
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skillnote_recommendation.ml.ml_recommender import MLRecommender


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="キャリア推薦システム - モデル学習",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 モデル学習と分析")
st.markdown("**ステップ2**: MLモデルを学習し、学習結果を分析します。")


# =========================================================
# データ読み込みチェック
# =========================================================
if not st.session_state.get("data_loaded", False):
    st.warning("⚠️ まずデータを読み込んでください。")
    st.info("👉 サイドバーから「データ読み込み」ページに戻ってCSVファイルをアップロードしてください。")
    st.stop()


# =========================================================
# 補助関数
# =========================================================
def build_ml_recommender(transformed_data: dict) -> MLRecommender:
    """
    MLRecommenderを学習済みの状態で作成する
    """
    recommender = MLRecommender.build(
        member_competence=transformed_data["member_competence"],
        competence_master=transformed_data["competence_master"],
        member_master=transformed_data["members_clean"]
    )
    return recommender


# =========================================================
# モデル学習
# =========================================================
st.subheader("🎓 MLモデル学習")

if st.session_state.get("model_trained", False):
    st.success("✅ MLモデルは既に学習済みです。")

    if st.button("🔄 モデルを再学習する"):
        st.session_state.model_trained = False
        st.session_state.ml_recommender = None
        st.rerun()
else:
    st.info("📚 NMF（非負値行列分解）を使用して、メンバーの力量習得パターンを学習します。")

    if st.button("🚀 MLモデル学習を実行", type="primary"):
        with st.spinner("MLモデルを学習中..."):
            try:
                ml_recommender = build_ml_recommender(
                    st.session_state.transformed_data
                )
                st.session_state.ml_recommender = ml_recommender
                st.session_state.model_trained = True
                st.success("✅ MLモデル学習が完了しました。")
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"❌ エラーが発生しました: {type(e).__name__}: {e}")
                st.code(traceback.format_exc())
                st.info("デバッグ情報:")
                st.write("transformed_data keys:", list(st.session_state.transformed_data.keys()))


# =========================================================
# 学習結果の分析
# =========================================================
if st.session_state.get("model_trained", False):
    st.markdown("---")
    st.subheader("📊 学習結果の分析")

    recommender = st.session_state.ml_recommender
    mf_model = recommender.mf_model

    # 基本統計
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("潜在因子数", mf_model.n_components)

    with col2:
        st.metric("メンバー数", len(mf_model.member_index))

    with col3:
        st.metric("力量数", len(mf_model.competence_index))

    with col4:
        error = mf_model.get_reconstruction_error()
        st.metric("再構成誤差", f"{error:.4f}")

    # NMF成分の分析
    st.markdown("### 🔍 NMF潜在因子の分析")

    st.markdown(
        "NMFはメンバー×力量マトリクスを**メンバー因子行列**と**力量因子行列**に分解します。\n"
        "各潜在因子は、特定の力量群（スキルセット）を表し、メンバーはこれらの因子の組み合わせで表現されます。"
    )

    # 各潜在因子の特徴を分析
    with st.expander("📈 潜在因子ごとの代表力量（トップ10）"):
        competence_master = st.session_state.transformed_data["competence_master"]

        n_factors_to_show = min(5, mf_model.n_components)

        for factor_idx in range(n_factors_to_show):
            st.markdown(f"#### 潜在因子 {factor_idx + 1}")

            # この因子で重みが高い力量を取得
            factor_weights = mf_model.H[factor_idx, :]
            top_indices = factor_weights.argsort()[-10:][::-1]
            top_competences = [mf_model.competence_codes[i] for i in top_indices]
            top_weights = [factor_weights[i] for i in top_indices]

            # 力量名を取得
            top_competence_names = []
            for comp_code in top_competences:
                comp_info = competence_master[competence_master["力量コード"] == comp_code]
                if len(comp_info) > 0:
                    top_competence_names.append(comp_info.iloc[0]["力量名"])
                else:
                    top_competence_names.append(comp_code)

            # データフレームで表示
            df_factor = pd.DataFrame({
                "力量名": top_competence_names,
                "重み": top_weights
            })

            col1, col2 = st.columns([2, 1])

            with col1:
                # 棒グラフ
                fig = px.bar(
                    df_factor,
                    x="重み",
                    y="力量名",
                    orientation="h",
                    title=f"潜在因子 {factor_idx + 1} の代表力量"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # テーブル
                st.dataframe(df_factor, use_container_width=True, height=400)

    # メンバーの潜在因子分布
    with st.expander("👥 メンバーの潜在因子分布"):
        st.markdown("各メンバーがどの潜在因子を強く持っているかを示します。")

        # ランダムに10名をサンプル
        import numpy as np

        n_members_to_show = min(10, len(mf_model.member_codes))
        random_indices = np.random.choice(len(mf_model.member_codes), n_members_to_show, replace=False)

        member_codes = [mf_model.member_codes[i] for i in random_indices]
        member_names = []
        members_df = st.session_state.transformed_data["members_clean"]
        for code in member_codes:
            member_info = members_df[members_df["メンバーコード"] == code]
            if len(member_info) > 0:
                member_names.append(member_info.iloc[0]["メンバー名"])
            else:
                member_names.append(code)

        # 各メンバーの潜在因子の重みを取得
        member_factors_data = []
        for i, (idx, member_code) in enumerate(zip(random_indices, member_codes)):
            factors = mf_model.W[idx, :]
            for factor_idx, weight in enumerate(factors):
                member_factors_data.append({
                    "メンバー": member_names[i],  # インデックスで直接参照
                    "潜在因子": f"因子{factor_idx + 1}",
                    "重み": weight
                })

        df_member_factors = pd.DataFrame(member_factors_data)

        # 重複チェックとデバッグ情報
        duplicates = df_member_factors[df_member_factors.duplicated(subset=["メンバー", "潜在因子"], keep=False)]
        if not duplicates.empty:
            st.warning(f"⚠️ 重複データが検出されました（{len(duplicates)}件）。重複を削除します。")
            df_member_factors = df_member_factors.drop_duplicates(subset=["メンバー", "潜在因子"], keep="first")

        # ヒートマップ
        pivot_table = df_member_factors.pivot_table(
            index="メンバー",
            columns="潜在因子",
            values="重み",
            aggfunc="mean"  # 万が一重複がある場合は平均を取る
        )

        fig = px.imshow(
            pivot_table,
            labels=dict(x="潜在因子", y="メンバー", color="重み"),
            title="メンバーの潜在因子分布ヒートマップ",
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # 力量の潜在因子分布
    with st.expander("💡 力量の潜在因子分布"):
        st.markdown("各力量がどの潜在因子に関連しているかを示します。")

        # ランダムに10個の力量をサンプル
        n_competences_to_show = min(10, len(mf_model.competence_codes))
        random_comp_indices = np.random.choice(len(mf_model.competence_codes), n_competences_to_show, replace=False)

        competence_codes = [mf_model.competence_codes[i] for i in random_comp_indices]
        competence_names = []
        for code in competence_codes:
            comp_info = competence_master[competence_master["力量コード"] == code]
            if len(comp_info) > 0:
                competence_names.append(comp_info.iloc[0]["力量名"])
            else:
                competence_names.append(code)

        # 各力量の潜在因子の重みを取得
        competence_factors_data = []
        for i, (idx, comp_code) in enumerate(zip(random_comp_indices, competence_codes)):
            factors = mf_model.H[:, idx]
            for factor_idx, weight in enumerate(factors):
                competence_factors_data.append({
                    "力量": competence_names[i],  # インデックスで直接参照
                    "潜在因子": f"因子{factor_idx + 1}",
                    "重み": weight
                })

        df_competence_factors = pd.DataFrame(competence_factors_data)

        # 重複チェックとデバッグ情報
        duplicates_comp = df_competence_factors[df_competence_factors.duplicated(subset=["力量", "潜在因子"], keep=False)]
        if not duplicates_comp.empty:
            st.warning(f"⚠️ 重複データが検出されました（{len(duplicates_comp)}件）。重複を削除します。")
            df_competence_factors = df_competence_factors.drop_duplicates(subset=["力量", "潜在因子"], keep="first")

        # ヒートマップ
        pivot_table_comp = df_competence_factors.pivot_table(
            index="力量",
            columns="潜在因子",
            values="重み",
            aggfunc="mean"  # 万が一重複がある場合は平均を取る
        )

        fig = px.imshow(
            pivot_table_comp,
            labels=dict(x="潜在因子", y="力量", color="重み"),
            title="力量の潜在因子分布ヒートマップ",
            color_continuous_scale="Greens"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # モデル評価指標
    with st.expander("📉 モデル評価指標"):
        st.markdown("### 再構成誤差の詳細")

        error = mf_model.get_reconstruction_error()

        st.metric("再構成誤差（Frobenius ノルム）", f"{error:.6f}")

        st.markdown(
            "**再構成誤差が低いほど、モデルは元のデータをよく再現できています。**\n\n"
            "- 誤差が0.1以下: 非常に良好\n"
            "- 誤差が0.1-0.3: 良好\n"
            "- 誤差が0.3-0.5: 許容範囲\n"
            "- 誤差が0.5以上: 改善の余地あり（潜在因子数の調整を推奨）"
        )

    st.markdown("---")
    st.success("✅ 学習結果の分析が完了しました。")
    st.info("👉 サイドバーから「推論」ページに移動して、推薦を実行してください。")
