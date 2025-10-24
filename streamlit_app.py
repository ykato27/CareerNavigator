"""
キャリア推薦システム Streamlitアプリ
- CSVアップロード（カテゴリ別・複数ファイル対応）
- データ読み込みと変換
- MLモデル学習（NMF）
- 推薦実行
- 推薦結果のダウンロード
"""

import os
import tempfile
from io import StringIO

import streamlit as st
import pandas as pd

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.ml.ml_recommender import MLRecommender


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="キャリア推薦システム",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 キャリア推薦システム")
st.markdown("ルールベースと機械学習による力量推薦のためのインターフェース")


# =========================================================
# セッション初期化
# =========================================================
def _init_session_state():
    defaults = {
        "data_loaded": False,
        "model_trained": False,
        "raw_data": None,
        "transformed_data": None,
        "ml_recommender": None,
        "temp_dir": None,
        "last_recommendations_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# =========================================================
# 補助関数
# =========================================================
def load_csv_to_memory(uploaded_file: "UploadedFile") -> pd.DataFrame:
    """アップロードされた単一CSVファイルをDataFrameに読み込む"""
    return pd.read_csv(uploaded_file, encoding="utf-8-sig")


def save_uploaded_files(temp_dir: str, subdir_name: str, uploaded_files):
    """
    指定カテゴリ用のサブディレクトリを作成し
    その中に複数のCSVをUTF-8で保存する
    """
    category_dir = os.path.join(temp_dir, subdir_name)
    os.makedirs(category_dir, exist_ok=True)

    for i, file in enumerate(uploaded_files):
        df = load_csv_to_memory(file)
        filename_base = os.path.splitext(file.name)[0]
        out_path = os.path.join(category_dir, f"{filename_base}_{i+1}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")


def create_temp_dir_with_csv(uploaded_dict: dict) -> str:
    """
    カテゴリごとの複数CSVを一時ディレクトリ下に保存する
    uploaded_dictのキーは 'members' 'skills' 'education' 'license' 'categories' 'acquired'
    """
    temp_dir = tempfile.mkdtemp()
    for category, files in uploaded_dict.items():
        if files:
            save_uploaded_files(temp_dir, category, files)
    return temp_dir


def build_transformed_data(raw_data: dict) -> dict:
    """
    DataTransformerを使い
    推薦や学習に使う形式にまとめる
    """
    transformer = DataTransformer()

    competence_master = transformer.create_competence_master(raw_data)
    member_competence, valid_members = transformer.create_member_competence(
        raw_data,
        competence_master
    )
    skill_matrix = transformer.create_skill_matrix(member_competence)
    members_clean = transformer.clean_members_data(raw_data)

    transformed_data = {
        "competence_master": competence_master,
        "member_competence": member_competence,
        "skill_matrix": skill_matrix,
        "members_clean": members_clean,
        "valid_members": valid_members,
    }
    return transformed_data


def build_ml_recommender(transformed_data: dict) -> MLRecommender:
    """
    MLRecommenderを学習済みの状態で作成する
    """
    recommender = MLRecommender.build(
        member_competence=transformed_data["member_competence"],
        competence_master=transformed_data["competence_master"]
    )
    return recommender


def convert_recommendations_to_dataframe(recommendations) -> pd.DataFrame:
    """
    Recommendationオブジェクトのリストを表示用/ダウンロード用のDataFrameに変換する
    """
    if not recommendations:
        return pd.DataFrame()

    # Recommendation.to_dict() がある前提
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


# =========================================================
# アップロードUI
# =========================================================
st.subheader("📁 データアップロード")
st.markdown("カテゴリごとにCSVファイルをドラッグ＆ドロップしてください。複数ファイルに対応します。")

col_left, col_right = st.columns(2)

with col_left:
    uploaded_members = st.file_uploader(
        "👥 メンバー",
        type=["csv"],
        accept_multiple_files=True,
        key="members"
    )
    uploaded_skills = st.file_uploader(
        "🧠 力量（スキル）",
        type=["csv"],
        accept_multiple_files=True,
        key="skills"
    )
    uploaded_education = st.file_uploader(
        "📘 力量（教育）",
        type=["csv"],
        accept_multiple_files=True,
        key="education"
    )

with col_right:
    uploaded_license = st.file_uploader(
        "🎓 力量（資格）",
        type=["csv"],
        accept_multiple_files=True,
        key="license"
    )
    uploaded_categories = st.file_uploader(
        "🗂 力量カテゴリー",
        type=["csv"],
        accept_multiple_files=True,
        key="categories"
    )
    uploaded_acquired = st.file_uploader(
        "📊 保有力量",
        type=["csv"],
        accept_multiple_files=True,
        key="acquired"
    )

st.markdown("---")


# =========================================================
# データ読み込み処理
# =========================================================
st.subheader("📥 データ読み込み")

if st.button("データ読み込みを実行", type="primary"):
    # 全カテゴリが少なくとも1つアップロードされているか確認
    if all([
        uploaded_members,
        uploaded_skills,
        uploaded_education,
        uploaded_license,
        uploaded_categories,
        uploaded_acquired
    ]):
        with st.spinner("データを読み込み・変換中..."):
            try:
                # 一時ディレクトリを作成しアップロード済みファイルを書き出す
                temp_dir = create_temp_dir_with_csv({
                    "members": uploaded_members,
                    "skills": uploaded_skills,
                    "education": uploaded_education,
                    "license": uploaded_license,
                    "categories": uploaded_categories,
                    "acquired": uploaded_acquired
                })

                # DataLoaderで生データを読み込む
                loader = DataLoader(data_dir=temp_dir)
                raw_data = loader.load_all_data()

                # DataTransformerで変換済みデータを構築
                transformed_data = build_transformed_data(raw_data)

                # セッションに保存
                st.session_state.temp_dir = temp_dir
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.data_loaded = True

                st.success("✅ データが正常に読み込まれました。")
                st.session_state.model_trained = False
                st.session_state.ml_recommender = None
                st.session_state.last_recommendations_df = None

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")
    else:
        st.warning("全てのカテゴリで少なくとも1つのCSVファイルをアップロードしてください。")


# ここまででデータ未読込ならガイドを表示
if not st.session_state.data_loaded:
    st.markdown("### 必要なデータ")
    st.markdown(
        "1. メンバー\n"
        "2. 力量（スキル）\n"
        "3. 力量（教育）\n"
        "4. 力量（資格）\n"
        "5. 力量カテゴリー\n"
        "6. 保有力量"
    )


# =========================================================
# MLモデル学習
# =========================================================
if st.session_state.data_loaded:
    st.markdown("---")
    st.subheader("🤖 MLモデル学習")

    if st.button("MLモデル学習を実行"):
        with st.spinner("MLモデルを学習中..."):
            try:
                ml_recommender = build_ml_recommender(
                    st.session_state.transformed_data
                )
                st.session_state.ml_recommender = ml_recommender
                st.session_state.model_trained = True
                st.success("✅ MLモデル学習が完了しました。")
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")


# =========================================================
# 推薦処理と結果表示
# =========================================================
if st.session_state.data_loaded and st.session_state.model_trained and st.session_state.ml_recommender:
    st.markdown("---")
    st.subheader("🎯 推薦処理")

    td = st.session_state.transformed_data
    members_df = td["members_clean"]

    # メンバー選択プルダウン
    member_codes = members_df["メンバーコード"].tolist()
    code_to_name = dict(
        zip(members_df["メンバーコード"], members_df["メンバー名"])
    )

    selected_member_code = st.selectbox(
        "推薦対象のメンバーを選択してください",
        options=member_codes,
        format_func=lambda code: f"{code} : {code_to_name.get(code, '')}"
    )

    # 推薦件数
    top_n = st.slider(
        "推薦件数",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    # 力量タイプフィルタ
    competence_type_ui = st.selectbox(
        "力量タイプフィルタ",
        options=["全て", "SKILL", "EDUCATION", "LICENSE"],
        index=0
    )
    competence_type = None if competence_type_ui == "全て" else competence_type_ui

    # 多様性戦略
    diversity_strategy = st.selectbox(
        "多様性戦略",
        options=["hybrid", "mmr", "category", "type"],
        index=0
    )

    # 推薦実行ボタン
    if st.button("推薦を実行"):
        with st.spinner("推薦を生成中..."):
            try:
                recommender = st.session_state.ml_recommender

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
                else:
                    df_result = convert_recommendations_to_dataframe(recs)
                    st.session_state.last_recommendations_df = df_result

                    st.success(f"{len(df_result)}件の推薦が生成されました。")
                    st.dataframe(df_result, use_container_width=True)

            except Exception as e:
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
                        f"2. データ登録後、「MLモデル学習を実行」ボタンをクリックして再学習してください\n"
                        f"3. 再学習後、再度推薦を実行してください\n\n"
                        f"**または:**\n"
                        f"- ルールベース推薦（ML以外）をお試しください（新規会員でも利用可能）"
                    )
                elif isinstance(e, MLModelNotTrainedError):
                    st.error("❌ MLモデルが学習されていません")
                    st.info("「MLモデル学習を実行」ボタンをクリックしてから、推薦を実行してください。")
                else:
                    st.error(f"❌ 推薦処理中にエラーが発生しました: {e}")


# =========================================================
# 推薦結果ダウンロード
# =========================================================
if st.session_state.last_recommendations_df is not None:
    st.markdown("---")
    st.subheader("💾 推薦結果のダウンロード")

    csv_buffer = StringIO()
    st.session_state.last_recommendations_df.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="推薦結果をCSVでダウンロード",
        data=csv_buffer.getvalue(),
        file_name="recommendations.csv",
        mime="text/csv"
    )


# フッタ
st.markdown("---")
st.caption("Generated by CareerNavigator UI")
