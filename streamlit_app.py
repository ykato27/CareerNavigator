"""
キャリア推薦システム Streamlitアプリ - データ読み込み
"""

import os
import tempfile

import streamlit as st
import pandas as pd

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="キャリア推薦システム - データ読み込み",
    page_icon="📁",
    layout="wide"
)

st.title("📁 データ読み込み")
st.markdown("**ステップ1**: 6種類のCSVファイルをアップロードしてデータを準備します。")


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
def load_csv_to_memory(uploaded_file) -> pd.DataFrame:
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


# =========================================================
# アップロードUI
# =========================================================
st.subheader("📤 CSVファイルのアップロード")

st.info("以下の6種類のCSVファイルをアップロードしてください。")

uploaded_dict = {}

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1️⃣ メンバー")
    uploaded_dict["members"] = st.file_uploader(
        "メンバーマスタ",
        type=["csv"],
        accept_multiple_files=True,
        key="members"
    )

    st.markdown("#### 2️⃣ スキル")
    uploaded_dict["skills"] = st.file_uploader(
        "力量（スキル）マスタ",
        type=["csv"],
        accept_multiple_files=True,
        key="skills"
    )

    st.markdown("#### 3️⃣ 教育")
    uploaded_dict["education"] = st.file_uploader(
        "力量（教育）マスタ",
        type=["csv"],
        accept_multiple_files=True,
        key="education"
    )

with col2:
    st.markdown("#### 4️⃣ 資格")
    uploaded_dict["license"] = st.file_uploader(
        "力量（資格）マスタ",
        type=["csv"],
        accept_multiple_files=True,
        key="license"
    )

    st.markdown("#### 5️⃣ カテゴリー")
    uploaded_dict["categories"] = st.file_uploader(
        "力量カテゴリーマスタ",
        type=["csv"],
        accept_multiple_files=True,
        key="categories"
    )

    st.markdown("#### 6️⃣ 保有力量")
    uploaded_dict["acquired"] = st.file_uploader(
        "保有力量データ",
        type=["csv"],
        accept_multiple_files=True,
        key="acquired"
    )


# =========================================================
# データ読み込みボタン
# =========================================================
all_uploaded = all(uploaded_dict.values())

if all_uploaded:
    if st.button("📥 データを読み込む", type="primary"):
        with st.spinner("データを読み込み中..."):
            try:
                # 一時ディレクトリに保存
                temp_dir = create_temp_dir_with_csv(uploaded_dict)

                # DataLoaderで読み込み
                loader = DataLoader(temp_dir)
                raw_data = loader.load_all_data()

                # DataTransformerで変換済みデータを構築
                transformed_data = build_transformed_data(raw_data)

                # Knowledge Graphを構築
                from skillnote_recommendation.graph import CompetenceKnowledgeGraph

                with st.spinner("Knowledge Graphを構築中..."):
                    knowledge_graph = CompetenceKnowledgeGraph(
                        member_competence=transformed_data['member_competence'],
                        member_master=transformed_data['members_clean'],
                        competence_master=transformed_data['competence_master']
                    )

                # セッションに保存
                st.session_state.temp_dir = temp_dir
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.knowledge_graph = knowledge_graph
                st.session_state.data_loaded = True

                # モデル学習状態をリセット
                st.session_state.model_trained = False
                st.session_state.ml_recommender = None
                st.session_state.last_recommendations_df = None

                st.success("✅ データが正常に読み込まれました。")
                st.info("👉 サイドバーから「モデル学習」ページに移動して、MLモデルを学習してください。")

            except Exception as e:
                import traceback
                st.error(f"❌ データ読み込み中にエラーが発生しました: {type(e).__name__}: {e}")

                # 詳細なトレースバックを表示
                with st.expander("🔍 詳細なエラー情報を表示"):
                    st.code(traceback.format_exc())

                    st.markdown("### デバッグ情報")
                    st.write("**エラー型:**", type(e).__name__)
                    st.write("**エラーメッセージ:**", str(e))

                    # アップロードされたファイルの情報
                    st.markdown("### アップロードされたファイル")
                    for category, files in uploaded_dict.items():
                        if files:
                            st.write(f"**{category}:**", [f.name for f in files])

                    st.info("💡 このエラー情報をスクリーンショットして開発者に共有してください。")
else:
    st.warning("全てのカテゴリで少なくとも1つのCSVファイルをアップロードしてください。")


# =========================================================
# データ読み込み状態の表示
# =========================================================
if st.session_state.data_loaded:
    st.markdown("---")
    st.subheader("📊 読み込まれたデータの概要")

    td = st.session_state.transformed_data

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("メンバー数", len(td["members_clean"]))

    with col2:
        st.metric("力量数", len(td["competence_master"]))

    with col3:
        st.metric("保有力量レコード数", len(td["member_competence"]))

    # データプレビュー
    with st.expander("📋 データプレビュー"):
        st.markdown("#### メンバー")
        st.dataframe(td["members_clean"].head(10))

        st.markdown("#### 力量マスタ")
        st.dataframe(td["competence_master"].head(10))

        st.markdown("#### メンバー保有力量")
        st.dataframe(td["member_competence"].head(10))

else:
    st.markdown("---")
    st.markdown("### 📌 必要なデータ")
    st.markdown(
        "1. **メンバー**: 登録メンバー情報\n"
        "2. **力量（スキル）**: スキルマスタ\n"
        "3. **力量（教育）**: 研修マスタ\n"
        "4. **力量（資格）**: 資格マスタ\n"
        "5. **力量カテゴリー**: カテゴリーマスタ\n"
        "6. **保有力量**: メンバーの保有力量データ"
    )

