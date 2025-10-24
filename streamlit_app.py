"""
キャリア推薦システム Streamlitアプリ
（メイン画面アップロード版・DataTransformer修正版）
"""

import streamlit as st
import pandas as pd
import tempfile
import os

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.recommendation_system import RecommendationSystem
from skillnote_recommendation.core.role_model import RoleModelFinder
from skillnote_recommendation.ml import MLRecommender


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="キャリア推薦システム",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 キャリア推薦システム")
st.markdown("**ルールベース** と **機械学習（ML）** による力量推薦")


# =========================================================
# セッション初期化
# =========================================================
for key in ["data_loaded", "recommendation_system", "ml_recommender", "role_model_finder", "raw_data", "transformed_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "data_loaded" else False


# =========================================================
# 補助関数
# =========================================================
def load_csv_to_memory(uploaded_file):
    """アップロードされたCSVをDataFrameに読み込み"""
    return pd.read_csv(uploaded_file, encoding="utf-8-sig")


def save_uploaded_files(temp_dir, subdir_name, uploaded_files):
    """指定サブディレクトリに複数CSVを保存"""
    os.makedirs(os.path.join(temp_dir, subdir_name), exist_ok=True)
    for i, file in enumerate(uploaded_files):
        df = load_csv_to_memory(file)
        path = os.path.join(temp_dir, subdir_name, f"{os.path.splitext(file.name)[0]}_{i+1}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")


def create_temp_dir_with_csv(uploaded_dict):
    """全カテゴリのCSVを一時ディレクトリにまとめて保存"""
    temp_dir = tempfile.mkdtemp()
    for category, files in uploaded_dict.items():
        if files:
            save_uploaded_files(temp_dir, category, files)
    return temp_dir


# =========================================================
# メイン画面：データアップロードUI
# =========================================================
st.subheader("📁 データアップロード")

st.markdown("各カテゴリごとにCSVファイルをアップロードしてください（複数可）")

col1, col2 = st.columns(2)

with col1:
    uploaded_members = st.file_uploader("👥 メンバー", type=["csv"], accept_multiple_files=True, key="members")
    uploaded_skills = st.file_uploader("🧠 力量（スキル）", type=["csv"], accept_multiple_files=True, key="skills")
    uploaded_education = st.file_uploader("📘 力量（教育）", type=["csv"], accept_multiple_files=True, key="education")

with col2:
    uploaded_license = st.file_uploader("🎓 力量（資格）", type=["csv"], accept_multiple_files=True, key="license")
    uploaded_categories = st.file_uploader("🗂 力量カテゴリー", type=["csv"], accept_multiple_files=True, key="categories")
    uploaded_acquired = st.file_uploader("📊 保有力量", type=["csv"], accept_multiple_files=True, key="acquired")

st.markdown("---")

# =========================================================
# データ読み込み処理
# =========================================================
if st.button("📥 データ読み込み", type="primary"):
    if all([
        uploaded_members, uploaded_skills, uploaded_education,
        uploaded_license, uploaded_categories, uploaded_acquired
    ]):
        with st.spinner("データを読み込み中..."):
            try:
                # 一時ディレクトリにファイル保存
                temp_dir = create_temp_dir_with_csv({
                    "members": uploaded_members,
                    "skills": uploaded_skills,
                    "education": uploaded_education,
                    "license": uploaded_license,
                    "categories": uploaded_categories,
                    "acquired": uploaded_acquired
                })

                # --- データ読み込み ---
                loader = DataLoader(data_dir=temp_dir)
                raw_data = loader.load_all_data()

                # --- データ変換 ---
                transformer = DataTransformer()

                competence_master = transformer.create_competence_master(raw_data)
                member_competence, valid_members = transformer.create_member_competence(raw_data, competence_master)
                skill_matrix = transformer.create_skill_matrix(member_competence)
                members_clean = transformer.clean_members_data(raw_data)

                transformed_data = {
                    "competence_master": competence_master,
                    "member_competence": member_competence,
                    "skill_matrix": skill_matrix,
                    "members_clean": members_clean
                }

                # --- 推薦システム初期化 ---
                rec_system = RecommendationSystem(output_dir=temp_dir)
                role_finder = RoleModelFinder(
                    members=members_clean,
                    member_competence=member_competence,
                    competence_master=competence_master
                )

                # --- セッション保存 ---
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.recommendation_system = rec_system
                st.session_state.role_model_finder = role_finder
                st.session_state.temp_dir = temp_dir
                st.session_state.data_loaded = True

                st.success("✅ データ読み込みと変換が完了しました。")
                st.rerun()

            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")
    else:
        st.warning("⚠️ 全てのカテゴリで少なくとも1つのCSVをアップロードしてください。")


# =========================================================
# MLモデル学習処理
# =========================================================
if st.session_state.data_loaded:
    st.markdown("### 🤖 MLモデル学習")

    if st.button("MLモデル学習を実行"):
        with st.spinner("MLモデルを学習中..."):
            try:
                ml_recommender = MLRecommender(st.session_state.raw_data)
                st.session_state.ml_recommender = ml_recommender
                st.success("✅ MLモデル学習が完了しました。")
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")


# =========================================================
# 画面表示制御
# =========================================================
if not st.session_state.data_loaded:
    st.markdown("### 📋 必要なファイル一覧")
    st.markdown("""
    1. 👥 **メンバー**（member_skillnote.csv など）  
    2. 🧠 **力量（スキル）**（skill_skillnote.csv など）  
    3. 📘 **力量（教育）**（education_skillnote.csv など）  
    4. 🎓 **力量（資格）**（license_skillnote.csv など）  
    5. 🗂 **力量カテゴリー**（competence_category_skillnote.csv など）  
    6. 📊 **保有力量**（acquiredCompetenceLevel.csv など）
    """)
else:
    st.success("✅ データが正常に読み込まれました。")
    st.markdown("次のステップとして推薦処理や分析機能を実行できます。")

st.markdown("---")
st.caption("🤖 Generated with ChatGPT（DataTransformer対応版）")
