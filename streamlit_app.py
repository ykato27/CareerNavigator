"""
キャリア推薦システム Streamlitアプリ
（カテゴリ別・複数CSVドラッグ＆ドロップ対応版）
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from io import StringIO
import plotly.express as px

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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 キャリア推薦システム")
st.markdown("**ルールベース** と **機械学習（ML）** による力量推薦")

# =========================================================
# セッション初期化
# =========================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendation_system' not in st.session_state:
    st.session_state.recommendation_system = None
if 'ml_recommender' not in st.session_state:
    st.session_state.ml_recommender = None
if 'role_model_finder' not in st.session_state:
    st.session_state.role_model_finder = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None


# =========================================================
# 補助関数
# =========================================================
def load_csv_to_memory(uploaded_file):
    """アップロードされたCSVをDataFrameに読み込み"""
    return pd.read_csv(uploaded_file, encoding='utf-8-sig')


def save_uploaded_files(temp_dir, subdir_name, uploaded_files):
    """指定サブディレクトリに複数CSVを保存"""
    os.makedirs(os.path.join(temp_dir, subdir_name), exist_ok=True)
    saved_paths = []
    for i, file in enumerate(uploaded_files):
        df = load_csv_to_memory(file)
        file_path = os.path.join(temp_dir, subdir_name, f"{os.path.splitext(file.name)[0]}_{i+1}.csv")
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        saved_paths.append(file_path)
    return saved_paths


def create_temp_dir_with_csv(uploaded_files_dict):
    """全カテゴリのCSVを一時ディレクトリにまとめて保存"""
    temp_dir = tempfile.mkdtemp()
    for category, files in uploaded_files_dict.items():
        if files:
            save_uploaded_files(temp_dir, category, files)
    return temp_dir


# =========================================================
# サイドバー：カテゴリ別アップロード
# =========================================================
st.sidebar.header("📁 データアップロード")

st.sidebar.markdown("各カテゴリごとにCSVファイルをアップロードしてください（複数可）")

uploaded_members = st.sidebar.file_uploader(
    "👥 メンバー (member_skillnote.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='members'
)

uploaded_skills = st.sidebar.file_uploader(
    "🧠 力量（スキル） (skill_skillnote.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='skills'
)

uploaded_education = st.sidebar.file_uploader(
    "📘 力量（教育） (education_skillnote.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='education'
)

uploaded_license = st.sidebar.file_uploader(
    "🎓 力量（資格） (license_skillnote.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='license'
)

uploaded_categories = st.sidebar.file_uploader(
    "🗂 力量カテゴリー (competence_category_skillnote.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='categories'
)

uploaded_acquired = st.sidebar.file_uploader(
    "📊 保有力量 (acquiredCompetenceLevel.csv など)",
    type=['csv'],
    accept_multiple_files=True,
    key='acquired'
)

# =========================================================
# データ読み込み
# =========================================================
if st.sidebar.button("データ読み込み", type="primary"):
    if all([
        uploaded_members, uploaded_skills, uploaded_education,
        uploaded_license, uploaded_categories, uploaded_acquired
    ]):
        with st.spinner("データを読み込み中..."):
            try:
                # ファイルを一時ディレクトリに保存
                temp_dir = create_temp_dir_with_csv({
                    'members': uploaded_members,
                    'skills': uploaded_skills,
                    'education': uploaded_education,
                    'license': uploaded_license,
                    'categories': uploaded_categories,
                    'acquired': uploaded_acquired
                })

                # データ読み込み
                loader = DataLoader(data_dir=temp_dir)
                raw_data = loader.load_all_data()

                # データ変換
                transformer = DataTransformer(raw_data)
                transformed_data = transformer.transform_all()

                # 推薦システム初期化
                rec_system = RecommendationSystem(output_dir=temp_dir)
                role_finder = RoleModelFinder(
                    members=transformed_data['members_clean'],
                    member_competence=transformed_data['member_competence'],
                    competence_master=transformed_data['competence_master']
                )

                # セッション保存
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.recommendation_system = rec_system
                st.session_state.role_model_finder = role_finder
                st.session_state.temp_dir = temp_dir
                st.session_state.data_loaded = True

                st.sidebar.success("✅ データ読み込み完了")
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"エラーが発生しました: {str(e)}")

    else:
        st.sidebar.warning("⚠️ 全てのカテゴリで少なくとも1つのCSVファイルをアップロードしてください。")


# =========================================================
# MLモデル学習
# =========================================================
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 機械学習モデル")

    if st.sidebar.button("MLモデル学習", type="secondary"):
        with st.spinner("MLモデルを学習中..."):
            try:
                ml_recommender = MLRecommender(st.session_state.raw_data)
                st.session_state.ml_recommender = ml_recommender
                st.sidebar.success("✅ MLモデル学習完了")
            except Exception as e:
                st.sidebar.error(f"エラー: {str(e)}")


# =========================================================
# メイン画面
# =========================================================
if not st.session_state.data_loaded:
    st.info("👈 左のサイドバーでカテゴリごとにCSVをアップロードし、「データ読み込み」をクリックしてください。")

    st.markdown("### 📋 必要なカテゴリ一覧")
    st.markdown("""
    1. 👥 **メンバー**（member_skillnote.csv など）  
    2. 🧠 **力量（スキル）**（skill_skillnote.csv など）  
    3. 📘 **力量（教育）**（education_skillnote.csv など）  
    4. 🎓 **力量（資格）**（license_skillnote.csv など）  
    5. 🗂 **力量カテゴリー**（competence_category_skillnote.csv など）  
    6. 📊 **保有力量**（acquiredCompetenceLevel.csv など）
    """)
else:
    st.success("✅ データ読み込みが完了しました。")
    st.markdown("この後に推薦処理や分析機能を実行できます。")
