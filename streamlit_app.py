"""
キャリア推薦システム Streamlitアプリ
（ドラッグ＆ドロップ対応版・日本語表記保持）
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import tempfile
import os

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.recommendation_system import RecommendationSystem
from skillnote_recommendation.core.role_model import RoleModelFinder
from skillnote_recommendation.ml import MLRecommender


# ページ設定
st.set_page_config(
    page_title="キャリア推薦システム",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🎯 キャリア推薦システム")
st.markdown("**ルールベース** と **機械学習（ML）** による力量推薦")

# セッション状態の初期化
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


def load_csv_to_memory(uploaded_file):
    """アップロードされたCSVをDataFrameに読み込み"""
    return pd.read_csv(uploaded_file, encoding='utf-8-sig')


def create_temp_dir_with_csv(uploaded_files_dict):
    """一時ディレクトリを作成してCSVファイルを配置"""
    temp_dir = tempfile.mkdtemp()

    # 各サブディレクトリを作成
    for dirname in ['members', 'acquired', 'skills', 'education', 'license', 'categories']:
        os.makedirs(os.path.join(temp_dir, dirname), exist_ok=True)

    # CSVファイルを配置
    file_mapping = {
        'members': 'members',
        'acquired': 'acquired',
        'skills': 'skills',
        'education': 'education',
        'license': 'license',
        'categories': 'categories'
    }

    for key, dirname in file_mapping.items():
        if key in uploaded_files_dict and uploaded_files_dict[key] is not None:
            df = load_csv_to_memory(uploaded_files_dict[key])
            filepath = os.path.join(temp_dir, dirname, f'{key}.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')

    return temp_dir


# =========================================================
# サイドバー：ドラッグ＆ドロップによる一括アップロード
# =========================================================
st.sidebar.header("📁 データアップロード")
st.sidebar.markdown("### 📂 CSVファイルをドラッグ＆ドロップでまとめてアップロード")

uploaded_files = st.sidebar.file_uploader(
    "全6種類のCSVをまとめてドロップしてください",
    type=['csv'],
    accept_multiple_files=True
)

# 必須ファイル名の定義（日本語名保持）
required_files = {
    'members': 'member_skillnote.csv',  # メンバー
    'acquired': 'acquiredCompetenceLevel.csv',  # 保有力量
    'skills': 'skill_skillnote.csv',  # 力量（スキル）
    'education': 'education_skillnote.csv',  # 力量（教育）
    'license': 'license_skillnote.csv',  # 力量（資格）
    'categories': 'competence_category_skillnote.csv'  # 力量カテゴリー
}

# アップロードされたファイルをマッピング
uploaded_dict = {}
if uploaded_files:
    for file in uploaded_files:
        for key, filename in required_files.items():
            if filename in file.name:
                uploaded_dict[key] = file

# データ読み込みボタン
if st.sidebar.button("データ読み込み", type="primary"):
    if len(uploaded_dict) == len(required_files):
        with st.spinner("データを読み込み中..."):
            try:
                # 一時ディレクトリに保存
                temp_dir = create_temp_dir_with_csv(uploaded_dict)

                # データ読み込みと変換
                loader = DataLoader(data_dir=temp_dir)
                raw_data = loader.load_all_data()
                transformer = DataTransformer(raw_data)
                transformed_data = transformer.transform_all()

                # 推薦システム初期化
                rec_system = RecommendationSystem(output_dir=temp_dir)
                role_finder = RoleModelFinder(
                    members=transformed_data['members_clean'],
                    member_competence=transformed_data['member_competence'],
                    competence_master=transformed_data['competence_master']
                )

                # セッションへ格納
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.recommendation_system = rec_system
                st.session_state.role_model_finder = role_finder
                st.session_state.temp_dir = temp_dir
                st.session_state.data_loaded = True

                st.sidebar.success("✅ データ読み込みが完了しました")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"エラーが発生しました: {str(e)}")
    else:
        missing = set(required_files.keys()) - set(uploaded_dict.keys())
        missing_names = [required_files[k] for k in missing]
        st.sidebar.warning(f"⚠️ 以下のファイルが不足しています: {', '.join(missing_names)}")


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
                st.sidebar.success("✅ MLモデル学習が完了しました")
            except Exception as e:
                st.sidebar.error(f"エラー: {str(e)}")


# =========================================================
# メイン画面
# =========================================================
if not st.session_state.data_loaded:
    st.info("👈 左のサイドバーに全6種類のCSVファイルをまとめてドラッグ＆ドロップし、「データ読み込み」をクリックしてください。")

    st.markdown("### 📋 必要なファイル一覧")
    st.markdown("""
    1. **メンバー** → `member_skillnote.csv`  
    2. **力量（スキル）** → `skill_skillnote.csv`  
    3. **力量（教育）** → `education_skillnote.csv`  
    4. **力量（資格）** → `license_skillnote.csv`  
    5. **力量カテゴリー** → `competence_category_skillnote.csv`  
    6. **保有力量** → `acquiredCompetenceLevel.csv`
    """)
else:
    st.success("✅ データ読み込みが完了しました。")
    st.markdown("この後に推薦処理や分析機能を実行できます。")
