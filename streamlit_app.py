"""
スキルノート推薦システム Streamlitアプリ

ルールベースとML両方の推薦機能を提供するWebアプリケーション
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
import tempfile
import os

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer
from skillnote_recommendation.core.recommendation_system import RecommendationSystem
from skillnote_recommendation.core.role_model import RoleModelFinder
from skillnote_recommendation.ml import MLRecommender


# ページ設定
st.set_page_config(
    page_title="スキルノート推薦システム",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトル
st.title("🎯 スキルノート推薦システム")
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


# サイドバー: データアップロード
st.sidebar.header("📁 データアップロード")

st.sidebar.markdown("### 必須データ（全6種類）")

uploaded_members = st.sidebar.file_uploader(
    "会員データ (members.csv)",
    type=['csv'],
    key='members'
)

uploaded_acquired = st.sidebar.file_uploader(
    "習得力量データ (acquired.csv)",
    type=['csv'],
    key='acquired'
)

uploaded_skills = st.sidebar.file_uploader(
    "スキル力量マスター (skills.csv)",
    type=['csv'],
    key='skills'
)

uploaded_education = st.sidebar.file_uploader(
    "教育力量マスター (education.csv)",
    type=['csv'],
    key='education'
)

uploaded_license = st.sidebar.file_uploader(
    "資格力量マスター (license.csv)",
    type=['csv'],
    key='license'
)

uploaded_categories = st.sidebar.file_uploader(
    "カテゴリマスター (categories.csv)",
    type=['csv'],
    key='categories'
)

# データ読み込みボタン
if st.sidebar.button("データ読み込み", type="primary"):
    if all([uploaded_members, uploaded_acquired, uploaded_skills,
            uploaded_education, uploaded_license, uploaded_categories]):

        with st.spinner("データを読み込み中..."):
            try:
                # 一時ディレクトリにCSVを配置
                temp_dir = create_temp_dir_with_csv({
                    'members': uploaded_members,
                    'acquired': uploaded_acquired,
                    'skills': uploaded_skills,
                    'education': uploaded_education,
                    'license': uploaded_license,
                    'categories': uploaded_categories
                })

                # データ読み込み
                loader = DataLoader(data_dir=temp_dir)
                raw_data = loader.load_all_data()

                # データ変換
                transformer = DataTransformer(raw_data)
                transformed_data = transformer.transform_all()

                # ルールベース推薦システム初期化
                rec_system = RecommendationSystem(
                    output_dir=temp_dir
                )

                # ロールモデル検索機能初期化
                role_finder = RoleModelFinder(
                    members=transformed_data['members_clean'],
                    member_competence=transformed_data['member_competence'],
                    competence_master=transformed_data['competence_master']
                )

                # セッション状態に保存
                st.session_state.raw_data = raw_data
                st.session_state.transformed_data = transformed_data
                st.session_state.recommendation_system = rec_system
                st.session_state.role_model_finder = role_finder
                st.session_state.temp_dir = temp_dir
                st.session_state.data_loaded = True

                st.sidebar.success("✅ データ読み込み完了")
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"エラー: {str(e)}")
    else:
        st.sidebar.warning("⚠️ 全6種類のCSVファイルをアップロードしてください")

# ML学習ボタン
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

# メインエリア
if not st.session_state.data_loaded:
    st.info("👈 左のサイドバーから全6種類のCSVファイルをアップロードして、「データ読み込み」をクリックしてください")

    st.markdown("### 📋 必要なファイル")
    st.markdown("""
    1. **会員データ** (members.csv) - 会員マスター
    2. **習得力量データ** (acquired.csv) - 会員の習得力量
    3. **スキル力量マスター** (skills.csv) - SKILLタイプの力量
    4. **教育力量マスター** (education.csv) - EDUCATIONタイプの力量
    5. **資格力量マスター** (license.csv) - LICENSEタイプの力量
    6. **カテゴリマスター** (categories.csv) - 力量カテゴリ
    """)

    st.markdown("### 🆕 新規ユーザーの推薦")
    st.markdown("""
    新規ユーザーの推薦を行う場合は、以下のテンプレートをダウンロードして編集してください：
    - [新規ユーザーCSVテンプレート](templates/new_user_template.csv)
    - [テンプレート使い方ガイド](templates/README.md)
    """)

else:
    # データ読み込み済み: タブで機能を分ける
    tab1, tab2 = st.tabs(["👤 既存会員の推薦", "🆕 新規ユーザーの推薦"])

    with tab1:
        st.header("既存会員への推薦")

        members_df = st.session_state.transformed_data['members_clean']

        # 会員選択
        member_options = members_df.apply(
            lambda row: f"{row['会員コード']} - {row['会員名']} ({row['職能等級']})",
            axis=1
        ).tolist()

        selected_member_text = st.selectbox(
            "推薦対象の会員を選択",
            options=member_options,
            key='selected_member'
        )

        selected_member_code = selected_member_text.split(' - ')[0]

        # 推薦設定
        col1, col2, col3 = st.columns(3)

        with col1:
            top_n = st.slider("推薦件数", min_value=5, max_value=50, value=10, step=5)

        with col2:
            competence_type = st.selectbox(
                "力量タイプフィルタ",
                options=['全て', 'SKILL', 'EDUCATION', 'LICENSE']
            )
            competence_type = None if competence_type == '全て' else competence_type

        with col3:
            diversity_strategy = st.selectbox(
                "多様性戦略（ML）",
                options=['hybrid', 'mmr', 'category', 'type'],
                index=0
            )

        if st.button("推薦実行", type="primary", key='recommend_existing'):
            st.markdown("---")

            # 会員情報表示
            member_info = st.session_state.recommendation_system.get_member_info(selected_member_code)

            st.subheader(f"📊 会員情報: {member_info['name']}")

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            info_col1.metric("職能等級", member_info['grade'])
            info_col2.metric("SKILL", f"{member_info['skill_count']}件")
            info_col3.metric("EDUCATION", f"{member_info['education_count']}件")
            info_col4.metric("LICENSE", f"{member_info['license_count']}件")

            st.markdown("---")

            # 推薦実行
            with st.spinner("推薦を実行中..."):
                # ルールベース推薦
                rule_recommendations = st.session_state.recommendation_system.recommend_competences(
                    selected_member_code,
                    competence_type=competence_type,
                    top_n=top_n
                )

                # ML推薦（モデルが学習済みの場合）
                ml_recommendations = None
                ml_diversity_metrics = None

                if st.session_state.ml_recommender is not None:
                    ml_result = st.session_state.ml_recommender.recommend(
                        member_code=selected_member_code,
                        top_n=top_n,
                        competence_type=competence_type,
                        use_diversity=True,
                        diversity_strategy=diversity_strategy
                    )
                    ml_recommendations = ml_result

                    # 多様性メトリクス計算
                    ml_diversity_metrics = st.session_state.ml_recommender.calculate_diversity_metrics(
                        ml_recommendations,
                        st.session_state.ml_recommender.competence_master
                    )

            # 推薦結果表示（2カラム）
            st.subheader("🎯 推薦結果")

            result_col1, result_col2 = st.columns(2)

            # ルールベース推薦結果
            with result_col1:
                st.markdown("### 📋 ルールベース推薦")

                if rule_recommendations:
                    for idx, rec in enumerate(rule_recommendations, 1):
                        with st.expander(f"**{idx}. {rec.competence_name}** (スコア: {rec.priority_score:.2f})"):
                            st.markdown(f"**力量コード**: {rec.competence_code}")
                            st.markdown(f"**タイプ**: {rec.competence_type}")
                            st.markdown(f"**カテゴリ**: {rec.category_name}")
                            st.markdown(f"**推薦理由**: {rec.reason}")

                            # ロールモデル表示
                            role_models = st.session_state.role_model_finder.find_role_models(
                                competence_code=rec.competence_code,
                                target_member_code=selected_member_code,
                                top_n=3
                            )

                            if role_models:
                                st.markdown("**👥 参考となる会員（この力量を習得済み）:**")
                                for rm in role_models:
                                    st.markdown(
                                        f"- {rm['member_name']} (Lv.{rm['competence_level']}) - "
                                        f"総習得力量: {rm['total_competences']}件"
                                    )
                else:
                    st.info("推薦できる力量がありません")

            # ML推薦結果
            with result_col2:
                st.markdown("### 🤖 ML推薦")

                if st.session_state.ml_recommender is None:
                    st.warning("⚠️ MLモデルが未学習です。サイドバーから「MLモデル学習」を実行してください。")
                elif ml_recommendations is not None and len(ml_recommendations) > 0:
                    for idx, rec in ml_recommendations.iterrows():
                        with st.expander(f"**{idx+1}. {rec['力量名']}** (スコア: {rec['MLスコア']:.3f})"):
                            st.markdown(f"**力量コード**: {rec['力量コード']}")
                            st.markdown(f"**タイプ**: {rec['力量種別']}")
                            st.markdown(f"**カテゴリ**: {rec['カテゴリ名']}")
                            st.markdown(f"**推薦理由**: {rec['推薦理由']}")

                            # ロールモデル表示
                            role_models = st.session_state.role_model_finder.find_role_models(
                                competence_code=rec['力量コード'],
                                target_member_code=selected_member_code,
                                top_n=3
                            )

                            if role_models:
                                st.markdown("**👥 参考となる会員（この力量を習得済み）:**")
                                for rm in role_models:
                                    st.markdown(
                                        f"- {rm['member_name']} (Lv.{rm['competence_level']}) - "
                                        f"総習得力量: {rm['total_competences']}件"
                                    )
                else:
                    st.info("推薦できる力量がありません")

            # 多様性メトリクス表示（ML）
            if ml_diversity_metrics:
                st.markdown("---")
                st.subheader("📊 多様性メトリクス（ML推薦）")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("カテゴリ多様性", f"{ml_diversity_metrics['category_diversity']:.3f}")
                metric_col2.metric("タイプ多様性", f"{ml_diversity_metrics['type_diversity']:.3f}")
                metric_col3.metric("カバレッジ", f"{ml_diversity_metrics['coverage']:.3f}")
                metric_col4.metric("リスト内多様性", f"{ml_diversity_metrics['intra_list_diversity']:.3f}")

                # グラフ表示
                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    # カテゴリ分布
                    if ml_diversity_metrics.get('category_distribution'):
                        cat_dist = ml_diversity_metrics['category_distribution']
                        fig_cat = px.bar(
                            x=list(cat_dist.keys()),
                            y=list(cat_dist.values()),
                            labels={'x': 'カテゴリ', 'y': '件数'},
                            title='カテゴリ分布'
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)

                with graph_col2:
                    # タイプ分布
                    if ml_diversity_metrics.get('type_distribution'):
                        type_dist = ml_diversity_metrics['type_distribution']
                        fig_type = px.pie(
                            names=list(type_dist.keys()),
                            values=list(type_dist.values()),
                            title='タイプ分布'
                        )
                        st.plotly_chart(fig_type, use_container_width=True)

            # CSVダウンロード
            st.markdown("---")
            st.subheader("💾 推薦結果ダウンロード")

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                if rule_recommendations:
                    rule_df = pd.DataFrame([
                        {
                            '順位': idx,
                            '力量コード': rec.competence_code,
                            '力量名': rec.competence_name,
                            'タイプ': rec.competence_type,
                            'カテゴリ': rec.category_name,
                            'スコア': rec.priority_score,
                            '推薦理由': rec.reason
                        }
                        for idx, rec in enumerate(rule_recommendations, 1)
                    ])

                    csv_buffer = StringIO()
                    rule_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

                    st.download_button(
                        label="ルールベース推薦結果をダウンロード",
                        data=csv_buffer.getvalue(),
                        file_name=f"rule_based_recommendations_{selected_member_code}.csv",
                        mime="text/csv"
                    )

            with download_col2:
                if ml_recommendations is not None and len(ml_recommendations) > 0:
                    ml_df = ml_recommendations.copy()
                    ml_df.insert(0, '順位', range(1, len(ml_df) + 1))

                    csv_buffer = StringIO()
                    ml_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

                    st.download_button(
                        label="ML推薦結果をダウンロード",
                        data=csv_buffer.getvalue(),
                        file_name=f"ml_recommendations_{selected_member_code}.csv",
                        mime="text/csv"
                    )

    with tab2:
        st.header("新規ユーザーへの推薦")

        st.markdown("""
        新規ユーザーの習得力量データをCSVでアップロードしてください。

        **CSVフォーマット:**
        ```
        会員コード,会員名,力量コード,力量レベル
        m999,新規ユーザー,s001,3
        m999,新規ユーザー,s002,5
        ```

        [テンプレートをダウンロード](templates/new_user_template.csv)
        """)

        uploaded_new_user = st.file_uploader(
            "新規ユーザーCSVをアップロード",
            type=['csv'],
            key='new_user'
        )

        if uploaded_new_user is not None:
            try:
                new_user_df = load_csv_to_memory(uploaded_new_user)

                # 新規ユーザー情報を表示
                st.subheader("📋 新規ユーザー情報")

                member_code = new_user_df['会員コード'].iloc[0]
                member_name = new_user_df['会員名'].iloc[0]

                st.markdown(f"**会員コード**: {member_code}")
                st.markdown(f"**会員名**: {member_name}")
                st.markdown(f"**習得力量数**: {len(new_user_df)}件")

                st.dataframe(new_user_df, use_container_width=True)

                # 推薦設定（既存会員と同様）
                st.markdown("---")
                st.subheader("⚙️ 推薦設定")

                col1, col2, col3 = st.columns(3)

                with col1:
                    new_top_n = st.slider("推薦件数", min_value=5, max_value=50, value=10, step=5, key='new_top_n')

                with col2:
                    new_competence_type = st.selectbox(
                        "力量タイプフィルタ",
                        options=['全て', 'SKILL', 'EDUCATION', 'LICENSE'],
                        key='new_competence_type'
                    )
                    new_competence_type = None if new_competence_type == '全て' else new_competence_type

                with col3:
                    new_diversity_strategy = st.selectbox(
                        "多様性戦略（ML）",
                        options=['hybrid', 'mmr', 'category', 'type'],
                        index=0,
                        key='new_diversity_strategy'
                    )

                if st.button("推薦実行", type="primary", key='recommend_new'):
                    st.markdown("---")

                    # 新規ユーザー情報の統計
                    st.subheader(f"📊 {member_name}さんの保有力量")

                    # タイプ別にカウント
                    comp_master = st.session_state.transformed_data['competence_master']
                    user_comps_with_type = new_user_df.merge(
                        comp_master[['力量コード', '力量種別']],
                        on='力量コード',
                        how='left'
                    )

                    skill_count = len(user_comps_with_type[user_comps_with_type['力量種別'] == 'SKILL'])
                    education_count = len(user_comps_with_type[user_comps_with_type['力量種別'] == 'EDUCATION'])
                    license_count = len(user_comps_with_type[user_comps_with_type['力量種別'] == 'LICENSE'])

                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    info_col1.metric("総習得力量", f"{len(new_user_df)}件")
                    info_col2.metric("SKILL", f"{skill_count}件")
                    info_col3.metric("EDUCATION", f"{education_count}件")
                    info_col4.metric("LICENSE", f"{license_count}件")

                    st.markdown("---")

                    # 新規ユーザーのデータを一時的に追加して推薦実行
                    with st.spinner("推薦を実行中..."):
                        # 新規会員データを追加
                        temp_members = st.session_state.transformed_data['members_clean'].copy()
                        new_member_row = pd.DataFrame([{
                            '会員コード': member_code,
                            '会員名': member_name,
                            '役職': '未設定',
                            '職能等級': '未設定'
                        }])
                        temp_members = pd.concat([temp_members, new_member_row], ignore_index=True)

                        # 新規習得力量データを追加
                        temp_member_competence = st.session_state.transformed_data['member_competence'].copy()
                        new_competences = new_user_df.copy()
                        new_competences.columns = ['会員コード', '会員名_drop', '力量コード', 'レベル']
                        new_competences = new_competences.drop(columns=['会員名_drop'])
                        temp_member_competence = pd.concat([temp_member_competence, new_competences], ignore_index=True)

                        # スキルマトリクス更新
                        from skillnote_recommendation.core.data_transformer import DataTransformer
                        temp_transformer = DataTransformer({})
                        temp_skill_matrix = temp_transformer.create_skill_matrix(temp_member_competence)

                        # 一時的な推薦システムを作成（データを上書き）
                        temp_rec_system = st.session_state.recommendation_system
                        temp_rec_system.members = temp_members
                        temp_rec_system.member_competence = temp_member_competence
                        temp_rec_system.skill_matrix = temp_skill_matrix

                        # ルールベース推薦実行
                        rule_recommendations = temp_rec_system.recommend_competences(
                            member_code,
                            competence_type=new_competence_type,
                            top_n=new_top_n
                        )

                        # ML推薦（モデルが学習済みの場合）
                        ml_recommendations = None
                        ml_diversity_metrics = None

                        if st.session_state.ml_recommender is not None:
                            # 新規ユーザーのデータでMLモデルを再学習
                            temp_raw_data = st.session_state.raw_data.copy()
                            temp_raw_data['members'] = pd.concat([
                                temp_raw_data['members'],
                                pd.DataFrame([{'会員コード': member_code, '会員名': member_name}])
                            ], ignore_index=True)

                            # 習得力量データ追加
                            acquired_data = new_user_df.copy()
                            acquired_data['取得日'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                            temp_raw_data['acquired'] = pd.concat([
                                temp_raw_data['acquired'],
                                acquired_data
                            ], ignore_index=True)

                            # MLモデル再学習
                            with st.spinner("新規ユーザーデータでMLモデルを再学習中..."):
                                temp_ml_recommender = MLRecommender(temp_raw_data)

                            ml_result = temp_ml_recommender.recommend(
                                member_code=member_code,
                                top_n=new_top_n,
                                competence_type=new_competence_type,
                                use_diversity=True,
                                diversity_strategy=new_diversity_strategy
                            )
                            ml_recommendations = ml_result

                            # 多様性メトリクス計算
                            ml_diversity_metrics = temp_ml_recommender.calculate_diversity_metrics(
                                ml_recommendations,
                                temp_ml_recommender.competence_master
                            )

                    # 推薦結果表示（2カラム） - 既存会員と同じUI
                    st.subheader("🎯 推薦結果")

                    result_col1, result_col2 = st.columns(2)

                    # ルールベース推薦結果
                    with result_col1:
                        st.markdown("### 📋 ルールベース推薦")

                        if rule_recommendations:
                            for idx, rec in enumerate(rule_recommendations, 1):
                                with st.expander(f"**{idx}. {rec.competence_name}** (スコア: {rec.priority_score:.2f})"):
                                    st.markdown(f"**力量コード**: {rec.competence_code}")
                                    st.markdown(f"**タイプ**: {rec.competence_type}")
                                    st.markdown(f"**カテゴリ**: {rec.category_name}")
                                    st.markdown(f"**推薦理由**: {rec.reason}")

                                    # ロールモデル表示
                                    role_models = st.session_state.role_model_finder.find_role_models(
                                        competence_code=rec.competence_code,
                                        target_member_code=None,  # 新規ユーザーは除外しない
                                        top_n=3
                                    )

                                    if role_models:
                                        st.markdown("**👥 参考となる会員（この力量を習得済み）:**")
                                        for rm in role_models:
                                            st.markdown(
                                                f"- {rm['member_name']} (Lv.{rm['competence_level']}) - "
                                                f"総習得力量: {rm['total_competences']}件"
                                            )
                        else:
                            st.info("推薦できる力量がありません")

                    # ML推薦結果
                    with result_col2:
                        st.markdown("### 🤖 ML推薦")

                        if st.session_state.ml_recommender is None:
                            st.warning("⚠️ MLモデルが未学習です。サイドバーから「MLモデル学習」を実行してください。")
                        elif ml_recommendations is not None and len(ml_recommendations) > 0:
                            for idx, rec in ml_recommendations.iterrows():
                                with st.expander(f"**{idx+1}. {rec['力量名']}** (スコア: {rec['MLスコア']:.3f})"):
                                    st.markdown(f"**力量コード**: {rec['力量コード']}")
                                    st.markdown(f"**タイプ**: {rec['力量種別']}")
                                    st.markdown(f"**カテゴリ**: {rec['カテゴリ名']}")
                                    st.markdown(f"**推薦理由**: {rec['推薦理由']}")

                                    # ロールモデル表示
                                    role_models = st.session_state.role_model_finder.find_role_models(
                                        competence_code=rec['力量コード'],
                                        target_member_code=None,
                                        top_n=3
                                    )

                                    if role_models:
                                        st.markdown("**👥 参考となる会員（この力量を習得済み）:**")
                                        for rm in role_models:
                                            st.markdown(
                                                f"- {rm['member_name']} (Lv.{rm['competence_level']}) - "
                                                f"総習得力量: {rm['total_competences']}件"
                                            )
                        else:
                            st.info("推薦できる力量がありません")

                    # 多様性メトリクス表示（ML）
                    if ml_diversity_metrics:
                        st.markdown("---")
                        st.subheader("📊 多様性メトリクス（ML推薦）")

                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        metric_col1.metric("カテゴリ多様性", f"{ml_diversity_metrics['category_diversity']:.3f}")
                        metric_col2.metric("タイプ多様性", f"{ml_diversity_metrics['type_diversity']:.3f}")
                        metric_col3.metric("カバレッジ", f"{ml_diversity_metrics['coverage']:.3f}")
                        metric_col4.metric("リスト内多様性", f"{ml_diversity_metrics['intra_list_diversity']:.3f}")

                        # グラフ表示
                        graph_col1, graph_col2 = st.columns(2)

                        with graph_col1:
                            # カテゴリ分布
                            if ml_diversity_metrics.get('category_distribution'):
                                cat_dist = ml_diversity_metrics['category_distribution']
                                fig_cat = px.bar(
                                    x=list(cat_dist.keys()),
                                    y=list(cat_dist.values()),
                                    labels={'x': 'カテゴリ', 'y': '件数'},
                                    title='カテゴリ分布'
                                )
                                st.plotly_chart(fig_cat, use_container_width=True)

                        with graph_col2:
                            # タイプ分布
                            if ml_diversity_metrics.get('type_distribution'):
                                type_dist = ml_diversity_metrics['type_distribution']
                                fig_type = px.pie(
                                    names=list(type_dist.keys()),
                                    values=list(type_dist.values()),
                                    title='タイプ分布'
                                )
                                st.plotly_chart(fig_type, use_container_width=True)

                    # CSVダウンロード
                    st.markdown("---")
                    st.subheader("💾 推薦結果ダウンロード")

                    download_col1, download_col2 = st.columns(2)

                    with download_col1:
                        if rule_recommendations:
                            rule_df = pd.DataFrame([
                                {
                                    '順位': idx,
                                    '力量コード': rec.competence_code,
                                    '力量名': rec.competence_name,
                                    'タイプ': rec.competence_type,
                                    'カテゴリ': rec.category_name,
                                    'スコア': rec.priority_score,
                                    '推薦理由': rec.reason
                                }
                                for idx, rec in enumerate(rule_recommendations, 1)
                            ])

                            csv_buffer = StringIO()
                            rule_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

                            st.download_button(
                                label="ルールベース推薦結果をダウンロード",
                                data=csv_buffer.getvalue(),
                                file_name=f"rule_based_recommendations_{member_code}.csv",
                                mime="text/csv"
                            )

                    with download_col2:
                        if ml_recommendations is not None and len(ml_recommendations) > 0:
                            ml_df = ml_recommendations.copy()
                            ml_df.insert(0, '順位', range(1, len(ml_df) + 1))

                            csv_buffer = StringIO()
                            ml_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

                            st.download_button(
                                label="ML推薦結果をダウンロード",
                                data=csv_buffer.getvalue(),
                                file_name=f"ml_recommendations_{member_code}.csv",
                                mime="text/csv"
                            )

            except Exception as e:
                st.error(f"CSVファイルの読み込みエラー: {str(e)}")

st.markdown("---")
st.markdown("🤖 Generated with [Claude Code](https://claude.com/claude-code)")
