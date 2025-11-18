"""
スキル依存関係分析ページ

時系列データから学習順序パターンを抽出し、
スキル間の依存関係を可視化します。
"""

import streamlit as st
import pandas as pd

from skillnote_recommendation.core.skill_dependency_analyzer import (
    SkillDependencyAnalyzer,
    LearningPath
)
from skillnote_recommendation.utils.visualization import (
    create_dependency_graph,
    create_learning_path_timeline
)
from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    display_error_details
)
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)

# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="CareerNavigator - スキル依存関係",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply rich UI styles
apply_rich_ui_styles()

# リッチなヘッダー
render_gradient_header(
    title="スキル依存関係分析",
    icon="🔗",
    description="時系列データから学習順序パターンを抽出し、推奨される学習パスを表示します"
)

# =========================================================
# 前提条件チェック
# =========================================================

check_data_loaded()

# =========================================================
# データ準備
# =========================================================

td = st.session_state.transformed_data
member_competence = td["member_competence"]
competence_master = td["competence_master"]

# 取得日カラムの存在チェック
if '取得日' not in member_competence.columns:
    st.error("❌ スキル依存関係分析には「取得日」データが必要です")
    st.info("""
    **対処方法:**
    1. CSVファイルに取得日カラムを追加してください
    2. データを再アップロードしてください

    **必要な形式:**
    - カラム名: `取得日`
    - 形式: YYYY/MM/DD または YYYY-MM-DD
    """)
    st.stop()

# =========================================================
# 分析設定
# =========================================================

st.markdown("---")
st.subheader("⚙️ 分析設定")

col1, col2, col3 = st.columns(3)

with col1:
    time_window_days = st.slider(
        "遷移とみなす最大期間（日数）",
        min_value=30,
        max_value=365,
        value=180,
        step=30,
        help="この期間内に連続して習得したスキルペアを分析対象とします"
    )

with col2:
    min_transition_count = st.slider(
        "最小遷移人数",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="この人数以上が同じ順序で学んだパターンのみを抽出します"
    )

with col3:
    confidence_threshold = st.slider(
        "依存関係の信頼度閾値",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="この信頼度以上の遷移を依存関係とみなします"
    )

# =========================================================
# 分析実行
# =========================================================

st.markdown("---")
st.subheader("🚀 分析実行")

if st.button("依存関係を分析", type="primary"):
    with st.spinner("スキル依存関係を分析中..."):
        try:
            # アナライザーを初期化
            analyzer = SkillDependencyAnalyzer(
                member_competence=member_competence,
                competence_master=competence_master,
                time_window_days=time_window_days,
                min_transition_count=min_transition_count,
                confidence_threshold=confidence_threshold
            )

            # 学習パスを生成
            learning_paths = analyzer.generate_learning_paths()

            # グラフデータを取得
            graph_data = analyzer.get_dependency_graph_data()

            # セッション状態に保存
            st.session_state.skill_dependencies = {
                'analyzer': analyzer,
                'learning_paths': learning_paths,
                'graph_data': graph_data
            }

            st.success(f"✅ 分析完了！{len(learning_paths)}個のスキルの学習パスを生成しました")

        except Exception as e:
            display_error_details(e, "依存関係分析")

# =========================================================
# 分析結果表示
# =========================================================

if 'skill_dependencies' in st.session_state:
    dep_data = st.session_state.skill_dependencies
    learning_paths = dep_data['learning_paths']
    graph_data = dep_data['graph_data']

    st.markdown("---")

    # サマリー情報
    st.markdown("### 📊 分析サマリー")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_skills = len(learning_paths)
        st.markdown(f"""
        <div class="metric-card">
            <h3>分析スキル数</h3>
            <h1>{total_skills}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_edges = len(graph_data.get('edges', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>依存関係数</h3>
            <h1>{total_edges}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # 強い依存関係の数
        strong_deps = sum(1 for edge in graph_data.get('edges', []) if edge.get('strength') == '強')
        st.markdown(f"""
        <div class="metric-card">
            <h3>強い依存関係</h3>
            <h1>{strong_deps}</h1>
        </div>
        """, unsafe_allow_html=True)

    # タブで表示
    tab1, tab2, tab3 = st.tabs([
        "🕸️ 依存関係グラフ",
        "📋 学習パス一覧",
        "🔍 スキル詳細検索"
    ])

    with tab1:
        st.markdown("### スキル依存関係グラフ")
        st.markdown("""
        **グラフの見方:**
        - 🔵 青: SKILL
        - 🟢 緑: EDUCATION
        - 🟡 黄: LICENSE
        - 矢印の向き: 学習順序（AからBへの矢印 = Aを先に学ぶべき）
        - 線の色:
          - 🔴 赤（太線）: 強い依存関係（信頼度 ≥ 70%）
          - 🟠 橙（中線）: 中程度の依存関係（信頼度 50-70%）
          - ⚫ 灰（細線）: 弱い依存関係（信頼度 30-50%）
        """)

        if graph_data.get('edges'):
            fig = create_dependency_graph(graph_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("依存関係が見つかりませんでした。分析設定を調整してみてください。")

    with tab2:
        st.markdown("### 全スキルの学習パス")

        # フィルタ
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            type_filter = st.multiselect(
                "タイプでフィルタ",
                options=['SKILL', 'EDUCATION', 'LICENSE'],
                default=['SKILL', 'EDUCATION', 'LICENSE']
            )

        with filter_col2:
            difficulty_filter = st.multiselect(
                "難易度でフィルタ",
                options=['初級', '中級', '上級'],
                default=['初級', '中級', '上級']
            )

        # 学習パスをフィルタして表示
        filtered_paths = {
            code: path for code, path in learning_paths.items()
            if path.competence_type in type_filter and path.estimated_difficulty in difficulty_filter
        }

        if filtered_paths:
            # データフレームで表示
            path_data = []
            for code, path in filtered_paths.items():
                path_data.append({
                    '力量コード': code,
                    '力量名': path.competence_name,
                    'タイプ': path.competence_type,
                    '難易度': path.estimated_difficulty,
                    '前提スキル数': len(path.recommended_prerequisites),
                    '並列学習可能': len(path.can_learn_in_parallel),
                    'アンロック': len(path.unlocks),
                    '成功率': f"{int(path.success_rate * 100)}%"
                })

            df_paths = pd.DataFrame(path_data)
            st.dataframe(df_paths, use_container_width=True, height=400)

            st.markdown(f"**表示中:** {len(filtered_paths)} / {len(learning_paths)} スキル")
        else:
            st.info("フィルタ条件に一致するスキルがありません")

    with tab3:
        st.markdown("### スキル詳細検索")

        # スキル選択
        skill_options = {
            path.competence_name: code
            for code, path in learning_paths.items()
        }

        selected_skill_name = st.selectbox(
            "スキルを選択してください",
            options=list(skill_options.keys())
        )

        if selected_skill_name:
            selected_code = skill_options[selected_skill_name]
            selected_path = learning_paths[selected_code]

            # スキル情報
            st.markdown("---")
            st.markdown(f"## 📚 {selected_path.competence_name}")

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:
                st.metric("タイプ", selected_path.competence_type)
            with info_col2:
                st.metric("難易度", selected_path.estimated_difficulty)
            with info_col3:
                st.metric("予測成功率", f"{int(selected_path.success_rate * 100)}%")

            # 前提スキル
            if selected_path.recommended_prerequisites:
                st.markdown("### 📖 推奨前提スキル")
                st.info("このスキルを学ぶ前に習得しておくと良いスキルです")

                for i, prereq in enumerate(selected_path.recommended_prerequisites, 1):
                    strength_badge = {
                        '強': 'badge-strong',
                        '中': 'badge-medium',
                        '弱': 'badge-weak'
                    }.get(prereq.get('dependency_strength', ''), '')

                    st.markdown(f"""
                    **{i}. {prereq['skill_name']}**
                    <span class="badge {strength_badge}">{prereq.get('dependency_strength', '不明')}</span>

                    - {prereq['reason']}
                    - 平均学習間隔: {prereq['average_time_gap_days']}日前
                    - 根拠: {prereq['evidence']}
                    """, unsafe_allow_html=True)

                # タイムライン表示
                st.markdown("---")
                st.markdown("#### ⏱️ 学習タイムライン")
                timeline_fig = create_learning_path_timeline(selected_path)
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.success("✨ このスキルは前提知識不要で学習可能です！")

            # 並列学習可能なスキル
            if selected_path.can_learn_in_parallel:
                st.markdown("---")
                st.markdown("### 🔀 並列学習可能なスキル")
                st.info("このスキルと同時に学んでも問題ないスキルです")

                for parallel in selected_path.can_learn_in_parallel:
                    st.markdown(f"- **{parallel.get('skill_name', parallel['skill_code'])}**: {parallel['reason']}")

            # このスキルを習得後に学べるスキル
            if selected_path.unlocks:
                st.markdown("---")
                st.markdown("### 🔓 アンロックされるスキル")
                st.info("このスキルを習得すると学べるようになるスキルです")

                for unlock in selected_path.unlocks:
                    st.markdown(f"- **{unlock['skill_name']}**: {unlock['reason']}")

            # ハイライトされた依存関係グラフ
            st.markdown("---")
            st.markdown("### 🎯 このスキルに関連する依存関係")

            highlight_fig = create_dependency_graph(
                graph_data,
                highlight_competence=selected_code
            )
            st.plotly_chart(highlight_fig, use_container_width=True)

else:
    st.info("👆 上の「依存関係を分析」ボタンをクリックして分析を開始してください")

# =========================================================
# ヘルプセクション
# =========================================================

with st.expander("❓ この機能について"):
    st.markdown("""
    ## スキル依存関係分析とは？

    この機能は、**時系列データ**から学習順序のパターンを抽出し、
    スキル間の依存関係を推定します。

    ### どうやって依存関係を推定しているの？

    1. **遷移パターンの抽出**
       - メンバーごとに、スキルの取得順序を時系列で分析
       - 一定期間内に連続して取得されたスキルペアを記録

    2. **信頼度の計算**
       - 「スキルAを学んだ人のうち、何%がその後スキルBを学んだか」を計算
       - これを「信頼度」として数値化

    3. **双方向比較**
       - A→Bの遷移とB→Aの遷移を比較
       - 一方向が圧倒的に多い場合、その方向を依存関係と判定

    ### 注意事項

    - これは「関連性」であり、「観測された学習パターン」です
    - 「AなしではBを学べない」という絶対的な依存関係ではありません
    - 「多くの人がこの順序で学んでいる」という傾向を示しています

    ### 活用方法

    - 新しいスキルを学ぶ際の参考情報として活用
    - 学習計画の立案に役立てる
    - 組織内の学習パターンを理解する
    """)
