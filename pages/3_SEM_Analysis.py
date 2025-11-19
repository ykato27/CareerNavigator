"""
SEM分析ページ

UnifiedSEM と HierarchicalSEM を使用した構造方程式モデリング分析。

主な機能:
- 統一SEM推定器による力量構造分析
- 階層的SEM推定器による大規模データ分析
- インタラクティブな可視化
- 標準的な適合度指標
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import importlib.util
from pathlib import Path

# プロジェクトルートを取得（環境非依存）
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
ml_dir = project_root / "skillnote_recommendation" / "ml"
graph_dir = project_root / "skillnote_recommendation" / "graph"

# UnifiedSEMEstimatorを直接import
def load_unified_sem():
    """UnifiedSEMEstimatorを動的にロード"""
    unified_sem_path = ml_dir / "unified_sem_estimator.py"

    spec = importlib.util.spec_from_file_location(
        "unified_sem_estimator",
        str(unified_sem_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_hierarchical_sem():
    """HierarchicalSEMEstimatorを動的にロード"""
    # まずUnifiedSEMをsys.modulesに登録
    unified_module = load_unified_sem()
    sys.modules['skillnote_recommendation.ml.unified_sem_estimator'] = unified_module

    hierarchical_sem_path = ml_dir / "hierarchical_sem_estimator.py"

    spec = importlib.util.spec_from_file_location(
        "hierarchical_sem_estimator",
        str(hierarchical_sem_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_sem_network_visualizer():
    """SEMNetworkVisualizerを動的にロード"""
    visualizer_path = graph_dir / "sem_network_visualizer.py"

    spec = importlib.util.spec_from_file_location(
        "sem_network_visualizer",
        str(visualizer_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_skill_dependency_analyzer():
    """SkillDependencyAnalyzerを動的にロード"""
    core_dir = project_root / "skillnote_recommendation" / "core"

    # Configをロード
    config_path = core_dir / "config.py"
    config_spec = importlib.util.spec_from_file_location("config", str(config_path))
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules['skillnote_recommendation.core.config'] = config_module
    config_spec.loader.exec_module(config_module)

    # SkillDependencyAnalyzerをロード
    analyzer_path = core_dir / "skill_dependency_analyzer.py"
    spec = importlib.util.spec_from_file_location(
        "skill_dependency_analyzer",
        str(analyzer_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_efa():
    """ExploratoryFactorAnalyzerを動的にロード"""
    core_dir = project_root / "skillnote_recommendation" / "core"
    efa_path = core_dir / "exploratory_factor_analysis.py"

    spec = importlib.util.spec_from_file_location(
        "exploratory_factor_analysis",
        str(efa_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_visualization_utils():
    """visualization utilsを動的にロード"""
    utils_dir = project_root / "skillnote_recommendation" / "utils"
    viz_path = utils_dir / "visualization.py"

    spec = importlib.util.spec_from_file_location(
        "visualization_utils",
        str(viz_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="CareerNavigator - SEM分析",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムスタイル
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .comparison-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.markdown("""
<div class="main-header">
    <h1>🧬 SEM分析</h1>
    <p>統一SEM推定器と階層的SEM推定器による構造方程式モデリング</p>
</div>
""", unsafe_allow_html=True)

# 重要な説明
st.info("""
🎯 **新機能**: このページでは最新のSEM実装を使用します

- ✅ **統一された目的関数**: 最尤推定による全パラメータ同時推定
- ✅ **明示的な共分散構造**: 力量同士の関係性を明確にモデル化
- ✅ **標準的な適合度指標**: RMSEA, CFI, TLI, AIC, BIC
- ✅ **スキル1000個対応**: 階層的推定により大規模データを高速処理
""")

# 使い方ガイド
with st.expander("📖 使い方ガイド", expanded=False):
    st.markdown("""
    ### 🚀 SEM分析の使い方

    **1. モデル選択**
    - **UnifiedSEM**: ~200スキルまでの標準的なSEM分析
    - **HierarchicalSEM**: 200~1000スキルの大規模データ分析

    **2. モデル構築**
    - ドメイン定義を設定（自動検出も可能）
    - 測定モデルと構造モデルを指定

    **3. 推定と評価**
    - 最尤推定による推定実行
    - 適合度指標で評価

    ### 💡 技術的背景

    **統一SEM推定器**:
    ```
    目的関数: F_ML(θ) = log|Σ(θ)| + tr(S·Σ⁻¹) - log|S| - p
    共分散構造: Σ(θ) = Λ·(I-B)⁻¹·Ψ·(I-B)⁻¹ᵀ·Λᵀ + Θ
    ```

    **階層的SEM推定器**:
    ```
    総合力量 → ドメイン力量 → 個別スキル (3層構造)
    推定時間: O(n_domains) × O(skills_per_domain)
    ```
    """)

# =========================================================
# データ確認
# =========================================================

# データがロードされているか確認
if 'transformed_data' not in st.session_state:
    st.warning("⚠️ データがロードされていません。まずデータ読み込みページでデータをアップロードしてください。")
    st.stop()

td = st.session_state.transformed_data

# デバッグ: transformed_dataの型を確認
if not isinstance(td, dict):
    st.error(f"❌ transformed_dataが辞書ではありません。型: {type(td)}")
    st.info("データ読み込みページで再度データをアップロードしてください。")
    st.stop()

# データが必要なキーを持っているか確認
required_keys = ["member_competence", "competence_master", "members_clean"]
missing_keys = [key for key in required_keys if key not in td]
if missing_keys:
    st.error(f"❌ 必要なデータが不足しています: {', '.join(missing_keys)}")
    st.info(f"利用可能なキー: {list(td.keys())}")
    st.info("データ読み込みページで再度データをアップロードしてください。")
    st.stop()

member_competence_all = td["member_competence"]
competence_master = td["competence_master"]
members_clean = td["members_clean"]

# =========================================================
# メンバーフィルタリング機能
# =========================================================

st.markdown("---")
st.subheader("👥 メンバーフィルタリング")

with st.expander("🔍 対象メンバーの選択", expanded=False):
    st.markdown("""
    分析対象とするメンバーを絞り込むことができます。

    - **全メンバー**: すべてのメンバーを対象にします（デフォルト）
    - **ユーザー選択**: 特定のメンバーを個別に選択します
    - **職能・等級選択**: 職能・等級、職種などで分類してフィルタリングします
    - **役職選択**: 役職別にメンバーをフィルタリングします
    """)

    # フィルタリングモード選択
    filter_mode = st.radio(
        "フィルタリングモード",
        options=["全メンバー", "ユーザー選択", "職能・等級選択", "役職選択", "詳細フィルタ（複合条件）"],
        index=0,
        help="分析対象のメンバーを絞り込む方法を選択してください"
    )

    # フィルタリング対象のメンバーコードを保存するリスト
    filtered_member_codes = None

    if filter_mode == "全メンバー":
        st.info("✅ 全メンバーを対象に分析します")
        filtered_member_codes = members_clean['メンバーコード'].tolist()

    elif filter_mode == "ユーザー選択":
        st.markdown("#### 👤 対象ユーザーの選択")

        # メンバー名とコードのマッピング
        member_options = [
            f"{row['メンバー名']} ({row['メンバーコード']})"
            for _, row in members_clean.iterrows()
        ]

        selected_members = st.multiselect(
            "メンバーを選択",
            options=member_options,
            help="分析対象とするメンバーを選択してください（複数選択可）"
        )

        if selected_members:
            # 選択されたメンバーのコードを抽出
            filtered_member_codes = [
                member.split("(")[-1].rstrip(")")
                for member in selected_members
            ]
            st.success(f"✅ {len(filtered_member_codes)}名のメンバーを選択しました")
        else:
            st.warning("⚠️ メンバーを選択してください")

    elif filter_mode == "職能・等級選択":
        st.markdown("#### 📊 職能・等級選択")

        # カラムを動的に検出（職能・等級を優先）
        classification_column = None
        classification_keywords = ["職能", "等級", "職種", "組織", "部署", "所属", "部門", "課", "グループ", "チーム"]

        for col in members_clean.columns:
            for keyword in classification_keywords:
                if keyword in col:
                    classification_column = col
                    break
            if classification_column:
                break

        # カラムが自動検出できない場合、ユーザーに選択させる
        if not classification_column:
            st.info("💡 カラムを自動検出できませんでした。使用するカラムを選択してください。")

            # メンバーコードとメンバー名以外のカラムを候補として表示
            exclude_cols = ["メンバーコード", "メンバー名", "ログインコード", "パスワード",
                          "メールアドレス", "よみがな", "生年月日", "SSOマッチングコード"]
            available_cols = [col for col in members_clean.columns if col not in exclude_cols]

            if available_cols:
                classification_column = st.selectbox(
                    "フィルタリングに使用するカラムを選択",
                    options=available_cols,
                    help="職種、職能・等級、社員区分などを選択できます"
                )
            else:
                st.error("❌ 利用可能なカラムがありません")
                st.stop()

        if classification_column:
            st.success(f"✅ 使用するカラム: **{classification_column}**")

            # 値の一覧を取得
            classification_values = members_clean[classification_column].dropna().unique().tolist()

            if len(classification_values) == 0:
                st.warning(f"⚠️ {classification_column}に有効な値がありません")
            else:
                selected_values = st.multiselect(
                    f"{classification_column}を選択",
                    options=classification_values,
                    help="分析対象とする値を選択してください（複数選択可）"
                )

                if selected_values:
                    filtered_members = members_clean[
                        members_clean[classification_column].isin(selected_values)
                    ]
                    filtered_member_codes = filtered_members['メンバーコード'].tolist()
                    st.success(f"✅ {len(filtered_member_codes)}名のメンバーを選択しました")
                else:
                    st.warning("⚠️ 値を選択してください")

    elif filter_mode == "役職選択":
        st.markdown("#### 💼 役職別フィルタリング")

        # 役職カラムを確認
        if "役職" in members_clean.columns:
            # 役職の一覧を取得
            position_values = members_clean["役職"].dropna().unique().tolist()

            selected_positions = st.multiselect(
                "役職を選択",
                options=position_values,
                help="分析対象とする役職を選択してください（複数選択可）"
            )

            if selected_positions:
                filtered_members = members_clean[
                    members_clean["役職"].isin(selected_positions)
                ]
                filtered_member_codes = filtered_members['メンバーコード'].tolist()
                st.success(f"✅ {len(filtered_member_codes)}名のメンバーを選択しました")
            else:
                st.warning("⚠️ 役職を選択してください")
        else:
            st.error("❌ 「役職」カラムが見つかりません")
            st.info(f"利用可能なカラム: {list(members_clean.columns)}")

    elif filter_mode == "詳細フィルタ（複合条件）":
        st.markdown("#### 🔧 詳細フィルタ（複合条件）")
        st.info("複数の条件を組み合わせてメンバーをフィルタリングできます")

        # フィルタリング条件を保存するリスト
        filter_conditions = []

        # ユーザー選択
        with st.container():
            st.markdown("##### 👤 ユーザー選択")
            use_member_filter = st.checkbox("ユーザーで絞り込む")

            if use_member_filter:
                member_options = [
                    f"{row['メンバー名']} ({row['メンバーコード']})"
                    for _, row in members_clean.iterrows()
                ]

                selected_members = st.multiselect(
                    "メンバーを選択",
                    options=member_options,
                    help="分析対象とするメンバーを選択してください（複数選択可）",
                    key="detail_members"
                )

                if selected_members:
                    selected_member_codes = [
                        member.split("(")[-1].rstrip(")")
                        for member in selected_members
                    ]
                    filter_conditions.append(
                        members_clean['メンバーコード'].isin(selected_member_codes)
                    )

        # 職能・等級選択
        with st.container():
            st.markdown("##### 📊 職能・等級選択")

            # カラムを動的に検出（職能・等級を優先）
            classification_column = None
            classification_keywords = ["職能", "等級", "職種", "組織", "部署", "所属", "部門", "課", "グループ", "チーム"]

            for col in members_clean.columns:
                for keyword in classification_keywords:
                    if keyword in col:
                        classification_column = col
                        break
                if classification_column:
                    break

            # カラムが自動検出できない場合、ユーザーに選択させる
            if not classification_column:
                # メンバーコードとメンバー名以外のカラムを候補として表示
                exclude_cols = ["メンバーコード", "メンバー名", "ログインコード", "パスワード",
                              "メールアドレス", "よみがな", "生年月日", "SSOマッチングコード"]
                available_cols = [col for col in members_clean.columns if col not in exclude_cols]

                if available_cols:
                    classification_column = st.selectbox(
                        "フィルタリングに使用するカラムを選択",
                        options=available_cols,
                        help="職種、職能・等級、社員区分などを選択できます",
                        key="detail_classification_column_select"
                    )

            if classification_column:
                use_classification_filter = st.checkbox(f"{classification_column}で絞り込む")

                if use_classification_filter:
                    classification_values = members_clean[classification_column].dropna().unique().tolist()

                    if len(classification_values) > 0:
                        selected_values = st.multiselect(
                            f"{classification_column}を選択",
                            options=classification_values,
                            help="分析対象とする値を選択してください（複数選択可）",
                            key="detail_classifications"
                        )

                        if selected_values:
                            filter_conditions.append(
                                members_clean[classification_column].isin(selected_values)
                            )
            else:
                st.caption("職能・等級に関するカラムが見つかりません")

        # 役職選択
        with st.container():
            st.markdown("##### 💼 役職選択")

            if "役職" in members_clean.columns:
                use_position_filter = st.checkbox("役職で絞り込む")

                if use_position_filter:
                    position_values = members_clean["役職"].dropna().unique().tolist()

                    selected_positions = st.multiselect(
                        "役職を選択",
                        options=position_values,
                        help="分析対象とする役職を選択してください（複数選択可）",
                        key="detail_positions"
                    )

                    if selected_positions:
                        filter_conditions.append(
                            members_clean["役職"].isin(selected_positions)
                        )
            else:
                st.caption("「役職」カラムが見つかりません")

        # フィルタ条件を適用
        if filter_conditions:
            # すべての条件をANDで結合
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition

            filtered_members = members_clean[combined_filter]
            filtered_member_codes = filtered_members['メンバーコード'].tolist()
            st.success(f"✅ {len(filtered_member_codes)}名のメンバーが条件に一致しました")
        else:
            st.warning("⚠️ 少なくとも1つの条件を設定してください")

# フィルタリング後のデータを作成
if filtered_member_codes is not None and len(filtered_member_codes) > 0:
    member_competence = member_competence_all[
        member_competence_all['メンバーコード'].isin(filtered_member_codes)
    ]
    members_clean_filtered = members_clean[
        members_clean['メンバーコード'].isin(filtered_member_codes)
    ]
    st.session_state['sem_filtered_member_codes'] = filtered_member_codes
else:
    # フィルタが設定されていない場合は全データを使用
    member_competence = member_competence_all
    members_clean_filtered = members_clean
    filtered_member_codes = members_clean['メンバーコード'].tolist()
    st.session_state['sem_filtered_member_codes'] = filtered_member_codes

# データサイズの表示
st.markdown("---")
st.subheader("📊 データ概要")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("対象メンバー数", len(members_clean_filtered))
with col2:
    n_skills = len(competence_master)
    st.metric("スキル数", n_skills)
with col3:
    n_records = len(member_competence)
    st.metric("習得記録数", n_records)
with col4:
    avg_skills = n_records / len(members_clean_filtered) if len(members_clean_filtered) > 0 else 0
    st.metric("平均習得数", f"{avg_skills:.1f}")

# =========================================================
# モデル選択
# =========================================================

st.markdown("---")
st.subheader("🎯 モデル選択")

# 2つの手法の違いを説明
with st.expander("❓ UnifiedSEMとHierarchicalSEMの違い", expanded=False):
    st.markdown("""
    ## 📊 2つの手法の違い

    UnifiedSEMとHierarchicalSEMは**全く異なる分析手法**です。結果が異なるのは正常です。

    ### 🔵 UnifiedSEM（統一型）

    **特徴:**
    - すべてのスキルとカテゴリーを**同時に1つのモデルで推定**
    - 全体の構造を統一的に把握できる
    - カテゴリー間の関係性も同時に分析

    **推定方法:**
    ```
    すべてのスキル → カテゴリー → 総合力量
    （1つの大きなモデルで一度に推定）
    ```

    **結果の見方:**
    - スキル間ネットワーク: すべてのスキルの関連性を一度に可視化
    - 測定モデル: すべてのカテゴリーのローディングを同時に表示
    - 構造モデル: カテゴリー間の関連性を表示

    **適用場面:**
    - スキル数が少ない場合（~200個）
    - 全体の構造を俯瞰したい場合
    - カテゴリー間の関係を知りたい場合

    ---

    ### 🟢 HierarchicalSEM（階層型）

    **特徴:**
    - カテゴリーごとに**独立したモデルを個別に推定**
    - その後、カテゴリー同士の関係を統合層で推定
    - 各カテゴリーの詳細な分析が可能

    **推定方法:**
    ```
    【段階1】各カテゴリーで独立に推定
    カテゴリーA: スキル1, 2, 3 → カテゴリーAスコア
    カテゴリーB: スキル4, 5, 6 → カテゴリーBスコア
    ...

    【段階2】統合層で推定
    カテゴリーAスコア、カテゴリーBスコア... → 総合力量
    ```

    **結果の見方:**
    - ドメイン別適合度: 各カテゴリーのモデルの精度を個別に評価
    - ドメインスコア: 各メンバーの各カテゴリーでのスキルレベル
    - 統合モデル: カテゴリー同士の関係性

    **適用場面:**
    - スキル数が多い場合（200~1000個）
    - 各カテゴリーの詳細を知りたい場合
    - メンバーのカテゴリー別スコアを知りたい場合

    ---

    ### 🔍 なぜ結果が違うのか？

    | 項目 | UnifiedSEM | HierarchicalSEM |
    |------|-----------|-----------------|
    | **推定単位** | 全体を一度に | カテゴリーごとに独立 |
    | **カテゴリー間の影響** | 考慮する | 第2段階でのみ考慮 |
    | **計算量** | O(全スキル数²) | O(カテゴリー数 × カテゴリー内スキル数²) |
    | **適合度指標** | 全体で1つ | カテゴリーごと + 統合層 |
    | **メンバースコア** | 総合力量のみ | カテゴリーごとのスコア |

    **結論:**
    - **UnifiedSEM**: 全体構造を把握したい → 俯瞰的な分析
    - **HierarchicalSEM**: 各カテゴリーの詳細を把握したい → 詳細な分析

    どちらも正しい結果ですが、**見ている視点が異なる**ため、結果も異なります。
    """)

model_type = st.radio(
    "使用するSEMモデル",
    options=["UnifiedSEM（実データ）", "HierarchicalSEM（実データ）"],
    index=0,
    help="データサイズに応じて適切なモデルを選択してください。UnifiedSEM: ~200スキル、HierarchicalSEM: 200~1000スキル"
)

# =========================================================
# UnifiedSEM（実データ）
# =========================================================

if model_type == "UnifiedSEM（実データ）":
    st.markdown("---")
    st.subheader("🎯 スキル選択")

    st.info("""
    **📊 UnifiedSEM分析について**

    UnifiedSEMは、スキル間の関連性を統計的にモデル化し、スキルネットワーク全体の構造を明らかにします。
    """)

    skill_selection_mode = st.radio(
        "**分析に使用するスキルの範囲を選択してください**",
        options=["🌐 全スキル使用（推奨）", "📂 カテゴリーで絞り込む"],
        index=0,
        help="""
        ・全スキル使用：すべてのスキルを対象に、全体のネットワーク構造を分析します（推奨）
        ・カテゴリーで絞り込む：特定のカテゴリーに絞って、詳細な構造を分析します
        """
    )

    # 利用可能なカテゴリーを取得
    available_categories = competence_master['力量カテゴリー名'].unique().tolist()
    available_categories = [cat for cat in available_categories if pd.notna(cat)]

    # カテゴリーごとのスキル数を計算
    category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()

    # 選択されたカテゴリーを保存する変数
    selected_categories = []
    selected_competences = pd.DataFrame()
    total_skills = 0

    if skill_selection_mode == "🌐 全スキル使用（推奨）":
        st.success("✅ **全スキルを対象にSEM分析を実行します**")
        st.markdown("""
        すべてのスキルを使用することで、組織全体のスキルネットワーク構造を包括的に把握できます。
        スキル間の関連性、カテゴリー間の関係性が明らかになります。
        """)

        # 全カテゴリーを自動選択
        selected_categories = available_categories
        selected_competences = competence_master
        total_skills = len(competence_master)

        # 統計情報の表示
        st.markdown("---")
        st.markdown("#### 📊 分析対象データ")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 カテゴリー数", len(selected_categories))
        with col2:
            st.metric("🎯 スキル総数", total_skills)
        with col3:
            recommend_model = "UnifiedSEM" if total_skills <= 200 else "HierarchicalSEM"
            st.metric("🔍 推奨モデル", recommend_model)

        if total_skills > 200:
            st.warning(
                f"⚠️ **スキル数が多い場合の推奨**\n\n"
                f"現在のスキル数: **{total_skills}個**\n\n"
                f"スキル数が200個を超える場合、より適切な結果を得るために **HierarchicalSEM** の使用を推奨します。\n\n"
                f"**または**、「📂 カテゴリーで絞り込む」を選択して特定カテゴリーに絞り込むこともできます。"
            )

    else:  # カテゴリーで絞り込む
        with st.expander("🔧 カテゴリー選択", expanded=True):
            st.markdown("### 力量カテゴリーの選択")
            st.info("特定のカテゴリーに絞り込んで分析することで、より詳細な構造を把握できます。")

            # ========================================
            # 初心者向け：推奨セット選択
            # ========================================
            st.markdown("#### 📋 推奨セットから選択（初心者向け）")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🎯 バランス型（推奨）", use_container_width=True):
                    # スキル数が50-150個になるようなセットを自動選択
                    target_skills = 100
                    threshold = 50
                    selected = []
                    total = 0
                    for cat in sorted(available_categories, key=lambda x: -category_counts.get(x, 0)):
                        cat_skills = category_counts.get(cat, 0)
                        if total + cat_skills <= target_skills + threshold:
                            selected.append(cat)
                            total += cat_skills
                        if len(selected) >= 5:  # 最大5カテゴリー
                            break
                    if selected:
                        st.session_state['unified_selected_categories'] = selected
                        st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            with col2:
                if st.button("📚 大規模型", use_container_width=True):
                    # スキル数が150-250個になるようなセットを自動選択
                    selected = []
                    total = 0
                    for cat in sorted(available_categories, key=lambda x: -category_counts.get(x, 0)):
                        cat_skills = category_counts.get(cat, 0)
                        if total + cat_skills <= 250:
                            selected.append(cat)
                            total += cat_skills
                        if len(selected) >= 8:  # 最大8カテゴリー
                            break
                    if selected:
                        st.session_state['unified_selected_categories'] = selected
                        st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            with col3:
                if st.button("⚡ コンパクト型", use_container_width=True):
                    # スキル数が20-50個になるようなセットを自動選択
                    selected = []
                    total = 0
                    for cat in sorted(available_categories, key=lambda x: -category_counts.get(x, 0)):
                        cat_skills = category_counts.get(cat, 0)
                        if total + cat_skills <= 50:
                            selected.append(cat)
                            total += cat_skills
                        if len(selected) >= 3:  # 最大3カテゴリー
                            break
                    if selected:
                        st.session_state['unified_selected_categories'] = selected
                        st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            # ========================================
            # 上級者向け：カテゴリー詳細調整
            # ========================================
            with st.expander("🔧 カテゴリーを詳細調整（上級者向け）", expanded=False):
                st.write("複数のカテゴリーを選択してください（推奨: 2~5カテゴリー、スキル数50~200個）")

                # 全件選択ボタン（チェックボックスではなくボタンで実装）
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    if st.button("🌍 全件選択", key="unified_select_all_btn", use_container_width=True):
                        # 全カテゴリーを選択してsession_stateに保存
                        st.session_state['unified_selected_categories'] = available_categories[:]
                        st.success(f"✅ 全{len(available_categories)}カテゴリーを選択しました")

                with col_b:
                    if st.button("🗑️ 選択解除", key="unified_clear_all_btn", use_container_width=True):
                        # 選択を解除
                        if 'unified_selected_categories' in st.session_state:
                            del st.session_state['unified_selected_categories']
                        st.info("選択を解除しました")

                # カテゴリー情報の表示
                category_info = [f"{cat} ({category_counts.get(cat, 0)}個)" for cat in available_categories]

                # session_stateから現在の選択を取得
                current_selection = []
                if 'unified_selected_categories' in st.session_state:
                    current_categories = st.session_state['unified_selected_categories']
                    current_selection = [f"{cat} ({category_counts.get(cat, 0)}個)"
                                        for cat in current_categories if cat in available_categories]

                selected_categories_display = st.multiselect(
                    "力量カテゴリー",
                    options=category_info,
                    default=current_selection,
                    help="複数のカテゴリーを選択してください。UnifiedSEMは200スキル程度まで推奨",
                    key="unified_multiselect"
                )

                # 表示名から実際のカテゴリー名を抽出
                if selected_categories_display:
                    selected_categories = [cat.rsplit(' (', 1)[0] for cat in selected_categories_display]
                    st.session_state['unified_selected_categories'] = selected_categories

            # ========================================
            # 選択状況の確認
            # ========================================
            if 'unified_selected_categories' in st.session_state:
                selected_categories = st.session_state['unified_selected_categories']
                selected_competences = competence_master[
                    competence_master['力量カテゴリー名'].isin(selected_categories)
                ]
                total_skills = len(selected_competences)

                st.markdown("---")
                st.markdown("#### 📊 選択状況")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("選択カテゴリー数", len(selected_categories))
                with col2:
                    st.metric("スキル総数", total_skills)
                with col3:
                    recommend_model = "UnifiedSEM" if total_skills <= 200 else "HierarchicalSEM"
                    st.metric("推奨モデル", recommend_model)

                if total_skills > 200:
                    st.warning(f"⚠️ スキル数が{total_skills}個と多いです。UnifiedSEMは200個程度まで推奨。HierarchicalSEMの使用を検討してください。")
                elif total_skills < 10:
                    st.error("❌ スキル数が少なすぎます。最低10個以上を選択してください。")
            else:
                selected_categories = []
                selected_competences = pd.DataFrame()
                total_skills = 0

    # EFAオプション
    st.markdown("---")
    st.markdown("### ⚙️ 分析オプション")

    use_efa = False
    n_efa_factors = None

    if total_skills >= 50:  # 50スキル以上でEFAオプションを表示
        with st.expander("🔬 探索的因子分析（EFA）オプション", expanded=(total_skills >= 150)):
            st.markdown("""
            **探索的因子分析（EFA）とは？**

            データから自動的に潜在因子を発見する手法です。事前に定義されたカテゴリーに依存せず、
            スキル間の相関構造から「実際にどのような能力の次元があるか」を統計的に推定します。

            **メリット:**
            - 🚀 **高速化**: 因子数が少なくなるため、大規模データ（150+スキル）で特に効果的
            - 📊 **データ駆動**: カテゴリー定義の誤りに影響されない
            - 🔍 **新発見**: 既存カテゴリーでは捉えられない能力の次元を発見できる可能性

            **推奨:**
            - スキル数150+: 強く推奨
            - スキル数100-149: 推奨
            - スキル数50-99: オプション
            """)

            if total_skills >= 150:
                st.info(f"💡 現在のスキル数（{total_skills}個）ではEFAの使用を強く推奨します。")
                default_use_efa = True
            elif total_skills >= 100:
                st.info(f"💡 現在のスキル数（{total_skills}個）ではEFAの使用を推奨します。")
                default_use_efa = True
            else:
                default_use_efa = False

            use_efa = st.checkbox(
                "探索的因子分析（EFA）を使用する",
                value=default_use_efa,
                help="データから自動的に潜在因子を発見します。因子数は自動決定されます。"
            )

            if use_efa:
                st.success("✅ EFAを使用します。因子数は自動決定されます（Kaiser基準 + 累積寄与率80%）")

                efa_advanced = st.checkbox("詳細設定", value=False)
                if efa_advanced:
                    col1, col2 = st.columns(2)
                    with col1:
                        manual_n_factors = st.number_input(
                            "因子数を手動指定（オプション）",
                            min_value=3,
                            max_value=20,
                            value=None,
                            help="Noneの場合は自動決定します"
                        )
                        if manual_n_factors:
                            n_efa_factors = int(manual_n_factors)
                    with col2:
                        st.caption("自動決定の場合、データの相関構造から最適な因子数が計算されます")

    # 実行ボタン
    st.markdown("---")
    st.markdown("### 🚀 分析実行")

    # ボタンの有効/無効の判定
    can_execute = bool(selected_categories) and total_skills >= 10

    if not can_execute:
        if not selected_categories:
            st.error("❌ カテゴリーが選択されていません。上記で「カテゴリーで絞り込む」を選択し、カテゴリーを選んでください。")
        elif total_skills < 10:
            st.error(f"❌ スキル数が少なすぎます（現在: {total_skills}個）。最低10個以上のスキルを含むカテゴリーを選択してください。")

    if st.button(
        "🚀 UnifiedSEM分析を開始",
        type="primary",
        disabled=not can_execute,
        use_container_width=True,
        help="選択したスキルを対象にSEM分析を実行します"
    ):
        with st.spinner("データを準備中..."):
            try:
                # データの準備: member_competence からピボットテーブルを作成
                selected_skill_codes = selected_competences['力量コード'].tolist()

                # フィルタリング
                filtered_mc = member_competence[
                    member_competence['力量コード'].isin(selected_skill_codes)
                ]

                # ピボット: 行=メンバー、列=力量コード、値=正規化レベル
                pivot_data = filtered_mc.pivot_table(
                    index='メンバーコード',
                    columns='力量コード',
                    values='正規化レベル',
                    aggfunc='first'
                ).fillna(0)  # 未習得は0

                st.success(f"✅ データ準備完了: {len(pivot_data)}人 × {len(pivot_data.columns)}スキル")

            except Exception as e:
                st.error(f"❌ データ準備エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())
                st.stop()

        # EFA使用判定とキャッシュ
        efa_result = None
        if use_efa:
            # キャッシュキーを作成
            skill_codes_key = "_".join(sorted(pivot_data.columns.tolist())[:10])  # 先頭10スキルでキー生成
            cache_key_efa = f"efa_{len(pivot_data.columns)}_{len(pivot_data)}_{skill_codes_key}"

            if cache_key_efa not in st.session_state:
                with st.spinner("探索的因子分析（EFA）実行中..."):
                    try:
                        # EFAモジュールロード
                        efa_module = load_efa()
                        ExploratoryFactorAnalyzer = efa_module.ExploratoryFactorAnalyzer

                        # スキル名マッピングを作成（ピボットデータのスキルコードに対応）
                        skill_code_to_name = dict(
                            zip(selected_competences['力量コード'], selected_competences['力量名'])
                        )

                        # EFA実行
                        efa = ExploratoryFactorAnalyzer(
                            pivot_data=pivot_data,
                            skill_name_mapping=skill_code_to_name,
                            n_factors=n_efa_factors,  # Noneの場合は自動決定
                        )
                        efa_result = efa.fit()

                        # キャッシュに保存
                        st.session_state[cache_key_efa] = efa_result

                        st.success(f"✅ EFA完了！{efa_result['n_factors']}個の因子を発見しました（累積寄与率: {np.sum(efa_result['explained_variance']):.1%}）")

                        # 因子解釈を表示
                        interpretation = efa.get_factor_interpretation(top_n=3)
                        with st.expander("🔍 発見された因子の解釈", expanded=True):
                            for factor_name, top_skills in interpretation.items():
                                st.markdown(f"**{factor_name}** (寄与率: {efa_result['explained_variance'][int(factor_name.replace('因子', ''))-1]:.1%})")
                                skills_text = ", ".join([f"{name}({loading:.2f})" for name, loading in top_skills])
                                st.caption(f"主要スキル: {skills_text}")

                    except Exception as e:
                        st.error(f"❌ EFA実行エラー: {e}")
                        import traceback
                        with st.expander("エラー詳細"):
                            st.code(traceback.format_exc())
                        st.stop()
            else:
                efa_result = st.session_state[cache_key_efa]
                st.info(f"✅ EFA結果をキャッシュから読み込みました（{efa_result['n_factors']}個の因子）")

        with st.spinner("UnifiedSEM推定中..."):
            try:
                # モジュールロード
                unified_sem_module = load_unified_sem()
                UnifiedSEMEstimator = unified_sem_module.UnifiedSEMEstimator
                MeasurementModelSpec = unified_sem_module.MeasurementModelSpec
                StructuralModelSpec = unified_sem_module.StructuralModelSpec

                if use_efa and efa_result:
                    # EFAベースの測定モデル仕様
                    st.info("🔬 EFAで発見した因子を使用してSEMを構築します")

                    measurement_specs = []
                    valid_factors = []

                    # 各スキルの主因子を特定（最大ローディング）
                    # これにより、各スキルは1つの因子にのみ割り当てられる
                    from collections import defaultdict
                    skill_primary_factor = {}

                    for skill_idx, skill_code in enumerate(efa_result['skill_codes']):
                        loadings_for_skill = efa_result['factor_loadings'][skill_idx, :]
                        max_loading_idx = np.argmax(np.abs(loadings_for_skill))
                        max_loading = np.abs(loadings_for_skill[max_loading_idx])

                        # 閾値以上の場合のみ割り当て
                        if max_loading > 0.3:
                            skill_primary_factor[skill_code] = max_loading_idx

                    # 因子ごとにスキルをグループ化
                    factor_to_skills = defaultdict(list)
                    for skill_code, factor_idx in skill_primary_factor.items():
                        # ピボットデータに存在するスキルのみ
                        if skill_code in pivot_data.columns:
                            factor_to_skills[factor_idx].append(skill_code)

                    # measurement_specs作成
                    for factor_idx in range(efa_result['n_factors']):
                        factor_name = efa_result['factor_names'][factor_idx]
                        factor_skills = factor_to_skills.get(factor_idx, [])

                        if len(factor_skills) >= 2:
                            measurement_specs.append(
                                MeasurementModelSpec(
                                    latent_name=factor_name,
                                    observed_vars=factor_skills,
                                    reference_indicator=factor_skills[0]
                                )
                            )
                            valid_factors.append(factor_name)

                    # 割り当てられたスキル数を表示
                    total_assigned = sum(len(skills) for skills in factor_to_skills.values())
                    st.caption(f"📊 {total_assigned}個のスキルを{len(valid_factors)}個の因子に割り当てました（各スキルは主因子のみに割り当て）")

                    # 構造モデル仕様
                    structural_specs = []
                    for i, from_factor in enumerate(valid_factors):
                        for j, to_factor in enumerate(valid_factors):
                            if i < j:
                                structural_specs.append(
                                    StructuralModelSpec(from_latent=from_factor, to_latent=to_factor)
                                )

                    st.info(f"📐 EFAモデル: {len(measurement_specs)}個の因子、構造モデル: {len(structural_specs)}個のパス")

                else:
                    # カテゴリーベースの測定モデル仕様（従来）
                    measurement_specs = []
                    valid_categories = []
                    for category in selected_categories:
                        cat_competences = selected_competences[
                            selected_competences['力量カテゴリー名'] == category
                        ]
                        skill_codes = cat_competences['力量コード'].tolist()

                        # ピボットデータに存在するスキルのみを使用
                        skill_codes = [code for code in skill_codes if code in pivot_data.columns]

                        if len(skill_codes) >= 2:  # 最低2個のスキルが必要
                            measurement_specs.append(
                                MeasurementModelSpec(
                                    latent_name=category,
                                    observed_vars=skill_codes,
                                    reference_indicator=skill_codes[0]  # 最初のスキルを参照指標に
                                )
                            )
                            valid_categories.append(category)  # 有効なカテゴリーを記録

                    # 構造モデル仕様の作成（測定モデルに含まれるカテゴリーのみ使用）
                    structural_specs = []
                    for i, from_cat in enumerate(valid_categories):
                        for j, to_cat in enumerate(valid_categories):
                            if i < j:  # 上三角のみ（一方向の関係）
                                structural_specs.append(
                                    StructuralModelSpec(from_latent=from_cat, to_latent=to_cat)
                                )

                    # 除外されたカテゴリーを警告
                    excluded_categories = set(selected_categories) - set(valid_categories)
                    if excluded_categories:
                        st.warning(f"⚠️ スキル数が2個未満のため除外されたカテゴリー: {', '.join(excluded_categories)}")

                    st.info(f"📐 測定モデル: {len(measurement_specs)}個の潜在変数、構造モデル: {len(structural_specs)}個のパス")

                # UnifiedSEM推定
                sem = UnifiedSEMEstimator(measurement_specs, structural_specs, method='ML')
                sem.fit(pivot_data)

                # 推定結果をsession_stateに保存（スライダー変更時も結果を保持）
                st.session_state['unified_sem_result'] = sem
                st.session_state['unified_sem_selected_competences'] = selected_competences
                st.session_state['unified_sem_use_efa'] = use_efa

                if use_efa:
                    st.success(f"✅ 推定完了！EFAで発見した{efa_result['n_factors']}個の因子を使用したSEMモデルが構築されました。")
                else:
                    st.success("✅ 推定完了！結果は下部に表示されます。")

            except Exception as e:
                st.error(f"❌ 推定エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())

    # =========================================================
    # 結果表示セクション（ボタンブロックの外）
    # =========================================================
    if 'unified_sem_result' in st.session_state:
        sem = st.session_state['unified_sem_result']
        selected_competences = st.session_state['unified_sem_selected_competences']

        # 結果表示
        st.markdown("---")
        st.subheader("📊 推定結果")

        # 適合度指標
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 適合度指標")
            fit = sem.fit_indices

            metrics_df = pd.DataFrame({
                '指標': ['RMSEA', 'CFI', 'TLI', 'GFI', 'SRMR', 'AIC', 'BIC'],
                '値': [
                    f"{fit.rmsea:.3f}",
                    f"{fit.cfi:.3f}",
                    f"{fit.tli:.3f}",
                    f"{fit.gfi:.3f}",
                    f"{fit.srmr:.3f}",
                    f"{fit.aic:.1f}",
                    f"{fit.bic:.1f}",
                ],
                '判定基準': [
                    '< 0.06 (良好)',
                    '> 0.95 (良好)',
                    '> 0.95 (良好)',
                    '> 0.90 (良好)',
                    '< 0.06 (良好)',
                    '小さいほど良い',
                    '小さいほど良い',
                ]
            })

            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # 総合判定
            if fit.is_excellent_fit():
                st.success("✅ 優れた適合度です！")
            elif fit.is_good_fit():
                st.info("✅ 良好な適合度です")
            else:
                st.warning("⚠️ 適合度が低いです。モデル仕様の見直しを推奨します。")

        with col2:
            st.markdown("### 構造係数（力量カテゴリー間の関係性）")
            relationships = sem.get_skill_relationships()

            if len(relationships) > 0:
                st.dataframe(
                    relationships[['from_skill', 'to_skill', 'coefficient', 'p_value', 'is_significant']],
                    use_container_width=True,
                    hide_index=True
                )

                # 構造係数の可視化
                fig = go.Figure()

                for _, row in relationships.iterrows():
                    color = 'green' if row['is_significant'] else 'gray'
                    fig.add_trace(go.Bar(
                        x=[f"{row['from_skill']}→{row['to_skill']}"],
                        y=[row['coefficient']],
                        marker_color=color,
                        name='有意' if row['is_significant'] else '非有意',
                        showlegend=False,
                    ))

                fig.update_layout(
                    title='構造係数の大きさ',
                    yaxis_title='係数',
                    height=300,
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("構造パスが定義されていません")

        # ファクターローディング
        st.markdown("### ファクターローディング行列")

        loading_df = pd.DataFrame(
            sem.Lambda,
            index=sem.observed_vars,
            columns=sem.latent_vars
        )

        # 力量コードを力量名に変換して表示
        skill_code_to_name = dict(zip(
            competence_master['力量コード'],
            competence_master['力量名']
        ))
        loading_df.index = [skill_code_to_name.get(code, code) for code in loading_df.index]

        # ヒートマップ
        fig = px.imshow(
            loading_df.T,
            labels=dict(x="スキル", y="潜在変数", color="ローディング"),
            aspect="auto",
            color_continuous_scale='RdBu_r',
        )
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

        # ============================================
        # ネットワークグラフ可視化
        # ============================================
        st.markdown("---")
        st.markdown("## 📊 ネットワークグラフ可視化")

        with st.spinner("ネットワークグラフを生成中..."):
            try:
                # グラフ可視化モジュールをロード
                visualizer_module = load_sem_network_visualizer()
                SEMNetworkVisualizer = visualizer_module.SEMNetworkVisualizer

                visualizer = SEMNetworkVisualizer()

                # タブで表示方法を選択（スキル間ネットワークを最初に）
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["🕸️ スキル間ネットワーク", "📈 統合モデル", "🔬 測定モデル", "🎯 カテゴリー間因果関係"]
                )

                with tab1:
                    st.markdown(
                        "### スキル間ネットワーク（有向グラフ）\n"
                        "スキル間の学習順序・前提関係を可視化"
                    )

                    st.info("""
                    **📊 グラフの方向性について:**
                    - 取得日データがある場合: 実際の学習パターンから方向性を推定（A→B = Aを先に学ぶべき）
                    - 取得日データがない場合: カテゴリー内の関連性を表示（無向グラフ）
                    """)

                    # スキルコード → スキル名（日本語）のマッピングを作成
                    skill_code_to_name = dict(zip(
                        competence_master['力量コード'],
                        competence_master['力量名']
                    ))

                    # スキルコード → カテゴリー名のマッピングを作成
                    skill_code_to_category = dict(zip(
                        competence_master['力量コード'],
                        competence_master['力量カテゴリー名']
                    ))

                    # 設定エリア
                    st.markdown("#### ⚙️ 表示設定")

                    # ネットワーク表示モード選択
                    st.markdown("##### 🎯 表示モード")
                    network_display_mode = st.radio(
                        "ネットワークの範囲を選択",
                        options=["全スキル表示", "カテゴリー別表示", "個別スキル選択"],
                        index=0,
                        help="表示するスキルの範囲を選択してください",
                        key="unified_network_display_mode",
                        horizontal=True
                    )

                    # フィルタリング対象のスキルコードリスト
                    filtered_skill_codes = sem.observed_vars.copy()

                    if network_display_mode == "カテゴリー別表示":
                        st.markdown("##### 📂 カテゴリー選択")
                        # 分析に使用されているカテゴリーのみを抽出
                        categories_in_analysis = set()
                        for skill_code in sem.observed_vars:
                            category = skill_code_to_category.get(skill_code)
                            if category:
                                categories_in_analysis.add(category)

                        categories_list = sorted(list(categories_in_analysis))

                        if len(categories_list) > 0:
                            selected_category = st.selectbox(
                                "表示するカテゴリーを選択",
                                options=categories_list,
                                help="選択したカテゴリーに属するスキルのみを表示します"
                            )

                            # 選択されたカテゴリーに属するスキルのみをフィルタ
                            filtered_skill_codes = [
                                code for code in sem.observed_vars
                                if skill_code_to_category.get(code) == selected_category
                            ]

                            st.info(f"✅ {selected_category}: {len(filtered_skill_codes)}個のスキル")
                        else:
                            st.warning("⚠️ カテゴリー情報が見つかりません")

                    elif network_display_mode == "個別スキル選択":
                        st.markdown("##### 🔍 スキル選択")

                        # スキル名のリストを作成（コード付き）
                        skill_options = [
                            f"{skill_code_to_name.get(code, code)} ({code})"
                            for code in sem.observed_vars
                        ]

                        selected_skills = st.multiselect(
                            "表示するスキルを選択",
                            options=skill_options,
                            help="選択したスキルとその関連スキルのみを表示します（最大20個推奨）"
                        )

                        if selected_skills:
                            # 選択されたスキルのコードを抽出
                            filtered_skill_codes = [
                                skill.split("(")[-1].rstrip(")")
                                for skill in selected_skills
                            ]
                            st.info(f"✅ {len(filtered_skill_codes)}個のスキルを選択")
                        else:
                            st.warning("⚠️ スキルを選択してください")
                            filtered_skill_codes = []

                    # メンバー選択
                    st.markdown("##### 👤 メンバー別表示（オプション）")
                    # フィルタリング後のメンバーを使用
                    member_names = members_clean_filtered['メンバー名'].tolist()
                    member_codes = members_clean_filtered['メンバーコード'].tolist()

                    member_options = ["（全体表示）"] + [f"{name} ({code})" for name, code in zip(member_names, member_codes)]

                    selected_member_display = st.selectbox(
                        "メンバーを選択",
                        options=member_options,
                        help="メンバーを選択すると、そのメンバーの取得済み/未取得力量が色分けされます",
                        key="unified_sem_selected_member"
                    )

                    # 選択されたメンバーの取得済みスキルを取得
                    acquired_skills = None
                    if selected_member_display != "（全体表示）":
                        # メンバーコードを抽出
                        selected_member_code = selected_member_display.split("(")[-1].rstrip(")")

                        # このメンバーの取得済みスキルを取得
                        member_skills = member_competence[
                            member_competence['メンバーコード'] == selected_member_code
                        ]['力量コード'].tolist()
                        acquired_skills = set(member_skills)

                        st.caption(f"✅ 取得済み力量: {len(acquired_skills)}個")

                    st.markdown("---")

                    col_threshold, col_edge = st.columns(2)

                    with col_threshold:
                        loading_threshold = st.slider(
                            "ローディング閾値",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.2,
                            step=0.05,
                            help="この値以上のファクターローディングを持つ力量のみ表示します。値を下げると表示される力量が増えます。",
                            key="unified_sem_loading_threshold",
                        )
                        st.caption(f"現在の閾値: {loading_threshold:.2f}")

                    # 全接続数を計算（edge_limit なしで実行、loading_threshold を使用）
                    temp_edges = []
                    for j in range(len(sem.latent_vars)):
                        contributing_skills = [
                            (i, abs(sem.Lambda[i, j]))
                            for i in range(len(sem.observed_vars))
                            if abs(sem.Lambda[i, j]) > loading_threshold
                        ]
                        for k1 in range(len(contributing_skills)):
                            for k2 in range(k1 + 1, len(contributing_skills)):
                                temp_edges.append(True)

                    max_edges = len(temp_edges)

                    with col_edge:
                        # スライダーで表示する接続数の範囲を調整（session_state で状態保持）
                        slider_start_key = "unified_sem_skill_network_edge_start"
                        slider_end_key = "unified_sem_skill_network_edge_end"

                        # max_edges が変更された場合、スライダーの値を調整
                        if slider_start_key not in st.session_state:
                            st.session_state[slider_start_key] = 1 if max_edges > 0 else 1

                        if slider_end_key not in st.session_state:
                            st.session_state[slider_end_key] = min(20, max_edges) if max_edges > 0 else 1

                        # max_edges を超えないようにvalidate
                        if st.session_state[slider_end_key] > max_edges and max_edges > 0:
                            st.session_state[slider_end_key] = max_edges

                        st.markdown("##### 接続範囲指定（関係性が強い順）")

                        # 開始位置スライダー
                        edge_start = st.slider(
                            "開始位置（番目から）",
                            min_value=1,
                            max_value=max(1, max_edges),
                            step=1,
                            help=f"最小: 1、最大: {max_edges}",
                            key=slider_start_key,
                        )

                        # 終了位置スライダー
                        edge_end = st.slider(
                            "終了位置（番目まで）",
                            min_value=1,
                            max_value=max(1, max_edges),
                            step=1,
                            help=f"開始位置以上の値で指定してください。最大: {max_edges}",
                            key=slider_end_key,
                        )

                        # 開始位置が終了位置より大きい場合は調整
                        if edge_start > edge_end:
                            edge_start, edge_end = edge_end, edge_start
                            st.warning(f"開始位置が終了位置より大きいため、自動調整しました: {edge_start}～{edge_end}")

                        st.caption(f"表示中: {edge_start}～{edge_end}番目 （全 {max_edges} 接続）")

                    st.markdown("---")

                    # 学習順序分析（取得日データがある場合）
                    dependency_edges = None
                    use_learning_order = False

                    if '取得日' in member_competence.columns:
                        use_learning_order = st.checkbox(
                            "🎓 学習順序ロジックを使用（取得日データから分析）",
                            value=True,
                            help="実際の取得パターンから学習順序を推定し、有向グラフの方向性を決定します",
                            key="unified_use_learning_order"
                        )

                        if use_learning_order:
                            # キャッシュキーを作成（メンバーフィルタリング状態を含む）
                            cache_key = f"unified_dep_{len(st.session_state.get('filtered_member_codes', []))}"

                            if cache_key not in st.session_state:
                                with st.spinner("学習順序を分析中..."):
                                    try:
                                        # フィルタリングされたメンバーの力量データを取得
                                        if 'filtered_member_codes' in st.session_state and st.session_state.filtered_member_codes:
                                            filtered_competence = member_competence[
                                                member_competence['メンバーコード'].isin(st.session_state.filtered_member_codes)
                                            ]
                                        else:
                                            filtered_competence = member_competence

                                        # SkillDependencyAnalyzerをロード
                                        analyzer_module = load_skill_dependency_analyzer()
                                        SkillDependencyAnalyzer = analyzer_module.SkillDependencyAnalyzer

                                        # アナライザーを初期化（デフォルトパラメータ）
                                        analyzer = SkillDependencyAnalyzer(
                                            member_competence=filtered_competence,
                                            competence_master=competence_master,
                                            time_window_days=180,
                                            min_transition_count=2,
                                            confidence_threshold=0.2,
                                        )

                                        # グラフデータを取得
                                        graph_data = analyzer.get_dependency_graph_data()

                                        # セッション状態に保存
                                        st.session_state[cache_key] = graph_data.get('edges', [])

                                        st.success(f"✅ 学習順序分析完了！{len(st.session_state[cache_key])}個の依存関係を検出")

                                    except Exception as e:
                                        st.warning(f"⚠️ 学習順序分析エラー: {e}")
                                        st.info("Lambda行列ベースのネットワークを表示します")
                                        st.session_state[cache_key] = []

                            dependency_edges = st.session_state.get(cache_key, [])

                    # フィルタリングされたスキルに対応するLambda行列の行インデックスを取得
                    if len(filtered_skill_codes) > 0:
                        filtered_indices = [
                            i for i, code in enumerate(sem.observed_vars)
                            if code in filtered_skill_codes
                        ]

                        # フィルタリングされた行のみを抽出
                        import numpy as np
                        filtered_lambda = sem.Lambda[filtered_indices, :]

                        # パフォーマンス最適化の適用状況を表示
                        n_skills = len(filtered_skill_codes)
                        if n_skills >= 200:
                            st.info(f"⚡ 超大規模グラフ最適化適用中（{n_skills}スキル）: Kamada-Kawai レイアウト + 高度なエッジ削減")
                        elif n_skills >= 150:
                            st.info(f"⚡ 大規模グラフ最適化適用中（{n_skills}スキル）: Kamada-Kawai レイアウト + エッジ削減")
                        elif n_skills >= 100:
                            st.info(f"⚡ 中規模グラフ最適化適用中（{n_skills}スキル）: 高速レイアウト + エッジ制限")

                        fig_skill_network = visualizer.visualize_skill_network(
                            lambda_matrix=filtered_lambda,
                            latent_vars=sem.latent_vars,
                            observed_vars=filtered_skill_codes,
                            skill_name_mapping=skill_code_to_name,
                            loading_threshold=loading_threshold,
                            edge_limit_start=edge_start,
                            edge_limit_end=edge_end,
                            acquired_skills=acquired_skills,
                            dependency_edges=dependency_edges if dependency_edges else None,
                        )
                        st.plotly_chart(fig_skill_network, use_container_width=True)

                        # 使用したロジックを表示
                        if use_learning_order and dependency_edges:
                            st.caption(f"🎓 学習順序ロジック使用中（{len(dependency_edges)}個の依存関係）")
                        else:
                            st.caption("📊 Lambda行列ベースのネットワーク")
                    else:
                        st.warning("⚠️ 表示するスキルがありません。スキルを選択してください。")

                with tab2:
                    st.markdown(
                        "### 📊 統合SEM構造（全体像）\n"
                        "スキル習得 → 力量カテゴリー形成 → キャリア発展の構造"
                    )

                    with st.expander("📖 この図の見方", expanded=True):
                        st.markdown("""
                        #### 構造図
                        ```
                            力量カテゴリーA        力量カテゴリーB
                            （青い丸）            （青い丸）
                                 ▲                    ▲
                                 │ ローディング        │ ローディング
                                 │ (関係の強さ)       │
                            ─────┴─────            ─────┴─────
                            Python基礎  Git        SQL基礎  DB設計
                            （マゼンタ丸）         （マゼンタ丸）
                                 ◀ スキル ▶

                        力量カテゴリーA  ──→ 力量カテゴリーB
                            (関連性の矢印)
                        ```

                        #### 色・太さの意味
                        - **マゼンタ丸（●）**: スキル（習得する具体的な技術）
                          - Python基礎、Git、SQL基礎、DB設計 など
                        - **青い丸（●）**: 力量カテゴリー（複合的な能力）
                          - 初級力量、中級力量、システム設計力 など
                        - **矢印の太さ**: 関係の強さ
                          - 太い → 強い関係
                          - 細い → 弱い関係
                        - **緑色の矢印**: 統計的に有意な関連性
                        - **グレーの矢印**: 統計的に有意でない可能性

                        #### このタブで分かること
                        1. **スキル→力量**: どのスキルがどの力量に貢献しているか
                        2. **力量→力量**: 力量カテゴリー間の発展段階
                        3. **全体パス**: 初級スキル→高度な力量への学習パス
                        """)


                    # パス有意性の辞書を作成
                    path_significance = {}
                    relationships = sem.get_skill_relationships()
                    for _, row in relationships.iterrows():
                        path_significance[(row["from_skill"], row["to_skill"])] = (
                            row["is_significant"]
                        )

                    # スキル名マッピングの作成
                    skill_code_to_name = dict(zip(
                        competence_master['力量コード'],
                        competence_master['力量名']
                    ))

                    fig_combined = visualizer.visualize_combined_model(
                        lambda_matrix=sem.Lambda,
                        b_matrix=sem.B,
                        latent_vars=sem.latent_vars,
                        observed_vars=sem.observed_vars,
                        loading_threshold=0.2,
                        path_significance=path_significance,
                        skill_name_mapping=skill_code_to_name,
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)

                with tab3:
                    st.markdown(
                        "### 🔬 測定モデル（スキル→力量）\n"
                        "各スキルが力量カテゴリーの形成にどの程度貢献しているか"
                    )

                    with st.expander("📖 この図の見方", expanded=True):
                        st.markdown("""
                        #### 構造図
                        ```
                        スキル層          力量カテゴリー層
                        （左側）          （右側）

                        Python基礎 ──────┐
                        Git      ──────→ 初級力量カテゴリー
                        SQL基礎   ──────┘

                        Webフレーム ──────┐
                        Docker    ──────→ 開発技術力量カテゴリー
                        Linux     ──────┘
                        ```

                        #### 矢印の意味
                        - **太い矢印**: 強いローディング（0.7~1.0）
                          - 例：「Python基礎」は「初級力量」の形成に大きく貢献
                        - **細い矢印**: 弱いローディング（0.3~0.5）
                          - 例：「Git」は「初級力量」に多少貢献

                        #### ローディングとは
                        - 0.0〜1.0の値
                        - **0.7以上**: スキルは重要（学習必須）
                        - **0.5~0.7**: スキルはまあまあ重要
                        - **0.3~0.5**: スキルは補助的

                        #### このタブで分かること
                        1. **各スキルの重要度**: どのスキルが力量形成に欠かせないか
                        2. **スキル選択**: 限られた時間で何から習得すべきか
                        3. **関連スキル**: 特定の力量を身につけるために必要なスキルセット
                        """)


                    fig_measurement = visualizer.visualize_measurement_model(
                        lambda_matrix=sem.Lambda,
                        latent_vars=sem.latent_vars,
                        observed_vars=sem.observed_vars,
                        loading_threshold=0.2,
                        skill_name_mapping=skill_code_to_name,
                    )
                    st.plotly_chart(fig_measurement, use_container_width=True)

                with tab4:
                    st.markdown(
                        "### 🎯 カテゴリー間因果関係（有向グラフ）\n"
                        "力量カテゴリー間の因果関係と学習発展段階"
                    )

                    with st.expander("📖 この図の見方", expanded=True):
                        st.markdown("""
                        #### 構造図（キャリア発展段階）
                        ```
                        初級力量 ──→ 中級力量 ──→ 上級力量
                        （基礎）    （応用）     （エキスパート）

                        例：プログラミング分野
                        基礎スキル習得 → 実務開発経験 → アーキテクチャ設計
                        ```

                        #### 矢印の意味
                        - **緑色の矢印（→）**: 統計的に有意な関連性
                          - p値 < 0.05（関係がある確率95%以上）
                          - 実務で確認されている段階的成長
                        - **グレーの矢印（→）**: 統計的に有意でない
                          - 直接的な関連性が見つからない可能性
                          - 他の要因を経由して影響する可能性

                        #### 矢印の太さ
                        - **太い矢印**: 因果係数が大きい（強い影響）
                          - 例：初級力量 → 中級力量（係数0.8）
                          - 初級力量の習得が中級力量習得に大きく貢献
                        - **細い矢印**: 因果係数が小さい（弱い影響）
                          - 例：初級力量 → 上級力量（係数0.2）
                          - 直接的な寄与は小さい

                        #### 因果係数（Path Coefficient）
                        - -1.0〜+1.0の値
                        - **0.7以上**: 強い影響
                        - **0.3~0.7**: 中程度の影響
                        - **0.3未満**: 弱い影響

                        #### このタブで分かること
                        1. **学習段階**: スキル習得の最適な順序
                        2. **前提条件**: 高度な力量を習得する前に何を習得すべきか
                        3. **キャリアパス**: メンバーのキャリア発展の方向性
                        4. **効率性**: どの力量習得が次のステップに最も貢献するか
                        """)

                    fig_structural = visualizer.visualize_structural_model(
                        b_matrix=sem.B,
                        latent_vars=sem.latent_vars,
                        path_significance=path_significance,
                    )
                    st.plotly_chart(fig_structural, use_container_width=True)

                st.success("✅ ネットワークグラフを生成しました")

            except Exception as e:
                st.error(f"❌ グラフ生成エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())

        # 詳細データ
        with st.expander("📋 詳細データ"):
            st.markdown("#### ファクターローディング")
            st.dataframe(loading_df, use_container_width=True)

            st.markdown("#### 構造係数行列 B")
            st.dataframe(
                pd.DataFrame(sem.B, index=sem.latent_vars, columns=sem.latent_vars),
                use_container_width=True
            )

            st.markdown("#### 潜在変数の分散 Ψ")
            st.dataframe(
                pd.DataFrame(sem.Psi, index=sem.latent_vars, columns=sem.latent_vars),
                use_container_width=True
            )

# =========================================================
# HierarchicalSEM（実データ）
# =========================================================

elif model_type == "HierarchicalSEM（実データ）":
    st.markdown("---")
    st.subheader("🎯 スキル選択")

    st.info("""
    **📊 HierarchicalSEM分析について**

    HierarchicalSEMは、大規模なスキルデータ（200~1000個）に対応した階層的な分析手法です。
    カテゴリーごとに独立してモデルを推定し、その後統合層で全体の関係性を明らかにします。
    """)

    skill_selection_mode_hier = st.radio(
        "**分析に使用するスキルの範囲を選択してください**",
        options=["🌐 全スキル使用（推奨）", "📂 カテゴリーで絞り込む"],
        index=0,
        help="""
        ・全スキル使用：すべてのスキルを対象に、階層的に分析します（推奨）
        ・カテゴリーで絞り込む：特定のカテゴリーに絞って分析します
        """,
        key="hier_skill_selection_mode"
    )

    # 利用可能なカテゴリーを取得
    available_categories = competence_master['力量カテゴリー名'].unique().tolist()
    available_categories = [cat for cat in available_categories if pd.notna(cat)]

    # カテゴリーごとのスキル数を計算
    category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()

    # 選択されたカテゴリーを保存する変数
    selected_categories = []
    selected_competences = pd.DataFrame()
    total_skills = 0

    if skill_selection_mode_hier == "🌐 全スキル使用（推奨）":
        st.success("✅ **全スキルを対象にHierarchicalSEM分析を実行します**")
        st.markdown("""
        すべてのスキルを使用することで、組織全体のスキル構造を階層的に把握できます。
        各カテゴリーの詳細な分析と、カテゴリー間の関係性が明らかになります。
        """)

        # 全カテゴリーを自動選択
        selected_categories = available_categories
        selected_competences = competence_master
        total_skills = len(competence_master)

        # 統計情報の表示
        st.markdown("---")
        st.markdown("#### 📊 分析対象データ")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 カテゴリー数", len(selected_categories))
        with col2:
            st.metric("🎯 スキル総数", total_skills)
        with col3:
            if total_skills <= 400:
                est_time = "~5分"
            elif total_skills <= 800:
                est_time = "5-15分"
            else:
                est_time = "15分以上"
            st.metric("⏱️ 推定時間", est_time)

        if total_skills > 1000:
            st.warning(
                f"⚠️ **スキル数が非常に多い場合**\n\n"
                f"現在のスキル数: **{total_skills}個**\n\n"
                f"処理に時間がかかる場合があります。「📂 カテゴリーで絞り込む」を選択して特定カテゴリーに絞り込むことも検討してください。"
            )

    else:  # カテゴリーで絞り込む
        with st.expander("🔧 カテゴリー選択", expanded=True):
            st.markdown("### 力量カテゴリーの選択")
            st.info("特定のカテゴリーに絞り込んで階層的に分析することで、より詳細な構造を把握できます。")

            # ========================================
            # 初心者向け：推奨セット選択
            # ========================================
            st.markdown("#### 📋 推奨セットから選択（初心者向け）")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🎯 標準型（推奨）", use_container_width=True, key="hier_standard"):
                    # スキル数が200-400個になるようなセットを自動選択
                    selected = []
                    total = 0
                    for cat in sorted(available_categories, key=lambda x: -category_counts.get(x, 0)):
                        cat_skills = category_counts.get(cat, 0)
                        if total + cat_skills <= 400:
                            selected.append(cat)
                            total += cat_skills
                        if len(selected) >= 8:  # 最大8カテゴリー
                            break
                    if selected:
                        st.session_state['hierarchical_selected_categories'] = selected
                        st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            with col2:
                if st.button("📚 大規模型", use_container_width=True, key="hier_large"):
                    # スキル数が400-800個になるようなセットを自動選択
                    selected = []
                    total = 0
                    for cat in sorted(available_categories, key=lambda x: -category_counts.get(x, 0)):
                        cat_skills = category_counts.get(cat, 0)
                        if total + cat_skills <= 800:
                            selected.append(cat)
                            total += cat_skills
                        if len(selected) >= 15:  # 最大15カテゴリー
                            break
                    if selected:
                        st.session_state['hierarchical_selected_categories'] = selected
                        st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            with col3:
                if st.button("🌍 全カテゴリー", use_container_width=True, key="hier_all"):
                    # 全カテゴリーを選択
                    selected = available_categories[:]
                    total = sum(category_counts.get(cat, 0) for cat in selected)
                    st.session_state['hierarchical_selected_categories'] = selected
                    st.success(f"✅ {len(selected)}個のカテゴリーを選択しました（{total}個のスキル）")

            # ========================================
            # 上級者向け：カテゴリー詳細調整
            # ========================================
            with st.expander("🔧 カテゴリーを詳細調整（上級者向け）", expanded=False):
                st.write("複数のカテゴリーを選択してください（推奨: 5~20カテゴリー、200~1000スキル）")

                # 全件選択ボタン（チェックボックスではなくボタンで実装）
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    if st.button("🌍 全件選択", key="hier_select_all_btn", use_container_width=True):
                        # 全カテゴリーを選択してsession_stateに保存
                        st.session_state['hierarchical_selected_categories'] = available_categories[:]
                        st.success(f"✅ 全{len(available_categories)}カテゴリーを選択しました")

                with col_b:
                    if st.button("🗑️ 選択解除", key="hier_clear_all_btn", use_container_width=True):
                        # 選択を解除
                        if 'hierarchical_selected_categories' in st.session_state:
                            del st.session_state['hierarchical_selected_categories']
                        st.info("選択を解除しました")

                # カテゴリー情報の表示
                category_info = [f"{cat} ({category_counts.get(cat, 0)}個)" for cat in available_categories]

                # session_stateから現在の選択を取得
                current_selection = []
                if 'hierarchical_selected_categories' in st.session_state:
                    current_categories = st.session_state['hierarchical_selected_categories']
                    current_selection = [f"{cat} ({category_counts.get(cat, 0)}個)"
                                        for cat in current_categories if cat in available_categories]

                selected_categories_display = st.multiselect(
                    "力量カテゴリー",
                    options=category_info,
                    default=current_selection,
                    help="複数のカテゴリーを選択してください。HierarchicalSEMは1000スキルまで対応",
                    key="hier_multiselect"
                )

                # 表示名から実際のカテゴリー名を抽出
                if selected_categories_display:
                    selected_categories = [cat.rsplit(' (', 1)[0] for cat in selected_categories_display]
                    st.session_state['hierarchical_selected_categories'] = selected_categories

            # ========================================
            # 選択状況の確認
            # ========================================
            if 'hierarchical_selected_categories' in st.session_state:
                selected_categories = st.session_state['hierarchical_selected_categories']
                selected_competences = competence_master[
                    competence_master['力量カテゴリー名'].isin(selected_categories)
                ]
                total_skills = len(selected_competences)

                st.markdown("---")
                st.markdown("#### 📊 選択状況")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("選択カテゴリー数", len(selected_categories))
                with col2:
                    st.metric("スキル総数", total_skills)
                with col3:
                    if total_skills <= 400:
                        est_time = "~5分"
                    elif total_skills <= 800:
                        est_time = "5-15分"
                    else:
                        est_time = "15分以上"
                    st.metric("推定時間", est_time)

                if total_skills > 1000:
                    st.warning(f"⚠️ スキル数が{total_skills}個と非常に多いです。処理に時間がかかる場合があります。")
                elif total_skills < 20:
                    st.error("❌ スキル数が少なすぎます。最低20個以上を選択してください。")
            else:
                selected_categories = []
                selected_competences = pd.DataFrame()
                total_skills = 0

    # 並列処理設定
    st.markdown("---")
    st.markdown("### ⚙️ 処理設定")

    # session_stateで状態を保持（スライダー変更時に他の設定が初期化されるのを防ぐ）
    if 'hsem_use_parallel' not in st.session_state:
        st.session_state.hsem_use_parallel = True
    if 'hsem_n_jobs' not in st.session_state:
        st.session_state.hsem_n_jobs = 4

    use_parallel = st.checkbox(
        "並列処理を有効化（高速化）",
        value=st.session_state.hsem_use_parallel,
        key="hsem_parallel_checkbox"
    )
    st.session_state.hsem_use_parallel = use_parallel

    if use_parallel:
        n_jobs = st.slider(
            "並列ジョブ数",
            1, 8,
            value=st.session_state.hsem_n_jobs,
            help="CPUコア数に応じて調整してください",
            key="hsem_n_jobs_slider"
        )
        st.session_state.hsem_n_jobs = n_jobs
    else:
        n_jobs = 1

    # 実行ボタン
    st.markdown("---")
    st.markdown("### 🚀 分析実行")

    # ボタンの有効/無効の判定
    can_execute = bool(selected_categories) and total_skills >= 20

    if not can_execute:
        if not selected_categories:
            st.error("❌ カテゴリーが選択されていません。上記で「カテゴリーで絞り込む」を選択し、カテゴリーを選んでください。")
        elif total_skills < 20:
            st.error(f"❌ スキル数が少なすぎます（現在: {total_skills}個）。最低20個以上のスキルを含むカテゴリーを選択してください。")

    if st.button(
        "🚀 HierarchicalSEM分析を開始",
        type="primary",
        disabled=not can_execute,
        use_container_width=True,
        help="選択したスキルを対象にHierarchicalSEM分析を実行します"
    ):
        with st.spinner("データを準備中..."):
            try:
                # データの準備
                selected_skill_codes = selected_competences['力量コード'].tolist()

                # フィルタリング
                filtered_mc = member_competence[
                    member_competence['力量コード'].isin(selected_skill_codes)
                ]

                # ピボット: 行=メンバー、列=力量コード、値=正規化レベル
                pivot_data = filtered_mc.pivot_table(
                    index='メンバーコード',
                    columns='力量コード',
                    values='正規化レベル',
                    aggfunc='first'
                ).fillna(0)

                st.success(f"✅ データ準備完了: {len(pivot_data)}人 × {len(pivot_data.columns)}スキル")

            except Exception as e:
                st.error(f"❌ データ準備エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())
                st.stop()

        with st.spinner("階層構造を構築中..."):
            try:
                # モジュールロード
                hierarchical_sem_module = load_hierarchical_sem()
                HierarchicalSEMEstimator = hierarchical_sem_module.HierarchicalSEMEstimator
                DomainDefinition = hierarchical_sem_module.DomainDefinition

                # ドメイン定義の作成
                domain_definitions = []

                # Level 1: カテゴリーごとのドメイン
                for category in selected_categories:
                    cat_competences = selected_competences[
                        selected_competences['力量カテゴリー名'] == category
                    ]
                    skill_codes = cat_competences['力量コード'].tolist()

                    # ピボットデータに存在するスキルのみを使用
                    skill_codes = [code for code in skill_codes if code in pivot_data.columns]

                    if len(skill_codes) >= 2:
                        domain_definitions.append(
                            DomainDefinition(
                                domain_name=category,
                                skills=skill_codes,
                                parent_domain='全体力量',
                                level=1
                            )
                        )

                # Level 2: 統合レベル（全カテゴリーを統合）
                domain_definitions.append(
                    DomainDefinition(
                        domain_name='全体力量',
                        skills=selected_categories,  # カテゴリー名をスキルとして扱う
                        level=2
                    )
                )

                st.success(f"✅ 階層構造構築完了: {len(domain_definitions)-1}個のドメイン + 統合層")

            except Exception as e:
                st.error(f"❌ 階層構造構築エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())
                st.stop()

        with st.spinner(f"HierarchicalSEM推定中（並列度: {n_jobs}）..."):
            try:
                import time
                start_time = time.time()

                # HierarchicalSEM推定
                hsem = HierarchicalSEMEstimator(
                    domain_definitions=domain_definitions,
                    confidence_level=0.95,
                    method='ML'
                )
                result = hsem.fit(pivot_data, n_jobs=n_jobs, use_multiprocessing=False)

                elapsed_time = time.time() - start_time
                st.success(f"✅ 推定完了！（{elapsed_time:.1f}秒）")

                # 結果表示
                st.markdown("---")
                st.subheader("📊 推定結果")

                # 統合モデルの適合度
                if result.integration_model and result.integration_fit_indices:
                    st.markdown("### 統合モデルの適合度指標")
                    fit = result.integration_fit_indices

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSEA", f"{fit.rmsea:.3f}", delta="良好" if fit.rmsea < 0.08 else "要改善", delta_color="inverse")
                    with col2:
                        st.metric("CFI", f"{fit.cfi:.3f}", delta="良好" if fit.cfi > 0.90 else "要改善", delta_color="normal")
                    with col3:
                        st.metric("TLI", f"{fit.tli:.3f}", delta="良好" if fit.tli > 0.90 else "要改善", delta_color="normal")
                    with col4:
                        st.metric("SRMR", f"{fit.srmr:.3f}", delta="良好" if fit.srmr < 0.08 else "要改善", delta_color="inverse")

                    # 総合判定
                    if fit.is_excellent_fit():
                        st.success("✅ 優れた適合度です！")
                    elif fit.is_good_fit():
                        st.info("✅ 良好な適合度です")
                    else:
                        st.warning("⚠️ 適合度が低いです。")

                # ドメイン別の適合度
                st.markdown("### ドメイン別の適合度")

                domain_fit_data = []
                for domain_name, fit in result.domain_fit_indices.items():
                    domain_fit_data.append({
                        'ドメイン': domain_name,
                        'RMSEA': f"{fit.rmsea:.3f}",
                        'CFI': f"{fit.cfi:.3f}",
                        'TLI': f"{fit.tli:.3f}",
                        'SRMR': f"{fit.srmr:.3f}",
                        'AIC': f"{fit.aic:.1f}",
                        'BIC': f"{fit.bic:.1f}",
                        '判定': '優秀' if fit.is_excellent_fit() else ('良好' if fit.is_good_fit() else '要改善')
                    })

                domain_fit_df = pd.DataFrame(domain_fit_data)
                st.dataframe(domain_fit_df, use_container_width=True, hide_index=True)

                # ドメインスコア
                if result.domain_scores is not None:
                    st.markdown("### 📊 ドメインスコア統計")

                    st.info(
                        "**ドメインスコアとは？**\n\n"
                        "各メンバーがそれぞれの力量カテゴリー（ドメイン）でどの程度のスキルレベルを持っているかを示す指標です。\n\n"
                        "- **高いスコア**: そのカテゴリーのスキルを多く習得している\n"
                        "- **低いスコア**: そのカテゴリーのスキル習得が少ない\n\n"
                        "このスコアを使って、メンバーの得意分野や成長機会を把握できます。"
                    )

                    score_stats = result.domain_scores.describe().T
                    score_stats = score_stats[['mean', 'std', 'min', 'max']]
                    score_stats.columns = ['平均', '標準偏差', '最小値', '最大値']
                    st.dataframe(score_stats, use_container_width=True)

                    # ドメインスコアの分布（改善版）
                    st.markdown("#### カテゴリー別スコア分布")

                    # カテゴリー数に応じて高さを動的調整
                    n_categories = len(result.domain_scores.columns)
                    # 基本: 400px、カテゴリーが多い場合は追加（1カテゴリーあたり20px）
                    dynamic_height = max(400, 300 + n_categories * 20)

                    fig = go.Figure()
                    for col in result.domain_scores.columns:
                        fig.add_trace(go.Box(
                            y=result.domain_scores[col],
                            name=col,
                            boxmean='sd',
                            marker=dict(
                                color='lightblue',
                                line=dict(color='darkblue', width=1.5)
                            ),
                            line=dict(color='darkblue'),
                            fillcolor='rgba(100, 149, 237, 0.5)'
                        ))

                    fig.update_layout(
                        title='各カテゴリーのスコア分布（箱ひげ図）<br><sub>箱：25%-75%範囲、線：中央値、×：平均値</sub>',
                        yaxis_title='ドメインスコア',
                        xaxis_title='力量カテゴリー',
                        height=dynamic_height,
                        showlegend=False,
                        plot_bgcolor='#F8F9FA',
                        font=dict(size=12),
                        margin=dict(b=150, l=80, r=40, t=120),
                    )

                    # X軸のラベルを斜めに表示
                    fig.update_xaxes(tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)

                # ============================================
                # ネットワークグラフ可視化（HierarchicalSEM用）
                # ============================================
                st.markdown("---")
                st.markdown("## 📊 ネットワークグラフ可視化")

                with st.spinner("ネットワークグラフを生成中..."):
                    try:
                        # グラフ可視化モジュールをロード
                        visualizer_module = load_sem_network_visualizer()
                        SEMNetworkVisualizer = visualizer_module.SEMNetworkVisualizer

                        visualizer = SEMNetworkVisualizer()

                        # スキルコード → スキル名（日本語）のマッピングを作成
                        skill_code_to_name = dict(zip(
                            competence_master['力量コード'],
                            competence_master['力量名']
                        ))

                        # タブで表示方法を選択
                        tab1, tab2, tab3 = st.tabs([
                            "🕸️ スキル間ネットワーク（ドメイン別）",
                            "📈 カテゴリー別スコア相関",
                            "🎯 カテゴリー間因果関係"
                        ])

                        with tab1:
                            st.markdown(
                                "### スキル間ネットワーク（ドメイン別・有向グラフ）\n"
                                "各カテゴリー内でのスキル同士の学習順序・前提関係を可視化"
                            )

                            st.info("""
                            **📊 グラフの方向性について:**
                            - 取得日データがある場合: 実際の学習パターンから方向性を推定（A→B = Aを先に学ぶべき）
                            - 取得日データがない場合: カテゴリー内の関連性を表示（無向グラフ）
                            """)

                            # ドメイン選択
                            domain_names = [name for name in result.domain_models.keys() if name != '全体力量']

                            if len(domain_names) > 0:
                                selected_domain = st.selectbox(
                                    "表示するカテゴリーを選択",
                                    options=domain_names,
                                    help="各カテゴリー内のスキル間ネットワークを表示します",
                                    key="hier_sem_domain_select"
                                )

                                # 選択されたドメインのモデルを取得
                                domain_model = result.domain_models[selected_domain]

                                # ネットワーク表示モード選択
                                st.markdown("##### 🎯 表示モード")
                                network_display_mode_hier = st.radio(
                                    "ネットワークの範囲を選択",
                                    options=["全スキル表示", "個別スキル選択"],
                                    index=0,
                                    help="このカテゴリー内で表示するスキルの範囲を選択してください",
                                    key=f"hier_network_display_mode_{selected_domain}",
                                    horizontal=True
                                )

                                # フィルタリング対象のスキルコードリスト
                                filtered_skill_codes_hier = domain_model.observed_vars.copy()

                                if network_display_mode_hier == "個別スキル選択":
                                    st.markdown("##### 🔍 スキル選択")

                                    # スキル名のリストを作成（コード付き）
                                    skill_options_hier = [
                                        f"{skill_code_to_name.get(code, code)} ({code})"
                                        for code in domain_model.observed_vars
                                    ]

                                    selected_skills_hier = st.multiselect(
                                        "表示するスキルを選択",
                                        options=skill_options_hier,
                                        help="選択したスキルとその関連スキルのみを表示します（最大20個推奨）",
                                        key=f"hier_skill_select_{selected_domain}"
                                    )

                                    if selected_skills_hier:
                                        # 選択されたスキルのコードを抽出
                                        filtered_skill_codes_hier = [
                                            skill.split("(")[-1].rstrip(")")
                                            for skill in selected_skills_hier
                                        ]
                                        st.info(f"✅ {len(filtered_skill_codes_hier)}個のスキルを選択")
                                    else:
                                        st.warning("⚠️ スキルを選択してください")
                                        filtered_skill_codes_hier = []

                                # メンバー選択
                                st.markdown("##### 👤 メンバー別表示（オプション）")
                                # フィルタリング後のメンバーを使用
                                member_names_hier = members_clean_filtered['メンバー名'].tolist()
                                member_codes_hier = members_clean_filtered['メンバーコード'].tolist()

                                member_options_hier = ["（全体表示）"] + [f"{name} ({code})" for name, code in zip(member_names_hier, member_codes_hier)]

                                selected_member_display_hier = st.selectbox(
                                    "メンバーを選択",
                                    options=member_options_hier,
                                    help="メンバーを選択すると、そのメンバーの取得済み/未取得力量が色分けされます",
                                    key="hier_sem_selected_member"
                                )

                                # 選択されたメンバーの取得済みスキルを取得
                                acquired_skills_hier = None
                                if selected_member_display_hier != "（全体表示）":
                                    # メンバーコードを抽出
                                    selected_member_code_hier = selected_member_display_hier.split("(")[-1].rstrip(")")

                                    # このメンバーの取得済みスキルを取得
                                    member_skills_hier = member_competence[
                                        member_competence['メンバーコード'] == selected_member_code_hier
                                    ]['力量コード'].tolist()
                                    acquired_skills_hier = set(member_skills_hier)

                                    st.caption(f"✅ 取得済み力量: {len(acquired_skills_hier)}個")

                                st.markdown("---")

                                col_threshold_hier, col_edge_hier = st.columns(2)

                                with col_threshold_hier:
                                    loading_threshold_hier = st.slider(
                                        "ローディング閾値",
                                        min_value=0.0,
                                        max_value=1.0,
                                        value=0.2,
                                        step=0.05,
                                        help="この値以上のファクターローディングを持つ力量のみ表示します",
                                        key="hier_sem_loading_threshold",
                                    )
                                    st.caption(f"現在の閾値: {loading_threshold_hier:.2f}")

                                # 全接続数を計算
                                temp_edges_hier = []
                                for j in range(len(domain_model.latent_vars)):
                                    contributing_skills = [
                                        (i, abs(domain_model.Lambda[i, j]))
                                        for i in range(len(domain_model.observed_vars))
                                        if abs(domain_model.Lambda[i, j]) > loading_threshold_hier
                                    ]
                                    for k1 in range(len(contributing_skills)):
                                        for k2 in range(k1 + 1, len(contributing_skills)):
                                            temp_edges_hier.append(True)

                                max_edges_hier = len(temp_edges_hier)

                                with col_edge_hier:
                                    # スライダーで表示する接続数の範囲を調整
                                    slider_start_key_hier = f"hier_sem_skill_network_edge_start_{selected_domain}"
                                    slider_end_key_hier = f"hier_sem_skill_network_edge_end_{selected_domain}"

                                    if slider_start_key_hier not in st.session_state:
                                        st.session_state[slider_start_key_hier] = 1 if max_edges_hier > 0 else 1

                                    if slider_end_key_hier not in st.session_state:
                                        st.session_state[slider_end_key_hier] = min(20, max_edges_hier) if max_edges_hier > 0 else 1

                                    if st.session_state[slider_end_key_hier] > max_edges_hier and max_edges_hier > 0:
                                        st.session_state[slider_end_key_hier] = max_edges_hier

                                    st.markdown("##### 接続範囲指定（関係性が強い順）")

                                    edge_start_hier = st.slider(
                                        "開始位置（番目から）",
                                        min_value=1,
                                        max_value=max(1, max_edges_hier),
                                        step=1,
                                        help=f"最小: 1、最大: {max_edges_hier}",
                                        key=slider_start_key_hier,
                                    )

                                    edge_end_hier = st.slider(
                                        "終了位置（番目まで）",
                                        min_value=1,
                                        max_value=max(1, max_edges_hier),
                                        step=1,
                                        help=f"開始位置以上の値で指定してください",
                                        key=slider_end_key_hier,
                                    )

                                    if edge_start_hier > edge_end_hier:
                                        edge_start_hier, edge_end_hier = edge_end_hier, edge_start_hier
                                        st.warning(f"開始位置が終了位置より大きいため、自動調整しました: {edge_start_hier}～{edge_end_hier}")

                                    st.caption(f"表示中: {edge_start_hier}～{edge_end_hier}番目 （全 {max_edges_hier} 接続）")

                                st.markdown("---")

                                # 学習順序分析（取得日データがある場合）
                                dependency_edges_hier = None
                                use_learning_order_hier = False

                                if '取得日' in member_competence.columns:
                                    use_learning_order_hier = st.checkbox(
                                        "🎓 学習順序ロジックを使用（取得日データから分析）",
                                        value=True,
                                        help="実際の取得パターンから学習順序を推定し、有向グラフの方向性を決定します",
                                        key=f"hier_use_learning_order_{selected_domain}"
                                    )

                                    if use_learning_order_hier:
                                        # キャッシュキーを作成
                                        cache_key_hier = f"hier_dep_{selected_domain}_{len(st.session_state.get('filtered_member_codes', []))}"

                                        if cache_key_hier not in st.session_state:
                                            with st.spinner("学習順序を分析中..."):
                                                try:
                                                    # フィルタリングされたメンバーの力量データを取得
                                                    if 'filtered_member_codes' in st.session_state and st.session_state.filtered_member_codes:
                                                        filtered_competence_hier = member_competence[
                                                            member_competence['メンバーコード'].isin(st.session_state.filtered_member_codes)
                                                        ]
                                                    else:
                                                        filtered_competence_hier = member_competence

                                                    # SkillDependencyAnalyzerをロード
                                                    analyzer_module = load_skill_dependency_analyzer()
                                                    SkillDependencyAnalyzer = analyzer_module.SkillDependencyAnalyzer

                                                    # アナライザーを初期化
                                                    analyzer_hier = SkillDependencyAnalyzer(
                                                        member_competence=filtered_competence_hier,
                                                        competence_master=competence_master,
                                                        time_window_days=180,
                                                        min_transition_count=2,
                                                        confidence_threshold=0.2,
                                                    )

                                                    # グラフデータを取得
                                                    graph_data_hier = analyzer_hier.get_dependency_graph_data()

                                                    # セッション状態に保存
                                                    st.session_state[cache_key_hier] = graph_data_hier.get('edges', [])

                                                    st.success(f"✅ 学習順序分析完了！{len(st.session_state[cache_key_hier])}個の依存関係を検出")

                                                except Exception as e:
                                                    st.warning(f"⚠️ 学習順序分析エラー: {e}")
                                                    st.info("Lambda行列ベースのネットワークを表示します")
                                                    st.session_state[cache_key_hier] = []

                                        dependency_edges_hier = st.session_state.get(cache_key_hier, [])

                                # フィルタリングされたスキルに対応するLambda行列の行インデックスを取得
                                if len(filtered_skill_codes_hier) > 0:
                                    filtered_indices_hier = [
                                        i for i, code in enumerate(domain_model.observed_vars)
                                        if code in filtered_skill_codes_hier
                                    ]

                                    # フィルタリングされた行のみを抽出
                                    import numpy as np
                                    filtered_lambda_hier = domain_model.Lambda[filtered_indices_hier, :]

                                    # パフォーマンス最適化の適用状況を表示
                                    n_skills_hier = len(filtered_skill_codes_hier)
                                    if n_skills_hier >= 200:
                                        st.info(f"⚡ 超大規模グラフ最適化適用中（{n_skills_hier}スキル）: Kamada-Kawai レイアウト + 高度なエッジ削減")
                                    elif n_skills_hier >= 150:
                                        st.info(f"⚡ 大規模グラフ最適化適用中（{n_skills_hier}スキル）: Kamada-Kawai レイアウト + エッジ削減")
                                    elif n_skills_hier >= 100:
                                        st.info(f"⚡ 中規模グラフ最適化適用中（{n_skills_hier}スキル）: 高速レイアウト + エッジ制限")

                                    if max_edges_hier > 0:
                                        fig_skill_network_hier = visualizer.visualize_skill_network(
                                            lambda_matrix=filtered_lambda_hier,
                                            latent_vars=domain_model.latent_vars,
                                            observed_vars=filtered_skill_codes_hier,
                                            skill_name_mapping=skill_code_to_name,
                                            loading_threshold=loading_threshold_hier,
                                            edge_limit_start=edge_start_hier,
                                            edge_limit_end=edge_end_hier,
                                            acquired_skills=acquired_skills_hier,
                                            dependency_edges=dependency_edges_hier if dependency_edges_hier else None,
                                        )
                                        st.plotly_chart(fig_skill_network_hier, use_container_width=True)

                                        # 使用したロジックを表示
                                        if use_learning_order_hier and dependency_edges_hier:
                                            st.caption(f"🎓 学習順序ロジック使用中（{len(dependency_edges_hier)}個の依存関係）")
                                        else:
                                            st.caption("📊 Lambda行列ベースのネットワーク")
                                    else:
                                        st.info(f"💡 {selected_domain}には表示可能なスキル間接続がありません（ローディング閾値を下げてみてください）")
                                else:
                                    st.warning("⚠️ 表示するスキルがありません。スキルを選択してください。")
                            else:
                                st.info("💡 ドメインモデルが見つかりません")

                        with tab2:
                            st.markdown(
                                "### 📈 カテゴリー別スコア相関\n"
                                "各カテゴリースコア間の相関関係を表示します"
                            )

                            if result.domain_scores is not None and len(result.domain_scores.columns) > 1:
                                # 相関行列を計算
                                corr_matrix = result.domain_scores.corr()

                                # ヒートマップで表示
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    labels=dict(x="カテゴリー", y="カテゴリー", color="相関係数"),
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    zmin=-1,
                                    zmax=1,
                                )
                                fig_corr.update_layout(
                                    title="カテゴリースコア相関マトリクス",
                                    height=600
                                )

                                st.plotly_chart(fig_corr, use_container_width=True)

                                st.markdown("""
                                **読み方:**
                                - 値が1に近い: 正の相関（一方が高いと他方も高い）
                                - 値が-1に近い: 負の相関（一方が高いと他方は低い）
                                - 値が0に近い: 相関なし
                                """)
                            else:
                                st.info("💡 カテゴリースコアが見つかりません")

                        with tab3:
                            st.markdown(
                                "### 🎯 カテゴリー間因果関係（有向グラフ）\n"
                                "統合層における力量カテゴリー間の因果関係と学習発展段階"
                            )

                            # 統合モデルのB行列を確認
                            if result.integration_model is not None and hasattr(result.integration_model, 'B'):
                                integration_model = result.integration_model

                                # B行列が存在し、非ゼロ要素があるか確認
                                if integration_model.B is not None and np.any(np.abs(integration_model.B) > 0.001):
                                    with st.expander("📖 この図の見方", expanded=True):
                                        st.markdown("""
                                        #### 構造図（カテゴリー間因果関係）
                                        ```
                                        カテゴリーA ──→ カテゴリーB ──→ カテゴリーC
                                        （基礎）         （応用）        （エキスパート）
                                        ```

                                        #### 矢印の意味
                                        - **緑色の矢印（→）**: 統計的に有意な因果関係
                                          - p値 < 0.05（関係がある確率95%以上）
                                          - 実務で確認されている段階的成長
                                        - **グレーの矢印（→）**: 統計的に有意でない
                                          - 直接的な因果関係が見つからない可能性

                                        #### 矢印の太さ
                                        - **太い矢印**: 因果係数が大きい（強い影響）
                                        - **細い矢印**: 因果係数が小さい（弱い影響）

                                        #### このタブで分かること
                                        1. **学習段階**: カテゴリー習得の最適な順序
                                        2. **前提条件**: 高度なカテゴリーを習得する前に何を習得すべきか
                                        3. **キャリアパス**: メンバーのキャリア発展の方向性
                                        """)

                                    # パス有意性の辞書を作成（もしあれば）
                                    path_significance_hier = {}
                                    if hasattr(integration_model, 'params'):
                                        for param_name, param_obj in integration_model.params.items():
                                            if param_name.startswith('β_'):
                                                # β_fromVar→toVar の形式からfromVarとtoVarを抽出
                                                parts = param_name[2:].split('→')
                                                if len(parts) == 2:
                                                    path_significance_hier[(parts[0], parts[1])] = param_obj.is_significant

                                    # 構造モデルを可視化
                                    fig_structural_hier = visualizer.visualize_structural_model(
                                        b_matrix=integration_model.B,
                                        latent_vars=integration_model.latent_vars,
                                        path_significance=path_significance_hier if path_significance_hier else None,
                                    )
                                    st.plotly_chart(fig_structural_hier, use_container_width=True)

                                    # B行列の詳細データ
                                    with st.expander("📋 構造係数行列 B（カテゴリー間因果係数）"):
                                        b_df = pd.DataFrame(
                                            integration_model.B,
                                            index=integration_model.latent_vars,
                                            columns=integration_model.latent_vars
                                        )
                                        st.dataframe(b_df, use_container_width=True)
                                        st.markdown("""
                                        **読み方:**
                                        - 行→列の因果係数を表示
                                        - 正の値: 促進効果（行のカテゴリーが列のカテゴリーを促進）
                                        - 負の値: 抑制効果（まれ）
                                        - 0に近い値: 因果関係なし
                                        """)
                                else:
                                    st.info("💡 統合モデルに構造パス（B行列）が定義されていません。")
                                    st.markdown("""
                                    **理由:**
                                    - HierarchicalSEMの統合層は、通常カテゴリースコアを統合するのみで、カテゴリー間の因果関係は定義されません。
                                    - カテゴリー間の関連性は「カテゴリー別スコア相関」タブで確認できます。
                                    """)
                            else:
                                st.info("💡 統合モデルが見つかりません")

                        st.success("✅ ネットワークグラフを生成しました")

                    except Exception as e:
                        st.error(f"❌ グラフ生成エラー: {e}")
                        import traceback
                        with st.expander("エラー詳細"):
                            st.code(traceback.format_exc())

                # 詳細データ
                with st.expander("📋 詳細データ"):
                    st.markdown("#### 統合モデル（カテゴリー間の関係）")
                    if result.integration_model:
                        # 構造係数（カテゴリー間の関連性）
                        st.markdown("##### 構造係数（カテゴリー間の関連パス）")
                        relationships = result.integration_model.get_skill_relationships()
                        if len(relationships) > 0:
                            st.dataframe(relationships, use_container_width=True, hide_index=True)
                        else:
                            st.info("💡 構造パスが定義されていません（カテゴリー間に関連性を仮定していないモデルです）")

                        # ファクターローディング（カテゴリー → 統合力量）
                        st.markdown("##### ファクターローディング（各カテゴリーの統合力量への貢献度）")

                        loading_df = pd.DataFrame(
                            result.integration_model.Lambda,
                            index=result.integration_model.observed_vars,
                            columns=result.integration_model.latent_vars
                        )

                        # 絶対値でソート
                        loading_df['最大ローディング'] = loading_df.abs().max(axis=1)
                        loading_df = loading_df.sort_values('最大ローディング', ascending=False)
                        loading_df = loading_df.drop(columns=['最大ローディング'])

                        # matplotlibに依存しないシンプルなスタイリング
                        def color_loading(val):
                            """ローディング値に応じた色付け（matplotlib不要）"""
                            if pd.isna(val):
                                return ''
                            # 緑（正）から赤（負）のグラデーション
                            if val > 0.7:
                                return 'background-color: #90EE90'  # 明るい緑
                            elif val > 0.4:
                                return 'background-color: #D4EDA7'  # 薄い緑
                            elif val > 0.1:
                                return 'background-color: #FFFFCC'  # 薄い黄色
                            elif val > -0.1:
                                return 'background-color: #FFFFFF'  # 白
                            elif val > -0.4:
                                return 'background-color: #FFD4D4'  # 薄いピンク
                            elif val > -0.7:
                                return 'background-color: #FFB6B6'  # ピンク
                            else:
                                return 'background-color: #FF9999'  # 赤

                        st.dataframe(
                            loading_df.style.applymap(color_loading),
                            use_container_width=True
                        )

                        st.markdown("""
                        **読み方:**
                        - 値が大きいほど、そのカテゴリーが統合力量に強く影響している
                        - 正の値: 正の相関（カテゴリースコアが高いと統合力量も高い）
                        - 負の値: 負の相関（稀）
                        """)

                    st.markdown("#### ドメイン別モデルの詳細")
                    for domain_name, model in result.domain_models.items():
                        with st.expander(f"🔍 {domain_name}"):
                            st.write(f"**観測変数数**: {len(model.observed_vars)}")
                            st.write(f"**潜在変数数**: {len(model.latent_vars)}")

                            loading_df = pd.DataFrame(
                                model.Lambda,
                                index=model.observed_vars,
                                columns=model.latent_vars
                            )

                            # 力量コードを力量名に変換
                            skill_code_to_name = dict(zip(
                                competence_master['力量コード'],
                                competence_master['力量名']
                            ))
                            loading_df.index = [skill_code_to_name.get(code, code) for code in loading_df.index]

                            st.dataframe(loading_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 推定エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())


# =========================================================
# フッター
# =========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🧬 SEM分析 | Powered by UnifiedSEM & HierarchicalSEM</p>
    <p>構造方程式モデリングによる科学的な力量分析</p>
</div>
""", unsafe_allow_html=True)
