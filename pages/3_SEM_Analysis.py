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

member_competence = td["member_competence"]
competence_master = td["competence_master"]
members_clean = td["members_clean"]

# データサイズの表示
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("メンバー数", len(members_clean))
with col2:
    n_skills = len(competence_master)
    st.metric("スキル数", n_skills)
with col3:
    n_records = len(member_competence)
    st.metric("習得記録数", n_records)
with col4:
    avg_skills = n_records / len(members_clean) if len(members_clean) > 0 else 0
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
    - 構造モデル: カテゴリー間の因果関係を表示

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
    st.info("📊 実データを使用したUnifiedSEM推定を実行します")

    # カテゴリー選択
    with st.expander("🔧 ドメイン設定", expanded=True):
        st.markdown("### 力量カテゴリーの選択")

        # 利用可能なカテゴリーを取得
        available_categories = competence_master['力量カテゴリー名'].unique().tolist()
        available_categories = [cat for cat in available_categories if pd.notna(cat)]

        # カテゴリーごとのスキル数を計算
        category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()

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

    if st.button("🚀 UnifiedSEM推定を実行", type="primary", disabled=not selected_categories or total_skills < 10):
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

                # サンプル数とスキル数に基づいて、最適なカテゴリー数を計算
                num_members = len(pivot_data)
                num_skills = len(pivot_data.columns)

                # 推奨：スキル数 ≤ メンバー数 / 3（SEM推定の安定性確保）
                recommended_skills = max(10, num_members // 3)

                if num_skills > recommended_skills:
                    # スキルが多すぎる場合、カテゴリー数を自動調整
                    original_categories = len(selected_categories)

                    # カテゴリーごとのスキル数を計算
                    category_skill_counts = {}
                    for category in selected_categories:
                        cat_competences = selected_competences[
                            selected_competences['力量カテゴリー名'] == category
                        ]
                        skill_codes = cat_competences['力量コード'].tolist()
                        skill_codes = [code for code in skill_codes if code in pivot_data.columns]
                        if len(skill_codes) >= 2:
                            category_skill_counts[category] = len(skill_codes)

                    # スキル数が多いカテゴリーから順に選択
                    sorted_categories = sorted(
                        category_skill_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # スキル数がちょうど良くなるまでカテゴリーを選別
                    adjusted_categories = []
                    total_adjusted_skills = 0

                    for category, skill_count in sorted_categories:
                        if total_adjusted_skills + skill_count <= recommended_skills:
                            adjusted_categories.append(category)
                            total_adjusted_skills += skill_count
                        else:
                            # 次のカテゴリーを追加するかどうか判定
                            # 現在のスキル数と推奨値の差が小さい場合は追加
                            if abs(total_adjusted_skills + skill_count - recommended_skills) < abs(total_adjusted_skills - recommended_skills):
                                adjusted_categories.append(category)
                                total_adjusted_skills += skill_count

                    if len(adjusted_categories) < original_categories:
                        st.info(
                            f"ℹ️ **カテゴリー自動調整**\n\n"
                            f"現在のメンバー数（{num_members}人）に対して、"
                            f"スキルが多すぎる可能性があります。\n\n"
                            f"- 元のカテゴリー数: {original_categories}\n"
                            f"- 調整後のカテゴリー数: {len(adjusted_categories)}\n"
                            f"- スキル数: {num_skills} → {total_adjusted_skills}\n\n"
                            f"推奨スキル数: {recommended_skills}個以下（メンバー数 ÷ 3）"
                        )
                        selected_categories = adjusted_categories
                    else:
                        st.warning(
                            f"⚠️ **サンプル数に対してスキルが多い可能性があります**\n\n"
                            f"- メンバー数: {num_members}人\n"
                            f"- スキル数: {num_skills}個\n"
                            f"- 推奨スキル数: {recommended_skills}個以下\n\n"
                            f"**推奨対応:**\n"
                            f"1. 「詳細調整」でカテゴリーを絞る\n"
                            f"2. または大規模型・全カテゴリー型を避ける\n"
                            f"3. 推奨型（バランス型）を選択してください"
                        )

            except Exception as e:
                st.error(f"❌ データ準備エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())
                st.stop()

        with st.spinner("UnifiedSEM推定中..."):
            try:
                # モジュールロード
                unified_sem_module = load_unified_sem()
                UnifiedSEMEstimator = unified_sem_module.UnifiedSEMEstimator
                MeasurementModelSpec = unified_sem_module.MeasurementModelSpec
                StructuralModelSpec = unified_sem_module.StructuralModelSpec

                # 測定モデル仕様の作成（カテゴリーごと）
                measurement_specs = []
                valid_categories = []  # 測定モデルに含まれるカテゴリーを記録
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
                    '< 0.08 (良好)',
                    '> 0.90 (良好)',
                    '> 0.90 (良好)',
                    '> 0.90 (良好)',
                    '< 0.08 (良好)',
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
                tab1, tab2, tab3 = st.tabs(
                    ["🕸️ スキル間ネットワーク", "📈 統合モデル", "🔬 測定モデル"]
                )

                with tab1:
                    st.markdown(
                        "### スキル間ネットワーク\n"
                        "同じ力量カテゴリーに統話するスキル同士の関連性"
                    )

                    # スキルコード → スキル名（日本語）のマッピングを作成
                    skill_code_to_name = dict(zip(
                        competence_master['力量コード'],
                        competence_master['力量名']
                    ))

                    # 設定エリア
                    st.markdown("#### ⚙️ 表示設定")

                    # メンバー選択
                    st.markdown("##### 👤 メンバー別表示（オプション）")
                    member_names = td["members_clean"]['メンバー名'].tolist()
                    member_codes = td["members_clean"]['メンバーコード'].tolist()

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
                        # スライダーで表示する接続数を調整（session_state で状態保持）
                        slider_key = "unified_sem_skill_network_edge_limit"

                        # max_edges が変更された場合、スライダーの値を調整
                        if slider_key not in st.session_state:
                            st.session_state[slider_key] = min(20, max_edges) if max_edges > 0 else 1

                        # max_edges を超えないようにvalidate
                        if st.session_state[slider_key] > max_edges and max_edges > 0:
                            st.session_state[slider_key] = max_edges

                        edge_limit = st.slider(
                            "表示接続数（強度順）",
                            min_value=1,
                            max_value=max(1, max_edges),
                            value=min(st.session_state[slider_key], max(1, max_edges)),
                            step=1,
                            help=f"接続の強度が強い順に表示します。最大：{max_edges}接続",
                            key=slider_key,
                        )
                        st.caption(f"表示中: {edge_limit}/{max_edges}接続")

                    st.markdown("---")

                    fig_skill_network = visualizer.visualize_skill_network(
                        lambda_matrix=sem.Lambda,
                        latent_vars=sem.latent_vars,
                        observed_vars=sem.observed_vars,
                        skill_name_mapping=skill_code_to_name,
                        loading_threshold=loading_threshold,
                        edge_limit=edge_limit,
                        acquired_skills=acquired_skills,
                    )
                    st.plotly_chart(fig_skill_network, use_container_width=True)

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
                            (因果関係の矢印)
                        ```

                        #### 色・太さの意味
                        - **マゼンタ丸（●）**: スキル（習得する具体的な技術）
                          - Python基礎、Git、SQL基礎、DB設計 など
                        - **青い丸（●）**: 力量カテゴリー（複合的な能力）
                          - 初級力量、中級力量、システム設計力 など
                        - **矢印の太さ**: 関係の強さ
                          - 太い → 強い関係
                          - 細い → 弱い関係
                        - **緑色の矢印**: 統計的に有意な因果関係
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

                with tab2:
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

                with tab3:
                    st.markdown(
                        "### ⚙️ 構造モデル（力量→力量）\n"
                        "力量カテゴリー間の因果関係と発展段階"
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
                        - **緑色の矢印（→）**: 統計的に有意な因果関係
                          - p値 < 0.05（関係がある確率95%以上）
                          - 実務で確認されている段階的成長
                        - **グレーの矢印（→）**: 統計的に有意でない
                          - 直接的な因果関係が見つからない可能性
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
    st.info("📊 実データを使用したHierarchicalSEM推定を実行します（大規模データ対応）")

    # カテゴリー選択
    with st.expander("🔧 階層構造設定", expanded=True):
        st.markdown("### 力量カテゴリーの選択")

        # 利用可能なカテゴリーを取得
        available_categories = competence_master['力量カテゴリー名'].unique().tolist()
        available_categories = [cat for cat in available_categories if pd.notna(cat)]

        # カテゴリーごとのスキル数を計算
        category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()

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
        # 選択状況の確認と並列処理設定
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
        st.markdown("#### ⚙️ 処理設定")
        use_parallel = st.checkbox("並列処理を有効化（高速化）", value=True)
        if use_parallel:
            n_jobs = st.slider("並列ジョブ数", 1, 8, 4, help="CPUコア数に応じて調整してください")
        else:
            n_jobs = 1

    if st.button("🚀 HierarchicalSEM推定を実行", type="primary", disabled=not selected_categories or total_skills < 20):
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

                # サンプル数とスキル数に基づいて、最適なカテゴリー数を計算
                num_members = len(pivot_data)
                num_skills = len(pivot_data.columns)

                # 推奨：スキル数 ≤ メンバー数 × 2.5（HierarchicalSEM用、UnifiedSEMより緩い）
                recommended_skills = max(50, int(num_members * 2.5))

                if num_skills > recommended_skills:
                    # スキルが多すぎる場合、カテゴリー数を自動調整
                    original_categories = len(selected_categories)

                    # カテゴリーごとのスキル数を計算
                    category_skill_counts = {}
                    for category in selected_categories:
                        cat_competences = selected_competences[
                            selected_competences['力量カテゴリー名'] == category
                        ]
                        skill_codes = cat_competences['力量コード'].tolist()
                        skill_codes = [code for code in skill_codes if code in pivot_data.columns]
                        if len(skill_codes) >= 1:
                            category_skill_counts[category] = len(skill_codes)

                    # スキル数が多いカテゴリーから順に選択（降順）
                    sorted_categories = sorted(
                        category_skill_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    # スキル数がちょうど良くなるまでカテゴリーを選別
                    adjusted_categories = []
                    total_adjusted_skills = 0

                    for category, skill_count in sorted_categories:
                        if total_adjusted_skills + skill_count <= recommended_skills:
                            adjusted_categories.append(category)
                            total_adjusted_skills += skill_count
                        else:
                            # 次のカテゴリーを追加するかどうか判定
                            # 現在のスキル数と推奨値の差が小さい場合は追加
                            if abs(total_adjusted_skills + skill_count - recommended_skills) < abs(total_adjusted_skills - recommended_skills):
                                adjusted_categories.append(category)
                                total_adjusted_skills += skill_count

                    if len(adjusted_categories) < original_categories:
                        st.info(
                            f"ℹ️ **カテゴリー自動調整**\n\n"
                            f"現在のメンバー数（{num_members}人）に対して、"
                            f"スキルが多すぎる可能性があります。\n\n"
                            f"- 元のカテゴリー数: {original_categories}\n"
                            f"- 調整後のカテゴリー数: {len(adjusted_categories)}\n"
                            f"- スキル数: {num_skills} → {total_adjusted_skills}\n\n"
                            f"推奨スキル数: {recommended_skills}個以下（メンバー数 × 2.5）"
                        )
                        selected_categories = adjusted_categories
                    else:
                        st.warning(
                            f"⚠️ **サンプル数に対してスキルが多い可能性があります**\n\n"
                            f"- メンバー数: {num_members}人\n"
                            f"- スキル数: {num_skills}個\n"
                            f"- 推奨スキル数: {recommended_skills}個以下\n\n"
                            f"**推奨対応:**\n"
                            f"1. 「カテゴリーを詳細調整」でカテゴリーを絞る\n"
                            f"2. 或いは「標準型」を選択してください\n"
                            f"3. メンバー数が少ない場合は推奨型を避ける"
                        )

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
                        height=500,
                        showlegend=False,
                        plot_bgcolor='#F8F9FA',
                        font=dict(size=12),
                        margin=dict(b=100, l=60, r=40, t=100),
                    )

                    # X軸のラベルを斜めに表示
                    fig.update_xaxes(tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)

                # 詳細データ
                with st.expander("📋 詳細データ"):
                    st.markdown("#### 統合モデル（カテゴリー間の関係）")
                    if result.integration_model:
                        # 構造係数（カテゴリー間の因果関係）
                        st.markdown("##### 構造係数（カテゴリー間の因果パス）")
                        relationships = result.integration_model.get_skill_relationships()
                        if len(relationships) > 0:
                            st.dataframe(relationships, use_container_width=True, hide_index=True)
                        else:
                            st.info("💡 構造パスが定義されていません（カテゴリー間に因果関係を仮定していないモデルです）")

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

                        st.dataframe(
                            loading_df.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-1, vmax=1),
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
