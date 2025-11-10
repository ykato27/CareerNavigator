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
        st.write("分析対象とする力量カテゴリーを選択してください（推奨: 2~5カテゴリー、スキル数50~200個）")

        # 利用可能なカテゴリーを取得
        available_categories = competence_master['力量カテゴリー名'].unique().tolist()
        available_categories = [cat for cat in available_categories if pd.notna(cat)]

        # カテゴリーごとのスキル数を表示
        category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()
        category_info = [f"{cat} ({category_counts.get(cat, 0)}個)" for cat in available_categories]

        selected_categories_display = st.multiselect(
            "力量カテゴリー",
            options=category_info,
            default=category_info[:min(3, len(category_info))],
            help="複数のカテゴリーを選択してください。UnifiedSEMは200スキル程度まで推奨"
        )

        # 表示名から実際のカテゴリー名を抽出
        selected_categories = [cat.rsplit(' (', 1)[0] for cat in selected_categories_display]

        # 選択されたカテゴリーの統計
        if selected_categories:
            selected_competences = competence_master[
                competence_master['力量カテゴリー名'].isin(selected_categories)
            ]
            total_skills = len(selected_competences)
            st.metric("選択されたスキル数", total_skills)

            if total_skills > 200:
                st.warning(f"⚠️ スキル数が{total_skills}個と多いです。UnifiedSEMは200個程度まで推奨。HierarchicalSEMの使用を検討してください。")
            elif total_skills < 10:
                st.error("❌ スキル数が少なすぎます。最低10個以上を選択してください。")

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

                # 最低サンプル数のチェック
                min_samples = max(50, total_skills * 3)
                if len(pivot_data) < min_samples:
                    st.warning(f"⚠️ サンプル数が少ない可能性があります（推奨: {min_samples}人以上、現在: {len(pivot_data)}人）")

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

                st.success("✅ 推定完了！")

                # 結果表示（デモモードと同じ形式）
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

            except Exception as e:
                st.error(f"❌ 推定エラー: {e}")
                import traceback
                with st.expander("エラー詳細"):
                    st.code(traceback.format_exc())

# =========================================================
# HierarchicalSEM（実データ）
# =========================================================

elif model_type == "HierarchicalSEM（実データ）":
    st.info("📊 実データを使用したHierarchicalSEM推定を実行します（大規模データ対応）")

    # カテゴリー選択
    with st.expander("🔧 階層構造設定", expanded=True):
        st.markdown("### 力量カテゴリーの選択")
        st.write("分析対象とする力量カテゴリーを選択してください（推奨: 5~20カテゴリー、200~1000スキル）")

        # 利用可能なカテゴリーを取得
        available_categories = competence_master['力量カテゴリー名'].unique().tolist()
        available_categories = [cat for cat in available_categories if pd.notna(cat)]

        # カテゴリーごとのスキル数を表示
        category_counts = competence_master.groupby('力量カテゴリー名').size().to_dict()
        category_info = [f"{cat} ({category_counts.get(cat, 0)}個)" for cat in available_categories]

        selected_categories_display = st.multiselect(
            "力量カテゴリー",
            options=category_info,
            default=category_info[:min(10, len(category_info))],
            help="複数のカテゴリーを選択してください。HierarchicalSEMは1000スキルまで対応"
        )

        # 表示名から実際のカテゴリー名を抽出
        selected_categories = [cat.rsplit(' (', 1)[0] for cat in selected_categories_display]

        # 並列処理設定
        use_parallel = st.checkbox("並列処理を有効化（高速化）", value=True)
        if use_parallel:
            n_jobs = st.slider("並列ジョブ数", 1, 8, 4, help="CPUコア数に応じて調整してください")
        else:
            n_jobs = 1

        # 選択されたカテゴリーの統計
        if selected_categories:
            selected_competences = competence_master[
                competence_master['力量カテゴリー名'].isin(selected_categories)
            ]
            total_skills = len(selected_competences)
            st.metric("選択されたスキル数", total_skills)

            if total_skills > 1000:
                st.warning(f"⚠️ スキル数が{total_skills}個と非常に多いです。処理に時間がかかる場合があります。")
            elif total_skills < 20:
                st.error("❌ スキル数が少なすぎます。最低20個以上を選択してください。")

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
                    st.markdown("### ドメインスコア統計")

                    score_stats = result.domain_scores.describe().T
                    score_stats = score_stats[['mean', 'std', 'min', 'max']]
                    score_stats.columns = ['平均', '標準偏差', '最小値', '最大値']
                    st.dataframe(score_stats, use_container_width=True)

                    # ドメインスコアの分布
                    fig = go.Figure()
                    for col in result.domain_scores.columns:
                        fig.add_trace(go.Box(
                            y=result.domain_scores[col],
                            name=col,
                            boxmean='sd'
                        ))

                    fig.update_layout(
                        title='ドメインスコアの分布',
                        yaxis_title='スコア',
                        height=400,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # 詳細データ
                with st.expander("📋 詳細データ"):
                    st.markdown("#### 統合モデルの構造係数")
                    if result.integration_model:
                        relationships = result.integration_model.get_skill_relationships()
                        if len(relationships) > 0:
                            st.dataframe(relationships, use_container_width=True, hide_index=True)
                        else:
                            st.info("構造パスが定義されていません")

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
