"""
CareerNavigator - SEM階層的スキル推薦

スキル領域の階層構造（初級→中級→上級）に基づいた推薦
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from skillnote_recommendation.ml.acquisition_order_hierarchy import AcquisitionOrderHierarchy
    from skillnote_recommendation.ml.acquisition_order_sem_model import AcquisitionOrderSEMModel
    from skillnote_recommendation.utils.ui_components import (
        apply_rich_ui_styles,
        render_gradient_header
    )
    IMPORTS_OK = True
except ImportError as e:
    st.error(f"❌ インポートエラー: {e}")
    st.error("このページは現在利用できません。")
    st.stop()
    IMPORTS_OK = False


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - SEM階層的スキル推薦",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply rich UI styles
apply_rich_ui_styles()

# Header
render_gradient_header(
    "【工事中】🎓 SEM階層的スキル推薦（取得順序ベース）",
    # "【工事中】スキルの取得順序から学習する完全データドリブンな段階的学習パス推薦"
    ""
)

# =========================================================
# データ読み込みチェック
# =========================================================
if "transformed_data" not in st.session_state:
    st.warning("⚠️ データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
    st.stop()

transformed_data = st.session_state.transformed_data

# データの型と内容を検証
if transformed_data is None:
    st.error("❌ データが正しく読み込まれていません。「データ読み込み」ページからデータを再度読み込んでください。")
    st.stop()

if not isinstance(transformed_data, dict):
    st.error(f"❌ データ形式が不正です。expected: dict, actual: {type(transformed_data).__name__}")
    st.stop()

# 必要なキーの存在確認
required_keys = ["competence_master", "member_competence", "members_clean"]
missing_keys = [key for key in required_keys if key not in transformed_data]
if missing_keys:
    st.error(f"❌ 必要なデータが不足しています: {', '.join(missing_keys)}")
    st.info(f"利用可能なキー: {', '.join(transformed_data.keys())}")
    st.warning("「データ読み込み」ページからデータを再度読み込んでください。")
    st.stop()

competence_master = transformed_data["competence_master"]
member_competence = transformed_data["member_competence"]
members_clean = transformed_data["members_clean"]

# デバッグ: データ読み込み状態を確認
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 デバッグ情報")
st.sidebar.write(f"✅ データ読み込み済み")
st.sidebar.write(f"- competence_master: {len(competence_master)}件")
st.sidebar.write(f"- member_competence: {len(member_competence)}件")
st.sidebar.write(f"- members_clean: {len(members_clean)}件")

# =========================================================
# SEMモデルの学習
# =========================================================
st.markdown("---")
st.subheader("🔧 SEMモデルの設定と学習")

with st.expander("📖 SEM階層的スキル推薦（取得順序ベース）とは？", expanded=False):
    st.markdown("""
    ### 概要
    実際のスキル取得順序データから学習する、完全にデータドリブンなSEMモデルです。
    恣意的なドメイン分類を排除し、実データに基づいた段階的学習パスを推薦します。

    ### 階層構造の構築方法
    1. **取得順序の計算**: 各メンバーのスキル取得日から、スキルごとの「平均取得順序」を計算
    2. **ステージ分割**: 平均取得順序に基づき、スキルを3段階（初級→中級→上級）に自動分類
       - **Stage 1（初級）**: 早期に取得されるスキル（平均取得順序0～10など）
       - **Stage 2（中級）**: 中期に取得されるスキル（平均取得順序11～20など）
       - **Stage 3（上級）**: 後期に取得されるスキル（平均取得順序21以降など）
    3. **SEM構築**: 各ステージを潜在変数とし、Stage 1 → Stage 2 → Stage 3 の因果関係をモデル化

    ### 推薦ロジック
    1. メンバーの現在のステージを推定（進捗率を計算）
    2. 進捗率が80%以上の場合、次のステージのスキルを推薦
    3. SEMの潜在変数スコアで優先度を調整

    ### メリット
    - **完全データドリブン**: 人間が定義したカテゴリに依存しない
    - **実データに基づく**: 実際の取得順序から学習
    - **時系列因果モデリング**: SEMの正しい使い方
    - **説明可能性**: 「このスキルは平均的に〇番目に取得されます」と具体的に説明可能
    """)

# SEMモデル学習のUI
col1, col2, col3 = st.columns(3)

with col1:
    n_stages = st.number_input(
        "ステージ数",
        min_value=2,
        max_value=5,
        value=3,
        help="スキルを何段階に分割するか（デフォルト: 3段階 = 初級/中級/上級）"
    )

with col2:
    min_competences_per_stage = st.number_input(
        "各ステージで最低限必要な力量数",
        min_value=2,
        max_value=10,
        value=3,
        help="各ステージで最低限この数以上の力量がないと、SEMモデルは学習されません"
    )

with col3:
    min_acquisition_count = st.number_input(
        "分析対象とする最小取得人数",
        min_value=1,
        max_value=10,
        value=3,
        help="このスキルを取得した人数がこの値未満の場合、分析対象から除外されます"
    )

# SEMモデルを学習
if st.button("🚀 SEMモデルを学習", type="primary"):
    with st.spinner("SEMモデルを学習中..."):
        try:
            # 取得日列の存在確認
            if '取得日' not in member_competence.columns:
                st.error("❌ member_competenceに '取得日' 列が存在しません。")
                st.stop()

            # ステップ1: 取得順序階層を構築
            with st.spinner("📊 ステップ1: 取得順序階層を構築中..."):
                acquisition_hierarchy = AcquisitionOrderHierarchy(
                    member_competence=member_competence,
                    competence_master=competence_master,
                    n_stages=int(n_stages),
                    min_acquisition_count=int(min_acquisition_count)
                )
            st.success("✅ ステップ1: 取得順序階層の構築完了")

            # 統計情報を表示
            st.markdown("### 📊 スキル取得順序の統計")
            stats_df = acquisition_hierarchy.get_statistics()
            st.dataframe(stats_df, use_container_width=True)

            # デバッグ: 階層の詳細を表示
            with st.expander("🔍 デバッグ: 取得順序階層の詳細", expanded=True):
                st.write(f"**ステージ数:** {n_stages}")
                st.write(f"**分析されたスキル数:** {len(acquisition_hierarchy.skill_acquisition_stats)}")
                st.write("**各ステージのスキル数:**")
                for stage_id in range(1, int(n_stages) + 1):
                    stage_skills = acquisition_hierarchy.get_skills_by_stage(stage_id)
                    stage_name = acquisition_hierarchy.get_stage_name(stage_id)
                    st.write(f"- Stage {stage_id} ({stage_name}): {len(stage_skills)}個")

            # ステップ2: SEMモデルを初期化
            with st.spinner("🧮 ステップ2: SEMモデルを初期化中..."):
                sem_model = AcquisitionOrderSEMModel(
                    member_competence=member_competence,
                    competence_master=competence_master,
                    acquisition_hierarchy=acquisition_hierarchy,
                    n_stages=int(n_stages)
                )
            st.success("✅ ステップ2: SEMモデル初期化完了")

            # ステップ3: フィッティング（プログレス表示付き）
            st.markdown("### ⚙️ ステップ3: 最尤推定フィッティング")
            st.write(f"最小スキル数: {int(min_competences_per_stage)}")

            # プログレスバーを作成
            progress_bar = st.progress(0)
            status_text = st.empty()

            # フィッティング実行時のステータス表示
            try:
                status_text.write("📊 ステップ3.1: データ準備中...")
                progress_bar.progress(10)

                status_text.write("🔍 ステップ3.2: 測定モデル構築中...")
                progress_bar.progress(30)

                status_text.write("⚙️ ステップ3.3: 最尤推定を実行中（⏳ 1-2分かかります）...")
                progress_bar.progress(50)

                # フィッティング実行（最も時間がかかる処理）
                with st.spinner("🔄 最適化アルゴリズム実行中..."):
                    sem_model.fit(min_competences_per_stage=int(min_competences_per_stage))

                status_text.write("📈 ステップ3.4: 適合度指標を計算中...")
                progress_bar.progress(90)

                status_text.write("✅ ステップ3: フィッティング完了")
                progress_bar.progress(100)

            except Exception as e:
                status_text.error(f"❌ フィッティング中にエラーが発生しました: {e}")
                progress_bar.progress(100)
                raise

            st.success(f"✅ ステップ3: フィッティング完了")

            # Session stateに保存
            st.session_state.sem_model = sem_model
            st.session_state.acquisition_hierarchy = acquisition_hierarchy

            # デバッグ: SEMモデルの学習結果を表示
            with st.expander("🔍 デバッグ: SEMモデル学習結果", expanded=True):
                st.write(f"**学習済みSEMモデル:** {'あり' if sem_model.is_fitted else 'なし'}")
                if sem_model.is_fitted:
                    st.write(f"**パス係数:** {[f'{c:.3f}' for c in sem_model.path_coefficients]}")
                    st.write(f"**潜在変数スコア推定メンバー数:** {len(sem_model.latent_scores)}")
                st.write(f"**階層統計:**")
                st.dataframe(stats_df)

            if sem_model.is_fitted:
                st.success(f"✅ SEMモデル学習完了")
            else:
                st.warning(f"⚠️ SEMモデル学習失敗 - データ不足の可能性があります")

        except Exception as e:
            st.error(f"❌ SEMモデル学習エラー: {e}")
            import traceback
            st.code(traceback.format_exc())


# =========================================================
# SEMモデルが学習済みの場合の分析・推薦
# =========================================================
if "sem_model" in st.session_state and st.session_state.sem_model.is_fitted:
    sem_model = st.session_state.sem_model
    acquisition_hierarchy = st.session_state.acquisition_hierarchy

    st.markdown("---")
    st.subheader("👤 メンバー別スキルプロファイル")

    # メンバー選択
    member_codes = sorted(member_competence["メンバーコード"].unique())

    # メンバーコードと名前のマッピングを作成
    member_dict = {}
    for code in member_codes:
        matched = members_clean[members_clean['メンバーコード'] == code]
        if len(matched) > 0 and 'メンバー名' in matched.columns:
            member_dict[code] = f"{code} - {matched['メンバー名'].iloc[0]}"
        else:
            member_dict[code] = code

    selected_member = st.selectbox(
        "メンバーを選択",
        options=member_codes,
        format_func=lambda x: member_dict.get(x, x)
    )

    if selected_member:
        # デバッグ: メンバーのスキル保有状況を確認
        member_skills = member_competence[member_competence['メンバーコード'] == selected_member]
        with st.expander("🔍 デバッグ: メンバースキル情報", expanded=False):
            st.write(f"**選択メンバー:** {selected_member}")
            st.write(f"**保有スキル数:** {len(member_skills)}")
            if len(member_skills) > 0:
                st.write("**保有スキル（最初の5件）:**")
                st.dataframe(member_skills.head(5))
            else:
                st.warning("このメンバーはスキルデータがありません")

        # メンバーの現在のステージを推定
        current_stage, progress, acquired_skills = acquisition_hierarchy.estimate_member_stage(selected_member)
        stage_name = acquisition_hierarchy.get_stage_name(current_stage)

        # メンバーの潜在変数スコアを取得
        latent_scores = sem_model.get_member_latent_scores(selected_member)

        # プロファイル情報を表示
        st.markdown("### 📈 スキル習得プロファイル")

        col_profile1, col_profile2, col_profile3 = st.columns(3)

        with col_profile1:
            st.metric("現在のステージ", f"Stage {current_stage}")

        with col_profile2:
            st.metric("ステージ名", stage_name)

        with col_profile3:
            st.metric("進捗率", f"{progress * 100:.1f}%")

        # ステージ別の習得率を可視化
        if latent_scores:
            st.markdown("#### 📊 ステージ別スキル習得状況")

            # データを準備
            stages = []
            stage_names = []
            latent_score_values = []

            for stage_id in sorted(latent_scores.keys()):
                stages.append(f"Stage {stage_id}")
                stage_names.append(acquisition_hierarchy.get_stage_name(stage_id))
                latent_score_values.append(latent_scores[stage_id])

            # 棒グラフを作成
            fig_stages = go.Figure()

            fig_stages.add_trace(go.Bar(
                x=stages,
                y=latent_score_values,
                text=[f"{v:.2f}" for v in latent_score_values],
                textposition='auto',
                marker=dict(
                    color=latent_score_values,
                    colorscale='Viridis',
                    showscale=True
                ),
                hovertemplate='<b>%{x}</b><br>潜在変数スコア: %{y:.3f}<extra></extra>'
            ))

            fig_stages.update_layout(
                title=f"{selected_member}のステージ別潜在変数スコア",
                xaxis_title="ステージ",
                yaxis_title="潜在変数スコア",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_stages, use_container_width=True)

            # テーブル表示
            st.markdown("#### 📋 ステージ詳細")
            profile_data = []
            for i, stage_id in enumerate(sorted(latent_scores.keys())):
                profile_data.append({
                    'ステージ': f"Stage {stage_id}",
                    'ステージ名': stage_names[i],
                    '潜在変数スコア': f"{latent_score_values[i]:.3f}"
                })

            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)

        else:
            st.warning("⚠️ このメンバーの潜在変数スコアが取得できませんでした")

        # =========================================================
        # SEM推薦結果
        # =========================================================
        st.markdown("---")
        st.markdown("### 🎯 SEM階層的スキル推薦結果")

        top_n_sem = st.number_input(
            "推薦数",
            min_value=1,
            max_value=20,
            value=10,
            key="top_n_sem"
        )

        # 推薦を生成
        with st.spinner("SEM推薦を生成中..."):
            recommendations = sem_model.recommend_next_skills(
                member_code=selected_member,
                top_n=int(top_n_sem)
            )

        if len(recommendations) > 0:
            # 推薦結果を表示
            st.success(f"✅ {len(recommendations)}件の推薦を生成しました")

            # 推薦結果をDataFrameに変換
            rec_df = pd.DataFrame(recommendations)

            # 表示用に整形
            display_rec_df = rec_df[[
                'competence_name',
                'stage',
                'stage_name',
                'category',
                'avg_acquisition_order',
                'adjusted_priority_score'
            ]].copy()

            display_rec_df = display_rec_df.rename(columns={
                'competence_name': '力量名',
                'stage': 'ステージ',
                'stage_name': 'ステージ名',
                'category': 'カテゴリー',
                'avg_acquisition_order': '平均取得順序',
                'adjusted_priority_score': '優先度スコア',
            })

            display_rec_df['平均取得順序'] = display_rec_df['平均取得順序'].round(1)
            display_rec_df['優先度スコア'] = display_rec_df['優先度スコア'].round(3)

            st.dataframe(display_rec_df, use_container_width=True)

            # ステージ別の推薦数を可視化
            st.markdown("#### 📊 ステージ別推薦数")

            stage_counts = rec_df['stage'].value_counts().sort_index()
            stage_labels = [f"Stage {s}" for s in stage_counts.index]

            fig_stage = px.bar(
                x=stage_labels,
                y=stage_counts.values,
                labels={'x': 'ステージ', 'y': '推薦数'},
                title='ステージ別推薦数',
                color=stage_counts.values,
                color_continuous_scale='viridis',
            )

            fig_stage.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_stage, use_container_width=True)

            # カテゴリー別の推薦数を可視化
            if 'category' in rec_df.columns and rec_df['category'].notna().any():
                st.markdown("#### 📊 カテゴリー別推薦数")

                category_counts = rec_df['category'].value_counts()

                fig_category = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    labels={'x': 'カテゴリー', 'y': '推薦数'},
                    title='カテゴリー別推薦数',
                    color=category_counts.values,
                    color_continuous_scale='blues',
                )

                fig_category.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_category, use_container_width=True)

            # CSVダウンロード
            st.markdown("#### 📥 推薦結果のダウンロード")

            csv_data = display_rec_df.to_csv(index=False).encode('utf-8-sig')

            st.download_button(
                label="📥 推薦結果をCSVでダウンロード",
                data=csv_data,
                file_name=f"sem_recommendations_{selected_member}.csv",
                mime="text/csv",
            )

        else:
            st.info("💡 このメンバーに推薦できるスキルがありません。")


# =========================================================
# SEMモデルの詳細分析
# =========================================================
if "sem_model" in st.session_state and st.session_state.sem_model.is_fitted:
    st.markdown("---")
    st.subheader("🔍 SEMモデルの詳細分析")

    sem_model = st.session_state.sem_model

    with st.expander("📈 SEMモデルの詳細", expanded=False):
        st.markdown("### モデル概要")

        col_model1, col_model2, col_model3 = st.columns(3)

        with col_model1:
            st.metric("ステージ数", sem_model.n_stages)

        with col_model2:
            st.metric("パス係数数", len(sem_model.path_coefficients))

        with col_model3:
            st.metric("推定メンバー数", len(sem_model.latent_scores))

        # パス係数を表示
        if sem_model.path_coefficients:
            st.markdown("### パス係数（因果効果の強さ）")

            path_data = []
            for i, coef in enumerate(sem_model.path_coefficients):
                from_stage = i + 1
                to_stage = i + 2
                path_data.append({
                    '因果パス': f"Stage {from_stage} → Stage {to_stage}",
                    'パス係数': f"{coef:.3f}",
                    '解釈': '強い' if coef > 0.7 else '中程度' if coef > 0.5 else '弱い'
                })

            path_df = pd.DataFrame(path_data)
            st.dataframe(path_df, use_container_width=True)

            # パス係数を可視化
            fig_path = go.Figure()

            fig_path.add_trace(go.Bar(
                x=[p['因果パス'] for p in path_data],
                y=sem_model.path_coefficients,
                text=[f"{c:.3f}" for c in sem_model.path_coefficients],
                textposition='auto',
                marker=dict(
                    color=sem_model.path_coefficients,
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=0,
                    cmax=1
                )
            ))

            fig_path.update_layout(
                title="ステージ間のパス係数",
                xaxis_title="因果パス",
                yaxis_title="パス係数",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig_path, use_container_width=True)

        # SEMモデルの適合度指標
        if hasattr(sem_model.sem_model, 'fit_info') and sem_model.sem_model.fit_info:
            st.markdown("### 適合度指標")

            fit_info = sem_model.sem_model.fit_info

            col_fit1, col_fit2, col_fit3 = st.columns(3)

            with col_fit1:
                gfi = fit_info.get('gfi', 'N/A')
                if isinstance(gfi, (int, float)):
                    st.metric("GFI", f"{gfi:.3f}", help="Goodness of Fit Index (1に近いほど良い)")
                else:
                    st.metric("GFI", gfi)

            with col_fit2:
                agfi = fit_info.get('agfi', 'N/A')
                if isinstance(agfi, (int, float)):
                    st.metric("AGFI", f"{agfi:.3f}", help="Adjusted Goodness of Fit Index")
                else:
                    st.metric("AGFI", agfi)

            with col_fit3:
                rmsea = fit_info.get('rmsea', 'N/A')
                if isinstance(rmsea, (int, float)):
                    st.metric("RMSEA", f"{rmsea:.3f}", help="Root Mean Square Error of Approximation (0.05以下が望ましい)")
                else:
                    st.metric("RMSEA", rmsea)
