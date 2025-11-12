"""
CareerNavigator - SEM階層的スキル推薦

スキル領域の階層構造（初級→中級→上級）に基づいた推薦
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from skillnote_recommendation.ml.skill_domain_hierarchy import SkillDomainHierarchy
from skillnote_recommendation.ml.skill_domain_sem_model import SkillDomainSEMModel
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)


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
    "🎓 SEM階層的スキル推薦",
    "スキル領域の階層構造に基づいた段階的な学習パス推薦"
)

# =========================================================
# データ読み込みチェック
# =========================================================
if "transformed_data" not in st.session_state:
    st.warning("⚠️ データが読み込まれていません。「データ読み込み」ページからデータを読み込んでください。")
    st.stop()

transformed_data = st.session_state.transformed_data

competence_master = transformed_data["competence_master"]
member_competence = transformed_data["member_competence"]
members_clean = transformed_data["members_clean"]

# デバッグ: members_cleanの構造を確認
with st.expander("🔍 デバッグ情報", expanded=False):
    st.write("**members_cleanのカラム:**", list(members_clean.columns))
    st.write("**members_cleanのサンプル（最初の3行）:**")
    st.dataframe(members_clean.head(3))


# =========================================================
# SEMモデルの学習
# =========================================================
st.markdown("---")
st.subheader("🔧 SEMモデルの設定と学習")

with st.expander("📖 SEM階層的スキル推薦とは？", expanded=False):
    st.markdown("""
    ### 概要
    構造方程式モデリング（SEM）を用いて、スキルの階層構造を明示的にモデル化します。

    ### 階層構造
    各スキル領域（プログラミング、データベースなど）を3段階に分類：
    - **Level 1（初級）**: 基礎的なスキル（例: Python基礎、Git）
    - **Level 2（中級）**: 応用的なスキル（例: Web開発、API開発）
    - **Level 3（上級）**: 専門的なスキル（例: システム設計、アーキテクチャ）

    ### 推薦ロジック
    1. メンバーの現在のスキルレベルを推定
    2. 現在のレベルが一定以上（デフォルト0.6）の場合、次のレベルを推薦
    3. 推薦理由を「〇〇領域の初級スキルを習得済み。次は中級スキルがおすすめ」のように明確に説明

    ### メリット
    - **説明可能性**: なぜそのスキルを推薦するのかが明確
    - **段階的学習**: 初級→中級→上級の順に学習できる
    - **個別化**: メンバーの現在のレベルに合わせた推薦
    """)

# SEMモデル学習のUI
col1, col2 = st.columns(2)

with col1:
    min_competences_per_level = st.number_input(
        "各レベルで最低限必要な力量数",
        min_value=2,
        max_value=10,
        value=3,
        help="各レベル（初級、中級、上級）で最低限この数以上の力量がないと、そのドメインのSEMモデルは学習されません"
    )

with col2:
    min_current_level_score = st.slider(
        "現在のレベルと判定する最小スコア",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="このスコア以上であれば、そのレベルを「習得済み」と判定します"
    )

# SEMモデルを学習
if st.button("🚀 SEMモデルを学習", type="primary"):
    with st.spinner("SEMモデルを学習中..."):
        try:
            # ドメイン階層を構築
            st.info("ステップ1: ドメイン階層を構築中...")
            domain_hierarchy = SkillDomainHierarchy(competence_master)

            # 統計情報を表示
            st.markdown("### 📊 スキル領域の統計")
            stats_df = domain_hierarchy.get_domain_statistics()
            st.dataframe(stats_df, use_container_width=True)

            # デバッグ: ドメイン階層の詳細を表示
            with st.expander("🔍 デバッグ: ドメイン階層の詳細", expanded=True):
                st.write(f"**総ドメイン数:** {len(domain_hierarchy.domains)}")
                st.write(f"**ドメインリスト:**")
                for domain in domain_hierarchy.domains:
                    st.write(f"- {domain.domain_name}: Level1={len(domain.level_1_competences)}, Level2={len(domain.level_2_competences)}, Level3={len(domain.level_3_competences)}")

            # SEMモデルを学習
            st.info("ステップ2: SEMモデルを学習中...")
            sem_model = SkillDomainSEMModel(
                member_competence=member_competence,
                competence_master=competence_master,
                domain_hierarchy=domain_hierarchy,
            )

            st.info(f"ステップ3: フィッティング開始（min_competences_per_level={int(min_competences_per_level)}）...")
            sem_model.fit(min_competences_per_level=int(min_competences_per_level))
            st.info(f"ステップ4: フィッティング完了")

            # Session stateに保存
            st.session_state.sem_model = sem_model
            st.session_state.domain_hierarchy = domain_hierarchy
            st.session_state.min_current_level_score = min_current_level_score

            # デバッグ: SEMモデルの学習結果を表示
            with st.expander("🔍 デバッグ: SEMモデル学習結果", expanded=False):
                st.write(f"**学習済みドメイン数:** {len(sem_model.sem_models)}")
                st.write(f"**学習済みドメイン:**", list(sem_model.sem_models.keys()))
                st.write(f"**ドメイン階層統計:**")
                st.dataframe(stats_df)

            st.success(f"✅ SEMモデル学習完了（{len(sem_model.sem_models)}ドメイン）")
            st.rerun()

        except Exception as e:
            st.error(f"❌ SEMモデル学習エラー: {e}")
            import traceback
            st.code(traceback.format_exc())


# =========================================================
# SEMモデルが学習済みの場合の分析・推薦
# =========================================================
if "sem_model" in st.session_state and st.session_state.sem_model.is_fitted:
    sem_model = st.session_state.sem_model
    domain_hierarchy = st.session_state.domain_hierarchy
    min_current_level_score = st.session_state.get("min_current_level_score", 0.6)

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

        # メンバーのスキルプロファイルを取得
        profile_df = sem_model.get_member_skill_profile(selected_member)

        # デバッグ: プロファイルの内容を確認
        with st.expander("🔍 デバッグ: プロファイル情報", expanded=False):
            st.write(f"**プロファイルの行数:** {len(profile_df)}")
            if len(profile_df) > 0:
                st.write("**プロファイルの内容:**")
                st.dataframe(profile_df)

        if len(profile_df) > 0:
            st.markdown("### 📈 スキルレベルプロファイル（レーダーチャート）")

            # レーダーチャートを作成
            fig_radar = go.Figure()

            categories = profile_df['Domain'].tolist()

            # Level 1（初級）
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_df['Level_1_Score'].tolist(),
                theta=categories,
                fill='toself',
                name='初級',
                line=dict(color='#3498db', width=2),
            ))

            # Level 2（中級）
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_df['Level_2_Score'].tolist(),
                theta=categories,
                fill='toself',
                name='中級',
                line=dict(color='#e74c3c', width=2),
            ))

            # Level 3（上級）
            fig_radar.add_trace(go.Scatterpolar(
                r=profile_df['Level_3_Score'].tolist(),
                theta=categories,
                fill='toself',
                name='上級',
                line=dict(color='#2ecc71', width=2),
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title=f"{selected_member}のスキルレベルプロファイル",
                height=500,
            )

            st.plotly_chart(fig_radar, use_container_width=True)

            # テーブル表示
            st.markdown("### 📊 スキルレベル詳細")

            # スコアを%表示に変換
            display_df = profile_df.copy()
            display_df['Level_1_Score'] = (display_df['Level_1_Score'] * 100).round(1).astype(str) + '%'
            display_df['Level_2_Score'] = (display_df['Level_2_Score'] * 100).round(1).astype(str) + '%'
            display_df['Level_3_Score'] = (display_df['Level_3_Score'] * 100).round(1).astype(str) + '%'

            display_df = display_df.rename(columns={
                'Domain': 'スキル領域',
                'Level_1_Score': '初級スコア',
                'Level_2_Score': '中級スコア',
                'Level_3_Score': '上級スコア',
            })

            st.dataframe(display_df, use_container_width=True)

        else:
            st.warning("⚠️ このメンバーのスキルプロファイルがありません")

        # =========================================================
        # SEM推薦結果
        # =========================================================
        st.markdown("---")
        st.markdown("### 🎯 SEM階層的スキル推薦結果")

        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            top_n_sem = st.number_input(
                "推薦数",
                min_value=1,
                max_value=20,
                value=10,
                key="top_n_sem"
            )

        with col_rec2:
            current_level_threshold = st.slider(
                "現在のレベル判定閾値",
                min_value=0.0,
                max_value=1.0,
                value=min_current_level_score,
                step=0.05,
                key="current_level_threshold"
            )

        # 推薦を生成
        with st.spinner("SEM推薦を生成中..."):
            recommendations = sem_model.recommend_next_skills(
                member_code=selected_member,
                top_n=int(top_n_sem),
                min_current_level_score=float(current_level_threshold)
            )

        if len(recommendations) > 0:
            # 推薦結果を表示
            st.success(f"✅ {len(recommendations)}件の推薦を生成しました")

            # 推薦結果をDataFrameに変換
            rec_df = pd.DataFrame(recommendations)

            # レベルを日本語に変換
            level_map = {1: '初級', 2: '中級', 3: '上級'}
            rec_df['level_name'] = rec_df['level'].map(level_map)

            # 表示用に整形
            display_rec_df = rec_df[[
                'competence_name',
                'domain',
                'level_name',
                'score',
                'reason'
            ]].copy()

            display_rec_df = display_rec_df.rename(columns={
                'competence_name': '力量名',
                'domain': 'スキル領域',
                'level_name': 'レベル',
                'score': 'スコア',
                'reason': '推薦理由',
            })

            display_rec_df['スコア'] = (display_rec_df['スコア'] * 100).round(1).astype(str) + '%'

            st.dataframe(display_rec_df, use_container_width=True)

            # ドメイン別の推薦数を可視化
            st.markdown("#### 📊 ドメイン別推薦数")

            domain_counts = rec_df['domain'].value_counts()

            fig_domain = px.bar(
                x=domain_counts.index,
                y=domain_counts.values,
                labels={'x': 'スキル領域', 'y': '推薦数'},
                title='ドメイン別推薦数',
                color=domain_counts.values,
                color_continuous_scale='viridis',
            )

            fig_domain.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_domain, use_container_width=True)

            # レベル別の推薦数を可視化
            st.markdown("#### 📊 レベル別推薦数")

            level_counts = rec_df['level'].value_counts().sort_index()
            level_names = [level_map[l] for l in level_counts.index]

            fig_level = px.bar(
                x=level_names,
                y=level_counts.values,
                labels={'x': 'レベル', 'y': '推薦数'},
                title='レベル別推薦数',
                color=level_counts.values,
                color_continuous_scale='blues',
            )

            fig_level.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_level, use_container_width=True)

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
            st.info("💡 このメンバーに推薦できるスキルがありません。現在のレベル判定閾値を下げてみてください。")


# =========================================================
# SEMモデルの詳細分析
# =========================================================
if "sem_model" in st.session_state and st.session_state.sem_model.is_fitted:
    st.markdown("---")
    st.subheader("🔍 SEMモデルの詳細分析")

    sem_model = st.session_state.sem_model

    with st.expander("📈 学習済みドメインの一覧", expanded=False):
        st.markdown(f"**学習済みドメイン数**: {len(sem_model.sem_models)}")

        for domain, sem_estimator in sem_model.sem_models.items():
            st.markdown(f"### {domain}")

            # 適合度指標
            if hasattr(sem_estimator, 'fit_info') and sem_estimator.fit_info:
                fit_info = sem_estimator.fit_info

                col_fit1, col_fit2, col_fit3 = st.columns(3)

                with col_fit1:
                    gfi = fit_info.get('gfi', 'N/A')
                    if isinstance(gfi, (int, float)):
                        st.metric("GFI", f"{gfi:.3f}")
                    else:
                        st.metric("GFI", gfi)

                with col_fit2:
                    agfi = fit_info.get('agfi', 'N/A')
                    if isinstance(agfi, (int, float)):
                        st.metric("AGFI", f"{agfi:.3f}")
                    else:
                        st.metric("AGFI", agfi)

                with col_fit3:
                    rmsea = fit_info.get('rmsea', 'N/A')
                    if isinstance(rmsea, (int, float)):
                        st.metric("RMSEA", f"{rmsea:.3f}")
                    else:
                        st.metric("RMSEA", rmsea)

            # パラメータ数
            if hasattr(sem_estimator, 'params') and sem_estimator.params:
                st.markdown(f"**推定パラメータ数**: {len(sem_estimator.params)}")
