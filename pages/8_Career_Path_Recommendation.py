"""
CareerNavigator - キャリアパス因果構造推薦

役職ごとの標準的なスキル習得パスをSEMでモデル化し、
メンバーの現在位置から次のステップを推薦
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from skillnote_recommendation.ml.career_path_hierarchy import CareerPathHierarchy
from skillnote_recommendation.ml.career_path_sem_model import CareerPathSEMModel
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header
)


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - キャリアパス推薦",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply rich UI styles
apply_rich_ui_styles()

# Header
render_gradient_header(
    "🎯 キャリアパス因果構造推薦",
    "役職ごとの標準的なスキル習得パスに基づくキャリア支援"
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
required_keys = ["competence_master", "member_competence"]
missing_keys = [key for key in required_keys if key not in transformed_data]
if missing_keys:
    st.error(f"❌ 必要なデータが不足しています: {', '.join(missing_keys)}")
    st.info(f"利用可能なキー: {', '.join(transformed_data.keys())}")
    st.warning("「データ読み込み」ページからデータを再度読み込んでください。")
    st.stop()

competence_master = transformed_data["competence_master"]
member_competence = transformed_data["member_competence"]

# member_masterの処理 (members_cleanがキーに存在する場合)
if "member_master" in transformed_data:
    member_master = transformed_data["member_master"]
elif "members_clean" in transformed_data:
    member_master = transformed_data["members_clean"]
else:
    st.error("❌ メンバーマスタデータが見つかりません（'member_master' または 'members_clean'）")
    st.info(f"利用可能なキー: {', '.join(transformed_data.keys())}")
    st.stop()

# 役職情報の確認
if '役職' not in member_master.columns:
    st.error("❌ メンバーマスタに「役職」列が存在しません。このページを使用するには役職情報が必要です。")
    st.stop()


# =========================================================
# キャリアパスSEMモデルの学習
# =========================================================
st.markdown("---")
st.subheader("🔧 キャリアパスSEMモデルの設定と学習")

with st.expander("📖 キャリアパス因果構造推薦とは？", expanded=False):
    st.markdown("""
    ### 概要
    構造方程式モデリング（SEM）を用いて、役職ごとのキャリアパスを因果構造としてモデル化します。

    ### キャリアステージ
    各役職を3～4段階のキャリアステージに分類：
    - **Stage 0（入門期）**: 基礎スキルの習得
    - **Stage 1（成長期）**: 応用スキルの習得
    - **Stage 2（熟達期）**: 専門スキルの習得
    - **Stage 3（エキスパート期）**: 高度な専門スキル（一部役職のみ）

    ### 因果構造
    ```
    入門期（潜在変数） → [観測スキル1, スキル2, ...]
         ↓ (パス係数 β=0.65)
    成長期（潜在変数） → [観測スキル3, スキル4, ...]
         ↓ (パス係数 β=0.58)
    熟達期（潜在変数） → [観測スキル5, スキル6, ...]
    ```

    ### 推薦ロジック
    1. メンバーの現在のキャリアステージと進捗率を推定
    2. 進捗率が80%未満 → 現在のステージのスキルを強化
    3. 進捗率が80%以上 → 次のステージへステップアップ

    ### メリット
    - **キャリアの見える化**: 現在位置と次のステップが明確
    - **段階的成長**: 基礎から順に学習できる
    - **個別化**: メンバーごとの進捗に合わせた推薦
    """)

# SEMモデル学習のUI
col1, col2 = st.columns(2)

with col1:
    min_members_per_role = st.number_input(
        "役職ごとの最低メンバー数",
        min_value=3,
        max_value=20,
        value=5,
        help="この数以上のメンバーがいる役職のみSEMモデルを学習します"
    )

with col2:
    min_skills_per_stage = st.number_input(
        "ステージごとの最低スキル数",
        min_value=2,
        max_value=10,
        value=3,
        help="各ステージでこの数以上のスキルがないと、そのステージは学習されません"
    )

# SEMモデルを学習
if st.button("🚀 キャリアパスSEMモデルを学習", type="primary"):
    with st.spinner("キャリアパスSEMモデルを学習中..."):
        try:
            # キャリアパス階層を構築
            career_hierarchy = CareerPathHierarchy(
                member_master, member_competence, competence_master
            )

            # 統計情報を表示
            st.markdown("### 📊 キャリアパスの統計")
            stats_df = career_hierarchy.get_career_path_statistics()
            st.dataframe(stats_df, use_container_width=True)

            # SEMモデルを学習
            career_sem_model = CareerPathSEMModel(
                member_master, member_competence, competence_master,
                career_path_hierarchy=career_hierarchy
            )

            career_sem_model.fit(
                min_members_per_role=int(min_members_per_role),
                min_skills_per_stage=int(min_skills_per_stage)
            )

            # Session stateに保存
            st.session_state.career_sem_model = career_sem_model
            st.session_state.career_hierarchy = career_hierarchy

            st.success(f"✅ キャリアパスSEMモデル学習完了（{len(career_sem_model.sem_models)}役職）")
            st.rerun()

        except Exception as e:
            st.error(f"❌ キャリアパスSEMモデル学習エラー: {e}")
            import traceback
            st.code(traceback.format_exc())


# =========================================================
# SEMモデルが学習済みの場合の分析・推薦
# =========================================================
if "career_sem_model" in st.session_state and st.session_state.career_sem_model.is_fitted:
    career_sem_model = st.session_state.career_sem_model
    career_hierarchy = st.session_state.career_hierarchy

    st.markdown("---")
    st.subheader("👤 メンバー別キャリア進捗")

    # メンバー選択
    member_codes = sorted(member_competence["メンバーコード"].unique())

    selected_member = st.selectbox(
        "メンバーを選択",
        options=member_codes,
        format_func=lambda x: f"{x} - {member_master[member_master['メンバーコード'] == x]['メンバー名'].values[0] if len(member_master[member_master['メンバーコード'] == x]) > 0 else x}"
    )

    if selected_member:
        # メンバーの現在位置を取得
        role, current_stage, progress = career_sem_model.get_member_position(selected_member)

        if role:
            st.markdown("### 📈 キャリア進捗サマリー")

            # 進捗サマリーを取得
            summary = career_sem_model.get_career_progression_summary(selected_member)

            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

            with col_sum1:
                st.metric("役職", summary['role'])

            with col_sum2:
                st.metric("現在のステージ", summary['current_stage_name'])

            with col_sum3:
                st.metric("進捗率", f"{summary['progress']*100:.0f}%")

            with col_sum4:
                st.metric("完了予測", f"約{summary['estimated_completion_months']}ヶ月")

            # プログレスバー
            st.progress(summary['progress'])

            # ステージ説明
            current_stage_info = career_hierarchy.get_stage_info(role, current_stage)

            if current_stage_info:
                st.info(
                    f"**{current_stage_info['name']}**: {current_stage_info['description']}\n\n"
                    f"標準的な期間: {current_stage_info['typical_duration_months']}ヶ月"
                )

            # キャリアパス全体の可視化
            st.markdown("### 🗺️ キャリアパス全体像")

            stages = career_hierarchy.get_role_stages(role)

            if stages:
                # フローチャート風の表示
                fig = go.Figure()

                stage_names = [s['name'] for s in stages]
                stage_nums = list(range(len(stages)))

                # ノードを追加
                for i, stage_info in enumerate(stages):
                    # 現在のステージを強調
                    if i == current_stage:
                        color = '#e74c3c'  # 赤
                        size = 30
                    elif i < current_stage:
                        color = '#2ecc71'  # 緑（完了）
                        size = 25
                    else:
                        color = '#95a5a6'  # グレー（未到達）
                        size = 25

                    fig.add_trace(go.Scatter(
                        x=[i],
                        y=[0],
                        mode='markers+text',
                        marker=dict(size=size, color=color),
                        text=[stage_info['name']],
                        textposition='top center',
                        name=stage_info['name'],
                        hovertemplate=(
                            f"<b>{stage_info['name']}</b><br>"
                            f"{stage_info['description']}<br>"
                            f"標準期間: {stage_info['typical_duration_months']}ヶ月<br>"
                            f"<extra></extra>"
                        )
                    ))

                # エッジ（矢印）を追加
                for i in range(len(stages) - 1):
                    # パス係数を取得
                    path_coefs = career_sem_model.path_coefficients.get(role, [])
                    if i < len(path_coefs):
                        beta = path_coefs[i]
                        annotation_text = f"β={beta:.2f}"
                    else:
                        annotation_text = ""

                    fig.add_annotation(
                        x=i + 0.5,
                        y=0,
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=12, color='#34495e')
                    )

                    fig.add_shape(
                        type="line",
                        x0=i,
                        y0=0,
                        x1=i + 1,
                        y1=0,
                        line=dict(color='#34495e', width=2),
                    )

                    fig.add_annotation(
                        x=i + 0.9,
                        y=0,
                        ax=i + 1,
                        ay=0,
                        xref='x',
                        yref='y',
                        axref='x',
                        ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='#34495e'
                    )

                fig.update_layout(
                    title=f"{role}のキャリアパス（現在: {summary['current_stage_name']}）",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 0.5]),
                    showlegend=False,
                    height=300,
                    hovermode='closest'
                )

                st.plotly_chart(fig, use_container_width=True)

            # 役職のキャリアパスサマリー
            with st.expander("📋 役職別キャリアパス詳細", expanded=False):
                path_summary_df = career_sem_model.get_role_path_summary(role)
                st.dataframe(path_summary_df, use_container_width=True)

        else:
            st.warning("⚠️ このメンバーの役職情報がありません")

        # =========================================================
        # キャリアパス推薦結果
        # =========================================================
        st.markdown("---")
        st.markdown("### 🎯 キャリアパス推薦結果")

        top_n_career = st.number_input(
            "推薦数",
            min_value=1,
            max_value=20,
            value=10,
            key="top_n_career"
        )

        # 推薦を生成
        with st.spinner("キャリアパス推薦を生成中..."):
            recommendations = career_sem_model.recommend_next_steps(
                member_code=selected_member,
                top_n=int(top_n_career)
            )

        # デバッグ情報を表示
        with st.expander("🔍 デバッグ情報", expanded=False):
            st.write(f"**選択メンバー:** {selected_member}")
            st.write(f"**役職:** {role}")
            st.write(f"**現在ステージ:** {current_stage}")
            st.write(f"**進捗率:** {progress:.2%}")

            # メンバーの習得スキル数
            member_skills = member_competence[
                member_competence['メンバーコード'] == selected_member
            ]
            st.write(f"**習得スキル数:** {len(member_skills)}")

            # 役職のステージ情報
            stages = career_hierarchy.get_role_stages(role) if role else []
            st.write(f"**役職のステージ数:** {len(stages)}")

            # 各ステージのスキル数をカウント
            if role and stages:
                st.write("**各ステージのスキル数（未習得のみ）:**")
                acquired_skills = set(member_skills['力量コード'].tolist())
                for i in range(len(stages)):
                    stage_skills = career_hierarchy.get_skills_by_stage(
                        role, i, acquired_skills
                    )
                    st.write(f"  - Stage {i} ({stages[i]['name']}): {len(stage_skills)}個")

            st.write(f"**推薦結果数:** {len(recommendations)}")

        if len(recommendations) > 0:
            st.success(f"✅ {len(recommendations)}件の推薦を生成しました")

            # 推薦結果をDataFrameに変換
            rec_df = pd.DataFrame(recommendations)

            # Path Alignment Scoreを計算して追加
            path_scores = []
            for rec in recommendations:
                score = career_sem_model.calculate_path_alignment_score(
                    selected_member,
                    rec['competence_code']
                )
                path_scores.append(score)

            rec_df['path_alignment_score'] = path_scores

            # 表示用に整形
            display_rec_df = rec_df[[
                'competence_name',
                'stage',
                'stage_name',
                'path_coefficient',
                'path_alignment_score',
                'reason'
            ]].copy()

            display_rec_df = display_rec_df.rename(columns={
                'competence_name': '力量名',
                'stage': 'ステージ番号',
                'stage_name': 'ステージ名',
                'path_coefficient': 'パス係数',
                'path_alignment_score': 'パス親和性',
                'reason': '推薦理由',
            })

            st.dataframe(display_rec_df, use_container_width=True)

            # 各推薦の詳細を展開可能にする
            st.markdown("#### 📝 推薦の詳細説明")

            for i, rec in enumerate(recommendations[:5]):  # 上位5件を表示
                with st.expander(f"{i+1}. {rec['competence_name']} (パス親和性: {path_scores[i]:.2f})"):
                    # 推薦理由を生成
                    explanation = career_sem_model.generate_path_explanation(
                        selected_member,
                        rec['competence_code']
                    )

                    st.markdown(explanation)

                    # スキル詳細情報
                    st.markdown("---")
                    st.markdown("**スキル詳細**")

                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.write(f"力量コード: `{rec['competence_code']}`")
                        st.write(f"推薦ステージ: {rec['stage_name']} (Stage {rec['stage']})")

                    with detail_col2:
                        st.write(f"Path Alignment Score: **{path_scores[i]:.2f}**")
                        st.write(f"パス係数: **{rec.get('path_coefficient', 0.0):.2f}**")

                    # スキルの意味を解説
                    if path_scores[i] >= 0.8:
                        st.success("✅ このスキルは現在または次のステージで重要です。優先的に習得することをお勧めします。")
                    elif path_scores[i] >= 0.5:
                        st.info("ℹ️ このスキルは将来のステージで重要です。先行学習として検討してください。")
                    else:
                        st.warning("⚠️ このスキルは優先度が低いか、パス上にないスキルです。")

            # ステージ別の推薦数を可視化
            st.markdown("#### 📊 ステージ別推薦数")

            stage_counts = rec_df['stage_name'].value_counts()

            fig_stage = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                labels={'x': 'ステージ', 'y': '推薦数'},
                title='ステージ別推薦数',
                color=stage_counts.values,
                color_continuous_scale='viridis',
            )

            fig_stage.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_stage, use_container_width=True)

            # パス係数の分布
            if 'path_coefficient' in rec_df.columns:
                path_coefs = rec_df[rec_df['path_coefficient'] > 0]['path_coefficient']

                if len(path_coefs) > 0:
                    st.markdown("#### 📊 パス係数の分布")

                    fig_coef = px.histogram(
                        path_coefs,
                        nbins=10,
                        labels={'value': 'パス係数', 'count': '頻度'},
                        title='パス係数の分布（ステージ間の因果関係の強さ）',
                    )

                    fig_coef.update_layout(height=400)
                    st.plotly_chart(fig_coef, use_container_width=True)

                    st.info(
                        "💡 **パス係数（β）の解釈**:\n"
                        "- β > 0.6: 前のステージの完了が次のステージ進出に強く影響\n"
                        "- 0.4 < β ≤ 0.6: 中程度の影響\n"
                        "- β ≤ 0.4: 弱い影響"
                    )

            # CSVダウンロード
            st.markdown("#### 📥 推薦結果のダウンロード")

            csv_data = display_rec_df.to_csv(index=False).encode('utf-8-sig')

            st.download_button(
                label="📥 推薦結果をCSVでダウンロード",
                data=csv_data,
                file_name=f"career_path_recommendations_{selected_member}.csv",
                mime="text/csv",
            )

        else:
            st.info("💡 このメンバーに推薦できるスキルがありません。")


# =========================================================
# 役職別の全体分析
# =========================================================
if "career_sem_model" in st.session_state and st.session_state.career_sem_model.is_fitted:
    st.markdown("---")
    st.subheader("🔍 役職別キャリアパス分析")

    career_sem_model = st.session_state.career_sem_model

    # 学習済み役職を選択
    trained_roles = list(career_sem_model.sem_models.keys())

    if trained_roles:
        selected_role = st.selectbox(
            "役職を選択",
            options=trained_roles
        )

        if selected_role:
            # 役職のキャリアパス全体
            st.markdown(f"### 📋 {selected_role}のキャリアパス")

            path_summary_df = career_sem_model.get_role_path_summary(selected_role)
            st.dataframe(path_summary_df, use_container_width=True)

            # この役職のメンバー分布
            st.markdown(f"### 📊 {selected_role}メンバーのステージ分布")

            role_members = member_master[
                member_master['役職'] == selected_role
            ]['メンバーコード'].tolist()

            if role_members:
                stage_distribution = {}

                for member_code in role_members:
                    _, stage, _ = career_sem_model.get_member_position(member_code)

                    if stage not in stage_distribution:
                        stage_distribution[stage] = 0

                    stage_distribution[stage] += 1

                # グラフ化
                stages_list = sorted(stage_distribution.keys())
                counts_list = [stage_distribution[s] for s in stages_list]

                # ステージ名を取得
                stage_names_list = []
                for s in stages_list:
                    stage_info = career_hierarchy.get_stage_info(selected_role, s)
                    stage_names_list.append(
                        stage_info['name'] if stage_info else f'Stage {s}'
                    )

                fig_dist = px.bar(
                    x=stage_names_list,
                    y=counts_list,
                    labels={'x': 'ステージ', 'y': 'メンバー数'},
                    title=f'{selected_role}メンバーのステージ分布',
                    color=counts_list,
                    color_continuous_scale='blues',
                )

                fig_dist.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
