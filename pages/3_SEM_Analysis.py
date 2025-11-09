"""
SEM分析ページ

構造方程式モデリング（SEM）を使用して、
力量（スキル、資格、教育）の習得構造を分析し、
メンバーの現在の習得状況から次に取るべき力量を推薦します。

主な機能:
- メンバーの領域別プロファイル可視化（レーダーチャート）
- 持っている力量/持っていない力量の可視化
- SEMベースの推薦（次に取るべき力量）
- 領域別のスキル依存関係ネットワーク可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from skillnote_recommendation.ml.sem_only_recommender import SEMOnlyRecommender
from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    display_error_details,
)
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header,
    render_section_divider,
)

# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="CareerNavigator - SEM分析",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply rich UI styles
apply_rich_ui_styles()

# リッチなヘッダー
render_gradient_header(
    title="🔬 SEM分析 - 力量構造分析",
    icon="📊",
    description="構造方程式モデリング（SEM）を使用して、力量の習得構造を分析し、次に取るべき力量を推薦します"
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
members_clean = td["members_clean"]

# =========================================================
# SEMレコメンダーの初期化
# =========================================================

@st.cache_resource
def initialize_sem_recommender(_member_competence, _competence_master, _members_clean, num_domains):
    """SEMレコメンダーを初期化（キャッシュ付き）"""
    return SEMOnlyRecommender(
        member_competence_df=_member_competence,
        competence_master_df=_competence_master,
        member_master_df=_members_clean,
        num_domain_categories=num_domains,
    )

# サイドバーで設定
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ SEM設定")

num_domain_categories = st.sidebar.slider(
    "スキル領域の分類数",
    min_value=5,
    max_value=15,
    value=8,
    step=1,
    help="力量を何個の領域に分類するか"
)

# SEMレコメンダーを初期化
with st.spinner("SEMモデルを初期化中..."):
    try:
        sem_recommender = initialize_sem_recommender(
            member_competence,
            competence_master,
            members_clean,
            num_domain_categories
        )
        st.success("✅ SEMモデルの初期化が完了しました")
    except Exception as e:
        st.error(f"❌ SEMモデルの初期化に失敗しました: {e}")
        display_error_details(e, "SEMモデル初期化")
        st.stop()

# 全領域を取得
all_domains = sem_recommender.get_all_domains()

st.info(f"📊 力量を{len(all_domains)}個の領域に分類しました: {', '.join(all_domains)}")

# =========================================================
# メンバー選択
# =========================================================

st.markdown("---")
st.subheader("👤 メンバー選択")

# メンバーコードのリストを取得
member_codes = sorted(members_clean['メンバーコード'].unique())

# メンバー選択
col1, col2 = st.columns([3, 1])

with col1:
    selected_member = st.selectbox(
        "分析するメンバーを選択",
        options=member_codes,
        help="SEMで分析するメンバーを選択してください"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 分析を実行", type="primary", use_container_width=True):
        st.rerun()

# メンバー情報を表示
member_info = members_clean[members_clean['メンバーコード'] == selected_member]
if not member_info.empty:
    member_row = member_info.iloc[0]
    member_name = member_row.get('メンバー名', selected_member)

    st.markdown(f"### 📋 {member_name} ({selected_member})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("職種", member_row.get('職種', 'N/A'))
    with col2:
        st.metric("役職", member_row.get('役職名', 'N/A'))
    with col3:
        st.metric("職能等級", member_row.get('職能等級', 'N/A'))
    with col4:
        # 習得力量数
        member_comp_count = len(member_competence[member_competence['メンバーコード'] == selected_member])
        st.metric("習得力量数", member_comp_count)

# =========================================================
# メンバープロファイル取得
# =========================================================

member_profile = sem_recommender.get_member_profile(selected_member)
domain_scores = member_profile['overall_scores']
acquired_competences = member_profile['acquired_competences']

# =========================================================
# タブで表示
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 領域別プロファイル",
    "✅ 持っている力量 / ❌ 持っていない力量",
    "🎯 SEM推薦",
    "🕸️ 領域別ネットワーク"
])

# =========================================================
# タブ1: 領域別プロファイル
# =========================================================

with tab1:
    st.markdown("### 📊 メンバーの領域別習得度")

    st.info(
        "メンバーの各領域における習得度をレーダーチャートで可視化します。"
        "スコアが高いほど、その領域の力量を多く習得しています。"
    )

    # レーダーチャートを作成
    if domain_scores:
        fig = go.Figure()

        domains = list(domain_scores.keys())
        scores = [domain_scores[d] * 100 for d in domains]  # 0-100スケール

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=domains,
            fill='toself',
            name=member_name,
            marker=dict(color='#1f77b4'),
            line=dict(color='#1f77b4', width=2),
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                )
            ),
            showlegend=True,
            title="領域別習得度",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数値表示
        st.markdown("#### 📈 領域別習得度（数値）")

        domain_df = pd.DataFrame([
            {
                '領域': domain,
                '習得度': f"{score*100:.1f}%",
                'スコア': score,
            }
            for domain, score in domain_scores.items()
        ]).sort_values('スコア', ascending=False)

        st.dataframe(
            domain_df[['領域', '習得度']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("領域別スコアが取得できませんでした")

# =========================================================
# タブ2: 持っている力量 / 持っていない力量
# =========================================================

with tab2:
    st.markdown("### ✅ 持っている力量 / ❌ 持っていない力量")

    # 領域フィルタ
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_domain_for_gap = st.selectbox(
            "表示する領域を選択",
            options=['全領域'] + all_domains,
            key='domain_gap_filter'
        )

    # ギャップ分析
    if selected_domain_for_gap == '全領域':
        gaps = sem_recommender.get_competence_gaps(selected_member)
    else:
        gaps = sem_recommender.get_competence_gaps(selected_member, domain=selected_domain_for_gap)

    # 領域ごとに表示
    for domain, gap_list in gaps.items():
        with st.expander(f"📂 {domain} 領域", expanded=(selected_domain_for_gap == domain or selected_domain_for_gap == '全領域')):
            acquired = [g for g in gap_list if g['is_acquired']]
            not_acquired = [g for g in gap_list if not g['is_acquired']]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### ✅ 持っている力量 ({len(acquired)}件)")

                if acquired:
                    acquired_df = pd.DataFrame(acquired)
                    st.dataframe(
                        acquired_df[['competence_name', 'competence_type', 'level']].rename(columns={
                            'competence_name': '力量名',
                            'competence_type': 'タイプ',
                            'level': 'レベル',
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("この領域の力量はまだ習得していません")

            with col2:
                st.markdown(f"#### ❌ 持っていない力量 ({len(not_acquired)}件)")

                if not_acquired:
                    not_acquired_df = pd.DataFrame(not_acquired)
                    st.dataframe(
                        not_acquired_df[['competence_name', 'competence_type']].rename(columns={
                            'competence_name': '力量名',
                            'competence_type': 'タイプ',
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.success("✨ この領域の全ての力量を習得済みです！")

# =========================================================
# タブ3: SEM推薦
# =========================================================

with tab3:
    st.markdown("### 🎯 SEMベースの推薦 - 次に取るべき力量")

    st.info(
        "構造方程式モデリング（SEM）に基づいて、"
        "メンバーの現在の習得状況から統計的に次に取るべき力量を推薦します。"
    )

    # 推薦設定
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        top_n_sem = st.slider("推薦数", min_value=5, max_value=30, value=10, step=5)

    with col2:
        competence_types_sem = st.multiselect(
            "力量タイプ",
            options=['SKILL', 'EDUCATION', 'LICENSE'],
            default=['SKILL', 'EDUCATION', 'LICENSE'],
            key='sem_comp_types'
        )

    with col3:
        domain_filter_sem = st.selectbox(
            "領域フィルタ",
            options=['全領域'] + all_domains,
            key='sem_domain_filter'
        )

    with col4:
        min_significance = st.checkbox(
            "統計的に有意なもののみ",
            value=True,
            help="p < 0.05のパス係数を持つ推薦のみを表示"
        )

    # 推薦を実行
    if st.button("🚀 推薦を実行", type="primary", key='sem_recommend_btn'):
        with st.spinner("SEM推薦を実行中..."):
            try:
                recommendations = sem_recommender.recommend(
                    member_code=selected_member,
                    top_n=top_n_sem,
                    competence_type=competence_types_sem if competence_types_sem else None,
                    domain_filter=domain_filter_sem if domain_filter_sem != '全領域' else None,
                    min_significance=min_significance,
                )

                st.session_state.sem_recommendations = recommendations

                if recommendations:
                    st.success(f"✅ {len(recommendations)}件の推薦を生成しました")
                else:
                    st.warning("推薦できる力量が見つかりませんでした")

            except Exception as e:
                display_error_details(e, "SEM推薦")

    # 推薦結果を表示
    if 'sem_recommendations' in st.session_state:
        recommendations = st.session_state.sem_recommendations

        if recommendations:
            st.markdown("---")
            st.markdown("#### 📋 推薦結果")

            # 推薦をデータフレームに変換
            rec_data = []
            for i, rec in enumerate(recommendations, 1):
                rec_data.append({
                    '順位': i,
                    '力量名': rec.competence_name,
                    'タイプ': rec.competence_type,
                    '領域': rec.domain,
                    'SEMスコア': f"{rec.sem_score:.3f}",
                    '現在レベル': rec.current_level,
                    '目標レベル': rec.target_level,
                    'パス係数': f"{rec.path_coefficient:.3f}" if rec.path_coefficient else 'N/A',
                    '有意性': '✓' if rec.is_significant else '',
                })

            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, hide_index=True, use_container_width=True)

            # 詳細を展開表示
            st.markdown("---")
            st.markdown("#### 📖 推薦の詳細説明")

            for i, rec in enumerate(recommendations[:10], 1):  # 上位10件のみ
                with st.expander(f"#{i} {rec.competence_name}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**推薦理由:**")
                        st.write(rec.reason)

                        st.markdown(f"**カテゴリー:** {rec.category}")

                    with col2:
                        st.metric("SEMスコア", f"{rec.sem_score:.3f}")
                        st.metric("現在レベル", rec.current_level)
                        st.metric("目標レベル", rec.target_level)

                        if rec.path_coefficient:
                            st.metric("パス係数", f"{rec.path_coefficient:.3f}")

                        if rec.is_significant:
                            st.success("✓ 統計的に有意")
                        else:
                            st.info("統計的有意性なし")

            # CSVダウンロード
            st.markdown("---")
            csv = rec_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 推薦結果をCSVでダウンロード",
                data=csv,
                file_name=f'sem_recommendations_{selected_member}.csv',
                mime='text/csv',
            )

# =========================================================
# タブ4: 領域別ネットワーク
# =========================================================

with tab4:
    st.markdown("### 🕸️ 領域別スキル依存関係ネットワーク")

    st.info(
        "各領域内のスキル依存関係をネットワークグラフで可視化します。"
        "矢印は統計的に有意なパス（因果関係）を示しています。"
    )

    # 領域選択
    selected_network_domain = st.selectbox(
        "ネットワークを表示する領域",
        options=all_domains,
        key='network_domain'
    )

    # ネットワークを表示
    if st.button("📊 ネットワークを表示", type="primary", key='show_network_btn'):
        with st.spinner(f"{selected_network_domain} 領域のネットワークを生成中..."):
            try:
                fig = sem_recommender.visualize_domain_network(selected_network_domain)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # 領域情報を表示
                    domain_info = sem_recommender.get_domain_info(selected_network_domain)

                    if domain_info:
                        st.markdown("---")
                        st.markdown(f"#### 📊 {selected_network_domain} 領域の詳細情報")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("潜在変数数", domain_info['num_latent_factors'])

                        with col2:
                            total_paths = len(domain_info.get('path_coefficients', []))
                            st.metric("パス数", total_paths)

                        with col3:
                            sig_paths = sum(
                                1 for p in domain_info.get('path_coefficients', [])
                                if p.get('is_significant', False)
                            )
                            st.metric("有意なパス数", sig_paths)

                        # パス係数の詳細
                        if domain_info.get('path_coefficients'):
                            st.markdown("#### 📈 パス係数の詳細")

                            path_data = []
                            for p in domain_info['path_coefficients']:
                                path_data.append({
                                    '開始': p['from'].replace(f"{selected_network_domain}_", ""),
                                    '終了': p['to'].replace(f"{selected_network_domain}_", ""),
                                    'パス係数': f"{p['coefficient']:.3f}",
                                    't値': f"{p['t_value']:.3f}",
                                    'p値': f"{p['p_value']:.4f}",
                                    '有意性': '✓' if p['is_significant'] else '',
                                    '信頼区間': f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]"
                                })

                            path_df = pd.DataFrame(path_data)
                            st.dataframe(path_df, hide_index=True, use_container_width=True)
                else:
                    st.warning(f"{selected_network_domain} 領域のネットワークグラフを生成できませんでした")

            except Exception as e:
                display_error_details(e, "ネットワーク可視化")

# =========================================================
# フッター
# =========================================================

st.markdown("---")
st.markdown("""
### 💡 SEM分析について

**構造方程式モデリング（SEM）**は、観測データから潜在的な因果関係を推定する統計手法です。

このシステムでは：
- **測定モデル**: スキル → 潜在変数（初級/中級/上級）
- **構造モデル**: 潜在変数間の因果効果（初級→中級→上級）
- **統計的検定**: パス係数の有意性（p < 0.05）

を用いて、メンバーの習得構造を分析し、統計的根拠に基づいた推薦を行います。
""")
