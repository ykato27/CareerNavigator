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
from datetime import datetime

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

# 重要な説明
st.success("""
✨ **このページはNMFモデル学習なしで独立して使用できます！**

データ読み込みページでCSVファイルをアップロードすれば、
直接SEM分析を実行できます。モデル学習ページをスキップしてOKです。
""")

# 新しいSEM分析へのリンク
st.info("""
🆕 **新機能**: [高度なSEM分析ページ](/3_Advanced_SEM_Analysis)が利用可能です！

- ✅ 統一された目的関数による最尤推定
- ✅ 標準的な適合度指標（RMSEA, CFI, TLI）
- ✅ スキル1000個対応の階層的推定
- ✅ 既存モデルとの比較ダッシュボード

👉 サイドバーから「高度なSEM分析」を選択してください
""")

# 使い方ガイド
with st.expander("📖 使い方ガイド", expanded=False):
    st.markdown("""
    ### 🚀 SEM分析の使い方

    **1. データの準備**
    - データ読み込みページでCSVファイルをアップロード
    - ✅ NMFモデル学習は不要です！

    **2. メンバーを選択**
    - 分析したいメンバーを選択します

    **3. タブで分析**
    - **📊 領域別プロファイル**: メンバーの習得状況を可視化
    - **✅❌ 力量ギャップ**: 持っている/持っていない力量を確認
    - **🎯 SEM推薦**: 統計的根拠に基づく推薦を取得
    - **🕸️ ネットワーク**: 力量間の依存関係を可視化

    ### 💡 SEMとは？

    **構造方程式モデリング (Structural Equation Modeling)** は、
    観測データから潜在的な因果関係を推定する統計手法です。

    - **測定モデル**: 力量 → 潜在変数（初級/中級/上級）
    - **構造モデル**: 潜在変数間の因果効果（初級→中級→上級）
    - **統計的検定**: パス係数の有意性（p < 0.05）

    このシステムでは、メンバーの習得構造をSEMで分析し、
    統計的根拠に基づいた推薦を提供します。
    """)

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 領域別プロファイル",
    "✅ 持っている力量 / ❌ 持っていない力量",
    "🎯 SEM推薦",
    "🕸️ 領域別ネットワーク",
    "👥 メンバー比較"
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
            value=False,
            help="p < 0.05のパス係数を持つ推薦のみを表示（チェックを入れると推薦数が減る可能性があります）"
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
                    st.warning("⚠️ 推薦できる力量が見つかりませんでした")

                    # 診断情報を表示
                    st.info("""
                    **推薦が空になった可能性のある原因:**

                    1. **力量タイプフィルタが厳しすぎる**
                       - 現在の設定: {}
                       - 提案: すべてのタイプ（SKILL, EDUCATION, LICENSE）を選択してみてください

                    2. **「統計的に有意なもののみ」フィルタが有効**
                       - 現在の設定: {}
                       - 提案: チェックを外してみてください

                    3. **領域フィルタで絞り込みすぎている**
                       - 現在の設定: {}
                       - 提案: 「全領域」を選択してみてください

                    4. **既に多くの力量を習得済み**
                       - 未習得の力量が少ない可能性があります
                       - 提案: 推薦数を増やしてみてください
                    """.format(
                        competence_types_sem if competence_types_sem else "全て",
                        "有効" if min_significance else "無効",
                        domain_filter_sem
                    ))

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
    st.markdown("### 🕸️ 領域別スキル依存関係ネットワーク（インタラクティブ）")

    st.info(
        "**インタラクティブ機能:** マウスホイールでズーム、ドラッグでパン、ノード/エッジにホバーで詳細表示\n\n"
        "**ノード**: 個別の力量（スキル、教育、資格）を表示\n"
        "**エッジ**: 力量間の依存関係を表示\n"
        "**色**: 力量タイプ（🔵=スキル、🟠=教育、🟢=資格）"
    )

    # モデル再読み込みボタン
    if st.button("🔄 モデルを再読み込み", help="最新のコードでモデルを再構築します"):
        if 'sem_recommender' in st.session_state:
            del st.session_state['sem_recommender']
        st.rerun()

    # 領域選択とネットワークオプション
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_network_domain = st.selectbox(
            "ネットワークを表示する領域",
            options=all_domains,
            key='network_domain'
        )

    with col2:
        layout_type = st.selectbox(
            "レイアウト",
            options=["spring", "circular", "hierarchical"],
            index=0,
            key='network_layout',
            help="spring: 力学モデル（関係性が近いノードを近くに配置）\n"
                 "circular: 円形配置（全体を見やすく）\n"
                 "hierarchical: 階層配置（上下関係を重視）"
        )

    # フィルタリングオプション
    col1, col2, col3 = st.columns(3)

    with col1:
        show_all_edges = st.checkbox(
            "すべてのエッジを表示",
            value=True,
            help="有意でないパスも表示します（推奨：オン）"
        )

    with col2:
        # セッションステートで前回の値を保持
        if 'min_coefficient' not in st.session_state:
            st.session_state.min_coefficient = 0.0

        min_coefficient = st.slider(
            "最小パス係数",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.min_coefficient,
            step=0.05,
            help="この値未満のパスは表示しません",
            key='min_coef_slider'
        )

        # 値が変更されたら保存
        st.session_state.min_coefficient = min_coefficient

    # ネットワークを表示
    if st.button("📊 ネットワークを表示", type="primary", key='show_network_btn'):
        with st.spinner(f"{selected_network_domain} 領域のネットワークを生成中..."):
            try:
                # インタラクティブネットワークグラフを生成
                fig = sem_recommender.visualize_domain_network(
                    domain_name=selected_network_domain,
                    layout=layout_type,
                    show_all_edges=show_all_edges,
                    min_coefficient=min_coefficient
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # グラフ情報の表示
                    graph_data = sem_recommender.sem_model.get_skill_dependency_graph(selected_network_domain)
                    if graph_data:
                        n_nodes = len(graph_data.get('nodes', []))
                        n_edges = len(graph_data.get('edges', []))

                        if n_edges == 0:
                            st.warning(f"⚠️ このドメインにはエッジ（力量間の関係）がありません。パス係数が定義されていないか、すべてのスキルが1つの潜在変数に属している可能性があります。")

                        st.info(f"📊 ネットワーク情報: {n_nodes}個のノード（力量）、{n_edges}個のエッジ（関係）")

                    # モデル適合度指標を表示
                    st.markdown("---")
                    st.markdown(f"#### 📊 {selected_network_domain} 領域の統計情報")

                    fit_indices = sem_recommender.get_model_fit_indices(selected_network_domain)

                    if fit_indices:
                        # 基本統計
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("平均パス係数", f"{fit_indices['avg_path_coefficient']:.3f}")

                        with col2:
                            st.metric(
                                "有意なパス",
                                f"{fit_indices['significant_paths']}/{fit_indices['total_paths']}"
                            )

                        with col3:
                            st.metric("平均因子負荷量", f"{fit_indices['avg_loading']:.3f}")

                        with col4:
                            st.metric("平均効果サイズ", f"{fit_indices['avg_effect_size']:.3f}")

                        # モデル適合度指標
                        st.markdown("#### 🎯 モデル適合度指標")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            gfi = fit_indices['gfi']
                            gfi_status = "良好" if gfi >= 0.9 else "要改善"
                            st.metric(
                                "GFI (適合度指標)",
                                f"{gfi:.3f}",
                                delta=gfi_status,
                                delta_color="normal" if gfi >= 0.9 else "inverse"
                            )
                            st.caption("0.9以上が望ましい")

                        with col2:
                            nfi = fit_indices['nfi']
                            nfi_status = "良好" if nfi >= 0.9 else "要改善"
                            st.metric(
                                "NFI (規準適合度)",
                                f"{nfi:.3f}",
                                delta=nfi_status,
                                delta_color="normal" if nfi >= 0.9 else "inverse"
                            )
                            st.caption("0.9以上が望ましい")

                        with col3:
                            var_explained = fit_indices['variance_explained']
                            st.metric("説明分散 (R²)", f"{var_explained:.3f}")
                            st.caption("1に近いほど良好")

                    # 領域情報を表示
                    domain_info = sem_recommender.get_domain_info(selected_network_domain)

                    if domain_info:
                        st.markdown("---")
                        st.markdown(f"#### 📋 {selected_network_domain} 領域の構造詳細")

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
                                # 効果サイズの判定
                                coeff_abs = abs(p['coefficient'])
                                if coeff_abs < 0.2:
                                    effect_size = "小"
                                elif coeff_abs < 0.5:
                                    effect_size = "中"
                                else:
                                    effect_size = "大"

                                path_data.append({
                                    '開始': p['from'].replace(f"{selected_network_domain}_", ""),
                                    '終了': p['to'].replace(f"{selected_network_domain}_", ""),
                                    'パス係数': f"{p['coefficient']:.3f}",
                                    '効果サイズ': effect_size,
                                    't値': f"{p['t_value']:.3f}",
                                    'p値': f"{p['p_value']:.4f}",
                                    '有意性': '✓' if p['is_significant'] else '',
                                    '信頼区間': f"[{p['ci'][0]:.3f}, {p['ci'][1]:.3f}]"
                                })

                            path_df = pd.DataFrame(path_data)
                            st.dataframe(path_df, hide_index=True, use_container_width=True)

                            # 説明を追加
                            with st.expander("📖 統計指標の説明"):
                                st.markdown("""
                                **パス係数**: 潜在変数間の因果効果の強さ（-1～1）

                                **効果サイズ（Cohen's d）**:
                                - **小**: |係数| < 0.2（小さな効果）
                                - **中**: 0.2 ≤ |係数| < 0.5（中程度の効果）
                                - **大**: |係数| ≥ 0.5（大きな効果）

                                **t値**: パス係数の有意性を検定する統計量

                                **p値**: 統計的有意性（p < 0.05で有意）

                                **信頼区間**: パス係数の95%信頼区間

                                **GFI (Goodness of Fit Index)**: モデルの適合度（0.9以上が望ましい）

                                **NFI (Normed Fit Index)**: 規準適合度指標（0.9以上が望ましい）

                                **R² (説明分散)**: モデルが説明する分散の割合（1に近いほど良好）
                                """)
                else:
                    st.error(f"❌ {selected_network_domain} 領域のネットワークグラフを生成できませんでした")
                    st.info("""
                    **考えられる原因:**
                    - この領域にスキルデータがない
                    - この領域の潜在変数構造が構築されていない
                    - データが不足している

                    別の領域を選択してみてください。
                    """)

            except Exception as e:
                display_error_details(e, "ネットワーク可視化")

# =========================================================
# タブ5: メンバー比較
# =========================================================

with tab5:
    st.markdown("### 👥 複数メンバーの領域プロファイル比較")

    st.info(
        "複数のメンバーを選択して、領域別の習得度を比較できます。"
        "チームの傾向や個々のメンバーの強み・弱みを把握するのに役立ちます。"
    )

    # メンバー選択
    st.markdown("#### 📝 比較するメンバーを選択")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_members_for_comparison = st.multiselect(
            "比較するメンバーを選択（最大5名まで）",
            options=member_codes,
            default=[selected_member] if selected_member else [],
            max_selections=5,
            help="最大5名まで選択可能"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        compare_btn = st.button("📊 比較を実行", type="primary", use_container_width=True, key='compare_btn')

    if compare_btn and selected_members_for_comparison:
        with st.spinner("メンバープロファイルを比較中..."):
            try:
                # 各メンバーのプロファイルを取得
                comparison_data = []

                for member_code in selected_members_for_comparison:
                    profile = sem_recommender.get_member_profile(member_code)
                    member_info = members_clean[members_clean['メンバーコード'] == member_code]
                    member_name = member_info.iloc[0].get('メンバー名', member_code) if not member_info.empty else member_code

                    comparison_data.append({
                        'member_code': member_code,
                        'member_name': member_name,
                        'domain_scores': profile['overall_scores'],
                        'total_competences': profile['total_competences_count']
                    })

                # レーダーチャートで比較
                st.markdown("---")
                st.markdown("#### 📊 領域別習得度の比較（レーダーチャート）")

                fig = go.Figure()

                colors = px.colors.qualitative.Plotly

                for i, data in enumerate(comparison_data):
                    domains = list(data['domain_scores'].keys())
                    scores = [data['domain_scores'][d] * 100 for d in domains]

                    fig.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=domains,
                        fill='toself',
                        name=data['member_name'],
                        marker=dict(color=colors[i % len(colors)]),
                        line=dict(color=colors[i % len(colors)], width=2),
                        opacity=0.7,
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
                    title="メンバー別領域習得度",
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

                # 数値表で比較
                st.markdown("---")
                st.markdown("#### 📈 領域別習得度（数値表）")

                # 領域ごとにメンバーのスコアを集計
                all_domains_for_comparison = list(comparison_data[0]['domain_scores'].keys())

                comparison_table = []
                for domain in all_domains_for_comparison:
                    row = {'領域': domain}
                    for data in comparison_data:
                        score = data['domain_scores'].get(domain, 0.0)
                        row[data['member_name']] = f"{score*100:.1f}%"
                    comparison_table.append(row)

                comparison_df = pd.DataFrame(comparison_table)
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                # メンバーサマリー
                st.markdown("---")
                st.markdown("#### 📋 メンバーサマリー")

                summary_cols = st.columns(len(comparison_data))

                for i, data in enumerate(comparison_data):
                    with summary_cols[i]:
                        st.markdown(f"**{data['member_name']}**")
                        st.metric("習得力量数", data['total_competences'])

                        # 最も得意な領域
                        best_domain = max(data['domain_scores'].items(), key=lambda x: x[1])
                        st.metric("得意領域", best_domain[0])
                        st.caption(f"習得度: {best_domain[1]*100:.1f}%")

                        # 最も弱い領域
                        worst_domain = min(data['domain_scores'].items(), key=lambda x: x[1])
                        st.metric("成長領域", worst_domain[0])
                        st.caption(f"習得度: {worst_domain[1]*100:.1f}%")

                # CSV ダウンロード
                st.markdown("---")
                csv_comparison = comparison_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 比較結果をCSVでダウンロード",
                    data=csv_comparison,
                    file_name=f'member_comparison_{len(selected_members_for_comparison)}members.csv',
                    mime='text/csv',
                )

            except Exception as e:
                display_error_details(e, "メンバー比較")

    elif not selected_members_for_comparison:
        st.warning("比較するメンバーを選択してください")

# =========================================================
# レポート生成
# =========================================================

st.markdown("---")
st.markdown("## 📄 HTMLレポート生成")

st.info(
    "現在のメンバーの分析結果を包括的なHTMLレポートとして生成できます。"
    "ブラウザで開いてPDFとして保存することも可能です。"
)

if st.button("📥 HTMLレポートを生成", type="primary", key='generate_report_btn'):
    with st.spinner("レポートを生成中..."):
        try:
            from skillnote_recommendation.utils.report_generator import generate_html_report

            # 推薦データを取得（既に生成されている場合）
            if 'sem_recommendations' in st.session_state:
                recommendations = st.session_state.sem_recommendations
            else:
                # 推薦を新規生成
                recommendations = sem_recommender.recommend(
                    member_code=selected_member,
                    top_n=10,
                    min_significance=True,
                )

            # ギャップデータを取得
            gaps = sem_recommender.get_competence_gaps(selected_member)

            # メンバー情報を取得
            member_info_row = members_clean[members_clean['メンバーコード'] == selected_member]
            member_info_dict = {}
            if not member_info_row.empty:
                member_info_dict = {
                    '職種': member_info_row.iloc[0].get('職種', 'N/A'),
                    '役職名': member_info_row.iloc[0].get('役職名', 'N/A'),
                    '職能等級': member_info_row.iloc[0].get('職能等級', 'N/A'),
                }

            # 全領域のモデル適合度指標を取得
            fit_indices_all = {}
            for domain in all_domains:
                fit_indices_all[domain] = sem_recommender.get_model_fit_indices(domain)

            # HTMLレポートを生成
            html_report = generate_html_report(
                member_code=selected_member,
                member_name=member_name,
                member_info=member_info_dict,
                domain_scores=domain_scores,
                recommendations=recommendations,
                gaps_by_domain=gaps,
                fit_indices=fit_indices_all
            )

            # ダウンロードボタンを表示
            st.success("✅ レポートの生成が完了しました！")

            st.download_button(
                label="📥 HTMLレポートをダウンロード",
                data=html_report,
                file_name=f'SEM_Report_{selected_member}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html',
                mime='text/html',
                key='download_html_report'
            )

            st.info("""
            **💡 PDFとして保存する方法:**
            1. ダウンロードしたHTMLファイルをブラウザで開く
            2. ブラウザの印刷機能（Ctrl+P または Cmd+P）を開く
            3. 「送信先」または「プリンター」で「PDFに保存」を選択
            4. 保存ボタンをクリック
            """)

        except Exception as e:
            display_error_details(e, "レポート生成")

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
