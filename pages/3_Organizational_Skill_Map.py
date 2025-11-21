import streamlit as st
import pandas as pd
import numpy as np

from skillnote_recommendation.organizational.skill_gap_analyzer import SkillGapAnalyzer
from skillnote_recommendation.organizational import org_metrics
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)
from skillnote_recommendation.utils.org_ui_components import (
    render_skill_heatmap,
    render_skill_distribution_chart,
    render_gap_ranking_table,
    render_skill_matrix_table,
    render_export_buttons,
    render_metric_cards_row,
    render_cross_tab_heatmap
)

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - 組織スキルマップ",
    page_icon="🏢",
    layout="wide"
)

apply_enterprise_styles()

render_page_header(
    title="組織スキルマップ",
    icon="🏢",
    description="組織全体のスキル保有状況を可視化し、スキルギャップを分析します"
)

# =========================================================
# データチェック
# =========================================================
if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
    st.warning("まずはトップページでデータを読み込んでください。")
    st.stop()

td = st.session_state.transformed_data

# 必要なデータの確認
required_keys = ["member_competence", "competence_master", "members_clean"]
missing_keys = [key for key in required_keys if key not in td]

if missing_keys:
    st.error(f"必要なデータが不足しています: {', '.join(missing_keys)}")
    st.stop()

member_competence_df = td["member_competence"]
competence_master_df = td["competence_master"]
members_df = td["members_clean"]

# =========================================================
# データクリーニング: カラム名の正規化
# =========================================================
def clean_column_name(col_name: str) -> str:
    """カラム名から ###[...]### を削除"""
    import re
    return re.sub(r'\s*###\[.*?\]###', '', col_name).strip()

members_df.columns = [clean_column_name(col) for col in members_df.columns]

# =========================================================
# タブ構成
# =========================================================
st.markdown("---")
tab1, tab2, tab3 = st.tabs([
    "📊 組織スキルマップダッシュボード",
    "📉 スキルギャップ分析", 
    "👥 人材スキルマトリクス"
])

# =========================================================
# タブ1: 組織スキルマップダッシュボード
# =========================================================
with tab1:
    st.subheader("📊 組織全体のスキル保有状況")
    
    # KPIメトリクス
    total_members = len(members_df)
    total_skills = len(competence_master_df)
    total_skill_records = len(member_competence_df)
    avg_skills_per_member = total_skill_records / total_members if total_members > 0 else 0
    
    coverage_info = org_metrics.calculate_skill_coverage(
        member_competence_df, competence_master_df
    )
    
    concentration_info = org_metrics.calculate_skill_concentration(
        member_competence_df, threshold=3
    )
    
    diversity_index = org_metrics.calculate_skill_diversity_index(
        member_competence_df
    )
    
    # メトリクスカード表示
    metrics = [
        {"label": "総メンバー数", "value": f"{total_members:,}人"},
        {"label": "1人あたり平均スキル数", "value": f"{avg_skills_per_member:.1f}"},
        {"label": "スキルカバレッジ率", "value": f"{coverage_info['coverage_rate']*100:.1f}%"},
        {"label": "スキル多様性指標", "value": f"{diversity_index:.2f}"}
    ]
    render_metric_cards_row(metrics)
    
    st.markdown("---")
    
    # スキルカテゴリ別分布
    st.markdown("### 📈 スキルカテゴリ別分布")
    
    if "力量タイプ" in member_competence_df.columns:
        category_dist = member_competence_df["力量タイプ"].value_counts().reset_index()
        category_dist.columns = ["カテゴリ", "保有件数"]
        
        render_skill_distribution_chart(
            category_dist,
            x_col="カテゴリ",
            y_col="保有件数",
            title="スキルカテゴリ別保有件数"
        )
    else:
        st.info("スキルカテゴリ情報がありません")
    
    st.markdown("---")
    
    # 職種×役職別クロス集計
    st.markdown("### 🔲 職種×役職別スキル集計")
    
    if "職種" in members_df.columns and "役職" in members_df.columns:
        try:
            cross_tab = org_metrics.calculate_cross_group_summary(
                member_competence_df,
                members_df,
                group_by_1="職種",
                group_by_2="役職"
            )
            
            st.write("**1人あたり平均スキル数**")
            render_cross_tab_heatmap(cross_tab, title="職種×役職別平均スキル数")
            
        except Exception as e:
            st.error(f"クロス集計の計算中にエラーが発生しました: {e}")
    else:
        st.warning("職種または役職情報がメンバーマスタに含まれていません")
    
    st.markdown("---")
    
    # 等級別集計
    st.markdown("### 📊 等級別スキル集計")
    
    if "職能・等級" in members_df.columns:
        try:
            grade_summary = org_metrics.calculate_group_skill_summary(
                member_competence_df,
                members_df,
                group_by="職能・等級"
            )
            
            st.dataframe(grade_summary, use_container_width=True, height=300)
            
        except Exception as e:
            st.error(f"等級別集計の計算中にエラーが発生しました: {e}")
    else:
        st.warning("等級情報がメンバーマスタに含まれていません")

# =========================================================
# タブ2: スキルギャップ分析
# =========================================================
with tab2:
    st.subheader("📉 スキルギャップ分析")
    
    st.markdown("""
    組織として目指すべきスキル水準と現状のギャップを分析します。
    **上位N%のメンバー**の平均スキルを目標として設定します。
    """)
    
    # ターゲット設定
    with st.expander("⚙️ ターゲット設定", expanded=True):
        percentile = st.slider(
            "上位何%のメンバーを目標とするか",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="スキル保有数が多い上位N%のメンバーの平均を目標として設定します"
        ) / 100.0
        
        if st.button("🎯 ギャップを計算", type="primary"):
            with st.spinner("ギャップを計算中..."):
                try:
                    # ギャップ分析エンジンを初期化
                    analyzer = SkillGapAnalyzer()
                    
                    # 現在のプロファイル計算
                    current_profile = analyzer.calculate_current_profile(
                        member_competence_df,
                        competence_master_df
                    )
                    
                    # 目標プロファイル計算（上位N%方式）
                    target_profile = analyzer.calculate_target_profile_top_percentile(
                        member_competence_df,
                        competence_master_df,
                        percentile=percentile
                    )
                    
                    # ギャップ計算
                    gap_df = analyzer.calculate_gap(current_profile, target_profile)
                    
                    # セッションステートに保存
                    st.session_state.gap_analyzer = analyzer
                    st.session_state.gap_df = gap_df
                    st.session_state.percentile_used = percentile
                    
                    st.success(f"✅ ギャップ計算が完了しました（目標: 上位{percentile*100:.0f}%）")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"ギャップ計算中にエラーが発生しました: {e}")
                    st.exception(e)
    
    # ギャップ結果の表示
    if "gap_df" in st.session_state and st.session_state.gap_df is not None:
        gap_df = st.session_state.gap_df
        percentile_used = st.session_state.get("percentile_used", 0.2)
        
        st.info(f"📊 **設定中の目標**: 上位{percentile_used*100:.0f}%のメンバーの平均スキルセット")
        
        st.markdown("---")
        st.markdown("### 🔝 ギャップが大きいスキルTop 10")
        
        render_gap_ranking_table(gap_df, top_n=10)
        
        # エクスポート
        st.markdown("### 💾 データエクスポート")
        render_export_buttons(gap_df, filename_prefix="skill_gap_analysis")
        
        st.markdown("---")
        
        # クリティカルスキル
        st.markdown("### ⚠️ クリティカルスキル（ギャップ率30%以上)")
        
        critical_threshold = st.slider(
            "クリティカルスキルの閾値（ギャップ率）",
            min_value=10,
            max_value=70,
            value=30,
            step=10
        ) / 100.0
        
        analyzer = st.session_state.get("gap_analyzer")
        if analyzer:
            critical_skills = analyzer.identify_critical_skills(
                gap_df, threshold=critical_threshold
            )
            
            if len(critical_skills) > 0:
                st.warning(f"⚠️ {len(critical_skills)}件のクリティカルスキルが見つかりました")
                
                # クリティカルスキルの詳細表示
                for idx, row in critical_skills.head(5).iterrows():
                    with st.container():
                        st.markdown(f"#### {idx+1}. {row['力量名']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("現在保有率", f"{row['現在保有率']*100:.1f}%")
                        with col2:
                            st.metric("目標保有率", f"{row['目標保有率']*100:.1f}%")
                        with col3:
                            st.metric("ギャップ", f"{row['保有率ギャップ']*100:.1f}%", 
                                    delta=f"{row['保有率ギャップ率']*100:.1f}%", delta_color="inverse")
                        st.markdown("---")
            else:
                st.success("✅ クリティカルスキルはありません")
    else:
        st.info("👆 上記の「ギャップを計算」ボタンをクリックしてギャップ分析を開始してください")

# =========================================================
# タブ3: 人材スキルマトリクス
# =========================================================
with tab3:
    st.subheader("👥 人材スキルマトリクス")
    
    st.markdown("メンバー × スキルのマトリクスを表示します。フィルタリングして絞り込みができます。")
    
    # フィルタUI
    with st.expander("🔍 フィルター設定", expanded=True):
        filter_cols = st.columns(3)
        
        filters = {}
        
        with filter_cols[0]:
            if "職種" in members_df.columns:
                occupation_options = members_df["職種"].dropna().unique().tolist()
                selected_occupations = st.multiselect(
                    "職種でフィルター",
                    options=occupation_options,
                    default=[]
                )
                if selected_occupations:
                    filters["職種"] = selected_occupations
        
        with filter_cols[1]:
            if "役職" in members_df.columns:
                position_options = members_df["役職"].dropna().unique().tolist()
                selected_positions = st.multiselect(
                    "役職でフィルター",
                    options=position_options,
                    default=[]
                )
                if selected_positions:
                    filters["役職"] = selected_positions
        
        with filter_cols[2]:
            if "職能・等級" in members_df.columns:
                grade_options = members_df["職能・等級"].dropna().unique().tolist()
                selected_grades = st.multiselect(
                    "等級でフィルター",
                    options=grade_options,
                    default=[]
                )
                if selected_grades:
                    filters["等級"] = selected_grades
    
    st.markdown("---")
    
    # マトリクステーブル表示
    try:
        matrix_df = render_skill_matrix_table(
            member_competence_df,
            competence_master_df,
            members_df,
            filters=filters
        )
        
        st.success(f"✅ {len(matrix_df)}人のメンバーを表示中")
        
        # エクスポート
        st.markdown("### 💾 データエクスポート")
        render_export_buttons(matrix_df, filename_prefix="skill_matrix")
        
    except Exception as e:
        st.error(f"マトリクステーブルの表示中にエラーが発生しました: {e}")
        st.exception(e)
