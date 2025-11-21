import streamlit as st
import pandas as pd
import re

from skillnote_recommendation.strategic.succession_planner import SuccessionPlanner
from skillnote_recommendation.strategic.org_simulator import OrganizationSimulator
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)
from skillnote_recommendation.utils.strategic_ui_components import (
    render_succession_candidate_table,
    render_readiness_gauge,
    render_skill_gap_comparison,
    render_transfer_simulator_ui,
    render_before_after_comparison,
    render_skill_distribution_comparison
)
from skillnote_recommendation.utils.org_ui_components import render_export_buttons

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="CareerNavigator - 戦略的人材配置",
    page_icon="🎯",
    layout="wide"
)

apply_enterprise_styles()

render_page_header(
    title="戦略的人材配置",
    icon="🎯",
    description="後継者計画と組織シミュレーションで戦略的な人材配置を支援します"
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
    return re.sub(r'\s*###\[.*?\]###', '', col_name).strip()

members_df.columns = [clean_column_name(col) for col in members_df.columns]
competence_master_df.columns = [clean_column_name(col) for col in competence_master_df.columns]

# =========================================================
# タブ構成
# =========================================================
st.markdown("---")
tab1, tab2 = st.tabs([
    "👔 後継者計画（サクセッションプラン）",
    "🔄 組織シミュレーション"
])

# =========================================================
# タブ1: 後継者計画
# =========================================================
with tab1:
    st.subheader("👔 後継者計画（サクセッションプラン）")
    
    st.markdown("""
    重要ポジション（役職）の後継者候補を特定し、準備度を評価します。
    """)
    
    # 役職選択
    with st.expander("⚙️ 対象役職設定", expanded=True):
        if "役職" in members_df.columns:
            position_options = members_df["役職"].dropna().unique().tolist()
            
            # 重要ポジションを自動抽出
            planner = SuccessionPlanner()
            critical_positions = planner.identify_critical_positions(members_df, position_column="役職")
            
            if critical_positions:
                st.info(f"💡 自動検出された重要ポジション: {', '.join(critical_positions[:5])}")
                default_position = critical_positions[0] if critical_positions else position_options[0]
            else:
                default_position = position_options[0] if position_options else None
            
            selected_position = st.selectbox(
                "後継者を探す役職",
                options=position_options,
                index=position_options.index(default_position) if default_position in position_options else 0
            )
            
            if st.button("🔍 後継者候補を検索", type="primary"):
                with st.spinner("後継者候補を検索中..."):
                    try:
                        # スキルプロファイル計算
                        profile = planner.calculate_position_skill_profile(
                            selected_position,
                            members_df,
                            member_competence_df,
                            competence_master_df,
                            position_column="役職"
                        )
                        
                        # 候補者検索
                        candidates = planner.find_succession_candidates(
                            selected_position,
                            members_df,
                            member_competence_df,
                            competence_master_df,
                            position_column="役職",
                            grade_column="職能・等級",
                            exclude_current_holders=True,
                            max_candidates=20
                        )
                        
                        # セッションステートに保存
                        st.session_state.succession_planner = planner
                        st.session_state.succession_candidates = candidates
                        st.session_state.target_position = selected_position
                        st.session_state.target_profile = profile
                        
                        st.success(f"✅ {len(candidates)}人の候補者が見つかりました")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"候補者検索中にエラーが発生しました: {e}")
                        st.exception(e)
        else:
            st.warning("⚠️ メンバーマスタに役職カラムがありません")
    
    # 結果表示
    if "succession_candidates" in st.session_state and st.session_state.succession_candidates is not None:
        candidates_df = st.session_state.succession_candidates
        target_position = st.session_state.target_position
        
        st.markdown("---")
        st.markdown(f"### 🎯 **{target_position}** の後継者候補ランキング")
        
        if len(candidates_df) > 0:
            # ランキングテーブル
            render_succession_candidate_table(candidates_df, top_n=10)
            
            # エクスポート
            st.markdown("### 💾 データエクスポート")
            render_export_buttons(candidates_df[[
                "メンバーコード", "メンバー名", "現在の役職", "現在の等級",
                "準備度スコア", "スキルマッチ度", "保有スキル数", "不足スキル数"
            ]], filename_prefix=f"succession_candidates_{target_position}")
            
            st.markdown("---")
            
            # Top3候補の詳細
            st.markdown("### 🌟 Top 3 候補の詳細")
            
            for idx, row in candidates_df.head(3).iterrows():
                with st.expander(f"#{idx+1}: {row['メンバー名']} (準備度: {row['準備度スコア']*100:.1f}%)", expanded=(idx==0)):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("準備度スコア", f"{row['準備度スコア']*100:.1f}%")
                    with col2:
                        st.metric("スキルマッチ度", f"{row['スキルマッチ度']*100:.1f}%")
                    with col3:
                        st.metric("保有スキル", f"{row['保有スキル数']}個")
                    with col4:
                        timeline = SuccessionPlanner().estimate_development_timeline(row['不足スキル数'])
                        st.metric("推定育成期間", timeline)
                    
                    # 不足スキルリスト
                    if "総合スコア詳細" in row and "missing_skill_codes" in row["総合スコア詳細"]:
                        missing_codes = row["総合スコア詳細"]["missing_skill_codes"]
                        if missing_codes:
                            missing_names = competence_master_df[
                                competence_master_df["力量コード"].isin(missing_codes)
                            ]["力量名"].tolist()[:10]
                            st.markdown(f"**不足スキル（上位10件）**: {', '.join(missing_names)}")
        else:
            st.info("候補者が見つかりませんでした")
    else:
        st.info("👆 上記の「後継者候補を検索」ボタンをクリックして分析を開始してください")

# =========================================================
# タブ2: 組織シミュレーション
# =========================================================
with tab2:
    st.subheader("🔄 組織シミュレーション")
    
    st.markdown("""
    職種間のメンバー異動をシミュレーションし、スキル分布への影響を分析します。
    """)
    
    # シミュレーター初期化
    if "org_simulator" not in st.session_state:
        st.session_state.org_simulator = OrganizationSimulator()
    
    simulator = st.session_state.org_simulator
    
    # 現状キャプチャ
    col_capture, col_reset = st.columns([3, 1])
    
    with col_capture:
        if st.button("📸 現在の組織状態をキャプチャ", type="primary"):
            with st.spinner("組織状態をキャプチャ中..."):
                try:
                    current_state = simulator.capture_current_state(
                        members_df,
                        member_competence_df,
                        competence_master_df,
                        group_by="職種"
                    )
                    st.success("✅ 現在の組織状態をキャプチャしました")
                    st.session_state.org_current_captured = True
                except Exception as e:
                    st.error(f"キャプチャ中にエラーが発生しました: {e}")
    
    with col_reset:
        if st.button("🔄 リセット"):
            simulator.reset_simulation()
            st.session_state.org_current_captured = False
            st.success("シミュレーションをリセットしました")
            st.rerun()
    
    if st.session_state.get("org_current_captured", False):
        st.markdown("---")
        
        # 異動設定UI
        if "職種" in members_df.columns:
            transfer_config = render_transfer_simulator_ui(members_df, group_column="職種")
            
            if st.button("➕ 異動を追加"):
                simulator.simulate_transfer(
                    transfer_config["member_code"],
                    transfer_config["from_group"],
                    transfer_config["to_group"],
                    group_column="職種"
                )
                st.success(f"✅ {transfer_config['member_name']} の異動を追加しました")
                st.session_state.transfers_added = True
            
            # 追加済み異動リスト
            if len(simulator.transfers) > 0:
                st.markdown("#### 設定済み異動リスト")
                for i, transfer in enumerate(simulator.transfers):
                    st.text(f"{i+1}. {transfer['from_group']} → {transfer['to_group']}")
        
        # シミュレーション実行
        if len(simulator.transfers) > 0:
            st.markdown("---")
            
            if st.button("🚀 シミュレーション実行", type="primary"):
                with st.spinner("シミュレーション実行中..."):
                    try:
                        simulated_state = simulator.execute_simulation(competence_master_df)
                        st.session_state.org_simulated = True
                        st.success("✅ シミュレーションが完了しました")
                        st.rerun()
                    except Exception as e:
                        st.error(f"シミュレーション実行中にエラーが発生しました: {e}")
                        st.exception(e)
        
        # 結果表示
        if st.session_state.get("org_simulated", False):
            st.markdown("---")
            st.markdown("### 📊 シミュレーション結果")
            
            try:
                # 前後比較
                comparison_df = simulator.compare_states()
                
                render_before_after_comparison(comparison_df)
                
                # グラフ表示
                st.markdown("---")
                render_skill_distribution_comparison(
                    simulator.current_state["group_summary"],
                    simulator.simulated_state["group_summary"]
                )
                
                # バランススコア
                st.markdown("---")
                st.markdown("### ⚖️ 組織バランススコア")
                
                current_balance = simulator.calculate_balance_score(simulator.current_state)
                simulated_balance = simulator.calculate_balance_score(simulator.simulated_state)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("現在", f"{current_balance:.3f}")
                with col2:
                    st.metric("シミュレーション後", f"{simulated_balance:.3f}")
                with col3:
                    delta = simulated_balance - current_balance
                    st.metric("変化", f"{delta:+.3f}", delta=f"{delta:+.3f}")
                
                # エクスポート
                st.markdown("---")
                st.markdown("### 💾 データエクスポート")
                render_export_buttons(comparison_df, filename_prefix="org_simulation_comparison")
                
            except Exception as e:
                st.error(f"結果表示中にエラーが発生しました: {e}")
                st.exception(e)
    else:
        st.info("👆 「現在の組織状態をキャプチャ」ボタンをクリックしてシミュレーションを開始してください")
