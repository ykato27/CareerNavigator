"""
因果推薦システムの診断ページ

LiNGAMの学習状況と推薦ロジックをデバッグするためのページ
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender

st.set_page_config(page_title="推薦システム診断", page_icon="🔍", layout="wide")

st.title("🔍 因果推薦システム診断")
st.caption("LiNGAMの学習状況と推薦ロジックをデバッグ")

# データ読み込み
@st.cache_data
def load_all_data():
    """全データを読み込む"""
    import os
    
    # Streamlit Cloudではカレントディレクトリがプロジェクトルート
    data_dir = Path("data")
    
    # ローカル環境の場合
    if not data_dir.exists():
        data_dir = project_root / "data"
    
    return {
        "member_competence": pd.read_csv(data_dir / "member_competence.csv"),
        "competence": pd.read_csv(data_dir / "competence.csv"),
    }

try:
    td = load_all_data()
except Exception as e:
    st.error(f"データ読み込みエラー: {e}")
    st.info("他のページ（Causal Recommendation）でデータが正常に読み込まれているか確認してください。")
    st.stop()

# 推薦モデルの構築
@st.cache_resource
def build_recommender():
    recommender = CausalGraphRecommender(
        member_competence=td["member_competence"],
        competence_master=td["competence"]
    )
    recommender.fit(min_members_per_skill=5)
    return recommender

with st.spinner("推薦モデルを構築中..."):
    recommender = build_recommender()

st.success("✅ モデル構築完了")

# 基本統計
st.header("📊 基本統計")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("スキル数", len(recommender.skill_matrix_.columns))
with col2:
    st.metric("メンバー数", len(recommender.skill_matrix_.index))
with col3:
    # 因果関係の数を計算
    adj_matrix = recommender.learner.get_adjacency_matrix()
    causal_edges = (adj_matrix.abs() > 0.01).sum().sum()
    st.metric("因果関係数（>0.01）", int(causal_edges))

# 因果効果の分布
st.header("📈 因果効果の分布")
adj_matrix = recommender.learner.get_adjacency_matrix()
effects = adj_matrix.values.flatten()
effects_nonzero = effects[effects != 0]

col1, col2 = st.columns(2)
with col1:
    st.metric("非ゼロ因果効果数", len(effects_nonzero))
    st.metric("平均因果効果（絶対値）", f"{np.abs(effects_nonzero).mean():.4f}")
with col2:
    st.metric("最大因果効果", f"{effects_nonzero.max():.4f}")
    st.metric("最小因果効果", f"{effects_nonzero.min():.4f}")

# ヒストグラム
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(np.abs(effects_nonzero), bins=50, edgecolor='black')
ax.axvline(x=0.01, color='red', linestyle='--', label='閾値 0.01')
ax.set_xlabel('因果効果（絶対値）')
ax.set_ylabel('頻度')
ax.set_title('因果効果の分布')
ax.legend()
st.pyplot(fig)

# メンバー別診断
st.header("👤 メンバー別診断")

member_list = td["member_competence"]["メンバーコード"].unique().tolist()
selected_member = st.selectbox("メンバーを選択", member_list)

if selected_member:
    # 保有スキル
    member_skills_codes = td["member_competence"][
        td["member_competence"]["メンバーコード"] == selected_member
    ]["力量コード"].tolist()
    
    code_to_name = recommender.code_to_name
    member_skill_names = [code_to_name.get(c, c) for c in member_skills_codes]
    
    st.subheader(f"保有スキル（{len(member_skill_names)}個）")
    st.write(", ".join(member_skill_names[:10]) + ("..." if len(member_skill_names) > 10 else ""))
    
    # 推薦結果
    recommendations = recommender.recommend(selected_member, top_n=10)
    
    st.subheader("推薦結果の詳細分析")
    
    if not recommendations:
        st.warning("推薦結果がありません")
    else:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['competence_name']} (スコア: {rec['score']:.3f})"):
                details = rec['details']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総合スコア", f"{rec['score']:.3f}")
                with col2:
                    st.metric("準備度", f"{details['readiness_score']:.3f}")
                with col3:
                    st.metric("将来性", f"{details['utility_score']:.3f}")
                
                # Readiness詳細
                st.markdown("**準備度の内訳**")
                if details['readiness_reasons']:
                    readiness_df = pd.DataFrame(
                        details['readiness_reasons'],
                        columns=['保有スキル', '因果効果']
                    )
                    st.dataframe(readiness_df, use_container_width=True)
                else:
                    st.info("保有スキルからの因果効果なし（すべて < 0.01）")
                
                # Utility詳細
                st.markdown("**将来性の内訳**")
                if details['utility_reasons']:
                    utility_df = pd.DataFrame(
                        details['utility_reasons'][:10],
                        columns=['将来スキル', '因果効果']
                    )
                    st.dataframe(utility_df, use_container_width=True)
                else:
                    st.info("将来スキルへの因果効果なし（すべて < 0.01）")

# 因果関係の詳細
st.header("🔗 因果関係の詳細")

skill_names = list(recommender.skill_matrix_.columns)
col1, col2 = st.columns(2)

with col1:
    cause_skill = st.selectbox("原因スキル", skill_names, key="cause")
with col2:
    effect_skill = st.selectbox("結果スキル", skill_names, key="effect")

if cause_skill and effect_skill:
    effect_value = recommender._get_effect(cause_skill, effect_skill)
    
    if abs(effect_value) > 0.001:
        st.success(f"**{cause_skill}** → **{effect_skill}**: {effect_value:.4f}")
    else:
        st.info(f"因果効果なし（{effect_value:.6f}）")

# 推奨事項
st.header("💡 診断結果と推奨事項")

avg_effect = np.abs(effects_nonzero).mean()
threshold_percentile = np.percentile(np.abs(effects_nonzero), 90)

st.markdown(f"""
### 現在の状況

- **平均因果効果**: {avg_effect:.4f}
- **90パーセンタイル**: {threshold_percentile:.4f}
- **現在の閾値**: 0.01

### 推奨事項

""")

if avg_effect < 0.01:
    st.warning(f"""
    ⚠️ **問題**: 平均因果効果（{avg_effect:.4f}）が閾値（0.01）より小さいです。
    
    **原因**: 
    - データのスケールが小さい
    - 因果関係が弱い
    - LiNGAMのパラメータが適切でない
    
    **対策**:
    1. 閾値を{avg_effect/2:.4f}程度に下げる
    2. データの正規化方法を見直す
    3. LiNGAMのパラメータを調整する
    """)
else:
    st.success("✅ 因果効果の強度は適切です")

if causal_edges < len(skill_names) * 2:
    st.warning(f"""
    ⚠️ **問題**: 因果関係数（{int(causal_edges)}）が少なすぎます。
    
    **対策**:
    - 閾値を下げる
    - データ量を増やす
    """)
