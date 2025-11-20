"""
階層的ベイジアン推薦システム - Streamlit UI

3層アーキテクチャによる統計的に妥当な推薦システム
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.ml.hierarchical_bayesian_recommender import (
    HierarchicalBayesianRecommender
)

st.set_page_config(
    page_title="階層的ベイジアン推薦",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 階層的ベイジアン推薦システム")

st.markdown("""
### 3層アーキテクチャによる統計的に妥当な推薦システム

- **Layer 1**: ベイジアンネットワーク（大カテゴリレベル）
- **Layer 2**: 条件付き確率学習（中カテゴリレベル）
- **Layer 3**: カテゴリ別行列分解（スキルレベル）

**特徴**:
- 統計的妥当性の確保（176サンプル vs 10-20カテゴリ）
- 階層的で解釈可能な説明
- スキルレベルの精密な推薦
""")

# セッション状態の初期化
if 'hb_recommender' not in st.session_state:
    st.session_state.hb_recommender = None
if 'hb_trained' not in st.session_state:
    st.session_state.hb_trained = False

# サイドバー: データ読み込みとモデル学習
with st.sidebar:
    st.header("⚙️ モデル設定")
    
    # データ読み込み
    if st.button("📂 データを読み込む", use_container_width=True):
        with st.spinner("データを読み込み中..."):
            try:
                loader = DataLoader()
                
                # データ読み込み
                member_competence = loader.load_member_competence()
                competence_master = loader.load_competence_master()
                
                # カテゴリとスキルのCSVパス
                data_dir = project_root / 'data'
                category_csv = data_dir / 'categories' / 'competence_category_skillnote.csv'
                skill_csv = data_dir / 'skills' / 'skill_skillnote.csv'
                
                # 推薦システムを初期化
                st.session_state.hb_recommender = HierarchicalBayesianRecommender(
                    member_competence=member_competence,
                    competence_master=competence_master,
                    category_csv_path=str(category_csv),
                    skill_csv_path=str(skill_csv),
                    max_indegree=3,
                    n_components=10
                )
                
                st.session_state.member_competence = member_competence
                st.session_state.competence_master = competence_master
                
                st.success("✅ データ読み込み完了！")
                
                # データ統計を表示
                n_users = member_competence['メンバーコード'].nunique()
                skill_data = member_competence[
                    member_competence['力量タイプ  ###[competence_type]###'] == 'SKILL'
                ]
                n_skills = skill_data['力量コード'].nunique()
                
                st.info(f"""
                **データ統計**:
                - ユーザー数: {n_users}
                - スキル数: {n_skills}
                """)
                
            except Exception as e:
                st.error(f"❌ データ読み込みエラー: {e}")
    
    st.divider()
    
    # モデル学習
    if st.session_state.hb_recommender is not None:
        st.subheader("🧠 モデル学習")
        
        if st.button("🚀 モデルを学習", use_container_width=True):
            with st.spinner("モデルを学習中... (数分かかる場合があります)"):
                try:
                    st.session_state.hb_recommender.fit()
                    st.session_state.hb_trained = True
                    st.success("✅ 学習完了！")
                    
                    # モデル情報を表示
                    if st.session_state.hb_recommender.hierarchy:
                        hierarchy = st.session_state.hb_recommender.hierarchy
                        st.info(f"""
                        **カテゴリ階層**:
                        - L1カテゴリ: {len(hierarchy.level1_categories)}個
                        - L2カテゴリ: {len(hierarchy.level2_categories)}個
                        - L3カテゴリ: {len(hierarchy.level3_categories)}個
                        - 総スキル数: {len(hierarchy.skill_to_category)}個
                        """)
                    
                    if st.session_state.hb_recommender.network_learner:
                        network_info = st.session_state.hb_recommender.network_learner.get_network_info()
                        if network_info:
                            st.info(f"""
                            **ベイジアンネットワーク (Layer 1)**:
                            - ノード数: {network_info.get('n_nodes', 'N/A')}
                            - エッジ数: {network_info.get('n_edges', 'N/A')}
                            """)
                    
                except Exception as e:
                    st.error(f"❌ 学習エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# メインエリア: 推薦生成
if st.session_state.hb_trained:
    st.header("💡 推薦生成")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # メンバー選択
        member_codes = st.session_state.member_competence['メンバーコード'].unique()
        selected_member = st.selectbox(
            "メンバーを選択",
            options=member_codes,
            help="推薦を生成するメンバーを選択してください"
        )
    
    with col2:
        # 推薦件数
        top_n = st.slider(
            "推薦件数",
            min_value=5,
            max_value=30,
            value=10,
            help="推薦するスキルの数"
        )
    
    if st.button("🎯 推薦を生成", type="primary", use_container_width=True):
        with st.spinner(f"{selected_member} への推薦を生成中..."):
            try:
                recommendations = st.session_state.hb_recommender.recommend(
                    member_code=selected_member,
                    top_n=top_n
                )
                
                if recommendations:
                    st.success(f"✅ {len(recommendations)}件の推薦を生成しました！")
                    
                    # 推薦結果を表示
                    st.subheader("📊 推薦結果")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"**{i}. {rec['力量名']}** (スコア: {rec['スコア']:.4f})"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown(f"""
                                **基本情報**:
                                - 力量コード: `{rec['力量コード']}`
                                - カテゴリ: {rec['カテゴリ']}
                                """)
                            
                            with col_b:
                                st.markdown(f"""
                                **推薦スコア**:
                                - 総合スコア: {rec['スコア']:.4f}
                                """)
                            
                            # 階層的説明
                            st.markdown("**📝 階層的説明**:")
                            st.info(rec['説明'])
                    
                    # 推薦結果をDataFrameで表示
                    st.subheader("📋 推薦一覧")
                    df_recommendations = pd.DataFrame(recommendations)
                    st.dataframe(
                        df_recommendations[['力量名', 'スコア', '説明', 'カテゴリ']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # CSVダウンロード
                    csv = df_recommendations.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 推薦結果をCSVでダウンロード",
                        data=csv,
                        file_name=f"hierarchical_bayesian_recommendations_{selected_member}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    st.warning("推薦が生成されませんでした。")
                    
            except Exception as e:
                st.error(f"❌ 推薦生成エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("""
    👈 サイドバーから以下の手順で開始してください：
    
    1. **データを読み込む** ボタンをクリック
    2. **モデルを学習** ボタンをクリック
    3. メンバーを選択して推薦を生成
    """)

# フッター
st.divider()
st.markdown("""
### 📚 階層的ベイジアン推薦システムについて

**3層アーキテクチャ**:
- **Layer 1 (ベイジアンネットワーク)**: 大カテゴリ間の依存関係を学習し、統計的妥当性を確保
- **Layer 2 (条件付き確率)**: P(中カテゴリ | 大カテゴリ)の関係を学習
- **Layer 3 (カテゴリ別MF)**: 各中カテゴリ内でスキルレベルの推薦を生成

**スコア統合式**:
```
最終スコア = (L1_準備度^0.3) × (L2_確率^0.3) × (L3_スキルスコア^0.4)
```

**特徴**:
- 統計的妥当性: 176サンプル vs 10-20カテゴリ（適切な比率）
- 解釈可能性: 階層的な説明文を生成
- 精密な推薦: スキルレベルでの推薦スコア
""")
