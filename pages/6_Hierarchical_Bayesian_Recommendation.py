"""
Hierarchical Bayesian Recommendation System - Streamlit UI

Statistically valid recommendation system with 3-layer architecture
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.ml.hierarchical_bayesian_recommender import (
    HierarchicalBayesianRecommender
)
from skillnote_recommendation.utils.ui_components import (
    apply_enterprise_styles,
    render_page_header
)

st.set_page_config(
    page_title="Hierarchical Bayesian Recommendation",
    page_icon="🎯",
    layout="wide"
)

# Apply modern UI styles
apply_enterprise_styles()

# Page header
render_page_header(
    title="Hierarchical Bayesian Recommendation",
    icon="🎯",
    description="階層的ベイジアン推薦システム - 3層アーキテクチャによる統計的に妥当な推薦"
)

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

# データ読み込みチェック
if 'transformed_data' not in st.session_state or st.session_state.transformed_data is None:
    st.error("❌ **データがロードされていません**")
    st.markdown("""
    ### 📋 次の手順でデータを読み込んでください:
    
    1. **左サイドバーのナビゲーション**から「🧭 CareerNavigator」（ホームページ）を選択
    2. **6種類のCSVファイル**をアップロード
    3. **「📥 データを読み込む」ボタン**をクリック
    4. データ読み込み完了後、このページに戻ってください
    """)
    st.stop()

# Streamlit appでインポートしたデータを取得
td = st.session_state.transformed_data
member_competence = td["member_competence"]
competence_master = td["competence_master"]

# categoriesとskillsのデータを取得
categories_df = td.get("categories")
# competence_masterからSKILLのみを抽出してskills_dfとして使用
skills_df = competence_master[competence_master['力量タイプ'] == 'SKILL'].copy()

# セッション状態の初期化
if 'hb_recommender' not in st.session_state:
    st.session_state.hb_recommender = None
if 'hb_trained' not in st.session_state:
    st.session_state.hb_trained = False
if 'hb_recommendations' not in st.session_state:
    st.session_state.hb_recommendations = None
if 'hb_selected_member' not in st.session_state:
    st.session_state.hb_selected_member = None

# データ統計を表示
n_users = member_competence['メンバーコード'].nunique()
skill_data = member_competence[
    member_competence['力量タイプ'] == 'SKILL'
]
n_skills = skill_data['力量コード'].nunique()

# サイドバー: データ統計のみ
with st.sidebar:
    st.header("⚙️ データ統計")

    st.info(f"""
    **データ統計**:
    - ユーザー数: {n_users}
    - スキル数: {n_skills}
    """)

    if categories_df is not None:
        st.success(f"""
        ✅ **カテゴリ情報**:
        - カテゴリ数: {len(categories_df)}個
        """)
    else:
        st.error("""
        ❌ **カテゴリ情報が未読み込み**

        階層的ベイジアン推薦には
        カテゴリーマスタが必要です。

        ホームページでアップロードしてください。
        """)

# メインエリア: モデル学習
st.markdown("---")
st.subheader("🧠 モデル学習")

if st.session_state.hb_trained:
    st.success("✅ モデルは既に学習済みです。")

    # モデル情報を表示
    if st.session_state.hb_recommender.hierarchy:
        hierarchy = st.session_state.hb_recommender.hierarchy

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("L1カテゴリ", f"{len(hierarchy.level1_categories)}個")
        with col2:
            st.metric("L2カテゴリ", f"{len(hierarchy.level2_categories)}個")
        with col3:
            st.metric("L3カテゴリ", f"{len(hierarchy.level3_categories)}個")
        with col4:
            st.metric("総スキル数", f"{len(hierarchy.skill_to_category)}個")

    if st.button("🔄 モデルを再学習する"):
        st.session_state.hb_trained = False
        st.session_state.hb_recommender = None
        st.rerun()
else:
    st.info("📚 階層的ベイジアン推薦システムを学習します。3層アーキテクチャで統計的に妥当な推薦を実現します。")

    with st.expander("⚙️ モデル設定", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            max_indegree = st.number_input(
                "ベイジアンネットワークの最大入次数",
                min_value=1,
                max_value=5,
                value=3,
                help="ベイジアンネットワークで各ノードが持つ親ノードの最大数"
            )

        with col2:
            n_components = st.number_input(
                "行列分解の潜在因子数",
                min_value=5,
                max_value=30,
                value=10,
                help="Layer 3の行列分解で使用する潜在因子の数"
            )

    if st.button("🚀 モデルを学習", type="primary", use_container_width=True):
        with st.spinner("モデルを初期化・学習中... (数分かかる場合があります)"):
            try:
                # カテゴリ情報の確認
                if categories_df is None:
                    st.error("❌ カテゴリ情報が見つかりません。")
                    st.warning("""
                    **解決方法:**

                    1. ホームページ（🧭 CareerNavigator）に戻る
                    2. **「5️⃣ カテゴリー」** セクションで「力量カテゴリーマスタ」CSVファイルをアップロード
                    3. **「📥 データを読み込む」** ボタンをクリック
                    4. このページに戻って再度学習を実行

                    ※ カテゴリーマスタは階層的ベイジアン推薦に必須です
                    """)
                    st.stop()

                # 推薦システムを初期化（DataFrameを直接渡す）
                st.session_state.hb_recommender = HierarchicalBayesianRecommender(
                    member_competence=member_competence,
                    competence_master=competence_master,
                    category_df=categories_df,
                    skill_df=skills_df,
                    max_indegree=int(max_indegree),
                    n_components=int(n_components)
                )

                # モデル学習
                st.session_state.hb_recommender.fit()
                st.session_state.hb_trained = True
                st.success("✅ 学習完了！")

                # UIを更新するためにリラン
                st.rerun()

            except Exception as e:
                st.error(f"❌ 学習エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

# 推薦生成エリア
st.markdown("---")
st.subheader("💡 推薦生成")

if st.session_state.hb_trained:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # メンバー選択
        member_codes = member_competence['メンバーコード'].unique()
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
    
    # 推薦ボタンまたは既に推薦結果がある場合
    generate_recommendations = st.button("🎯 推薦を生成", type="primary", use_container_width=True)

    # メンバーが変わったら推薦結果をクリア
    if st.session_state.hb_selected_member != selected_member:
        st.session_state.hb_recommendations = None
        st.session_state.hb_selected_member = selected_member

    if generate_recommendations:
        with st.spinner(f"{selected_member} への推薦を生成中..."):
            try:
                recommendations = st.session_state.hb_recommender.recommend(
                    member_code=selected_member,
                    top_n=top_n
                )
                st.session_state.hb_recommendations = recommendations

            except Exception as e:
                st.error(f"❌ 推薦生成エラー: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 推薦結果を表示（セッション状態から取得）
    if st.session_state.hb_recommendations is not None:
        recommendations = st.session_state.hb_recommendations

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

            # 階層グラフの可視化
            st.markdown("---")
            st.subheader("🔗 階層グラフ可視化")
            st.caption("推薦スキルのカテゴリ階層とあなたの保有スキルとの関係")

            # 推薦スキルから選択（上位10個まで）
            skill_options = [f"{i+1}. {rec['力量名']} (スコア: {rec['スコア']:.4f})"
                            for i, rec in enumerate(recommendations[:10])]
            selected_skill_idx = st.selectbox(
                "グラフを表示する推薦スキルを選択",
                range(min(10, len(recommendations))),
                format_func=lambda x: skill_options[x],
                help="上位10個の推薦スキルから選択できます。"
            )

            if selected_skill_idx is not None:
                import streamlit.components.v1 as components

                try:
                    selected_rec = recommendations[selected_skill_idx]
                    skill_code = selected_rec['力量コード']

                    # 階層グラフを生成
                    html_path = st.session_state.hb_recommender.generate_hierarchy_graph(
                        skill_code=skill_code,
                        member_code=selected_member,
                        output_path=f"hierarchy_graph_{skill_code}.html",
                        height="600px"
                    )

                    if html_path:
                        # HTMLファイルを読み込んで表示
                        with open(html_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()
                        components.html(source_code, height=620, scrolling=False)

                        # 凡例を表示
                        st.caption(
                            "🔴 **赤**: L1カテゴリ（大カテゴリ） | "
                            "🟠 **橙**: L2カテゴリ（中カテゴリ） | "
                            "🟡 **黄**: L3カテゴリ（小カテゴリ） | "
                            "🔵 **青**: 推薦スキル | "
                            "🟢 **緑**: あなたの保有スキル | "
                            "⚪ **灰**: 関連スキル"
                        )

                        st.info("""
                        **グラフの見方**:
                        - 上から下へ階層構造（L1→L2→L3→スキル）が表示されます
                        - 青いノードが選択した推薦スキルです
                        - 緑のノードはあなたが既に保有しているスキルです
                        - 同じL3カテゴリ内の関連スキルが表示されます（保有スキルは全て、その他は最大10個）
                        - L2カテゴリ配下の他のL3カテゴリとそのスキルも表示されます（最大2カテゴリ）
                        - ノードをドラッグして移動、マウスホイールでズームできます
                        """)
                    else:
                        st.warning("グラフを生成できませんでした。")

                except Exception as e:
                    st.error(f"グラフ描画エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        else:
            st.warning("推薦が生成されませんでした。")

else:
    st.info("""
    💡 まずモデルを学習してください。

    1. 上記の **「🚀 モデルを学習」** ボタンをクリック
    2. 学習完了後、メンバーを選択して推薦を生成できます
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
