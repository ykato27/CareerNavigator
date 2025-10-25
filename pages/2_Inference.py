"""
キャリア推薦システム - 推論ページ

このページでは、学習済みMLモデルを使用して、メンバーへの力量推薦を実行し、
推薦結果の詳細と参考人物の可視化を提供します。

主な機能:
- メンバー選択と推論設定
- 力量推薦の実行
- 推薦理由と参考人物の表示
- メンバーポジショニングマップの可視化
- 推薦結果のCSVダウンロード
"""

from io import StringIO
from typing import List

import streamlit as st
import pandas as pd

from skillnote_recommendation.utils.streamlit_helpers import (
    check_data_loaded,
    check_model_trained,
    display_error_details,
)
from skillnote_recommendation.utils.visualization import (
    create_member_positioning_data,
    create_positioning_plot,
    prepare_positioning_display_dataframe,
)
from skillnote_recommendation.core.models import Recommendation


# =========================================================
# ヘルパー関数
# =========================================================

def convert_hybrid_to_recommendation(hybrid_rec) -> Recommendation:
    """
    HybridRecommendationを標準のRecommendationオブジェクトに変換

    Args:
        hybrid_rec: HybridRecommendationオブジェクト

    Returns:
        Recommendationオブジェクト
    """
    return Recommendation(
        competence_code=hybrid_rec.competence_code,
        competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code),
        competence_type=hybrid_rec.competence_info.get('力量タイプ', 'UNKNOWN'),
        category=hybrid_rec.competence_info.get('カテゴリー', ''),
        priority_score=hybrid_rec.score,
        category_importance=0.5,  # デフォルト値
        acquisition_ease=0.5,  # デフォルト値
        popularity=0.5,  # デフォルト値
        reason=', '.join(hybrid_rec.reasons) if hybrid_rec.reasons else 'グラフベース推薦',
        reference_persons=[]
    )


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="キャリア推薦システム - 推論",
    page_icon="🎯",
    layout="wide"
)

# カスタムCSSでリッチなUIを実現
st.markdown("""
<style>
    /* グローバルスタイル */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* サイドバースタイル */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* サイドバーテキストの色 */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* サイドバーの入力要素 */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] .stMultiSelect > div > div:hover {
        border-color: rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* スライダー */
    [data-testid="stSidebar"] .stSlider {
        padding: 1rem 0;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }

    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: white;
        border: 2px solid rgba(255, 255, 255, 0.5);
    }

    /* チェックボックス */
    [data-testid="stSidebar"] .stCheckbox {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.9rem;
    }

    /* サイドバー内のinfo/warning/error */
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    /* サイドバーセクション区切り */
    [data-testid="stSidebar"] hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
        margin: 1.5rem 0;
    }

    /* サイドバー内のボタン */
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: #667eea;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* サイドバーキャプション */
    [data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem;
    }

    /* カードスタイル */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }

    /* グラデーションヘッダー */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* メトリクスカード */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .metric-card-green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .metric-card-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }

    .metric-card-purple {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }

    /* バッジ */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }

    .badge-success {
        background: #28a745;
        color: white;
    }

    .badge-info {
        background: #17a2b8;
        color: white;
    }

    .badge-warning {
        background: #ffc107;
        color: black;
    }

    .badge-danger {
        background: #dc3545;
        color: white;
    }

    /* アニメーション */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }

    /* タイトル装飾 */
    .title-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
        vertical-align: middle;
    }

    /* プログレスバー */
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.6s ease;
    }

    /* ボタンホバー効果 */
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* タブスタイル */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }

    /* セクション区切り */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# リッチなヘッダー
st.markdown("""
<div class="gradient-header fade-in">
    <h1><span class="title-icon">🎯</span>推論実行</h1>
    <p style="font-size: 1.1rem; margin: 0;">学習済みMLモデルを使用して、メンバーへの力量推薦を実行します</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 前提条件チェック
# =========================================================

check_data_loaded()
check_model_trained()


# =========================================================
# データ準備
# =========================================================

td = st.session_state.transformed_data
members_df = td["members_clean"]
recommender = st.session_state.ml_recommender
mf_model = recommender.mf_model


# =========================================================
# ヘルパー関数
# =========================================================

def convert_recommendations_to_dataframe(recommendations) -> pd.DataFrame:
    """
    Recommendationオブジェクトのリストを表示用/ダウンロード用のDataFrameに変換する。

    Args:
        recommendations: Recommendationオブジェクトのリスト

    Returns:
        推薦結果のDataFrame（順位列付き）
    """
    if not recommendations:
        return pd.DataFrame()

    rows = []
    for rank, rec in enumerate(recommendations, start=1):
        rec_dict = rec.to_dict()
        rec_dict["順位"] = rank
        rows.append(rec_dict)

    # 順位を先頭列にする
    df = pd.DataFrame(rows)
    cols = ["順位"] + [c for c in df.columns if c != "順位"]
    return df[cols]


def get_reference_person_codes(recommendations) -> List[str]:
    """
    推薦結果から参考人物のコードリストを抽出する。

    Args:
        recommendations: Recommendationオブジェクトのリスト

    Returns:
        ユニークな参考人物コードのリスト
    """
    reference_codes = []
    for rec in recommendations:
        if rec.reference_persons:
            for ref_person in rec.reference_persons:
                if ref_person.member_code not in reference_codes:
                    reference_codes.append(ref_person.member_code)
    return reference_codes


def display_reference_person(ref_person):
    """
    参考人物の情報を表示する。

    Args:
        ref_person: ReferencePersonオブジェクト
    """
    # 参考タイプのアイコンとラベル
    if ref_person.reference_type == "similar_career":
        st.markdown("#### 🤝 類似キャリア")
    elif ref_person.reference_type == "role_model":
        st.markdown("#### ⭐ ロールモデル")
    else:
        st.markdown("#### 🌟 異なるキャリアパス")

    st.markdown(f"**{ref_person.member_name}さん**")
    st.markdown(ref_person.reason)

    # 差分分析を表示
    st.markdown("**📊 力量の比較**")
    st.metric("共通力量", f"{len(ref_person.common_competences)}個")
    st.metric("参考力量", f"{len(ref_person.unique_competences)}個")
    st.metric("類似度", f"{int(ref_person.similarity_score * 100)}%")


def display_recommendation_details(rec, idx: int):
    """
    推薦結果の詳細を展開可能なセクションで表示する。

    Args:
        rec: Recommendationオブジェクト
        idx: 推薦順位
    """
    with st.expander(
        f"🎯 推薦 {idx}: {rec.competence_name} (優先度: {rec.priority_score:.1f})"
    ):
        # 推薦理由
        st.markdown("### 📋 推薦理由")
        st.markdown(rec.reason)

        # 参考人物
        if rec.reference_persons:
            st.markdown("---")
            st.markdown("### 👥 参考になる人物")

            cols = st.columns(len(rec.reference_persons))
            for col_idx, ref_person in enumerate(rec.reference_persons):
                with cols[col_idx]:
                    display_reference_person(ref_person)


def display_positioning_maps(
    position_df: pd.DataFrame,
    target_code: str,
    reference_codes: List[str]
):
    """
    メンバーポジショニングマップを複数のタブで表示する。

    Args:
        position_df: メンバー位置データ
        target_code: 対象メンバーコード
        reference_codes: 参考人物コードのリスト
    """
    # リッチなセクション区切り
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # カードベースのヘッダー
    st.markdown("""
    <div class="card fade-in">
        <h2>🗺️ メンバーポジショニングマップ</h2>
        <p>あなたと参考人物が、全メンバーの中でどの位置にいるかを可視化します</p>
        <div>
            <span class="badge badge-danger">あなた</span>
            <span class="badge badge-info">参考人物</span>
            <span class="badge">その他のメンバー</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # タブを作成
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 スキルレベル vs 保有力量数",
        "📈 平均レベル vs 保有力量数",
        "🔮 潜在因子マップ",
        "📋 データテーブル"
    ])

    with tab1:
        st.markdown("### 総合スキルレベル vs 保有力量数")
        st.markdown(
            "**X軸**: 総合スキルレベル（全保有力量の正規化レベルの合計）\n\n"
            "**Y軸**: 保有力量数\n\n"
            "右上に行くほど、多くの力量を高いレベルで保有していることを示します。"
        )
        fig1 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "総合スキルレベル", "保有力量数",
            "総合スキルレベル vs 保有力量数"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.markdown("### 平均レベル vs 保有力量数")
        st.markdown(
            "**X軸**: 保有力量数（スキルの幅）\n\n"
            "**Y軸**: 平均レベル（スキルの深さ）\n\n"
            "右上に行くほど、幅広い力量を深く習得していることを示します。"
        )
        fig2 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "保有力量数", "平均レベル",
            "スキルの幅 vs 深さ"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 潜在因子マップ（NMF空間）")
        st.markdown(
            "**X軸**: 潜在因子1（第1スキルパターン）\n\n"
            "**Y軸**: 潜在因子2（第2スキルパターン）\n\n"
            "NMFモデルが学習したスキルパターンの空間で、メンバーを配置します。\n"
            "近くにいる人は似たスキルパターンを持っています。"
        )
        fig3 = create_positioning_plot(
            position_df, target_code, reference_codes,
            "潜在因子1", "潜在因子2",
            "潜在因子空間でのメンバー分布"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.markdown("### 全メンバーのデータ")
        display_df = prepare_positioning_display_dataframe(
            position_df, target_code, reference_codes
        )
        st.dataframe(display_df, use_container_width=True, height=400)


# =========================================================
# メンバー選択UI
# =========================================================

st.subheader("👤 推論対象メンバーの選択")

# 学習データに存在するメンバーのみをフィルタ（コールドスタート問題を回避）
trained_member_codes = set(mf_model.member_codes)
available_members = members_df[
    members_df["メンバーコード"].isin(trained_member_codes)
]

if len(available_members) == 0:
    st.error("❌ 推論可能なメンバーが存在しません。")
    st.stop()

st.info(
    f"📊 推論可能なメンバー数: {len(available_members)} / {len(members_df)} 名\n\n"
    f"💡 **コールドスタート問題の回避**: 学習データに含まれるメンバーのみが選択可能です。\n"
    f"保有力量が未登録のメンバーは、データ登録後にモデルを再学習してください。"
)

# メンバー選択プルダウン
member_options = dict(
    zip(available_members["メンバーコード"], available_members["メンバー名"])
)

selected_member_code = st.selectbox(
    "推論対象メンバーを選択してください",
    options=list(member_options.keys()),
    format_func=lambda x: f"{member_options[x]} ({x})"
)


# =========================================================
# 推論設定UI
# =========================================================

st.subheader("⚙️ 推論設定")

col1, col2, col3 = st.columns(3)

with col1:
    top_n = st.slider(
        "推薦数",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

with col2:
    st.markdown("**力量タイプフィルタ**")
    selected_types = st.multiselect(
        "推薦する力量タイプを選択してください",
        options=["SKILL", "EDUCATION", "LICENSE"],
        default=["SKILL", "EDUCATION", "LICENSE"],
        help="複数選択可能。例: スキルのみ、スキルと教育、など"
    )

    # 空リストの場合はNoneに変換（全てを推薦）
    competence_type = selected_types if selected_types else None

    # 選択が空の場合の警告
    if not selected_types:
        st.warning("⚠️ 力量タイプが選択されていません。全てのタイプから推薦します。")
    else:
        # 選択されたタイプを確認表示
        st.caption(f"選択中: {', '.join(selected_types)}")

with col3:
    diversity_strategy = st.selectbox(
        "多様性戦略",
        options=["hybrid", "mmr", "category", "type"],
        index=0,
        help="推薦結果の多様性を確保する戦略を選択"
    )

# =========================================================
# 推薦手法選択
# =========================================================

st.markdown("---")
st.subheader("🎯 推薦手法の選択")

recommendation_method = st.radio(
    "使用する推薦手法を選択してください",
    options=["NMF推薦", "グラフベース推薦", "ハイブリッド推薦"],
    index=0,
    help="推薦手法を選択します。NMFは高速、グラフベースは説明可能性が高い、ハイブリッドは両方の良いところを組み合わせます。",
    horizontal=True
)

# 選択された手法の説明を表示
if recommendation_method == "NMF推薦":
    st.info("📊 **NMF推薦（機械学習ベース）**: 協調フィルタリングに基づく高速な推薦。メンバー間の類似性から推薦を生成します。")
elif recommendation_method == "グラフベース推薦":
    st.info("🔗 **グラフベース推薦（RWR）**: 知識グラフ構造を活用した推薦。推薦パスを可視化でき、説明可能性が高いです。")
else:
    st.info("🎯 **ハイブリッド推薦（NMF + Graph）**: NMFとグラフベースの両方の強みを組み合わせた推薦。")

# グラフベースまたはハイブリッドの場合、追加設定を表示
if recommendation_method in ["グラフベース推薦", "ハイブリッド推薦"]:
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if recommendation_method == "ハイブリッド推薦":
            rwr_weight = st.slider(
                "グラフベーススコアの重み",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0.5 = グラフとNMFを同等に評価、1.0 = グラフのみ、0.0 = NMFのみ"
            )
        else:
            rwr_weight = 1.0  # グラフベース推薦の場合は常に1.0

    with col_g2:
        show_paths = st.checkbox(
            "推薦パスを表示",
            value=True,
            help="推薦理由をグラフで可視化します"
        )
else:
    rwr_weight = 0.5  # デフォルト値
    show_paths = False


# =========================================================
# 推論実行
# =========================================================

st.subheader("🚀 推論実行")

if st.button("推薦を実行", type="primary"):
    with st.spinner(f"{recommendation_method}を生成中..."):
        try:
            import time
            from skillnote_recommendation.graph import HybridGraphRecommender

            # 実行時間を計測
            start_time = time.time()

            # 選択された推薦手法のみを実行
            if recommendation_method == "NMF推薦":
                # NMF推薦のみ
                recs = recommender.recommend(
                    member_code=selected_member_code,
                    top_n=top_n,
                    competence_type=competence_type,
                    category_filter=None,
                    use_diversity=True,
                    diversity_strategy=diversity_strategy
                )
                # グラフ情報はなし
                graph_recommendations = None

            elif recommendation_method in ["グラフベース推薦", "ハイブリッド推薦"]:
                # Knowledge Graphの確認
                if 'knowledge_graph' not in st.session_state:
                    st.error("❌ Knowledge Graphが初期化されていません。データ読み込みページで再度データを読み込んでください。")
                    st.stop()

                # HybridGraphRecommenderを初期化
                hybrid_recommender = HybridGraphRecommender(
                    knowledge_graph=st.session_state.knowledge_graph,
                    ml_recommender=recommender,
                    rwr_weight=rwr_weight
                )

                # グラフベースまたはハイブリッド推薦を実行
                graph_recommendations = hybrid_recommender.recommend(
                    member_code=selected_member_code,
                    top_n=top_n,
                    competence_type=competence_type,
                    category_filter=None,
                    use_diversity=True
                )

                # HybridRecommendationを標準のRecommendationに変換
                recs = [convert_hybrid_to_recommendation(hr) for hr in graph_recommendations]

            # 実行時間を計測
            elapsed_time = time.time() - start_time

            # セッション状態に保存
            st.session_state.last_recommendations = recs
            st.session_state.last_target_member_code = selected_member_code
            st.session_state.last_execution_time = elapsed_time
            st.session_state.last_recommendation_method = recommendation_method
            if graph_recommendations:
                st.session_state.graph_recommendations = graph_recommendations

            # セッション状態に保存
            if not recs:
                st.warning("⚠️ 推薦できる力量がありません。")

                # 診断情報を表示
                st.info("### 💡 推薦が空になった理由:")

                # 選択された力量タイプを表示
                if competence_type:
                    type_str = "、".join(competence_type) if isinstance(competence_type, list) else competence_type
                    st.write(f"**選択された力量タイプ**: {type_str}")
                else:
                    st.write("**選択された力量タイプ**: 全て")

                # 保有力量の情報を表示
                member_comp = td["member_competence"][
                    td["member_competence"]["メンバーコード"] == selected_member_code
                ]
                acquired_count = len(member_comp)
                st.write(f"**既習得力量数**: {acquired_count}個")

                # タイプ別の保有力量数を表示
                if len(member_comp) > 0:
                    comp_master = td["competence_master"]
                    acquired_codes = member_comp["力量コード"].unique()
                    acquired_info = comp_master[comp_master["力量コード"].isin(acquired_codes)]

                    type_counts = acquired_info["力量タイプ"].value_counts().to_dict()
                    st.write("**タイプ別保有力量数**:")
                    for comp_type, count in type_counts.items():
                        st.write(f"  - {comp_type}: {count}個")

                # 改善案を提示
                st.markdown("### 🔧 改善案:")
                suggestions = []

                if competence_type and len(competence_type) < 3:
                    suggestions.append("- **力量タイプを追加**: 他の力量タイプも選択してみてください")

                if acquired_count > 50:
                    suggestions.append("- **すでに多くの力量を習得**: 新しい分野への挑戦も検討してみてください")

                suggestions.append("- **推薦数を増やす**: スライダーで推薦数を増やしてみてください")
                suggestions.append("- **多様性戦略を変更**: 異なる多様性戦略を試してみてください")

                for suggestion in suggestions:
                    st.write(suggestion)

                st.session_state.last_recommendations_df = None
                st.session_state.last_recommendations = None
                st.session_state.last_target_member_code = None
            else:
                # ハイブリッド推薦をメインとして保存
                df_result = convert_recommendations_to_dataframe(recs)
                st.session_state.last_recommendations_df = df_result
                st.session_state.last_recommendations = recs

                # リッチな成功メッセージ（実行時間を表示）
                st.markdown(f"""
                <div class="card metric-card-green fade-in" style="text-align: left;">
                    <h2 style="margin: 0;">🎉 推薦完了！</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">{recommendation_method}で{len(recs)}件の力量を推薦しました</p>
                    <p style="font-size: 0.9rem; margin: 0; opacity: 0.9;">⚡ 実行時間: {elapsed_time:.2f}秒</p>
                </div>
                """, unsafe_allow_html=True)

                # 推薦結果の表示
                st.markdown("---")

                # NMF推薦の場合
                if recommendation_method == "NMF推薦":
                    # 推薦結果の詳細表示
                    for idx, rec in enumerate(recs, 1):
                        display_recommendation_details(rec, idx)

                # グラフベースまたはハイブリッド推薦の場合
                elif recommendation_method in ["グラフベース推薦", "ハイブリッド推薦"]:
                    graph_recs_display = st.session_state.get('graph_recommendations', [])

                    if graph_recs_display:
                        # 推薦結果の詳細表示
                        for idx, hybrid_rec in enumerate(graph_recs_display, 1):
                            rec = convert_hybrid_to_recommendation(hybrid_rec)

                            # スコア表示のタイトルを決定
                            if recommendation_method == "グラフベース推薦":
                                title = f"🎯 推薦 {idx}: {rec.competence_name} (グラフスコア: {hybrid_rec.rwr_score:.3f})"
                            else:
                                title = f"🎯 推薦 {idx}: {rec.competence_name} (総合スコア: {hybrid_rec.score:.3f})"

                            with st.expander(title):
                                # スコア情報を表示
                                if recommendation_method == "グラフベース推薦":
                                    col_s1, col_s2 = st.columns(2)
                                    with col_s1:
                                        st.metric("グラフスコア（RWR）", f"{hybrid_rec.rwr_score:.3f}")
                                    with col_s2:
                                        st.metric("NMFスコア（参考）", f"{hybrid_rec.nmf_score:.3f}")
                                else:  # ハイブリッド推薦
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    with col_s1:
                                        st.metric("総合スコア", f"{hybrid_rec.score:.3f}")
                                    with col_s2:
                                        st.metric("グラフスコア", f"{hybrid_rec.rwr_score:.3f}")
                                    with col_s3:
                                        st.metric("NMFスコア", f"{hybrid_rec.nmf_score:.3f}")

                                # 推薦理由
                                st.markdown("### 📋 推薦理由")
                                st.markdown(rec.reason)

                                # パス可視化
                                if show_paths and hybrid_rec.paths:
                                    st.markdown("---")
                                    st.markdown("### 🔗 推薦パスの可視化")

                                    from skillnote_recommendation.graph import RecommendationPathVisualizer
                                    from skillnote_recommendation.graph.visualization_utils import (
                                        ExplanationGenerator,
                                        format_explanation_for_display,
                                        export_figure_as_html
                                    )

                                    visualizer = RecommendationPathVisualizer()
                                    category_hierarchy = st.session_state.knowledge_graph.category_hierarchy if st.session_state.get('knowledge_graph') else None
                                    explainer = ExplanationGenerator(category_hierarchy=category_hierarchy)

                                    # 詳細説明を生成
                                    explanation = explainer.generate_detailed_explanation(
                                        paths=hybrid_rec.paths,
                                        rwr_score=hybrid_rec.rwr_score,
                                        nmf_score=hybrid_rec.nmf_score,
                                        competence_info=hybrid_rec.competence_info
                                    )

                                    # グラフ可視化と詳細説明をタブで表示
                                    tab1, tab2 = st.tabs(["📊 グラフ可視化", "📝 詳細説明"])

                                    with tab1:
                                        member_name = members_df[
                                            members_df["メンバーコード"] == selected_member_code
                                        ]["メンバー名"].iloc[0]

                                        fig = visualizer.visualize_recommendation_path(
                                            paths=hybrid_rec.paths,
                                            target_member_name=member_name,
                                            target_competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                        # エクスポートボタン
                                        if st.button(f"📥 HTMLとしてエクスポート", key=f"export_{idx}"):
                                            try:
                                                filename = f"recommendation_path_{hybrid_rec.competence_code}.html"
                                                filepath = export_figure_as_html(fig, filename)
                                                st.success(f"✅ エクスポート完了: {filepath}")
                                            except Exception as e:
                                                st.error(f"エクスポートエラー: {str(e)}")

                                    with tab2:
                                        formatted_explanation = format_explanation_for_display(explanation)
                                        st.markdown(formatted_explanation)

                # テーブル表示
                st.markdown("---")
                st.markdown("### 📊 推薦結果一覧")
                st.dataframe(df_result, use_container_width=True)

        except Exception as e:
            # エラー処理
            from skillnote_recommendation.ml.exceptions import (
                ColdStartError,
                MLModelNotTrainedError
            )

            if isinstance(e, ColdStartError):
                st.error("❌ コールドスタート問題が発生しました")
                st.warning(
                    f"**メンバーコード `{e.member_code}` の保有力量が登録されていないため、"
                    f"ML推薦ができません。**\n\n"
                    f"**原因:**\n"
                    f"- このメンバーの力量データがMLモデルの学習データに含まれていません。\n\n"
                    f"**対処方法:**\n"
                    f"1. このメンバーの力量データ（保有力量）を登録してください\n"
                    f"2. データ登録後、「モデル学習」ページで再学習してください\n"
                    f"3. 再学習後、再度推薦を実行してください"
                )
            elif isinstance(e, MLModelNotTrainedError):
                st.error("❌ MLモデルが学習されていません")
                st.info(
                    "「モデル学習」ページでMLモデルを学習してから、"
                    "推薦を実行してください。"
                )
            else:
                display_error_details(e, "推薦処理中")


# =========================================================
# 推薦結果のダウンロード & 可視化
# =========================================================

if st.session_state.get("last_recommendations_df") is not None:
    # セクション区切り
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # CSVダウンロード（カードスタイル）
    st.markdown("""
    <div class="card fade-in">
        <h2>💾 推薦結果のダウンロード</h2>
        <p>推薦結果をCSV形式でダウンロードして、さらなる分析や共有に活用できます</p>
    </div>
    """, unsafe_allow_html=True)

    csv_buffer = StringIO()
    st.session_state.last_recommendations_df.to_csv(
        csv_buffer,
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 推薦結果をCSVでダウンロード",
        data=csv_buffer.getvalue(),
        file_name="recommendations.csv",
        mime="text/csv"
    )

    # メンバーポジショニングマップ
    if st.session_state.get("last_recommendations") is not None:
        # 参考人物のコードを収集
        reference_codes = get_reference_person_codes(
            st.session_state.last_recommendations
        )

        # ポジショニングデータを作成
        position_df = create_member_positioning_data(
            td["member_competence"],
            td["members_clean"],
            mf_model
        )

        # 可視化を表示
        display_positioning_maps(
            position_df,
            st.session_state.last_target_member_code,
            reference_codes
        )

        # キャリアパス推薦
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card fade-in">
            <h2>🎯 キャリアパス推薦</h2>
            <p>目標とするメンバーを選択して、そのメンバーに近づくための学習パスを確認できます</p>
        </div>
        """, unsafe_allow_html=True)

        # 目標メンバー選択
        members_df = td["members_clean"]
        target_member_options = members_df["メンバー名"].tolist()

        # 現在のメンバーを除外
        current_member_name = members_df[
            members_df["メンバーコード"] == st.session_state.last_target_member_code
        ]["メンバー名"].iloc[0] if len(members_df[
            members_df["メンバーコード"] == st.session_state.last_target_member_code
        ]) > 0 else None

        if current_member_name in target_member_options:
            target_member_options.remove(current_member_name)

        col1, col2 = st.columns([3, 1])
        with col1:
            target_member_name = st.selectbox(
                "目標メンバーを選択",
                options=target_member_options,
                key="career_path_target_member"
            )

        with col2:
            analyze_button = st.button(
                "📊 分析実行",
                type="primary",
                key="analyze_career_path"
            )

        if analyze_button and target_member_name:
            with st.spinner("キャリアパスを分析中..."):
                try:
                    from skillnote_recommendation.graph import (
                        CareerGapAnalyzer,
                        LearningPathGenerator,
                        CareerPathVisualizer,
                        format_career_path_summary
                    )

                    # 目標メンバーコードを取得
                    target_member_code = members_df[
                        members_df["メンバー名"] == target_member_name
                    ]["メンバーコード"].iloc[0]

                    # ギャップ分析
                    gap_analyzer = CareerGapAnalyzer(
                        knowledge_graph=st.session_state.knowledge_graph,
                        member_competence_df=td["member_competence"],
                        competence_master_df=td["competence_master"]
                    )

                    gap_analysis = gap_analyzer.analyze_gap(
                        source_member_code=st.session_state.last_target_member_code,
                        target_member_code=target_member_code
                    )

                    # 学習パス生成
                    path_generator = LearningPathGenerator(
                        knowledge_graph=st.session_state.knowledge_graph,
                        category_hierarchy=st.session_state.knowledge_graph.category_hierarchy
                    )

                    career_path = path_generator.generate_learning_path(
                        gap_analysis=gap_analysis,
                        max_per_phase=5
                    )

                    # 可視化
                    visualizer = CareerPathVisualizer()

                    # タブで表示
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 サマリー",
                        "📅 ロードマップ",
                        "🎯 到達度",
                        "📈 カテゴリー分析"
                    ])

                    with tab1:
                        # サマリーを表示
                        summary = format_career_path_summary(career_path, target_member_name)
                        st.markdown(summary)

                    with tab2:
                        # ロードマップを表示
                        roadmap_fig = visualizer.create_roadmap(career_path, target_member_name)
                        st.plotly_chart(roadmap_fig, use_container_width=True)

                    with tab3:
                        # 到達度ゲージを表示
                        gauge_fig = visualizer.create_progress_gauge(career_path.estimated_completion_rate)
                        st.plotly_chart(gauge_fig, use_container_width=True)

                        # 詳細情報（リッチなメトリクスカード）
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-card metric-card-green fade-in">
                                <h3 style="margin: 0;">✅ 共通力量</h3>
                                <h1 style="margin: 0.5rem 0;">{len(career_path.common_competences)}<span style="font-size: 1.5rem;">個</span></h1>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f"""
                            <div class="metric-card metric-card-orange fade-in">
                                <h3 style="margin: 0;">📚 不足力量</h3>
                                <h1 style="margin: 0.5rem 0;">{len(career_path.missing_competences)}<span style="font-size: 1.5rem;">個</span></h1>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_c:
                            st.markdown(f"""
                            <div class="metric-card metric-card-blue fade-in">
                                <h3 style="margin: 0;">📊 ギャップスコア</h3>
                                <h1 style="margin: 0.5rem 0;">{career_path.gap_score:.2f}</h1>
                            </div>
                            """, unsafe_allow_html=True)

                    with tab4:
                        # カテゴリー別分析を表示
                        category_fig = visualizer.create_category_breakdown(career_path)
                        st.plotly_chart(category_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ キャリアパス分析エラー: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
