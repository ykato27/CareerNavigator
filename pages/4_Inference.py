"""
CareerNavigator - AI推薦実行

このページでは、学習済みAIモデルを使用して、メンバーへの力量推薦を実行し、
推薦結果の詳細と参考人物の可視化を提供します。

主な機能:
- メンバー選択と推論設定
- AI推薦の実行（キャリアパターン別・NMF・グラフベース・ハイブリッド）
- 推薦理由と参考人物の表示
- メンバーポジショニングマップの可視化
- 推薦結果のCSVダウンロード
"""

from io import StringIO
from typing import List
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ロガーの設定
logger = logging.getLogger(__name__)

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
from skillnote_recommendation.utils.ui_components import (
    apply_rich_ui_styles,
    render_gradient_header,
    render_section_divider,
    render_success_message
)


# =========================================================
# ヘルパー関数
# =========================================================

def create_growth_path_timeline(growth_path, role_name: str):
    """
    役職の成長パス（スキル取得シナリオ）をタイムライン形式で可視化

    Args:
        growth_path: RoleGrowthPathオブジェクト
        role_name: 役職名

    Returns:
        Plotlyのfigureオブジェクト
    """
    if not growth_path or not growth_path.skills_in_order:
        return None

    # データ準備
    skills = growth_path.skills_in_order

    # 優先度スコアを計算（取得率と取得順序を組み合わせる）
    # スコアが高いほど早期に習得すべきスキル
    skills_with_priority = []
    for skill in skills:
        # 取得率スコア：多くの人が取っているほど高い（0.0～1.0）
        acquisition_score = skill.acquisition_rate

        # 順序スコア：早期に取得されているほど高い（0.0～1.0）
        # 最大順序を取得して正規化
        max_order = max(s.average_order for s in skills)
        order_score = 1.0 - (skill.average_order / (max_order + 1)) if max_order > 0 else 0.5

        # 優先度スコア：取得率を重視（60%）、順序を考慮（40%）
        priority_score = (acquisition_score * 0.6) + (order_score * 0.4)

        skills_with_priority.append({
            'skill': skill,
            'priority_score': priority_score,
            'acquisition_score': acquisition_score,
            'order_score': order_score
        })

    # 優先度スコアでソート（降順：高い方が先）
    skills_with_priority.sort(key=lambda x: x['priority_score'], reverse=True)

    # ソート済みスキルリストを取得
    sorted_skills = [item['skill'] for item in skills_with_priority]

    # 成長段階を決定（取得率に基づく）
    # 取得率が高い = 多くの人が習得 = 基本スキル = 初級
    # 取得率が低い = 一部の専門家のみ = 高度なスキル = 上級
    stages = []
    colors = []
    for skill in sorted_skills:
        if skill.acquisition_rate >= 0.7:
            stages.append("🌱 初級")
            colors.append("#90EE90")  # Light green - 基本的・必須スキル
        elif skill.acquisition_rate >= 0.3:
            stages.append("🌿 中級")
            colors.append("#4CAF50")  # Green - 中堅レベルのスキル
        else:
            stages.append("🌳 上級")
            colors.append("#2E7D32")  # Dark green - 専門的・高度なスキル

    # スキル名（長すぎる場合は省略）
    skill_names = [
        skill.competence_name[:25] + "..." if len(skill.competence_name) > 25
        else skill.competence_name
        for skill in sorted_skills
    ]

    # 推奨取得順序（1から始まる連番）
    recommended_orders = list(range(1, len(sorted_skills) + 1))

    # 取得率（パーセント）
    acquisition_rates = [skill.acquisition_rate * 100 for skill in sorted_skills]

    # 優先度スコアを取得
    priority_scores = [item['priority_score'] for item in skills_with_priority]

    # ホバーテキスト
    hover_texts = [
        f"<b>{skill.competence_name}</b><br>"
        f"推奨取得順序: {rec_order}番目<br>"
        f"優先度スコア: {priority:.3f}<br>"
        f"<br>"
        f"【実データ】<br>"
        f"実際の平均取得順序: {skill.average_order:.1f}番目<br>"
        f"役職内取得率: {skill.acquisition_rate*100:.1f}% ({skill.acquisition_count}/{skill.total_members}名)<br>"
        f"成長段階: {stage}<br>"
        f"カテゴリー: {skill.category}"
        for skill, rec_order, priority, stage in zip(sorted_skills, recommended_orders, priority_scores, stages)
    ]

    # 累積スキル取得割合を計算
    total_skills = len(sorted_skills)
    cumulative_percentages = [(i + 1) / total_skills * 100 for i in range(total_skills)]

    # 時間軸（平均取得順序）を取得
    time_axis = [skill.average_order for skill in sorted_skills]

    # マーカーサイズ：取得率に応じて変化（10～30）
    marker_sizes = [10 + (rate / 100) * 20 for rate in acquisition_rates]

    # 折れ線グラフを作成
    fig = go.Figure()

    # メインの折れ線：累積スキル取得割合
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=cumulative_percentages,
        mode='lines+markers',
        name='累積スキル習得割合',
        line=dict(color='#2E7D32', width=3),
        marker=dict(
            size=marker_sizes,
            color=acquisition_rates,  # 取得率で色分け
            colorscale=[
                [0, '#2E7D32'],      # 0%: 濃緑（上級・専門スキル）
                [0.3, '#4CAF50'],    # 30%: 緑（中級）
                [0.7, '#90EE90'],    # 70%: 薄緑（初級・基本スキル）
                [1, '#C8E6C9']       # 100%: 非常に薄い緑
            ],
            colorbar=dict(
                title="取得率(%)",
                thickness=15,
                len=0.7
            ),
            line=dict(color='white', width=2),
            symbol='circle',
            showscale=True
        ),
        hovertext=hover_texts,
        hoverinfo='text'
    ))

    # 重要なマイルストーン（25%, 50%, 75%, 100%）にグリッド線を追加
    for milestone in [25, 50, 75, 100]:
        fig.add_hline(
            y=milestone,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"{milestone}%",
            annotation_position="right"
        )

    # 上位5件のスキル名を表示
    for i in range(min(5, len(sorted_skills))):
        fig.add_annotation(
            x=time_axis[i],
            y=cumulative_percentages[i],
            text=skill_names[i],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor='gray',
            ax=60 if i % 2 == 0 else -60,
            ay=-40 if i % 2 == 0 else 40,
            font=dict(size=9, color='black'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            borderpad=2
        )

    # レイアウト設定
    fig.update_layout(
        title=dict(
            text=f"<b>役職「{role_name}」のスキル取得シナリオ（累積表示）</b><br>"
                 f"<sup>時間軸に沿った累積スキル習得割合の推移（{growth_path.total_members}名のデータから分析）</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="<b>時間軸（平均取得順序：番目）</b>",
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title="<b>累積スキル習得割合 (%)</b>",
            gridcolor='lightgray',
            showgrid=True,
            range=[0, 105]
        ),
        height=600,
        margin=dict(l=80, r=100, t=100, b=80),
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=False
    )

    # グリッド線を追加
    fig.update_xaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)

    return fig


def create_growth_path_stages_chart(growth_path, role_name: str):
    """
    役職の成長パスを段階別に可視化（初級・中級・上級）

    Args:
        growth_path: RoleGrowthPathオブジェクト
        role_name: 役職名

    Returns:
        Plotlyのfigureオブジェクト
    """
    if not growth_path or not growth_path.skills_in_order:
        return None

    # 段階別にスキルを分類
    early_skills = growth_path.get_early_stage_skills(threshold=0.3)
    mid_skills = growth_path.get_mid_stage_skills(early_threshold=0.3, late_threshold=0.7)
    late_skills = growth_path.get_late_stage_skills(threshold=0.7)

    stages_data = [
        {
            'stage': '🌱 初級段階',
            'count': len(early_skills),
            'avg_acquisition_rate': sum(s.acquisition_rate for s in early_skills) / len(early_skills) * 100 if early_skills else 0,
            'color': '#90EE90'
        },
        {
            'stage': '🌿 中級段階',
            'count': len(mid_skills),
            'avg_acquisition_rate': sum(s.acquisition_rate for s in mid_skills) / len(mid_skills) * 100 if mid_skills else 0,
            'color': '#4CAF50'
        },
        {
            'stage': '🌳 上級段階',
            'count': len(late_skills),
            'avg_acquisition_rate': sum(s.acquisition_rate for s in late_skills) / len(late_skills) * 100 if late_skills else 0,
            'color': '#2E7D32'
        }
    ]

    # サンキー図を作成
    fig = go.Figure()

    # 棒グラフで表示
    fig.add_trace(go.Bar(
        x=[d['stage'] for d in stages_data],
        y=[d['count'] for d in stages_data],
        marker=dict(color=[d['color'] for d in stages_data]),
        text=[f"{d['count']}個<br>平均取得率: {d['avg_acquisition_rate']:.1f}%" for d in stages_data],
        textposition='auto',
        hovertext=[
            f"<b>{d['stage']}</b><br>"
            f"スキル数: {d['count']}個<br>"
            f"平均取得率: {d['avg_acquisition_rate']:.1f}%"
            for d in stages_data
        ],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>役職「{role_name}」の成長段階別スキル分布</b>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title="成長段階"),
        yaxis=dict(title="スキル数"),
        height=400,
        plot_bgcolor='white',
        showlegend=False
    )

    return fig


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
        reason='\n'.join(hybrid_rec.reasons) if hybrid_rec.reasons else 'グラフベース推薦',
        reference_persons=[]
    )


# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="CareerNavigator - AI推薦",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Apply rich UI styles
apply_rich_ui_styles()

# リッチなヘッダー
render_gradient_header(
    title="🧭 CareerNavigator",
    icon="🎯",
    description="AI推薦実行 - 学習済みAIモデルを使用して、メンバーへの力量推薦を実行します"
)


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

# Knowledge Graphの初期化（グラフベース推薦とハイブリッド推薦で必要）
# デフォルトパラメータをsession_stateで管理
if 'graph_similarity_threshold' not in st.session_state:
    from skillnote_recommendation.core.config import Config
    st.session_state.graph_similarity_threshold = Config.GRAPH_PARAMS['member_similarity_threshold']
    st.session_state.graph_similarity_top_k = Config.GRAPH_PARAMS['member_similarity_top_k']

if 'knowledge_graph' not in st.session_state:
    from skillnote_recommendation.graph import CompetenceKnowledgeGraph
    with st.spinner("Knowledge Graphを初期化中..."):
        st.session_state.knowledge_graph = CompetenceKnowledgeGraph(
            member_competence=td["member_competence"],
            member_master=td["members_clean"],
            competence_master=td["competence_master"],
            use_category_hierarchy=True,
            member_similarity_threshold=st.session_state.graph_similarity_threshold,
            member_similarity_top_k=st.session_state.graph_similarity_top_k
        )


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
    st.caption(f"メンバーコード: `{ref_person.member_code}`")
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
    reference_codes: List[str] = None,
    similar_career_codes: List[str] = None,
    different_career1_codes: List[str] = None,
    different_career2_codes: List[str] = None
):
    """
    メンバーポジショニングマップを複数のタブで表示する。

    Args:
        position_df: メンバー位置データ
        target_code: 対象メンバーコード
        reference_codes: 参考人物コードのリスト（従来型）
        similar_career_codes: 類似キャリアの参考人物コード（キャリアパターン別）
        different_career1_codes: 異なるキャリア1の参考人物コード
        different_career2_codes: 異なるキャリア2の参考人物コード
    """
    # キャリアパターン別推薦かどうかを判定
    use_pattern_based = (similar_career_codes is not None or
                        different_career1_codes is not None or
                        different_career2_codes is not None)

    # デフォルト値の設定
    if similar_career_codes is None:
        similar_career_codes = []
    if different_career1_codes is None:
        different_career1_codes = []
    if different_career2_codes is None:
        different_career2_codes = []
    if reference_codes is None:
        reference_codes = []
    # リッチなセクション区切り
    render_section_divider()

    # カードベースのヘッダー
    if use_pattern_based:
        st.markdown("""
        <div class="card fade-in">
            <h2>🗺️ メンバーポジショニングマップ</h2>
            <p>あなたと参考人物（キャリアパターン別）が、全メンバーの中でどの位置にいるかを可視化します</p>
            <div>
                <span class="badge badge-danger">あなた</span>
                <span class="badge badge-info">💼 類似キャリア</span>
                <span class="badge" style="background-color: #4CAF50;">🌟 異なるキャリア1</span>
                <span class="badge" style="background-color: #FF9800;">🚀 異なるキャリア2</span>
                <span class="badge">その他のメンバー</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
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
        if use_pattern_based:
            from skillnote_recommendation.utils.visualization import create_positioning_plot_with_patterns
            fig1 = create_positioning_plot_with_patterns(
                position_df, target_code,
                similar_career_codes, different_career1_codes, different_career2_codes,
                "総合スキルレベル", "保有力量数",
                "総合スキルレベル vs 保有力量数"
            )
        else:
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
        if use_pattern_based:
            from skillnote_recommendation.utils.visualization import create_positioning_plot_with_patterns
            fig2 = create_positioning_plot_with_patterns(
                position_df, target_code,
                similar_career_codes, different_career1_codes, different_career2_codes,
                "保有力量数", "平均レベル",
                "スキルの幅 vs 深さ"
            )
        else:
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
        if use_pattern_based:
            from skillnote_recommendation.utils.visualization import create_positioning_plot_with_patterns
            fig3 = create_positioning_plot_with_patterns(
                position_df, target_code,
                similar_career_codes, different_career1_codes, different_career2_codes,
                "潜在因子1", "潜在因子2",
                "潜在因子空間でのメンバー分布"
            )
        else:
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

st.subheader("👤 推薦対象メンバーの選択")

# 学習データに存在するメンバーのみをフィルタ
trained_member_codes = set(mf_model.member_codes)
available_members = members_df[
    members_df["メンバーコード"].isin(trained_member_codes)
]

if len(available_members) == 0:
    st.error("❌ 推薦可能なメンバーが存在しません。モデルを学習してください。")
    st.stop()

# メンバー選択プルダウン
member_options = dict(
    zip(available_members["メンバーコード"], available_members["メンバー名"])
)

selected_member_code = st.selectbox(
    "メンバーを選択",
    options=list(member_options.keys()),
    format_func=lambda x: f"{member_options[x]} ({x})",
    help=f"推薦可能なメンバー: {len(available_members)}名"
)


# =========================================================
# 基本設定
# =========================================================

st.subheader("⚙️ 基本設定")

col1, col2 = st.columns(2)

with col1:
    top_n = st.slider(
        "推薦数",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
        help="推薦する力量の数"
    )

with col2:
    selected_types = st.multiselect(
        "推薦する力量タイプ",
        options=["SKILL", "EDUCATION", "LICENSE"],
        default=["SKILL", "EDUCATION", "LICENSE"],
        help="SKILLのみ、EDUCATIONのみ等、絞り込みが可能です"
    )

    # 空リストの場合はNoneに変換（全てを推薦）
    competence_type = selected_types if selected_types else None

# =========================================================
# 詳細設定（オプション）
# =========================================================

st.markdown("---")

with st.expander("⚙️ 詳細設定（オプション）"):
    st.markdown("### 推薦手法の選択")

    # デフォルトはハイブリッド推薦（最も精度が高い）
    recommendation_method = st.radio(
        "推薦方法",
        options=["ハイブリッド推薦（推奨）", "NMF推薦", "グラフベース推薦", "キャリアパターン別推薦", "役職ベースの成長パス推薦"],
        index=0,
        help="通常はハイブリッド推薦をお勧めします",
        horizontal=False
    )

    st.markdown("---")
    st.markdown("### 比較モード")

    comparison_mode = st.checkbox(
        "複数の推薦方法を比較する",
        value=False,
        help="異なる推薦方法を同時実行して結果を比較できます"
    )

    if comparison_mode:
        methods_to_compare = st.multiselect(
            "比較する手法",
            options=["NMF推薦", "グラフベース推薦", "ハイブリッド推薦"],
            default=["NMF推薦", "グラフベース推薦"]
        )
        recommendation_method = None
    else:
        methods_to_compare = None

    st.markdown("---")
    st.markdown("### グラフ設定（グラフベース・ハイブリッド推薦のみ）")

    # メンバー類似度パラメータ調整
    st.markdown("#### 🔧 メンバー類似度パラメータ")

    col1, col2 = st.columns(2)

    with col1:
        new_threshold = st.slider(
            "類似度閾値",
            min_value=0.05,
            max_value=0.5,
            value=st.session_state.graph_similarity_threshold,
            step=0.05,
            help="メンバー間の類似度がこの値以上の場合にエッジを張ります。小さいほど多くの接続が生成されます。"
        )

    with col2:
        new_top_k = st.slider(
            "類似メンバー数",
            min_value=3,
            max_value=20,
            value=st.session_state.graph_similarity_top_k,
            step=1,
            help="各メンバーから接続する類似メンバーの最大数。多いほど推薦パスが豊富になります。"
        )

    # パラメータが変更された場合の通知
    params_changed = (
        new_threshold != st.session_state.graph_similarity_threshold or
        new_top_k != st.session_state.graph_similarity_top_k
    )

    if params_changed:
        st.info("⚠️ パラメータが変更されました。下のボタンでグラフを再構築してください。")

    # グラフ再構築ボタン
    if st.button("🔄 Knowledge Graphを再構築", help="新しいパラメータでグラフを再構築します"):
        st.session_state.graph_similarity_threshold = new_threshold
        st.session_state.graph_similarity_top_k = new_top_k

        from skillnote_recommendation.graph import CompetenceKnowledgeGraph
        with st.spinner("Knowledge Graphを再構築中..."):
            st.session_state.knowledge_graph = CompetenceKnowledgeGraph(
                member_competence=td["member_competence"],
                member_master=td["members_clean"],
                competence_master=td["competence_master"],
                use_category_hierarchy=True,
                member_similarity_threshold=new_threshold,
                member_similarity_top_k=new_top_k
            )
        st.success(f"✅ グラフを再構築しました！（閾値={new_threshold}, 類似メンバー数={new_top_k}）")
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 パス表示設定")

    show_paths = st.checkbox(
        "学習パスを表示",
        value=True,
        help="推薦理由を可視化します"
    )

    max_path_length = st.slider(
        "パスの最大ステップ数",
        min_value=2,
        max_value=20,
        value=10,
        step=2
    )

    max_paths = st.slider(
        "表示するパス数",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

    st.markdown("---")
    st.markdown("#### 👔 役職ベース推薦設定")

    min_acquisition_rate = st.slider(
        "最小取得率",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="役職内でこの割合以上のメンバーが習得しているスキルのみを推薦します。0に近いほど多くのスキルが推薦されます。"
    )

    st.info(f"📊 現在の設定: 役職内の{min_acquisition_rate*100:.0f}%以上のメンバーが習得しているスキルを推薦")

# デフォルト値の設定
diversity_strategy = "hybrid"  # 常にハイブリッド戦略を使用
rwr_weight = 0.5  # グラフとNMFを同等に評価


# =========================================================
# 推薦実行
# =========================================================

st.markdown("---")

if st.button("🚀 推薦を実行する", type="primary", use_container_width=True):
    # 比較モードの場合
    if comparison_mode:
        if not methods_to_compare:
            st.error("❌ 比較する手法を選択してください")
            st.stop()

        st.success(f"🔬 比較モード: {len(methods_to_compare)}個の手法を実行中...")

        # 比較モード処理（後で実装）
        comparison_results = {}

        import time
        from skillnote_recommendation.graph import build_hybrid_recommender

        for method in methods_to_compare:
            with st.spinner(f"{method}を実行中..."):
                try:
                    start_time = time.time()

                    if method == "NMF推薦":
                        recs = recommender.recommend(
                            member_code=selected_member_code,
                            top_n=top_n,
                            competence_type=competence_type,
                            category_filter=None,
                            use_diversity=True,
                            diversity_strategy=diversity_strategy
                        )
                        comparison_results[method] = {
                            'recommendations': recs,
                            'execution_time': time.time() - start_time,
                            'method_type': 'nmf'
                        }

                    elif method == "グラフベース推薦":
                        # Knowledge Graphの確認
                        if 'knowledge_graph' not in st.session_state:
                            st.error("❌ Knowledge Graphが初期化されていません")
                            continue

                        # RWRで推薦
                        from skillnote_recommendation.graph import RandomWalkRecommender
                        rwr = RandomWalkRecommender(
                            knowledge_graph=st.session_state['knowledge_graph'],
                            max_path_length=max_path_length,
                            max_paths=max_paths
                        )

                        graph_recommendations_raw = rwr.recommend(
                            member_code=selected_member_code,
                            top_n=top_n,
                            return_paths=show_paths,
                            competence_type=competence_type
                        )

                        # Recommendation形式に変換
                        from skillnote_recommendation.core.models import Recommendation
                        recs = []
                        for comp_code, score, paths in graph_recommendations_raw:
                            comp_info_row = td["competence_master"][
                                td["competence_master"]["力量コード"] == comp_code
                            ]
                            if not comp_info_row.empty:
                                recs.append(Recommendation(
                                    competence_code=comp_code,
                                    competence_name=comp_info_row.iloc[0]['力量名'],
                                    competence_type=comp_info_row.iloc[0]['力量タイプ'],
                                    category=comp_info_row.iloc[0].get('力量カテゴリー名', 'UNKNOWN'),
                                    priority_score=score,
                                    category_importance=0.5,
                                    interpretability_score=0.9,
                                    paths=paths if show_paths else []
                                ))

                        comparison_results[method] = {
                            'recommendations': recs,
                            'execution_time': time.time() - start_time,
                            'method_type': 'graph'
                        }

                    elif method == "ハイブリッド推薦":
                        # Knowledge Graphの確認
                        if 'knowledge_graph' not in st.session_state:
                            st.error("❌ Knowledge Graphが初期化されていません")
                            continue

                        # HybridGraphRecommenderを作成
                        hybrid_recommender = build_hybrid_recommender(
                            member_competence=td["member_competence"],
                            competence_master=td["competence_master"],
                            member_master=td["members_clean"],
                            graph_weight=rwr_weight,
                            cf_weight=1.0 - rwr_weight,
                            content_weight=0.0,
                            max_path_length=max_path_length,
                            max_paths=max_paths
                        )

                        # ハイブリッド推薦を実行
                        hybrid_recs = hybrid_recommender.recommend(
                            member_code=selected_member_code,
                            top_n=top_n,
                            competence_type=competence_type,
                            category_filter=None,
                            use_diversity=True
                        )

                        # Recommendation形式に変換
                        from skillnote_recommendation.core.models import Recommendation
                        recs = []
                        for hybrid_rec in hybrid_recs:
                            recs.append(Recommendation(
                                competence_code=hybrid_rec.competence_code,
                                competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code),
                                competence_type=hybrid_rec.competence_info.get('力量タイプ', 'UNKNOWN'),
                                category=hybrid_rec.competence_info.get('カテゴリー', ''),
                                priority_score=hybrid_rec.score,
                                category_importance=0.5,
                                interpretability_score=0.8,
                                paths=hybrid_rec.paths if show_paths else []
                            ))

                        comparison_results[method] = {
                            'recommendations': recs,
                            'execution_time': time.time() - start_time,
                            'method_type': 'hybrid'
                        }

                except Exception as e:
                    st.error(f"❌ {method}の実行中にエラーが発生しました: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # 比較結果を表示
        if comparison_results:
            st.success(f"✅ {len(comparison_results)}個の手法の実行が完了しました")

            # 空の結果がある場合は警告
            empty_methods = [method for method, result in comparison_results.items()
                           if len(result['recommendations']) == 0]
            if empty_methods:
                st.warning(f"⚠️ 以下の手法で推薦結果が0件でした: {', '.join(empty_methods)}\n\n"
                          "考えられる原因:\n"
                          "- 既に多くの力量を習得済み\n"
                          "- 力量タイプフィルタが厳しすぎる\n"
                          "- 推薦数を増やしてみてください")

            # 比較テーブルを作成
            max_len = max((len(result['recommendations']) for result in comparison_results.values()), default=0)

            if max_len > 0:
                st.markdown("---")
                st.subheader("📊 推薦結果の比較")

                # 比較テーブルを作成
                comparison_data = []

                for i in range(max_len):
                    row = {'順位': i + 1}

                    for method, result in comparison_results.items():
                        recs = result['recommendations']
                        if i < len(recs):
                            rec = recs[i]
                            row[f'{method}_力量名'] = rec.competence_name
                            row[f'{method}_スコア'] = f"{rec.priority_score:.3f}"
                            row[f'{method}_タイプ'] = rec.competence_type
                        else:
                            row[f'{method}_力量名'] = '-'
                            row[f'{method}_スコア'] = '-'
                            row[f'{method}_タイプ'] = '-'

                    comparison_data.append(row)

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, height=400)

            # 実行時間の比較
            st.markdown("### ⏱️ 実行時間の比較")
            time_cols = st.columns(len(comparison_results))
            for idx, (method, result) in enumerate(comparison_results.items()):
                with time_cols[idx]:
                    st.metric(
                        label=method,
                        value=f"{result['execution_time']:.2f}秒"
                    )

            # 詳細結果（個別タブで表示）
            st.markdown("### 📋 詳細結果")
            tabs = st.tabs(list(comparison_results.keys()))

            for idx, (method, result) in enumerate(comparison_results.items()):
                with tabs[idx]:
                    st.markdown(f"#### {method}の詳細結果")
                    recs = result['recommendations']

                    for rec in recs[:10]:  # 上位10件を表示
                        with st.expander(f"{rec.rank if hasattr(rec, 'rank') else '?'}. {rec.competence_name} (スコア: {rec.priority_score:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**力量タイプ**: {rec.competence_type}")
                            with col2:
                                st.markdown(f"**カテゴリ**: {rec.category}")

                            if hasattr(rec, 'paths') and rec.paths:
                                st.markdown("**推薦パス**:")
                                for path_idx, path in enumerate(rec.paths[:3], 1):
                                    path_names = [node.get('name', node.get('id', '?')) for node in path]
                                    st.caption(f"パス{path_idx}: {' → '.join(path_names)}")

        st.stop()  # 比較モードの場合はここで終了

    # 通常モード（単一手法）
    # 表示名から内部名に変換
    method_map = {
        "ハイブリッド推薦（推奨）": "ハイブリッド推薦",
        "NMF推薦": "NMF推薦",
        "グラフベース推薦": "グラフベース推薦",
        "キャリアパターン別推薦": "キャリアパターン別推薦",
        "役職ベースの成長パス推薦": "役職ベースの成長パス推薦"
    }
    internal_method = method_map.get(recommendation_method, recommendation_method)

    with st.spinner(f"推薦を生成中..."):
        try:
            import time
            from skillnote_recommendation.graph import HybridGraphRecommender

            # 実行時間を計測
            start_time = time.time()

            # 選択された推薦手法のみを実行
            if internal_method == "キャリアパターン別推薦":
                # キャリアパターン別推薦
                from skillnote_recommendation.core.config import Config
                from skillnote_recommendation.ml.career_pattern_classifier import create_classifier_from_config
                from skillnote_recommendation.ml.multi_pattern_recommender import create_multi_pattern_recommender

                # キャリアパターン分類器を作成
                classifier = create_classifier_from_config(
                    member_competence=td["member_competence"],
                    member_master=td["members_clean"],
                    mf_model=recommender.mf_model,
                    config=Config
                )

                # マルチパターン推薦器を作成
                multi_recommender = create_multi_pattern_recommender(
                    classifier=classifier,
                    competence_master=td["competence_master"],
                    member_competence=td["member_competence"],
                    mf_model=recommender.mf_model
                )

                # 各パターンでの推薦件数
                top_k_per_pattern = {
                    'similar': Config.CAREER_PATTERN_PARAMS['similar_career_top_k'],
                    'different1': Config.CAREER_PATTERN_PARAMS['different_career1_top_k'],
                    'different2': Config.CAREER_PATTERN_PARAMS['different_career2_top_k']
                }

                # パターン別推薦を実行
                pattern_recommendations = multi_recommender.recommend_by_patterns(
                    target_member_code=selected_member_code,
                    top_k_per_pattern=top_k_per_pattern,
                    competence_type=competence_type
                )

                # セッションステートに保存
                st.session_state.pattern_recommendations = pattern_recommendations

                # recsには全パターンの推薦を統合（CSV出力用）
                recs = []
                for pattern_name, pattern_rec in pattern_recommendations.items():
                    recs.extend(pattern_rec.recommendations)

                graph_recommendations = None

            elif internal_method == "役職ベースの成長パス推薦":
                # 役職ベースの成長パス推薦
                from skillnote_recommendation.graph import RoleBasedGrowthPathAnalyzer

                # 役職情報が含まれているか確認
                if '役職' not in td["members_clean"].columns:
                    st.error("❌ メンバーマスタに「役職」カラムが含まれていません。")
                    st.stop()

                # 取得日情報が含まれているか確認
                if '取得日' not in td["member_competence"].columns:
                    st.error("❌ メンバー保有力量データに「取得日」カラムが含まれていません。")
                    st.stop()

                # RoleBasedGrowthPathAnalyzerを初期化
                analyzer = RoleBasedGrowthPathAnalyzer(
                    members_df=td["members_clean"],
                    member_competence_df=td["member_competence"],
                    competence_master_df=td["competence_master"]
                )

                # 全役職の成長パスを分析
                with st.spinner("役職ごとの成長パスを分析中..."):
                    growth_paths = analyzer.analyze_all_roles(min_members=3)

                if not growth_paths:
                    st.warning("⚠️ 成長パスを生成できませんでした。各役職に最低3名のメンバーが必要です。")
                    recs = []
                    graph_recommendations = None
                else:
                    # 全役職について推薦を生成（役職全体の視点で）
                    all_role_recommendations = {}

                    with st.spinner("各役職の推薦を生成中..."):
                        for role_name, growth_path in growth_paths.items():
                            # 役職全体に対して推薦を生成（個人ベースではない）
                            role_recs = analyzer.recommend_for_role(
                                role_name=role_name,
                                top_n=top_n,
                                min_acquisition_rate=min_acquisition_rate
                            )

                            all_role_recommendations[role_name] = role_recs

                            # デバッグログ出力（ユーザーには表示しない）
                            logger.info(f"役職 '{role_name}': {len(role_recs)}件の推薦を生成")

                    # セッションステートに保存
                    st.session_state.role_based_growth_paths = growth_paths
                    st.session_state.role_based_analyzer = analyzer
                    st.session_state.role_based_recommendations = all_role_recommendations

                    # 統合用のrecsは空にする（役職別に表示するため）
                    recs = []
                    graph_recommendations = None

                # パターン別推薦情報をクリア
                if 'pattern_recommendations' in st.session_state:
                    del st.session_state['pattern_recommendations']

            elif internal_method == "NMF推薦":
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
                # パターン別推薦情報をクリア
                if 'pattern_recommendations' in st.session_state:
                    del st.session_state['pattern_recommendations']

            elif internal_method == "グラフベース推薦":
                # Knowledge Graphの確認
                if 'knowledge_graph' not in st.session_state:
                    st.error("❌ Knowledge Graphが初期化されていません。データ読み込みページで再度データを読み込んでください。")
                    st.stop()

                # RandomWalkRecommenderを作成（max_path_lengthとmax_pathsを設定）
                from skillnote_recommendation.graph.random_walk import RandomWalkRecommender
                rwr = RandomWalkRecommender(
                    knowledge_graph=st.session_state.knowledge_graph,
                    max_path_length=max_path_length,
                    max_paths=max_paths
                )

                # グラフベース推薦を実行
                graph_recommendations_raw = rwr.recommend(
                    member_code=selected_member_code,
                    top_n=top_n,
                    return_paths=show_paths,
                    competence_type=competence_type
                )

                # RWRの結果をHybridRecommendation形式に変換
                from skillnote_recommendation.graph.hybrid_recommender import HybridRecommendation
                graph_recommendations = []
                kg = st.session_state.knowledge_graph

                for comp_code, score, paths in graph_recommendations_raw:
                    # 力量情報を取得
                    comp_info_row = td["competence_master"][
                        td["competence_master"]["力量コード"] == comp_code
                    ]
                    if not comp_info_row.empty:
                        comp_info = {
                            '力量名': comp_info_row.iloc[0]['力量名'],
                            '力量タイプ': comp_info_row.iloc[0]['力量タイプ'],
                            'カテゴリー': comp_info_row.iloc[0].get('力量カテゴリー名', 'UNKNOWN'),
                            '概要': None
                        }
                    else:
                        comp_info = {
                            '力量名': comp_code,
                            '力量タイプ': 'UNKNOWN',
                            'カテゴリー': 'UNKNOWN',
                            '概要': None
                        }

                    # パスを人間が読める形式に変換
                    readable_paths = []
                    for path in paths:
                        readable_path = []
                        for node in path:
                            node_info = kg.get_node_info(node)
                            node_type = node_info.get('node_type', 'unknown')
                            node_name = node_info.get('name', node)

                            # メンバーノードの場合はメンバーコードを追加
                            if node_type == 'member':
                                member_code = node_info.get('code', '')
                                if member_code:
                                    node_name_with_code = f"{node_name} ({member_code})"
                                else:
                                    node_name_with_code = node_name
                            else:
                                node_name_with_code = node_name

                            readable_path.append({
                                'id': node,
                                'type': node_type,
                                'name': node_name_with_code,
                            })
                        readable_paths.append(readable_path)

                    # パスから推薦理由を生成（各パスの詳細を表示）
                    reasons = []

                    if len(readable_paths) > 0:
                        reasons.append(f"📊 抽出されたパス数: **{len(readable_paths)}個**")
                        reasons.append("")  # 空行

                        # パスのタイプ別に分類
                        direct_paths = []
                        category_paths = []
                        member_paths = []
                        competence_paths = []

                        for i, path in enumerate(readable_paths, 1):
                            if len(path) < 2:
                                continue

                            path_types = [n['type'] for n in path]
                            path_names = [n['name'] for n in path]

                            # パスの説明を生成
                            if len(path) == 2:
                                # 直接パス
                                direct_paths.append(f"  {i}. {path_names[0]} → {path_names[1]}")
                            elif 'category' in path_types:
                                # カテゴリー経由
                                category_paths.append(f"  {i}. {' → '.join(path_names)}")
                            elif path_types.count('member') > 1:
                                # 類似メンバー経由
                                member_paths.append(f"  {i}. {' → '.join(path_names)}")
                            elif 'competence' in path_types and len(path) >= 3:
                                # 既習得力量経由
                                competence_paths.append(f"  {i}. {' → '.join(path_names)}")
                            else:
                                # その他のパス
                                competence_paths.append(f"  {i}. {' → '.join(path_names)}")

                        # パスタイプ別に表示
                        if direct_paths:
                            reasons.append(f"**🎯 直接パス ({len(direct_paths)}個):**")
                            reasons.extend(direct_paths[:5])  # 最大5個表示
                            if len(direct_paths) > 5:
                                reasons.append(f"  ... 他{len(direct_paths) - 5}個")
                            reasons.append("")

                        if category_paths:
                            reasons.append(f"**📁 カテゴリー経由パス ({len(category_paths)}個):**")
                            reasons.extend(category_paths[:5])
                            if len(category_paths) > 5:
                                reasons.append(f"  ... 他{len(category_paths) - 5}個")
                            reasons.append("")

                        if member_paths:
                            reasons.append(f"**👥 類似メンバー経由パス ({len(member_paths)}個):**")
                            reasons.extend(member_paths[:5])
                            if len(member_paths) > 5:
                                reasons.append(f"  ... 他{len(member_paths) - 5}個")
                            reasons.append("")

                        if competence_paths:
                            reasons.append(f"**🔗 既習得力量経由パス ({len(competence_paths)}個):**")
                            reasons.extend(competence_paths[:5])
                            if len(competence_paths) > 5:
                                reasons.append(f"  ... 他{len(competence_paths) - 5}個")
                    else:
                        # カテゴリーベースまたは類似メンバーベースの推薦
                        reasons.append("**📋 カテゴリー・類似メンバーベースの推薦**")
                        reasons.append("")
                        reasons.append("あなたの既習得力量と同じカテゴリー、または類似メンバーの保有力量から推薦しました。")

                    # 理由がない場合のフォールバック
                    if len(reasons) == 0:
                        reasons = [f"📊 グラフ構造に基づく推薦"]

                    # HybridRecommendationを作成
                    hybrid_rec = HybridRecommendation(
                        competence_code=comp_code,
                        score=score,
                        graph_score=score,
                        cf_score=0.0,
                        content_score=0.0,
                        paths=readable_paths,
                        reasons=reasons,
                        competence_info=comp_info
                    )
                    graph_recommendations.append(hybrid_rec)

                # HybridRecommendationを標準のRecommendationに変換
                recs = [convert_hybrid_to_recommendation(hr) for hr in graph_recommendations]

                # 学習パスを生成（グラフベース推薦専用）
                from skillnote_recommendation.graph import generate_learning_path_from_recommendations
                learning_path = generate_learning_path_from_recommendations(
                    recommendations=graph_recommendations_raw,
                    knowledge_graph=st.session_state.knowledge_graph,
                    member_code=selected_member_code,
                    competence_master_df=td["competence_master"],
                    member_competence_df=td["member_competence"]
                )
                # セッションステートに保存
                st.session_state.graph_learning_path = learning_path

            elif internal_method == "ハイブリッド推薦":
                # Knowledge Graphの確認
                if 'knowledge_graph' not in st.session_state:
                    st.error("❌ Knowledge Graphが初期化されていません。データ読み込みページで再度データを読み込んでください。")
                    st.stop()

                # HybridGraphRecommenderを作成
                from skillnote_recommendation.graph import build_hybrid_recommender
                hybrid_recommender = build_hybrid_recommender(
                    member_competence=td["member_competence"],
                    competence_master=td["competence_master"],
                    member_master=td["members_clean"],
                    graph_weight=rwr_weight,
                    cf_weight=1.0 - rwr_weight,
                    content_weight=0.0,  # コンテンツベースは無効化（feature_engineerが必要なため）
                    max_path_length=max_path_length,
                    max_paths=max_paths
                )

                # ハイブリッド推薦を実行
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
            st.session_state.last_recommendation_method = internal_method
            if graph_recommendations:
                st.session_state.graph_recommendations = graph_recommendations

            # セッション状態に保存
            # 役職ベース推薦の場合は特別な判定が必要
            if internal_method == "役職ベースの成長パス推薦":
                role_based_recs = st.session_state.get('role_based_recommendations', {})
                # 全役職の推薦件数の合計を計算
                total_recs = sum(len(role_recs) for role_recs in role_based_recs.values())

                if total_recs == 0:
                    st.warning("⚠️ 推薦できる力量がありません。")

                    # 診断情報を表示
                    st.info("### 💡 推薦が空になった理由:")

                    st.write("**全ての役職で推薦可能なスキルがありませんでした。**")
                    st.write("これは以下のいずれかの理由が考えられます：")
                    st.write("- 各役職のメンバーが成長パス上の全スキルを既に習得済み")
                    st.write("- 最小取得率の設定が高すぎる（現在の設定を下げてみてください）")

                    # 改善案を提示
                    st.markdown("### 🔧 改善案:")
                    suggestions = []
                    suggestions.append("- **最小取得率を下げる**: 詳細設定で最小取得率を0.0～0.1に下げてみてください")
                    suggestions.append("- **推薦数を増やす**: スライダーで推薦数を増やしてみてください")

                    for suggestion in suggestions:
                        st.write(suggestion)

                    st.session_state.last_recommendations_df = None
                    st.session_state.last_recommendations = None
                    st.session_state.last_target_member_code = None
                else:
                    # 役職ベース推薦は成功
                    # DataFrame作成はスキップ（役職別に表示するため）
                    st.session_state.last_recommendations_df = None
                    st.session_state.last_recommendations = None

                    # リッチな成功メッセージ（実行時間を表示）
                    render_success_message(
                        title="✅ 推薦が完了しました",
                        message=f"全{len(role_based_recs)}役職で合計{total_recs}件の力量を推薦しました",
                        additional_info=f"実行時間: {elapsed_time:.2f}秒"
                    )

                    # 推薦結果の表示
                    st.markdown("---")

                    # 成長パス情報を取得
                    analyzer = st.session_state.get('role_based_analyzer')
                    growth_paths = st.session_state.get('role_based_growth_paths', {})

                    if analyzer:
                        # 対象メンバーの進捗状況を表示
                        progress_info = analyzer.get_member_progress(selected_member_code)

                        if progress_info:
                            st.markdown("## 📊 あなたの成長パス上での進捗状況")

                            # メトリクス表示
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("役職", progress_info['role_name'])
                            with col2:
                                st.metric("進捗率", f"{progress_info['progress_rate']*100:.1f}%")
                            with col3:
                                st.metric("習得済み", f"{progress_info['acquired_count']}個")
                            with col4:
                                st.metric("未習得", f"{progress_info['not_acquired_count']}個")

                            # プログレスバー
                            st.progress(progress_info['progress_rate'])

                    # 全役職の推薦を表示
                    if role_based_recs:
                        st.markdown("---")
                        st.markdown("## 🎯 役職別：次に習得すべきスキル")
                        st.info("各役職の成長パスを分析し、実際にその役職の人たちが習得してきた順序に基づいて、次のステップとして推薦すべきスキルを提示します。")

                        # 役職ごとにシンプルに表示
                        for role_name, role_recs_list in role_based_recs.items():
                            st.markdown(f"### 役職: {role_name}")

                            # この役職の情報を表示
                            if role_name in growth_paths:
                                growth_path = growth_paths[role_name]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("メンバー数", f"{growth_path.total_members}名")
                                with col2:
                                    st.metric("分析されたスキル数", f"{len(growth_path.skills_in_order)}個")

                                # 成長パスの可視化を追加
                                st.markdown("#### 📈 スキル取得シナリオ")
                                st.info("この役職のメンバーの実データ（取得率と取得時期）を分析し、推奨取得順序を算出しています。左から右へ：多くの人が早期に習得しているスキル順です。")

                                # タブで表示
                                timeline_tab, stages_tab = st.tabs(["🔄 取得順序タイムライン", "📊 段階別分布"])

                                with timeline_tab:
                                    # タイムライン図を作成
                                    timeline_fig = create_growth_path_timeline(growth_path, role_name)
                                    if timeline_fig:
                                        st.plotly_chart(timeline_fig, use_container_width=True)
                                        st.caption("💡 横軸：時間軸（平均取得順序）、縦軸：累積スキル習得割合（0%→100%）。マーカーの色：各スキルの取得率（濃緑=専門スキル、薄緑=基本スキル）、マーカーのサイズ：取得率の高さ。点線は25%/50%/75%/100%のマイルストーン。上位5件のスキル名を表示。")

                                with stages_tab:
                                    # 段階別チャートを作成
                                    stages_fig = create_growth_path_stages_chart(growth_path, role_name)
                                    if stages_fig:
                                        st.plotly_chart(stages_fig, use_container_width=True)
                                        st.caption("💡 成長パス上のスキルを、早期（初級）・中期（中級）・後期（上級）の3段階に分類して表示しています。")

                            st.markdown("---")

                            # 推薦が0件の場合
                            if not role_recs_list:
                                st.info(f"💡 **役職「{role_name}」の推薦スキルはありません。**\n\n"
                                       "これは以下のいずれかの理由が考えられます：\n"
                                       "- この役職のメンバーが成長パス上の全スキルを既に習得済み\n"
                                       "- 最小取得率の設定が高すぎる（詳細設定で下げてみてください）\n"
                                       "- この役職の成長パスで推薦可能なスキルが存在しない")
                                continue

                            # 推薦結果をシンプルなリストで表示
                            for idx, rec_dict in enumerate(role_recs_list, 1):
                                title = f"🎯 推薦 {idx}: {rec_dict['competence_name']} (優先度スコア: {rec_dict['priority_score']:.3f})"

                                with st.expander(title):
                                    # スキル情報
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"**力量タイプ:** {rec_dict['competence_type']}")
                                        st.markdown(f"**カテゴリー:** {rec_dict['category']}")
                                    with col2:
                                        st.markdown(f"**優先度スコア:** {rec_dict['priority_score']:.3f}")
                                        st.markdown(f"**平均取得順序:** {rec_dict['average_order']:.1f}番目")
                                    with col3:
                                        st.markdown(f"**役職内取得率:** {rec_dict['acquisition_rate']*100:.1f}%")
                                        # 成長段階のラベル
                                        # 取得率が高い = 基本スキル = 初級、取得率が低い = 専門スキル = 上級
                                        if rec_dict['acquisition_rate'] >= 0.7:
                                            stage = "🌱 初級（基本スキル）"
                                        elif rec_dict['acquisition_rate'] >= 0.3:
                                            stage = "🌿 中級（中堅スキル）"
                                        else:
                                            stage = "🌳 上級（専門スキル）"
                                        st.markdown(f"**成長段階:** {stage}")

                                    # 推薦理由
                                    st.markdown("---")
                                    st.markdown("### 📋 推薦理由")
                                    st.markdown(rec_dict['reason'])

                            st.markdown("---")
            elif not recs:
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
                render_success_message(
                    title="✅ 推薦が完了しました",
                    message=f"{len(recs)}件の力量を推薦しました",
                    additional_info=f"実行時間: {elapsed_time:.2f}秒"
                )

                # 推薦結果の表示
                st.markdown("---")

                # キャリアパターン別推薦の場合
                if internal_method == "キャリアパターン別推薦":
                    pattern_recs = st.session_state.get('pattern_recommendations', {})

                    if pattern_recs:
                        # 3つのパターンそれぞれを表示
                        for pattern_name in ['similar', 'different1', 'different2']:
                            if pattern_name not in pattern_recs:
                                continue

                            pattern_rec = pattern_recs[pattern_name]

                            # セクション区切り
                            st.markdown("---")
                            st.markdown(f"## {pattern_rec.pattern_label}")

                            # メッセージがある場合（参考人物が少ないなど）
                            if pattern_rec.message:
                                st.warning(pattern_rec.message)
                                continue

                            # 参考人物を表示
                            if pattern_rec.reference_persons:
                                st.markdown("### 👥 参考人物（あなたより総合スキルレベルが高いメンバー）")

                                # フィルタリング情報を表示
                                if pattern_rec.filtered_count > 0 and pattern_rec.total_count > 0:
                                    st.info(
                                        f"このパターンの全{pattern_rec.total_count}名のうち、"
                                        f"あなたより総合スキルレベルが高い{pattern_rec.filtered_count}名を参考人物として選定しています。"
                                    )

                                ref_person_names = []
                                for ref_person in pattern_rec.reference_persons:
                                    name_with_sim = f"{ref_person['name']} (類似度: {ref_person['similarity']})"
                                    ref_person_names.append(name_with_sim)

                                st.markdown("、".join(ref_person_names))
                                st.markdown("")  # 空行

                            # 推薦力量を表示
                            if pattern_rec.recommendations:
                                st.markdown("### 📋 推薦力量")

                                for idx, rec in enumerate(pattern_rec.recommendations, 1):
                                    with st.expander(f"**推薦 {idx}**: {rec.competence_name} (スコア: {rec.priority_score:.2f})"):
                                        # 力量情報
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**力量タイプ**: {rec.competence_type}")
                                        with col2:
                                            st.markdown(f"**カテゴリ**: {rec.category}")

                                        # 推薦理由
                                        st.markdown("---")
                                        st.markdown("**推薦理由**")
                                        st.markdown(rec.reason)
                            else:
                                st.info("このパターンからの推薦はありません。")

                    else:
                        st.error("キャリアパターン別推薦の結果が見つかりません。")

                # NMF推薦の場合
                elif internal_method == "NMF推薦":
                    # 推薦結果の詳細表示
                    for idx, rec in enumerate(recs, 1):
                        display_recommendation_details(rec, idx)

                # グラフベース推薦の場合（学習パス表示）
                elif internal_method == "グラフベース推薦":
                    # 学習パスを表示
                    learning_path = st.session_state.get('graph_learning_path')

                    if learning_path:
                        st.markdown("---")
                        st.markdown("## 📚 段階的な学習ロードマップ")
                        st.info("推薦された力量を、習得しやすい順序で3つのフェーズに分類しました。基礎から順番に学習することをお勧めします。")

                        # Phase 1: 基礎固め
                        if learning_path.phase_1_basic:
                            st.markdown("### 🌱 Phase 1: 基礎固め")
                            st.markdown(f"**{len(learning_path.phase_1_basic)}個の力量**　まずはこれらから始めましょう")

                            for idx, comp in enumerate(learning_path.phase_1_basic, 1):
                                with st.expander(f"**{idx}. {comp['competence_name']}** (優先度: {comp['priority_score']:.2f})"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("グラフスコア", f"{comp['rwr_score']:.3f}")
                                    with col2:
                                        st.metric("習得容易性", f"{comp['ease_score']:.2f}")
                                    with col3:
                                        st.markdown(f"**カテゴリ**: {comp['category']}")

                                    st.caption(f"力量タイプ: {comp['competence_type']} | 階層レベル: {comp['hierarchy_level']}")

                        # Phase 2: 専門性構築
                        if learning_path.phase_2_intermediate:
                            st.markdown("---")
                            st.markdown("### 🌿 Phase 2: 専門性構築")
                            st.markdown(f"**{len(learning_path.phase_2_intermediate)}個の力量**　Phase 1の後に取り組みましょう")

                            for idx, comp in enumerate(learning_path.phase_2_intermediate, 1):
                                with st.expander(f"**{idx}. {comp['competence_name']}** (優先度: {comp['priority_score']:.2f})"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("グラフスコア", f"{comp['rwr_score']:.3f}")
                                    with col2:
                                        st.metric("習得容易性", f"{comp['ease_score']:.2f}")
                                    with col3:
                                        st.markdown(f"**カテゴリ**: {comp['category']}")

                                    st.caption(f"力量タイプ: {comp['competence_type']} | 階層レベル: {comp['hierarchy_level']}")

                        # Phase 3: エキスパート
                        if learning_path.phase_3_expert:
                            st.markdown("---")
                            st.markdown("### 🌳 Phase 3: エキスパート")
                            st.markdown(f"**{len(learning_path.phase_3_expert)}個の力量**　高度な専門性を身につけましょう")

                            for idx, comp in enumerate(learning_path.phase_3_expert, 1):
                                with st.expander(f"**{idx}. {comp['competence_name']}** (優先度: {comp['priority_score']:.2f})"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("グラフスコア", f"{comp['rwr_score']:.3f}")
                                    with col2:
                                        st.metric("習得容易性", f"{comp['ease_score']:.2f}")
                                    with col3:
                                        st.markdown(f"**カテゴリ**: {comp['category']}")

                                    st.caption(f"力量タイプ: {comp['competence_type']} | 階層レベル: {comp['hierarchy_level']}")

                    # 従来の詳細表示も残す
                    st.markdown("---")
                    st.markdown("## 📋 推薦詳細（パス可視化）")

                    graph_recs_display = st.session_state.get('graph_recommendations', [])
                    if graph_recs_display:
                        for idx, hybrid_rec in enumerate(graph_recs_display, 1):
                            rec = convert_hybrid_to_recommendation(hybrid_rec)
                            title = f"🎯 推薦 {idx}: {rec.competence_name} (グラフスコア: {hybrid_rec.graph_score:.3f})"

                            with st.expander(title):
                                # スコア情報を表示
                                col_s1, col_s2 = st.columns(2)
                                with col_s1:
                                    st.metric("グラフスコア（RWR）", f"{hybrid_rec.graph_score:.3f}")
                                with col_s2:
                                    st.metric("パス数", f"{len(hybrid_rec.paths)}個")

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
                                        rwr_score=hybrid_rec.graph_score,
                                        nmf_score=hybrid_rec.cf_score,
                                        competence_info=hybrid_rec.competence_info
                                    )

                                    # グラフ可視化と詳細説明をタブで表示
                                    tab1, tab2 = st.tabs(["📊 グラフ可視化", "📝 詳細説明"])

                                    with tab1:
                                        member_name = members_df[
                                            members_df["メンバーコード"] == selected_member_code
                                        ]["メンバー名"].iloc[0]

                                        # 学習パス情報からフェーズマッピングを作成
                                        phase_info = {}
                                        if learning_path:
                                            for comp in learning_path.phase_1_basic:
                                                phase_info[comp['competence_code']] = 1
                                            for comp in learning_path.phase_2_intermediate:
                                                phase_info[comp['competence_code']] = 2
                                            for comp in learning_path.phase_3_expert:
                                                phase_info[comp['competence_code']] = 3

                                        # 段階的な学習パスを生成（Phase 1 → Phase 2 → Phase 3）
                                        combined_paths = list(hybrid_rec.paths) if hybrid_rec.paths else []
                                        if learning_path:
                                            from skillnote_recommendation.graph import generate_progressive_learning_paths
                                            progressive_paths = generate_progressive_learning_paths(
                                                learning_path=learning_path,
                                                member_code=selected_member_code,
                                                member_name=member_name,
                                                max_paths=3  # 各フェーズから最大3つの力量でパスを生成
                                            )
                                            # 既存のRWRパスと段階的な学習パスを結合
                                            combined_paths.extend(progressive_paths)

                                        fig = visualizer.visualize_recommendation_path(
                                            paths=combined_paths,
                                            target_member_name=member_name,
                                            target_competence_name=hybrid_rec.competence_info.get('力量名', hybrid_rec.competence_code),
                                            phase_info=phase_info if phase_info else None
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

                # ハイブリッド推薦の場合
                elif recommendation_method in ["ハイブリッド推薦"]:
                    graph_recs_display = st.session_state.get('graph_recommendations', [])

                    if graph_recs_display:
                        # 推薦結果の詳細表示
                        for idx, hybrid_rec in enumerate(graph_recs_display, 1):
                            rec = convert_hybrid_to_recommendation(hybrid_rec)

                            # スコア表示のタイトルを決定
                            if recommendation_method == "グラフベース推薦":
                                title = f"🎯 推薦 {idx}: {rec.competence_name} (グラフスコア: {hybrid_rec.graph_score:.3f})"
                            else:
                                title = f"🎯 推薦 {idx}: {rec.competence_name} (総合スコア: {hybrid_rec.score:.3f})"

                            with st.expander(title):
                                # スコア情報を表示
                                if recommendation_method == "グラフベース推薦":
                                    col_s1, col_s2 = st.columns(2)
                                    with col_s1:
                                        st.metric("グラフスコア（RWR）", f"{hybrid_rec.graph_score:.3f}")
                                    with col_s2:
                                        st.metric("パス数", f"{len(hybrid_rec.paths)}個")
                                else:  # ハイブリッド推薦
                                    col_s1, col_s2, col_s3 = st.columns(3)
                                    with col_s1:
                                        st.metric("総合スコア", f"{hybrid_rec.score:.3f}")
                                    with col_s2:
                                        st.metric("グラフスコア", f"{hybrid_rec.graph_score:.3f}")
                                    with col_s3:
                                        st.metric("NMFスコア", f"{hybrid_rec.cf_score:.3f}")

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
                                        rwr_score=hybrid_rec.graph_score,
                                        nmf_score=hybrid_rec.cf_score,
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
    render_section_divider()

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
        # ポジショニングデータを作成
        position_df = create_member_positioning_data(
            td["member_competence"],
            td["members_clean"],
            mf_model
        )

        # キャリアパターン別推薦がある場合はパターンコードを使用
        pattern_recs = st.session_state.get('pattern_recommendations', {})
        if pattern_recs:
            # 各パターンから参考人物コードを抽出
            similar_codes = []
            different1_codes = []
            different2_codes = []

            if 'similar' in pattern_recs:
                similar_codes = [p['code'] for p in pattern_recs['similar'].reference_persons]
            if 'different1' in pattern_recs:
                different1_codes = [p['code'] for p in pattern_recs['different1'].reference_persons]
            if 'different2' in pattern_recs:
                different2_codes = [p['code'] for p in pattern_recs['different2'].reference_persons]

            # パターンベースのポジショニングマップを表示
            display_positioning_maps(
                position_df,
                st.session_state.last_target_member_code,
                similar_career_codes=similar_codes,
                different_career1_codes=different1_codes,
                different_career2_codes=different2_codes
            )
        else:
            # 従来の参考人物ベースのポジショニングマップを表示
            reference_codes = get_reference_person_codes(
                st.session_state.last_recommendations
            )
            display_positioning_maps(
                position_df,
                st.session_state.last_target_member_code,
                reference_codes=reference_codes
            )

        # キャリアパス推薦
        render_section_divider()

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
