"""
ハイブリッド推薦システムのビルダー

ハイブリッド推薦システム（グラフベース + 協調フィルタリング + コンテンツベース）を
簡単に構築するためのビルダー関数を提供
"""

import pandas as pd
from typing import Optional

from .knowledge_graph import CompetenceKnowledgeGraph
from .hybrid_recommender import HybridGraphRecommender
from ..ml.ml_recommender import MLRecommender
from ..ml.content_based_recommender import ContentBasedRecommender
from ..ml.feature_engineering import FeatureEngineer


def build_hybrid_recommender(
    member_competence: pd.DataFrame,
    competence_master: pd.DataFrame,
    member_master: pd.DataFrame,
    graph_weight: float = 0.4,
    cf_weight: float = 0.3,
    content_weight: float = 0.3,
    max_path_length: int = 10,
    use_tuning: bool = False,
    enable_cache: bool = True,
    category_hierarchy: Optional[dict] = None
) -> HybridGraphRecommender:
    """
    ハイブリッド推薦システムを構築

    Args:
        member_competence: メンバー習得力量DataFrame
        competence_master: 力量マスタDataFrame
        member_master: メンバーマスタDataFrame
        graph_weight: グラフベーススコアの重み（デフォルト: 0.4）
        cf_weight: 協調フィルタリングスコアの重み（デフォルト: 0.3）
        content_weight: コンテンツベーススコアの重み（デフォルト: 0.3）
        max_path_length: 推薦パスの最大長さ/ステップ数（デフォルト: 10）
        use_tuning: ハイパーパラメータチューニングを使用するか
        enable_cache: グラフのキャッシュを有効にするか
        category_hierarchy: カテゴリ階層構造

    Returns:
        構築されたHybridGraphRecommender

    Example:
        >>> hybrid_recommender = build_hybrid_recommender(
        ...     member_competence=member_competence_df,
        ...     competence_master=competence_master_df,
        ...     member_master=member_master_df,
        ...     graph_weight=0.4,
        ...     cf_weight=0.3,
        ...     content_weight=0.3
        ... )
        >>> recommendations = hybrid_recommender.recommend(
        ...     member_code='M001',
        ...     top_n=10
        ... )
    """
    print("\n" + "=" * 80)
    print("ハイブリッド推薦システム構築開始")
    print("=" * 80)

    # 1. 知識グラフの構築
    print("\n[1/5] 知識グラフを構築...")
    kg = CompetenceKnowledgeGraph()
    kg.build_graph(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=member_master
    )

    # 2. ML推薦エンジンの構築
    print("\n[2/5] 協調フィルタリング推薦エンジンを構築...")
    ml_recommender = MLRecommender.build(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=member_master,
        use_tuning=use_tuning
    )

    # 3. 特徴量エンジニアの構築
    print("\n[3/5] 特徴量エンジニアを構築...")
    feature_engineer = FeatureEngineer(
        member_master=member_master,
        competence_master=competence_master,
        member_competence=member_competence,
        category_hierarchy=category_hierarchy
    )

    # 4. コンテンツベース推薦エンジンの構築
    print("\n[4/5] コンテンツベース推薦エンジンを構築...")
    content_recommender = ContentBasedRecommender(
        feature_engineer=feature_engineer,
        member_master=member_master,
        competence_master=competence_master,
        member_competence=member_competence
    )

    # 5. ハイブリッド推薦エンジンの構築
    print("\n[5/5] ハイブリッド推薦エンジンを構築...")
    hybrid_recommender = HybridGraphRecommender(
        knowledge_graph=kg,
        ml_recommender=ml_recommender,
        content_recommender=content_recommender,
        feature_engineer=feature_engineer,
        graph_weight=graph_weight,
        cf_weight=cf_weight,
        content_weight=content_weight,
        max_path_length=max_path_length,
        enable_cache=enable_cache
    )

    print("\n" + "=" * 80)
    print("ハイブリッド推薦システム構築完了")
    print("=" * 80)

    return hybrid_recommender


def quick_recommend(
    member_code: str,
    member_competence: pd.DataFrame,
    competence_master: pd.DataFrame,
    member_master: pd.DataFrame,
    top_n: int = 10,
    graph_weight: float = 0.4,
    cf_weight: float = 0.3,
    content_weight: float = 0.3,
    max_path_length: int = 10
):
    """
    クイック推薦（ハイブリッド推薦システムを構築して推薦を実行）

    Args:
        member_code: 対象メンバーコード
        member_competence: メンバー習得力量DataFrame
        competence_master: 力量マスタDataFrame
        member_master: メンバーマスタDataFrame
        top_n: 推薦件数
        graph_weight: グラフベーススコアの重み
        cf_weight: 協調フィルタリングスコアの重み
        content_weight: コンテンツベーススコアの重み
        max_path_length: 推薦パスの最大長さ/ステップ数

    Returns:
        推薦結果のリスト

    Example:
        >>> recommendations = quick_recommend(
        ...     member_code='M001',
        ...     member_competence=member_competence_df,
        ...     competence_master=competence_master_df,
        ...     member_master=member_master_df,
        ...     top_n=10
        ... )
    """
    hybrid_recommender = build_hybrid_recommender(
        member_competence=member_competence,
        competence_master=competence_master,
        member_master=member_master,
        graph_weight=graph_weight,
        cf_weight=cf_weight,
        content_weight=content_weight,
        max_path_length=max_path_length
    )

    recommendations = hybrid_recommender.recommend(
        member_code=member_code,
        top_n=top_n
    )

    return recommendations
