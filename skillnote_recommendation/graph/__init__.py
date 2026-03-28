"""
Graph-based recommendation modules

Knowledge Graph構築とグラフベースの推薦アルゴリズム
"""

from .knowledge_graph import CompetenceKnowledgeGraph
from .random_walk import RandomWalkRecommender
from .hybrid_recommender import HybridGraphRecommender, HybridRecommendation
from .hybrid_builder import build_hybrid_recommender
from .path_visualizer import RecommendationPathVisualizer, visualize_multiple_recommendations
from .category_hierarchy import CategoryHierarchy
from .career_path import (
    CareerGapAnalyzer,
    LearningPathGenerator,
    CareerPathAnalysis,
    CompetenceGap,
    RecommendationLearningPath,
    generate_learning_path_from_recommendations,
    generate_progressive_learning_paths,
)
from .career_path_visualizer import CareerPathVisualizer, format_career_path_summary
from .role_based_growth_path import (
    RoleBasedGrowthPathAnalyzer,
    RoleGrowthPath,
    SkillAcquisitionPattern,
)
from .role_growth_path_visualizer import RoleGrowthPathVisualizer

# 新しい改善版モジュール
from .enhanced_path_visualizer import EnhancedPathVisualizer, EdgeStatistics, create_comparison_view
from .sankey_visualizer import SkillTransitionSankeyVisualizer, TimeBasedSankeyVisualizer

__all__ = [
    "CompetenceKnowledgeGraph",
    "RandomWalkRecommender",
    "HybridGraphRecommender",
    "HybridRecommendation",
    "build_hybrid_recommender",
    "RecommendationPathVisualizer",
    "visualize_multiple_recommendations",
    "CategoryHierarchy",
    "CareerGapAnalyzer",
    "LearningPathGenerator",
    "CareerPathAnalysis",
    "CompetenceGap",
    "RecommendationLearningPath",
    "generate_learning_path_from_recommendations",
    "generate_progressive_learning_paths",
    "CareerPathVisualizer",
    "format_career_path_summary",
    # 役職ベースの成長パス
    "RoleBasedGrowthPathAnalyzer",
    "RoleGrowthPath",
    "SkillAcquisitionPattern",
    "RoleGrowthPathVisualizer",
    # 新しい改善版
    "EnhancedPathVisualizer",
    "EdgeStatistics",
    "create_comparison_view",
    "SkillTransitionSankeyVisualizer",
    "TimeBasedSankeyVisualizer",
]

# Force redeploy
