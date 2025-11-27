"""
機械学習ベースの推薦システム

Matrix Factorization、多様性再ランキング、MLベース推薦エンジンを提供
"""

# 循環インポートを避けるため、exceptionsを最初にインポート
from skillnote_recommendation.ml.exceptions import ColdStartError, MLModelNotTrainedError
from skillnote_recommendation.ml.matrix_factorization import MatrixFactorizationModel
from skillnote_recommendation.ml.diversity import DiversityReranker
from skillnote_recommendation.ml.ml_recommender import MLRecommender
from skillnote_recommendation.ml.causal_structure_learner import CausalStructureLearner
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender
from skillnote_recommendation.ml.unified_sem_estimator import UnifiedSEMEstimator

# Optional dependencies
try:
    from skillnote_recommendation.ml.enhanced_graph_recommender import (
        EnhancedSkillTransitionGraphRecommender,
    )
    _HAS_NODE2VEC = True
except ImportError:
    _HAS_NODE2VEC = False

__all__ = [
    "ColdStartError",
    "MLModelNotTrainedError",
    "MatrixFactorizationModel",
    "DiversityReranker",
    "MLRecommender",
    "CausalStructureLearner",
    "CausalGraphRecommender",
    "UnifiedSEMEstimator",
]

if _HAS_NODE2VEC:
    __all__.append("EnhancedSkillTransitionGraphRecommender")
