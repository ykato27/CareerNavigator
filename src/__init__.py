"""
スキルノート推薦システム

パッケージ初期化
"""

from .config import Config
from .models import Member, Competence, MemberCompetence, Recommendation
from .data_loader import DataLoader
from .data_transformer import DataTransformer
from .similarity_calculator import SimilarityCalculator
from .recommendation_engine import RecommendationEngine
from .recommendation_system import RecommendationSystem

__version__ = '1.0.0'
__all__ = [
    'Config',
    'Member',
    'Competence',
    'MemberCompetence',
    'Recommendation',
    'DataLoader',
    'DataTransformer',
    'SimilarityCalculator',
    'RecommendationEngine',
    'RecommendationSystem',
]
