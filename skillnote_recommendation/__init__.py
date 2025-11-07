"""
スキルノート推薦システム

技術者向けの力量推薦を行うシステム（機械学習ベース）
"""

from skillnote_recommendation.core.config import Config
from skillnote_recommendation.core.models import (
    Member,
    Competence,
    MemberCompetence,
    Recommendation,
)
from skillnote_recommendation.core.data_loader import DataLoader
from skillnote_recommendation.core.data_transformer import DataTransformer

__version__ = "1.0.0"
__all__ = [
    "Config",
    "Member",
    "Competence",
    "MemberCompetence",
    "Recommendation",
    "DataLoader",
    "DataTransformer",
]
