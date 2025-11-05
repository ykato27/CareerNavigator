"""
Persistence layer for data storage and session management.

This module provides functionality for:
- User management and authentication
- Recommendation history storage
- Model persistence and reuse
- Session management
"""

from .database import DatabaseManager
from .models import User, RecommendationHistory, ModelMetadata, UserSession
from .repository import UserRepository, RecommendationHistoryRepository, ModelRepository
from .session_manager import SessionManager

__all__ = [
    "DatabaseManager",
    "User",
    "RecommendationHistory",
    "ModelMetadata",
    "UserSession",
    "UserRepository",
    "RecommendationHistoryRepository",
    "ModelRepository",
    "SessionManager",
]
