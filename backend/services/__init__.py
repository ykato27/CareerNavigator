"""Services package initialization."""

from backend.services.training_service import TrainingService, training_service
from backend.services.recommendation_service import (
    RecommendationService,
    recommendation_service,
)
from backend.services.weights_service import WeightsService, weights_service

__all__ = [
    "TrainingService",
    "training_service",
    "RecommendationService",
    "recommendation_service",
    "WeightsService",
    "weights_service",
]
