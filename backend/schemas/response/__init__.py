"""Response schemas initialization."""

from backend.schemas.response.common import (
    ErrorDetail,
    ErrorResponse,
    SuccessResponse,
    MessageResponse,
    PaginatedResponse,
)
from backend.schemas.response.training import (
    ModelSummary,
    TrainModelResponse,
    GetModelSummaryResponse,
    DeleteModelResponse,
)
from backend.schemas.response.recommendation import (
    RecommendationItem,
    RecommendationsResponse,
)
from backend.schemas.response.weights import WeightsResponse

__all__ = [
    # Common
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",
    "MessageResponse",
    "PaginatedResponse",
    # Training
    "ModelSummary",
    "TrainModelResponse",
    "GetModelSummaryResponse",
    "DeleteModelResponse",
    # Recommendation
    "RecommendationItem",
    "RecommendationsResponse",
    # Weights
    "WeightsResponse",
]
