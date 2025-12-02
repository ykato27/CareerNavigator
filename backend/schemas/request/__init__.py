"""Request schemas initialization."""

from backend.schemas.request.training import TrainModelRequest, DeleteModelRequest
from backend.schemas.request.recommendation import GetRecommendationsRequest
from backend.schemas.request.weights import UpdateWeightsRequest, GetWeightsRequest

__all__ = [
    "TrainModelRequest",
    "DeleteModelRequest",
    "GetRecommendationsRequest",
    "UpdateWeightsRequest",
    "GetWeightsRequest",
]
