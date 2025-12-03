"""
Recommendation API endpoints.

This module provides endpoints for generating skill recommendations.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.request.recommendation import GetRecommendationsRequest
from backend.schemas.response.recommendation import RecommendationsResponse
from backend.services.recommendation_service import recommendation_service
from backend.core.exceptions import AppException
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/recommend", response_model=RecommendationsResponse)
async def get_recommendations(request: GetRecommendationsRequest):
    """
    Get skill recommendations for a member.

    Args:
        request: Request containing model_id, member_id, and top_n

    Returns:
        RecommendationsResponse: List of recommended skills with scores

    Raises:
        HTTPException: If model or member not found
    """
    try:
        result = await recommendation_service.get_recommendations(
            model_id=request.model_id,
            member_id=request.member_id,
            top_n=request.top_n,
        )

        return RecommendationsResponse(
            success=True,
            model_id=result["model_id"],
            member_id=result["member_id"],
            member_name=result.get("member_name", ""),
            recommendations=result["recommendations"],
            metadata=result.get("metadata", {}),
        )

    except AppException:
        # Custom exceptions are handled by error_handler_middleware
        raise
    except Exception as e:
        logger.error("Unexpected error generating recommendations", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")
