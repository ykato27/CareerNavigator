from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from backend.utils import session_manager

router = APIRouter()
logger = logging.getLogger(__name__)


class RecommendationRequest(BaseModel):
    model_id: str
    member_id: str
    top_n: int = 10


@router.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Get skill recommendations for a member using a trained model.

    Args:
        request: Contains model_id, member_id, and top_n

    Returns:
        dict: Recommendations with scores and explanations

    Raises:
        HTTPException: If model or member not found
    """
    # Get model from session manager
    recommender = session_manager.get_model(request.model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Please train a model first."
        )

    try:
        logger.info(f"[RECOMMEND] Getting recommendations for member {request.member_id}")

        recommendations = recommender.recommend(request.member_id, top_n=request.top_n)

        if not recommendations:
            logger.info(f"[RECOMMEND] No recommendations found for member {request.member_id}")
            return {
                "member_id": request.member_id,
                "recommendations": [],
                "message": "推奨できるスキルが見つかりませんでした"
            }

        logger.info(f"[RECOMMEND] Found {len(recommendations)} recommendations")

        return {
            "member_id": request.member_id,
            "recommendations": recommendations
        }

    except KeyError:
        logger.error(f"[RECOMMEND] Member ID '{request.member_id}' not found")
        raise HTTPException(
            status_code=404,
            detail=f"Member ID '{request.member_id}' not found in the training data"
        )
    except Exception as e:
        logger.error(f"[RECOMMEND] Recommendation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )
