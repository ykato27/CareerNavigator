from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

from backend.utils import session_manager, validate_weights

router = APIRouter()
logger = logging.getLogger(__name__)


class UpdateWeightsRequest(BaseModel):
    model_id: str
    weights: Dict[str, float]


class OptimizeWeightsRequest(BaseModel):
    model_id: str
    n_trials: int = 50
    n_jobs: int = -1
    holdout_ratio: float = 0.2
    top_k: int = 10


@router.post("/weights/update")
async def update_weights(request: UpdateWeightsRequest):
    """
    Manually update recommendation weights.

    Args:
        request: Contains model_id and weights dictionary

    Returns:
        dict: Updated normalized weights

    Raises:
        HTTPException: If model not found or weights invalid
    """
    # Get model from session manager
    recommender = session_manager.get_model(request.model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Please train a model first."
        )

    try:
        logger.info(f"[WEIGHTS] Updating weights for model {request.model_id}")

        # Validate weight keys
        required_keys = {'readiness', 'bayesian', 'utility'}
        if set(request.weights.keys()) != required_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Weights must include exactly: {required_keys}"
            )

        # Validate weight values
        total = sum(request.weights.values())
        if total == 0:
            raise HTTPException(
                status_code=400,
                detail="Weight sum cannot be zero"
            )

        # Normalize weights
        normalized_weights = {k: v / total for k, v in request.weights.items()}

        # Validate normalized weights sum to 1.0
        validate_weights(normalized_weights)

        # Update model weights
        if hasattr(recommender, 'set_weights'):
            recommender.set_weights(normalized_weights)
        else:
            recommender.weights = normalized_weights

        logger.info(f"[WEIGHTS] Weights updated successfully: {normalized_weights}")

        return {
            "success": True,
            "weights": normalized_weights,
            "message": "重みを更新しました"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WEIGHTS] Weight update failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Weight update failed: {str(e)}"
        )


@router.post("/weights/optimize")
async def optimize_weights(request: OptimizeWeightsRequest):
    """
    Automatically optimize weights using Bayesian optimization.

    Args:
        request: Contains model_id and optimization parameters

    Returns:
        dict: Optimized weights and success status

    Raises:
        HTTPException: If model not found or optimization fails
    """
    # Get model from session manager
    recommender = session_manager.get_model(request.model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Please train a model first."
        )

    if not hasattr(recommender, 'optimize_weights'):
        raise HTTPException(
            status_code=400,
            detail="Model does not support weight optimization"
        )

    try:
        logger.info(
            f"[WEIGHTS] Starting weight optimization for model {request.model_id} "
            f"(trials={request.n_trials})"
        )

        best_weights = recommender.optimize_weights(
            n_trials=request.n_trials,
            n_jobs=request.n_jobs,
            holdout_ratio=request.holdout_ratio,
            top_k=request.top_k
        )

        logger.info(f"[WEIGHTS] Optimization completed: {best_weights}")

        return {
            "success": True,
            "optimized_weights": best_weights,
            "message": "重みの最適化が完了しました"
        }

    except Exception as e:
        logger.error(f"[WEIGHTS] Optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.get("/weights/{model_id}")
async def get_weights(model_id: str):
    """
    Get current weights for a model.

    Args:
        model_id: The unique model identifier

    Returns:
        dict: Current weights

    Raises:
        HTTPException: If model not found
    """
    # Get model from session manager
    recommender = session_manager.get_model(model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Please train a model first."
        )

    weights = (
        recommender.get_weights()
        if hasattr(recommender, 'get_weights')
        else recommender.weights
    )

    return {
        "model_id": model_id,
        "weights": weights
    }
