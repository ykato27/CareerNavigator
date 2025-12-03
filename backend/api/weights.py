"""
Weights management API endpoints.

This module provides endpoints for managing recommendation weights.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.request.weights import UpdateWeightsRequest, GetWeightsRequest
from backend.schemas.response.weights import WeightsResponse
from backend.services.weights_service import weights_service
from backend.core.exceptions import AppException
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/weights/update", response_model=WeightsResponse)
async def update_weights(request: UpdateWeightsRequest):
    """
    Update recommendation weights for a model.

    Args:
        request: Request containing model_id and new weights

    Returns:
        WeightsResponse: Updated weights and confirmation

    Raises:
        HTTPException: If model not found or weights invalid
    """
    try:
        updated_weights = await weights_service.update_weights(
            model_id=request.model_id, weights=request.weights
        )

        return WeightsResponse(
            success=True,
            model_id=request.model_id,
            weights=updated_weights,
            message="Weights updated successfully",
        )

    except AppException:
        # Custom exceptions are handled by error_handler_middleware
        raise
    except Exception as e:
        logger.error("Unexpected error updating weights", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")


@router.post("/weights/get", response_model=WeightsResponse)
async def get_weights(request: GetWeightsRequest):
    """
    Get current recommendation weights for a model.

    Args:
        request: Request containing model_id

    Returns:
        WeightsResponse: Current weights

    Raises:
        HTTPException: If model not found
    """
    try:
        current_weights = await weights_service.get_weights(model_id=request.model_id)

        return WeightsResponse(
            success=True,
            model_id=request.model_id,
            weights=current_weights,
            message="",
        )

    except AppException:
        raise
    except Exception as e:
        logger.error("Unexpected error getting weights", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
