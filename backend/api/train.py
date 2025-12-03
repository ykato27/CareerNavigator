"""
Training API endpoints.

This module provides endpoints for training causal inference models.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.request.training import TrainModelRequest
from backend.schemas.response.training import (
    TrainModelResponse,
    GetModelSummaryResponse,
    DeleteModelResponse,
    ModelSummary,
)
from backend.services.training_service import training_service
from backend.core.exceptions import AppException
from backend.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/train", response_model=TrainModelResponse)
async def train_causal_model(request: TrainModelRequest):
    """
    Train a causal inference model using LiNGAM algorithm.

    Args:
        request: Training parameters including session_id, min_members_per_skill,
                correlation_threshold, and optional custom weights

    Returns:
        TrainModelResponse: Model ID, training summary, and success message

    Raises:
        HTTPException: If session not found or training fails
    """
    try:
        result = await training_service.train_model(
            session_id=request.session_id,
            min_members_per_skill=request.min_members_per_skill,
            correlation_threshold=request.correlation_threshold,
            weights=request.weights,
        )

        return TrainModelResponse(
            success=True,
            model_id=result["model_id"],
            message=result["message"],
            summary=ModelSummary(**result["summary"]),
        )

    except AppException:
        # Custom exceptions are handled by error_handler_middleware
        raise
    except Exception as e:
        logger.error("Unexpected error during training", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model/{model_id}/summary", response_model=GetModelSummaryResponse)
async def get_model_summary(model_id: str):
    """
    Get summary information about a trained model.

    Args:
        model_id: The unique model identifier

    Returns:
        GetModelSummaryResponse: Model summary including dimensions and weights

    Raises:
        HTTPException: If model not found
    """
    try:
        summary_data = await training_service.get_model_summary(model_id)

        return GetModelSummaryResponse(success=True, **summary_data)

    except AppException:
        raise
    except Exception as e:
        logger.error("Unexpected error getting model summary", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model/{model_id}", response_model=DeleteModelResponse)
async def delete_model(model_id: str):
    """
    Delete a trained model from memory.

    Args:
        model_id: The unique model identifier

    Returns:
        DeleteModelResponse: Deletion status

    Raises:
        HTTPException: If model not found
    """
    try:
        result = await training_service.delete_model(model_id)

        return DeleteModelResponse(success=True, message=result["message"], model_id=model_id)

    except AppException:
        raise
    except Exception as e:
        logger.error("Unexpected error deleting model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
