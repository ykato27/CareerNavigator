from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import time
import logging

from backend.utils import (
    load_and_transform_session_data,
    DEFAULT_WEIGHTS,
    validate_weights,
    session_manager
)
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender

router = APIRouter()
logger = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    session_id: str
    min_members_per_skill: int = 5
    correlation_threshold: float = 0.2
    weights: Optional[Dict[str, float]] = None


class TrainResponse(BaseModel):
    success: bool
    model_id: str
    summary: Dict[str, Any]
    message: str


@router.post("/train", response_model=TrainResponse)
async def train_causal_model(request: TrainRequest):
    """
    Train a causal model using LiNGAM algorithm.

    Args:
        request: Training parameters including session_id, min_members_per_skill,
                correlation_threshold, and optional custom weights

    Returns:
        TrainResponse: Model ID, training summary, and success message

    Raises:
        HTTPException: If session not found or training fails
    """
    try:
        logger.info(f"[TRAIN] Starting model training for session {request.session_id}")
        start_time = time.time()

        # Load and transform data (with caching)
        transformed_data = load_and_transform_session_data(request.session_id)

        # Validate and prepare weights
        weights = request.weights or DEFAULT_WEIGHTS.copy()
        if request.weights:
            validate_weights(weights)

        logger.info(f"[TRAIN] Data loaded and transformed, starting model fit...")
        train_start = time.time()

        # Create and train recommender
        recommender = CausalGraphRecommender(
            member_competence=transformed_data["member_competence"],
            competence_master=transformed_data["competence_master"],
            learner_params={
                "correlation_threshold": request.correlation_threshold,
                "min_cluster_size": 3
            },
            weights=weights
        )

        recommender.fit(min_members_per_skill=request.min_members_per_skill)

        learning_time = time.time() - train_start
        total_time = time.time() - start_time

        logger.info(f"[TRAIN] Model training completed in {learning_time:.2f}s")

        # Generate unique model ID
        model_id = f"model_{request.session_id}_{int(time.time())}"

        # Store trained model using session manager
        session_manager.add_model(model_id, recommender)

        # Prepare training summary
        summary = {
            "num_members": len(recommender.skill_matrix_.index),
            "num_skills": len(recommender.skill_matrix_.columns),
            "learning_time": round(learning_time, 2),
            "total_time": round(total_time, 2),
            "weights": weights,
            "parameters": {
                "min_members_per_skill": request.min_members_per_skill,
                "correlation_threshold": request.correlation_threshold
            }
        }

        logger.info(f"[TRAIN] Model {model_id} stored successfully")

        return TrainResponse(
            success=True,
            model_id=model_id,
            summary=summary,
            message="因果構造の学習が完了しました"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except FileNotFoundError as e:
        logger.error(f"[TRAIN] Data file not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"[TRAIN] Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/model/{model_id}/summary")
async def get_model_summary(model_id: str):
    """
    Get summary information about a trained model.

    Args:
        model_id: The unique model identifier

    Returns:
        dict: Model summary including dimensions and weights

    Raises:
        HTTPException: If model not found
    """
    recommender = session_manager.get_model(model_id)

    if not recommender:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Please train a model first."
        )

    # Get weights (try method first, then attribute)
    weights = (
        recommender.get_weights()
        if hasattr(recommender, 'get_weights')
        else recommender.weights
    )

    return {
        "model_id": model_id,
        "num_members": len(recommender.skill_matrix_.index),
        "num_skills": len(recommender.skill_matrix_.columns),
        "weights": weights,
        "has_causal_graph": hasattr(recommender, 'adj_matrix_')
    }


@router.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a trained model from memory.

    Args:
        model_id: The unique model identifier

    Returns:
        dict: Deletion status

    Raises:
        HTTPException: If model not found
    """
    removed = session_manager.remove_model(model_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )

    return {
        "success": True,
        "message": f"Model {model_id} deleted successfully"
    }
