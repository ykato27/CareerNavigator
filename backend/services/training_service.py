"""
Service layer for model training operations.

This module contains business logic for training causal inference models.
"""

import time
from typing import Dict, Any
from backend.repositories.session_repository import session_repository
from backend.core.exceptions import (
    SessionNotFoundException,
    ModelNotFoundException,
    InsufficientDataException,
)
from backend.core.logging import get_logger
from backend.utils.common import load_and_transform_session_data, DEFAULT_WEIGHTS, validate_weights
from skillnote_recommendation.ml.causal_graph_recommender import CausalGraphRecommender

logger = get_logger(__name__)


class TrainingService:
    """Service for handling model training operations."""

    def __init__(self):
        self.repository = session_repository

    async def train_model(
        self,
        session_id: str,
        min_members_per_skill: int = 5,
        correlation_threshold: float = 0.2,
        weights: Dict[str, float] | None = None,
        apply_constraints: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a causal inference model.

        Args:
            session_id: Session ID containing uploaded data
            min_members_per_skill: Minimum members required per skill
            correlation_threshold: Minimum correlation threshold
            weights: Optional custom recommendation weights

        Returns:
            Dictionary containing model_id and training summary

        Raises:
            SessionNotFoundException: If session not found
            InsufficientDataException: If data is insufficient for training
        """
        logger.info("Starting model training", session_id=session_id)
        start_time = time.time()

        # Validate session exists
        if not self.repository.session_exists(session_id):
            raise SessionNotFoundException(session_id)

        # Load and transform data
        try:
            transformed_data = load_and_transform_session_data(session_id)
        except FileNotFoundError as e:
            raise SessionNotFoundException(session_id) from e

        # Validate and prepare weights
        final_weights = weights or DEFAULT_WEIGHTS.copy()
        if weights:
            validate_weights(final_weights)

        logger.info("Data loaded and transformed, starting model fit")
        train_start = time.time()

        # Create and train recommender
        try:
            recommender = CausalGraphRecommender(
                member_competence=transformed_data["member_competence"],
                competence_master=transformed_data["competence_master"],
                learner_params={
                    "correlation_threshold": correlation_threshold,
                    "min_cluster_size": 3,
                },
                weights=final_weights,
            )

            recommender.fit(min_members_per_skill=min_members_per_skill)
        except Exception as e:
            logger.error("Model training failed", error=str(e), exc_info=True)
            raise InsufficientDataException(operation="model training", reason=str(e)) from e

        learning_time = time.time() - train_start
        total_time = time.time() - start_time

        logger.info("Model training completed", learning_time_sec=round(learning_time, 2))

        # Generate unique model ID
        model_id = f"model_{session_id}_{int(time.time())}"

        # Prepare training summary (flattened structure for test compatibility)
        summary = {
            "model_id": model_id,
            "session_id": session_id,
            "num_members": len(recommender.skill_matrix_.index),
            "num_skills": len(recommender.skill_matrix_.columns),
            "learning_time": round(learning_time, 2),
            "total_time": round(total_time, 2),
            "weights": final_weights,
            "min_members_per_skill": min_members_per_skill,
            "correlation_threshold": correlation_threshold,
        }
        
        # Prepare metadata for disk persistence
        metadata = {
            "model_id": model_id,
            "session_id": session_id,
            "num_members": len(recommender.skill_matrix_.index),
            "num_skills": len(recommender.skill_matrix_.columns),
            "weights": final_weights,
            "created_at": time.time(),
            "min_members_per_skill": min_members_per_skill,
            "correlation_threshold": correlation_threshold,
        }

        # Apply constraints if enabled
        constraints_applied = 0
        if apply_constraints:
            from backend.utils.constraint_manager import constraint_manager
            constraints = constraint_manager.load_constraints(session_id)
            if constraints:
                recommender.learner.apply_constraints(constraints)
                constraints_applied = len(constraints)
                logger.info(f"Applied {constraints_applied} constraints to model")

        # Store trained model (will save to disk automatically)
        self.repository.add_model(model_id, recommender)

        logger.info("Model stored successfully", model_id=model_id)

        return {
            "model_id": model_id,
            "summary": summary,
            "message": "因果構造の学習が完了しました",
        }

    async def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get summary information about a trained model.

        Args:
            model_id: Model identifier

        Returns:
            Model summary dictionary

        Raises:
            ModelNotFoundException: If model not found
        """
        recommender = self.repository.get_model(model_id)

        if not recommender:
            raise ModelNotFoundException(model_id)

        # Get weights (try method first, then attribute)
        weights = (
            recommender.get_weights()
            if hasattr(recommender, "get_weights")
            else recommender.weights
        )

        return {
            "model_id": model_id,
            "num_members": len(recommender.skill_matrix_.index),
            "num_skills": len(recommender.skill_matrix_.columns),
            "weights": weights,
            "has_causal_graph": hasattr(recommender, "adj_matrix_"),
        }

    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a trained model.

        Args:
            model_id: Model identifier

        Returns:
            Deletion confirmation dictionary

        Raises:
            ModelNotFoundException: If model not found
        """
        removed = self.repository.remove_model(model_id)

        if not removed:
            raise ModelNotFoundException(model_id)

        logger.info("Model deleted", model_id=model_id)

        return {"model_id": model_id, "message": f"Model {model_id} deleted successfully"}


# Singleton instance
training_service = TrainingService()
