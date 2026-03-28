"""
Service layer for recommendation weights management.

This module contains business logic for managing recommendation weights.
"""

from typing import Dict
from backend.repositories.session_repository import session_repository
from backend.core.exceptions import ModelNotFoundException, InvalidWeightsException
from backend.core.logging import get_logger
from backend.utils.common import validate_weights

logger = get_logger(__name__)


class WeightsService:
    """Service for handling recommendation weights operations."""

    def __init__(self):
        self.repository = session_repository

    async def update_weights(self, model_id: str, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update recommendation weights for a model.

        Args:
            model_id: Model identifier
            weights: New weights dictionary

        Returns:
            Updated weights dictionary

        Raises:
            ModelNotFoundException: If model not found
            InvalidWeightsException: If weights are invalid
        """
        logger.info("Updating weights", model_id=model_id, weights=weights)

        # Get trained model
        recommender = self.repository.get_model(model_id)
        if not recommender:
            raise ModelNotFoundException(model_id)

        # Validate weights
        try:
            validate_weights(weights)
        except ValueError as e:
            raise InvalidWeightsException(str(e)) from e

        # Update weights (use method if available, otherwise set attribute)
        if hasattr(recommender, "set_weights"):
            recommender.set_weights(weights)
        else:
            recommender.weights = weights

        logger.info("Weights updated successfully", model_id=model_id)

        return weights

    async def get_weights(self, model_id: str) -> Dict[str, float]:
        """
        Get current recommendation weights for a model.

        Args:
            model_id: Model identifier

        Returns:
            Current weights dictionary

        Raises:
            ModelNotFoundException: If model not found
        """
        # Get trained model
        recommender = self.repository.get_model(model_id)
        if not recommender:
            raise ModelNotFoundException(model_id)

        # Get weights (try method first, then attribute)
        weights = (
            recommender.get_weights()
            if hasattr(recommender, "get_weights")
            else recommender.weights
        )

        logger.info("Retrieved weights", model_id=model_id, weights=weights)

        return weights

    async def optimize_weights(
        self,
        model_id: str,
        n_trials: int = 50,
        n_jobs: int = -1,
        holdout_ratio: float = 0.2,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Optimize recommendation weights for a model using Bayesian optimization.

        Args:
            model_id: Model identifier
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs (-1 for all cores)
            holdout_ratio: Ratio of skills to holdout for evaluation
            top_k: Number of recommendations to evaluate

        Returns:
            Optimized weights dictionary

        Raises:
            ModelNotFoundException: If model not found
        """
        logger.info(
            "Optimizing weights",
            model_id=model_id,
            n_trials=n_trials,
            n_jobs=n_jobs,
            holdout_ratio=holdout_ratio,
            top_k=top_k
        )

        # Get trained model
        recommender = self.repository.get_model(model_id)
        if not recommender:
            raise ModelNotFoundException(model_id)

        # Check if model has optimize_weights method
        if not hasattr(recommender, "optimize_weights"):
            logger.warning(
                "Model does not support weight optimization. Using current weights.",
                model_id=model_id
            )
            # Return current weights instead of failing
            return await self.get_weights(model_id)

        # Run optimization
        try:
            optimized_weights = recommender.optimize_weights(
                n_trials=n_trials,
                n_jobs=n_jobs,
                holdout_ratio=holdout_ratio,
                top_k=top_k
            )

            logger.info(
                "Weights optimized successfully",
                model_id=model_id,
                optimized_weights=optimized_weights
            )

            return optimized_weights

        except Exception as e:
            logger.error(
                "Failed to optimize weights",
                model_id=model_id,
                error=str(e),
                exc_info=True
            )
            raise


# Singleton instance
weights_service = WeightsService()
