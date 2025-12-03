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


# Singleton instance
weights_service = WeightsService()
