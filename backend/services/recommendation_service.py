"""
Service layer for recommendation operations.

This module contains business logic for generating skill recommendations.
"""

from typing import Dict, Any
from backend.repositories.session_repository import session_repository
from backend.core.exceptions import ModelNotFoundException, MemberNotFoundException
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """Service for handling recommendation operations."""

    def __init__(self):
        self.repository = session_repository

    async def get_recommendations(
        self, model_id: str, member_id: str, top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Get skill recommendations for a member.

        Args:
            model_id: Trained model identifier
            member_id: Member code to get recommendations for
            top_n: Number of recommendations to return

        Returns:
            Dictionary containing recommendations and metadata

        Raises:
            ModelNotFoundException: If model not found
            MemberNotFoundException: If member not found in model data
        """
        logger.info("Getting recommendations", model_id=model_id, member_id=member_id, top_n=top_n)

        # Get trained model
        recommender = self.repository.get_model(model_id)
        if not recommender:
            raise ModelNotFoundException(model_id)

        # Check if member exists
        if member_id not in recommender.skill_matrix_.index:
            raise MemberNotFoundException(member_id)

        # Generate recommendations
        try:
            recommendations_list = recommender.recommend(member_code=member_id, top_n=top_n)
        except Exception as e:
            logger.error(
                "Recommendation generation failed",
                model_id=model_id,
                member_id=member_id,
                error=str(e),
            )
            recommendations_list = []

        # Format recommendations
        formatted_recommendations = []
        for rec in recommendations_list:
            formatted_recommendations.append(
                {
                    "skill_code": rec.get("skill_code", ""),
                    "skill_name": rec.get("skill_name", ""),
                    "category": rec.get("category", ""),
                    "readiness_score": rec.get("readiness_score", 0.0),
                    "probability_score": rec.get("probability_score", 0.0),
                    "utility_score": rec.get("utility_score", 0.0),
                    "final_score": rec.get("final_score", 0.0),
                    "reason": rec.get("reason", ""),
                    "dependencies": rec.get("dependencies", []),
                }
            )

        # Get current weights
        weights = (
            recommender.get_weights()
            if hasattr(recommender, "get_weights")
            else recommender.weights
        )

        logger.info(
            "Recommendations generated",
            model_id=model_id,
            member_id=member_id,
            count=len(formatted_recommendations),
        )

        return {
            "model_id": model_id,
            "member_id": member_id,
            "member_name": "",  # Could be enriched from session data
            "recommendations": formatted_recommendations,
            "metadata": {
                "weights": weights,
                "total_candidates": len(formatted_recommendations),
            },
            "message": (
                "推薦を生成しました" if formatted_recommendations else "利用可能な推薦がありません"
            ),
        }


# Singleton instance
recommendation_service = RecommendationService()
