"""
Unit tests for backend/services/recommendation_service.py
"""
import pytest
from unittest.mock import Mock
import pandas as pd

from backend.services.recommendation_service import RecommendationService, recommendation_service
from backend.core.exceptions import (
    ModelNotFoundException,
    MemberNotFoundException,
)


class TestRecommendationService:
    """Tests for RecommendationService business logic."""

    @pytest.fixture
    def service(self):
        """RecommendationService instance."""
        return RecommendationService()

    @pytest.fixture
    def mock_repository(self, mocker):
        """Mock SessionRepository."""
        mock_repo = mocker.Mock()
        return mock_repo

    @pytest.fixture
    def mock_recommender(self, mocker):
        """Mock CausalGraphRecommender with recommendations."""
        recommender = mocker.Mock()
        recommender.skill_matrix_ = pd.DataFrame({
            'Python': [3, 0, 2],
            'Java': [2, 3, 1],
            'SQL': [1, 2, 3]
        }, index=['M001', 'M002', 'M003'])
        
        # Mock recommend method - returns list of dictionaries
        recommender.recommend.return_value = [
            {
                'skill_code': 'Java',
                'skill_name': 'Java Programming',
                'category': 'Programming',
                'readiness_score': 0.8,
                'probability_score': 0.9,
                'utility_score': 0.85,
                'final_score': 0.85,
                'reason': 'High utility',
                'dependencies': []
            },
            {
                'skill_code': 'SQL',
                'skill_name': 'SQL Database',
                'category': 'Database',
                'readiness_score': 0.7,
                'probability_score': 0.75,
                'utility_score': 0.71,
                'final_score': 0.72,
                'reason': 'Good foundation',
                'dependencies': []
            }
        ]
        recommender.get_weights.return_value = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
        
        return recommender

    @pytest.mark.asyncio
    async def test_get_recommendations_success(self, service, mock_repository, mock_recommender):
        """Test getting recommendations successfully."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        result = await service.get_recommendations(
            model_id="model_123",
            member_id="M001",
            top_n=5
        )
        
        assert result["model_id"] == "model_123"
        assert result["member_id"] == "M001"
        assert "recommendations" in result
        assert "metadata" in result
        assert len(result["recommendations"]) == 2
        assert result["recommendations"][0]["skill_code"] == "Java"
        
        mock_repository.get_model.assert_called_once_with("model_123")
        mock_recommender.recommend.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recommendations_model_not_found(self, service, mock_repository):
        """Test getting recommendations for non-existent model raises ModelNotFoundException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = None
        
        with pytest.raises(ModelNotFoundException) as exc_info:
            await service.get_recommendations(
                model_id="nonexistent_model",
                member_id="M001"
            )
        
        assert "nonexistent_model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_recommendations_member_not_found(self, service, mock_repository, mock_recommender):
        """Test getting recommendations for non-existent member raises MemberNotFoundException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        with pytest.raises(MemberNotFoundException) as exc_info:
            await service.get_recommendations(
                model_id="model_123",
                member_id="NONEXISTENT"
            )
        
        assert "NONEXISTENT" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_recommendations_empty_result(self, service, mock_repository, mocker):
        """Test getting recommendations when no recommendations are available."""
        service.repository = mock_repository
        
        empty_recommender = mocker.Mock()
        empty_recommender.skill_matrix_ = pd.DataFrame({
            'Python': [3]
        }, index=['M001'])
        empty_recommender.recommend.return_value = []
        empty_recommender.get_weights.return_value = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
        
        mock_repository.get_model.return_value = empty_recommender
        
        result = await service.get_recommendations(
            model_id="model_123",
            member_id="M001"
        )
        
        assert result["recommendations"] == []

    @pytest.mark.asyncio
    async def test_get_recommendations_with_top_n(self, service, mock_repository, mocker):
        """Test getting recommendations with top_n parameter."""
        service.repository = mock_repository
        
        recommender = mocker.Mock()
        recommender.skill_matrix_ = pd.DataFrame({'s1': [1]}, index=['M001'])
        recommender.recommend.return_value = [
            {
                "skill_code": "s1",
                "skill_name": "Skill 1",
                "category": "Cat1",
                "readiness_score": 0.9,
                "probability_score": 0.9,
                "utility_score": 0.9,
                "final_score": 0.9,
                "reason": "",
                "dependencies": []
            },
            {
                "skill_code": "s2",
                "skill_name": "Skill 2",
                "category": "Cat2",
                "readiness_score": 0.8,
                "probability_score": 0.8,
                "utility_score": 0.8,
                "final_score": 0.8,
                "reason": "",
                "dependencies": []
            },
            {
                "skill_code": "s3",
                "skill_name": "Skill 3",
                "category": "Cat3",
                "readiness_score": 0.7,
                "probability_score": 0.7,
                "utility_score": 0.7,
                "final_score": 0.7,
                "reason": "",
                "dependencies": []
            }
        ]
        recommender.get_weights.return_value = {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1}
        
        mock_repository.get_model.return_value = recommender
        
        result = await service.get_recommendations(
            model_id="model_123",
            member_id="M001",
            top_n=3
        )
        
        # Verify recommend was called with correct parameters
        call_args = recommender.recommend.call_args
        assert call_args[1]['member_code'] == 'M001'
        assert call_args[1]['top_n'] == 3
        assert len(result["recommendations"]) == 3


class TestRecommendationServiceSingleton:
    """Test RecommendationService singleton instance."""

    def test_singleton_instance_exists(self):
        """Test that recommendation_service singleton exists."""
        assert recommendation_service is not None
        assert isinstance(recommendation_service, RecommendationService)
