"""
Unit tests for backend/services/weights_service.py
"""
import pytest
from unittest.mock import Mock

from backend.services.weights_service import WeightsService, weights_service
from backend.core.exceptions import (
    ModelNotFoundException,
    ValidationException,
)


class TestWeightsService:
    """Tests for WeightsService business logic."""

    @pytest.fixture
    def service(self):
        """WeightsService instance."""
        return WeightsService()

    @pytest.fixture
    def mock_repository(self, mocker):
        """Mock SessionRepository."""
        mock_repo = mocker.Mock()
        return mock_repo

    @pytest.fixture
    def mock_recommender(self, mocker):
        """Mock CausalGraphRecommender."""
        recommender = mocker.Mock()
        recommender.get_weights.return_value = {
            'readiness': 0.6,
            'bayesian': 0.3,
            'utility': 0.1
        }
        recommender.set_weights = mocker.Mock()
        return recommender

    @pytest.mark.asyncio
    async def test_update_weights_success(self, service, mock_repository, mock_recommender):
        """Test updating weights successfully."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        new_weights = {'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.2}
        
        result = await service.update_weights(
            model_id="model_123",
            weights=new_weights
        )
        
        assert result == new_weights
        mock_repository.get_model.assert_called_once_with("model_123")
        mock_recommender.set_weights.assert_called_once_with(new_weights)

    @pytest.mark.asyncio
    async def test_update_weights_model_not_found(self, service, mock_repository):
        """Test updating weights for non-existent model raises ModelNotFoundException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = None
        
        with pytest.raises(ModelNotFoundException) as exc_info:
            await service.update_weights(
                model_id="nonexistent_model",
                weights={'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.2}
            )
        
        assert "nonexistent_model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_weights_validation_error(self, service, mock_repository, mock_recommender, mocker):
        """Test updating weights with invalid sum raises ValidationException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        # Mock validate_weights to raise exception
        mocker.patch(
            'backend.services.weights_service.validate_weights',
            side_effect=ValidationException("Weights must sum to 1.0")
        )
        
        invalid_weights = {'readiness': 0.5, 'bayesian': 0.5, 'utility': 0.5}
        
        with pytest.raises(ValidationException) as exc_info:
            await service.update_weights(
                model_id="model_123",
                weights=invalid_weights
            )
        
        assert "sum to 1.0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_weights_success(self, service, mock_repository, mock_recommender):
        """Test getting current weights successfully."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        result = await service.get_weights("model_123")
        
        assert result["readiness"] == 0.6
        assert result["bayesian"] == 0.3
        assert result["utility"] == 0.1
        mock_repository.get_model.assert_called_once_with("model_123")

    @pytest.mark.asyncio
    async def test_get_weights_model_not_found(self, service, mock_repository):
        """Test getting weights for non-existent model raises ModelNotFoundException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = None
        
        with pytest.raises(ModelNotFoundException) as exc_info:
            await service.get_weights("nonexistent_model")
        
        assert "nonexistent_model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_weights_fallback_to_attribute(self, service, mock_repository, mocker):
        """Test update_weights falls back to weights attribute if set_weights doesn't exist."""
        service.repository = mock_repository
        
        # Create recommender without set_weights method
        recommender = mocker.Mock(spec=['get_weights', 'weights'])
        recommender.weights = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
        recommender.get_weights.return_value = recommender.weights
        
        # Remove set_weights method
        del recommender.set_weights
        
        mock_repository.get_model.return_value = recommender
        
        new_weights = {'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.2}
        
        result = await service.update_weights(
            model_id="model_123",
            weights=new_weights
        )
        
        assert result == new_weights
        assert recommender.weights == new_weights


class TestWeightsServiceSingleton:
    """Test WeightsService singleton instance."""

    def test_singleton_instance_exists(self):
        """Test that weights_service singleton exists."""
        assert weights_service is not None
        assert isinstance(weights_service, WeightsService)
