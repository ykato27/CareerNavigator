"""
Unit tests for backend/services/training_service.py
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from backend.services.training_service import TrainingService, training_service
from backend.core.exceptions import (
    SessionNotFoundException,
    ModelNotFoundException,
    InsufficientDataException,
)


class TestTrainingService:
    """Tests for TrainingService business logic."""

    @pytest.fixture
    def service(self):
        """TrainingService instance."""
        return TrainingService()

    @pytest.fixture
    def mock_repository(self, mocker):
        """Mock SessionRepository."""
        mock_repo = mocker.Mock()
        mock_repo.session_exists.return_value = True
        mock_repo.add_model = mocker.Mock()
        mock_repo.get_model.return_value = None
        mock_repo.remove_model.return_value = True
        return mock_repo

    @pytest.fixture
    def mock_recommender(self, mocker):
        """Mock CausalGraphRecommender."""
        recommender = mocker.Mock()
        recommender.skill_matrix_ = pd.DataFrame({
            'skill1': [1, 2, 3],
            'skill2': [4, 5, 6]
        })
        recommender.get_weights.return_value = {
            'readiness': 0.6,
            'bayesian': 0.3,
            'utility': 0.1
        }
        recommender.weights = {'readiness': 0.6, 'bayesian': 0.3, 'utility': 0.1}
        recommender.adj_matrix_ = pd.DataFrame()
        return recommender

    @pytest.mark.asyncio
    async def test_train_model_success(self, service, mock_repository, mocker):
        """Test successful model training."""
        # Mock dependencies
        service.repository = mock_repository
        
        mock_transform_data = mocker.patch(
            'backend.services.training_service.load_and_transform_session_data',
            return_value={
                'member_competence': pd.DataFrame(),
                'competence_master': pd.DataFrame()
            }
        )
        
        mock_recommender = Mock()
        mock_recommender.skill_matrix_ = pd.DataFrame([
            [1, 2, 3],
            [4, 5, 6]
        ], columns=['skill1', 'skill2', 'skill3'])
        
        mock_cgr = mocker.patch(
            'backend.services.training_service.CausalGraphRecommender',
            return_value=mock_recommender
        )
        
        # Execute
        result = await service.train_model(
            session_id="test_session_001",
            min_members_per_skill=5,
            correlation_threshold=0.2
        )
        
        # Assert
        assert "model_id" in result
        assert "summary" in result
        assert "message" in result
        assert result["summary"]["num_members"] == 2
        assert result["summary"]["num_skills"] == 3
        mock_repository.session_exists.assert_called_once_with("test_session_001")
        mock_repository.add_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_model_session_not_found(self, service, mock_repository):
        """Test training with non-existent session raises SessionNotFoundException."""
        service.repository = mock_repository
        mock_repository.session_exists.return_value = False
        
        with pytest.raises(SessionNotFoundException) as exc_info:
            await service.train_model(
                session_id="nonexistent_session",
                min_members_per_skill=5
            )
        
        assert "nonexistent_session" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_train_model_with_custom_weights(self, service, mock_repository, mocker):
        """Test training with custom weights."""
        service.repository = mock_repository
        
        mocker.patch(
            'backend.services.training_service.load_and_transform_session_data',
            return_value={
                'member_competence': pd.DataFrame(),
                'competence_master': pd.DataFrame()
            }
        )
        
        mock_recommender = Mock()
        mock_recommender.skill_matrix_ = pd.DataFrame([[1, 2]], columns=['s1', 's2'])
        
        mocker.patch(
            'backend.services.training_service.CausalGraphRecommender',
            return_value=mock_recommender
        )
        
        custom_weights = {'readiness': 0.5, 'bayesian': 0.3, 'utility': 0.2}
        
        result = await service.train_model(
            session_id="test_session_001",
            weights=custom_weights
        )
        
        assert result["summary"]["weights"] == custom_weights

    @pytest.mark.asyncio
    async def test_train_model_insufficient_data(self, service, mock_repository, mocker):
        """Test training with insufficient data raises InsufficientDataException."""
        service.repository = mock_repository
        
        mocker.patch(
            'backend.services.training_service.load_and_transform_session_data',
            return_value={
                'member_competence': pd.DataFrame(),
                'competence_master': pd.DataFrame()
            }
        )
        
        # Mock CausalGraphRecommender.fit to raise exception
        mock_recommender = Mock()
        mock_recommender.fit.side_effect = ValueError("Insufficient data")
        
        mocker.patch(
            'backend.services.training_service.CausalGraphRecommender',
            return_value=mock_recommender
        )
        
        with pytest.raises(InsufficientDataException):
            await service.train_model(session_id="test_session_001")

    @pytest.mark.asyncio
    async def test_get_model_summary_success(self, service, mock_repository, mock_recommender):
        """Test getting model summary successfully."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = mock_recommender
        
        result = await service.get_model_summary("model_123")
        
        assert result["model_id"] == "model_123"
        assert "num_members" in result
        assert "num_skills" in result
        assert "weights" in result
        assert result["num_members"] == 3
        assert result["num_skills"] == 2

    @pytest.mark.asyncio
    async def test_get_model_summary_not_found(self, service, mock_repository):
        """Test getting summary for non-existent model raises ModelNotFoundException."""
        service.repository = mock_repository
        mock_repository.get_model.return_value = None
        
        with pytest.raises(ModelNotFoundException) as exc_info:
            await service.get_model_summary("nonexistent_model")
        
        assert "nonexistent_model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_model_success(self, service, mock_repository):
        """Test deleting a model successfully."""
        service.repository = mock_repository
        mock_repository.remove_model.return_value = True
        
        result = await service.delete_model("model_123")
        
        assert result["model_id"] == "model_123"
        assert "message" in result
        mock_repository.remove_model.assert_called_once_with("model_123")

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, service, mock_repository):
        """Test deleting non-existent model raises ModelNotFoundException."""
        service.repository = mock_repository
        mock_repository.remove_model.return_value = False
        
        with pytest.raises(ModelNotFoundException) as exc_info:
            await service.delete_model("nonexistent_model")
        
        assert "nonexistent_model" in str(exc_info.value)


class TestTrainingServiceSingleton:
    """Test TrainingService singleton instance."""

    def test_singleton_instance_exists(self):
        """Test that training_service singleton exists."""
        assert training_service is not None
        assert isinstance(training_service, TrainingService)
