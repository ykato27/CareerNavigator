"""
Unit tests for backend/schemas/response/*.py
"""
import pytest
from pydantic import ValidationError

from backend.schemas.response.training import (
    ModelSummary,
    TrainModelResponse,
    GetModelSummaryResponse,
    DeleteModelResponse
)
from backend.schemas.response.weights import WeightsResponse
from backend.schemas.response.recommendation import RecommendationsResponse
from backend.schemas.response.common import ErrorResponse


class TestModelSummary:
    """Tests for ModelSummary schema."""

    def test_valid_summary(self):
        """Test valid model summary."""
        data = {
            "model_id": "model_123",
            "session_id": "session_001",
            "num_members": 100,
            "num_skills": 50,
            "learning_time": 2.5,
            "total_time": 5.0,
            "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1},
            "min_members_per_skill": 5,
            "correlation_threshold": 0.2
        }
        summary = ModelSummary(**data)
        
        assert summary.model_id == "model_123"
        assert summary.num_members == 100
        assert summary.weights["readiness"] == 0.6


class TestTrainModelResponse:
    """Tests for TrainModelResponse schema."""

    def test_valid_response(self):
        """Test valid train model response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "summary": {
                "model_id": "model_123",
                "session_id": "session_001",
                "num_members": 100,
                "num_skills": 50,
                "learning_time": 2.5,
                "total_time": 5.0,
                "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1},
                "min_members_per_skill": 5,
                "correlation_threshold": 0.2
            },
            "message": "Model trained successfully"
        }
        response = TrainModelResponse(**data)
        
        assert response.success is True
        assert response.model_id == "model_123"
        assert isinstance(response.summary, ModelSummary)


class TestGetModelSummaryResponse:
    """Tests for GetModelSummaryResponse schema."""

    def test_valid_response(self):
        """Test valid get model summary response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "num_members": 100,
            "num_skills": 50,
            "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1},
            "has_causal_graph": True
        }
        response = GetModelSummaryResponse(**data)
        
        assert response.success is True
        assert response.model_id == "model_123"
        assert response.num_members == 100


class TestDeleteModelResponse:
    """Tests for DeleteModelResponse schema."""

    def test_valid_response(self):
        """Test valid delete model response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "message": "Model deleted successfully"
        }
        response = DeleteModelResponse(**data)
        
        assert response.success is True
        assert response.model_id == "model_123"


class TestWeightsResponse:
    """Tests for WeightsResponse schema."""

    def test_valid_response(self):
        """Test valid weights response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "weights": {"readiness": 0.5, "bayesian": 0.3, "utility": 0.2}
        }
        response = WeightsResponse(**data)
        
        assert response.success is True
        assert response.weights["readiness"] == 0.5

    def test_valid_get_weights(self):
        """Test valid get weights response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1}
        }
        response = WeightsResponse(**data)
        
        assert response.success is True
        assert response.weights["utility"] == 0.1


class TestGetRecommendationsResponse:
    """Tests for GetRecommendationsResponse schema."""

    def test_valid_response(self):
        """Test valid get recommendations response."""
        data = {
            "success": True,
            "model_id": "model_123",
            "member_id": "M001",
            "member_name": "John Doe",
            "recommendations": [
                {
                    "skill_code": "Python",
                    "skill_name": "Python Programming",
                    "category": "Programming",
                    "readiness_score": 0.8,
                    "probability_score": 0.9,
                    "utility_score": 0.85,
                    "final_score": 0.85,
                    "reason": "High utility",
                    "dependencies": ["SQL"]
                }
            ],
            "metadata": {
                "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1},
                "total_candidates": 1
            },
            "message": "Recommendations generated"
        }
        response = RecommendationsResponse(**data)
        
        assert response.success is True
        assert response.model_id == "model_123"
        assert len(response.recommendations) == 1
        assert response.recommendations[0].skill_code == "Python"

    def test_valid_empty_recommendations(self):
        """Test valid response with empty recommendations."""
        data = {
            "success": True,
            "model_id": "model_123",
            "member_id": "M001",
            "member_name": "",
            "recommendations": [],
            "metadata": {
                "weights": {"readiness": 0.6, "bayesian": 0.3, "utility": 0.1},
                "total_candidates": 0
            },
            "message": "No recommendations available"
        }
        response = RecommendationsResponse(**data)
        
        assert response.recommendations == []


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_valid_error_response(self):
        """Test valid error response."""
        data = {
            "error": {
                "code": "MODEL_NOT_FOUND",
                "message": "Model not found"
            }
        }
        response = ErrorResponse(**data)
        
        assert response.error.code == "MODEL_NOT_FOUND"
        assert response.error.message == "Model not found"

    def test_valid_error_response_with_details(self):
        """Test valid error response with details."""
        data = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid weights",
                "details": {"field": "weights", "issue": "sum > 1.0"}
            }
        }
        response = ErrorResponse(**data)
        
        assert response.error.details["field"] == "weights"
