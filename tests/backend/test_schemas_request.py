"""
Unit tests for backend/schemas/request/*.py
"""
import pytest
from pydantic import ValidationError

from backend.schemas.request.training import TrainModelRequest
from backend.schemas.request.weights import UpdateWeightsRequest
from backend.schemas.request.recommendation import GetRecommendationsRequest


class TestTrainModelRequest:
    """Tests for TrainModelRequest schema validation."""

    def test_valid_request_minimal(self):
        """Test valid request with minimal required fields."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5
        }
        request = TrainModelRequest(**data)
        
        assert request.session_id == "test_session_001"
        assert request.min_members_per_skill == 5
        assert request.correlation_threshold == 0.2  # default
        assert request.weights is None

    def test_valid_request_with_custom_weights(self):
        """Test valid request with custom weights."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 3,
            "weights": {
                "readiness": 0.5,
                "bayesian": 0.3,
                "utility": 0.2
            }
        }
        request = TrainModelRequest(**data)
        
        assert request.weights["readiness"] == 0.5
        assert request.weights["bayesian"] == 0.3
        assert request.weights["utility"] == 0.2

    def test_valid_request_all_fields(self):
        """Test valid request with all fields."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 10,
            "correlation_threshold": 0.3,
            "weights": {
                "readiness": 0.6,
                "bayesian": 0.3,
                "utility": 0.1
            }
        }
        request = TrainModelRequest(**data)
        
        assert request.session_id == "test_session_001"
        assert request.min_members_per_skill == 10
        assert request.correlation_threshold == 0.3

    def test_invalid_weights_sum_too_high(self):
        """Test that weights sum > 1.0 raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5,
            "weights": {
                "readiness": 0.5,
                "bayesian": 0.5,
                "utility": 0.5  # Sum = 1.5
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "sum to 1.0" in str(exc_info.value).lower()

    def test_invalid_weights_sum_too_low(self):
        """Test that weights sum < 1.0 raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5,
            "weights": {
                "readiness": 0.2,
                "bayesian": 0.2,
                "utility": 0.2  # Sum = 0.6
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "sum to 1.0" in str(exc_info.value).lower()

    def test_invalid_weights_missing_key(self):
        """Test that missing weight key raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5,
            "weights": {
                "readiness": 0.7,
                "bayesian": 0.3
                # Missing 'utility'
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "utility" in str(exc_info.value).lower() or "readiness" in str(exc_info.value).lower()

    def test_invalid_weights_extra_key(self):
        """Test that extra weight key raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5,
            "weights": {
                "readiness": 0.4,
                "bayesian": 0.3,
                "utility": 0.3,
                "extra_key": 0.0  # Extra key
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "extra" in str(exc_info.value).lower() or "exactly" in str(exc_info.value).lower()

    def test_invalid_min_members_negative(self):
        """Test that negative min_members_per_skill raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": -1
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value).lower()

    def test_invalid_correlation_threshold_out_of_range(self):
        """Test that correlation_threshold outside [0, 1] raises ValidationError."""
        data = {
            "session_id": "test_session_001",
            "min_members_per_skill": 5,
            "correlation_threshold": 1.5
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrainModelRequest(**data)
        
        assert "less than or equal to 1" in str(exc_info.value).lower()


class TestUpdateWeightsRequest:
    """Tests for UpdateWeightsRequest schema validation."""

    def test_valid_request(self):
        """Test valid weights update request."""
        data = {
            "model_id": "model_123",
            "weights": {
                "readiness": 0.5,
                "bayesian": 0.3,
                "utility": 0.2
            }
        }
        request = UpdateWeightsRequest(**data)
        
        assert request.model_id == "model_123"
        assert request.weights["readiness"] == 0.5
        assert request.weights["bayesian"] == 0.3
        assert request.weights["utility"] == 0.2

    def test_invalid_weights_sum(self):
        """Test that invalid weights sum raises ValidationError."""
        data = {
            "model_id": "model_123",
            "weights": {
                "readiness": 0.6,
                "bayesian": 0.6,
                "utility": 0.6
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UpdateWeightsRequest(**data)
        
        assert "sum to 1.0" in str(exc_info.value).lower()

    def test_invalid_missing_model_id(self):
        """Test that missing model_id raises ValidationError."""
        data = {
            "weights": {
                "readiness": 0.5,
                "bayesian": 0.3,
                "utility": 0.2
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UpdateWeightsRequest(**data)
        
        assert "model_id" in str(exc_info.value).lower()


class TestGetRecommendationsRequest:
    """Tests for GetRecommendationsRequest schema validation."""

    def test_valid_request_minimal(self):
        """Test valid request with minimal fields."""
        data = {
            "model_id": "model_123",
            "member_id": "M001"
        }
        request = GetRecommendationsRequest(**data)
        
        assert request.model_id == "model_123"
        assert request.member_id == "M001"
        assert request.top_n == 10  # default

    def test_valid_request_with_top_n(self):
        """Test valid request with custom top_n."""
        data = {
            "model_id": "model_123",
            "member_id": "M001",
            "top_n": 5
        }
        request = GetRecommendationsRequest(**data)
        
        assert request.top_n == 5

    def test_invalid_top_n_negative(self):
        """Test that negative top_n raises ValidationError."""
        data = {
            "model_id": "model_123",
            "member_id": "M001",
            "top_n": -1
        }
        
        with pytest.raises(ValidationError) as exc_info:
            GetRecommendationsRequest(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value).lower()

    def test_invalid_top_n_zero(self):
        """Test that zero top_n raises ValidationError."""
        data = {
            "model_id": "model_123",
            "member_id": "M001",
            "top_n": 0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            GetRecommendationsRequest(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value).lower()
