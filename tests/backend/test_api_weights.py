"""
Tests for backend/api/weights.py
"""
import pytest


class TestUpdateWeights:
    """Tests for /api/weights/update endpoint."""
    
    def test_update_weights_success(self, client, mock_trained_model, session_manager_reset):
        """Test updating weights successfully."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/weights/update",
            json={
                "model_id": model_id,
                "weights": {
                    "readiness": 0.5,
                    "bayesian": 0.3,
                    "utility": 0.2
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert data["weights"]["readiness"] == 0.5
    
    def test_update_weights_model_not_found(self, client):
        """Test updating weights for non-existent model."""
        response = client.post(
            "/api/weights/update",
            json={
                "model_id": "nonexistent",
                "weights": {
                    "readiness": 0.5,
                    "bayesian": 0.3,
                    "utility": 0.2
                }
            }
        )
        
        assert response.status_code == 404
    
    def test_update_weights_invalid_sum(self, client, mock_trained_model):
        """Test updating with weights that don't sum to 1.0."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/weights/update",
            json={
                "model_id": model_id,
                "weights": {
                    "readiness": 0.5,
                    "bayesian": 0.3,
                    "utility": 0.3  # Sum = 1.1
                }
            }
        )
        
        assert response.status_code == 422  # Pydantic validation error


class TestGetWeights:
    """Tests for /api/weights/{model_id} endpoint."""
    
    def test_get_weights_success(self, client, mock_trained_model, session_manager_reset):
        """Test getting current weights."""
        model_id, _ = mock_trained_model
        
        response = client.get(f"/api/weights/{model_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert "readiness" in data["weights"]
    
    def test_get_weights_model_not_found(self, client):
        """Test getting weights for non-existent model."""
        response = client.get("/api/weights/nonexistent")
        
        assert response.status_code == 404
