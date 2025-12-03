"""
Tests for backend/api/train.py
"""
import pytest


class TestTrainCausalModel:
    """Tests for /api/train endpoint."""
    
    def test_train_model_success(self, client, mock_session_data, session_manager_reset):
        """Test successful model training."""
        response = client.post(
            "/api/train",
            json={
                "session_id": mock_session_data,
                "min_members_per_skill": 1,
                "correlation_threshold": 0.2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "model_id" in data
        assert "summary" in data
        assert data["summary"]["num_members"] > 0
        assert data["summary"]["num_skills"] > 0
    
    def test_train_model_session_not_found(self, client):
        """Test training with non-existent session."""
        response = client.post(
            "/api/train",
            json={
                "session_id": "nonexistent_session",
                "min_members_per_skill": 1
            }
        )
        
        assert response.status_code == 404
    
    def test_train_model_with_custom_weights(self, client, mock_session_data, session_manager_reset):
        """Test training with custom weights."""
        response = client.post(
            "/api/train",
            json={
                "session_id": mock_session_data,
                "min_members_per_skill": 1,
                "correlation_threshold": 0.2,
                "weights": {
                    "readiness": 0.5,
                    "bayesian": 0.3,
                    "utility": 0.2
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["weights"]["readiness"] == 0.5
    
    def test_train_model_invalid_weights(self, client, mock_session_data):
        """Test training with invalid weights."""
        response = client.post(
            "/api/train",
            json={
                "session_id": mock_session_data,
                "min_members_per_skill": 1,
                "weights": {
                    "readiness": 0.5,
                    "bayesian": 0.3,
                    "utility": 0.3  # Sum > 1.0
                }
            }
        )
        
        assert response.status_code == 422  # Pydantic validation error


class TestGetModelSummary:
    """Tests for /api/model/{model_id}/summary endpoint."""
    
    def test_get_model_summary_success(self, client, mock_trained_model, session_manager_reset):
        """Test getting model summary successfully."""
        model_id, _ = mock_trained_model
        
        response = client.get(f"/api/model/{model_id}/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == model_id
        assert "num_members" in data
        assert "num_skills" in data
        assert "weights" in data
    
    def test_get_model_summary_not_found(self, client):
        """Test getting summary for non-existent model."""
        response = client.get("/api/model/nonexistent_model/summary")
        
        assert response.status_code == 404


class TestDeleteModel:
    """Tests for /api/model/{model_id} DELETE endpoint."""
    
    def test_delete_model_success(self, client, mock_trained_model, session_manager_reset):
        """Test deleting a model successfully."""
        model_id, _ = mock_trained_model
        
        response = client.delete(f"/api/model/{model_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify model is deleted
        get_response = client.get(f"/api/model/{model_id}/summary")
        assert get_response.status_code == 404
    
    def test_delete_model_not_found(self, client):
        """Test deleting non-existent model."""
        response = client.delete("/api/model/nonexistent_model")
        
        assert response.status_code == 404
