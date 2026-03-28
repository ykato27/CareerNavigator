"""
Tests for backend/api/recommendation.py
"""
import pytest
from fastapi import HTTPException


class TestGetRecommendations:
    """Tests for the /api/recommend endpoint."""
    
    def test_get_recommendations_success(self, client, mock_trained_model, session_manager_reset):
        """Test getting recommendations successfully."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/recommend",
            json={
                "model_id": model_id,
                "member_id": "M001",
                "top_n": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["member_id"] == "M001"
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
    
    def test_get_recommendations_model_not_found(self, client):
        """Test getting recommendations with non-existent model."""
        response = client.post(
            "/api/recommend",
            json={
                "model_id": "nonexistent_model",
                "member_id":  "M001",
                "top_n": 5
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_recommendations_member_not_found(self, client, mock_trained_model, session_manager_reset):
        """Test getting recommendations for non-existent member."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/recommend",
            json={
                "model_id": model_id,
                "member_id": "NONEXISTENT",
                "top_n": 5
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_recommendations_no_results(self, client, mock_trained_model, session_manager_reset):
        """Test when no recommendations are available."""
        model_id, recommender = mock_trained_model
        
        # Mock recommender to return empty list
        original_recommend = recommender.recommend
        recommender.recommend = lambda *args, **kwargs: []
        
        response = client.post(
            "/api/recommend",
            json={
                "model_id": model_id,
                "member_id": "M001",
                "top_n": 5
            }
        )
        
        # Restore original method
        recommender.recommend = original_recommend
        
        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"] == []
        assert "message" in data
    
    def test_get_recommendations_custom_top_n(self, client, mock_trained_model, session_manager_reset):
        """Test with custom top_n parameter."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/recommend",
            json={
                "model_id": model_id,
                "member_id": "M001",
                "top_n": 3
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should return at most 3 recommendations
        assert len(data["recommendations"]) <= 3
