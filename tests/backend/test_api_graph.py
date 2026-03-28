"""
Tests for backend/api/graph.py
"""
import pytest


class TestGetEgoNetwork:
    """Tests for /api/graph/ego endpoint."""
    
    def test_get_ego_network_success(self, client, mock_trained_model, session_manager_reset):
        """Test generating ego network graph successfully."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/graph/ego",
            json={
                "model_id": model_id,
                "center_node": "Python",
                "radius": 1,
                "min_edge_weight": 0.01
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "html_content" in data or "graph" in data
    
    def test_get_ego_network_model_not_found(self, client):
        """Test ego network with non-existent model."""
        response = client.post(
            "/api/graph/ego",
            json={
                "model_id": "nonexistent",
                "center_node": "Python",
                "radius": 1
            }
        )
        
        assert response.status_code == 404


class TestGetFullGraph:
    """Tests for /api/graph/full endpoint."""
    
    def test_get_full_graph_success(self, client, mock_trained_model, session_manager_reset):
        """Test generating full causal graph successfully."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/api/graph/full",
            json={
                "model_id": model_id,
                "min_edge_weight": 0.01,
                "layout": "spring"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "html_content" in data or "graph" in data
    
    def test_get_full_graph_model_not_found(self, client):
        """Test full graph with non-existent model."""
        response = client.post(
            "/api/graph/full",
            json={
                "model_id": "nonexistent",
                "min_edge_weight": 0.01
            }
        )
        
        assert response.status_code == 404
