"""
Tests for backend/api/organizational.py
"""
import pytest


class TestGetOrganizationalMetrics:
    """Tests for /api/organizational/metrics endpoint."""
    
    def test_get_metrics_success(self, client, mock_session_data, session_manager_reset):
        """Test getting organizational metrics successfully."""
        response = client.post(
            "/api/organizational/metrics",
            json={"session_id": mock_session_data}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "metrics" in data
        assert "total_members" in data["metrics"]
        assert "total_skills" in data["metrics"]
    
    def test_get_metrics_session_not_found(self, client):
        """Test metrics with non-existent session."""
        response = client.post(
            "/api/organizational/metrics",
            json={"session_id": "nonexistent"}
        )
        
        assert response.status_code == 404 or response.status_code == 500


class TestAnalyzeSkillGap:
    """Tests for /api/organizational/skill-gap endpoint."""
    
    def test_analyze_skill_gap_success(self, client, mock_session_data, session_manager_reset):
        """Test skill gap analysis successfully."""
        response = client.post(
            "/api/organizational/skill-gap",
            json={
                "session_id": mock_session_data,
                "percentile": 0.2
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "gap_analysis" in data
        assert "critical_skills" in data
    
    def test_analyze_skill_gap_session_not_found(self, client):
        """Test skill gap with non-existent session."""
        response = client.post(
            "/api/organizational/skill-gap",
            json={
                "session_id": "nonexistent",
                "percentile": 0.2
            }
        )
        
        assert response.status_code == 404 or response.status_code == 500


class TestFindSuccessionCandidates:
    """Tests for /api/organizational/succession endpoint."""
    
    def test_find_succession_candidates_success(self, client, mock_session_data, session_manager_reset):
        """Test finding succession candidates successfully."""
        response = client.post(
            "/api/organizational/succession",
            json={
                "session_id": mock_session_data,
                "target_position": "部長",
                "max_candidates": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "candidates" in data
    
    def test_find_succession_session_not_found(self, client):
        """Test succession with non-existent session."""
        response = client.post(
            "/api/organizational/succession",
            json={
                "session_id": "nonexistent",
                "target_position": "部長"
            }
        )
        
        assert response.status_code == 404 or response.status_code == 500


class TestSimulateOrganization:
    """Tests for /api/organizational/simulate endpoint."""
    
    def test_simulate_organization_success(self, client, mock_session_data, session_manager_reset):
        """Test organization simulation successfully."""
        response = client.post(
            "/api/organizational/simulate",
            json={
                "session_id": mock_session_data,
                "transfers": [
                    {"member_code": "M001", "from_group": "開発", "to_group": "営業"}
                ],
                "group_column": "職種"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "comparison" in data
        assert "balance_scores" in data
    
    def test_simulate_session_not_found(self, client):
        """Test simulation with non-existent session."""
        response = client.post(
            "/api/organizational/simulate",
            json={
                "session_id": "nonexistent",
                "transfers": [],
                "group_column": "職種"
            }
        )
        
        assert response.status_code == 404 or response.status_code == 500
