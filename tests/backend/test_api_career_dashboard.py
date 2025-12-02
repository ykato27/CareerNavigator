"""
Tests for backend/api/career_dashboard.py
"""
import pytest


class TestGetAvailableMembers:
    """Tests for /members endpoint."""
    
    def test_get_available_members_success(self, client, mock_session_data, session_manager_reset):
        """Test getting available members successfully."""
        response = client.get(f"/members?session_id={mock_session_data}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "members" in data
        assert "total_count" in data


class TestGetMemberSkills:
    """Tests for /member-skills endpoint."""
    
    def test_get_member_skills_success(self, client, mock_session_data, session_manager_reset):
        """Test getting member skills successfully."""
        response = client.post(
            "/member-skills",
            json={
                "session_id": mock_session_data,
                "member_code": "M001"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "current_skills" in data


class TestAnalyzeCareerGap:
    """Tests for /gap-analysis endpoint."""
    
    def test_analyze_career_gap_success(self, client, mock_session_data, mock_trained_model, session_manager_reset):
        """Test career gap analysis successfully."""
        model_id, _ = mock_trained_model
        
        response = client.post(
            "/gap-analysis",
            json={
                "session_id": mock_session_data,
                "model_id": model_id,
                "source_member_code": "M001",
                "target_member_code": "M002"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "gap_skills" in data
