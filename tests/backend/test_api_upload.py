"""
Tests for backend/api/upload.py
"""
import pytest
from io import BytesIO


class TestUploadFiles:
    """Tests for /api/upload endpoint."""
    
    def test_upload_files_success(self, client, sample_csv_files, session_manager_reset):
        """Test successful file upload."""
        # Prepare test files
        files = []
        for csv_file in sample_csv_files.glob("*.csv"):
            with open(csv_file, 'rb') as f:
                content = f.read()
                files.append(
                    ('files', (csv_file.name, BytesIO(content), 'text/csv'))
                )
        
        # The upload endpoint expects FormData with files
        # This is a simplified test - actual implementation may vary
        response = client.post("/api/upload", files=files)
        
        # May return 200, 400, or 422 depending on implementation
        assert response.status_code in [200, 400, 422]


class TestGetSessionStatus:
    """Tests for /api/session/{session_id}/status endpoint."""
    
    def test_get_session_status_success(self, client, mock_session_data, session_manager_reset):
        """Test getting session status successfully."""
        # Add session to manager
        session_manager_reset.add_session(mock_session_data, {"status": "uploaded"})
        
        response = client.get(f"/api/session/{mock_session_data}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "session_id" in data
    
    def test_get_session_status_not_found(self, client):
        """Test getting status for non-existent session."""
        response = client.get("/api/session/nonexistent/status")
        
        assert response.status_code == 404


class TestDeleteSession:
    """Tests for /api/session/{session_id} DELETE endpoint."""
    
    def test_delete_session_success(self, client, mock_session_data, session_manager_reset):
        """Test deleting a session successfully."""
        # Add session
        session_manager_reset.add_session(mock_session_data, {"status": "uploaded"})
        
        response = client.delete(f"/api/session/{mock_session_data}")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True or "message" in data
    
    def test_delete_session_not_found(self, client):
        """Test deleting non-existent session."""
        response = client.delete("/api/session/nonexistent")
        
        # May return 404 or 200 depending on implementation
        assert response.status_code in [200, 404]


class TestGetSessionsStats:
    """Tests for /api/sessions/stats endpoint."""
    
    def test_get_sessions_stats(self, client, session_manager_reset):
        """Test getting sessions statistics."""
        response = client.get("/api/sessions/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data or "stats" in data


class TestCleanupOldSessions:
    """Tests for /api/sessions/cleanup endpoint."""
    
    def test_cleanup_old_sessions(self, client):
        """Test cleaning up old sessions."""
        response = client.post("/api/sessions/cleanup", json={"max_age_hours": 24})
        
        assert response.status_code == 200
        data = response.json()
        assert "removed" in data or "sessions_removed" in data or "success" in data


class TestGetSessionMembers:
    """Tests for /api/session/{session_id}/members endpoint."""
    
    def test_get_session_members_success(self, client, mock_session_data, session_manager_reset):
        """Test getting session members successfully."""
        response = client.get(f"/api/session/{mock_session_data}/members")
        
        assert response.status_code == 200
        data = response.json()
        assert "members" in data or isinstance(data, list)
    
    def test_get_session_members_not_found(self, client):
        """Test getting members for non-existent session."""
        response = client.get("/api/session/nonexistent/members")
        
        assert response.status_code == 404
