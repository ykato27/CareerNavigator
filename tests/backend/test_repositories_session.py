"""
Unit tests for backend/repositories/session_repository.py
"""
import pytest
from unittest.mock import Mock

from backend.repositories.session_repository import SessionRepository, session_repository


class TestSessionRepository:
    """Tests for SessionRepository data access layer."""

    @pytest.fixture
    def repository(self):
        """SessionRepository instance."""
        return SessionRepository()

    @pytest.fixture
    def mock_manager(self, mocker):
        """Mock SessionManager."""
        mock_mgr = mocker.Mock()
        return mock_mgr

    # Session operations tests
    def test_get_session_exists(self, repository, mock_manager):
        """Test getting existing session."""
        repository._manager = mock_manager
        mock_manager.get_session.return_value = {"data": "test_data", "timestamp": 1234567890}
        
        result = repository.get_session("session_123")
        
        assert result == {"data": "test_data", "timestamp": 1234567890}
        mock_manager.get_session.assert_called_once_with("session_123")

    def test_get_session_not_exists(self, repository, mock_manager):
        """Test getting non-existent session returns None."""
        repository._manager = mock_manager
        mock_manager.get_session.return_value = None
        
        result = repository.get_session("nonexistent")
        
        assert result is None
        mock_manager.get_session.assert_called_once_with("nonexistent")

    def test_add_session(self, repository, mock_manager):
        """Test adding a session."""
        repository._manager = mock_manager
        
        session_data = {"key": "value"}
        metadata = {"created_by": "test"}
        
        repository.add_session("session_123", session_data, metadata)
        
        mock_manager.add_session.assert_called_once_with("session_123", session_data, metadata)

    def test_remove_session_success(self, repository, mock_manager):
        """Test removing existing session."""
        repository._manager = mock_manager
        mock_manager.remove_session.return_value = True
        
        result = repository.remove_session("session_123")
        
        assert result is True
        mock_manager.remove_session.assert_called_once_with("session_123")

    def test_remove_session_not_found(self, repository, mock_manager):
        """Test removing non-existent session."""
        repository._manager = mock_manager
        mock_manager.remove_session.return_value = False
        
        result = repository.remove_session("nonexistent")
        
        assert result is False
        mock_manager.remove_session.assert_called_once_with("nonexistent")

    def test_session_exists_true(self, repository, mock_manager):
        """Test session_exists returns True for existing session."""
        repository._manager = mock_manager
        mock_manager.get_session.return_value = {"data": "test"}
        
        result = repository.session_exists("session_123")
        
        assert result is True
        mock_manager.get_session.assert_called_once_with("session_123")

    def test_session_exists_false(self, repository, mock_manager):
        """Test session_exists returns False for non-existent session."""
        repository._manager = mock_manager
        mock_manager.get_session.return_value = None
        
        result = repository.session_exists("nonexistent")
        
        assert result is False
        mock_manager.get_session.assert_called_once_with("nonexistent")

    def test_get_all_sessions(self, repository, mock_manager):
        """Test getting all session IDs."""
        repository._manager = mock_manager
        mock_manager.get_statistics.return_value = {
            "sessions": {"session_1": {}, "session_2": {}},
            "models": {}
        }
        
        result = repository.get_all_sessions()
        
        assert result == ["session_1", "session_2"]
        mock_manager.get_statistics.assert_called_once()

    # Model operations tests
    def test_get_model_exists(self, repository, mock_manager):
        """Test getting existing model."""
        repository._manager = mock_manager
        mock_model = Mock()
        mock_manager.get_model.return_value = mock_model
        
        result = repository.get_model("model_123")
        
        assert result == mock_model
        mock_manager.get_model.assert_called_once_with("model_123")

    def test_get_model_not_exists(self, repository, mock_manager):
        """Test getting non-existent model returns None."""
        repository._manager = mock_manager
        mock_manager.get_model.return_value = None
        
        result = repository.get_model("nonexistent")
        
        assert result is None
        mock_manager.get_model.assert_called_once_with("nonexistent")

    def test_add_model(self, repository, mock_manager):
        """Test adding a model."""
        repository._manager = mock_manager
        mock_model = Mock()
        
        repository.add_model("model_123", mock_model)
        
        mock_manager.add_model.assert_called_once_with("model_123", mock_model)

    def test_remove_model_success(self, repository, mock_manager):
        """Test removing existing model."""
        repository._manager = mock_manager
        mock_manager.remove_model.return_value = True
        
        result = repository.remove_model("model_123")
        
        assert result is True
        mock_manager.remove_model.assert_called_once_with("model_123")

    def test_remove_model_not_found(self, repository, mock_manager):
        """Test removing non-existent model."""
        repository._manager = mock_manager
        mock_manager.remove_model.return_value = False
        
        result = repository.remove_model("nonexistent")
        
        assert result is False
        mock_manager.remove_model.assert_called_once_with("nonexistent")

    def test_model_exists_true(self, repository, mock_manager):
        """Test model_exists returns True for existing model."""
        repository._manager = mock_manager
        mock_model = Mock()
        mock_manager.get_model.return_value = mock_model
        
        result = repository.model_exists("model_123")
        
        assert result is True
        mock_manager.get_model.assert_called_once_with("model_123")

    def test_model_exists_false(self, repository, mock_manager):
        """Test model_exists returns False for non-existent model."""
        repository._manager = mock_manager
        mock_manager.get_model.return_value = None
        
        result = repository.model_exists("nonexistent")
        
        assert result is False
        mock_manager.get_model.assert_called_once_with("nonexistent")

    def test_get_all_models(self, repository, mock_manager):
        """Test getting all model IDs."""
        repository._manager = mock_manager
        mock_manager.get_statistics.return_value = {
            "sessions": {},
            "models": {"model_1": {}, "model_2": {}, "model_3": {}}
        }
        
        result = repository.get_all_models()
        
        assert result == ["model_1", "model_2", "model_3"]
        mock_manager.get_statistics.assert_called_once()

    # Statistics tests
    def test_get_statistics(self, repository, mock_manager):
        """Test getting statistics."""
        repository._manager = mock_manager
        stats = {
            "active_sessions": 5,
            "trained_models": 3,
            "cached_sessions": 2
        }
        mock_manager.get_statistics.return_value = stats
        
        result = repository.get_statistics()
        
        assert result == stats
        mock_manager.get_statistics.assert_called_once()


class TestSessionRepositorySingleton:
    """Test SessionRepository singleton instance."""

    def test_singleton_instance_exists(self):
        """Test that session_repository singleton exists."""
        assert session_repository is not None
        assert isinstance(session_repository, SessionRepository)

    def test_singleton_has_manager(self):
        """Test that singleton has SessionManager instance."""
        assert hasattr(session_repository, '_manager')
        assert session_repository._manager is not None
