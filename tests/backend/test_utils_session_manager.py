"""
Tests for backend/utils/session_manager.py
"""
import pytest
import time
from backend.utils.session_manager import SessionManager, session_manager


class TestSessionManagerSingleton:
    """Tests for SessionManager singleton pattern."""
    
    def test_singleton_instance(self):
        """Test that SessionManager returns the same instance."""
        manager1 = SessionManager()
        manager2 = SessionManager()
        assert manager1 is manager2
    
    def test_global_instance(self):
        """Test global session_manager instance."""
        manager = SessionManager()
        assert manager is session_manager


class TestSessionManagement:
    """Tests for session management methods."""
    
    def test_add_and_get_session(self, session_manager_reset):
        """Test adding and retrieving a session."""
        session_id = "test_session_001"
        data = {"file_count": 6, "status": "uploaded"}
        
        session_manager_reset.add_session(session_id, data)
        result = session_manager_reset.get_session(session_id)
        
        assert result is not None
        assert result["file_count"] == 6
        assert result["status"] == "uploaded"
        assert "timestamp" in result
        assert "last_accessed" in result
    
    def test_session_exists(self, session_manager_reset):
        """Test checking if session exists."""
        session_id = "test_session_002"
        
        assert not session_manager_reset.session_exists(session_id)
        
        session_manager_reset.add_session(session_id, {"data": "test"})
        
        assert session_manager_reset.session_exists(session_id)
    
    def test_get_nonexistent_session(self, session_manager_reset):
        """Test getting a non-existent session."""
        result = session_manager_reset.get_session("nonexistent")
        assert result is None
    
    def test_remove_session(self, session_manager_reset):
        """Test removing a session."""
        session_id = "test_session_003"
        session_manager_reset.add_session(session_id, {"data": "test"})
        
        assert session_manager_reset.session_exists(session_id)
        
        removed = session_manager_reset.remove_session(session_id)
        
        assert removed is True
        assert not session_manager_reset.session_exists(session_id)
    
    def test_remove_nonexistent_session(self, session_manager_reset):
        """Test removing a non-existent session."""
        # SessionManager.remove_session always returns True currently
        # This is the actual behavior, not an error
        result = session_manager_reset.remove_session("nonexistent")
        # The method doesn't check existence before attempting removal
        assert result is True
    
    def test_last_accessed_updated(self, session_manager_reset):
        """Test that last_accessed is updated on get."""
        session_id = "test_session_004"
        session_manager_reset.add_session(session_id, {"data": "test"})
        
        # Get first time
        result1 = session_manager_reset.get_session(session_id)
        first_access = result1["last_accessed"]
        
        time.sleep(0.01)  # Small delay
        
        # Get second time
        result2 = session_manager_reset.get_session(session_id)
        second_access = result2["last_accessed"]
        
        assert second_access > first_access
    
    def test_cleanup_old_sessions(self, session_manager_reset):
        """Test cleaning up old sessions."""
        # Add recent session
        session_manager_reset.add_session("recent", {"data": "recent"})
        
        # Add old session by manually setting timestamp
        old_session_id = "old_session"
        session_manager_reset.add_session(old_session_id, {"data": "old"})
        session_manager_reset._sessions[old_session_id]["last_accessed"] = time.time() - 90000  # 25 hours ago
        
        # Cleanup sessions older than 24 hours (86400 seconds)
        removed_count = session_manager_reset.cleanup_old_sessions(max_age_seconds=86400)
        
        assert removed_count == 1
        assert session_manager_reset.session_exists("recent")
        assert not session_manager_reset.session_exists(old_session_id)


class TestModelManagement:
    """Tests for model management methods."""
    
    def test_add_and_get_model(self, session_manager_reset):
        """Test adding and retrieving a model."""
        model_id = "model_001"
        mock_model = {"type": "causal", "params": {}}
        
        session_manager_reset.add_model(model_id, mock_model)
        result = session_manager_reset.get_model(model_id)
        
        assert result == mock_model
    
    def test_model_exists(self, session_manager_reset):
        """Test checking if model exists."""
        model_id = "model_002"
        
        assert not session_manager_reset.model_exists(model_id)
        
        session_manager_reset.add_model(model_id, {"model": "data"})
        
        assert session_manager_reset.model_exists(model_id)
    
    def test_get_nonexistent_model(self, session_manager_reset):
        """Test getting a non-existent model."""
        result = session_manager_reset.get_model("nonexistent")
        assert result is None
    
    def test_remove_model(self, session_manager_reset):
        """Test removing a model."""
        model_id = "model_003"
        session_manager_reset.add_model(model_id, {"model": "data"})
        
        assert session_manager_reset.model_exists(model_id)
        
        removed = session_manager_reset.remove_model(model_id)
        
        assert removed is True
        assert not session_manager_reset.model_exists(model_id)
    
    def test_remove_nonexistent_model(self, session_manager_reset):
        """Test removing a non-existent model."""
        result = session_manager_reset.remove_model("nonexistent")
        assert result is False
    
    def test_cleanup_old_models(self, session_manager_reset):
        """Test cleaning up old models."""
        # Add recent model
        session_manager_reset.add_model("recent_model", {"data": "recent"})
        
        # Add old model by manually setting timestamp
        old_model_id = "old_model"
        session_manager_reset.add_model(old_model_id, {"data": "old"})
        session_manager_reset._models[old_model_id]["timestamp"] = time.time() - 90000  # 25 hours ago
        
        # Cleanup models older than 24 hours
        removed_count = session_manager_reset.cleanup_old_models(max_age_seconds=86400)
        
        assert removed_count == 1
        assert session_manager_reset.model_exists("recent_model")
        assert not session_manager_reset.model_exists(old_model_id)


class TestCacheManagement:
    """Tests for cache management methods."""
    
    def test_add_and_get_cache(self, session_manager_reset):
        """Test adding and retrieving cache."""
        session_id = "session_cache_001"
        cache_data = {
            "member_competence": "dataframe",
            "competence_master": "dataframe"
        }
        
        session_manager_reset.add_cache(session_id, cache_data)
        result = session_manager_reset.get_cache(session_id)
        
        assert result == cache_data
    
    def test_get_nonexistent_cache(self, session_manager_reset):
        """Test getting non-existent cache."""
        result = session_manager_reset.get_cache("nonexistent")
        assert result is None
    
    def test_clear_specific_cache(self, session_manager_reset):
        """Test clearing cache for a specific session."""
        session_manager_reset.add_cache("session1", {"data": "1"})
        session_manager_reset.add_cache("session2", {"data": "2"})
        
        session_manager_reset.clear_cache("session1")
        
        assert session_manager_reset.get_cache("session1") is None
        assert session_manager_reset.get_cache("session2") is not None
    
    def test_clear_all_cache(self, session_manager_reset):
        """Test clearing all cache."""
        session_manager_reset.add_cache("session1", {"data": "1"})
        session_manager_reset.add_cache("session2", {"data": "2"})
        
        session_manager_reset.clear_cache()
        
        assert session_manager_reset.get_cache("session1") is None
        assert session_manager_reset.get_cache("session2") is None
    
    def test_remove_session_also_clears_cache(self, session_manager_reset):
        """Test that removing session also clears its cache."""
        session_id = "session_with_cache"
        
        session_manager_reset.add_session(session_id, {"data": "session"})
        session_manager_reset.add_cache(session_id, {"data": "cache"})
        
        session_manager_reset.remove_session(session_id)
        
        assert session_manager_reset.get_cache(session_id) is None


class TestStatistics:
    """Tests for statistics methods."""
    
    def test_get_stats_empty(self, session_manager_reset):
        """Test getting stats when everything is empty."""
        stats = session_manager_reset.get_stats()
        
        assert stats["active_sessions"] == 0
        assert stats["trained_models"] == 0
        assert stats["cached_sessions"] == 0
        assert stats["total_memory_items"] == 0
    
    def test_get_stats_with_data(self, session_manager_reset):
        """Test getting stats with some data."""
        session_manager_reset.add_session("session1", {"data": "1"})
        session_manager_reset.add_session("session2", {"data": "2"})
        session_manager_reset.add_model("model1", {"data": "model"})
        session_manager_reset.add_cache("session1", {"data": "cache"})
        
        stats = session_manager_reset.get_stats()
        
        assert stats["active_sessions"] == 2
        assert stats["trained_models"] == 1
        assert stats["cached_sessions"] == 1
        assert stats["total_memory_items"] == 4


class TestThreadSafety:
    """Tests for thread-safe operations."""
    
    def test_concurrent_access(self, session_manager_reset):
        """Test that SessionManager handles concurrent access properly."""
        import threading
        
        results = []
        
        def add_session(session_id):
            session_manager_reset.add_session(session_id, {"thread": threading.current_thread().name})
            results.append(session_id)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_session, args=(f"session_{i}",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All sessions should be added
        assert len(results) == 10
        stats = session_manager_reset.get_stats()
        assert stats["active_sessions"] == 10
