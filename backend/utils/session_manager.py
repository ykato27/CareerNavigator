"""
Session and model state management for the backend API.

This module provides centralized management of:
- Uploaded session data
- Trained models
- Cached transformed data
"""
from typing import Dict, Any, Optional
import time
from threading import Lock


class SessionManager:
    """
    Singleton class for managing session data, trained models, and cache.
    Thread-safe implementation to prevent race conditions.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, Any] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._data_lock = Lock()
        self._initialized = True

    # Session management
    def add_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Add a new session with metadata."""
        with self._data_lock:
            self._sessions[session_id] = {
                **data,
                "timestamp": time.time(),
                "last_accessed": time.time()
            }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data and update last accessed time."""
        with self._data_lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_accessed"] = time.time()
                return self._sessions[session_id]
            return None

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def remove_session(self, session_id: str) -> bool:
        """Remove a session and its associated cache."""
        with self._data_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if session_id in self._cache:
                del self._cache[session_id]
            return True
        return False

    def cleanup_old_sessions(self, max_age_seconds: int = 86400) -> int:
        """
        Remove sessions older than max_age_seconds (default: 24 hours).
        Returns the number of sessions removed.
        """
        current_time = time.time()
        removed_count = 0

        with self._data_lock:
            sessions_to_remove = [
                sid for sid, data in self._sessions.items()
                if current_time - data.get("last_accessed", 0) > max_age_seconds
            ]

            for session_id in sessions_to_remove:
                self.remove_session(session_id)
                removed_count += 1

        return removed_count

    # Model management
    def add_model(self, model_id: str, model: Any) -> None:
        """Store a trained model."""
        with self._data_lock:
            self._models[model_id] = {
                "model": model,
                "timestamp": time.time()
            }

    def get_model(self, model_id: str) -> Optional[Any]:
        """Retrieve a trained model."""
        with self._data_lock:
            if model_id in self._models:
                return self._models[model_id]["model"]
            return None

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        return model_id in self._models

    def remove_model(self, model_id: str) -> bool:
        """Remove a trained model."""
        with self._data_lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    def cleanup_old_models(self, max_age_seconds: int = 86400) -> int:
        """
        Remove models older than max_age_seconds (default: 24 hours).
        Returns the number of models removed.
        """
        current_time = time.time()
        removed_count = 0

        with self._data_lock:
            models_to_remove = [
                mid for mid, data in self._models.items()
                if current_time - data.get("timestamp", 0) > max_age_seconds
            ]

            for model_id in models_to_remove:
                self.remove_model(model_id)
                removed_count += 1

        return removed_count

    # Cache management
    def add_cache(self, session_id: str, data: Dict[str, Any]) -> None:
        """Cache transformed data for a session."""
        with self._data_lock:
            self._cache[session_id] = {
                "data": data,
                "timestamp": time.time()
            }

    def get_cache(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data for a session."""
        with self._data_lock:
            if session_id in self._cache:
                return self._cache[session_id]["data"]
            return None

    def clear_cache(self, session_id: str = None) -> None:
        """Clear cache for a specific session or all sessions."""
        with self._data_lock:
            if session_id:
                if session_id in self._cache:
                    del self._cache[session_id]
            else:
                self._cache.clear()

    # Statistics
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions, models, and cache."""
        with self._data_lock:
            return {
                "active_sessions": len(self._sessions),
                "trained_models": len(self._models),
                "cached_sessions": len(self._cache),
                "total_memory_items": len(self._sessions) + len(self._models) + len(self._cache)
            }


# Global singleton instance
session_manager = SessionManager()
