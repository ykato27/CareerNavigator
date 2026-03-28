"""
Repository layer for session and model management.

This module provides data access abstraction over the SessionManager singleton.
"""

from typing import Optional, Dict, Any, List
from backend.utils.session_manager import session_manager


class SessionRepository:
    """Repository for managing sessions and models."""

    def __init__(self):
        self._manager = session_manager

    # Session operations
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None if not found
        """
        return self._manager.get_session(session_id)

    def add_session(
        self, session_id: str, data: Dict[str, Any], metadata: Optional[Dict] = None
    ) -> None:
        """
        Add or update a session.

        Args:
            session_id: Session identifier
            data: Session data to store
            metadata: Optional metadata about the session
        """
        self._manager.add_session(session_id, data, metadata)

    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was removed, False if not found
        """
        return self._manager.remove_session(session_id)

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        return self.get_session(session_id) is not None

    def get_all_sessions(self) -> List[str]:
        """
        Get all session IDs.

        Returns:
            List of session IDs
        """
        stats = self._manager.get_statistics()
        return list(stats.get("sessions", {}).keys())

    # Model operations
    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Retrieve a trained model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model object or None if not found
        """
        return self._manager.get_model(model_id)

    def add_model(self, model_id: str, model: Any) -> None:
        """
        Store a trained model.

        Args:
            model_id: Model identifier
            model: Model object to store
        """
        self._manager.add_model(model_id, model)

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a trained model.

        Args:
            model_id: Model identifier

        Returns:
            True if model was removed, False if not found
        """
        return self._manager.remove_model(model_id)

    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_id: Model identifier

        Returns:
            True if model exists
        """
        return self.get_model(model_id) is not None

    def get_all_models(self) -> List[str]:
        """
        Get all model IDs.

        Returns:
            List of model IDs
        """
        stats = self._manager.get_statistics()
        return list(stats.get("models", {}).keys())

    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with session and model statistics
        """
        return self._manager.get_statistics()


# Singleton instance
session_repository = SessionRepository()
