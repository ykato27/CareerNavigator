"""
Session manager for handling user sessions and state persistence.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from .database import DatabaseManager
from .models import UserSession
from .repository import UserRepository

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions and state persistence."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize session manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.user_repo = UserRepository(db_manager)

    def create_session(self, user_id: str) -> UserSession:
        """
        Create a new user session.

        Args:
            user_id: User ID

        Returns:
            Created UserSession object
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        query = """
            INSERT INTO user_sessions
            (session_id, user_id, created_at, last_active, data_loaded,
             model_trained, current_model_id, state_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            session_id,
            user_id,
            now.isoformat(),
            now.isoformat(),
            0,  # data_loaded
            0,  # model_trained
            None,  # current_model_id
            self.db.serialize_json({}),  # state_data
        )

        self.db.execute_insert(query, params)

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            data_loaded=False,
            model_trained=False,
            current_model_id=None,
            state_data={},
        )

        # Update user's last login
        self.user_repo.update_last_login(user_id)

        logger.info(f"Created session: {session_id} for user: {user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            UserSession object or None
        """
        query = "SELECT * FROM user_sessions WHERE session_id = ?"
        results = self.db.execute_query(query, (session_id,))

        if not results:
            return None

        row = results[0]
        return self._row_to_session(row)

    def get_user_sessions(self, user_id: str, active_only: bool = False) -> list:
        """
        Get sessions for a user.

        Args:
            user_id: User ID
            active_only: If True, only return recent active sessions

        Returns:
            List of UserSession objects
        """
        if active_only:
            # Consider sessions active if last_active within 24 hours
            query = """
                SELECT * FROM user_sessions
                WHERE user_id = ?
                AND datetime(last_active) > datetime('now', '-1 day')
                ORDER BY last_active DESC
            """
        else:
            query = """
                SELECT * FROM user_sessions
                WHERE user_id = ?
                ORDER BY last_active DESC
            """

        results = self.db.execute_query(query, (user_id,))
        return [self._row_to_session(row) for row in results]

    def update_session_activity(self, session_id: str):
        """
        Update session's last active timestamp.

        Args:
            session_id: Session ID
        """
        query = "UPDATE user_sessions SET last_active = ? WHERE session_id = ?"
        self.db.execute_update(query, (datetime.now().isoformat(), session_id))

    def update_session_state(
        self,
        session_id: str,
        data_loaded: Optional[bool] = None,
        model_trained: Optional[bool] = None,
        current_model_id: Optional[str] = None,
        state_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Update session state.

        Args:
            session_id: Session ID
            data_loaded: Data loaded flag
            model_trained: Model trained flag
            current_model_id: Current model ID
            state_data: Additional state data
        """
        updates = []
        params = []

        if data_loaded is not None:
            updates.append("data_loaded = ?")
            params.append(1 if data_loaded else 0)

        if model_trained is not None:
            updates.append("model_trained = ?")
            params.append(1 if model_trained else 0)

        if current_model_id is not None:
            updates.append("current_model_id = ?")
            params.append(current_model_id)

        if state_data is not None:
            updates.append("state_data = ?")
            params.append(self.db.serialize_json(state_data))

        if not updates:
            return

        # Always update last_active
        updates.append("last_active = ?")
        params.append(datetime.now().isoformat())

        # Add session_id for WHERE clause
        params.append(session_id)

        query = f"UPDATE user_sessions SET {', '.join(updates)} WHERE session_id = ?"
        self.db.execute_update(query, tuple(params))
        logger.info(f"Updated session state: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM user_sessions WHERE session_id = ?"
        affected = self.db.execute_update(query, (session_id,))
        return affected > 0

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Number of days to keep sessions

        Returns:
            Number of deleted sessions
        """
        query = """
            DELETE FROM user_sessions
            WHERE datetime(last_active) < datetime('now', ?)
        """
        affected = self.db.execute_update(query, (f"-{days} days",))
        logger.info(f"Cleaned up {affected} old sessions")
        return affected

    def _row_to_session(self, row) -> UserSession:
        """Convert database row to UserSession object."""
        return UserSession(
            session_id=row["session_id"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
            data_loaded=bool(row["data_loaded"]),
            model_trained=bool(row["model_trained"]),
            current_model_id=row["current_model_id"],
            state_data=self.db.deserialize_json(row["state_data"]),
        )
