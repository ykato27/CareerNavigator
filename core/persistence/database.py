"""
Database manager for SQLite persistence.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and schema."""

    def __init__(self, db_path: str = "career_navigator.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_directory()
        self._initialized = False

    def _ensure_directory(self):
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Get database connection context manager.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def initialize_schema(self):
        """Initialize database schema."""
        if self._initialized:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    settings TEXT DEFAULT '{}'
                )
            """)

            # Recommendation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_history (
                    history_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    member_code TEXT NOT NULL,
                    member_name TEXT,
                    method TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    reference_persons TEXT DEFAULT '[]',
                    parameters TEXT DEFAULT '{}',
                    execution_time REAL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Model metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parameters TEXT DEFAULT '{}',
                    metrics TEXT DEFAULT '{}',
                    file_path TEXT NOT NULL,
                    data_hash TEXT,
                    description TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    data_loaded INTEGER DEFAULT 0,
                    model_trained INTEGER DEFAULT 0,
                    current_model_id TEXT,
                    state_data TEXT DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (current_model_id) REFERENCES model_metadata (model_id)
                )
            """)

            # Create indices for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_user_id
                ON recommendation_history(user_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_timestamp
                ON recommendation_history(timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_user_id
                ON model_metadata(user_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON user_sessions(user_id)
            """)

            conn.commit()
            self._initialized = True
            logger.info(f"Database initialized at {self.db_path}")

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[sqlite3.Row]:
        """
        Execute SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of query results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def execute_insert(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute INSERT query and return last row ID.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Last inserted row ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid

    def execute_update(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute UPDATE/DELETE query and return affected rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount

    @staticmethod
    def serialize_json(data: Any) -> str:
        """Serialize data to JSON string."""
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def deserialize_json(json_str: str) -> Any:
        """Deserialize JSON string to data."""
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {}
