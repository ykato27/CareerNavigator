"""
Repository classes for data access layer.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .database import DatabaseManager
from .models import User, RecommendationHistory, ModelMetadata, UserSession

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user data operations."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize user repository.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create_user(
        self, username: str, email: Optional[str] = None, settings: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            settings: User settings dictionary

        Returns:
            Created User object
        """
        user_id = str(uuid.uuid4())
        created_at = datetime.now()

        query = """
            INSERT INTO users (user_id, username, email, created_at, settings)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            user_id,
            username,
            email,
            created_at.isoformat(),
            self.db.serialize_json(settings or {}),
        )

        self.db.execute_insert(query, params)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            created_at=created_at,
            settings=settings or {},
        )
        logger.info(f"Created user: {username} ({user_id})")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        query = "SELECT * FROM users WHERE user_id = ?"
        results = self.db.execute_query(query, (user_id,))

        if not results:
            return None

        row = results[0]
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            settings=self.db.deserialize_json(row["settings"]),
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.

        Args:
            username: Username

        Returns:
            User object or None
        """
        query = "SELECT * FROM users WHERE username = ?"
        results = self.db.execute_query(query, (username,))

        if not results:
            return None

        row = results[0]
        return User(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            settings=self.db.deserialize_json(row["settings"]),
        )

    def update_last_login(self, user_id: str):
        """
        Update user's last login timestamp.

        Args:
            user_id: User ID
        """
        query = "UPDATE users SET last_login = ? WHERE user_id = ?"
        self.db.execute_update(query, (datetime.now().isoformat(), user_id))

    def update_settings(self, user_id: str, settings: Dict[str, Any]):
        """
        Update user settings.

        Args:
            user_id: User ID
            settings: New settings dictionary
        """
        query = "UPDATE users SET settings = ? WHERE user_id = ?"
        self.db.execute_update(query, (self.db.serialize_json(settings), user_id))
        logger.info(f"Updated settings for user: {user_id}")

    def list_users(self) -> List[User]:
        """
        List all users.

        Returns:
            List of User objects
        """
        query = "SELECT * FROM users ORDER BY created_at DESC"
        results = self.db.execute_query(query)

        users = []
        for row in results:
            users.append(
                User(
                    user_id=row["user_id"],
                    username=row["username"],
                    email=row["email"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_login=(
                        datetime.fromisoformat(row["last_login"]) if row["last_login"] else None
                    ),
                    settings=self.db.deserialize_json(row["settings"]),
                )
            )
        return users


class RecommendationHistoryRepository:
    """Repository for recommendation history operations."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize recommendation history repository.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create_history(self, history: RecommendationHistory) -> RecommendationHistory:
        """
        Create a new recommendation history record.

        Args:
            history: RecommendationHistory object

        Returns:
            Created history with generated ID
        """
        if not history.history_id:
            history.history_id = str(uuid.uuid4())
        if not history.timestamp:
            history.timestamp = datetime.now()

        query = """
            INSERT INTO recommendation_history
            (history_id, user_id, member_code, member_name, method,
             timestamp, recommendations, reference_persons, parameters, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            history.history_id,
            history.user_id,
            history.member_code,
            history.member_name,
            history.method,
            history.timestamp.isoformat(),
            self.db.serialize_json(history.recommendations),
            self.db.serialize_json(history.reference_persons),
            self.db.serialize_json(history.parameters),
            history.execution_time,
        )

        self.db.execute_insert(query, params)
        logger.info(f"Created recommendation history: {history.history_id}")
        return history

    def get_history_by_id(self, history_id: str) -> Optional[RecommendationHistory]:
        """
        Get recommendation history by ID.

        Args:
            history_id: History ID

        Returns:
            RecommendationHistory object or None
        """
        query = "SELECT * FROM recommendation_history WHERE history_id = ?"
        results = self.db.execute_query(query, (history_id,))

        if not results:
            return None

        row = results[0]
        return self._row_to_history(row)

    def get_user_history(
        self, user_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[RecommendationHistory]:
        """
        Get recommendation history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of RecommendationHistory objects
        """
        query = """
            SELECT * FROM recommendation_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """

        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        results = self.db.execute_query(query, (user_id,))
        return [self._row_to_history(row) for row in results]

    def get_member_history(
        self, user_id: str, member_code: str, limit: Optional[int] = None
    ) -> List[RecommendationHistory]:
        """
        Get recommendation history for a specific member.

        Args:
            user_id: User ID
            member_code: Member code
            limit: Maximum number of records to return

        Returns:
            List of RecommendationHistory objects
        """
        query = """
            SELECT * FROM recommendation_history
            WHERE user_id = ? AND member_code = ?
            ORDER BY timestamp DESC
        """

        if limit is not None:
            query += f" LIMIT {limit}"

        results = self.db.execute_query(query, (user_id, member_code))
        return [self._row_to_history(row) for row in results]

    def delete_history(self, history_id: str) -> bool:
        """
        Delete a recommendation history record.

        Args:
            history_id: History ID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM recommendation_history WHERE history_id = ?"
        affected = self.db.execute_update(query, (history_id,))
        return affected > 0

    def _row_to_history(self, row) -> RecommendationHistory:
        """Convert database row to RecommendationHistory object."""
        return RecommendationHistory(
            history_id=row["history_id"],
            user_id=row["user_id"],
            member_code=row["member_code"],
            member_name=row["member_name"],
            method=row["method"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            recommendations=self.db.deserialize_json(row["recommendations"]),
            reference_persons=self.db.deserialize_json(row["reference_persons"]),
            parameters=self.db.deserialize_json(row["parameters"]),
            execution_time=row["execution_time"],
        )


class ModelRepository:
    """Repository for model metadata operations."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize model repository.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create_model(self, metadata: ModelMetadata) -> ModelMetadata:
        """
        Create a new model metadata record.

        Args:
            metadata: ModelMetadata object

        Returns:
            Created metadata
        """
        if not metadata.created_at:
            metadata.created_at = datetime.now()

        query = """
            INSERT INTO model_metadata
            (model_id, user_id, model_type, created_at, parameters,
             metrics, file_path, data_hash, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            metadata.model_id,
            metadata.user_id,
            metadata.model_type,
            metadata.created_at.isoformat(),
            self.db.serialize_json(metadata.parameters),
            self.db.serialize_json(metadata.metrics),
            metadata.file_path,
            metadata.data_hash,
            metadata.description,
        )

        self.db.execute_insert(query, params)
        logger.info(f"Created model metadata: {metadata.model_id}")
        return metadata

    def get_model_by_id(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.

        Args:
            model_id: Model ID

        Returns:
            ModelMetadata object or None
        """
        query = "SELECT * FROM model_metadata WHERE model_id = ?"
        results = self.db.execute_query(query, (model_id,))

        if not results:
            return None

        row = results[0]
        return self._row_to_metadata(row)

    def get_user_models(
        self, user_id: str, model_type: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        Get models for a user.

        Args:
            user_id: User ID
            model_type: Optional filter by model type

        Returns:
            List of ModelMetadata objects
        """
        if model_type:
            query = """
                SELECT * FROM model_metadata
                WHERE user_id = ? AND model_type = ?
                ORDER BY created_at DESC
            """
            results = self.db.execute_query(query, (user_id, model_type))
        else:
            query = """
                SELECT * FROM model_metadata
                WHERE user_id = ?
                ORDER BY created_at DESC
            """
            results = self.db.execute_query(query, (user_id,))

        return [self._row_to_metadata(row) for row in results]

    def get_latest_model(
        self, user_id: str, model_type: str, data_hash: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get the latest model for a user.

        Args:
            user_id: User ID
            model_type: Model type
            data_hash: Optional data hash to match

        Returns:
            ModelMetadata object or None
        """
        if data_hash:
            query = """
                SELECT * FROM model_metadata
                WHERE user_id = ? AND model_type = ? AND data_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            results = self.db.execute_query(query, (user_id, model_type, data_hash))
        else:
            query = """
                SELECT * FROM model_metadata
                WHERE user_id = ? AND model_type = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            results = self.db.execute_query(query, (user_id, model_type))

        if not results:
            return None

        return self._row_to_metadata(results[0])

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model metadata record.

        Args:
            model_id: Model ID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM model_metadata WHERE model_id = ?"
        affected = self.db.execute_update(query, (model_id,))
        return affected > 0

    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata object."""
        return ModelMetadata(
            model_id=row["model_id"],
            user_id=row["user_id"],
            model_type=row["model_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            parameters=self.db.deserialize_json(row["parameters"]),
            metrics=self.db.deserialize_json(row["metrics"]),
            file_path=row["file_path"],
            data_hash=row["data_hash"],
            description=row["description"],
        )
