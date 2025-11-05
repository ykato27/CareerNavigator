"""
Tests for persistence layer.
"""

import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

from skillnote_recommendation.core.persistence.database import DatabaseManager
from skillnote_recommendation.core.persistence.repository import (
    UserRepository,
    RecommendationHistoryRepository,
    ModelRepository
)
from skillnote_recommendation.core.persistence.session_manager import SessionManager
from skillnote_recommendation.core.persistence.model_storage import ModelStorage
from skillnote_recommendation.core.persistence.models import (
    User,
    RecommendationHistory,
    ModelMetadata
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db_manager = DatabaseManager(db_path)
    db_manager.initialize_schema()

    yield db_manager

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def user_repo(temp_db):
    """Create UserRepository with temporary database."""
    return UserRepository(temp_db)


@pytest.fixture
def history_repo(temp_db):
    """Create RecommendationHistoryRepository with temporary database."""
    return RecommendationHistoryRepository(temp_db)


@pytest.fixture
def model_repo(temp_db):
    """Create ModelRepository with temporary database."""
    return ModelRepository(temp_db)


@pytest.fixture
def session_manager(temp_db):
    """Create SessionManager with temporary database."""
    return SessionManager(temp_db)


@pytest.fixture
def model_storage(temp_db):
    """Create ModelStorage with temporary directory."""
    temp_dir = tempfile.mkdtemp()
    storage = ModelStorage(temp_db, temp_dir)

    yield storage

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestUserRepository:
    """Tests for UserRepository."""

    def test_create_user(self, user_repo):
        """Test creating a new user."""
        user = user_repo.create_user(
            username="testuser",
            email="test@example.com",
            settings={"theme": "dark"}
        )

        assert user.user_id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.settings == {"theme": "dark"}
        assert user.created_at is not None

    def test_get_user_by_id(self, user_repo):
        """Test retrieving user by ID."""
        created_user = user_repo.create_user("testuser")
        retrieved_user = user_repo.get_user_by_id(created_user.user_id)

        assert retrieved_user is not None
        assert retrieved_user.user_id == created_user.user_id
        assert retrieved_user.username == "testuser"

    def test_get_user_by_username(self, user_repo):
        """Test retrieving user by username."""
        user_repo.create_user("testuser")
        retrieved_user = user_repo.get_user_by_username("testuser")

        assert retrieved_user is not None
        assert retrieved_user.username == "testuser"

    def test_update_last_login(self, user_repo):
        """Test updating user's last login."""
        user = user_repo.create_user("testuser")
        assert user.last_login is None

        user_repo.update_last_login(user.user_id)

        updated_user = user_repo.get_user_by_id(user.user_id)
        assert updated_user.last_login is not None

    def test_update_settings(self, user_repo):
        """Test updating user settings."""
        user = user_repo.create_user("testuser", settings={"theme": "light"})

        new_settings = {"theme": "dark", "language": "ja"}
        user_repo.update_settings(user.user_id, new_settings)

        updated_user = user_repo.get_user_by_id(user.user_id)
        assert updated_user.settings == new_settings


class TestRecommendationHistoryRepository:
    """Tests for RecommendationHistoryRepository."""

    def test_create_history(self, user_repo, history_repo):
        """Test creating recommendation history."""
        user = user_repo.create_user("testuser")

        history = RecommendationHistory(
            user_id=user.user_id,
            member_code="M001",
            member_name="Test Member",
            method="nmf",
            recommendations=[
                {"competence_code": "C001", "priority_score": 0.9}
            ],
            parameters={"top_n": 10},
            execution_time=1.5
        )

        created_history = history_repo.create_history(history)

        assert created_history.history_id is not None
        assert created_history.user_id == user.user_id
        assert created_history.member_code == "M001"
        assert len(created_history.recommendations) == 1

    def test_get_user_history(self, user_repo, history_repo):
        """Test retrieving user's recommendation history."""
        user = user_repo.create_user("testuser")

        # Create multiple history records
        for i in range(3):
            history = RecommendationHistory(
                user_id=user.user_id,
                member_code=f"M00{i}",
                member_name=f"Member {i}",
                method="nmf",
                recommendations=[]
            )
            history_repo.create_history(history)

        histories = history_repo.get_user_history(user.user_id)

        assert len(histories) == 3
        # Should be ordered by timestamp DESC
        assert histories[0].member_code == "M002"

    def test_get_member_history(self, user_repo, history_repo):
        """Test retrieving history for specific member."""
        user = user_repo.create_user("testuser")

        # Create history for different members
        for i in range(3):
            history = RecommendationHistory(
                user_id=user.user_id,
                member_code="M001" if i % 2 == 0 else "M002",
                member_name=f"Member {i}",
                method="nmf",
                recommendations=[]
            )
            history_repo.create_history(history)

        m001_histories = history_repo.get_member_history(user.user_id, "M001")

        assert len(m001_histories) == 2


class TestModelRepository:
    """Tests for ModelRepository."""

    def test_create_model(self, user_repo, model_repo):
        """Test creating model metadata."""
        user = user_repo.create_user("testuser")

        metadata = ModelMetadata(
            model_id="model_001",
            user_id=user.user_id,
            model_type="nmf",
            parameters={"n_components": 10},
            metrics={"error": 0.05},
            file_path="/path/to/model.pkl",
            data_hash="abc123"
        )

        created_metadata = model_repo.create_model(metadata)

        assert created_metadata.model_id == "model_001"
        assert created_metadata.created_at is not None

    def test_get_user_models(self, user_repo, model_repo):
        """Test retrieving user's models."""
        user = user_repo.create_user("testuser")

        # Create multiple models
        for i in range(3):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                user_id=user.user_id,
                model_type="nmf",
                file_path=f"/path/to/model_{i}.pkl"
            )
            model_repo.create_model(metadata)

        models = model_repo.get_user_models(user.user_id)

        assert len(models) == 3

    def test_get_latest_model(self, user_repo, model_repo):
        """Test retrieving latest model."""
        user = user_repo.create_user("testuser")

        # Create models with different timestamps
        import time
        for i in range(3):
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                user_id=user.user_id,
                model_type="nmf",
                file_path=f"/path/to/model_{i}.pkl",
                data_hash="abc123"
            )
            model_repo.create_model(metadata)
            time.sleep(0.01)  # Ensure different timestamps

        latest = model_repo.get_latest_model(user.user_id, "nmf", "abc123")

        assert latest is not None
        assert latest.model_id == "model_2"


class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self, user_repo, session_manager):
        """Test creating a new session."""
        user = user_repo.create_user("testuser")

        session = session_manager.create_session(user.user_id)

        assert session.session_id is not None
        assert session.user_id == user.user_id
        assert session.data_loaded is False
        assert session.model_trained is False

    def test_update_session_state(self, user_repo, session_manager):
        """Test updating session state."""
        user = user_repo.create_user("testuser")
        session = session_manager.create_session(user.user_id)

        session_manager.update_session_state(
            session.session_id,
            data_loaded=True,
            model_trained=True,
            current_model_id="model_001"
        )

        updated_session = session_manager.get_session(session.session_id)

        assert updated_session.data_loaded is True
        assert updated_session.model_trained is True
        assert updated_session.current_model_id == "model_001"


class TestModelStorage:
    """Tests for ModelStorage."""

    def test_save_and_load_model(self, user_repo, model_storage):
        """Test saving and loading a model."""
        user = user_repo.create_user("testuser")

        # Create a simple model (dict for testing)
        model = {"test": "data", "value": 42}

        # Save model
        metadata = model_storage.save_model(
            model=model,
            user_id=user.user_id,
            model_type="test",
            parameters={"param": 1},
            metrics={"metric": 0.5},
            description="Test model"
        )

        assert metadata.model_id is not None
        assert Path(metadata.file_path).exists()

        # Load model
        loaded_model = model_storage.load_model(metadata.model_id)

        assert loaded_model is not None
        assert loaded_model["test"] == "data"
        assert loaded_model["value"] == 42

    def test_load_latest_model(self, user_repo, model_storage):
        """Test loading the latest model."""
        user = user_repo.create_user("testuser")

        # Create training data
        training_data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })

        # Save multiple models
        for i in range(3):
            model = {"version": i}
            model_storage.save_model(
                model=model,
                user_id=user.user_id,
                model_type="test",
                parameters={},
                metrics={},
                training_data=training_data
            )

        # Load latest
        loaded_model, metadata = model_storage.load_latest_model(
            user_id=user.user_id,
            model_type="test",
            training_data=training_data
        )

        assert loaded_model is not None
        assert loaded_model["version"] == 2

    def test_delete_model(self, user_repo, model_storage):
        """Test deleting a model."""
        user = user_repo.create_user("testuser")

        # Save model
        model = {"test": "data"}
        metadata = model_storage.save_model(
            model=model,
            user_id=user.user_id,
            model_type="test",
            parameters={},
            metrics={}
        )

        model_id = metadata.model_id
        file_path = Path(metadata.file_path)

        assert file_path.exists()

        # Delete model
        success = model_storage.delete_model(model_id)

        assert success is True
        assert not file_path.exists()
