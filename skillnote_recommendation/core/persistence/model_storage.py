"""
Model storage utilities for saving and loading trained models.
"""

import hashlib
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional, Dict
import logging
import pandas as pd

from .database import DatabaseManager
from .repository import ModelRepository
from .models import ModelMetadata

logger = logging.getLogger(__name__)


class ModelStorage:
    """Handles model serialization and storage."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        storage_dir: str = "models"
    ):
        """
        Initialize model storage.

        Args:
            db_manager: Database manager instance
            storage_dir: Directory for storing model files
        """
        self.db = db_manager
        self.model_repo = ModelRepository(db_manager)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        user_id: str,
        model_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_data: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
        use_joblib: bool = True
    ) -> ModelMetadata:
        """
        Save a trained model.

        Args:
            model: Trained model object
            user_id: User ID
            model_type: Model type (e.g., "nmf", "graph", "hybrid")
            parameters: Model parameters
            metrics: Model metrics
            training_data: Optional training data for hash computation
            description: Optional model description
            use_joblib: If True, use joblib; otherwise use pickle

        Returns:
            ModelMetadata object
        """
        # Compute data hash if training data provided
        data_hash = None
        if training_data is not None:
            data_hash = self._compute_data_hash(training_data)

        # Generate model ID and file path
        model_id = self._generate_model_id(user_id, model_type, data_hash)
        file_name = f"{model_id}.{'joblib' if use_joblib else 'pkl'}"
        file_path = self.storage_dir / file_name

        # Save model file
        try:
            if use_joblib:
                joblib.dump(model, file_path)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            logger.info(f"Saved model file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        # Create metadata record
        metadata = ModelMetadata(
            model_id=model_id,
            user_id=user_id,
            model_type=model_type,
            parameters=parameters,
            metrics=metrics,
            file_path=str(file_path),
            data_hash=data_hash,
            description=description
        )

        self.model_repo.create_model(metadata)
        return metadata

    def load_model(
        self,
        model_id: str,
        use_joblib: bool = True
    ) -> Optional[Any]:
        """
        Load a saved model.

        Args:
            model_id: Model ID
            use_joblib: If True, use joblib; otherwise use pickle

        Returns:
            Loaded model object or None
        """
        # Get metadata
        metadata = self.model_repo.get_model_by_id(model_id)
        if not metadata:
            logger.warning(f"Model metadata not found: {model_id}")
            return None

        # Load model file
        file_path = Path(metadata.file_path)
        if not file_path.exists():
            logger.warning(f"Model file not found: {file_path}")
            return None

        try:
            if use_joblib:
                model = joblib.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            logger.info(f"Loaded model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def load_latest_model(
        self,
        user_id: str,
        model_type: str,
        training_data: Optional[pd.DataFrame] = None,
        use_joblib: bool = True
    ) -> tuple[Optional[Any], Optional[ModelMetadata]]:
        """
        Load the latest model for a user.

        Args:
            user_id: User ID
            model_type: Model type
            training_data: Optional training data for hash matching
            use_joblib: If True, use joblib; otherwise use pickle

        Returns:
            Tuple of (model, metadata) or (None, None)
        """
        # Compute data hash if training data provided
        data_hash = None
        if training_data is not None:
            data_hash = self._compute_data_hash(training_data)

        # Get latest metadata
        metadata = self.model_repo.get_latest_model(
            user_id=user_id,
            model_type=model_type,
            data_hash=data_hash
        )

        if not metadata:
            logger.info(f"No saved model found for user {user_id}, type {model_type}")
            return None, None

        # Load model
        model = self.load_model(metadata.model_id, use_joblib=use_joblib)
        if model is None:
            return None, None

        return model, metadata

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a saved model.

        Args:
            model_id: Model ID

        Returns:
            True if deleted successfully
        """
        # Get metadata
        metadata = self.model_repo.get_model_by_id(model_id)
        if not metadata:
            return False

        # Delete file
        file_path = Path(metadata.file_path)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Deleted model file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete model file: {e}")

        # Delete metadata
        return self.model_repo.delete_model(model_id)

    def list_user_models(
        self,
        user_id: str,
        model_type: Optional[str] = None
    ) -> list[ModelMetadata]:
        """
        List models for a user.

        Args:
            user_id: User ID
            model_type: Optional filter by model type

        Returns:
            List of ModelMetadata objects
        """
        return self.model_repo.get_user_models(user_id, model_type)

    @staticmethod
    def _compute_data_hash(data: pd.DataFrame) -> str:
        """
        Compute hash of training data.

        Args:
            data: Training data DataFrame

        Returns:
            Hash string
        """
        # Use shape and column names for hash
        hash_input = f"{data.shape}_{','.join(data.columns)}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    @staticmethod
    def _generate_model_id(
        user_id: str,
        model_type: str,
        data_hash: Optional[str]
    ) -> str:
        """
        Generate unique model ID.

        Args:
            user_id: User ID
            model_type: Model type
            data_hash: Data hash

        Returns:
            Model ID string
        """
        import uuid
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        if data_hash:
            return f"{model_type}_{user_id[:8]}_{data_hash[:8]}_{timestamp}_{unique_id}"
        else:
            return f"{model_type}_{user_id[:8]}_{timestamp}_{unique_id}"
