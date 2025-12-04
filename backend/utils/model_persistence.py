"""
Model persistence utilities for saving and loading trained models.

This module provides functionality to persist CausalGraphRecommender models
to disk, enabling model reuse across backend restarts.
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from backend.core.logging import get_logger

logger = get_logger(__name__)


class ModelPersistence:
    """
    Handles saving and loading of trained models to/from disk.
    
    Models are saved in the following structure:
    saved_models/
    └── model_id/
        ├── model.pkl        # Serialized model object (joblib)
        └── metadata.json    # Model metadata
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize ModelPersistence.
        
        Args:
            base_dir: Base directory for saving models. If None, uses PROJECT_ROOT/backend/saved_models
        """
        if base_dir is None:
            from backend.utils import PROJECT_ROOT
            base_dir = PROJECT_ROOT / "backend" / "saved_models"
        
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()
    
    def _ensure_base_dir(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Model persistence directory: {self.base_dir}")
    
    def _get_model_dir(self, model_id: str) -> Path:
        """Get directory path for a specific model."""
        return self.base_dir / model_id
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get file path for model pickle."""
        return self._get_model_dir(model_id) / "model.pkl"
    
    def _get_metadata_path(self, model_id: str) -> Path:
        """Get file path for metadata JSON."""
        return self._get_model_dir(model_id) / "metadata.json"
    
    def save_model(
        self,
        model_id: str,
        recommender: Any,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model_id: Unique model identifier
            recommender: CausalGraphRecommender instance
            metadata: Model metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = self._get_model_dir(model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model using joblib
            model_path = self._get_model_path(model_id)
            joblib.dump(recommender, model_path, compress=3)
            
            # Save metadata as JSON
            metadata_path = self._get_metadata_path(model_id)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model saved to disk", model_id=model_id, path=str(model_dir))
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model", model_id=model_id, error=str(e), exc_info=True)
            return False
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load a trained model from disk.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            CausalGraphRecommender instance if successful, None otherwise
        """
        try:
            model_path = self._get_model_path(model_id)
            
            if not model_path.exists():
                logger.warning(f"Model file not found", model_id=model_id, path=str(model_path))
                return None
            
            recommender = joblib.load(model_path)
            logger.info(f"Model loaded from disk", model_id=model_id)
            return recommender
            
        except Exception as e:
            logger.error(f"Failed to load model", model_id=model_id, error=str(e), exc_info=True)
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Metadata dictionary if successful, None otherwise
        """
        try:
            metadata_path = self._get_metadata_path(model_id)
            
            if not metadata_path.exists():
                logger.warning(f"Metadata file not found", model_id=model_id)
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load metadata", model_id=model_id, error=str(e))
            return None
    
    def delete_saved_model(self, model_id: str) -> bool:
        """
        Delete a saved model from disk.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = self._get_model_dir(model_id)
            
            if not model_dir.exists():
                logger.warning(f"Model directory not found", model_id=model_id)
                return False
            
            # Delete all files in model directory
            for file_path in model_dir.iterdir():
                file_path.unlink()
            
            # Delete directory
            model_dir.rmdir()
            
            logger.info(f"Model deleted from disk", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model", model_id=model_id, error=str(e), exc_info=True)
            return False
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata.
        
        Returns:
            List of dictionaries containing model_id and metadata
        """
        saved_models = []
        
        try:
            if not self.base_dir.exists():
                return []
            
            for model_dir in self.base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_id = model_dir.name
                metadata = self.get_model_metadata(model_id)
                
                if metadata:
                    saved_models.append({
                        'model_id': model_id,
                        'metadata': metadata
                    })
            
            logger.debug(f"Found {len(saved_models)} saved models")
            return saved_models
            
        except Exception as e:
            logger.error(f"Failed to list saved models", error=str(e), exc_info=True)
            return []
    
    def model_exists_on_disk(self, model_id: str) -> bool:
        """
        Check if a model exists on disk.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if model exists, False otherwise
        """
        model_path = self._get_model_path(model_id)
        return model_path.exists()


# Singleton instance
model_persistence = ModelPersistence()
