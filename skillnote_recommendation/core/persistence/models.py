"""
Data models for persistence layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum


class RecommendationMethod(Enum):
    """Recommendation method types."""

    NMF = "nmf"
    GRAPH = "graph"
    HYBRID = "hybrid"
    CAREER_PATTERN = "career_pattern"


@dataclass
class User:
    """User model for authentication and data management."""

    user_id: str
    username: str
    email: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create from dictionary."""
        data = data.copy()
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_login"):
            data["last_login"] = datetime.fromisoformat(data["last_login"])
        return cls(**data)


@dataclass
class RecommendationHistory:
    """Recommendation history record."""

    history_id: Optional[str] = None
    user_id: str = ""
    member_code: str = ""
    member_name: str = ""
    method: str = ""
    timestamp: Optional[datetime] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    reference_persons: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "history_id": self.history_id,
            "user_id": self.user_id,
            "member_code": self.member_code,
            "member_name": self.member_name,
            "method": self.method,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "recommendations": self.recommendations,
            "reference_persons": self.reference_persons,
            "parameters": self.parameters,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationHistory":
        """Create from dictionary."""
        data = data.copy()
        if data.get("timestamp"):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ModelMetadata:
    """Model metadata for saved models."""

    model_id: str
    user_id: str
    model_type: str  # "nmf", "graph", "hybrid"
    created_at: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    file_path: str = ""
    data_hash: Optional[str] = None  # Hash of training data
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "user_id": self.user_id,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "file_path": self.file_path,
            "data_hash": self.data_hash,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data = data.copy()
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class UserSession:
    """User session for managing application state."""

    session_id: str
    user_id: str
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    data_loaded: bool = False
    model_trained: bool = False
    current_model_id: Optional[str] = None
    state_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "data_loaded": self.data_loaded,
            "model_trained": self.model_trained,
            "current_model_id": self.current_model_id,
            "state_data": self.state_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        """Create from dictionary."""
        data = data.copy()
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_active"):
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        return cls(**data)
