"""
Response schemas for the training API.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    """Summary information about a trained model."""

    model_id: str = Field(..., description="Unique identifier for the trained model")
    session_id: str = Field(..., description="Session ID used for training")
    created_at: Optional[str] = Field(None, description="Model creation timestamp")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Training parameters used")
    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Model statistics (e.g., number of nodes, edges)"
    )


class TrainModelResponse(BaseModel):
    """Response schema for successful model training."""

    success: bool = Field(True, description="Indicates successful training")
    model_id: str = Field(..., description="ID of the trained model")
    message: str = Field(..., description="Success message")
    summary: ModelSummary = Field(..., description="Model summary information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "model_id": "model_abc123",
                "message": "Model trained successfully",
                "summary": {
                    "model_id": "model_abc123",
                    "session_id": "session_1701234567",
                    "created_at": "2024-12-03T07:00:00Z",
                    "parameters": {"min_members_per_skill": 5, "correlation_threshold": 0.2},
                    "statistics": {"num_skills": 150, "num_relationships": 450},
                },
            }
        }
    }


class GetModelSummaryResponse(BaseModel):
    """Response schema for getting model summary."""

    success: bool = Field(True, description="Indicates successful operation")
    summary: ModelSummary = Field(..., description="Model summary information")


class DeleteModelResponse(BaseModel):
    """Response schema for model deletion."""

    success: bool = Field(True, description="Indicates successful deletion")
    message: str = Field(..., description="Deletion confirmation message")
    model_id: str = Field(..., description="ID of the deleted model")
