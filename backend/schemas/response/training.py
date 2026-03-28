"""
Response schemas for the training API.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    """Summary information about a trained model."""

    model_id: str = Field(..., description="Unique identifier for the trained model")
    session_id: Optional[str] = Field(None, description="Session ID used for training")
    num_members: Optional[int] = Field(None, description="Number of members in the model")
    num_skills: Optional[int] = Field(None, description="Number of skills in the model")
    learning_time: Optional[float] = Field(None, description="Time taken to train (seconds)")
    total_time: Optional[float] = Field(None, description="Total processing time (seconds)")
    weights: Optional[Dict[str, float]] = Field(None, description="Recommendation weights")
    min_members_per_skill: Optional[int] = Field(
        None, description="Minimum members per skill parameter"
    )
    correlation_threshold: Optional[float] = Field(
        None, description="Correlation threshold parameter"
    )
    has_causal_graph: Optional[bool] = Field(None, description="Whether causal graph exists")


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
                    "num_members": 100,
                    "num_skills": 150,
                    "learning_time": 12.5,
                    "total_time": 15.3,
                    "weights": {"readiness": 0.4, "bayesian": 0.3, "utility": 0.3},
                    "min_members_per_skill": 5,
                    "correlation_threshold": 0.2,
                },
            }
        }
    }


class GetModelSummaryResponse(BaseModel):
    """Response schema for getting model summary."""

    success: bool = Field(True, description="Indicates successful operation")
    model_id: str = Field(..., description="Unique identifier for the trained model")
    num_members: Optional[int] = Field(None, description="Number of members in the model")
    num_skills: Optional[int] = Field(None, description="Number of skills in the model")
    weights: Optional[Dict[str, float]] = Field(None, description="Recommendation weights")
    has_causal_graph: Optional[bool] = Field(None, description="Whether causal graph exists")


class DeleteModelResponse(BaseModel):
    """Response schema for model deletion."""

    success: bool = Field(True, description="Indicates successful deletion")
    message: str = Field(..., description="Deletion confirmation message")
    model_id: str = Field(..., description="ID of the deleted model")
