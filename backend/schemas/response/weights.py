"""
Response schemas for the weights API.
"""
from typing import Dict
from pydantic import BaseModel, Field


class WeightsResponse(BaseModel):
    """Response schema for weights operations."""
    
    success: bool = Field(True, description="Indicates successful operation")
    model_id: str = Field(..., description="ID of the model")
    weights: Dict[str, float] = Field(
        ...,
        description="Current recommendation weights"
    )
    message: str = Field(
        default="",
        description="Optional message"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "model_id": "model_abc123",
                "weights": {
                    "readiness": 0.4,
                    "probability": 0.3,
                    "utility": 0.3
                },
                "message": "Weights updated successfully"
            }
        }
    }
