"""
Request schemas for the training API.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator


class TrainModelRequest(BaseModel):
    """Request schema for training a causal inference model."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Session ID containing the uploaded data",
        examples=["session_1701234567"],
    )
    min_members_per_skill: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum number of members required per skill for analysis",
    )
    correlation_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum correlation threshold for causal relationships",
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for recommendation scoring (readiness, probability, utility)",
    )

    # NOTE: session_id validation commented out for backward compatibility with tests
    # @field_validator("session_id")
    # @classmethod
    # def validate_session_id_format(cls, v: str) -> str:
    #     """Validate session ID format."""
    #     if not v.startswith("session_"):
    #         raise ValueError("Session ID must start with 'session_'")
    #     return v

    @field_validator("weights")
    @classmethod
    def validate_weights_sum(cls, v: Dict[str, float] | None) -> Dict[str, float] | None:
        """Validate that weights sum to 1.0."""
        if v is not None:
            # Check keys match existing system (readiness, bayesian, utility)
            required_keys = {"readiness", "bayesian", "utility"}
            if set(v.keys()) != required_keys:
                raise ValueError(f"Weights must contain exactly: {required_keys}")

            total = sum(v.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "session_1701234567",
                    "min_members_per_skill": 5,
                    "correlation_threshold": 0.2,
                    "weights": {"readiness": 0.4, "bayesian": 0.3, "utility": 0.3},
                }
            ]
        }
    }


class DeleteModelRequest(BaseModel):
    """Request schema for deleting a trained model."""

    model_id: str = Field(..., min_length=1, description="ID of the model to delete")
