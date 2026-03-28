"""
Request schemas for the weights API.
"""

from typing import Dict
from pydantic import BaseModel, Field, field_validator


class UpdateWeightsRequest(BaseModel):
    """Request schema for updating recommendation weights."""

    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model to update weights for",
        examples=["model_abc123"],
    )
    weights: Dict[str, float] = Field(
        ...,
        description="New weights for recommendation scoring",
        examples=[{"readiness": 0.5, "bayesian": 0.3, "utility": 0.2}],
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights sum to 1.0."""
        required_keys = {"readiness", "bayesian", "utility"}
        if set(v.keys()) != required_keys:
            raise ValueError(f"Weights must contain exactly: {required_keys}")

        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        # Check all values are non-negative
        if any(w < 0 for w in v.values()):
            raise ValueError("All weights must be non-negative")

        return v


class GetWeightsRequest(BaseModel):
    """Request schema for getting current weights."""

    model_id: str = Field(..., min_length=1, description="ID of the model to get weights for")


class OptimizeWeightsRequest(BaseModel):
    """Request schema for optimizing weights."""

    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model to optimize weights for",
        examples=["model_abc123"],
    )
    n_trials: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of optimization trials",
        examples=[50],
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Number of parallel jobs (-1 for all cores)",
        examples=[-1],
    )
    holdout_ratio: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Ratio of skills to holdout for evaluation",
        examples=[0.2],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to evaluate",
        examples=[10],
    )
