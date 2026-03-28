"""
Request schemas for the recommendation API.
"""

from pydantic import BaseModel, Field, field_validator


class GetRecommendationsRequest(BaseModel):
    """Request schema for getting skill recommendations."""

    model_id: str = Field(
        ..., min_length=1, description="ID of the trained model to use", examples=["model_abc123"]
    )
    member_id: str = Field(
        ...,
        min_length=1,
        description="Member code to get recommendations for",
        examples=["M001", "m48"],
    )
    top_n: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")

    @field_validator("model_id")
    @classmethod
    def validate_model_id_format(cls, v: str) -> str:
        """Validate model ID format."""
        if not v.startswith("model_"):
            raise ValueError("Model ID must start with 'model_'")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{"model_id": "model_abc123", "member_id": "M001", "top_n": 10}]
        }
    }
