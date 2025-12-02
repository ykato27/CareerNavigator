"""
Response schemas for the recommendation API.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """A single skill recommendation."""
    
    skill_code: str = Field(..., description="Skill code")
    skill_name: str = Field(..., description="Skill name")
    category: str = Field(..., description="Skill category")
    readiness_score: float = Field(..., description="Readiness score (0-1)")
    probability_score: float = Field(..., description="Acquisition probability score (0-1)")
    utility_score: float = Field(..., description="Utility/usefulness score (0-1)")
    final_score: float = Field(..., description="Weighted final score")
    reason: str = Field(default="", description="Explanation for this recommendation")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of prerequisite skills"
    )


class RecommendationsResponse(BaseModel):
    """Response schema for skill recommendations."""
    
    success: bool = Field(True, description="Indicates successful operation")
    model_id: str = Field(..., description="ID of the model used")
    member_id: str = Field(..., description="Member code")
    member_name: str = Field(default="", description="Member name (if available)")
    recommendations: List[RecommendationItem] = Field(
        ...,
        description="List of recommended skills"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., weights used, statistics)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "model_id": "model_abc123",
                "member_id": "M001",
                "member_name": "山田太郎",
                "recommendations": [
                    {
                        "skill_code": "SKILL001",
                        "skill_name": "Python",
                        "category": "プログラミング",
                        "readiness_score": 0.85,
                        "probability_score": 0.75,
                        "utility_score": 0.90,
                        "final_score": 0.83,
                        "reason": "Based on your current skills in data analysis",
                        "dependencies": ["SKILL002", "SKILL003"]
                    }
                ],
                "metadata": {
                    "weights": {
                        "readiness": 0.4,
                        "probability": 0.3,
                        "utility": 0.3
                    },
                    "total_candidates": 50
                }
            }
        }
    }
