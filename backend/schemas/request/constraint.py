"""
Request schemas for constraint management.
"""

from pydantic import BaseModel, Field
from typing import Optional


class AddConstraintRequest(BaseModel):
    """Request to add a constraint to a session's causal graph."""
    
    from_skill: str = Field(..., description="Source skill name")
    to_skill: str = Field(..., description="Target skill name")
    constraint_type: str = Field(
        ...,
        description="Type of constraint: 'required', 'forbidden', or 'deleted'"
    )
    value: Optional[float] = Field(
        None,
        description="Optional value for 'required' constraints (default: 0.5)"
    )
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class ApplyConstraintsRequest(BaseModel):
    """Request to apply constraints to an existing model."""
    
    model_id: str = Field(..., description="Model identifier to apply constraints to")
