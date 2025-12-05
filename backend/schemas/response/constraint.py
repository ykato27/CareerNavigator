"""
Response schemas for constraint management.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class Constraint(BaseModel):
    """A single constraint."""
    
    id: str = Field(..., description="Unique constraint identifier")
    from_skill: str = Field(..., description="Source skill name")
    to_skill: str = Field(..., description="Target skill name")
    constraint_type: str = Field(..., description="Type: 'required', 'forbidden', 'deleted'")
    value: Optional[float] = Field(None, description="Value for 'required' constraints")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    created_by: Optional[str] = Field(None, description="User who created the constraint")


class GetConstraintsResponse(BaseModel):
    """Response for getting constraints."""
    
    success: bool = Field(default=True)
    session_id: str = Field(..., description="Session identifier")
    constraints: List[Constraint] = Field(..., description="List of constraints")
    count: int = Field(..., description="Number of constraints")


class AddConstraintResponse(BaseModel):
    """Response for adding a constraint."""
    
    success: bool = Field(default=True)
    constraint: Constraint = Field(..., description="The created/updated constraint")
    message: str = Field(default="制約を追加しました")


class DeleteConstraintResponse(BaseModel):
    """Response for deleting a constraint."""
    
    success: bool = Field(default=True)
    message: str = Field(default="制約を削除しました")


class ApplyConstraintsResponse(BaseModel):
    """Response for applying constraints to a model."""
    
    success: bool = Field(default=True)
    model_id: str = Field(..., description="Model identifier")
    applied_count: int = Field(..., description="Number of constraints applied")
    skipped_count: int = Field(..., description="Number of constraints skipped")
    message: str = Field(default="制約を適用しました")
