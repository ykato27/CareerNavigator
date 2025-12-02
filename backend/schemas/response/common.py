"""
Common response schemas used across all API endpoints.
"""
from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field


T = TypeVar('T')


class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: ErrorDetail
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "SESSION_NOT_FOUND",
                    "message": "Session 'session_123' not found",
                    "details": {},
                    "trace_id": "abc-def-123"
                }
            }
        }


class SuccessResponse(BaseModel, Generic[T]):
    """Generic success response wrapper."""
    success: bool = Field(True, description="Indicates successful operation")
    data: T = Field(..., description="Response payload")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"message": "Operation completed successfully"}
            }
        }


class MessageResponse(BaseModel):
    """Simple message response."""
    success: bool = Field(True, description="Indicates successful operation")
    message: str = Field(..., description="Success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Model trained successfully"
            }
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    success: bool = Field(True, description="Indicates successful operation")
    data: list[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": [],
                "total": 100,
                "page": 1,
                "page_size": 10
            }
        }
