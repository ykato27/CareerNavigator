"""
Core exceptions for the CareerNavigator application.

This module defines a hierarchy of custom exceptions used throughout the application.
All exceptions inherit from AppException which provides consistent error handling.
"""

from typing import Optional, Dict, Any


class AppException(Exception):
    """Base exception for all application errors.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code (e.g., "SESSION_NOT_FOUND")
        status_code: HTTP status code to return
        details: Additional error details/context
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": {"code": self.error_code, "message": self.message, "details": self.details}
        }


# Resource Not Found Exceptions
class ResourceNotFoundException(AppException):
    """Base exception for resource not found errors."""

    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            message=f"{resource_type} '{resource_id}' not found",
            error_code=f"{resource_type.upper()}_NOT_FOUND",
            status_code=404,
            **kwargs,
        )


class SessionNotFoundException(ResourceNotFoundException):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        super().__init__(resource_type="Session", resource_id=session_id)


class ModelNotFoundException(ResourceNotFoundException):
    """Raised when a trained model is not found."""

    def __init__(self, model_id: str):
        super().__init__(resource_type="Model", resource_id=model_id)


class MemberNotFoundException(ResourceNotFoundException):
    """Raised when a member is not found in the data."""

    def __init__(self, member_code: str):
        super().__init__(resource_type="Member", resource_id=member_code)


# Validation Exceptions
class ValidationException(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details,
            **kwargs,
        )


class InvalidWeightsException(ValidationException):
    """Raised when recommendation weights are invalid."""

    def __init__(self, message: str = "Weights must sum to 1.0"):
        super().__init__(message=message)


class InvalidFileFormatException(ValidationException):
    """Raised when uploaded file format is invalid."""

    def __init__(self, filename: str, expected_format: str):
        super().__init__(
            message=f"Invalid file format for '{filename}'. Expected: {expected_format}",
            field="file",
        )


# Business Logic Exceptions
class BusinessLogicException(AppException):
    """Base exception for business logic errors."""

    def __init__(self, message: str, error_code: str, **kwargs):
        super().__init__(message=message, error_code=error_code, status_code=400, **kwargs)


class InsufficientDataException(BusinessLogicException):
    """Raised when there is insufficient data for an operation."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Insufficient data for {operation}: {reason}", error_code="INSUFFICIENT_DATA"
        )


class ModelNotTrainedException(BusinessLogicException):
    """Raised when trying to use a model that hasn't been trained yet."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Model for session '{session_id}' has not been trained",
            error_code="MODEL_NOT_TRAINED",
        )


class NoRecommendationsException(BusinessLogicException):
    """Raised when no recommendations can be generated."""

    def __init__(self, member_code: str, reason: Optional[str] = None):
        message = f"No recommendations available for member '{member_code}'"
        if reason:
            message += f": {reason}"
        super().__init__(message=message, error_code="NO_RECOMMENDATIONS")


# Server Exceptions
class InternalServerException(AppException):
    """Raised for unexpected internal server errors."""

    def __init__(self, message: str = "An unexpected error occurred", **kwargs):
        super().__init__(
            message=message, error_code="INTERNAL_SERVER_ERROR", status_code=500, **kwargs
        )


class ServiceUnavailableException(AppException):
    """Raised when a required service is unavailable."""

    def __init__(self, service_name: str):
        super().__init__(
            message=f"Service '{service_name}' is currently unavailable",
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
        )
