"""
Error handling middleware for the CareerNavigator application.

This middleware catches all exceptions and converts them to standardized
error responses with proper logging.
"""

import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from backend.core.exceptions import AppException
from backend.core.logging import get_logger

logger = get_logger(__name__)


async def error_handler_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to handle all exceptions and convert them to standardized error responses.

    This middleware:
    1. Generates a unique trace_id for each request
    2. Catches all exceptions
    3. Logs errors with context
    4. Returns standardized error responses
    """
    # Generate trace ID for this request
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id

    try:
        response = await call_next(request)
        return response

    except AppException as exc:
        # Handle known application exceptions
        logger.warning(
            "Application exception occurred",
            error_code=exc.error_code,
            error_message=exc.message,
            status_code=exc.status_code,
            trace_id=trace_id,
            path=request.url.path,
            method=request.method,
            details=exc.details,
        )

        error_response = exc.to_dict()
        error_response["error"]["trace_id"] = trace_id

        return JSONResponse(status_code=exc.status_code, content=error_response)

    except Exception as exc:
        # Handle unexpected exceptions
        logger.error(
            "Unexpected exception occurred",
            error_type=type(exc).__name__,
            error_message=str(exc),
            trace_id=trace_id,
            path=request.url.path,
            method=request.method,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {},
                    "trace_id": trace_id,
                }
            },
        )
