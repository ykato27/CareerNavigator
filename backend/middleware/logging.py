"""
Request/Response logging middleware for the CareerNavigator application.

This middleware logs all incoming requests and outgoing responses with
timing information.
"""
import time
from typing import Callable
from fastapi import Request, Response
from backend.core.logging import get_logger, bind_request_context, clear_request_context

logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log all requests and responses with timing information.
    
    Logs:
    - Request method, path, client IP
    - Response status code
    - Request processing duration
    - Trace ID for request correlation
    """
    # Bind request context for all subsequent logs in this request
    trace_id = getattr(request.state, 'trace_id', 'unknown')
    bind_request_context(
        trace_id=trace_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    # Log incoming request
    logger.info(
        "Request started",
        query_params=dict(request.query_params) if request.query_params else {}
    )
    
    # Process request and measure duration
    start_time = time.time()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log successful response
        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )
        
        return response
    
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        
        # Log failed request (exception will be handled by error_handler_middleware)
        logger.error(
            "Request failed",
            error_type=type(exc).__name__,
            duration_ms=round(duration_ms, 2)
        )
        raise
    
    finally:
        # Clear request context after response
        clear_request_context()
