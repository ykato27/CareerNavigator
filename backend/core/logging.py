"""
Structured logging configuration for the CareerNavigator application.

This module sets up structlog for JSON-formatted, structured logging with
trace IDs, timestamps, and contextual information.
"""

import logging
import sys
from typing import Any
import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to all log entries."""
    event_dict["app"] = "career-navigator"
    event_dict["environment"] = "development"  # Should come from config
    return event_dict


def configure_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output logs in JSON format
    """
    # Define processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
    ]

    if json_logs:
        # Production: JSON format
        processors.extend(
            [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
        )
    else:
        # Development: Console format
        processors.extend([structlog.dev.set_exc_info, structlog.dev.ConsoleRenderer()])

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
        stream=sys.stdout,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Context managers for adding request context
def bind_request_context(**kwargs: Any) -> None:
    """
    Bind request-specific context to all subsequent log entries.

    Example:
        bind_request_context(trace_id="abc123", user_id="user456")
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear request-specific context."""
    structlog.contextvars.clear_contextvars()
