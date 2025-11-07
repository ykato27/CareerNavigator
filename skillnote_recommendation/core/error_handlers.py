"""
Error Handling Utilities

This module provides standardized error handling mechanisms
for the application, including logging, error formatting,
and user-friendly error messages.
"""

import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps
import streamlit as st

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Raised when data processing fails."""

    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    pass


class RecommendationError(Exception):
    """Raised when recommendation generation fails."""

    pass


class ErrorHandler:
    """
    Centralized error handling utility class.

    Provides methods for consistent error logging, user notification,
    and error recovery strategies.
    """

    @staticmethod
    def log_error(error: Exception, context: str, level: int = logging.ERROR) -> None:
        """
        Log an error with context information.

        Args:
            error: The exception that occurred
            context: Context description (e.g., "data loading", "model training")
            level: Logging level (default: ERROR)

        Examples:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     ErrorHandler.log_error(e, "processing data")
        """
        error_type = type(error).__name__
        error_msg = str(error)

        logger.log(level, f"Error in {context}: {error_type}: {error_msg}", exc_info=True)

    @staticmethod
    def format_user_message(
        error: Exception, context: str, suggestions: Optional[list] = None
    ) -> str:
        """
        Format a user-friendly error message.

        Args:
            error: The exception that occurred
            context: Context description
            suggestions: List of suggested actions for the user

        Returns:
            Formatted error message for display to users

        Examples:
            >>> msg = ErrorHandler.format_user_message(
            ...     ValueError("Invalid data"),
            ...     "loading CSV",
            ...     ["Check file format", "Verify column names"]
            ... )
        """
        error_type = type(error).__name__
        error_msg = str(error)

        message = f"❌ Error during {context}\n\n"
        message += f"**Error Type:** {error_type}\n\n"
        message += f"**Details:** {error_msg}\n\n"

        if suggestions:
            message += "**💡 Suggestions:**\n"
            for suggestion in suggestions:
                message += f"- {suggestion}\n"

        return message

    @staticmethod
    def display_streamlit_error(
        error: Exception,
        context: str,
        suggestions: Optional[list] = None,
        show_traceback: bool = False,
    ) -> None:
        """
        Display error in Streamlit with consistent formatting.

        Args:
            error: The exception that occurred
            context: Context description
            suggestions: List of suggested actions
            show_traceback: Whether to show full traceback in expander

        Examples:
            >>> try:
            ...     load_data()
            ... except Exception as e:
            ...     ErrorHandler.display_streamlit_error(
            ...         e, "loading data",
            ...         suggestions=["Check file path", "Verify permissions"]
            ...     )
        """
        # Log the error
        ErrorHandler.log_error(error, context)

        # Display user-friendly message
        message = ErrorHandler.format_user_message(error, context, suggestions)
        st.error(message)

        # Optionally show technical details
        if show_traceback:
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())

    @staticmethod
    def handle_data_processing_error(func: Callable) -> Callable:
        """
        Decorator for handling data processing errors.

        Args:
            func: Function to wrap with error handling

        Returns:
            Wrapped function with error handling

        Examples:
            >>> @ErrorHandler.handle_data_processing_error
            ... def load_csv(path):
            ...     return pd.read_csv(path)
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Data processing failed in {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise DataProcessingError(error_msg) from e

        return wrapper

    @staticmethod
    def safe_execute(
        func: Callable, *args, default: Any = None, context: str = "operation", **kwargs
    ) -> Any:
        """
        Safely execute a function with error handling.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            default: Default value to return on error
            context: Context description for logging
            **kwargs: Keyword arguments for func

        Returns:
            Function result or default value if error occurs

        Examples:
            >>> result = ErrorHandler.safe_execute(
            ...     risky_func, arg1, arg2,
            ...     default=None,
            ...     context="data processing"
            ... )
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.log_error(e, context, level=logging.WARNING)
            logger.warning(f"Returning default value: {default}")
            return default


class ErrorRecovery:
    """
    Error recovery strategies.

    Provides methods for recovering from common error scenarios
    and implementing fallback behaviors.
    """

    @staticmethod
    def retry_on_failure(
        func: Callable, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0
    ) -> Callable:
        """
        Decorator to retry a function on failure with exponential backoff.

        Args:
            func: Function to wrap
            max_retries: Maximum number of retry attempts
            delay: Initial delay in seconds
            backoff: Backoff multiplier for each retry

        Returns:
            Wrapped function with retry logic

        Examples:
            >>> @ErrorRecovery.retry_on_failure(max_retries=3)
            ... def unstable_api_call():
            ...     return requests.get(url)
        """
        import time

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for "
                            f"{func.__name__}: {str(e)}. Retrying in "
                            f"{current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper

    @staticmethod
    def with_fallback(
        primary_func: Callable, fallback_func: Callable, context: str = "operation"
    ) -> Any:
        """
        Execute primary function with fallback on error.

        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            context: Context description for logging

        Returns:
            Result from primary or fallback function

        Examples:
            >>> result = ErrorRecovery.with_fallback(
            ...     lambda: expensive_computation(),
            ...     lambda: cached_result(),
            ...     context="data calculation"
            ... )
        """
        try:
            return primary_func()
        except Exception as e:
            logger.warning(f"Primary {context} failed: {str(e)}. Using fallback.")
            return fallback_func()
