"""
Unit tests for backend/middleware/logging.py
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, Response

from backend.middleware.logging import logging_middleware


class TestLoggingMiddleware:
    """Tests for logging_middleware."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.trace_id = "test-trace-id"
        request.url.path = "/test"
        request.method = "GET"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.query_params = {"q": "test"}
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Mock call_next function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_logging_success(self, mock_request, mock_call_next):
        """Test logging for successful requests."""
        expected_response = Response(content="success", status_code=200)
        mock_call_next.return_value = expected_response

        with patch("backend.middleware.logging.logger") as mock_logger, \
             patch("backend.middleware.logging.bind_request_context") as mock_bind, \
             patch("backend.middleware.logging.clear_request_context") as mock_clear:
            
            response = await logging_middleware(mock_request, mock_call_next)

            assert response == expected_response
            
            # Verify context binding
            mock_bind.assert_called_once_with(
                trace_id="test-trace-id",
                method="GET",
                path="/test",
                client_ip="127.0.0.1"
            )
            
            # Verify logging
            assert mock_logger.info.call_count == 2
            mock_logger.info.assert_any_call("Request started", query_params={"q": "test"})
            
            # Verify completion log contains status code and duration
            args, kwargs = mock_logger.info.call_args_list[1]
            assert args[0] == "Request completed"
            assert kwargs["status_code"] == 200
            assert "duration_ms" in kwargs
            
            # Verify context clearing
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_logging_exception(self, mock_request, mock_call_next):
        """Test logging when exception occurs."""
        mock_call_next.side_effect = ValueError("Test error")

        with patch("backend.middleware.logging.logger") as mock_logger, \
             patch("backend.middleware.logging.bind_request_context") as mock_bind, \
             patch("backend.middleware.logging.clear_request_context") as mock_clear:
            
            with pytest.raises(ValueError):
                await logging_middleware(mock_request, mock_call_next)

            # Verify error logging
            mock_logger.error.assert_called_once()
            args, kwargs = mock_logger.error.call_args
            assert args[0] == "Request failed"
            assert kwargs["error_type"] == "ValueError"
            assert "duration_ms" in kwargs
            
            # Verify context clearing (finally block)
            mock_clear.assert_called_once()
