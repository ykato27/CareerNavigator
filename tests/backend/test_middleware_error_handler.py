"""
Unit tests for backend/middleware/error_handler.py
"""
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.middleware.error_handler import error_handler_middleware
from backend.core.exceptions import AppException


class TestErrorHandlerMiddleware:
    """Tests for error_handler_middleware."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        request = Mock(spec=Request)
        request.state = Mock()
        request.url.path = "/test"
        request.method = "GET"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Mock call_next function."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_success_pass_through(self, mock_request, mock_call_next):
        """Test that middleware passes through successful requests."""
        expected_response = Response(content="success", status_code=200)
        mock_call_next.return_value = expected_response

        response = await error_handler_middleware(mock_request, mock_call_next)

        assert response == expected_response
        assert hasattr(mock_request.state, "trace_id")
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_app_exception_handling(self, mock_request, mock_call_next):
        """Test handling of AppException."""
        app_exception = AppException(
            error_code="TEST_ERROR",
            message="Test error message",
            status_code=400,
            details={"field": "value"}
        )
        mock_call_next.side_effect = app_exception

        response = await error_handler_middleware(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        
        import json
        body = json.loads(response.body)
        assert body["error"]["code"] == "TEST_ERROR"
        assert body["error"]["message"] == "Test error message"
        assert body["error"]["details"] == {"field": "value"}
        assert "trace_id" in body["error"]

    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self, mock_request, mock_call_next):
        """Test handling of unexpected exceptions."""
        mock_call_next.side_effect = ValueError("Unexpected error")

        response = await error_handler_middleware(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 500
        
        import json
        body = json.loads(response.body)
        assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert body["error"]["message"] == "An unexpected error occurred"
        assert "trace_id" in body["error"]
