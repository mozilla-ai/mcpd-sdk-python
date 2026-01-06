from unittest.mock import Mock, patch

import pytest
import requests

from mcpd import McpdClient
from mcpd.exceptions import (
    _PIPELINE_ERROR_FLOWS,
    PIPELINE_FLOW_REQUEST,
    PIPELINE_FLOW_RESPONSE,
    AuthenticationError,
    ConnectionError,
    McpdError,
    PipelineError,
    ServerNotFoundError,
    ServerUnhealthyError,
    TimeoutError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
)


class TestMcpdError:
    def test_mcpd_error_inheritance(self):
        """Test that McpdError inherits from Exception."""
        assert issubclass(McpdError, Exception)

    def test_mcpd_error_basic_creation(self):
        """Test basic McpdError creation."""
        error = McpdError("Test error message")
        assert str(error) == "Test error message"

    def test_mcpd_error_empty_message(self):
        """Test McpdError with empty message."""
        error = McpdError("")
        assert str(error) == ""

    def test_mcpd_error_none_message(self):
        """Test McpdError with None message."""
        error = McpdError(None)
        assert str(error) == "None"

    def test_mcpd_error_with_args(self):
        """Test McpdError with multiple arguments."""
        error = McpdError("Error", "with", "multiple", "args")
        assert "Error" in str(error)

    def test_mcpd_error_raising(self):
        """Test that McpdError can be raised and caught."""
        with pytest.raises(McpdError):
            raise McpdError("Test error")

    def test_mcpd_error_catching_as_exception(self):
        """Test that McpdError can be caught as Exception."""
        with pytest.raises(McpdError):
            raise McpdError("Test error")

    def test_mcpd_error_chaining(self):
        """Test error chaining with McpdError."""
        original_error = ValueError("Original error")

        try:
            raise original_error
        except ValueError as e:
            chained_error = McpdError("Chained error")
            chained_error.__cause__ = e

        assert chained_error.__cause__ is original_error
        assert str(chained_error) == "Chained error"

    def test_mcpd_error_with_format_string(self):
        """Test McpdError with format string."""
        server_name = "test_server"
        tool_name = "test_tool"
        error = McpdError(f"Error calling tool '{tool_name}' on server '{server_name}'")

        expected_message = "Error calling tool 'test_tool' on server 'test_server'"
        assert str(error) == expected_message

    def test_mcpd_error_attributes(self):
        """Test that McpdError has expected attributes."""
        error = McpdError("Test error")

        # Should have standard Exception attributes
        assert hasattr(error, "args")
        assert error.args == ("Test error",)

    def test_mcpd_error_repr(self):
        """Test string representation of McpdError."""
        error = McpdError("Test error")
        repr_str = repr(error)

        assert "McpdError" in repr_str
        assert "Test error" in repr_str

    def test_mcpd_error_instance_check(self):
        """Test isinstance checks with McpdError."""
        error = McpdError("Test error")

        assert isinstance(error, McpdError)
        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)

    def test_mcpd_error_equality(self):
        """Test equality comparison of McpdError instances."""
        error1 = McpdError("Same message")
        error2 = McpdError("Same message")
        error3 = McpdError("Different message")

        # Note: Exception instances are not equal even with same message
        # This is standard Python behavior
        assert error1 is not error2
        assert error1 is not error3

    def test_mcpd_error_with_complex_message(self):
        """Test McpdError with complex message containing various data types."""
        data = {"server": "test", "tool": "example", "params": [1, 2, 3]}
        error = McpdError(f"Complex error with data: {data}")

        assert "Complex error with data:" in str(error)
        assert "test" in str(error)
        assert "example" in str(error)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from McpdError."""

    def test_all_exceptions_inherit_from_mcpd_error(self):
        """Verify exception hierarchy."""
        assert issubclass(ConnectionError, McpdError)
        assert issubclass(AuthenticationError, McpdError)
        assert issubclass(PipelineError, McpdError)
        assert issubclass(ServerNotFoundError, McpdError)
        assert issubclass(ServerUnhealthyError, McpdError)
        assert issubclass(ToolNotFoundError, McpdError)
        assert issubclass(ToolExecutionError, McpdError)
        assert issubclass(TimeoutError, McpdError)
        assert issubclass(ValidationError, McpdError)

    def test_backward_compatibility(self):
        """Test that catching McpdError still works for all subclasses."""
        exceptions = [
            ConnectionError("test"),
            AuthenticationError("test"),
            PipelineError("test", server_name="server1", operation="server1.tool1", pipeline_flow="request"),
            ServerNotFoundError("test", server_name="server1"),
            ServerUnhealthyError("test", server_name="server1", health_status="timeout"),
            ToolNotFoundError("test", server_name="server1", tool_name="tool1"),
            ToolExecutionError("test"),
            TimeoutError("test"),
            ValidationError("test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except McpdError:
                pass  # Should catch all subclasses
            except Exception:
                pytest.fail(f"{exc.__class__.__name__} not caught by McpdError")


class TestConnectionError:
    """Test ConnectionError is raised appropriately."""

    @patch("requests.Session")
    def test_connection_error_on_servers(self, mock_session_class):
        """Test ConnectionError when cannot connect to daemon."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate connection error
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ConnectionError) as exc_info:
            client.servers()

        assert "Cannot connect to mcpd daemon" in str(exc_info.value)
        assert "localhost:8090" in str(exc_info.value)

    @patch("requests.Session")
    def test_connection_error_on_tool_call(self, mock_session_class):
        """Test ConnectionError when calling a tool."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ConnectionError) as exc_info:
            client._perform_call("test_server", "test_tool", {})

        assert "Cannot connect to mcpd daemon" in str(exc_info.value)

    @patch("requests.Session")
    def test_connection_error_on_server_health(self, mock_session_class):
        """Test ConnectionError when checking server health."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ConnectionError) as exc_info:
            client.server_health("test_server")

        assert "Cannot connect to mcpd daemon" in str(exc_info.value)
        assert "localhost:8090" in str(exc_info.value)


class TestAuthenticationError:
    """Test AuthenticationError is raised appropriately."""

    @patch("requests.Session")
    def test_authentication_error_401(self, mock_session_class):
        """Test AuthenticationError on 401 response."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate 401 error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.get.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090", api_key="bad-key")  # pragma: allowlist secret

        with pytest.raises(AuthenticationError) as exc_info:
            client.servers()

        assert "Authentication failed" in str(exc_info.value)


class TestServerNotFoundError:
    """Test ServerNotFoundError is raised appropriately."""

    @patch("requests.Session")
    def test_server_not_found_404(self, mock_session_class):
        """Test ServerNotFoundError on 404 when getting tools."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.get.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ServerNotFoundError) as exc_info:
            client._get_tool_definitions("nonexistent_server")

        assert "Server 'nonexistent_server' not found" in str(exc_info.value)
        assert exc_info.value.server_name == "nonexistent_server"

    def test_server_not_found_attributes(self):
        """Test ServerNotFoundError attributes."""
        exc = ServerNotFoundError("Server not found", server_name="test_server")
        assert exc.server_name == "test_server"


class TestServerUnhealthyError:
    """Test ServerUnhealthyError is raised appropriately."""

    @patch("requests.Session")
    def test_server_unhealthy_error(self, mock_session_class):
        """Test ServerUnhealthyError when server health is not 'ok'."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock server health response with unhealthy status
        mock_response = Mock()
        mock_response.json.return_value = {"name": "unhealthy_server", "status": "timeout"}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ServerUnhealthyError) as exc_info:
            client._raise_for_server_health("unhealthy_server")

        assert "Server 'unhealthy_server' is not healthy" in str(exc_info.value)
        assert exc_info.value.server_name == "unhealthy_server"
        assert exc_info.value.health_status == "timeout"

    def test_server_unhealthy_attributes(self):
        """Test ServerUnhealthyError attributes."""
        exc = ServerUnhealthyError("Server unhealthy", server_name="test_server", health_status="timeout")
        assert exc.server_name == "test_server"
        assert exc.health_status == "timeout"


class TestToolNotFoundError:
    """Test ToolNotFoundError is raised appropriately."""

    @patch("requests.Session")
    def test_tool_not_found_in_dynamic_caller(self, mock_session_class):
        """Test ToolNotFoundError when tool doesn't exist."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Mock successful server list with existing tool
        mock_response = Mock()
        mock_response.json.return_value = {"tools": [{"name": "existing_tool"}]}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ToolNotFoundError) as exc_info:
            # This will trigger the has_tool check in dynamic_caller
            client.call.test_server.nonexistent_tool()

        assert "Tool 'nonexistent_tool' not found on server 'test_server'" in str(exc_info.value)
        assert exc_info.value.server_name == "test_server"
        assert exc_info.value.tool_name == "nonexistent_tool"

    def test_tool_not_found_attributes(self):
        """Test ToolNotFoundError attributes."""
        exc = ToolNotFoundError("Tool not found", server_name="test_server", tool_name="test_tool")
        assert exc.server_name == "test_server"
        assert exc.tool_name == "test_tool"


class TestToolExecutionError:
    """Test ToolExecutionError is raised appropriately."""

    @patch("requests.Session")
    def test_tool_execution_error_500(self, mock_session_class):
        """Test ToolExecutionError on 500 server error."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate 500 error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ToolExecutionError) as exc_info:
            client._perform_call("test_server", "test_tool", {"param": "value"})

        assert "Server error when executing 'test_tool'" in str(exc_info.value)
        assert exc_info.value.server_name == "test_server"
        assert exc_info.value.tool_name == "test_tool"

    @patch("requests.Session")
    def test_tool_execution_error_400(self, mock_session_class):
        """Test ToolExecutionError on 400 bad request."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate 400 error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ToolExecutionError) as exc_info:
            client._perform_call("test_server", "test_tool", {"bad": "param"})

        assert "Error calling tool 'test_tool'" in str(exc_info.value)
        assert exc_info.value.server_name == "test_server"
        assert exc_info.value.tool_name == "test_tool"

    def test_tool_execution_error_attributes(self):
        """Test ToolExecutionError attributes."""
        details = {"error_code": "INVALID_PARAMS"}
        exc = ToolExecutionError("Execution failed", server_name="test_server", tool_name="test_tool", details=details)
        assert exc.server_name == "test_server"
        assert exc.tool_name == "test_tool"
        assert exc.details == details


class TestTimeoutError:
    """Test TimeoutError is raised appropriately."""

    @patch("requests.Session")
    def test_timeout_error_on_servers(self, mock_session_class):
        """Test TimeoutError when request times out."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Simulate timeout
        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(TimeoutError) as exc_info:
            client.servers()

        assert "Request timed out after 5 seconds" in str(exc_info.value)
        assert exc_info.value.operation == "list servers"
        assert exc_info.value.timeout == 5

    @patch("requests.Session")
    def test_timeout_error_on_tool_call(self, mock_session_class):
        """Test TimeoutError when tool execution times out."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(TimeoutError) as exc_info:
            client._perform_call("slow_server", "slow_tool", {})

        assert "Tool execution timed out after 30 seconds" in str(exc_info.value)
        assert exc_info.value.operation == "slow_server.slow_tool"
        assert exc_info.value.timeout == 30

    @patch("requests.Session")
    def test_timeout_error_on_server_health(self, mock_session_class):
        """Test TimeoutError when server health times out."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_session.get.side_effect = requests.exceptions.Timeout("Request timed out")

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(TimeoutError) as exc_info:
            client.server_health("slow_server")

        assert "Request timed out after 5 seconds" in str(exc_info.value)
        assert exc_info.value.operation == "get health of slow_server"
        assert exc_info.value.timeout == 5

    def test_timeout_error_attributes(self):
        """Test TimeoutError attributes."""
        exc = TimeoutError("Operation timed out", operation="fetch_data", timeout=30.0)
        assert exc.operation == "fetch_data"
        assert exc.timeout == 30.0


class TestValidationError:
    """Test ValidationError is raised appropriately."""

    def test_validation_error_attributes(self):
        """Test ValidationError stores validation errors."""
        errors = ["Missing field 'name'", "Invalid type for 'age'"]
        exc = ValidationError("Validation failed", validation_errors=errors)

        assert exc.validation_errors == errors
        assert "Validation failed" in str(exc)

    def test_validation_error_empty_list(self):
        """Test ValidationError with no specific errors."""
        exc = ValidationError("Validation failed")
        assert exc.validation_errors == []


class TestPipelineError:
    """Test PipelineError is raised appropriately."""

    def test_pipeline_error_attributes(self):
        """Test PipelineError stores all attributes."""
        exc = PipelineError(
            "Pipeline failure",
            server_name="test_server",
            operation="test_server.test_tool",
            pipeline_flow=PIPELINE_FLOW_REQUEST,
        )
        assert exc.server_name == "test_server"
        assert exc.operation == "test_server.test_tool"
        assert exc.pipeline_flow == PIPELINE_FLOW_REQUEST
        assert "Pipeline failure" in str(exc)

    def test_pipeline_error_response_flow(self):
        """Test PipelineError with response flow."""
        exc = PipelineError(
            "Response pipeline failed",
            server_name="time",
            operation="time.get_current_time",
            pipeline_flow=PIPELINE_FLOW_RESPONSE,
        )
        assert exc.pipeline_flow == PIPELINE_FLOW_RESPONSE

    def test_pipeline_error_minimal(self):
        """Test PipelineError with only message."""
        exc = PipelineError("Minimal error")
        assert exc.server_name is None
        assert exc.operation is None
        assert exc.pipeline_flow is None

    def test_pipeline_flow_constants(self):
        """Test pipeline flow constants have expected values."""
        assert PIPELINE_FLOW_REQUEST == "request"
        assert PIPELINE_FLOW_RESPONSE == "response"

    def test_pipeline_error_flows_mapping(self):
        """Test _PIPELINE_ERROR_FLOWS mapping from header values to flow constants."""
        assert _PIPELINE_ERROR_FLOWS["request-pipeline-failure"] == PIPELINE_FLOW_REQUEST
        assert _PIPELINE_ERROR_FLOWS["response-pipeline-failure"] == PIPELINE_FLOW_RESPONSE

    @patch("requests.Session")
    def test_pipeline_error_on_request_failure(self, mock_session_class):
        """Test PipelineError raised on 500 with request-pipeline-failure header."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {"Mcpd-Error-Type": "request-pipeline-failure"}
        mock_response.text = "Request pipeline processing failed"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(PipelineError) as exc_info:
            client._perform_call("test_server", "test_tool", {"param": "value"})

        assert exc_info.value.pipeline_flow == PIPELINE_FLOW_REQUEST
        assert exc_info.value.server_name == "test_server"
        assert exc_info.value.operation == "test_server.test_tool"
        assert "Request pipeline processing failed" in str(exc_info.value)

    @patch("requests.Session")
    def test_pipeline_error_on_response_failure(self, mock_session_class):
        """Test PipelineError raised on 500 with response-pipeline-failure header."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {"Mcpd-Error-Type": "response-pipeline-failure"}
        mock_response.text = "Response pipeline processing failed"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(PipelineError) as exc_info:
            client._perform_call("test_server", "test_tool", {"param": "value"})

        assert exc_info.value.pipeline_flow == PIPELINE_FLOW_RESPONSE
        assert exc_info.value.server_name == "test_server"
        assert exc_info.value.operation == "test_server.test_tool"
        assert "Response pipeline processing failed" in str(exc_info.value)

    @patch("requests.Session")
    def test_500_without_header_is_tool_execution_error(self, mock_session_class):
        """Test that 500 without Mcpd-Error-Type header raises ToolExecutionError."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(ToolExecutionError) as exc_info:
            client._perform_call("test_server", "test_tool", {"param": "value"})

        assert "Server error when executing 'test_tool'" in str(exc_info.value)

    @patch("requests.Session")
    def test_pipeline_error_case_insensitive_header(self, mock_session_class):
        """Test PipelineError handles case-insensitive header values."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {"Mcpd-Error-Type": "REQUEST-PIPELINE-FAILURE"}
        mock_response.text = "Pipeline failed"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(PipelineError) as exc_info:
            client._perform_call("test_server", "test_tool", {})

        assert exc_info.value.pipeline_flow == PIPELINE_FLOW_REQUEST

    @patch("requests.Session")
    def test_pipeline_error_empty_body_uses_default_message(self, mock_session_class):
        """Test PipelineError uses default message when response body is empty."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.headers = {"Mcpd-Error-Type": "response-pipeline-failure"}
        mock_response.text = ""
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_session.post.return_value = mock_response

        client = McpdClient(api_endpoint="http://localhost:8090")

        with pytest.raises(PipelineError) as exc_info:
            client._perform_call("test_server", "test_tool", {})

        assert "Pipeline failure" in str(exc_info.value)
