"""Exception hierarchy for the mcpd SDK.

This module provides a structured exception hierarchy to help users handle
different error scenarios appropriately.

Constants:
    PIPELINE_FLOW_REQUEST: Flow constant indicating a request pipeline failure.
        The request was rejected before reaching the upstream server.
    PIPELINE_FLOW_RESPONSE: Flow constant indicating a response pipeline failure.
        The upstream request was processed but results cannot be returned.
"""

#: Flow constant for request pipeline failures.
#: The request was rejected before reaching the upstream server.
PIPELINE_FLOW_REQUEST: str = "request"

#: Flow constant for response pipeline failures.
#: The upstream request was processed but results cannot be returned.
PIPELINE_FLOW_RESPONSE: str = "response"

# Internal mapping from mcpd error type header values to flow constants.
_PIPELINE_ERROR_FLOWS: dict[str, str] = {
    "request-pipeline-failure": PIPELINE_FLOW_REQUEST,
    "response-pipeline-failure": PIPELINE_FLOW_RESPONSE,
}


class McpdError(Exception):
    """Base exception for all mcpd SDK errors.

    This exception wraps all errors that occur during interaction with the mcpd daemon,
    including network failures, authentication errors, server errors, and tool execution
    failures. The original exception is preserved via exception chaining for debugging.

    Common error scenarios:
    - Network connectivity issues with the mcpd daemon
    - Authentication failures (invalid or missing API key)
    - Server not found or unavailable
    - Tool not found on the specified server
    - Tool execution errors (invalid parameters, server-side failures)
    - Timeout errors during long-running operations

    Attributes:
        args: The error message and any additional arguments.
        __cause__: The original exception that triggered this error (if any).

    Example:
        >>> from mcpd import McpdClient, McpdError
        >>>
        >>> client = McpdClient(api_endpoint="http://localhost:8090")
        >>>
        >>> try:
        >>>     # Attempt to call a tool that might not exist
        >>>     result = client.call.unknown_server.unknown_tool()
        >>> except McpdError as e:
        >>>     print(f"Operation failed: {e}")
        >>>     # Access the original exception for debugging
        >>>     if e.__cause__:
        >>>         print(f"Underlying cause: {e.__cause__}")
        >>>
        >>> # Handle specific error scenarios
        >>> try:
        >>>     servers = client.servers()
        >>> except McpdError as e:
        >>>     if "authentication" in str(e).lower():
        >>>         print("Authentication failed - check your API key")
        >>>     elif "connection" in str(e).lower():
        >>>         print("Cannot reach mcpd daemon - is it running?")
        >>>     else:
        >>>         print(f"Unexpected error: {e}")

    Note:
        When catching McpdError, you can access the original exception through
        the `__cause__` attribute for more detailed error handling or logging.
        All SDK methods that interact with the mcpd daemon will raise McpdError
        or one of its subclasses on failure, providing a consistent error interface.
    """

    pass


class ConnectionError(McpdError):
    """Raised when unable to connect to the mcpd daemon.

    This typically indicates that:
    - The mcpd daemon is not running
    - The endpoint URL is incorrect
    - Network connectivity issues
    - Firewall blocking the connection

    Example:
        >>> try:
        >>>     client = McpdClient(api_endpoint="http://localhost:8090")
        >>>     servers = client.servers()
        >>> except ConnectionError as e:
        >>>     print("Cannot reach mcpd daemon - is it running?")
        >>>     print("Try running: mcpd start")
    """

    pass


class AuthenticationError(McpdError):
    """Raised when authentication with the mcpd daemon fails.

    This indicates that:
    - The API key is invalid or expired
    - The API key is missing but required
    - The authentication method is not supported

    Example:
        >>> try:
        >>>     client = McpdClient(
        >>>         api_endpoint="http://localhost:8090",
        >>>         api_key="invalid-key"  # pragma: allowlist secret
        >>>     )
        >>>     servers = client.servers()
        >>> except AuthenticationError as e:
        >>>     print("Authentication failed - check your API key")
    """

    pass


class ServerNotFoundError(McpdError):
    """Raised when a specified MCP server doesn't exist.

    This error occurs when trying to access a server that:
    - Is not configured in the mcpd daemon
    - Has been removed or renamed
    - Is temporarily unavailable

    Attributes:
        server_name: The name of the server that wasn't found.

    Example:
        >>> try:
        >>>     tools = client.tools("nonexistent_server")
        >>> except ServerNotFoundError as e:
        >>>     print(f"Server '{e.server_name}' not found")
        >>>     print(f"Available servers: {client.servers()}")
    """

    def __init__(self, message: str, server_name: str = None):
        """Initialize ServerNotFoundError.

        Args:
            message: The error message.
            server_name: The name of the server that was not found.
        """
        super().__init__(message)
        self.server_name = server_name


class ServerUnhealthyError(McpdError):
    """Raised when a specified MCP server is not healthy.

    This indicates that the server exists but is currently unhealthy:
    - The server is down or unreachable
    - Timeout occurred while checking health
    - No health data is available for the server

    Attributes:
        server_name: The name of the server that is unhealthy.
        health_status: Details about the server's health status (if available).
                      Can be one of timeout, unreachable, unknown.

    Example:
        >>> try:
        >>>     tools = client.tools("unhealthy_server")
        >>> except ServerUnhealthyError as e:
        >>>     print(f"Server '{e.server_name}' is unhealthy")
        >>>     if e.health_status:
        >>>         print(f"Health details: {e.health_status}")
    """

    def __init__(self, message: str, server_name: str, health_status: str):
        """Initialize ServerUnhealthyError.

        Args:
            message: The error message.
            server_name: The name of the server that is unhealthy.
            health_status: Details about the server's health status.
        """
        super().__init__(message)
        self.server_name = server_name
        self.health_status = health_status


class ToolNotFoundError(McpdError):
    """Raised when a specified tool doesn't exist on a server.

    This error occurs when trying to call a tool that:
    - Doesn't exist on the specified server
    - Has been removed or renamed
    - Is temporarily unavailable

    Attributes:
        server_name: The name of the server.
        tool_name: The name of the tool that wasn't found.

    Example:
        >>> try:
        >>>     result = client.call.time.nonexistent_tool()
        >>> except ToolNotFoundError as e:
        >>>     print(f"Tool '{e.tool_name}' not found on server '{e.server_name}'")
        >>>     tools = client.tools(e.server_name)
        >>>     print(f"Available tools: {[t['name'] for t in tools]}")
    """

    def __init__(self, message: str, server_name: str = None, tool_name: str = None):
        """Initialize ToolNotFoundError.

        Args:
            message: The error message.
            server_name: The name of the server where the tool was not found.
            tool_name: The name of the tool that was not found.
        """
        super().__init__(message)
        self.server_name = server_name
        self.tool_name = tool_name


class ToolExecutionError(McpdError):
    """Raised when a tool execution fails on the server side.

    This indicates that the tool was found and called, but failed during execution:
    - Invalid parameters provided
    - Server-side error during tool execution
    - Tool returned an error response
    - Timeout during tool execution

    Attributes:
        server_name: The name of the server.
        tool_name: The name of the tool that failed.
        details: Additional error details from the server (if available).

    Example:
        >>> try:
        >>>     # Call with invalid parameters
        >>>     result = client.call.filesystem.read_file(path="/nonexistent/file")
        >>> except ToolExecutionError as e:
        >>>     print(f"Tool execution failed: {e}")
        >>>     if e.details:
        >>>         print(f"Server error details: {e.details}")
    """

    def __init__(self, message: str, server_name: str = None, tool_name: str = None, details: dict = None):
        """Initialize ToolExecutionError.

        Args:
            message: The error message.
            server_name: The name of the server where the tool execution failed.
            tool_name: The name of the tool that failed to execute.
            details: Additional error details from the server.
        """
        super().__init__(message)
        self.server_name = server_name
        self.tool_name = tool_name
        self.details = details


class ValidationError(McpdError):
    """Raised when input validation fails.

    This occurs when:
    - Required parameters are missing
    - Parameter types don't match the schema
    - Parameter values don't meet constraints

    Attributes:
        validation_errors: List of specific validation failures.

    Example:
        >>> try:
        >>>     # Missing required parameter
        >>>     result = client.call.database.query()  # 'sql' parameter required
        >>> except ValidationError as e:
        >>>     print(f"Validation failed: {e}")
        >>>     for error in e.validation_errors:
        >>>         print(f"  - {error}")
    """

    def __init__(self, message: str, validation_errors: list = None):
        """Initialize ValidationError.

        Args:
            message: The error message.
            validation_errors: List of specific validation error messages.
        """
        super().__init__(message)
        self.validation_errors = validation_errors or []


class TimeoutError(McpdError):
    """Raised when an operation times out.

    This can occur during:
    - Long-running tool executions
    - Slow network connections
    - Unresponsive mcpd daemon

    Attributes:
        operation: Description of the operation that timed out.
        timeout: The timeout value in seconds.

    Example:
        >>> try:
        >>>     # Long-running operation
        >>>     result = client.call.analysis.process_large_dataset(data=huge_data)
        >>> except TimeoutError as e:
        >>>     print(f"Operation timed out after {e.timeout} seconds: {e.operation}")
    """

    def __init__(self, message: str, operation: str = None, timeout: float = None):
        """Initialize TimeoutError.

        Args:
            message: The error message.
            operation: The operation that timed out.
            timeout: The timeout value in seconds.
        """
        super().__init__(message)
        self.operation = operation
        self.timeout = timeout


class PipelineError(McpdError):
    """Raised when required pipeline processing fails.

    This indicates that a required plugin failed during request or response
    processing. This typically indicates a problem with a plugin or an external
    system that a plugin depends on (e.g., audit service, authentication provider).

    Pipeline Flow Distinction:
    - **response-pipeline-failure**: The upstream request was processed (the tool
      was called), but results cannot be returned due to a required response
      processing step failure. Note: This does not indicate whether the tool
      itself succeeded or failed - only that the response cannot be delivered.

    - **request-pipeline-failure**: The request was rejected before reaching the
      upstream server due to a required request processing step failure (such as
      authentication, authorization, validation, or rate limiting plugin failure).

    Attributes:
        server_name: The server name (when called through tool execution).
        operation: The operation (e.g., "time.get_current_time").
        pipeline_flow: Which pipeline flow failed ("request" or "response").

    Example:
        >>> try:
        >>>     result = client.call.time.get_current_time()
        >>> except PipelineError as e:
        >>>     print(f"Pipeline {e.pipeline_flow} failure: {e}")
        >>>     if e.pipeline_flow == "response":
        >>>         print("Tool was called but results cannot be delivered")
        >>>     else:
        >>>         print("Request was rejected by pipeline")
    """

    def __init__(
        self,
        message: str,
        server_name: str | None = None,
        operation: str | None = None,
        pipeline_flow: str | None = None,
    ) -> None:
        """Initialize PipelineError.

        Args:
            message: The error message.
            server_name: The name of the server (when called through tool execution).
            operation: The operation (e.g., "time.get_current_time").
            pipeline_flow: Which pipeline flow failed ("request" or "response").
        """
        super().__init__(message)
        self.server_name = server_name
        self.operation = operation
        self.pipeline_flow = pipeline_flow
