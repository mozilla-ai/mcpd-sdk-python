"""mcpd client for MCP server management and tool execution.

This module provides the main McpdClient class that interfaces with the mcpd
daemon to manage interactions with MCP servers and execute tools. It offers
multiple interaction patterns including direct API calls, dynamic calling
syntax, and agent-ready function generation.

The client handles authentication, error management, and provides a unified
interface for working with multiple MCP servers through the mcpd daemon.
"""

import threading
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, NoReturn, ParamSpec, Protocol, TypeVar

import requests
from cachetools import TTLCache, cached

from ._logger import Logger, create_logger
from .dynamic_caller import DynamicCaller
from .exceptions import (
    _PIPELINE_ERROR_FLOWS,
    AuthenticationError,
    ConnectionError,
    McpdError,
    PipelineError,
    ServerNotFoundError,
    ServerUnhealthyError,
    TimeoutError,
    ToolExecutionError,
)
from .function_builder import TOOL_SEPARATOR, FunctionBuilder

P = ParamSpec("P")
R = TypeVar("R")

# Header name for mcpd pipeline error type (internal).
_MCPD_ERROR_TYPE_HEADER = "Mcpd-Error-Type"


def _raise_for_http_error(
    error: requests.exceptions.HTTPError,
    server_name: str,
    tool_name: str,
) -> NoReturn:
    """Raise appropriate McpdError for HTTP error responses."""
    status = error.response.status_code

    if status == 401:
        raise AuthenticationError(
            f"Authentication failed when calling '{tool_name}' on '{server_name}': {error}"
        ) from error

    if status == 404:
        raise ServerNotFoundError(f"Server '{server_name}' not found", server_name=server_name) from error

    # Check for pipeline failure (500 with Mcpd-Error-Type header).
    if status == 500:
        error_type = (error.response.headers.get(_MCPD_ERROR_TYPE_HEADER) or "").lower()
        flow = _PIPELINE_ERROR_FLOWS.get(error_type)
        if flow:
            message = error.response.text or "Pipeline failure"
            raise PipelineError(
                message=message,
                server_name=server_name,
                operation=f"{server_name}.{tool_name}",
                pipeline_flow=flow,
            ) from error

    # 5xx server errors.
    if status >= 500:
        raise ToolExecutionError(
            f"Server error when executing '{tool_name}' on '{server_name}': {error}",
            server_name=server_name,
            tool_name=tool_name,
        ) from error

    # Other HTTP errors (4xx).
    raise ToolExecutionError(
        f"Error calling tool '{tool_name}' on server '{server_name}': {error}",
        server_name=server_name,
        tool_name=tool_name,
    ) from error


class _AgentFunction(Protocol):
    """Protocol for generated agent functions with metadata.

    Internal type for functions created by FunctionBuilder.
    Not exposed in public API to maintain compatibility with AI frameworks.
    """

    __name__: str
    _server_name: str
    _tool_name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class HealthStatus(Enum):
    """Enumeration of possible MCP server health statuses."""

    OK = "ok"
    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"

    @classmethod
    def is_transient(cls, status: str) -> bool:
        """Check if the given health status is a transient error state."""
        return status in (cls.TIMEOUT.value, cls.UNKNOWN.value)

    @classmethod
    def is_healthy(cls, status: str) -> bool:
        """Check if the given status string represents a healthy state."""
        return status == cls.OK.value


class McpdClient:
    """Client for interacting with MCP (Model Context Protocol) servers through an mcpd daemon.

    The McpdClient provides a high-level interface to discover, inspect, and invoke tools
    exposed by MCP servers running behind an mcpd daemon proxy/gateway.

    Thread Safety:
        This client is thread-safe. Multiple threads can safely share a single instance.
        The internal health check cache is protected by locks with negligible performance
        impact since network I/O dominates execution time.

    Attributes:
        call: Dynamic interface for invoking tools using dot notation.

    Example:
        >>> from mcpd import McpdClient
        >>>
        >>> # Initialize client
        >>> client = McpdClient(api_endpoint="http://localhost:8090")
        >>>
        >>> # List available servers
        >>> servers = client.servers()
        >>> print(servers)  # ['time', 'fetch', 'git']
        >>>
        >>> # Invoke a tool dynamically
        >>> result = client.call.time.get_current_time(timezone="UTC")
        >>> print(result)  # {'time': '2024-01-15T10:30:00Z'}
    """

    _CACHEABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        ServerNotFoundError,
        ServerUnhealthyError,
        AuthenticationError,
    )
    """Exception types that should be cached when raised during server health checks.
    These exceptions represent persistent server states that benefit from caching
    to avoid repeated failed health check requests within the TTL period."""

    _SERVER_HEALTH_CACHE_MAXSIZE: int = 100
    """Maximum number of server health entries to cache.
    Prevents unbounded memory growth while allowing legitimate large-scale monitoring."""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str | None = None,
        server_health_cache_ttl: float = 10,
        logger: Logger | None = None,
    ) -> None:
        """Initialize a new McpdClient instance.

        Args:
            api_endpoint: The base URL of the mcpd daemon (e.g., "http://localhost:8090").
                         Trailing slashes will be automatically removed.
            api_key: Optional API key for Bearer token authentication. If provided,
                    will be included in all requests as "Authorization: Bearer {api_key}".
            server_health_cache_ttl: Time to live in seconds for the cache of
                                    the server health API calls. A value of 0 means no caching.
            logger: Optional custom Logger implementation. If None, uses the default logger
                   controlled by the MCPD_LOG_LEVEL environment variable.

        Raises:
            ValueError: If api_endpoint is empty or invalid.

        Example:
            >>> # Basic initialization
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # With authentication
            >>> client = McpdClient(
            ...     api_endpoint="https://mcpd.example.com",
            ...     api_key="your-api-key-here"  # pragma: allowlist secret
            ... )
        """
        self._endpoint = api_endpoint.rstrip("/").strip()
        if self._endpoint == "":
            raise ValueError("api_endpoint must be set")
        self._api_key = api_key
        self._session = requests.Session()

        # Initialize components
        self._logger = create_logger(logger)
        self._function_builder = FunctionBuilder(self)

        # Set up authentication
        if self._api_key:
            self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        # Dynamic call interface
        self.call = DynamicCaller(self)

        # Thread-safe caching for server health checks
        self._cache_lock = threading.RLock()
        # A TTL cache for server health calls. Uses LRU eviction for least recently checked servers.
        self._server_health_cache = TTLCache(maxsize=self._SERVER_HEALTH_CACHE_MAXSIZE, ttl=server_health_cache_ttl)

    def _perform_call(self, server_name: str, tool_name: str, params: dict[str, Any]) -> Any:
        """Perform the actual API call to execute a tool on an MCP server.

        This method handles the low-level HTTP communication with the mcpd daemon
        and maps various failure modes to specific exception types. It is used
        internally by both the dynamic caller interface and generated agent functions.

        Args:
            server_name: The name of the MCP server hosting the tool.
            tool_name: The name of the tool to execute.
            params: Dictionary of parameters to pass to the tool. Should match
                   the tool's inputSchema requirements.

        Returns:
            The tool's response, typically a dictionary containing the results.
            The exact structure depends on the specific tool being called.

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon (daemon not
                           running, network issues, incorrect endpoint).
            TimeoutError: If the tool execution takes longer than 30 seconds.
            AuthenticationError: If the API key is invalid or missing (HTTP 401).
            ServerNotFoundError: If the specified server doesn't exist (HTTP 404).
            ToolExecutionError: If the tool execution fails on the server side
                               (HTTP 4xx/5xx errors, invalid parameters, server errors).
            McpdError: For any other unexpected request failures.

        Note:
            All raised exceptions use proper exception chaining (``raise ... from e``)
            to preserve the original HTTP/network error details. The original
            exception can be accessed via the ``__cause__`` attribute for debugging.

        Example:
            This method is typically called indirectly through the dynamic interface:

            >>> # This call:
            >>> client.call.time.get_current_time(timezone="UTC")
            >>> # Eventually calls:
            >>> client._perform_call("time", "get_current_time", {"timezone": "UTC"})
        """
        try:
            url = f"{self._endpoint}/api/v1/servers/{server_name}/tools/{tool_name}"
            response = self._session.post(url, json=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to mcpd daemon at {self._endpoint}: {e}") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                "Tool execution timed out after 30 seconds", operation=f"{server_name}.{tool_name}", timeout=30
            ) from e
        except requests.exceptions.HTTPError as e:
            _raise_for_http_error(e, server_name, tool_name)
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error calling tool '{tool_name}' on server '{server_name}': {e}") from e

    def servers(self) -> list[str]:
        """Retrieve a list of all available MCP server names.

        Queries the mcpd daemon to discover all configured and running MCP servers.
        Server names can be used with other methods to inspect tools or invoke them.

        Returns:
            A list of server name strings. Empty list if no servers are configured.

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If the request times out after 5 seconds.
            AuthenticationError: If the API key is invalid or missing.
            McpdError: If the mcpd daemon returns an error or the API endpoint
                      is not available (check daemon version/configuration).

        Example:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>> available_servers = client.servers()
            >>> print(available_servers)
            ['time', 'fetch', 'git', 'filesystem']
            >>>
            >>> # Check if a specific server exists
            >>> if 'git' in available_servers:
            ...     print("Git server is available!")
        """
        try:
            url = f"{self._endpoint}/api/v1/servers"
            response = self._session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to mcpd daemon at {self._endpoint}: {e}") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError("Request timed out after 5 seconds", operation="list servers", timeout=5) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed: {e}") from e
            elif e.response.status_code == 404:
                raise McpdError(
                    f"Servers API endpoint not found - ensure mcpd daemon is running and supports API version v1: {e}"
                ) from e
            elif e.response.status_code >= 500:
                raise McpdError(f"mcpd daemon server error: {e}") from e
            else:
                raise McpdError(f"Error listing servers: {e}") from e
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing servers: {e}") from e

    def tools(self, server_name: str | None = None) -> dict[str, list[dict]] | list[dict]:
        """Retrieve tool schema definitions from one or all MCP servers.

        Tool schemas describe the available tools, their parameters, and expected types.
        These schemas follow the JSON Schema specification and can be used to validate
        inputs or generate UI forms.

        When server_name is provided, queries that specific server directly. When None,
        first calls servers() to get all server names, then queries each server individually.

        Args:
            server_name: Optional name of a specific server to query. If None,
                        retrieves tools from all available servers.

        Returns:
            - If server_name is provided: A list of tool schema dictionaries for that server.
            - If server_name is None: A dictionary mapping server names to their tool schemas.

            Each tool schema contains:
            - 'name': The tool's identifier
            - 'description': Human-readable description
            - 'inputSchema': JSON Schema for the tool's parameters

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If requests to the daemon time out.
            AuthenticationError: If API key authentication fails.
            ServerNotFoundError: If the specified server doesn't exist (when server_name provided).
            McpdError: For other daemon errors or API issues.

        Examples:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Get tools from a specific server
            >>> time_tools = client.tools("time")
            >>> print(time_tools[0]['name'])
            'get_current_time'
            >>>
            >>> # Get all tools from all servers
            >>> all_tools = client.tools()
            >>> for server, tools in all_tools.items():
            ...     print(f"{server}: {len(tools)} tools")
            time: 2 tools
            fetch: 1 tools
            git: 5 tools

            >>> # Inspect a tool's schema
            >>> tool_schema = time_tools[0]
            >>> print(tool_schema['inputSchema']['properties'])
            {'timezone': {'type': 'string', 'description': 'IANA timezone'}}
        """
        if server_name:
            return self._get_tool_definitions(server_name)

        try:
            all_definitions_by_server = {}
            server_names = self.servers()
            for s_name in server_names:
                definitions = self._get_tool_definitions(s_name)
                all_definitions_by_server[s_name] = definitions
            return all_definitions_by_server
        except McpdError as e:
            raise McpdError(f"Could not retrieve all tool definitions: {e}") from e

    def _get_tool_definitions(self, server_name: str) -> list[dict[str, Any]]:
        """Get tool definitions for a specific server.

        Internal method that handles HTTP requests to retrieve tool schemas.
        Called by tools() and other public methods.

        Args:
            server_name: Name of the server to get tools from.

        Returns:
            List of tool definition dictionaries containing schemas and metadata.

        Raises:
            ConnectionError: If unable to connect to mcpd daemon.
            TimeoutError: If request times out after 5 seconds.
            AuthenticationError: If API key authentication fails (HTTP 401).
            ServerNotFoundError: If server doesn't exist (HTTP 404).
            McpdError: For other daemon errors or API issues.
        """
        try:
            url = f"{self._endpoint}/api/v1/servers/{server_name}/tools"
            response = self._session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to mcpd daemon at {self._endpoint}: {e}") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                "Request timed out after 5 seconds", operation=f"list tools for {server_name}", timeout=5
            ) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Authentication failed when accessing server '{server_name}': {e}") from e
            elif e.response.status_code == 404:
                raise ServerNotFoundError(f"Server '{server_name}' not found", server_name=server_name) from e
            else:
                raise McpdError(f"Error listing tool definitions for server '{server_name}': {e}") from e
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing tool definitions for server '{server_name}': {e}") from e

    def agent_tools(
        self,
        servers: list[str] | None = None,
        tools: list[str] | None = None,
        *,
        refresh_cache: bool = False,
    ) -> list[Callable[..., Any]]:
        """Generate callable Python functions for available tools, suitable for AI agents.

        This method queries servers and creates self-contained, deepcopy-safe functions
        that can be passed to agentic frameworks like any-agent, LangChain, or custom AI
        systems. Each function includes its schema as metadata and handles the MCP
        communication internally.

        By default, this method automatically filters out unhealthy servers by checking
        their health status before fetching tools. Unhealthy servers are silently skipped
        to ensure the method returns quickly without waiting for timeouts on failed servers.

        Generated functions are cached for performance. Once cached, subsequent calls return
        the cached functions immediately without refetching schemas from healthy servers,
        regardless of filter parameters.
        Use clear_agent_tools_cache() to clear the cache, or set refresh_cache to true
        to force regeneration when tool schemas have changed.

        Args:
            servers: Optional list of server names to filter by.
                     If None, returns tools from all servers.
                     If specified, only tools from the listed servers are included.
                     Non-existent server names are silently ignored.

            tools: Optional list of tool names to filter by. Supports both:
                   - Raw tool names: 'get_current_time' (matches across all servers)
                   - Fully qualified tool names: with server prefix and separator,
                        e.g. 'time__get_current_time' (matches specific {server}__{tool})
                   If None, returns all tools from selected servers.
                   Empty list returns no tools.

            refresh_cache: When true, clears the cache and fetches fresh tool schemas from healthy servers.
                           When false or undefined, returns cached functions if available.
                           Defaults to false.

        Returns:
            A list of callable functions, one for each matching tool from healthy servers.
            Each function has the following attributes:
            - __name__: The tool's qualified name (e.g., "time__get_current_time")
            - __doc__: The tool's description
            - _server_name: The server hosting this tool (original name)
            - _tool_name: The tool's name (original name)

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If requests to the daemon time out.
            AuthenticationError: If API key authentication fails.
            McpdError: If unable to retrieve server health status or generate functions.
                      Servers for which tool schemas cannot be retrieved will be ignored.

        Examples:
            >>> from any_agent import AnyAgent, AgentConfig
            >>> from mcpd import McpdClient
            >>>
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Get all tools from healthy servers (default)
            >>> tools = client.agent_tools()
            >>> print(f"Generated {len(tools)} callable tools")
            >>>
            >>> # Get tools from specific servers, only if healthy
            >>> time_tools = client.agent_tools(servers=['time'])
            >>> subset_tools = client.agent_tools(servers=['time', 'fetch'])
            >>>
            >>> # Filter by tool names (cross-cutting)
            >>> math_tools = client.agent_tools(tools=['add', 'multiply'])
            >>>
            >>> # Filter by qualified tool names
            >>> specific = client.agent_tools(tools=['time__get_current_time'])
            >>>
            >>> # Combine server and tool filtering
            >>> filtered = client.agent_tools(
            ...     servers=['time', 'math'],
            ...     tools=['add', 'get_current_time']
            ... )
            >>>
            >>> # Force refresh of cached functions
            >>> all_tools = client.agent_tools(refresh_cache=True)
            >>>
            >>> # Inspect metadata
            >>> tool = client.agent_tools()[0]
            >>> print(f"{tool._server_name}.{tool._tool_name}")
            >>>
            >>> # Use with an AI agent framework
            >>> agent_config = AgentConfig(
            ...     tools=tools,
            ...     model_id="gpt-4",
            ...     instructions="Help the user with their tasks."
            ... )
            >>> agent = AnyAgent.create("assistant", agent_config)
            >>>
            >>> # The agent can now call any MCP tool automatically
            >>> response = agent.run("What time is it in Tokyo?")

        Note:
            The generated functions capture the client instance and will use the
            same authentication and endpoint configuration. They are thread-safe
            but may not be suitable for pickling due to the embedded client state.
        """
        if refresh_cache:
            self.clear_agent_tools_cache()

        all_tools = self._agent_tools()

        return self._filter_agent_tools(all_tools, servers, tools)

    def _agent_tools(self) -> list[_AgentFunction]:
        """Get or build cached agent tool functions from all healthy servers.

        This internal method manages the agent tools cache. On first call, it builds
        the cache by fetching tools from all healthy servers. On subsequent calls,
        it returns the cached functions immediately without refetching schemas.

        Returns:
            List of callable agent functions.
            Returns cached functions if available, otherwise builds and caches functions from all healthy servers.

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If requests to the daemon time out.
            AuthenticationError: If API key authentication fails.
            McpdError: If unable to retrieve server health status.

        Note:
            Servers for which tool schemas cannot be retrieved will be ignored.
            When logging is enabled a warning will be logged for these servers.
        """
        # Return cached functions if available.
        cached_functions = self._function_builder.get_cached_functions()
        if cached_functions:
            return cached_functions

        agent_tools = []
        healthy_servers = self._get_healthy_servers(self.servers())
        for server_name in healthy_servers:
            try:
                tool_schemas = self.tools(server_name=server_name)
            except (ConnectionError, TimeoutError, AuthenticationError, ServerNotFoundError, McpdError) as e:
                # These servers were reported as healthy, so failures for schemas would be unexpected.
                self._logger.warn("Server '%s' became unavailable or unhealthy during tool fetch: %s", server_name, e)
                continue

            for tool_schema in tool_schemas:
                func = self._function_builder.create_function_from_schema(tool_schema, server_name)
                agent_tools.append(func)

        return agent_tools

    def _get_healthy_servers(self, server_names: list[str]) -> list[str]:
        """Filter server names to only those that are healthy.

        Args:
            server_names: List of server names to filter.

        Returns:
            List of server names that have health status 'ok'.

        Raises:
            McpdError: If unable to retrieve server health information.

        Note:
            When logging is enabled a warning will be logged for servers that don't exist
            or have an unhealthy status (timeout, unreachable, unknown).
        """
        health_map = self.server_health()

        def is_valid(name: str) -> bool:
            health = health_map.get(name)

            if not health:
                self._logger.warn("Skipping non-existent server '%s'", name)
                return False

            status = health.get("status")
            if not HealthStatus.is_healthy(status):
                self._logger.warn("Skipping unhealthy server '%s' with status '%s'", name, status)
                return False

            return True

        return [name for name in server_names if is_valid(name)]

    def _matches_tool_filter(self, func: _AgentFunction, tools: list[str]) -> bool:
        """Check if a tool matches the tool filter.

        Supports two formats:
        - Raw tool name: "get_current_time" (matches tool name only)
        - Fully qualified tool names: with server prefix and separator,
            e.g. 'time__get_current_time' (matches specific {server}__{tool})

        When a filter contains TOOL_SEPARATOR (__), it's checked as prefixed (exact match against func.__name__);
        then falls back to raw match. This handles tools whose names contain TOOL_SEPARATOR.

        Args:
            func: The generated function to check.
            tools: List of tool names to match against.

        Returns:
            True if the tool matches any item in the filter.
        """
        return any(
            filter_item == func._tool_name
            if TOOL_SEPARATOR not in filter_item
            else filter_item == func.__name__ or filter_item == func._tool_name
            for filter_item in tools
        )

    def _filter_agent_tools(
        self,
        functions: list[_AgentFunction],
        servers: list[str] | None,
        tools: list[str] | None,
    ) -> list[_AgentFunction]:
        """Filter agent tools by servers and/or tool names.

        Args:
            functions: List of agent functions to filter.
            servers: Optional list of server names to filter by.
            tools: Optional list of tool names to filter by.

        Returns:
            Filtered list of functions matching the criteria.
        """
        result = functions

        # Filter by servers if specified.
        if servers is not None:
            result = [func for func in result if func._server_name in servers]

        # Filter by tools if specified.
        if tools is not None:
            result = [func for func in result if self._matches_tool_filter(func, tools)]

        return result

    def has_tool(self, server_name: str, tool_name: str) -> bool:
        """Check if a specific tool exists on a given server.

        This method queries the server's tool definitions via tools(server_name) and
        searches for the specified tool. It's useful for validation before attempting
        to call a tool, especially when tool names are provided by user input or
        external sources.

        Args:
            server_name: The name of the MCP server to check.
            tool_name: The name of the tool to look for.

        Returns:
            True if the tool exists on the specified server, False otherwise.
            Returns False if the server doesn't exist, is unreachable, or if any
            other error occurs during the check.

        Example:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Check before calling
            >>> if client.has_tool("time", "get_current_time"):
            ...     result = client.call.time.get_current_time(timezone="UTC")
            ... else:
            ...     print("Tool not available")
            >>>
            >>> # Validate user input
            >>> user_server = input("Enter server name: ")
            >>> user_tool = input("Enter tool name: ")
            >>> if not client.has_tool(user_server, user_tool):
            ...     print(f"Error: Tool '{user_tool}' not found on '{user_server}'")
        """
        try:
            tool_defs = self.tools(server_name=server_name)
            return any(tool.get("name") == tool_name for tool in tool_defs)
        except McpdError:
            return False

    def clear_agent_tools_cache(self) -> None:
        """Clear the cache of generated callable functions from agent_tools().

        This method clears the internal FunctionBuilder cache that stores compiled
        function templates. Call this when server configurations have changed to
        ensure agent_tools() regenerates functions with the latest definitions.

        Call this method when:
        - MCP servers have been added or removed from the daemon
        - Tool definitions have changed on existing servers
        - You want to force regeneration of function wrappers

        This only affects the internal cache used by agent_tools(). It does not
        affect the mcpd daemon or MCP servers themselves.

        Returns:
            None

        Example:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Initial tool generation
            >>> tools_v1 = client.agent_tools()
            >>>
            >>> # ... MCP server configuration changes ...
            >>>
            >>> # Clear cache to get updated tools
            >>> client.clear_agent_tools_cache()
            >>> tools_v2 = client.agent_tools()  # Regenerates from latest definitions
        """
        self._function_builder.clear_cache()

    def _exception_to_result(
        self, func: Callable[P, R], cacheable_exceptions: tuple[type[Exception], ...]
    ) -> Callable[P, R | Exception]:
        """Decorator that executes the wrapped function and captures any exception as the return value.

        If the wrapped function raises an exception from the given cacheable_exceptions,
        the exception object is returned instead of propagating it. This is used to extend
        the functionality of the caches provided by the cachetools library which, by default
        do not cache results when exceptions are raised.

        Args:
            func: The function to wrap.
            cacheable_exceptions: A tuple of exception types that should be captured and returned.

        Returns:
            The result of the function, or the exception object if an exception was raised.
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except cacheable_exceptions as e:
                return e

        return wrapped

    def _result_to_exception(self, func: Callable[P, R | Exception]) -> Callable[P, R]:
        """Decorator that checks if the wrapped function returns an Exception and raises it.

        If the wrapped function returns an Exception object, this decorator raises it.
        Otherwise, it returns the result as normal. Useful for converting error-as-result
        patterns back into standard exception propagation.

        Args:
            func: The function to wrap.

        Returns:
            The result of the function, or raises the exception if the result is an Exception.
        """

        @wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, Exception):
                raise result
            return result

        return wrapped

    def _cache_with_selective_exceptions(self) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator which caches results of the wrapped function, including certain cacheable exceptions.

        The caching primitives provided by the cachetools library do not cache results when exceptions
        are raised by the wrapped function. This decorator allows caching certain exceptions by combining
        three behaviors:

        1. Captures certain exceptions as results (using _exception_to_result). See _CACHEABLE_EXCEPTIONS.
        2. Caches the results (including captured exceptions) using cachetools.cached.
        3. Propagates any captured exceptions as raised exceptions (using _result_to_exception).

        Returns:
            A decorator that applies all three behaviors in order.
        """

        def decorator(function):
            decorated = self._exception_to_result(function, cacheable_exceptions=self._CACHEABLE_EXCEPTIONS)
            decorated = cached(cache=self._server_health_cache, lock=self._cache_lock)(decorated)
            decorated = self._result_to_exception(decorated)
            return decorated

        return decorator

    @staticmethod
    def _cache_with_selective_exceptions_and_self(func: Callable[P, R]) -> Callable[P, R]:
        """Decorator to apply _cache_with_selective_exceptions to methods.

        This is a helper to apply the caching decorator to instance methods that
        need access to self.

        Args:
            func: The instance method to wrap.

        Returns:
            A decorator that can be applied to instance methods.
        """

        def wrapper(self, *args, **kwargs):
            return self._cache_with_selective_exceptions()(func)(self, *args, **kwargs)

        return wrapper

    @_cache_with_selective_exceptions_and_self
    def _get_server_health(self, server_name: str | None = None) -> list[dict] | dict:
        """Get health information for one or all MCP servers.

        Internal method that handles HTTP requests to mcpd daemon health endpoints.
        Called by server_health() and other public methods.

        Args:
            server_name: Optional name of specific server. If None, gets all servers.

        Returns:
            Health data for the server(s). See server_health() for format details.

        Raises:
            See server_health() for all possible exceptions.
        """
        try:
            if server_name:
                url = f"{self._endpoint}/api/v1/health/servers/{server_name}"
            else:
                url = f"{self._endpoint}/api/v1/health/servers"
            response = self._session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data if server_name else data.get("servers", [])
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to mcpd daemon at {self._endpoint}: {e}") from e
        except requests.exceptions.Timeout as e:
            operation = f"get health of {server_name}" if server_name else "get health of all servers"
            raise TimeoutError("Request timed out after 5 seconds", operation=operation, timeout=5) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                msg = (
                    f"Authentication failed when accessing server '{server_name}': {e}"
                    if server_name
                    else f"Authentication failed: {e}"
                )
                raise AuthenticationError(msg) from e
            elif e.response.status_code == 404:
                assert server_name is not None
                raise ServerNotFoundError(f"Server '{server_name}' not found", server_name=server_name) from e
            else:
                msg = (
                    f"Error retrieving health status for server '{server_name}': {e}"
                    if server_name
                    else f"Error retrieving health status for all servers: {e}"
                )
                raise McpdError(msg) from e
        except requests.exceptions.RequestException as e:
            msg = (
                f"Error retrieving health status for server '{server_name}': {e}"
                if server_name
                else f"Error retrieving health status for all servers: {e}"
            )
            raise McpdError(msg) from e

    def server_health(self, server_name: str | None = None) -> dict[str, dict] | dict:
        """Retrieve health information from one or all MCP servers.

        This method queries the mcpd daemon for health status of MCP servers.
        Health information includes status, latency, and timestamps of last checks.

        When server_name is provided, it queries only that server. Otherwise, it retrieves
        health information from all servers in a single query.

        The returned health information is cached for performance using a TTL cache. Use
        clear_server_health_cache() to force a fresh check.

        Args:
            server_name: Optional name of a specific server to query. If None,
                         retrieves health information from all available servers.

        Returns:
            - If server_name is provided: The health information for that server.
            - If server_name is None: A dictionary mapping server names to their health information.

            Server health is a dictionary that contains:
            - '$schema': A URL to the JSON schema
            - 'name': The server identifier
            - 'status': The current health status of the server ('ok', 'timeout', 'unreachable', 'unknown')
            - 'latency': The latency of the server in milliseconds (optional)
            - 'lastChecked': Time when ping was last attempted (optional)
            - 'lastSuccessful': Time of the most recent successful ping (optional)

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If requests to the daemon time out.
            AuthenticationError: If API key authentication fails.
            ServerNotFoundError: If the specified server doesn't exist (when server_name provided).
            McpdError: For other daemon errors or API issues.

        Examples:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Get health for a specific server
            >>> health_info = client.server_health(server_name="time")
            >>> print(health_info["status"])
            ok
            >>>
            >>> # Check if a server failure is temporary (useful for retry logic)
            >>> health_info = client.server_health(server_name="problematic_server")
            >>> if HealthStatus.is_transient(health_info["status"]):
            ...     print("Temporary issue, will retry")
            ... else:
            ...     print("Persistent problem, requires intervention")
            >>>
            >>> # Get health for all servers
            >>> all_health = client.server_health()
            >>> for server, health in all_health.items():
            ...     print(f"{server}: {health['status']}")
            fetch: ok
            time: ok
        """
        if server_name:
            return self._get_server_health(server_name)

        try:
            all_health = self._get_server_health()
            all_health_by_server = {}
            for health in all_health:
                all_health_by_server[health["name"]] = health
            return all_health_by_server
        except McpdError as e:
            raise McpdError(f"Could not retrieve all health information: {e}") from e

    def _raise_for_server_health(self, server_name: str):
        """Raise an error if the specified MCP server is not healthy.

        This internal method checks server health and raises ServerUnhealthyError
        if the server status is anything other than 'ok'. Used by methods that
        require a healthy server before proceeding.

        Args:
            server_name: Name of the server to check.

        Raises:
            ServerUnhealthyError: If the server status is not 'ok'.
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If health check request times out.
            AuthenticationError: If API key authentication fails.
            ServerNotFoundError: If the specified server doesn't exist.
            McpdError: For other daemon errors during health check.
        """
        health = self.server_health(server_name=server_name)
        status = health["status"]
        if not HealthStatus.is_healthy(status):
            raise ServerUnhealthyError(
                f"Server '{server_name}' is not healthy", server_name=server_name, health_status=status
            )

    def is_server_healthy(self, server_name: str) -> bool:
        """Check if the specified MCP server is healthy.

        This method queries the server's health status and determines whether the server is healthy
        and therefore can handle requests or not. It's useful for validating an MCP server is ready
        before attempting to call one of its tools.

        Args:
            server_name: The name of the MCP server to check.

        Returns:
            True if the server is healthy, False if the server is unhealthy or doesn't exist.

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If the health check request times out.
            AuthenticationError: If API key authentication fails.
            McpdError: For other daemon errors that prevent checking health status.

        Example:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Check before calling
            >>> if client.is_server_healthy("time"):
            ...     result = client.call.time.get_current_time(timezone="UTC")
            ... else:
            ...     print("The server is not ready to accept requests yet.")
        """
        try:
            self._raise_for_server_health(server_name)
            return True
        except (ServerUnhealthyError, ServerNotFoundError):
            # These specific exceptions represent servers that are unhealthy one way or another
            return False

    def clear_server_health_cache(self, server_name: str | None = None) -> None:
        """Clear the cached health information for one or all MCP servers.

        This method clears the internal cache that stores server health information.
        Call this when server statuses may have changed to ensure server_health() fetches
        fresh data from the mcpd daemon.

        Call this method when:
        - You want to force a fresh health check
        - You want to clear stale or potentially incorrect health data

        Note: Cache entries are automatically invalidated based on the TTL set
        during initialization (see `server_health_cache_ttl`).

        This only affects the internal server health cache used by server_health(). It does not
        affect the mcpd daemon or MCP servers themselves.

        Args:
            server_name: The name of the MCP server to clear the cache for. If None, clears all caches.

        Returns:
            None

        Example:
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Initial server health check
            >>> health_v1 = client.server_health("time")
            >>>
            >>> # ... Force a fresh health check ...
            >>>
            >>> # Clear cache to get updated health info
            >>> client.clear_server_health_cache("time")
            >>> health_v2 = client.server_health("time") # Fetches fresh health data
        """
        with self._cache_lock:
            if server_name is None:
                self._server_health_cache.clear()
            else:
                self._server_health_cache.pop((self, server_name), None)
