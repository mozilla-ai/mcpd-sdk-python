from typing import Any, Callable, Union

import requests

from .dynamic_caller import DynamicCaller
from .exceptions import (
    AuthenticationError,
    ConnectionError,
    McpdError,
    ServerNotFoundError,
    TimeoutError,
    ToolExecutionError,
)
from .function_builder import FunctionBuilder


class McpdClient:
    """
    Client for interacting with MCP (Model Context Protocol) servers through an mcpd daemon.

    The McpdClient provides a high-level interface to discover, inspect, and invoke tools
    exposed by MCP servers running behind an mcpd daemon proxy/gateway.

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

    def __init__(self, api_endpoint: str, api_key: str | None = None):
        """
        Initialize a new McpdClient instance.

        Args:
            api_endpoint: The base URL of the mcpd daemon (e.g., "http://localhost:8090").
                         Trailing slashes will be automatically removed.
            api_key: Optional API key for Bearer token authentication. If provided,
                    will be included in all requests as "Authorization: Bearer {api_key}".

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
        self._function_builder = FunctionBuilder(self)

        # Set up authentication
        if self._api_key:
            self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

        # Dynamic call interface
        self.call = DynamicCaller(self)

    def _perform_call(self, server_name: str, tool_name: str, params: dict[str, Any]) -> Any:
        """
        Perform the actual API call to execute a tool on an MCP server.

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
                f"Tool execution timed out after 30 seconds", operation=f"{server_name}.{tool_name}", timeout=30
            ) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(
                    f"Authentication failed when calling '{tool_name}' on '{server_name}': {e}"
                ) from e
            elif e.response.status_code == 404:
                raise ServerNotFoundError(f"Server '{server_name}' not found", server_name=server_name) from e
            elif e.response.status_code >= 500:
                raise ToolExecutionError(
                    f"Server error when executing '{tool_name}' on '{server_name}': {e}",
                    server_name=server_name,
                    tool_name=tool_name,
                ) from e
            else:
                raise ToolExecutionError(
                    f"Error calling tool '{tool_name}' on server '{server_name}': {e}",
                    server_name=server_name,
                    tool_name=tool_name,
                ) from e
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error calling tool '{tool_name}' on server '{server_name}': {e}") from e

    def servers(self) -> list[str]:
        """
        Retrieve a list of all available MCP server names.

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
            raise TimeoutError(f"Request timed out after 5 seconds", operation="list servers", timeout=5) from e
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
        """
        Retrieve tool schema definitions from one or all MCP servers.

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
        """Get tool definitions for a specific server."""
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
                f"Request timed out after 5 seconds", operation=f"list tools for {server_name}", timeout=5
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

    def agent_tools(self) -> list[Callable[..., Any]]:
        """
        Generate callable Python functions for all available tools, suitable for AI agents.

        This method queries all servers via `tools()` and creates self-contained,
        deepcopy-safe functions that can be passed to agentic frameworks like any-agent,
        LangChain, or custom AI systems. Each function includes its schema as metadata
        and handles the MCP communication internally.

        The generated functions are cached for performance. Use clear_agent_tools_cache()
        to force regeneration if servers or tools have changed.

        Returns:
            A list of callable functions, one for each tool across all servers.
            Each function has the following attributes:
            - __name__: The tool's qualified name (e.g., "time__get_current_time")
            - __doc__: The tool's description
            - _schema: The original JSON Schema
            - _server_name: The server hosting this tool
            - _tool_name: The tool's name

        Raises:
            ConnectionError: If unable to connect to the mcpd daemon.
            TimeoutError: If requests to the daemon time out.
            AuthenticationError: If API key authentication fails.
            ServerNotFoundError: If a server becomes unavailable during tool retrieval.
            McpdError: If unable to retrieve tool definitions or generate functions.

        Example:
            >>> from any_agent import AnyAgent, AgentConfig
            >>> from mcpd import McpdClient
            >>>
            >>> client = McpdClient(api_endpoint="http://localhost:8090")
            >>>
            >>> # Get all tools as callable functions
            >>> tools = client.agent_tools()
            >>> print(f"Generated {len(tools)} callable tools")
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
        agent_tools = []
        all_tools = self.tools()

        for server_name, tool_schemas in all_tools.items():
            for tool_schema in tool_schemas:
                func = self._function_builder.create_function_from_schema(tool_schema, server_name)
                agent_tools.append(func)

        return agent_tools

    def has_tool(self, server_name: str, tool_name: str) -> bool:
        """
        Check if a specific tool exists on a given server.

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
        """
        Clear the cache of generated callable functions from agent_tools().

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
