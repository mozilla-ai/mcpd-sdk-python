"""Dynamic tool invocation for mcpd client.

This module provides the DynamicCaller and ServerProxy classes that enable
natural Python syntax for calling MCP tools, such as:
    client.call.server.tool(**kwargs)

The dynamic calling system uses Python's __getattr__ magic method to create
a fluent interface that resolves server and tool names at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .exceptions import ToolNotFoundError
from .function_builder import TOOL_SEPARATOR

if TYPE_CHECKING:
    from .mcpd_client import McpdClient


class DynamicCaller:
    """Enables dynamic, attribute-based tool invocation using natural Python syntax.

    This class provides the magic behind the client.call.<server>.<tool>(**kwargs) syntax,
    allowing you to call MCP tools as if they were native Python methods. It uses Python's
    __getattr__ to dynamically resolve server and tool names at runtime.

    The DynamicCaller is automatically instantiated as the 'call' attribute on McpdClient
    and should not be created directly.

    Attributes:
        _client: Reference to the parent McpdClient instance.

    Example:
        >>> client = McpdClient(api_endpoint="http://localhost:8090")
        >>>
        >>> # Access tools through natural attribute syntax
        >>> # Instead of: client._perform_call("time", "get_current_time", {"timezone": "UTC"})
        >>> # You can write:
        >>> result = client.call.time.get_current_time(timezone="UTC")
        >>>
        >>> # Works with any server and tool name
        >>> weather = client.call.duckduckgo_mcp.searsch(query="Tokyo", max_results=3)
        >>> commits = client.call.mcp_discord.discord_read_messages(channelId="9223372036854775806", limit=10)

    Note:
        Tool and server names are resolved at runtime. If a server or tool doesn't exist,
        an McpdError will be raised when you attempt to call it. Use client.has_tool()
        to check availability before calling if needed.
    """

    def __init__(self, client: McpdClient):
        """Initialize the DynamicCaller with a reference to the client.

        Args:
            client: The McpdClient instance that owns this DynamicCaller.
        """
        self._client = client

    def __getattr__(self, server_name: str) -> ServerProxy:
        """Create a ServerProxy for the specified server name.

        This method is called when accessing an attribute on the DynamicCaller,
        e.g., client.call.time returns a ServerProxy for the "time" server.

        Args:
            server_name: The name of the MCP server to create a proxy for.

        Returns:
            A ServerProxy instance that can be used to call tools on that server.

        Example:
            >>> # When you write: client.call.time
            >>> # Python calls: client.call.__getattr__("time")
            >>> # Which returns: ServerProxy(client, "time")
        """
        return ServerProxy(self._client, server_name)


class ServerProxy:
    """Proxy for a specific MCP server, enabling tool invocation via attributes.

    This class represents a specific MCP server and allows calling its tools
    as if they were methods. It's created automatically by DynamicCaller and
    should not be instantiated directly.

    Attributes:
        _client: Reference to the McpdClient instance.
        _server_name: Name of the MCP server this proxy represents.

    Example:
        >>> # ServerProxy is created when you access a server:
        >>> time_server = client.call.time  # Returns ServerProxy(client, "time")
        >>>
        >>> # You can then call tools on it:
        >>> current_time = time_server.get_current_time(timezone="UTC")
        >>>
        >>> # Or chain it directly:
        >>> current_time = client.call.time.get_current_time(timezone="UTC")
    """

    def __init__(self, client: McpdClient, server_name: str):
        """Initialize a ServerProxy for a specific server.

        Args:
            client: The McpdClient instance to use for API calls.
            server_name: The name of the MCP server this proxy represents.
        """
        self._client = client
        self._server_name = server_name

    def __getattr__(self, tool_name: str) -> Callable:
        """Create a callable function for the specified tool.

        When you access an attribute on a ServerProxy (e.g., time_server.get_current_time),
        this method creates and returns a function that will call that tool when invoked.

        Args:
            tool_name: The name of the tool to create a callable for.

        Returns:
            A callable function that accepts keyword arguments and invokes the tool.

        Raises:
            McpdError: If the tool doesn't exist on this server.

        Example:
            >>> # When you write: client.call.time.get_current_time
            >>> # Python calls: ServerProxy.__getattr__("get_current_time")
            >>> # Which returns a function that calls the tool
            >>>
            >>> # The returned function can then be called:
            >>> result = client.call.time.get_current_time(timezone="UTC")
            >>>
            >>> # You can also store the function reference:
            >>> get_time = client.call.time.get_current_time
            >>> tokyo_time = get_time(timezone="Asia/Tokyo")
            >>> london_time = get_time(timezone="Europe/London")
        """
        if not self._client.has_tool(self._server_name, tool_name):
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found on server '{self._server_name}'. "
                f"Use client.tools('{self._server_name}') to see available tools.",
                server_name=self._server_name,
                tool_name=tool_name,
            )

        def tool_function(**kwargs):
            """Execute the MCP tool with the provided parameters.

            Args:
                **kwargs: Tool parameters as keyword arguments.
                         These should match the tool's inputSchema.

            Returns:
                The tool's response, typically a dictionary with the results.

            Raises:
                McpdError: If the tool execution fails for any reason.
            """
            return self._client._perform_call(self._server_name, tool_name, kwargs)

        # Add metadata to help with debugging and introspection
        tool_function.__name__ = f"{self._server_name}{TOOL_SEPARATOR}{tool_name}"
        tool_function.__qualname__ = f"ServerProxy.{tool_name}"

        return tool_function
