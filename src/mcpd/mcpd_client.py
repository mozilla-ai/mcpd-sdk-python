from typing import Any, Callable, Union

import requests

from .dynamic_caller import DynamicCaller
from .exceptions import McpdError
from .function_builder import FunctionBuilder


class McpdClient:
    """Client for interacting with MCP servers through a proxy endpoint."""

    def __init__(self, api_endpoint: str, api_key: str | None = None):
        """
        Initialize the MCP client.

        Args:
            api_endpoint: The proxy endpoint URL for MCP servers
            api_key: Optional API key for authentication
        """
        self.endpoint = api_endpoint.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        # Initialize components
        self._function_builder = FunctionBuilder(self)

        # Set up authentication
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        # Dynamic call interface
        self.call = DynamicCaller(self)

    def _perform_call(self, server_name: str, tool_name: str, params: dict[str, Any]) -> Any:
        """Perform the actual API call to the MCP server."""
        try:
            # url = f"{self.endpoint}/api/v1/servers/{server_name}/tools/{tool_name}"
            # TODO: Adjusting url for now, to make it compatible with getmcp.io server
            url = f"{self.endpoint}"
            response = self.session.post(url, json=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error calling tool '{tool_name}' on server '{server_name}': {e}") from e

    def servers(self) -> list[str]:
        """Get a list of all configured server names."""
        try:
            # url = f"{self.endpoint}/api/v1/servers"
            # TODO: Adjusting url for now, to make it compatible with getmcp.io server
            url = f"{self.endpoint}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing servers: {e}") from e

    def tools(self, server_name: str | None = None) -> dict[str, list[dict]] | list[dict]:
        """Get tool schema definitions."""
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
            # url = f"{self.endpoint}/api/v1/servers/{server_name}/tools"
            # TODO: Adjusting url for now, to make it compatible with getmcp.io server
            url = f"{self.endpoint}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing tool definitions for server '{server_name}'") from e

    def agent_tools(self) -> list[Callable[..., Any]]:
        """Get a list of callable functions suitable for agentic frameworks."""
        agent_tools = []
        all_tools = self.tools()

        for server_name, tool_schemas in all_tools.items():
            for tool_schema in tool_schemas:
                func = self._function_builder.create_function_from_schema(tool_schema, server_name)
                agent_tools.append(func)

        return agent_tools

    def has_tool(self, server_name: str, tool_name: str) -> bool:
        """Check if a specific tool exists on a given server."""
        try:
            tool_defs = self.tools(server_name=server_name)
            return any(tool.get("name") == tool_name for tool in tool_defs)
        except McpdError:
            return False

    def clear_agent_tools_cache(self):
        """Clear the cached compiled functions that act as agent tools."""
        self._function_builder.clear_cache()
