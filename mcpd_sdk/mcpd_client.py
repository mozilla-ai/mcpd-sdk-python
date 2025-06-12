import json
from inspect import Parameter, Signature
from typing import Any, Callable, overload

import requests


class McpdError(Exception):
    """Custom exception for MCPD client errors."""

    pass


class _ServerNamespace:
    """
    An internal helper class that represents a specific server's toolset.
    It dynamically creates callable methods for each tool.
    """

    def __init__(self, client: "McpdClient", server_name: str):
        self._client = client
        self._server_name = server_name

    def __getattr__(self, tool_name: str) -> Callable[..., Any]:
        """
        Dynamically creates a function that will call a tool on this server.
        """

        # The actual function that gets called, created on the fly.
        # It uses the client's internal _perform_call method.
        def dynamic_tool_caller(**kwargs):
            return self._client._perform_call(self._server_name, tool_name, params=kwargs)

        dynamic_tool_caller.__name__ = f"{self._server_name}_{tool_name}"
        dynamic_tool_caller.__doc__ = f"Dynamically calls the '{tool_name}' tool on the '{self._server_name}' server."
        return dynamic_tool_caller


class _CallNamespace:
    """
    An internal helper class that acts as the entry point for dynamic tool access.
    e.g., `client.call.time.get_current_time()`
    """

    def __init__(self, client: "McpdClient"):
        self._client = client

    def __getattr__(self, server_name: str) -> _ServerNamespace:
        """
        Dynamically creates a namespace for a specific server.
        """
        return _ServerNamespace(self._client, server_name)


class McpdClient:
    """
    A Python SDK client for interacting with the mcpd application API.
    """

    def __init__(self, endpoint: str, api_key: str = None):
        """
        Initializes the client with the base URL of the mcpd API.
        """
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

        if api_key:
            self.session.headers.update({"MCPD-API-KEY": api_key})

        self.call = _CallNamespace(self)

    def servers(self) -> list[str]:
        """
        Retrieves a list of all available servers from the mcpd API.
        """
        try:
            url = f"{self.endpoint}/api/v1/servers"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing servers: {e}") from e

    def _get_tool_definitions(self, server_name: str) -> list[dict[str, Any]]:
        """
        Internal method to retrieve the full tool definitions (schemas) for a specific server.
        """
        try:
            url = f"{self.endpoint}/api/v1/servers/{server_name}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing tool definitions for server '{server_name}'") from e

    @overload
    def tools(self) -> dict[str, list[dict[str, Any]]]: ...

    @overload
    def tools(self, server_name: str) -> list[dict[str, Any]]: ...

    def tools(self, server_name: str = None) -> list[dict[str, Any]]:
        """
        Fetches tool definitions (schemas).
        - If no server_name, returns a dictionary mapping server names to their tool definitions.
        - If server_name is provided, returns a list of definitions for that server's tools.
        """
        if server_name:
            return self._get_tool_definitions(server_name)
        else:
            all_definitions_by_server = {}
            try:
                server_names = self.servers()
                for s_name in server_names:
                    definitions = self._get_tool_definitions(s_name)
                    all_definitions_by_server[s_name] = definitions
                return all_definitions_by_server
            except McpdError as e:
                raise McpdError(f"Could not retrieve all tool definitions") from e

    def has_tool(self, server_name: str, tool_name: str) -> bool:
        """
        Checks if a specific tool is available on a given server.
        """
        try:
            tool_defs = self.tools(server_name=server_name)
            for tool in tool_defs:
                if tool.get("name") == tool_name:
                    return True
            return False
        except McpdError:
            return False

    def agent_tools(self) -> list[Callable[..., Any]]:
        """
        Generates a list of self-contained, deepcopy-safe callable functions,
        one for each tool, suitable for use with agentic frameworks like any-agent.
        """
        agent_tools = []
        try:
            server_names = self.servers()
            for server_name in server_names:
                definitions = self.tools(server_name=server_name)
                for tool_def in definitions:
                    original_tool_name = tool_def.get("name")
                    if not original_tool_name:
                        continue

                    def create_tool_func(s_name, t_name, endpoint, api_key, schema):
                        parameters = []
                        docstring_parts = [schema.get("description", "")]
                        input_schema = schema.get("inputSchema", {})
                        properties = input_schema.get("properties", {})
                        required_args = set(input_schema.get("required", []))

                        if properties:
                            docstring_parts.append("\nArgs:")

                        type_map = {
                            "string": str,
                            "integer": int,
                            "number": float,
                            "boolean": bool,
                            "array": list,
                            "object": dict,
                        }

                        for arg_name, arg_schema in properties.items():
                            param_type = type_map.get(arg_schema.get("type"), Any)
                            parameters.append(
                                Parameter(
                                    name=arg_name,
                                    kind=Parameter.KEYWORD_ONLY,
                                    default=Parameter.empty if arg_name in required_args else None,
                                    annotation=param_type,
                                )
                            )
                            # Add parameter to the docstring
                            arg_desc = arg_schema.get("description", "")
                            docstring_parts.append(f"    {arg_name} ({param_type.__name__}): {arg_desc}")

                        dynamic_sig = Signature(parameters, return_annotation=Any)

                        def standalone_tool_func(**kwargs):
                            # This creates a temporary client only when the agent calls the tool.
                            temp_client = McpdClient(endpoint=endpoint, api_key=api_key)
                            return temp_client._perform_call(s_name, t_name, params=kwargs)

                        standalone_tool_func.__name__ = f"{s_name}_{t_name}"
                        standalone_tool_func.__doc__ = "\n".join(docstring_parts)
                        standalone_tool_func.__signature__ = dynamic_sig

                        return standalone_tool_func

                    new_agent_tool = create_tool_func(
                        server_name, original_tool_name, self.endpoint, self.api_key, tool_def
                    )
                    agent_tools.append(new_agent_tool)

            return agent_tools
        except McpdError as e:
            raise McpdError(f"Could not generate agent tools") from e

    def _perform_call(self, server_name: str, tool_name: str, params: dict[str, Any] = None) -> Any:
        """
        Internal-only method to execute the HTTP request for a tool call.
        """
        if params is None:
            params = {}

        url = f"{self.endpoint}/api/v1/servers/{server_name}/{tool_name}"

        try:
            response = self.session.post(url, data=json.dumps(params), timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Error calling tool '{tool_name}' on server '{server_name}': {e}"
            if e.response is not None:
                error_message += f"\n  - Response body: {e.response.text}"
            raise McpdError(error_message) from e
