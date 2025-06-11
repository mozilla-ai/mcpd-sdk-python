import json
from functools import partial
from inspect import Parameter, Signature
from typing import Any, Callable, Sequence, overload

import requests


class McpdError(Exception):
    """Custom exception for MCPD mcpd_client errors."""

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

        def tool_caller(**kwargs):
            """The actual function that gets called."""
            return self._client.call(self._server_name, tool_name, params=kwargs)

        tool_caller.__name__ = tool_name
        tool_caller.__doc__ = (
            f"Dynamically generated method to call the '{tool_name}' tool on the '{self._server_name}' server."
        )

        return tool_caller


class _ToolCallerNamespace:
    """
    An internal helper class that acts as the entry point for dynamic tool access.
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
    A Python SDK mcpd_client for interacting with the mcpd application API.
    """

    def __init__(self, endpoint: str, api_key: str = None):
        """
        Initializes the mcpd_client with the base URL of the mcpd API.
        """
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

        if api_key:
            self.session.headers.update({"MCPD-API-KEY": api_key})

        self.tool_caller = _ToolCallerNamespace(self)

        print(f"MCPD SDK Client initialized for endpoint: {self.endpoint}")

    def list_servers(self) -> list[str]:
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

    def servers(self) -> list[str]:
        """
        An alias for list_servers() for API consistency.
        """
        return self.list_servers()

    def _list_tool_definitions(self, server_name: str) -> list[dict[str, Any]]:
        """
        Internal method to retrieve tool definitions for a specific server.
        """
        try:
            url = f"{self.endpoint}/api/v1/servers/{server_name}"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise McpdError(f"Error listing tools for server '{server_name}': {e}") from e

    def _get_all_callable_tools(self) -> Sequence[Callable[..., Any]]:
        """
        Internal method to fetch all tools and return them as callable functions.
        """
        callable_tools = []
        try:
            server_names = self.list_servers()
            for server_name in server_names:
                tools_definitions = self._list_tool_definitions(server_name)
                for tool_def in tools_definitions:
                    tool_name = tool_def.get("name")
                    if not tool_name:
                        continue

                    new_func = partial(self.call, server_name, tool_name)
                    new_func.__name__ = f"{server_name}_{tool_name}"
                    new_func.__doc__ = tool_def.get(
                        "description", f"Calls the '{tool_name}' tool on the '{server_name}' server."
                    )
                    callable_tools.append(new_func)
            return callable_tools
        except McpdError as e:
            print(f"Could not retrieve all tools: {e}")
            return []

    def get_all_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Fetches the tool definitions (schemas) for all available servers.
        This is useful for initializing agentic frameworks.
        """
        all_definitions = []
        try:
            server_names = self.list_servers()
            for server_name in server_names:
                definitions = self._list_tool_definitions(server_name)
                for tool_def in definitions:
                    tool_def["name"] = f"{server_name}_{tool_def['name']}"
                all_definitions.extend(definitions)
            return all_definitions
        except McpdError as e:
            print(f"Could not retrieve all tool definitions: {e}")
            return []

    def as_any_agent_tools(self) -> list[Callable[..., Any]]:
        """
        Generates a list of self-contained, deepcopy-safe callable functions,
        one for each tool, suitable for use with agentic frameworks like any-agent.
        """
        agent_tools = []
        try:
            server_names = self.list_servers()
            for server_name in server_names:
                definitions = self._list_tool_definitions(server_name)
                for tool_def in definitions:
                    original_tool_name = tool_def.get("name")
                    if not original_tool_name:
                        continue

                    # Create a new, standalone function for each tool.
                    def create_tool_func(s_name, t_name, endpoint, api_key, schema):
                        # --- Dynamically build the function signature ---
                        parameters = []
                        type_map = {
                            "string": str,
                            "integer": int,
                            "number": float,
                            "boolean": bool,
                            "array": list,
                            "object": dict,
                        }

                        input_schema = schema.get("inputSchema", {})
                        properties = input_schema.get("properties", {})
                        required_args = set(input_schema.get("required", []))

                        for arg_name, arg_schema in properties.items():
                            param_type = type_map.get(arg_schema.get("type"), Any)
                            # Make arguments keyword-only so they must be called by name
                            parameters.append(
                                Parameter(
                                    name=arg_name,
                                    kind=Parameter.KEYWORD_ONLY,
                                    default=Parameter.empty if arg_name in required_args else None,
                                    annotation=param_type,
                                )
                            )

                        dynamic_sig = Signature(parameters, return_annotation=Any)

                        # --- Define the function's core logic ---
                        def standalone_tool_func(**kwargs):
                            temp_client = McpdClient(endpoint=endpoint, api_key=api_key)
                            return temp_client.call(s_name, t_name, params=kwargs)

                        # --- Attach metadata to the function ---
                        standalone_tool_func.__name__ = f"{s_name}_{t_name}"
                        standalone_tool_func.__doc__ = schema.get("description", "")
                        standalone_tool_func.__signature__ = dynamic_sig  # This is the crucial part

                        return standalone_tool_func

                    # Create the function for the current tool
                    new_agent_tool = create_tool_func(
                        server_name, original_tool_name, self.endpoint, self.api_key, tool_def
                    )
                    agent_tools.append(new_agent_tool)

            return agent_tools
        except McpdError as e:
            print(f"Could not generate agent tools: {e}")
            return []

    @overload
    def tools(self) -> Sequence[Callable[..., Any]]: ...

    @overload
    def tools(self, server_name: str) -> list[dict[str, Any]]: ...

    def tools(self, server_name: str = None):
        """
        Fetches tools.
        - If no server_name, returns a sequence of callable functions for all tools.
        - If server_name is provided, returns the JSON schema definitions for that server's tools.
        """
        if server_name:
            return self._list_tool_definitions(server_name)
        else:
            return self._get_all_callable_tools()

    def call(self, server_name: str, tool_name: str, params: dict[str, Any] = None) -> Any:
        """
        The low-level method to call a specific tool.
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
