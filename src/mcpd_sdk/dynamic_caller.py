from .exceptions import McpdError


class DynamicCaller:
    """Helper class to enable client.call.<server_name>.<tool_name>(**kwargs) syntax."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, server_name: str):
        """Get a server proxy for dynamic tool calling."""
        return ServerProxy(self._client, server_name)


class ServerProxy:
    """Proxy class for server-specific tool calling."""

    def __init__(self, client, server_name: str):
        self._client = client
        self._server_name = server_name

    def __getattr__(self, tool_name: str):
        """Get a tool callable."""
        if not self._client.has_tool(self._server_name, tool_name):
            raise McpdError(f"Tool '{tool_name}' not found on server '{self._server_name}'")

        def tool_callable(**kwargs):
            return self._client._perform_call(self._server_name, tool_name, kwargs)

        return tool_callable
