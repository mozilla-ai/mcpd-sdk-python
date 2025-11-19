from collections.abc import Callable
from unittest.mock import Mock

import pytest

from mcpd import McpdClient


@pytest.fixture(scope="function")
def mock_client():
    client = Mock()
    client._perform_call = Mock(return_value={"result": "success"})
    client.has_tool = Mock(return_value=True)
    return client


@pytest.fixture(scope="function")
def mock_response():
    response = Mock()
    response.json.return_value = {"result": "success"}
    response.raise_for_status.return_value = None
    return response


@pytest.fixture(scope="session")
def basic_schema():
    return {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
            "required": ["param1"],
        },
    }


@pytest.fixture(scope="session")
def fqdn() -> str:
    return "http://localhost:8090"


@pytest.fixture(scope="session")
def api_url(fqdn) -> str:
    return fqdn + "/api/v1"


@pytest.fixture(scope="function")
def client(fqdn):
    return McpdClient(api_endpoint=fqdn)


@pytest.fixture(scope="function")
def client_with_auth(fqdn):
    return McpdClient(api_endpoint=fqdn, api_key="test-key")  # pragma: allowlist secret


@pytest.fixture
def tools_side_effect():
    """Factory for creating tools() mock side effects.

    Returns a function that creates side_effect functions for mocking tools().
    The side_effect returns the appropriate tool list based on server_name parameter.

    Usage:
        def test_something(tools_side_effect):
            tools_map = {
                "server1": [{"name": "tool1", "description": "Tool 1"}],
                "server2": [{"name": "tool2", "description": "Tool 2"}],
            }
            mock_tools.side_effect = tools_side_effect(tools_map)
    """

    def _create_side_effect(tools_map: dict[str, list[dict]]) -> Callable[[str | None], list[dict]]:
        def side_effect(server_name: str | None = None) -> list[dict]:
            return tools_map.get(server_name, [])

        return side_effect

    return _create_side_effect
