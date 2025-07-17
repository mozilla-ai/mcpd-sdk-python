from unittest.mock import Mock

import pytest

from mcpd_sdk import McpdClient, McpdError


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
