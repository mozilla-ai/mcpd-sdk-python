from unittest.mock import Mock, patch

import pytest
from requests import Session
from requests.exceptions import RequestException

from mcpd_sdk import McpdClient, McpdError


class TestMcpdClient:
    def test_init_basic(self):
        client = McpdClient(api_endpoint="http://localhost:9999")
        assert client.endpoint == "http://localhost:9999"
        assert client.api_key is None
        assert hasattr(client, "session")
        assert hasattr(client, "call")

    def test_init_with_auth(self):
        client = McpdClient(api_endpoint="http://localhost:9090", api_key="test-key123")
        assert client.endpoint == "http://localhost:9090"
        assert client.api_key == "test-key123"  # pragma: allowlist secret
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer test-key123"

    def test_init_strips_trailing_slash(self):
        client = McpdClient("http://localhost:8090/")
        assert client.endpoint == "http://localhost:8090"

    @patch.object(Session, "get")
    def test_servers_success(self, mock_get, client, api_url):
        servers = ["server1", "server2"]
        mock_response = Mock()
        mock_response.json.return_value = servers
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.servers()

        assert result == servers
        mock_get.assert_called_once_with(f"{api_url}/servers", timeout=5)

    @patch.object(Session, "get")
    def test_servers_request_error(self, mock_get, client):
        mock_get.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError, match="Error listing servers"):
            client.servers()

    @patch.object(Session, "get")
    def test_tools_single_server(self, mock_get, client):
        mock_response = Mock()
        mock_response.json.return_value = {"tools": [{"name": "tool1"}, {"name": "tool2"}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.tools("test_server")

        assert result == [{"name": "tool1"}, {"name": "tool2"}]
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/servers/test_server/tools", timeout=5)

    @patch.object(Session, "get")
    def test_tools_all_servers(self, mock_get, client):
        # Mock servers() call
        servers_response = Mock()
        servers_response.json.return_value = ["server1", "server2"]
        servers_response.raise_for_status.return_value = None

        # Mock tools calls for each server
        tools_response = Mock()
        tools_response.json.return_value = {"tools": [{"name": "tool1"}]}
        tools_response.raise_for_status.return_value = None

        mock_get.side_effect = [servers_response, tools_response, tools_response]

        result = client.tools()

        assert result == {"server1": [{"name": "tool1"}], "server2": [{"name": "tool1"}]}
        assert mock_get.call_count == 3

    @patch.object(Session, "get")
    def test_tools_request_error(self, mock_get, client):
        mock_get.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError, match="Error listing tool definitions"):
            client.tools("test_server")

    @patch.object(Session, "post")
    def test_perform_call_success(self, mock_post, client):
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = client._perform_call("test_server", "test_tool", {"param": "value"})

        assert result == {"result": "success"}
        mock_post.assert_called_once_with(
            "http://localhost:8090/api/v1/servers/test_server/tools/test_tool", json={"param": "value"}, timeout=30
        )

    @patch.object(Session, "post")
    def test_perform_call_request_error(self, mock_post, client):
        mock_post.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError, match="Error calling tool 'test_tool' on server 'test_server'"):
            client._perform_call("test_server", "test_tool", {"param": "value"})

    @patch.object(McpdClient, "tools")
    def test_agent_tools(self, mock_tools, client):
        mock_tools.return_value = {
            "server1": [{"name": "tool1", "description": "Test tool"}],
            "server2": [{"name": "tool2", "description": "Another tool"}],
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func2 = Mock()
            mock_create.side_effect = [mock_func1, mock_func2]

            result = client.agent_tools()

            assert result == [mock_func1, mock_func2]
            assert mock_create.call_count == 2

    @patch.object(McpdClient, "tools")
    def test_has_tool_exists(self, mock_tools, client):
        mock_tools.return_value = [{"name": "existing_tool"}, {"name": "another_tool"}]

        result = client.has_tool("test_server", "existing_tool")

        assert result is True
        mock_tools.assert_called_once_with(server_name="test_server")

    @patch.object(McpdClient, "tools")
    def test_has_tool_not_exists(self, mock_tools, client):
        mock_tools.return_value = [{"name": "existing_tool"}, {"name": "another_tool"}]

        result = client.has_tool("test_server", "nonexistent_tool")

        assert result is False

    @patch.object(McpdClient, "tools")
    def test_has_tool_server_error(self, mock_tools, client):
        mock_tools.side_effect = McpdError("Server error")

        result = client.has_tool("test_server", "any_tool")

        assert result is False

    def test_clear_agent_tools_cache(self, client):
        with patch.object(client._function_builder, "clear_cache") as mock_clear:
            client.clear_agent_tools_cache()
            mock_clear.assert_called_once()
