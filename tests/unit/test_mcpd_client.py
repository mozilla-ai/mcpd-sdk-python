import math
from unittest.mock import Mock, patch

import pytest
from requests import Session
from requests.exceptions import RequestException

from mcpd import AuthenticationError, HealthStatus, McpdClient, McpdError, ServerNotFoundError, ServerUnhealthyError


class TestHealthStatus:
    def test_enum_values(self):
        assert HealthStatus.OK.value == "ok"
        assert HealthStatus.TIMEOUT.value == "timeout"
        assert HealthStatus.UNREACHABLE.value == "unreachable"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_is_healthy(self):
        assert HealthStatus.is_healthy(HealthStatus.OK.value)


class TestMcpdClient:
    def test_init_basic(self):
        client = McpdClient(api_endpoint="http://localhost:9999")
        assert client._endpoint == "http://localhost:9999"
        assert client._api_key is None
        assert hasattr(client, "_session")
        assert hasattr(client, "call")

    def test_init_with_auth(self):
        client = McpdClient(api_endpoint="http://localhost:9090", api_key="test-key123")
        assert client._endpoint == "http://localhost:9090"
        assert client._api_key == "test-key123"  # pragma: allowlist secret
        assert "Authorization" in client._session.headers
        assert client._session.headers["Authorization"] == "Bearer test-key123"  # pragma: allowlist secret

    def test_init_strips_trailing_slash(self):
        client = McpdClient("http://localhost:8090/")
        assert client._endpoint == "http://localhost:8090"

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

    @patch.object(Session, "get")
    def test_health_single_server(self, mock_get, client):
        mock_response = Mock()
        mock_response.json.return_value = {"name": "test_server", "status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.server_health("test_server")

        assert result == {"name": "test_server", "status": "ok"}
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

    @patch.object(Session, "get")
    def test_health_all_servers(self, mock_get, client):
        mock_response = Mock()
        mock_response.json.return_value = {
            "servers": [{"name": "server1", "status": "ok"}, {"name": "server2", "status": "unreachable"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = client.server_health()

        assert result == {
            "server1": {"name": "server1", "status": "ok"},
            "server2": {"name": "server2", "status": "unreachable"},
        }
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers", timeout=5)

    @patch.object(Session, "get")
    def test_health_request_error(self, mock_get, client):
        mock_get.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError, match="Error retrieving health status for all servers"):
            client.server_health()

    @patch.object(McpdClient, "server_health")
    def test_is_healthy_true(self, mock_health, client):
        mock_health.return_value = {"name": "test_server", "status": "ok"}

        result = client.is_server_healthy("test_server")

        assert result is True
        mock_health.assert_called_once_with(server_name="test_server")

    @patch.object(McpdClient, "server_health")
    def test_is_healthy_false(self, mock_health, client):
        for status in ["timeout", "unknown", "unreachable"]:
            mock_health.return_value = {"name": "test_server", "status": status}

            result = client.is_server_healthy("test_server")

            assert result is False

    @patch.object(McpdClient, "server_health")
    def test_is_healthy_error(self, mock_health, client):
        mock_health.side_effect = McpdError("Health check failed")

        result = client.is_server_healthy("test_server")

        assert result is False

    def test_cacheable_exceptions(self):
        expected = {ServerNotFoundError, ServerUnhealthyError, AuthenticationError}
        assert expected == set(McpdClient._CACHEABLE_EXCEPTIONS)

    @patch.object(Session, "get")
    def test_server_health_cache(self, mock_get):
        client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)

        mock_response = Mock()
        mock_response.json.return_value = {"name": "test_server", "status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # First call should invoke the actual method
        result1 = client.server_health("test_server")
        assert result1 == {"name": "test_server", "status": "ok"}
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

        # Second call should use the cached result
        result2 = client.server_health("test_server")
        assert result2 == {"name": "test_server", "status": "ok"}
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

    def test_server_health_with_cacheable_exceptions(self):
        exceptions = [
            ServerNotFoundError("Server not found", "test_server"),
            ServerUnhealthyError("server is unhealthy", "test_server", "unreachable"),
            AuthenticationError(),
        ]

        for exc in exceptions:
            client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)
            with patch("requests.Session.get") as server_health_mock:
                # First call raises a cacheable exception
                server_health_mock.side_effect = exc
                with pytest.raises(type(exc)) as e:
                    client.server_health("test_server")
                    assert e in client._CACHEABLE_EXCEPTIONS
                server_health_mock.assert_called_once_with(
                    "http://localhost:8090/api/v1/health/servers/test_server", timeout=5
                )

                # Second call should use the cached exception
                with pytest.raises(type(exc)) as e:
                    client.server_health("test_server")
                    assert e in client._CACHEABLE_EXCEPTIONS
                server_health_mock.assert_called_once_with(
                    "http://localhost:8090/api/v1/health/servers/test_server", timeout=5
                )

    @patch.object(Session, "get")
    def test_server_health_cache_with_noncacheable_exception(self, mock_get):
        # Ensure no caching occurs for non-cacheable exceptions
        client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)
        mock_get.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError) as e:
            client.server_health("test_server")
            assert e not in client._CACHEABLE_EXCEPTIONS

        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

        with pytest.raises(McpdError):
            client.server_health("test_server")
            assert e not in client._CACHEABLE_EXCEPTIONS

        assert mock_get.call_count == 2

    @patch.object(Session, "get")
    def test_server_health_with_disabled_cache(self, mock_get):
        client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=0)

        mock_response = Mock()
        mock_response.json.return_value = {"name": "test_server", "status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result1 = client.server_health("test_server")
        assert result1 == {"name": "test_server", "status": "ok"}
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

        # Subsequent call should not use cache and invoke the actual method again
        result2 = client.server_health("test_server")
        assert result2 == {"name": "test_server", "status": "ok"}
        assert mock_get.call_count == 2

    @patch.object(Session, "get")
    def test_clear_server_health_cache(self, mock_get):
        client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)

        mock_response = Mock()
        mock_response.json.return_value = {"name": "test_server", "status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result1 = client.server_health("test_server")
        assert result1 == {"name": "test_server", "status": "ok"}
        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

        # Clear the cache
        client.clear_server_health_cache("test_server")

        # Subsequent call should not use cache and invoke the actual method again
        result2 = client.server_health("test_server")
        assert result2 == {"name": "test_server", "status": "ok"}
        assert mock_get.call_count == 2
