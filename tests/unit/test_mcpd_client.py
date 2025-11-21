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
        assert not HealthStatus.is_healthy(HealthStatus.TIMEOUT.value)
        assert not HealthStatus.is_healthy(HealthStatus.UNREACHABLE.value)
        assert not HealthStatus.is_healthy(HealthStatus.UNKNOWN.value)

    def test_is_transient(self):
        assert HealthStatus.is_transient(HealthStatus.TIMEOUT.value)
        assert HealthStatus.is_transient(HealthStatus.UNKNOWN.value)
        assert not HealthStatus.is_transient(HealthStatus.OK.value)
        assert not HealthStatus.is_transient(HealthStatus.UNREACHABLE.value)


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

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools(self, mock_health, mock_tools, mock_servers, client, tools_side_effect):
        mock_servers.return_value = ["server1", "server2"]

        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
                "server2": [{"name": "tool2", "description": "Another tool"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "ok"},
            "server2": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func2 = Mock()
            mock_create.side_effect = [mock_func1, mock_func2]

            result = client.agent_tools()

            assert result == [mock_func1, mock_func2]
            assert mock_create.call_count == 2

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_single_server(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering tools by a single server name."""
        mock_servers.return_value = ["server1", "server2"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
                "server2": [{"name": "tool2", "description": "Another tool"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "ok"},
            "server2": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1._server_name = "server1"
            mock_func1._tool_name = "tool1"
            mock_func2 = Mock()
            mock_func2._server_name = "server2"
            mock_func2._tool_name = "tool2"
            mock_create.side_effect = [mock_func1, mock_func2]

            result = client.agent_tools(servers=["server1"])

            # Cache is built from both servers, but filtered to only server1.
            assert result == [mock_func1]
            assert mock_create.call_count == 2

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_multiple_servers(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering tools by multiple server names."""
        mock_servers.return_value = ["server1", "server2", "server3"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
                "server2": [{"name": "tool2", "description": "Another tool"}],
                "server3": [{"name": "tool3", "description": "Third tool"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "ok"},
            "server2": {"status": "ok"},
            "server3": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1._server_name = "server1"
            mock_func1._tool_name = "tool1"
            mock_func2 = Mock()
            mock_func2._server_name = "server2"
            mock_func2._tool_name = "tool2"
            mock_func3 = Mock()
            mock_func3._server_name = "server3"
            mock_func3._tool_name = "tool3"
            mock_create.side_effect = [mock_func1, mock_func2, mock_func3]

            result = client.agent_tools(servers=["server1", "server2"])

            # Cache is built from all 3 servers, but filtered to only server1 and server2.
            assert result == [mock_func1, mock_func2]
            assert mock_create.call_count == 3

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_with_nonexistent_server(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering with server that doesn't exist."""
        mock_servers.return_value = ["server1"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
            }
        )
        mock_health.return_value = {
            "server1": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1._server_name = "server1"
            mock_func1._tool_name = "tool1"
            mock_create.return_value = mock_func1

            result = client.agent_tools(servers=["nonexistent"])

            # Cache is built from server1, but filtered to nonexistent server.
            assert result == []
            assert mock_create.call_count == 1

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_with_empty_servers_list(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering with empty server list."""
        mock_servers.return_value = ["server1"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
            }
        )
        mock_health.return_value = {
            "server1": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1._server_name = "server1"
            mock_func1._tool_name = "tool1"
            mock_create.return_value = mock_func1

            result = client.agent_tools(servers=[])

            # Cache is built from all servers, but filtered to empty list.
            assert result == []
            assert mock_create.call_count == 1

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_without_servers_parameter(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test existing behavior - returns all tools when servers parameter not provided."""
        mock_servers.return_value = ["server1", "server2"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Test tool"}],
                "server2": [{"name": "tool2", "description": "Another tool"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "ok"},
            "server2": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func2 = Mock()
            mock_create.side_effect = [mock_func1, mock_func2]

            result = client.agent_tools()

            assert result == [mock_func1, mock_func2]
            assert mock_create.call_count == 2

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filters_by_health_default(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test that agent_tools filters to healthy servers by default."""
        mock_servers.return_value = ["healthy_server", "unhealthy_server"]
        mock_tools.side_effect = tools_side_effect(
            {
                "healthy_server": [{"name": "tool1", "description": "Tool 1"}],
                "unhealthy_server": [{"name": "tool2", "description": "Tool 2"}],
            }
        )

        mock_health.return_value = {
            "healthy_server": {"status": "ok"},
            "unhealthy_server": {"status": "timeout"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func = Mock()
            mock_create.return_value = mock_func

            result = client.agent_tools()

            # Should only include tool from healthy server.
            assert result == [mock_func]
            assert mock_create.call_count == 1
            mock_create.assert_called_with({"name": "tool1", "description": "Tool 1"}, "healthy_server")

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_all_servers_unhealthy(self, mock_health, mock_tools, mock_servers, client, tools_side_effect):
        """Test behavior when all servers are unhealthy."""
        mock_servers.return_value = ["server1", "server2"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Tool 1"}],
                "server2": [{"name": "tool2", "description": "Tool 2"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "timeout"},
            "server2": {"status": "unreachable"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            result = client.agent_tools()

            # Should return empty list.
            assert result == []
            mock_create.assert_not_called()

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_with_servers_and_health_filtering(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test combining server filtering and health filtering."""
        mock_servers.return_value = ["server1", "server2", "server3"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Tool 1"}],
                "server2": [{"name": "tool2", "description": "Tool 2"}],
                "server3": [{"name": "tool3", "description": "Tool 3"}],
            }
        )

        mock_health.return_value = {
            "server1": {"status": "ok"},
            "server2": {"status": "timeout"},
            "server3": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1._server_name = "server1"
            mock_func1._tool_name = "tool1"
            mock_func3 = Mock()
            mock_func3._server_name = "server3"
            mock_func3._tool_name = "tool3"
            mock_create.side_effect = [mock_func1, mock_func3]

            # Request server1 and server2, but server2 is unhealthy.
            result = client.agent_tools(servers=["server1", "server2"])

            # Cache is built from server1 and server3 (server2 is unhealthy and skipped).
            # Filtered to only server1 and server2, but server2 not in cache.
            assert result == [mock_func1]
            assert mock_create.call_count == 2

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_health_check_exception(self, mock_health, mock_tools, mock_servers, client, tools_side_effect):
        """Test behavior when health check raises exception."""
        mock_servers.return_value = ["server1"]
        mock_tools.side_effect = tools_side_effect(
            {
                "server1": [{"name": "tool1", "description": "Tool 1"}],
            }
        )

        # Health check raises exception.
        mock_health.side_effect = Exception("Health check failed")

        # Exception should propagate.
        with pytest.raises(Exception, match="Health check failed"):
            client.agent_tools()

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_raw_tool_name(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering by raw tool name (cross-cutting)."""
        mock_servers.return_value = ["math", "calc"]
        mock_tools.side_effect = tools_side_effect(
            {
                "math": [
                    {"name": "add", "description": "Add numbers"},
                    {"name": "subtract", "description": "Subtract numbers"},
                ],
                "calc": [
                    {"name": "add", "description": "Calculator add"},
                    {"name": "multiply", "description": "Multiply"},
                ],
            }
        )

        mock_health.return_value = {
            "math": {"status": "ok"},
            "calc": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1.__name__ = "math__add"
            mock_func1._tool_name = "add"
            mock_func2 = Mock()
            mock_func2.__name__ = "calc__add"
            mock_func2._tool_name = "add"
            mock_func3 = Mock()
            mock_func3.__name__ = "math__subtract"
            mock_func3._tool_name = "subtract"
            mock_func4 = Mock()
            mock_func4.__name__ = "calc__multiply"
            mock_func4._tool_name = "multiply"
            mock_create.side_effect = [mock_func1, mock_func3, mock_func2, mock_func4]

            result = client.agent_tools(tools=["add"])

            # Should include 'add' from ALL servers that have it.
            assert len(result) == 2
            assert mock_create.call_count == 4  # All functions created, but only 2 match filter

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_prefixed_tool_name(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering by prefixed tool name (specific server+tool)."""
        mock_servers.return_value = ["time", "other"]
        mock_tools.side_effect = tools_side_effect(
            {
                "time": [{"name": "get_current_time", "description": "Get time"}],
                "other": [{"name": "get_current_time", "description": "Other time"}],
            }
        )

        mock_health.return_value = {
            "time": {"status": "ok"},
            "other": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1.__name__ = "time__get_current_time"
            mock_func1._tool_name = "get_current_time"
            mock_func2 = Mock()
            mock_func2.__name__ = "other__get_current_time"
            mock_func2._tool_name = "get_current_time"
            mock_create.side_effect = [mock_func1, mock_func2]

            result = client.agent_tools(tools=["time__get_current_time"])

            # Should include only the exact match.
            assert len(result) == 1
            assert result[0].__name__ == "time__get_current_time"
            assert mock_create.call_count == 2  # Both created, only one matches

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_mixed_formats(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering with both raw and prefixed names."""
        mock_servers.return_value = ["math", "time"]
        mock_tools.side_effect = tools_side_effect(
            {
                "math": [{"name": "add", "description": "Add"}, {"name": "subtract", "description": "Subtract"}],
                "time": [{"name": "get_current_time", "description": "Get time"}],
            }
        )

        mock_health.return_value = {
            "math": {"status": "ok"},
            "time": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1.__name__ = "math__add"
            mock_func1._tool_name = "add"
            mock_func2 = Mock()
            mock_func2.__name__ = "math__subtract"
            mock_func2._tool_name = "subtract"
            mock_func3 = Mock()
            mock_func3.__name__ = "time__get_current_time"
            mock_func3._tool_name = "get_current_time"
            mock_create.side_effect = [mock_func1, mock_func2, mock_func3]

            result = client.agent_tools(tools=["add", "time__get_current_time"])

            # Should include all 'add' tools plus the specific time tool.
            assert len(result) == 2
            assert mock_create.call_count == 3  # All created, only 2 match

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_by_servers_and_tools(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test combining server and tool filtering."""
        mock_servers.return_value = ["time", "math", "fetch"]
        mock_tools.side_effect = tools_side_effect(
            {
                "time": [{"name": "get_current_time", "description": "Get time"}],
                "math": [{"name": "add", "description": "Add"}],
                "fetch": [{"name": "fetch_url", "description": "Fetch"}],
            }
        )

        mock_health.return_value = {
            "time": {"status": "ok"},
            "math": {"status": "ok"},
            "fetch": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func1 = Mock()
            mock_func1.__name__ = "time__get_current_time"
            mock_func1._server_name = "time"
            mock_func1._tool_name = "get_current_time"
            mock_func2 = Mock()
            mock_func2.__name__ = "math__add"
            mock_func2._server_name = "math"
            mock_func2._tool_name = "add"
            mock_func3 = Mock()
            mock_func3.__name__ = "fetch__fetch_url"
            mock_func3._server_name = "fetch"
            mock_func3._tool_name = "fetch_url"
            mock_create.side_effect = [mock_func1, mock_func2, mock_func3]

            result = client.agent_tools(servers=["time", "math"], tools=["add", "get_current_time"])

            # Cache is built from all 3 servers, filtered by both servers and tools.
            assert len(result) == 2
            assert set(result) == {mock_func1, mock_func2}
            assert mock_create.call_count == 3

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_with_nonexistent_tool(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering with tool that doesn't exist."""
        mock_servers.return_value = ["time"]
        mock_tools.side_effect = tools_side_effect(
            {
                "time": [{"name": "get_current_time", "description": "Get time"}],
            }
        )

        mock_health.return_value = {
            "time": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func = Mock()
            mock_func.__name__ = "time__get_current_time"
            mock_func._tool_name = "get_current_time"
            mock_create.return_value = mock_func

            result = client.agent_tools(tools=["nonexistent_tool"])

            # Should return empty list (function created but filtered out).
            assert result == []
            assert mock_create.call_count == 1

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_filter_with_empty_tools_list(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test filtering with empty tool list."""
        mock_servers.return_value = ["time"]
        mock_tools.side_effect = tools_side_effect(
            {
                "time": [{"name": "get_current_time", "description": "Get time"}],
            }
        )

        mock_health.return_value = {
            "time": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func = Mock()
            mock_func.__name__ = "time__get_current_time"
            mock_func._tool_name = "get_current_time"
            mock_create.return_value = mock_func

            result = client.agent_tools(tools=[])

            # Should return empty list (function created but empty filter matches nothing).
            assert result == []
            assert mock_create.call_count == 1

    @patch.object(McpdClient, "servers")
    @patch.object(McpdClient, "tools")
    @patch.object(McpdClient, "server_health")
    def test_agent_tools_handles_tool_name_with_double_underscore(
        self, mock_health, mock_tools, mock_servers, client, tools_side_effect
    ):
        """Test that tool names containing __ are handled correctly."""
        mock_servers.return_value = ["test"]
        mock_tools.side_effect = tools_side_effect(
            {
                "test": [{"name": "my__special__tool", "description": "Special tool"}],
            }
        )

        mock_health.return_value = {
            "test": {"status": "ok"},
        }

        with patch.object(client._function_builder, "create_function_from_schema") as mock_create:
            mock_func = Mock()
            mock_func.__name__ = "test__my__special__tool"
            mock_func._tool_name = "my__special__tool"
            mock_create.return_value = mock_func

            # Filter by raw name (should work).
            result = client.agent_tools(tools=["my__special__tool"])
            assert len(result) == 1
            assert mock_create.call_count == 1

            # Reset mock.
            mock_create.reset_mock()
            mock_create.return_value = mock_func

            # Filter by prefixed name (should work).
            result = client.agent_tools(tools=["test__my__special__tool"])
            assert len(result) == 1
            assert mock_create.call_count == 1

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
        # Test that ServerUnhealthyError returns False
        mock_health.side_effect = ServerUnhealthyError(
            "Server is unhealthy", server_name="test_server", health_status="unreachable"
        )
        result = client.is_server_healthy("test_server")
        assert result is False

        # Test that ServerNotFoundError returns False
        mock_health.side_effect = ServerNotFoundError("Server not found", server_name="test_server")
        result = client.is_server_healthy("test_server")
        assert result is False

        # Test that generic McpdError propagates
        mock_health.side_effect = McpdError("Health check failed")
        with pytest.raises(McpdError, match="Health check failed"):
            client.is_server_healthy("test_server")

    def test_cacheable_exceptions(self):
        expected = {ServerNotFoundError, ServerUnhealthyError, AuthenticationError}
        assert expected == set(McpdClient._CACHEABLE_EXCEPTIONS)

    def test_server_health_cache_maxsize(self):
        assert McpdClient._SERVER_HEALTH_CACHE_MAXSIZE == 100

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

    @patch.object(Session, "get")
    def test_server_health_with_cacheable_exceptions(self, mock_get):
        exceptions = [
            ServerNotFoundError("Server not found", "test_server"),
            ServerUnhealthyError("server is unhealthy", "test_server", "unreachable"),
            AuthenticationError(),
        ]

        # Verify our test covers all cacheable exceptions
        client = McpdClient(api_endpoint="http://localhost:8090")
        assert len(exceptions) == len(client._CACHEABLE_EXCEPTIONS)

        for exc in exceptions:
            client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)
            # First call raises a cacheable exception
            mock_get.side_effect = exc
            with pytest.raises(type(exc)) as e:
                client.server_health("test_server")

            assert type(e.value) in client._CACHEABLE_EXCEPTIONS
            mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

            # Second call should use the cached exception
            with pytest.raises(type(exc)) as e2:
                client.server_health("test_server")

            # Verify this is still a cacheable exception type
            assert type(e2.value) in client._CACHEABLE_EXCEPTIONS
            # Verify the mock was still only called once (cache was used)
            mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

            # Reset the mock for next iteration
            mock_get.reset_mock()

    @patch.object(Session, "get")
    def test_server_health_cache_with_noncacheable_exception(self, mock_get):
        # Ensure no caching occurs for non-cacheable exceptions
        client = McpdClient(api_endpoint="http://localhost:8090", server_health_cache_ttl=math.inf)
        mock_get.side_effect = RequestException("Connection failed")

        with pytest.raises(McpdError) as e:
            client.server_health("test_server")

        assert not isinstance(e.value, client._CACHEABLE_EXCEPTIONS)

        mock_get.assert_called_once_with("http://localhost:8090/api/v1/health/servers/test_server", timeout=5)

        with pytest.raises(McpdError) as e2:
            client.server_health("test_server")
        assert not isinstance(e2.value, client._CACHEABLE_EXCEPTIONS)

        # Should be called twice since exception wasn't cached
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
