import pytest

from mcpd import McpdError
from mcpd.dynamic_caller import DynamicCaller, ServerProxy


class TestDynamicCaller:
    @pytest.fixture
    def dynamic_caller(self, mock_client):
        return DynamicCaller(mock_client)

    def test_init(self, mock_client):
        caller = DynamicCaller(mock_client)
        assert caller._client is mock_client

    def test_getattr_returns_server_proxy(self, dynamic_caller, mock_client):
        server_proxy = dynamic_caller.test_server

        assert isinstance(server_proxy, ServerProxy)
        assert server_proxy._client is mock_client
        assert server_proxy._server_name == "test_server"

    def test_multiple_server_access(self, dynamic_caller, mock_client):
        server1 = dynamic_caller.server1
        server2 = dynamic_caller.server2

        assert server1._server_name == "server1"
        assert server2._server_name == "server2"
        assert server1._client is mock_client
        assert server2._client is mock_client


class TestServerProxy:
    @pytest.fixture
    def server_proxy(self, mock_client):
        return ServerProxy(mock_client, "test_server")

    def test_init(self, mock_client):
        proxy = ServerProxy(mock_client, "test_server")
        assert proxy._client is mock_client
        assert proxy._server_name == "test_server"

    def test_getattr_tool_exists(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        tool_function = server_proxy.test_tool

        assert callable(tool_function)
        mock_client.has_tool.assert_called_once_with("test_server", "test_tool")

    def test_getattr_tool_not_exists(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = False

        with pytest.raises(McpdError, match="Tool 'nonexistent_tool' not found on server 'test_server'"):
            _ = server_proxy.nonexistent_tool

    def test_tool_function_execution(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        tool_function = server_proxy.test_tool
        result = tool_function(param1="value1", param2="value2")

        assert result == {"result": "success"}
        mock_client._perform_call.assert_called_once_with(
            "test_server", "test_tool", {"param1": "value1", "param2": "value2"}
        )

    def test_tool_function_no_params(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        tool_function = server_proxy.test_tool
        result = tool_function()

        assert result == {"result": "success"}
        mock_client._perform_call.assert_called_once_with("test_server", "test_tool", {})

    def test_tool_function_with_kwargs(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        tool_function = server_proxy.test_tool
        result = tool_function(
            string_param="test", int_param=42, bool_param=True, list_param=["a", "b", "c"], dict_param={"key": "value"}
        )

        expected_params = {
            "string_param": "test",
            "int_param": 42,
            "bool_param": True,
            "list_param": ["a", "b", "c"],
            "dict_param": {"key": "value"},
        }

        assert result == {"result": "success"}
        mock_client._perform_call.assert_called_once_with("test_server", "test_tool", expected_params)

    def test_multiple_tool_calls(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        tool1 = server_proxy.tool1
        tool2 = server_proxy.tool2

        result1 = tool1(param="value1")
        result2 = tool2(param="value2")

        assert result1 == {"result": "success"}
        assert result2 == {"result": "success"}

        assert mock_client._perform_call.call_count == 2
        mock_client._perform_call.assert_any_call("test_server", "tool1", {"param": "value1"})
        mock_client._perform_call.assert_any_call("test_server", "tool2", {"param": "value2"})

    def test_has_tool_called_for_each_access(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True

        # Access the same tool multiple times
        _ = server_proxy.test_tool
        _ = server_proxy.test_tool

        # has_tool should be called each time
        assert mock_client.has_tool.call_count == 2
        mock_client.has_tool.assert_any_call("test_server", "test_tool")

    def test_error_propagation(self, server_proxy, mock_client):
        mock_client.has_tool.return_value = True
        mock_client._perform_call.side_effect = Exception("API Error")

        tool_function = server_proxy.test_tool

        with pytest.raises(Exception, match="API Error"):
            tool_function(param="value")


class TestIntegration:
    def test_full_call_chain(self, mock_client):
        # Test the complete client.call.server.tool() chain
        caller = DynamicCaller(mock_client)

        result = caller.test_server.test_tool(param1="value1", param2="value2")

        assert result == {"result": "success"}
        mock_client.has_tool.assert_called_once_with("test_server", "test_tool")
        mock_client._perform_call.assert_called_once_with(
            "test_server", "test_tool", {"param1": "value1", "param2": "value2"}
        )

    def test_multiple_servers_and_tools(self, mock_client):
        caller = DynamicCaller(mock_client)

        # Call different tools on different servers
        result1 = caller.server1.tool1(param="value1")
        result2 = caller.server2.tool2(param="value2")
        result3 = caller.server1.tool3(param="value3")

        assert result1 == {"result": "success"}
        assert result2 == {"result": "success"}
        assert result3 == {"result": "success"}

        # Verify all calls were made correctly
        has_tool_calls = mock_client.has_tool.call_args_list
        perform_call_calls = mock_client._perform_call.call_args_list

        assert len(has_tool_calls) == 3
        assert len(perform_call_calls) == 3

        # Check that has_tool was called with correct arguments
        assert has_tool_calls[0][0] == ("server1", "tool1")
        assert has_tool_calls[1][0] == ("server2", "tool2")
        assert has_tool_calls[2][0] == ("server1", "tool3")

        # Check that _perform_call was called with correct arguments
        assert perform_call_calls[0][0] == ("server1", "tool1", {"param": "value1"})
        assert perform_call_calls[1][0] == ("server2", "tool2", {"param": "value2"})
        assert perform_call_calls[2][0] == ("server1", "tool3", {"param": "value3"})
