from types import FunctionType

import pytest

from mcpd.exceptions import McpdError
from mcpd.function_builder import FunctionBuilder


class TestFunctionBuilder:
    @pytest.fixture
    def function_builder(self, mock_client):
        return FunctionBuilder(mock_client)

    def test_safe_name_alphanumeric(self, function_builder):
        assert function_builder._safe_name("test_func") == "test_func"
        assert function_builder._safe_name("TestFunc123") == "TestFunc123"

    def test_safe_name_special_chars(self, function_builder):
        assert function_builder._safe_name("test-func") == "test_func"
        assert function_builder._safe_name("test.func") == "test_func"
        assert function_builder._safe_name("test func") == "test_func"
        assert function_builder._safe_name("test@func#") == "test_func_"

    def test_safe_name_leading_digit(self, function_builder):
        assert function_builder._safe_name("123test") == "_123test"
        assert function_builder._safe_name("9abc") == "_9abc"

    def test_function_name(self, function_builder):
        result = function_builder._function_name("test_server", "test_tool")
        assert result == "test_server__test_tool"

    def test_function_name_with_special_chars(self, function_builder):
        result = function_builder._function_name("test-server", "test.tool")
        assert result == "test_server__test_tool"

    def test_create_function_from_schema_basic(self, function_builder):
        schema = {
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

        func = function_builder.create_function_from_schema(schema, "test_server")

        assert isinstance(func, FunctionType)
        assert func.__name__ == "test_server__test_tool"
        assert "A test tool" in func.__doc__
        assert "param1" in func.__doc__
        assert "param2" in func.__doc__

    def test_create_function_from_schema_execution(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "First parameter"}},
                "required": ["param1"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "test_server")
        result = func(param1="test_value")

        assert result == {"result": "success"}
        function_builder._client._perform_call.assert_called_once_with(
            "test_server", "test_tool", {"param1": "test_value"}
        )

    def test_create_function_from_schema_missing_required(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param1": {"type": "string", "description": "First parameter"}},
                "required": ["param1"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "test_server")

        with pytest.raises(McpdError, match="Missing required parameters"):
            func(param1=None)

    def test_create_function_from_schema_optional_params(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Required parameter"},
                    "param2": {"type": "string", "description": "Optional parameter"},
                },
                "required": ["param1"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "test_server")

        # Test with only required param
        _ = func(param1="test")
        function_builder._client._perform_call.assert_called_with("test_server", "test_tool", {"param1": "test"})

        # Test with optional param
        _ = func(param1="test", param2="optional")
        function_builder._client._perform_call.assert_called_with(
            "test_server", "test_tool", {"param1": "test", "param2": "optional"}
        )

    def test_create_function_from_schema_no_params(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "A test tool with no parameters",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

        func = function_builder.create_function_from_schema(schema, "test_server")
        _ = func()

        function_builder._client._perform_call.assert_called_once_with("test_server", "test_tool", {})

    def test_create_function_from_schema_caching(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {"param1": {"type": "string"}}, "required": ["param1"]},
        }

        func1 = function_builder.create_function_from_schema(schema, "test_server")
        func2 = function_builder.create_function_from_schema(schema, "test_server")

        # Should be different instances but same functionality
        assert func1 is not func2
        assert func1.__name__ == func2.__name__
        # Cached instances should preserve metadata attributes.
        assert func1._server_name == func2._server_name == "test_server"
        assert func1._tool_name == func2._tool_name == "test_tool"

    def test_create_annotations_basic_types(self, function_builder):
        schema = {
            "inputSchema": {
                "type": "object",
                "properties": {
                    "str_param": {"type": "string"},
                    "int_param": {"type": "integer"},
                    "bool_param": {"type": "boolean"},
                },
                "required": ["str_param"],
            }
        }

        annotations = function_builder._create_annotations(schema)

        assert annotations["str_param"] is str
        assert annotations["int_param"] == (int | None)
        assert annotations["bool_param"] == (bool | None)

    def test_create_docstring_with_params(self, function_builder):
        schema = {
            "description": "Test function description",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        }

        docstring = function_builder._create_docstring(schema)

        assert "Test function description" in docstring
        assert "Args:" in docstring
        assert "param1: First parameter" in docstring
        assert "param2: Second parameter (optional)" in docstring
        assert "Returns:" in docstring
        assert "Raises:" in docstring

    def test_create_docstring_no_params(self, function_builder):
        schema = {
            "description": "Test function description",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

        docstring = function_builder._create_docstring(schema)

        assert "Test function description" in docstring
        assert "Args:" not in docstring
        assert "Returns:" in docstring

    def test_create_namespace(self, function_builder):
        namespace = function_builder._create_namespace()

        assert "McpdError" in namespace
        assert "client" in namespace
        assert namespace["client"] is function_builder._client
        assert "Any" in namespace
        assert "str" in namespace
        assert "int" in namespace

    def test_clear_cache(self, function_builder):
        # Add something to cache
        function_builder._function_cache["test"] = {"data": "test"}

        function_builder.clear_cache()

        assert len(function_builder._function_cache) == 0

    def test_build_function_code_structure(self, function_builder):
        schema = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Required parameter"},
                    "param2": {"type": "string", "description": "Optional parameter"},
                },
                "required": ["param1"],
            },
        }

        code = function_builder._build_function_code(schema, "test_server")

        assert "def test_server__test_tool(param1, param2=None):" in code
        assert "required_params = ['param1']" in code
        assert 'client._perform_call("test_server", "test_tool", params)' in code
        assert "Test tool" in code

    def test_create_function_compilation_error(self, function_builder):
        # Create a schema that would cause compilation issues
        schema = {
            "name": "test-tool",  # This should be sanitized
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

        # Should not raise an error due to name sanitization
        func = function_builder.create_function_from_schema(schema, "test-server")
        assert func.__name__ == "test_server__test_tool"

    def test_function_has_server_name_attribute(self, function_builder):
        """Test that generated function has _server_name attribute."""
        schema = {"name": "test_tool", "description": "Test", "inputSchema": {}}

        func = function_builder.create_function_from_schema(schema, "test_server")

        assert hasattr(func, "_server_name")
        assert func._server_name == "test_server"

    def test_function_has_tool_name_attribute(self, function_builder):
        """Test that generated function has _tool_name attribute."""
        schema = {"name": "test_tool", "description": "Test", "inputSchema": {}}

        func = function_builder.create_function_from_schema(schema, "test_server")

        assert hasattr(func, "_tool_name")
        assert func._tool_name == "test_tool"

    def test_metadata_attributes_with_special_characters(self, function_builder):
        """Test metadata attributes when names contain special characters."""
        schema = {"name": "test-tool.v2", "description": "Test", "inputSchema": {}}

        func = function_builder.create_function_from_schema(schema, "test-server")

        # Function name should be sanitized.
        assert func.__name__ == "test_server__test_tool_v2"
        # But metadata should have original names.
        assert func._server_name == "test-server"
        assert func._tool_name == "test-tool.v2"
