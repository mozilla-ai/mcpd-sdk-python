import contextlib
import inspect
from pathlib import Path
from types import FunctionType
from typing import Any, get_type_hints

import pytest

from mcpd.exceptions import McpdError, ToolBuildError, ValidationError
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

    def test_clear_cache(self, function_builder):
        schema = {
            "name": "cached_tool",
            "description": "A tool to populate the cache.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }
        function_builder.create_function_from_schema(schema, "server")
        assert len(function_builder._function_cache) == 1

        function_builder.clear_cache()

        assert len(function_builder._function_cache) == 0

    def test_get_cached_functions_is_empty_before_anything_is_built(self, function_builder):
        assert function_builder.get_cached_functions() == []

    def test_get_cached_functions_returns_one_function_per_cached_tool(self, function_builder):
        for tool_name in ["first_tool", "second_tool"]:
            function_builder.create_function_from_schema(
                {
                    "name": tool_name,
                    "description": "A tool to populate the cache.",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                },
                "server",
            )

        cached = function_builder.get_cached_functions()

        assert [func._tool_name for func in cached] == ["first_tool", "second_tool"]

    def test_get_cached_functions_returns_a_new_function_each_time(self, function_builder):
        function_builder.create_function_from_schema(
            {
                "name": "cached_tool",
                "description": "A tool to populate the cache.",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            "server",
        )

        assert function_builder.get_cached_functions()[0] is not function_builder.get_cached_functions()[0]

    def test_tool_name_that_is_not_an_identifier_becomes_a_safe_function_name(self, function_builder):
        schema = {
            "name": "test-tool",
            "description": "Test tool",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

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


def _canary_payload(canary_path: Path) -> str:
    """Build a Python expression that writes to `canary_path` when evaluated."""
    return f"__import__('pathlib').Path({str(canary_path)!r}).write_text('executed')"


class TestFunctionBuilderInjection:
    """Tool definitions are untrusted input and must never reach an evaluator."""

    @pytest.fixture
    def function_builder(self, mock_client):
        return FunctionBuilder(mock_client)

    @pytest.fixture
    def canary(self, tmp_path):
        return tmp_path / "canary"

    def test_injected_property_name_does_not_execute(self, function_builder, canary):
        schema = {
            "name": "lookup_word",
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "The word"},
                    f"lang={_canary_payload(canary)},_": {"type": "string", "description": "Language"},
                },
                "required": ["word"],
            },
        }

        with contextlib.suppress(McpdError):
            function_builder.create_function_from_schema(schema, "dictionary")

        assert not canary.exists()

    def test_property_name_that_is_not_an_identifier_is_rejected(self, function_builder):
        schema = {
            "name": "lookup_word",
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {"not-an-identifier": {"type": "string", "description": "Bad"}},
                "required": [],
            },
        }

        with pytest.raises(McpdError):
            function_builder.create_function_from_schema(schema, "dictionary")

    def test_property_name_that_is_a_keyword_is_rejected(self, function_builder):
        schema = {
            "name": "lookup_word",
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {"class": {"type": "string", "description": "Reserved"}},
                "required": [],
            },
        }

        with pytest.raises(McpdError):
            function_builder.create_function_from_schema(schema, "dictionary")

    def test_rejected_parameter_name_is_escaped_in_the_error(self, function_builder):
        schema = {
            "name": "lookup_word",
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {"bad\nWARNING: forged log line": {"type": "string"}},
                "required": [],
            },
        }

        with pytest.raises(McpdError) as exc_info:
            function_builder.create_function_from_schema(schema, "dictionary")

        assert "\n" not in str(exc_info.value)

    def test_injected_description_does_not_execute_when_called(self, function_builder, canary):
        schema = {
            "name": "lookup_word",
            "description": f'Look up a word."""; {_canary_payload(canary)}; """',
            "inputSchema": {
                "type": "object",
                "properties": {"word": {"type": "string", "description": "The word"}},
                "required": ["word"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "dictionary")
        func(word="apple")

        assert not canary.exists()

    def test_injected_tool_name_does_not_execute_when_called(self, function_builder, canary):
        schema = {
            "name": f'x", params) and {_canary_payload(canary)} and client._perform_call("s", "x',
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {"word": {"type": "string", "description": "The word"}},
                "required": ["word"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "dictionary")
        func(word="apple")

        assert not canary.exists()

    def test_injected_parameter_description_does_not_execute_when_called(self, function_builder, canary):
        schema = {
            "name": "lookup_word",
            "description": "Look up a word.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": f'The word."""; {_canary_payload(canary)}; """',
                    }
                },
                "required": ["word"],
            },
        }

        func = function_builder.create_function_from_schema(schema, "dictionary")
        func(word="apple")

        assert not canary.exists()


class TestFunctionBuilderIntrospectionContract:
    """Agent frameworks read these attributes to build the tool schema sent to the model."""

    @pytest.fixture
    def function_builder(self, mock_client):
        return FunctionBuilder(mock_client)

    @pytest.fixture
    def func(self, function_builder):
        schema = {
            "name": "get_current_time",
            "description": "Get the current time in a timezone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name"},
                    "format": {"type": "string", "description": "Output format"},
                },
                "required": ["timezone"],
            },
        }
        return function_builder.create_function_from_schema(schema, "time")

    def test_signature_reports_schema_parameters(self, func):
        assert str(inspect.signature(func)) == "(timezone: str, format: str | None = None) -> Any"

    def test_name_qualifies_server_and_tool(self, func):
        assert func.__name__ == "time__get_current_time"

    def test_docstring_starts_with_tool_description(self, func):
        assert func.__doc__.startswith("Get the current time in a timezone.")

    def test_annotations_are_resolved_types_not_strings(self, func):
        # NOTE: The builder's module enables PEP 563, which turns source annotations into strings.
        assert get_type_hints(func) == {"timezone": str, "format": str | None, "return": Any}

    def test_source_is_retrievable(self, func):
        # A framework that reads a tool's source needs a real source file.
        assert inspect.getsource(func).strip().startswith("def ")

    def test_code_object_does_not_describe_the_parameters(self, func):
        # The signature is attached rather than compiled, so a framework must read
        # it with inspect rather than from the code object.
        assert func.__code__.co_argcount == 0
        assert list(inspect.signature(func).parameters) == ["timezone", "format"]


class TestFunctionBuilderMalformedSchema:
    """A tool definition that does not match its contract is rejected as an McpdError."""

    @pytest.fixture
    def function_builder(self, mock_client):
        return FunctionBuilder(mock_client)

    def test_schema_without_a_name_is_rejected(self, function_builder):
        schema = {"inputSchema": {"type": "object", "properties": {}, "required": []}}

        with pytest.raises(McpdError):
            function_builder.create_function_from_schema(schema, "server")

    def test_property_that_is_not_an_object_is_rejected(self, function_builder):
        schema = {
            "name": "broken",
            "inputSchema": {"type": "object", "properties": {"param": "not an object"}, "required": []},
        }

        with pytest.raises(McpdError):
            function_builder.create_function_from_schema(schema, "server")

    def test_required_name_absent_from_properties_is_rejected(self, function_builder):
        schema = {
            "name": "ghost_param",
            "description": "Requires a parameter it never describes",
            "inputSchema": {
                "type": "object",
                "properties": {"present": {"type": "string"}},
                "required": ["present", "absent"],
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            function_builder.create_function_from_schema(schema, "server")

        assert "absent" in str(exc_info.value)

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"inputSchema": {"type": "object"}}, id="no-name"),
            pytest.param({"name": 42}, id="name-is-not-a-string"),
            pytest.param({"name": "t", "description": 42}, id="description-is-not-a-string"),
            pytest.param({"name": "t", "inputSchema": "nope"}, id="input-schema-is-not-an-object"),
            pytest.param({"name": "t", "inputSchema": {"properties": ["p"]}}, id="properties-is-not-an-object"),
            pytest.param({"name": "t", "inputSchema": {"properties": {42: {}}}}, id="parameter-name-is-not-a-string"),
            pytest.param({"name": "t", "inputSchema": {"properties": {"p": "nope"}}}, id="parameter-is-not-an-object"),
            pytest.param({"name": "t", "inputSchema": {"required": 5}}, id="required-is-not-a-list"),
            pytest.param({"name": "t", "inputSchema": {"required": [5]}}, id="required-name-is-not-a-string"),
            pytest.param({"name": "t", "inputSchema": {"properties": {"p": {"anyOf": []}}}}, id="empty-any-of"),
        ],
    )
    def test_malformed_definition_is_reported_as_a_validation_error(self, function_builder, schema):
        with pytest.raises(ValidationError):
            function_builder.create_function_from_schema(schema, "server")

    def test_unexpected_failure_is_reported_as_a_build_error(self, function_builder, monkeypatch):
        def defect(*args, **kwargs):
            raise RuntimeError("the builder is broken")

        monkeypatch.setattr(function_builder, "_create_docstring", defect)
        schema = {"name": "usable", "inputSchema": {"type": "object", "properties": {}, "required": []}}

        with pytest.raises(ToolBuildError):
            function_builder.create_function_from_schema(schema, "server")
