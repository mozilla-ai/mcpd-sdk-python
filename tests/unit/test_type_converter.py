from typing import Any, Literal

from mcpd.type_converter import TypeConverter


class TestTypeConverter:
    def test_json_type_to_python_type_string(self):
        result = TypeConverter.json_type_to_python_type("string", {})
        assert result is str

    def test_json_type_to_python_type_string_with_enum(self):
        schema = {"enum": ["option1", "option2", "option3"]}
        result = TypeConverter.json_type_to_python_type("string", schema)

        # Should be a Literal type
        assert hasattr(result, "__origin__")
        assert result.__origin__ is Literal

    def test_json_type_to_python_type_string_with_single_enum(self):
        schema = {"enum": ["single_option"]}
        result = TypeConverter.json_type_to_python_type("string", schema)

        # Should handle single enum value
        assert hasattr(result, "__origin__") or result == Literal["single_option"]

    def test_json_type_to_python_type_number(self):
        result = TypeConverter.json_type_to_python_type("number", {})
        assert result == (int | float)

    def test_json_type_to_python_type_integer(self):
        result = TypeConverter.json_type_to_python_type("integer", {})
        assert result is int

    def test_json_type_to_python_type_boolean(self):
        result = TypeConverter.json_type_to_python_type("boolean", {})
        assert result is bool

    def test_json_type_to_python_type_null(self):
        from types import NoneType

        result = TypeConverter.json_type_to_python_type("null", {})
        assert result is NoneType

    def test_json_type_to_python_type_array_with_items(self):
        schema = {"items": {"type": "string"}}
        result = TypeConverter.json_type_to_python_type("array", schema)
        assert result == list[str]

    def test_json_type_to_python_type_array_without_items(self):
        result = TypeConverter.json_type_to_python_type("array", {})
        assert result == list[Any]

    def test_json_type_to_python_type_object(self):
        result = TypeConverter.json_type_to_python_type("object", {})
        assert result == dict[str, Any]

    def test_json_type_to_python_type_unknown(self):
        result = TypeConverter.json_type_to_python_type("unknown_type", {})
        assert result == Any

    def test_parse_schema_type_simple_type(self):
        schema = {"type": "string"}
        result = TypeConverter.parse_schema_type(schema)
        assert result is str

    def test_parse_schema_type_anyof_simple(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = TypeConverter.parse_schema_type(schema)
        assert result == (str | int)

    def test_parse_schema_type_anyof_complex(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]}
        result = TypeConverter.parse_schema_type(schema)
        assert result == (str | int | bool)

    def test_parse_schema_type_anyof_with_arrays(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "integer"}}]}
        result = TypeConverter.parse_schema_type(schema)
        assert result == (str | list[int])

    def test_parse_schema_type_nested_anyof(self):
        schema = {"anyOf": [{"type": "string"}, {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}]}
        result = TypeConverter.parse_schema_type(schema)
        # Should handle nested anyOf
        assert result == (str | (int | bool))

    def test_parse_schema_type_no_type_field(self):
        schema = {"description": "Some schema without type"}
        result = TypeConverter.parse_schema_type(schema)
        assert result == Any

    def test_parse_schema_type_empty_schema(self):
        schema = {}
        result = TypeConverter.parse_schema_type(schema)
        assert result == Any

    def test_parse_schema_type_array_with_complex_items(self):
        schema = {"type": "array", "items": {"type": "object"}}
        result = TypeConverter.parse_schema_type(schema)
        assert result == list[dict[str, Any]]

    def test_parse_schema_type_enum_with_mixed_types(self):
        # Test enum with different value types
        schema = {"type": "string", "enum": ["string_val", "another_string"]}
        result = TypeConverter.parse_schema_type(schema)

        # Should be a Literal type
        assert hasattr(result, "__origin__")
        assert result.__origin__ is Literal

    def test_complex_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "nested_array": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}
            },
        }
        # This tests the overall parsing, not specific property parsing
        result = TypeConverter.parse_schema_type(schema)
        assert result == dict[str, Any]

    def test_array_with_anyof_items(self):
        schema = {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}
        result = TypeConverter.parse_schema_type(schema)
        assert result == list[str | int]

    def test_enum_fallback_handling(self):
        # Test the fallback mechanism for complex enum handling
        schema = {"type": "string", "enum": ["val1", "val2", "val3", "val4"]}
        result = TypeConverter.parse_schema_type(schema)

        # Should handle the enum properly
        assert hasattr(result, "__origin__")
        assert result.__origin__ is Literal

    def test_json_type_to_python_type_array_recursive(self):
        schema = {"items": {"type": "array", "items": {"type": "string"}}}
        result = TypeConverter.json_type_to_python_type("array", schema)
        assert result == list[list[str]]

    def test_parse_schema_type_with_null_in_anyof(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = TypeConverter.parse_schema_type(schema)
        # Should handle null type properly
        from types import NoneType

        assert result == (str | NoneType)  # null maps to NoneType
