from types import NoneType
from typing import Any, Literal, Union


class TypeConverter:
    """Handles JSON schema to Python type conversion."""

    @staticmethod
    def json_type_to_python_type(json_type: str, schema_def: dict[str, Any]) -> Any:
        """Convert JSON schema types to Python type annotations."""
        if json_type == "string":
            if "enum" in schema_def:
                enum_values = tuple(schema_def["enum"])
                try:
                    return Literal[enum_values]
                except TypeError:
                    # Fallback for complex enum handling
                    result = Literal[enum_values[0]]
                    for val in enum_values[1:]:
                        result = result | Literal[val]
                    return result
            return str
        elif json_type == "number":
            return int | float
        elif json_type == "integer":
            return int
        elif json_type == "boolean":
            return bool
        elif json_type == "array":
            if "items" in schema_def:
                item_type = TypeConverter.parse_schema_type(schema_def["items"])
                return list[item_type]
            return list[Any]
        elif json_type == "object":
            return dict[str, Any]
        else:
            return Any

    @staticmethod
    def parse_schema_type(schema_def: dict[str, Any]) -> Any:
        """Parse a schema definition and return the appropriate Python type."""
        if "anyOf" in schema_def:
            union_types = []
            for any_schema in schema_def["anyOf"]:
                union_types.append(TypeConverter.parse_schema_type(any_schema))

            result_type = union_types[0]
            for union_type in union_types[1:]:
                result_type = result_type | union_type
            return result_type

        if "type" in schema_def:
            return TypeConverter.json_type_to_python_type(schema_def["type"], schema_def)

        return Any
