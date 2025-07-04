from types import FunctionType
from typing import Any

from .exceptions import McpdError
from .type_converter import TypeConverter


class FunctionBuilder:
    """Builds callable functions from JSON schemas using string compilation."""

    def __init__(self, client):
        self.client = client
        self._function_cache = {}

    def create_function_from_schema(self, schema: dict[str, Any], server_name: str) -> FunctionType:
        """Create a callable function from a JSON schema."""
        cache_key = f"{server_name}__{schema.get('name', '')}"

        if cache_key in self._function_cache:
            cached_func = self._function_cache[cache_key]
            return cached_func["create_function"](cached_func["annotations"])

        try:
            function_code = self._build_function_code(schema, server_name)
            annotations = self._create_annotations(schema)
            compiled_code = compile(function_code, f"<{cache_key}>", "exec")

            # Execute and get the function
            namespace = self._create_namespace()
            exec(compiled_code, namespace)
            function_name = f"{server_name}__{schema['name']}"
            created_function = namespace[function_name]
            created_function.__annotations__ = annotations

            # Cache the function creation details
            def create_function_instance(annotations):
                temp_namespace = namespace.copy()
                exec(compiled_code, temp_namespace)
                new_func = temp_namespace[function_name]
                new_func.__annotations__ = annotations.copy()
                return new_func

            self._function_cache[cache_key] = {
                "compiled_code": compiled_code,
                "annotations": annotations,
                "create_function": create_function_instance,
            }

            return created_function

        except Exception as e:
            raise McpdError(f"Error creating function {cache_key}: {e}") from e

    def _build_function_code(self, schema: dict[str, Any], server_name: str) -> str:
        """Build the function code string."""
        function_name = f"{server_name}__{schema['name']}"
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        # Sort parameters: required first, then optional
        required_param_names = [p for p in properties.keys() if p in required_params]
        optional_param_names = [p for p in properties.keys() if p not in required_params]
        sorted_param_names = required_param_names + optional_param_names
        param_declarations = []

        # Build parameter signature
        # for param_name in properties.keys():
        for param_name in sorted_param_names:
            if param_name in required_params:
                param_declarations.append(param_name)
            else:
                param_declarations.append(f"{param_name}=None")

        param_signature = ", ".join(param_declarations)
        docstring = self._create_docstring(schema)

        function_lines = [
            f"def {function_name}({param_signature}):",
            f'    """{docstring}"""',
            "",
            "    # Validate required parameters",
            f"    required_params = {list(required_params)}",
            "    missing_params = []",
            "    locals_dict = locals()",
            "",
            "    for param in required_params:",
            "        if param not in locals_dict or locals_dict[param] is None:",
            "            missing_params.append(param)",
            "",
            "    if missing_params:",
            '        raise McpdError(f"Missing required parameters: {missing_params}")',
            "",
            "    # Build parameters dictionary",
            "    params = {}",
            "    locals_dict = locals()",
            "",
            f"    for param_name in {list(properties.keys())}:",
            "        if param_name in locals_dict and locals_dict[param_name] is not None:",
            "            params[param_name] = locals_dict[param_name]",
            "",
            "    # Make the API call",
            f'    return client._perform_call("{server_name}", "{schema["name"]}", params)',
        ]

        return "\n".join(function_lines)

    def _create_annotations(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Create type annotations for the function."""
        annotations = {}
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        for param_name, param_info in properties.items():
            is_required = param_name in required_params
            param_type = TypeConverter.parse_schema_type(param_info)

            if is_required:
                annotations[param_name] = param_type
            else:
                annotations[param_name] = param_type | None

        annotations["return"] = Any
        return annotations

    def _create_docstring(self, schema: dict[str, Any]) -> str:
        """Create a docstring for the function."""
        description = schema.get("description", "No description provided")
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        docstring_parts = [description]

        if properties:
            docstring_parts.append("")
            docstring_parts.append("Args:")

            for param_name, param_info in properties.items():
                is_required = param_name in required_params
                param_desc = param_info.get("description", "No description provided")
                required_text = "" if is_required else " (optional)"
                docstring_parts.append(f"    {param_name}: {param_desc}{required_text}")

        docstring_parts.extend(
            [
                "",
                "Returns:",
                "    Any: Function execution result",
                "",
                "Raises:",
                "    McpdError: If required parameters are missing or API call fails",
            ]
        )

        return "\n".join(docstring_parts)

    def _create_namespace(self) -> dict[str, Any]:
        """Create the namespace for function execution."""
        from types import NoneType
        from typing import Any, Literal, Union

        return {
            "McpdError": McpdError,
            "client": self.client,
            "Any": Any,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "Literal": Literal,
            "Union": Union,
            "NoneType": NoneType,
        }

    def clear_cache(self):
        """Clear the function cache."""
        self._function_cache.clear()
