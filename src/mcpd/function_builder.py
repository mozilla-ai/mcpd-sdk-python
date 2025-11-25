"""Function generation from MCP tool schemas.

This module provides the FunctionBuilder class that dynamically generates
callable Python functions from MCP tool JSON Schema definitions. These
functions can be used with AI agent frameworks and include proper parameter
validation, type annotations, and comprehensive docstrings.

The generated functions are self-contained and cached for performance.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .exceptions import McpdError, ValidationError
from .type_converter import TypeConverter

if TYPE_CHECKING:
    from .mcpd_client import McpdClient

TOOL_SEPARATOR = "__"
"""Separator used between server name and tool name in qualified tool names.

Format: `{serverName}{TOOL_SEPARATOR}{toolName}`
Example: "time__get_current_time" where "time" is server and "get_current_time" is tool.
"""


class FunctionBuilder:
    """Builds callable Python functions from MCP tool JSON schemas.

    This class generates self-contained functions that can be used with AI agent
    frameworks. It uses dynamic string compilation to create functions with proper
    parameter validation, type annotations, and docstrings based on the tool's
    JSON Schema definition.

    The generated functions are cached for performance, with cache invalidation
    controlled by the owning McpdClient via clear_cache().

    Attributes:
        _client: Reference to the McpdClient instance for tool execution.
        _function_cache: Cache of compiled function templates and metadata.

    Example:
        This class is typically used internally by McpdClient.agent_tools():

        >>> # Internal usage (not typical user code):
        >>> builder = FunctionBuilder(client)
        >>> schema = {"name": "get_time", "inputSchema": {...}}
        >>> func = builder.create_function_from_schema(schema, "time_server")
        >>> result = func(timezone="UTC")  # Executes the MCP tool
    """

    def __init__(self, client: McpdClient):
        """Initialize a FunctionBuilder for the given client.

        Args:
            client: The McpdClient instance that will be used to execute
                   the generated functions via _perform_call().
        """
        self._client = client
        self._function_cache = {}

    def _safe_name(self, name: str) -> str:
        """Convert a string into a safe Python identifier.

        This method sanitizes arbitrary strings (like server names or tool names) to create
        valid Python identifiers that can be used as function names or variable names.
        It replaces non-word characters and handles edge cases like leading digits.

        Args:
            name: The string to convert into a safe identifier. Can contain any characters.

        Returns:
            A string that is a valid Python identifier:
            - Contains only letters, digits, and underscores
            - Does not start with a digit
            - Non-word characters are replaced with underscores

        Example:
            >>> builder._safe_name("my-server")
            'my_server'
            >>> builder._safe_name("123tool")
            '_123tool'
            >>> builder._safe_name("special@chars!")
            'special_chars_'
            >>> builder._safe_name("valid_name")
            'valid_name'
        """
        return re.sub(r"\W|^(?=\d)", "_", name)  # replace non‑word chars, leading digit

    def _function_name(self, server_name: str, schema_name: str) -> str:
        """Generate a unique function name from server and tool names.

        This method creates a qualified function name by combining the server name
        and tool name with TOOL_SEPARATOR. Both names are sanitized using _safe_name()
        to ensure the result is a valid Python identifier.

        The separator helps distinguish the server and tool components while maintaining
        uniqueness across the entire function namespace.

        Args:
            server_name: The name of the MCP server hosting the tool.
            schema_name: The name of the tool from the schema definition.

        Returns:
            A qualified function name in the format "{safe_server}{TOOL_SEPARATOR}{safe_tool}".
            The result is guaranteed to be a valid Python identifier.

        Example:
            >>> builder._function_name("time-server", "get_current_time")
            'time_server__get_current_time'
            >>> builder._function_name("my@server", "tool-123")
            'my_server__tool_123'
            >>> builder._function_name("simple", "tool")
            'simple__tool'

        Note:
            This naming convention allows the generated function to be introspected
            to determine its originating server and tool names by splitting on TOOL_SEPARATOR.
        """
        return f"{self._safe_name(server_name)}{TOOL_SEPARATOR}{self._safe_name(schema_name)}"

    def create_function_from_schema(self, schema: dict[str, Any], server_name: str) -> Callable[..., Any]:
        """Create a callable Python function from an MCP tool's JSON Schema definition.

        This method generates a self-contained, callable function that validates parameters
        and executes the corresponding MCP tool. The function is dynamically compiled from
        a string template and includes proper type annotations, docstrings, and validation
        logic based on the tool's JSON Schema.

        Generated functions are cached for performance. If a function for the same
        server/tool combination already exists in the cache, it returns a new instance
        of the cached function template rather than recompiling.

        Args:
            schema: The MCP tool's JSON Schema definition, containing:
                   - 'name': Tool identifier (required)
                   - 'description': Human-readable description (optional)
                   - 'inputSchema': JSON Schema for parameters (optional)
            server_name: The name of the MCP server hosting this tool.

        Returns:
            A callable Python function with the following characteristics:
            - Parameter signature matches the tool's inputSchema
            - Required parameters first, then optional parameters with defaults
            - Type annotations based on JSON Schema types
            - Comprehensive docstring with parameter descriptions
            - Raises ValidationError for missing required parameters
            - Returns the tool's execution result via client._perform_call()
            - Has _server_name attribute containing the originating server name
            - Has _tool_name attribute containing the original tool name

        Raises:
            McpdError: If function compilation fails due to invalid schema,
                      malformed tool definition, or code generation errors.
                      The original exception is preserved via exception chaining.

        Example:
            >>> schema = {
            ...     "name": "get_current_time",
            ...     "description": "Get the current time in a timezone",
            ...     "inputSchema": {
            ...         "type": "object",
            ...         "properties": {
            ...             "timezone": {"type": "string", "description": "IANA timezone"}
            ...         },
            ...         "required": ["timezone"]
            ...     }
            ... }
            >>> func = builder.create_function_from_schema(schema, "time_server")
            >>> result = func(timezone="UTC")  # Calls the MCP tool

        Note:
            The generated function includes validation logic that checks for required
            parameters at runtime and builds a parameters dictionary for the API call.
            The function is cached using a key of "{server_name}{TOOL_SEPARATOR}{tool_name}".
        """
        cache_key = f"{server_name}{TOOL_SEPARATOR}{schema.get('name', '')}"

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
            function_name = self._function_name(server_name, schema["name"])
            created_function = namespace[function_name]
            created_function.__annotations__ = annotations

            # Add metadata attributes.
            created_function._server_name = server_name
            created_function._tool_name = schema["name"]

            # Cache the function creation details
            def create_function_instance(annotations: dict[str, Any]) -> Callable[..., Any]:
                temp_namespace = namespace.copy()
                exec(compiled_code, temp_namespace)
                new_func = temp_namespace[function_name]
                new_func.__annotations__ = annotations.copy()
                # Add metadata attributes to cached instances as well.
                new_func._server_name = server_name
                new_func._tool_name = schema["name"]
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
        """Generate Python function source code from an MCP tool's JSON Schema.

        This method is the core of the dynamic function generation system. It creates
        a complete Python function as a string that includes parameter validation,
        proper signature ordering, docstring generation, and API call execution.

        The generated function includes:
        - Required parameters first, then optional parameters with None defaults
        - Runtime validation that raises ValidationError for missing required params
        - Parameter dictionary building that excludes None values
        - Direct call to client._perform_call() with the tool's server and name

        Args:
            schema: The MCP tool's JSON Schema definition containing:
                   - 'name': Tool identifier (required)
                   - 'description': Tool description (optional)
                   - 'inputSchema': Parameter schema with 'properties' and 'required' (optional)
            server_name: The name of the MCP server hosting this tool.

        Returns:
            Complete Python function source code as a string, ready for compilation
            with compile() and execution with exec(). The function will be named
            using the _function_name() convention.

        Example:
            Given a schema like:
            ```python
            {
                "name": "get_time",
                "description": "Get current time",
                "inputSchema": {
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"]
                }
            }
            ```

            This method generates code equivalent to:
            ```python
            def server__get_time(timezone):
                '''Get current time

                Args:
                    timezone: No description provided

                Returns:
                    Any: Function execution result

                Raises:
                    ValidationError: If required parameters are missing
                    McpdError: If the API call fails
                '''
                # Validate required parameters
                required_params = ['timezone']
                missing_params = []
                locals_dict = locals()

                for param in required_params:
                    if param not in locals_dict or locals_dict[param] is None:
                        missing_params.append(param)

                if missing_params:
                    raise ValidationError(
                        f"Missing required parameters: {missing_params}",
                        validation_errors=missing_params,
                    )

                # Build parameters dictionary
                params = {}
                locals_dict = locals()

                for param_name in ['timezone']:
                    if param_name in locals_dict and locals_dict[param_name] is not None:
                        params[param_name] = locals_dict[param_name]

                # Make the API call
                return client._perform_call("server", "get_time", params)
            ```

        Note:
            The generated code uses string interpolation and list literals to embed
            the schema data directly into the function code. This creates a completely
            self-contained function that doesn't depend on the original schema object.
        """  # noqa: D214
        function_name = self._function_name(server_name, schema["name"])
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        # Sort parameters: required first, then optional
        required_param_names = [p for p in properties if p in required_params]
        optional_param_names = [p for p in properties if p not in required_params]
        sorted_param_names = required_param_names + optional_param_names
        param_declarations = []

        # Build parameter signature
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
            "        raise ValidationError(",
            '            f"Missing required parameters: {missing_params}",',
            "            validation_errors=missing_params,",
            "        )",
            "",
            "    # Build parameters dictionary",
            "    params = {}",
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
        """Generate Python type annotations from a tool's JSON Schema.

        This method converts JSON Schema type definitions into Python type hints
        that are attached to the generated function. It uses the TypeConverter
        utility to handle complex schema types and properly marks optional
        parameters with modern union syntax (type | None).

        The method processes each parameter in the schema's properties, determines
        if it's required, and creates appropriate type annotations. Required
        parameters get direct type annotations while optional parameters are
        automatically unioned with None.

        Args:
            schema: The MCP tool's JSON Schema definition containing:
                   - 'inputSchema': Object with 'properties' and 'required' arrays
                   - Properties define parameter names and their JSON Schema types
                   - Required array lists which parameters are mandatory

        Returns:
            A dictionary mapping parameter names to Python type objects, plus
            a 'return' key mapped to Any. This dictionary is directly assignable
            to a function's __annotations__ attribute.

        Example:
            Given a schema like:
            ```python
            {
                "inputSchema": {
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "active": {"type": "boolean"}
                    },
                    "required": ["name"]
                }
            }
            ```

            Returns annotations equivalent to:
            ```python
            {
                "name": str,                    # Required parameter
                "age": int | None,              # Optional parameter
                "active": bool | None,          # Optional parameter
                "return": Any                   # Return type
            }
            ```

        Note:
            - Uses TypeConverter.parse_schema_type() for complex type parsing
            - Always adds 'return': Any since MCP tool responses are untyped
            - Optional parameters use Python 3.10+ union syntax (type | None)
            - The annotations are used for IDE support and runtime introspection
        """
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
        """Generate a comprehensive docstring for the dynamically created function.

        This method builds a properly formatted Python docstring that includes the
        tool's description, parameter documentation with optional/required status,
        return value information, and exception documentation. The generated docstring
        follows Google/NumPy style conventions for consistency.

        The docstring is embedded directly into the generated function code and
        provides runtime documentation accessible via help() or __doc__.

        Args:
            schema: The MCP tool's JSON Schema definition containing:
                   - 'description': Human-readable tool description (optional)
                   - 'inputSchema': Schema with 'properties' and 'required' arrays
                   - Properties include parameter descriptions

        Returns:
            A multi-line string containing the complete docstring text, properly
            formatted with sections for description, arguments, returns, and raises.
            The string is ready to be embedded in triple quotes in the generated code.

        Example:
            Given a schema like:
            ```python
            {
                "description": "Search for items in database",
                "inputSchema": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
            ```

            Generates a docstring like:
            ```Search for items in database

            Args:
                query: Search query string
                limit: Maximum results to return (optional)

            Returns:
                Any: Function execution result

            Raises:
                ValidationError: If required parameters are missing
                McpdError: If the API call fails
            ```

        Note:
            - Parameters without descriptions get "No description provided"
            - Optional parameters are marked with "(optional)" suffix
            - The Raises section accurately documents both validation and execution errors
            - Empty properties result in a docstring without an Args section
        """  # noqa: D214
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
                "    ValidationError: If required parameters are missing",
                "    McpdError: If the API call fails",
            ]
        )

        return "\n".join(docstring_parts)

    def _create_namespace(self) -> dict[str, Any]:
        """Create the execution namespace for dynamically generated functions.

        This method builds a dictionary containing all the Python built-ins, types,
        and references that the generated function code needs at runtime. The namespace
        is used as the global scope when executing the compiled function code via exec().

        The namespace includes:
        - Exception classes for error handling (McpdError, ValidationError)
        - Reference to the client instance for making API calls
        - Python built-in types (str, int, float, bool, list, dict)
        - Type annotation utilities (Any, Literal, Union, NoneType)

        This ensures the generated function has access to everything it needs without
        relying on module imports or the surrounding scope.

        Returns:
            A dictionary mapping names to Python objects, suitable for use as the
            globals parameter in exec(). This namespace makes the generated function
            completely self-contained.

        Example:
            The namespace allows generated code like this to work:
            ```python
            def server__tool(param: str = None):
                # Uses ValidationError from namespace
                if missing_params:
                    raise ValidationError(...)

                # Uses 'client' from namespace
                return client._perform_call(...)
            ```

            Without the namespace, the function would fail with NameError
            when trying to access ValidationError or client.

        Note:
            - The client reference is captured at FunctionBuilder creation time
            - All standard Python types are included to support type annotations
            - NoneType is imported locally to avoid top-level import issues
            - The namespace is copied for each function instance to ensure isolation
        """
        from types import NoneType
        from typing import Any, Literal, Union

        return {
            "McpdError": McpdError,
            "ValidationError": ValidationError,
            "client": self._client,
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

    def clear_cache(self) -> None:
        """Clear the internal function compilation cache.

        This method removes all cached function templates created by previous calls
        to create_function_from_schema(). After clearing, subsequent calls to
        create_function_from_schema() will recompile functions from their JSON
        schemas rather than using cached templates.

        Use this method when:
        - MCP server tool definitions have changed
        - You want to force regeneration of function wrappers
        - Memory usage from cached functions becomes a concern

        The cache is automatically populated as functions are generated, so there's
        no need to explicitly populate it after clearing.

        Returns:
            None

        Example:
            >>> builder = FunctionBuilder(client)
            >>> func1 = builder.create_function_from_schema(schema, "server1")  # Compiles and caches
            >>> func2 = builder.create_function_from_schema(schema, "server1")  # Uses cache
            >>>
            >>> builder.clear_cache()
            >>> func3 = builder.create_function_from_schema(schema, "server1")  # Recompiles
        """
        self._function_cache.clear()

    def get_cached_functions(self) -> list[Callable[..., Any]]:
        """Get all cached functions.

        Returns:
            List of all cached agent functions, or empty list if cache is empty.
        """
        return [cached["create_function"](cached["annotations"]) for cached in self._function_cache.values()]
