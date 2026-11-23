"""Function generation from MCP tool schemas.

This module provides the FunctionBuilder class that generates callable Python
functions from MCP tool JSON Schema definitions. These functions can be used
with AI agent frameworks and include proper parameter validation, type
annotations, and comprehensive docstrings.

The generated functions are cached for performance.
"""

from __future__ import annotations

import inspect
import keyword
import re
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _ToolFunction:
    """The parts of a generated tool function that do not depend on the call."""

    server_name: str
    tool_name: str
    function_name: str
    signature: inspect.Signature
    annotations: dict[str, Any]
    docstring: str


class FunctionBuilder:
    """Builds callable Python functions from MCP tool JSON schemas.

    This class generates functions that can be used with AI agent frameworks, with
    parameter validation, type annotations, and docstrings taken from the tool's
    JSON Schema definition.

    NOTE: A tool's name, description and parameter names are server-supplied and
    therefore untrusted. They must never reach an evaluator.

    A tool's schema is read once and the result cached, so each request for that
    tool returns a new callable without re-reading the schema. Call clear_cache()
    to discard the cache.

    Attributes:
        _client: Reference to the McpdClient instance for tool execution.
        _function_cache: What each tool's schema describes, keyed by server and tool.

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
        self._function_cache: dict[str, _ToolFunction] = {}

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

        This method generates a callable function that validates parameters and executes
        the corresponding MCP tool, with type annotations, a docstring, and validation
        logic taken from the tool's JSON Schema.

        What the schema describes is cached per server and tool. A repeat request for
        the same tool returns a new callable built from that cached description.

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
            McpdError: If the function cannot be built from the schema, for example
                      when a parameter name is unusable or the tool definition is
                      malformed. The original exception is preserved via exception
                      chaining.

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
        if not isinstance(schema, dict):
            raise ValidationError("Tool definition is not an object")

        cache_key = f"{server_name}{TOOL_SEPARATOR}{schema.get('name', '')}"

        if cache_key in self._function_cache:
            return self._build_function(self._function_cache[cache_key])

        try:
            annotations = self._create_annotations(schema)
            tool = _ToolFunction(
                server_name=server_name,
                tool_name=schema["name"],
                function_name=self._function_name(server_name, schema["name"]),
                signature=self._build_signature(schema, annotations),
                annotations=annotations,
                docstring=self._create_docstring(schema),
            )
        except McpdError:
            raise
        except Exception as e:
            raise McpdError(f"Error creating function {cache_key!r}: {e}") from e

        self._function_cache[cache_key] = tool

        return self._build_function(tool)

    def _build_signature(self, schema: dict[str, Any], annotations: dict[str, Any]) -> inspect.Signature:
        """Build the call signature that a tool's JSON Schema describes.

        Required parameters precede optional ones, so the result is a usable
        signature even when the schema lists them in another order.

        Args:
            schema: The tool's JSON Schema definition.
            annotations: Parameter type annotations keyed by parameter name.

        Returns:
            The signature for the tool's generated function.

        Raises:
            ValidationError: If a property name cannot be used as a parameter name,
                or if a required parameter has no property describing it.
        """
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        undescribed_params = sorted(required_params - set(properties))
        if undescribed_params:
            raise ValidationError(
                f"Tool requires parameters it does not describe: {undescribed_params}",
                validation_errors=undescribed_params,
            )

        ordered_names = [name for name in properties if name in required_params]
        ordered_names += [name for name in properties if name not in required_params]

        parameters = []
        for param_name in ordered_names:
            if not param_name.isidentifier() or keyword.iskeyword(param_name):
                raise ValidationError(
                    f"Tool parameter {param_name!r} cannot be used as a parameter name",
                    validation_errors=[param_name],
                )

            is_required = param_name in required_params
            parameters.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if is_required else None,
                    annotation=annotations[param_name],
                )
            )

        return inspect.Signature(parameters, return_annotation=Any)

    def _build_function(self, tool: _ToolFunction) -> Callable[..., Any]:
        """Create a callable that validates its arguments and executes the tool.

        Args:
            tool: The generated function's identity, signature and documentation.

        Returns:
            A callable reporting `tool`'s signature, annotations and docstring
            under introspection.

        NOTE: The tool's name, description and parameter names are server-supplied
        data. They are attached to the callable as values and must never be
        rendered into executable source.

        NOTE: The returned callable accepts `*args, **kwargs`, so its code object
        does not describe the tool's parameters. A caller that reads
        `__code__.co_varnames` or `co_argcount` rather than `inspect.signature`
        should not be given this callable.
        """
        client = self._client

        def tool_function(*args: Any, **kwargs: Any) -> Any:
            bound = tool.signature.bind(*args, **kwargs)
            bound.apply_defaults()

            missing_params = [
                name
                for name, param in tool.signature.parameters.items()
                if param.default is inspect.Parameter.empty and bound.arguments.get(name) is None
            ]

            if missing_params:
                raise ValidationError(
                    f"Missing required parameters: {missing_params}",
                    validation_errors=missing_params,
                )

            params = {name: value for name, value in bound.arguments.items() if value is not None}

            return client._perform_call(tool.server_name, tool.tool_name, params)

        tool_function.__name__ = tool.function_name
        tool_function.__qualname__ = tool.function_name
        tool_function.__doc__ = tool.docstring
        tool_function.__signature__ = tool.signature
        tool_function.__annotations__ = dict(tool.annotations)
        tool_function._server_name = tool.server_name
        tool_function._tool_name = tool.tool_name

        return tool_function

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

    def clear_cache(self) -> None:
        """Clear the internal function cache.

        This method removes all cached functions created by previous calls to
        create_function_from_schema(). After clearing, subsequent calls to
        create_function_from_schema() will rebuild functions from their JSON
        schemas rather than using the cache.

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
            >>> func1 = builder.create_function_from_schema(schema, "server1")  # Builds and caches
            >>> func2 = builder.create_function_from_schema(schema, "server1")  # Uses cache
            >>>
            >>> builder.clear_cache()
            >>> func3 = builder.create_function_from_schema(schema, "server1")  # Rebuilds
        """
        self._function_cache.clear()
