# mcpd-sdk-python

`mcpd-sdk-python` is a lightweight Python SDK for interacting with the [mcpd](https://github.com/mozilla-ai/mcpd) application.

A daemon that exposes MCP server tools via a simple HTTP API.

This SDK provides high-level and dynamic access to those tools, making it easy to integrate with scripts, applications, or agentic frameworks.

## Features

- Discover and list available `mcpd` hosted MCP servers
- Retrieve tool definitions and schemas for one or all servers
- Dynamically invoke any tool using a clean, attribute-based syntax
- Generate self-contained, deepcopy-safe tool functions for frameworks like [any-agent](https://github.com/mozilla-ai/any-agent)
- Minimal dependencies (`requests` and `cachetools` only)

## Installation in your project

Assuming you are using [uv](https://github.com/astral-sh/uv), include it in your `pyproject.toml`:

```bash
uv add mcpd
```

## Dev Setup

Use the `Makefile` target to ensure `uv` is installed, and your virtual environment is active and sync'd.

```bash
make setup
```

## Testing

Ensure you have the correct dependencies installed for testing:

```bash
uv sync --group tests
```

Then to run all tests:

```bash
uv run pytest tests
```

... or via `Makefile`:

```bash
make test
```

Lint files using:

```bash
make lint
```

## Quick Start

```python
from mcpd import McpdClient, McpdError

client = McpdClient(api_endpoint="http://localhost:8090")

# List available servers
print(client.servers())
# Example: ['time', 'fetch', 'git']

# List tool definitions (schemas) for a specific server
print(client.tools(server_name="time"))

# Dynamically call a tool
try:
    result = client.call.time.get_current_time(timezone="UTC")
    print(result)
except McpdError as e:
    print(f"Error: {e}")

```

## Agentic Usage

Generate dynamic functions suitable for AI agents:

```python
from any_agent import AnyAgent, AgentConfig
from mcpd import McpdClient

# Assumes the mcpd daemon is running
client = McpdClient(api_endpoint="http://localhost:8090")

# Get all tools from healthy servers (default - filters out unhealthy servers)
all_tools = client.agent_tools()

# Get tools from specific servers, only if healthy
time_tools = client.agent_tools(servers=['time'])

# Get tools from multiple servers, only if healthy
subset_tools = client.agent_tools(servers=['time', 'fetch'])

# Filter by tool names (cross-cutting)
math_tools = client.agent_tools(tools=['add', 'multiply'])

# Filter by qualified tool names
specific = client.agent_tools(tools=['time__get_current_time'])

# Combine server and tool filtering
filtered = client.agent_tools(
    servers=['time', 'math'],
    tools=['add', 'get_current_time']
)

agent_config = AgentConfig(
    tools=client.agent_tools(),
    model_id="gpt-4.1-nano",  # Requires OPENAI_API_KEY to be set
    instructions="Use the tools to answer the user's question."
)
agent = AnyAgent.create("mcpd-agent", agent_config)

response = agent.run("What is the current time in Tokyo?")
print(response)
```

> [!IMPORTANT]
> Generated functions are cached for performance. Once cached, subsequent calls to `agent_tools()` return
> the cached functions immediately without refetching schemas, regardless of filter parameters.
> Use `refresh_cache=True` or call `client.clear_agent_tools_cache()` to force regeneration when tool schemas have changed.

```python
# Force refresh cache to get latest schemas
fresh_tools = client.agent_tools(refresh_cache=True)

# Or clear cache manually and call again
client.clear_agent_tools_cache()
fresh_tools = client.agent_tools()
```

## Examples

A working SDK examples are available in the `examples/` folder,
please refer to the relevant example for execution details.

| Method      | Docs                                        |
|-------------|---------------------------------------------|
| AnyAgent    | [README.md](examples/anyagent/README.md)    |
| Manual      | [README.md](examples/manual/README.md)      |
| Pydantic AI | [README.md](examples/pydantic-ai/README.md) |

## API

### Initialization

```python
from mcpd import McpdClient

# Initialize the client with your mcpd API endpoint.
# api_key is optional and sends an 'MCPD-API-KEY' header.
# server_health_cache_ttl is optional and sets the time in seconds to cache a server health response.
# logger is optional and allows you to provide a custom logger implementation (see Logging section).
client = McpdClient(api_endpoint="http://localhost:8090", api_key="optional-key", server_health_cache_ttl=10)
```

### Core Methods

* `client.servers() -> list[str]` - Returns a list of all configured server names.

* `client.tools() -> dict[str, list[dict]]` - Returns a dictionary mapping each server name to a list of its tool schema definitions.

* `client.tools(server_name: str) -> list[dict]` - Returns the tool schema definitions for only the specified server.

* `client.agent_tools(servers: list[str] | None = None, tools: list[str] | None = None, *, refresh_cache: bool = False) -> list[Callable]` - Returns a list of self-contained, callable functions suitable for agentic frameworks. By default, filters to healthy servers only. Use `servers` to filter by server names, `tools` to filter by tool names (supports both raw names like `'add'` and prefixed names like `'time__get_current_time'`), or `refresh_cache=True` to force regeneration of cached functions. Functions are cached - subsequent calls return cached functions immediately without refetching schemas.

* `client.clear_agent_tools_cache()` - Clears cached generated callable functions. Call this to force regeneration when tool schemas have changed.

* `client.has_tool(server_name: str, tool_name: str) -> bool` - Checks if a specific tool exists on a given server.

* `client.call.<server_name>.<tool_name>(**kwargs)` - The primary way to dynamically call any tool using keyword arguments.

* `client.server_health() -> dict[str, dict]` - Returns a dictionary mapping each server name to the health information of that server.

* `client.server_health(server_name: str) -> dict` - Returns the health information for only the specified server.

* `client.is_server_healthy(server_name: str) -> bool` - Checks if the specified server is healthy and can handle requests.

## Logging

The SDK includes built-in logging infrastructure that can be enabled via the `MCPD_LOG_LEVEL` environment variable. Logging is disabled by default to avoid contaminating stdout/stderr.

> [!IMPORTANT]
> Only enable `MCPD_LOG_LEVEL` in non-MCP-server contexts. MCP servers can use stdout for JSON-RPC communication,
> and any logging output will break the protocol.

### Available Log Levels

Set the `MCPD_LOG_LEVEL` environment variable to one of the following values (from most to least verbose):

* `trace` - Most verbose logging (includes all levels below)
* `debug` - Debug-level logging
* `info` - Informational logging
* `warn` - Warning-level logging (recommended for most use cases)
* `error` - Error-level logging only
* `off` - Disable all logging (default)

### Example Usage

```bash
# Enable warning-level logging
export MCPD_LOG_LEVEL=warn
python your_script.py
```

```python
from mcpd import McpdClient

# Warnings will be logged to stderr when MCPD_LOG_LEVEL=warn
client = McpdClient(api_endpoint="http://localhost:8090")

# For example, the SDK will log warnings for:
# - Non-existent servers when calling agent_tools()
# - Unhealthy servers when calling agent_tools()
# - Servers that become unavailable during tool fetching
```

### Custom Logger

You can provide your own logger implementation that implements the `Logger` protocol:

```python
import sys
from mcpd import McpdClient, Logger

class CustomLogger:
    """Custom logger that writes to stderr (safe for MCP server contexts)."""

    def trace(self, msg: str, *args: object) -> None:
        print(f"TRACE: {msg % args}", file=sys.stderr)

    def debug(self, msg: str, *args: object) -> None:
        print(f"DEBUG: {msg % args}", file=sys.stderr)

    def info(self, msg: str, *args: object) -> None:
        print(f"INFO: {msg % args}", file=sys.stderr)

    def warn(self, msg: str, *args: object) -> None:
        print(f"WARN: {msg % args}", file=sys.stderr)

    def error(self, msg: str, *args: object) -> None:
        print(f"ERROR: {msg % args}", file=sys.stderr)

# Use custom logger
client = McpdClient(
    api_endpoint="http://localhost:8090",
    logger=CustomLogger()
)
```

You can also provide a partial logger implementation. Any omitted methods will fall back to the default logger (which respects `MCPD_LOG_LEVEL`):

```python
import sys

class PartialLogger:
    """Partial logger - only override warn/error, others use default."""

    def warn(self, msg: str, *args: object) -> None:
        # Custom warning handler (writes to stderr).
        print(f"CUSTOM WARN: {msg % args}", file=sys.stderr)

    def error(self, msg: str, *args: object) -> None:
        # Custom error handler (writes to stderr).
        print(f"CUSTOM ERROR: {msg % args}", file=sys.stderr)
    # trace, debug, info use default logger (respects MCPD_LOG_LEVEL)

client = McpdClient(
    api_endpoint="http://localhost:8090",
    logger=PartialLogger()
)
```

## Error Handling

All SDK-level errors raise exceptions that inherit from `McpdError`. The original exception is chained via `__cause__` for full context.

### Exception Types

| Exception              | Description                             |
|------------------------|-----------------------------------------|
| `McpdError`            | Base exception for all SDK errors       |
| `ConnectionError`      | Unable to connect to `mcpd` daemon      |
| `AuthenticationError`  | Authentication failed (invalid API key) |
| `ServerNotFoundError`  | Specified server doesn't exist          |
| `ServerUnhealthyError` | Server exists but is not healthy        |
| `ToolNotFoundError`    | Specified tool doesn't exist on server  |
| `ToolExecutionError`   | Tool execution failed                   |
| `ValidationError`      | Input validation failed                 |
| `TimeoutError`         | Operation timed out                     |
| `PipelineError`        | Required pipeline processing failed     |

### Example

```python
from mcpd import (
    McpdClient,
    McpdError,
    PipelineError,
    PIPELINE_FLOW_REQUEST,
    PIPELINE_FLOW_RESPONSE,
    ServerNotFoundError,
    ToolExecutionError,
)

client = McpdClient(api_endpoint="http://localhost:8090")

try:
    result = client.call.time.get_current_time()
except PipelineError as e:
    # A required plugin failed during request or response processing.
    if e.pipeline_flow == PIPELINE_FLOW_RESPONSE:
        print("Tool was called but results cannot be delivered")
    elif e.pipeline_flow == PIPELINE_FLOW_REQUEST:
        print("Request was rejected by pipeline")
except ServerNotFoundError as e:
    print(f"Server '{e.server_name}' not found")
except ToolExecutionError as e:
    print(f"Tool '{e.tool_name}' failed: {e}")
except McpdError as e:
    # Catch-all for any other SDK errors.
    print(f"Operation failed: {e}")
```


## License

Apache-2.0
