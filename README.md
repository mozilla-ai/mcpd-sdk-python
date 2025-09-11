# mcpd-sdk-python

`mcpd-sdk-python` is a lightweight Python SDK for interacting with the [mcpd](https://github.com/mozilla-ai/mcpd) application.

A daemon that exposes MCP server tools via a simple HTTP API.

This SDK provides high-level and dynamic access to those tools, making it easy to integrate with scripts, applications, or agentic frameworks.

## Features

- Discover and list available `mcpd` hosted MCP servers
- Retrieve tool definitions and schemas for one or all servers
- Dynamically invoke any tool using a clean, attribute-based syntax
- Generate self-contained, deepcopy-safe tool functions for frameworks like [any-agent](https://github.com/mozilla-ai/any-agent)
- Minimal dependencies (`requests` only)

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

agent_config = AgentConfig(
    tools=client.agent_tools(),
    model_id="gpt-4.1-nano",  # Requires OPENAI_API_KEY to be set
    instructions="Use the tools to answer the user's question."
)
agent = AnyAgent.create("mcpd-agent", agent_config)

response = agent.run("What is the current time in Tokyo?")
print(response)
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
client = McpdClient(api_endpoint="http://localhost:8090", api_key="optional-key")
```

### Core Methods

* `client.servers() -> list[str]` - Returns a list of all configured server names.

* `client.tools() -> dict[str, list[dict]]` - Returns a dictionary mapping each server name to a list of its tool schema definitions.

* `client.tools(server_name: str) -> list[dict]` - Returns the tool schema definitions for only the specified server.

* `client.agent_tools() -> list[Callable]` - Returns a list of self-contained, callable functions suitable for agentic frameworks.

* `client.clear_agent_tools_cache()` - Clears cached generated callable functions that are created when calling agent_tools().

* `client.has_tool(server_name: str, tool_name: str) -> bool` - Checks if a specific tool exists on a given server.

* `client.call.<server_name>.<tool_name>(**kwargs)` - The primary way to dynamically call any tool using keyword arguments.

## Error Handling

All SDK-level errors, including HTTP and connection errors, will raise a `McpdError` exception.
The original exception is chained for full context.


## License

Apache-2.0
