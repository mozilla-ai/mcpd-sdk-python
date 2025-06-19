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

## Installation

Assuming you are using [uv](https://github.com/astral-sh/uv), include it in your `pyproject.toml`:

```toml
[tool.uv.sources]
mcpd_sdk = { path = "../mcpd-sdk", editable = true }

[project]
dependencies = [
  "mcpd_sdk"
]
```

Then run:

```bash
uv sync
```

## Quick Start

```python
from mcpd_sdk import McpdClient, McpdError

client = McpdClient(endpoint="http://localhost:8090")

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
from mcpd_sdk import McpdClient

# Assumes the mcpd daemon is running
client = McpdClient(endpoint="http://localhost:8090")

agent_config = AgentConfig(
    tools=client.agent_tools(),
    model_id="gpt-4.1-nano", # Requires OPENAI_API_KEY to be set
    instructions="Use the tools to answer the user's question."
)
agent = AnyAgent.create("mcpd-agent", agent_config)

response = agent.run("What is the current time in Tokyo?")
print(response)
```

## Examples

A working example using any-agent and this SDK is available in the `example/` folder.
Refer to [example/README.md](example/README.md) for setup and execution details.

## API

### Initialization

```python
from mcpd_sdk import McpdClient

# Initialize the client with your mcpd API endpoint.
# api_key is optional and sends an 'MCPD-API-KEY' header.
client = McpdClient(endpoint="http://localhost:8090", api_key="optional-key")
```

### Core Methods

* `client.servers() -> list[str]` - Returns a list of all configured server names.

* `client.tools() -> dict[str, list[dict]]` - Returns a dictionary mapping each server name to a list of its tool schema definitions.

* `client.tools(server_name: str) -> list[dict]` - Returns the tool schema definitions for only the specified server.

* `client.agent_tools() -> list[Callable]` - Returns a list of self-contained, callable functions suitable for agentic frameworks.

* `client.has_tool(server_name: str, tool_name: str) -> bool` - Checks if a specific tool exists on a given server.

* `client.call.<server_name>.<tool_name>(**kwargs)` - The primary way to dynamically call any tool using keyword arguments.

## Error Handling

All SDK-level errors, including HTTP and connection errors, will raise a `McpdError` exception.
The original exception is chained for full context.


## License

Apache-2.0
