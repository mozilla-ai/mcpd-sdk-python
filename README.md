# mcpd-sdk-python

`mcpd-sdk-python` is a lightweight Python SDK for interacting with the [mcpd](https://github.com/mozilla-ai/mcpd-cli) application.

A daemon that exposes MCP server tools via a simple HTTP API.

This SDK provides high-level and dynamic access to those tools, making it easy to integrate with scripts, applications, or agentic frameworks.

## Features

- Discover and list available `mcpd` hosted MCP servers
- Retrieve tool definitions and schemas
- Dynamically invoke any tool using structured arguments
- Generate self-contained tool functions for frameworks like [any-agent](https://github.com/mozilla-ai/any-agent)
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
from mcpd_sdk import McpdClient

client = McpdClient(endpoint="http://localhost:8090")

# List available servers
print(client.servers())

# List tool definitions for a specific server
print(client.tools("time"))

# Dynamically call a tool
print(client.tool_caller.time.get_current_time(timezone="UTC"))
```

## Agentic Usage

Generate dynamic functions suitable for AI agents:

```python
from any_agent import AnyAgent, AgentConfig
from mcpd_sdk import McpdClient

client = McpdClient(endpoint="http://localhost:8090")

agent_config = AgentConfig(
    tools=client.as_any_agent_tools(),
    model_id="gpt-4.1-nano",
    instructions="Use the tools to answer questions."
)
agent = AnyAgent.create("mcpd-agent", agent_config)

response = agent.run("What's the current time in Tokyo?")
print(response)
```

## Examples

A working example using any-agent and this SDK is available in the `example/` folder.
Refer to [example/README.md](example/README.md) for setup and execution details.

## API

### Initialization

```python
client = McpdClient(endpoint="http://localhost:8090", api_key="optional-key")
```

### Core Methods

* `McpdClient(endpoint: str, api_key: str = None)` -
Initialize the client with your mcpd API endpoint.

* `client.servers()` - Returns a list of server names.

* `client.tools(server_name: str)` - Returns tool schema definitions for the specified server.

* `client.tools()` - Returns a list of callable functions for all discovered tools.

* `client.tool_caller.<server>.<tool>(**kwargs)` - Dynamically call any tool using keyword arguments.

* `client.call(server_name, tool_name, params={})` - Low-level direct tool invocation.

* `client.as_any_agent_tools()` - Returns a list of callable functions with introspectable signatures, safe for use with LLM frameworks.

* `client.get_all_tool_definitions()` - Returns all tool schema definitions across all servers, with names scoped by server.

## Error Handling

All SDK-level errors raise a `McpdError` exception with details and any HTTP response body attached.

## License

Apache-2.0
