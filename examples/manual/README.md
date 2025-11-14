# Example App - `mcpd` Python SDK

This sample application demonstrates how to run `mcpd` in daemon mode to start MCP servers and then use our
application code to explicitly call tools on those MCP servers.

## Requirements

* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [mcpd](https://mozilla-ai.github.io/mcpd/installation/) installed
* `OPENAI_API_KEY` exported - this will be used by [any-agent](https://github.com/mozilla-ai/any-agent)

## Starting `mcpd`

### Execution context config file

`~/.config/mcpd/secrets.dev.toml` is the file that is used to provide user specific configuration to MCP servers via `mcpd`.

Here is an example of some custom configuration for the `mcp-server-time` (time) server:

```toml
[servers]
  [servers.time]
    args = ["--local-timezone=Europe/London"]
```

Run the following command to create this file if you don't want the time MCP Server to use defaults:

```bash
mcpd config args set time -- --local-timezone=Europe/London
```

### Project configuration file

The `.mcpd.toml` in this folder, is used alongside the following command to start specific versions of MCP servers:

```bash
mcpd daemon --log-level=DEBUG --log-path=$(pwd)/mcpd.log
```

We do this outside of code, and use the HTTP address given to us by `mcpd` to configure the SDK.

The `mcpd` daemon will start the servers, emitting messages to the terminal, but you can tail the log to see more info:

```bash
tail -f mcpd.log
```

## Running our agentic app

To run our application which will showcase some of the commands available via the SDK, and then how to wire up tools easily
to any-agent:

```bash
uv venv
source .venv/bin/activate
uv sync --group all
uv run python -m main
```
