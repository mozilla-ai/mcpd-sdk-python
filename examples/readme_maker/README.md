# Pydantic AI README Maker

This agent is built using the X framework and leverages GitHub's MCP server to automatically generate high-quality READMEs for your repositories.

## Key Features

* Generate READMEs based on a template repository
* Automatically open a pull request with the generated README on your target repository

## Getting Started

### Step 1: Set Up Your Environment
```bash
uv venv
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
uv sync --group all
```

### Step 3: Initialize a new MCPD project and add the GitHub MCP server
```bash
mcpd init
mcpd add github
```

### Step 4: Configure your agent's API key and the GitHub token for the MCP server
```bash
export OPENAI_API_KEY=your_openai_api_key # Required by the Agent
mcpd config env set github GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token # Required by the GitHub MCP server
```

### Step 5: Start the MCPD server
```bash
mcpd daemon
```

### Step 6: In another terminal, run the agent
```bash
uv run readme_maker.py
```

### (Optional): Run mcpd daemon in debug mode to see the server logs:
```bash
mcpd daemon --log-level=DEBUG --log-path=$(pwd)/mcpd.log
tail -f mcpd.log # Run in another terminal window
```
