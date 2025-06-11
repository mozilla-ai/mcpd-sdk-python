import json
import os

import requests
from any_agent import AgentConfig, AnyAgent
from mcpd_sdk import McpdClient, McpdError

if __name__ == "__main__":
    mcpd_endpoint = os.getenv("MCPD_ADDR", "http://localhost:8090")
    mcpd_api_key = os.getenv("MCPD_API_KEY")  # NOTE: Not used at present.

    try:
        mcpd_client = McpdClient(endpoint=mcpd_endpoint, api_key=mcpd_api_key)

        # 1. List Servers
        print("\n--- Listing MCP Servers ---", flush=True)
        servers = mcpd_client.servers()
        print(f"Found {len(servers)} servers: {servers}", flush=True)

        # 2. Get tool *definitions* for a specific server
        if "time" in servers:
            print("\n--- Listing Tool Definitions for 'time' ---", flush=True)
            time_tool_defs = mcpd_client.tools("time")
            print("Found definitions for 'time':, flush=True")
            print(json.dumps(time_tool_defs, indent=2), flush=True)

        # 3. Use the dynamic caller
        if "time" in servers:
            print("\n--- Calling Tool via Dynamic Caller ---", flush=True)
            result = mcpd_client.tool_caller.time.get_current_time(timezone="UTC")
            print("Result:", flush=True)
            print(json.dumps(result, indent=2), flush=True)

        # 4. Get all tools as callable functions
        print("\n--- Getting All Tools as Callable Functions ---", flush=True)
        all_callable_tools = mcpd_client.tools()
        print(f"Discovered {len(all_callable_tools)} callable tools in total.", flush=True)

        # 5. Get all tool *definitions* for an agentic framework
        print("\n--- Getting All Tool Definitions for an Agent ---", flush=True)
        all_tool_definitions = mcpd_client.get_all_tool_definitions()
        print(f"Discovered {len(all_tool_definitions)} tool definitions in total.", flush=True)

        if all_tool_definitions:
            print("Example of a combined tool definition:", flush=True)
            print(json.dumps(all_tool_definitions[0], indent=2), flush=True)

        # 6. Use tools with any-agent
        print("\n--- Initializing any-agent (Example) ---", flush=True)
        agent_config = AgentConfig(
            tools=mcpd_client.as_any_agent_tools(),
            model_id="gpt-4.1-nano",
            instructions="Use the tools to find an answer",
        )
        agent = AnyAgent.create("tinyagent", agent_config)

        agent_trace = agent.run("What time is it in Tokyo?")
        print(agent_trace, flush=True)

    except McpdError as e:
        print(f"\n[SDK ERROR] An error occurred: {e}", flush=True)
    except requests.exceptions.ConnectionError:
        print(f"\n[CONNECTION ERROR] Could not connect to the mcpd daemon at {mcpd_endpoint}", flush=True)
        print("Please ensure the mcpd application is running with the 'mcpd daemon' command.", flush=True)
