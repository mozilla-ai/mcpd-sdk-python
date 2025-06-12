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
        print("\n--- Listing MCP Servers ---")
        servers = mcpd_client.servers()
        print(f"Found {len(servers)} servers: {servers}")

        # 2. Get tool *definitions for all servers
        print("\n--- Listing Tool Definitions ---")
        tool_defs = mcpd_client.tools()
        print("Found definitions:")
        print(json.dumps(tool_defs, indent=2))

        # 3. Get tool *definitions* for a specific server
        if "time" in servers:
            print("\n--- Listing Tool Definitions for 'time' ---")
            time_tool_defs = mcpd_client.tools("time")
            print("Found definitions for 'time':")
            print(json.dumps(time_tool_defs, indent=2))

        # 4. Use the dynamic caller
        if "time" in servers:
            print("\n--- Calling Tool via Dynamic Caller ---")
            print("Calling `mcpd_client.call.time.get_current_time(timezone='UTC')`...")
            result = mcpd_client.call.time.get_current_time(timezone="UTC")
            print("Result:")
            print(json.dumps(result, indent=2))

        # 5. Check if a tool exists
        print("\n--- Checking for Tool Existence ---")
        has_it = mcpd_client.has_tool("time", "get_current_time")
        print(f"Does 'time' server have 'get_current_time' tool? {has_it}")
        has_it_not = mcpd_client.has_tool("time", "non_existent_tool")
        print(f"Does 'time' server have 'non_existent_tool' tool? {has_it_not}")

        # 6. Get tools in a format compatible with any-agent
        print("\n--- Getting Tools for any-agent ---")
        agent_compatible_tools = mcpd_client.agent_tools()
        print(f"Generated {len(agent_compatible_tools)} deepcopy-safe tools for an agent.")
        if agent_compatible_tools:
            import inspect

            example_tool = agent_compatible_tools[0]
            print(f"Example tool: {example_tool.__name__}")
            print(f"Signature: {inspect.signature(example_tool)}")
            print(f"Docstring: '{example_tool.__doc__}'")

        # 7. Use any-agent with agent friendly tools
        print("\n--- Using agent_tools with any-agent (Example using `mcpd_client.agent_tools()`) ---", flush=True)
        agent_config = AgentConfig(
            tools=mcpd_client.agent_tools(),
            model_id="gpt-4.1-nano",
            instructions="Use the tools to find an answer",
        )
        agent = AnyAgent.create("tinyagent", agent_config)
        agent_trace = agent.run("What time is it in Tokyo?")
        print(agent_trace)

    except McpdError as e:
        print("\n------------------------------")
        print(f"\n[SDK ERROR] An error occurred: {e}")
        print("\n------------------------------")
    except requests.exceptions.ConnectionError:
        print("\n------------------------------")
        print(f"\n[CONNECTION ERROR] Could not connect to the mcpd daemon at {mcpd_endpoint}")
        print("Please ensure the mcpd application is running with the 'daemon' command.")
        print("\n------------------------------")
