"""Example demonstrating manual tool execution with mcpd SDK."""

import json
import os

import requests
from mcpd import McpdClient, McpdError

if __name__ == "__main__":
    mcpd_endpoint = os.getenv("MCPD_ADDR", "http://localhost:8090")
    mcpd_api_key = os.getenv("MCPD_API_KEY")  # NOTE: Not used at present.

    try:
        mcpd_client = McpdClient(api_endpoint=mcpd_endpoint, api_key=mcpd_api_key)

        # List Servers
        print("\n--- Listing MCP Servers ---\n")
        servers = mcpd_client.servers()
        print(f"Found {len(servers)} servers: {servers}")

        # Get tool *definitions for all servers
        print("\n--- Listing Tool Definitions ---\n")
        tool_defs = mcpd_client.tools()
        print("Found definitions:")
        print(json.dumps(tool_defs, indent=2))

        # Get tool *definitions* for a specific server
        if "time" in servers:
            print("\n--- Listing Tool Definitions for 'time' ---\n")
            time_tool_defs = mcpd_client.tools("time")
            print("Found definitions for 'time':")
            print(json.dumps(time_tool_defs, indent=2))

        # Use the dynamic caller
        if "time" in servers:
            print("\n--- Calling Tool via Dynamic Caller ---\n")
            print("Calling `mcpd_client.call.time.get_current_time(timezone='Asia/Tokyo')`...")
            result = mcpd_client.call.time.get_current_time(timezone="Asia/Tokyo")
            print("Result:\n")
            print(json.dumps(result, indent=2))

        # Check if a tool exists
        print("\n--- Checking for Tool Existence ---\n")
        has_it = mcpd_client.has_tool("time", "get_current_time")
        print(f"Q: Does 'time' server have 'get_current_time' tool?\nA: {has_it}\n")
        has_it_not = mcpd_client.has_tool("time", "non_existent_tool")
        print(f"Does 'time' server have 'non_existent_tool' tool?\nA: {has_it_not}\n")

    except McpdError as e:
        print("\n------------------------------")
        print(f"\n[SDK ERROR] An error occurred: {e}")
        print("\n------------------------------")
    except requests.exceptions.ConnectionError:
        print("\n------------------------------")
        print(f"\n[CONNECTION ERROR] Could not connect to the mcpd daemon at {mcpd_endpoint}")
        print("Please ensure the mcpd application is running with the 'daemon' command.")
        print("\n------------------------------")
