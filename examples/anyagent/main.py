"""Example demonstrating mcpd SDK integration with AnyAgent framework."""

import os

import requests
from any_agent import AgentConfig, AnyAgent
from mcpd import McpdClient, McpdError

if __name__ == "__main__":
    mcpd_endpoint = os.getenv("MCPD_ADDR", "http://localhost:8090")
    mcpd_api_key = os.getenv("MCPD_API_KEY")  # NOTE: Not used at present.

    try:
        mcpd_client = McpdClient(api_endpoint=mcpd_endpoint, api_key=mcpd_api_key)

        # Use any-agent with agent friendly tools
        # Please ensure you have exported your OPENAI_API_KEY
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
