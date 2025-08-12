"""Example demonstrating mcpd SDK integration with Pydantic AI framework."""

import os

import requests
from mcpd import McpdClient, McpdError
from pydantic_ai import Agent

if __name__ == "__main__":
    mcpd_endpoint = os.getenv("MCPD_ADDR", "http://localhost:8090")
    mcpd_api_key = os.getenv("MCPD_API_KEY")  # NOTE: Not used at present.

    try:
        mcpd_client = McpdClient(api_endpoint=mcpd_endpoint, api_key=mcpd_api_key)

        agent = Agent(
            "openai:gpt-4o",
            system_prompt="Use the tools to find an answer",
            tools=mcpd_client.agent_tools(),
        )
        result_sync = agent.run_sync("What time is in Tokyo?")
        print(result_sync.output)

    except McpdError as e:
        print("\n------------------------------")
        print(f"\n[SDK ERROR] An error occurred: {e}")
        print("\n------------------------------")
    except requests.exceptions.ConnectionError:
        print("\n------------------------------")
        print(f"\n[CONNECTION ERROR] Could not connect to the mcpd daemon at {mcpd_endpoint}")
        print("Please ensure the mcpd application is running with the 'daemon' command.")
        print("\n------------------------------")
