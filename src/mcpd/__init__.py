"""mcpd Python SDK.

A Python SDK for interacting with the mcpd daemon, which manages
Model Context Protocol (MCP) servers and enables seamless tool execution
through natural Python syntax.

This package provides:
- McpdClient: Main client for server management and tool execution
- Dynamic calling: Natural syntax like client.call.server.tool(**kwargs)
- Agent-ready functions: Generate callable functions via agent_tools() for AI frameworks
- Type-safe function generation: Create callable functions from tool schemas
- Comprehensive error handling: Detailed exceptions for different failure modes
"""

from .exceptions import (
    AuthenticationError,
    ConnectionError,
    McpdError,
    ServerNotFoundError,
    TimeoutError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
)
from .mcpd_client import McpdClient

__all__ = [
    "McpdClient",
    "McpdError",
    "AuthenticationError",
    "ConnectionError",
    "ServerNotFoundError",
    "TimeoutError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ValidationError",
]
