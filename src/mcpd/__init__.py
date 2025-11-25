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

from ._logger import Logger, LogLevel
from .exceptions import (
    AuthenticationError,
    ConnectionError,
    McpdError,
    ServerNotFoundError,
    ServerUnhealthyError,
    TimeoutError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
)
from .mcpd_client import HealthStatus, McpdClient

__all__ = [
    "McpdClient",
    "HealthStatus",
    "Logger",
    "LogLevel",
    "McpdError",
    "AuthenticationError",
    "ConnectionError",
    "ServerNotFoundError",
    "ServerUnhealthyError",
    "TimeoutError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ValidationError",
]
