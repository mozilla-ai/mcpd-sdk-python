"""Internal logging infrastructure for the mcpd SDK.

This module provides a logging shim controlled by the MCPD_LOG_LEVEL environment
variable. Logging is disabled by default to avoid contaminating stdout/stderr in
MCP server contexts.

CRITICAL: Only enable MCPD_LOG_LEVEL in non-MCP-server contexts. MCP servers use
stdout for JSON-RPC communication, and any logging output will break the protocol.
"""

import logging
import os
from enum import Enum
from typing import Protocol


class LogLevel(str, Enum):
    """Valid log level values for MCPD_LOG_LEVEL environment variable.

    Aligns with mcpd server binary log levels for consistency across the mcpd ecosystem.
    """

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    OFF = "off"


class Logger(Protocol):
    """Logger protocol defining the SDK's logging interface.

    This protocol matches standard logging levels and allows custom logger injection.
    All methods accept a message and optional formatting arguments.
    """

    def trace(self, msg: str, *args: object) -> None:
        """Log a trace-level message (most verbose)."""
        ...

    def debug(self, msg: str, *args: object) -> None:
        """Log a debug-level message."""
        ...

    def info(self, msg: str, *args: object) -> None:
        """Log an info-level message."""
        ...

    def warn(self, msg: str, *args: object) -> None:
        """Log a warning-level message."""
        ...

    def error(self, msg: str, *args: object) -> None:
        """Log an error-level message."""
        ...


# Custom TRACE level (below DEBUG=10).
_TRACE = 5
logging.addLevelName(_TRACE, "TRACE")

_RANKS: dict[str, int] = {
    LogLevel.TRACE.value: _TRACE,
    LogLevel.DEBUG.value: logging.DEBUG,
    LogLevel.INFO.value: logging.INFO,
    LogLevel.WARN.value: logging.WARNING,
    "warning": logging.WARNING,  # Alias for backwards compatibility.
    LogLevel.ERROR.value: logging.ERROR,
    LogLevel.OFF.value: 1000,  # Higher than any standard level.
}


def _resolve_log_level(raw: str | None) -> str:
    """Resolve the log level from environment variable value.

    Args:
        raw: Raw value from MCPD_LOG_LEVEL environment variable.

    Returns:
        Valid log level string matching LogLevel enum values.
        Returns LogLevel.OFF.value if raw is None, empty, or not a valid level.
    """
    candidate = raw.strip().lower() if raw else None
    return candidate if candidate and candidate in _RANKS else LogLevel.OFF.value


def _get_level() -> str:
    """Get the current log level from environment variable (lazy evaluation).

    This function is called on each log statement to support dynamic level changes.

    Note:
        Dynamic level changes can facilitate testing.

    Returns:
        The resolved log level string.
    """
    return _resolve_log_level(os.getenv("MCPD_LOG_LEVEL"))


def _create_default_logger() -> Logger:
    """Create the default logger with lazy level evaluation.

    Returns:
        A Logger instance that checks MCPD_LOG_LEVEL on each log call,
        enabling dynamic level changes without module reloading.
    """
    # Create logger and handler once (not per-call).
    _logger = logging.getLogger(__name__)

    if not _logger.handlers:
        # Add stderr handler (default for StreamHandler).
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        _logger.addHandler(handler)
        _logger.propagate = False

    class _DefaultLogger:
        """Default logger that checks level on each call (lazy evaluation)."""

        def trace(self, msg: str, *args: object) -> None:
            """Log trace-level message."""
            lvl = _get_level()
            if lvl != LogLevel.OFF.value and _RANKS[lvl] <= _RANKS[LogLevel.TRACE.value]:
                _logger.setLevel(_TRACE)
                _logger.log(_TRACE, msg, *args)

        def debug(self, msg: str, *args: object) -> None:
            """Log debug-level message."""
            lvl = _get_level()
            if lvl != LogLevel.OFF.value and _RANKS[lvl] <= _RANKS[LogLevel.DEBUG.value]:
                _logger.setLevel(logging.DEBUG)
                _logger.debug(msg, *args)

        def info(self, msg: str, *args: object) -> None:
            """Log info-level message."""
            lvl = _get_level()
            if lvl != LogLevel.OFF.value and _RANKS[lvl] <= _RANKS[LogLevel.INFO.value]:
                _logger.setLevel(logging.INFO)
                _logger.info(msg, *args)

        def warn(self, msg: str, *args: object) -> None:
            """Log warning-level message."""
            lvl = _get_level()
            if lvl != LogLevel.OFF.value and _RANKS[lvl] <= _RANKS[LogLevel.WARN.value]:
                _logger.setLevel(logging.WARNING)
                _logger.warning(msg, *args)

        def error(self, msg: str, *args: object) -> None:
            """Log error-level message."""
            lvl = _get_level()
            if lvl != LogLevel.OFF.value and _RANKS[lvl] <= _RANKS[LogLevel.ERROR.value]:
                _logger.setLevel(logging.ERROR)
                _logger.error(msg, *args)

    return _DefaultLogger()


class _PartialLoggerWrapper:
    """Wrapper that combines partial custom logger with default logger fallback.

    This enables partial logger implementations where users can override specific
    methods while keeping defaults for others.
    """

    def __init__(self, custom: object, default: Logger) -> None:
        """Initialize the wrapper.

        Args:
            custom: Partial logger implementation (may not have all methods).
            default: Default logger to use for missing methods.
        """
        self._custom = custom
        self._default = default

    def trace(self, msg: str, *args: object) -> None:
        """Log trace-level message."""
        if hasattr(self._custom, LogLevel.TRACE.value):
            self._custom.trace(msg, *args)
        else:
            self._default.trace(msg, *args)

    def debug(self, msg: str, *args: object) -> None:
        """Log debug-level message."""
        if hasattr(self._custom, LogLevel.DEBUG.value):
            self._custom.debug(msg, *args)
        else:
            self._default.debug(msg, *args)

    def info(self, msg: str, *args: object) -> None:
        """Log info-level message."""
        if hasattr(self._custom, LogLevel.INFO.value):
            self._custom.info(msg, *args)
        else:
            self._default.info(msg, *args)

    def warn(self, msg: str, *args: object) -> None:
        """Log warning-level message."""
        if hasattr(self._custom, LogLevel.WARN.value):
            self._custom.warn(msg, *args)
        else:
            self._default.warn(msg, *args)

    def error(self, msg: str, *args: object) -> None:
        """Log error-level message."""
        if hasattr(self._custom, LogLevel.ERROR.value):
            self._custom.error(msg, *args)
        else:
            self._default.error(msg, *args)


def create_logger(impl: Logger | object | None = None) -> Logger:
    """Create a logger, optionally using a custom implementation.

    This function allows SDK users to inject their own logger implementation.
    Supports partial implementations - any omitted methods will fall back to the
    default logger, which respects the MCPD_LOG_LEVEL environment variable.

    Args:
        impl: Optional custom Logger implementation or partial implementation.
              If None, uses the default logger controlled by MCPD_LOG_LEVEL.
              If partially provided, custom methods are used and omitted methods
              fall back to default logger (which respects MCPD_LOG_LEVEL).

    Returns:
        A Logger instance with all methods implemented.

    Example:
        >>> # Use default logger (controlled by MCPD_LOG_LEVEL).
        >>> logger = create_logger()
        >>>
        >>> # Full custom logger.
        >>> class MyLogger:
        ...     def trace(self, msg, *args): pass
        ...     def debug(self, msg, *args): pass
        ...     def info(self, msg, *args): pass
        ...     def warn(self, msg, *args): print(f"WARN: {msg % args}")
        ...     def error(self, msg, *args): print(f"ERROR: {msg % args}")
        >>> logger = create_logger(MyLogger())
        >>>
        >>> # Partial logger: custom warn/error, default (MCPD_LOG_LEVEL-aware) for others.
        >>> class PartialLogger:
        ...     def warn(self, msg, *args): print(f"WARN: {msg % args}")
        ...     def error(self, msg, *args): print(f"ERROR: {msg % args}")
        ...     # trace, debug, info use default logger (respects MCPD_LOG_LEVEL)
        >>> logger = create_logger(PartialLogger())
    """
    if impl is None:
        return _default_logger

    # Check if it's a full Logger implementation (has all required methods).
    required_methods = [
        LogLevel.TRACE.value,
        LogLevel.DEBUG.value,
        LogLevel.INFO.value,
        LogLevel.WARN.value,
        LogLevel.ERROR.value,
    ]
    if all(hasattr(impl, method) for method in required_methods):
        return impl

    # Partial implementation - wrap with fallback to default logger.
    return _PartialLoggerWrapper(impl, _default_logger)


# Module-level default logger (created at import time).
_default_logger: Logger = _create_default_logger()
