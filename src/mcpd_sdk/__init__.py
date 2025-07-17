from importlib.metadata import PackageNotFoundError, version

from .mcpd_client import McpdClient, McpdError

try:
    __version__ = version("mcpd_sdk")
except PackageNotFoundError:
    # In the case of local development
    # i.e., running directly from the source directory without package being installed
    __version__ = "0.0.0-dev"

__all__ = [
    "McpdClient",
    "McpdError",
    "__version__",
]
