from importlib.metadata import PackageNotFoundError, version

from mlweave.mlweave import main

try:
    __version__ = version("mlweave")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__", "main"]
