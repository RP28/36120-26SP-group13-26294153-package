from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

def main() -> None:
    """Print the installed MLWeave version."""
    try:
        package_version = version("mlweave")
    except PackageNotFoundError:
        package_version = "unknown"
    print(f"mlweave {package_version}")
