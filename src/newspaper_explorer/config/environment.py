"""
Environment variable loading and path resolution.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def get_project_root() -> Path:
    """
    Get the project root directory.

    Navigates from config/environment.py -> config -> newspaper_explorer -> src -> project_root
    """
    return Path(__file__).parent.parent.parent.parent


def load_environment():
    """Load environment variables from .env file if it exists."""
    project_root = get_project_root()
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)


def get_env_path(env_var: str, default: Path) -> Path:
    """
    Get a path from environment variable or use default.

    Args:
        env_var: Environment variable name
        default: Default path if variable not set

    Returns:
        Resolved absolute path
    """
    value = os.getenv(env_var)
    if value:
        path = Path(value)
        # If relative path, make it relative to project root
        if not path.is_absolute():
            path = get_project_root() / path
        return path
    return default


# Load environment on import
load_environment()
