"""
Configuration management for newspaper explorer.
Loads settings from .env file and provides defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
# Path from utils/config.py -> utils -> newspaper_explorer -> src -> project_root
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)


class Config:
    """Configuration class for newspaper explorer."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Determine project root
        self.project_root = project_root

        # Data directories
        self.data_dir = self._get_path("DATA_DIR", self.project_root / "data")
        self.download_dir = self._get_path("DOWNLOAD_DIR", self.data_dir / "downloads")
        self.extracted_dir = self._get_path("EXTRACTED_DIR", self.data_dir / "extracted")
        self.sources_dir = self._get_path("SOURCES_DIR", self.data_dir / "sources")

        # Results directory
        self.results_dir = self._get_path("RESULTS_DIR", self.project_root / "results")

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def _get_path(self, env_var: str, default: Path) -> Path:
        """Get a path from environment variable or use default."""
        value = os.getenv(env_var)
        if value:
            path = Path(value)
            # If relative path, make it relative to project root
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return default

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
