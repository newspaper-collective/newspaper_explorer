"""
Base configuration class for newspaper explorer.
Centralized access to paths, settings, and environment variables.
"""

import os

from newspaper_explorer.config.environment import get_env_path, get_project_root


class Config:
    """Configuration class for newspaper explorer."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Determine project root
        self.project_root = get_project_root()

        # Data directories
        self.data_dir = get_env_path("DATA_DIR", self.project_root / "data")
        self.download_dir = get_env_path("DOWNLOAD_DIR", self.data_dir / "downloads")
        self.extracted_dir = get_env_path("EXTRACTED_DIR", self.data_dir / "extracted")
        self.sources_dir = get_env_path("SOURCES_DIR", self.data_dir / "sources")

        # Results directory
        self.results_dir = get_env_path("RESULTS_DIR", self.project_root / "results")

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # LLM settings
        self.llm_base_url = os.getenv("LLM_BASE_URL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    def get(self, key: str, default: str = "") -> str:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (e.g., "llm_base_url", "llm_api_key").
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        return getattr(self, key, default)

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config
