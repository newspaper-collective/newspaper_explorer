"""
Base configuration class for newspaper explorer.
Centralized access to paths, settings, and environment variables.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Configuration for newspaper explorer.

    Automatically loads from environment variables and .env file.
    All paths support both absolute and relative (to project root) paths.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Project root (computed, not from env)
    # Navigates from config/base.py -> config -> newspaper_explorer -> src -> project_root
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent.parent, exclude=True
    )

    # Data directories
    data_dir: Path = Field(default=Path("data"), description="Main data directory")
    download_dir: Path = Field(default=Path("data/downloads"), description="Download directory")
    extracted_dir: Path = Field(default=Path("data/extracted"), description="Extraction directory")
    sources_dir: Path = Field(default=Path("data/sources"), description="Sources config directory")

    # Results directory
    results_dir: Path = Field(default=Path("results"), description="Results output directory")

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    cli_log_format: str = Field(default="%(message)s", description="Log format for CLI output")

    # LLM settings
    llm_base_url: str = Field(default="", description="LLM API base URL")
    llm_api_key: str = Field(default="", description="LLM API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=2000, gt=0, description="LLM max tokens")

    # Validation settings
    min_image_size_bytes: int = Field(
        default=1024, gt=0, description="Minimum valid image file size in bytes"
    )
    default_alto_pattern: str = Field(
        default="**/fulltext/*.xml", description="Default glob pattern for ALTO XML files"
    )

    @field_validator("data_dir", "download_dir", "extracted_dir", "sources_dir", "results_dir")
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Convert relative paths to absolute (relative to project root)."""
        if not v.is_absolute():
            # config/base.py -> config -> newspaper_explorer -> src -> project_root
            project_root = Path(__file__).parent.parent.parent.parent
            return project_root / v
        return v

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


# Global config instance
_config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config


def get_project_root() -> Path:
    """Get the project root directory."""
    return _config.project_root
