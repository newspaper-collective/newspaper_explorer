"""
Source configuration management utilities.

Functions for loading source configurations and computing paths.
Uses models from newspaper_explorer.models.core.sources.
"""

from pathlib import Path

from natsort import natsorted

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.core.sources import SourceConfig


def list_available_sources() -> list[str]:
    """
    List all available sources from the sources directory.

    Returns:
        list[str]: Naturally sorted list of source names

    Example:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['der_tag']
    """
    config = get_config()

    if config.sources_dir.exists():
        sources = [source_file.stem for source_file in config.sources_dir.glob("*.json")]
    else:
        sources = []

    return natsorted(sources)


def load_source_config(source_name: str) -> SourceConfig:
    """
    Load and validate source configuration.

    Args:
        source_name: Name of the source (e.g., 'der_tag')

    Returns:
        SourceConfig: Validated Pydantic model

    Raises:
        ValueError: If source not found or validation fails

    Example:
        >>> config = load_source_config("der_tag")
        >>> print(config.metadata.newspaper_title)
        'Der Tag'
        >>> print(config.loading.pattern)
        '**/fulltext/*.xml'
        >>> print(len(config.parts))
        7
    """
    config = get_config()
    source_file = config.sources_dir / f"{source_name}.json"

    if not source_file.exists():
        available = list_available_sources()
        raise ValueError(
            f"Source '{source_name}' not found. Available sources: {', '.join(available)}"
        )

    try:
        return SourceConfig.from_json_file(source_file)

    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid source configuration for '{source_name}': {e}") from e


def get_source_paths(source_config: SourceConfig) -> dict[str, Path]:
    """
    Get standard paths for a source's data directories and files.

    Args:
        source_config: Validated SourceConfig object

    Returns:
        Dict[str, Path]: Dictionary with paths:
            - raw_dir: Raw XML/OCR files
            - text_dir: Parsed text data
            - images_dir: Downloaded images
            - output_file: Main parquet output

    Example:
        >>> config = load_source_config("der_tag")
        >>> paths = get_source_paths(config)
        >>> print(paths["raw_dir"])
        PosixPath('data/raw/der_tag/xml_ocr')
    """
    config = get_config()
    dataset_name = source_config.dataset_name
    data_type = source_config.data_type

    raw_dir = config.data_dir / "raw" / dataset_name / data_type
    text_dir = config.data_dir / "raw" / dataset_name / "text"
    images_dir = config.data_dir / "raw" / dataset_name / "images"
    output_file = text_dir / f"{dataset_name}_lines.parquet"

    return {
        "raw_dir": raw_dir,
        "text_dir": text_dir,
        "images_dir": images_dir,
        "output_file": output_file,
    }
