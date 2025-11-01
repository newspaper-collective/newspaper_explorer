"""Source configuration management utilities.

This module provides centralized functions for loading and managing source configurations
from the data/sources/ directory. All source-related operations (download, parse, analyze)
rely on these utilities to access configuration data.

Key Functions:
    - list_available_sources(): Get all available source names
    - load_source_config(): Load a source's JSON configuration
    - get_source_paths(): Calculate standard paths for a source's data

Example:
    >>> from newspaper_explorer.utils.sources import load_source_config, get_source_paths
    >>>
    >>> # Load source configuration
    >>> config = load_source_config("der_tag")
    >>> print(config["metadata"]["newspaper_title"])
    'Der Tag'
    >>>
    >>> # Get standard paths
    >>> paths = get_source_paths(config)
    >>> print(paths["output_file"])
    PosixPath('data/raw/der_tag/text/der_tag_lines.parquet')
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from natsort import natsorted

from newspaper_explorer.config.base import get_config


def list_available_sources() -> List[str]:
    """List all available sources from the sources directory.

    Scans the configured sources directory (typically data/sources/) for JSON
    configuration files and returns their names (without .json extension).
    Results are naturally sorted for consistent ordering.

    Returns:
        List[str]: Naturally sorted list of source names (e.g., ['der_tag', 'wiener_zeitung'])

    Example:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['der_tag']
        >>>
        >>> # Check if a source exists
        >>> if 'der_tag' in list_available_sources():
        ...     print("Source available!")
    """
    config = get_config()
    sources = []

    if config.sources_dir.exists():
        for source_file in config.sources_dir.glob("*.json"):
            sources.append(source_file.stem)

    return natsorted(sources)


def load_source_config(source_name: str) -> dict[str, Any]:
    """Load configuration for a specific source.

    Reads and parses the JSON configuration file for the given source from
    the sources directory. The configuration includes metadata (title, language,
    years), loading instructions (patterns, compression), and download parts.

    Args:
        source_name (str): Name of the source (e.g., 'der_tag'), corresponding
            to the filename without .json extension

    Returns:
        dict[str, Any]: Dictionary containing:
            - dataset_name (str): Internal dataset identifier
            - data_type (str): Type of data (e.g., 'xml_ocr')
            - metadata (dict): Newspaper metadata (title, language, years_available)
            - loading (dict): Loading configuration (pattern, compression)
            - parts (list): Download parts with URLs, checksums, filenames

    Raises:
        ValueError: If source configuration file not found. Error message includes
            list of available sources.

    Example:
        >>> config = load_source_config("der_tag")
        >>> print(config["metadata"]["newspaper_title"])
        'Der Tag'
        >>> print(config["loading"]["pattern"])
        '**/fulltext/*.xml'
        >>> print(len(config["parts"]))
        21
    """
    config = get_config()
    source_file = config.sources_dir / f"{source_name}.json"

    if not source_file.exists():
        available = list_available_sources()
        raise ValueError(
            f"Source '{source_name}' not found. Available sources: {', '.join(available)}"
        )

    with open(source_file, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
        return data


def get_source_paths(config_data: Dict[str, Any]) -> Dict[str, Path]:
    """Get standard paths for a source's data directories and files.

    Calculates the expected filesystem paths for a source based on its configuration.
    This ensures consistent path usage across download, loading, and analysis operations.
    All paths are based on the configured data directory (from environment or default).

    Args:
        config_data (Dict[str, Any]): Source configuration dictionary as returned
            by load_source_config(). Must contain 'dataset_name' and 'data_type' keys.

    Returns:
        Dict[str, Path]: Dictionary with the following keys:
            - raw_dir (Path): Raw XML/OCR files (data/raw/{dataset_name}/{data_type}/)
            - text_dir (Path): Parsed text data (data/raw/{dataset_name}/text/)
            - images_dir (Path): Downloaded images (data/raw/{dataset_name}/images/)
            - output_file (Path): Main parquet output (data/raw/{dataset_name}/text/{dataset_name}_lines.parquet)

    Example:
        >>> config = load_source_config("der_tag")
        >>> paths = get_source_paths(config)
        >>> print(paths["raw_dir"])
        PosixPath('data/raw/der_tag/xml_ocr')
        >>> print(paths["output_file"])
        PosixPath('data/raw/der_tag/text/der_tag_lines.parquet')
        >>>
        >>> # Check if data has been processed
        >>> if paths["output_file"].exists():
        ...     print("Source has been parsed!")
    """
    config = get_config()
    dataset_name = config_data["dataset_name"]
    data_type = config_data["data_type"]

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
