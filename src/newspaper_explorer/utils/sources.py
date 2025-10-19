"""
Source configuration management utilities.
Centralized functions for loading and managing source configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from newspaper_explorer.utils.config import get_config


def list_available_sources() -> List[str]:
    """
    List all available sources from the sources directory.

    Returns:
        List of source names (e.g., ['der_tag'])
    """
    config = get_config()
    sources = []

    if config.sources_dir.exists():
        for source_file in config.sources_dir.glob("*.json"):
            sources.append(source_file.stem)

    return sorted(sources)


def load_source_config(source_name: str) -> dict[str, Any]:
    """
    Load configuration for a specific source.

    Args:
        source_name: Name of the source (e.g., 'der_tag')

    Returns:
        Dictionary with source configuration and metadata

    Raises:
        ValueError: If source not found
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
    """
    Get relevant paths for a source.

    Args:
        config_data: Source configuration dictionary

    Returns:
        Dictionary with 'raw_dir', 'text_dir', 'images_dir', and 'output_file' paths
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
