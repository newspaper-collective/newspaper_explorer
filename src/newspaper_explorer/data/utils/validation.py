"""
Data validation utilities for newspaper data quality assessment.

Functions for checking data integrity, finding missing or empty files,
and validating data completeness.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import polars as pl
from natsort import natsorted

from newspaper_explorer.utils.sources import get_source_paths, load_source_config

logger = logging.getLogger(__name__)

# Default glob pattern for finding ALTO XML files
DEFAULT_ALTO_PATTERN = "**/fulltext/*.xml"


def find_empty_xml_files(source_name: str) -> Dict[str, Any]:
    """
    Find XML files without OCR transcription (no text content).

    Compares all XML files in the source directory with processed files
    in the parquet output to identify files that were skipped due to
    having no text content (e.g., pages with only images/graphics).

    Args:
        source_name: Name of the source to check (e.g., 'der_tag')

    Returns:
        Dictionary with:
        - total_xml_files: Total number of XML files found
        - processed_files: Number of files with text content
        - empty_files: Number of files without text
        - empty_file_list: List of paths to empty files

    Example:
        >>> result = find_empty_xml_files("der_tag")
        >>> print(f"Found {result['empty_files']} empty files")
    """
    # Load source configuration
    config = load_source_config(source_name)
    paths = get_source_paths(config)
    raw_dir = paths["raw_dir"]
    output_file = paths["output_file"]

    # Get loading config
    loading_config = config.get("loading", {})
    pattern = loading_config.get("pattern", DEFAULT_ALTO_PATTERN)

    logger.info(f"Scanning for XML files in {raw_dir}")
    all_files = natsorted([str(f.relative_to(raw_dir)) for f in raw_dir.glob(pattern)])

    logger.info(f"Found {len(all_files)} XML files")

    # Get processed files from parquet
    if not output_file.exists():
        logger.warning("No parquet file found - run data load first")
        return {
            "total_xml_files": len(all_files),
            "processed_files": 0,
            "empty_files": len(all_files),
            "empty_file_list": all_files,
        }

    logger.info("Loading processed files from parquet")
    df = pl.read_parquet(output_file)
    processed_filenames = set(df["filename"].unique())

    logger.info(f"Found {len(processed_filenames)} processed files")

    # Find files that weren't processed (empty)
    empty_files = [f for f in all_files if Path(f).name not in processed_filenames]

    return {
        "total_xml_files": len(all_files),
        "processed_files": len(processed_filenames),
        "empty_files": len(empty_files),
        "empty_file_list": empty_files,
    }
