"""
Data statistics extraction utilities.

This module provides functions for extracting statistics from DataFrames
for metadata tracking and analysis provenance.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


def extract_input_stats(
    df: pl.DataFrame,
    id_column: str = "line_id",
    date_column: str = "date",
    input_path: Optional[Union[Path, str]] = None,
) -> dict[str, Any]:
    """
    Extract statistics from input DataFrame for metadata.

    If input_path is provided and has associated preprocessing metadata,
    includes preprocessing information for full provenance tracking.

    Args:
        df: Input DataFrame
        id_column: Name of ID column
        date_column: Name of date column
        input_path: Optional path to input parquet file (for loading preprocessing metadata)

    Returns:
        Dictionary with input statistics (including preprocessing info if available)

    Example:
        >>> df = pl.read_parquet("data/processed/der_tag/text/normalize_lowercase_20251110/textblocks.parquet")
        >>> stats = extract_input_stats(df, input_path="data/processed/...")
        >>> print(stats["preprocessing"])  # Will include preprocessing metadata if available
        {'preprocessing_id': 'normalize_lowercase_20251110_120000', 'steps': [...]}
    """
    stats = {
        "row_count": len(df),
        "columns": df.columns,
        "schema": {col: str(dtype) for col, dtype in df.schema.items()},
    }

    # Add date range if date column exists
    if date_column in df.columns:
        try:
            dates = df.select(pl.col(date_column)).to_series()
            min_date = dates.min()
            max_date = dates.max()
            stats["date_range"] = [
                str(min_date) if min_date else None,
                str(max_date) if max_date else None,
            ]
        except Exception as e:
            # Date column exists but might not be datetime type
            stats["date_range"] = None

    # Add ID type info if available
    if id_column in df.columns:
        stats["id_column"] = id_column
        sample_id = df.select(pl.col(id_column)).head(1).item()
        if sample_id:
            from newspaper_explorer.data.utils.ids import identify_id_type

            stats["id_type"] = identify_id_type(sample_id)

    # Check for preprocessing metadata if input path provided
    if input_path:
        from newspaper_explorer.data.utils.metadata import find_metadata_for_parquet, load_metadata
        from newspaper_explorer.models.data.metadata import PreprocessingMetadata

        metadata_path = find_metadata_for_parquet(input_path)
        if metadata_path:
            try:
                metadata = load_metadata(metadata_path)
                if isinstance(metadata, PreprocessingMetadata):
                    # Include full preprocessing provenance
                    stats["preprocessing"] = {
                        "preprocessing_id": metadata.preprocessing_id,
                        "steps": metadata.get_all_steps(),  # Includes chained steps
                        "parameters": metadata.parameters,
                        "created_at": metadata.created_at,
                        "metadata_path": str(metadata_path),
                    }
                    logger.debug(f"Loaded preprocessing metadata: {metadata.preprocessing_id}")
            except Exception as e:
                logger.debug(f"Could not load preprocessing metadata from {metadata_path}: {e}")

    return stats


def extract_output_stats(df: pl.DataFrame) -> dict[str, Any]:
    """
    Extract statistics from output DataFrame for metadata.

    Args:
        df: Output DataFrame

    Returns:
        Dictionary with output statistics

    Example:
        >>> results_df = extractor.extract(input_df)
        >>> stats = extract_output_stats(results_df)
    """
    return {
        "row_count": len(df),
        "columns": df.columns,
        "schema": {col: str(dtype) for col, dtype in df.schema.items()},
    }
