"""
Analysis and preprocessing results I/O utilities.

This module provides functions for saving analysis and preprocessing results
with standardized directory structures and metadata tracking.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from newspaper_explorer.data.models import AnalysisMetadata, PreprocessingMetadata
from newspaper_explorer.data.utils.metadata import save_metadata

logger = logging.getLogger(__name__)


def save_analysis_results(
    results_df: pl.DataFrame,
    metadata: AnalysisMetadata,
    results_base_dir: Path,
    results_filename: Optional[str] = None,
) -> dict[str, Path]:
    """
    Save analysis results with standardized subdirectory structure.

    This is the unified way to save analysis results across all analysis types.
    Creates structure: {results_base_dir}/{analysis_type}/{analysis_id}/

    Args:
        results_df: Results DataFrame to save
        metadata: Analysis metadata object
        results_base_dir: Base results directory (e.g., config.results_dir / source_name)
        results_filename: Name for results file (default: based on analysis_type)

    Returns:
        Dictionary with paths:
            - output_dir: Directory containing results
            - results_path: Path to parquet file
            - metadata_path: Path to metadata JSON

    Example:
        >>> from newspaper_explorer.data.utils.results import save_analysis_results
        >>> from newspaper_explorer.data.models import AnalysisMetadata
        >>> from newspaper_explorer.config.base import get_config
        >>>
        >>> metadata = AnalysisMetadata(
        ...     analysis_type="keywords",
        ...     method_type="keybert",
        ...     model_name="multi-v1",
        ...     source="der_tag",
        ...     parameters={"top_k": 10}
        ... )
        >>>
        >>> config = get_config()
        >>> results_base = config.results_dir / "der_tag"
        >>> paths = save_analysis_results(results_df, metadata, results_base)
        >>>
        >>> # Results saved to:
        >>> # results/der_tag/keywords/keybert_multi_v1_20251109_120000/
        >>> #   ├── keywords.parquet
        >>> #   └── metadata.json
    """
    # Default filenames based on analysis type
    default_filenames = {
        "entities": "entities.parquet",
        "keywords": "keywords.parquet",
        "topics": "topics.parquet",
        "emotions": "emotions.parquet",
        "layout": "layout.parquet",
        "concepts": "concepts.parquet",
    }

    if results_filename is None:
        results_filename = default_filenames.get(
            metadata.analysis_type, f"{metadata.analysis_type}.parquet"
        )

    # Create subdirectory structure: {base}/{analysis_type}/{analysis_id}/
    analysis_id = metadata.analysis_id or "unknown"
    output_dir = results_base_dir / metadata.analysis_type / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results parquet
    results_path = output_dir / results_filename
    results_df.write_parquet(results_path, compression="zstd")
    logger.info(f"Saved {len(results_df)} rows to {results_path}")

    # Save metadata
    metadata_path = save_metadata(metadata, results_path)

    logger.info(f"Analysis output directory: {output_dir}")
    logger.info(f"Analysis ID: {metadata.analysis_id}")

    return {
        "output_dir": output_dir,
        "results_path": results_path,
        "metadata_path": metadata_path,
    }


def save_preprocessing_results(
    results_df: pl.DataFrame,
    metadata: PreprocessingMetadata,
    processed_base_dir: Path,
    results_filename: str = "textblocks.parquet",
) -> dict[str, Path]:
    """
    Save preprocessing results with standardized subdirectory structure.

    Creates structure: {processed_base_dir}/{source}/text/{preprocessing_id}/

    Args:
        results_df: Preprocessed DataFrame to save
        metadata: Preprocessing metadata object
        processed_base_dir: Base processed directory (e.g., config.processed_dir)
        results_filename: Name for results file (default: textblocks.parquet)

    Returns:
        Dictionary with paths:
            - output_dir: Directory containing results
            - results_path: Path to parquet file
            - metadata_path: Path to metadata JSON

    Example:
        >>> from newspaper_explorer.data.utils.results import save_preprocessing_results
        >>> from newspaper_explorer.data.models import PreprocessingMetadata
        >>> from newspaper_explorer.config.base import get_config
        >>> config = get_config()
        >>>
        >>> metadata = PreprocessingMetadata(
        ...     source="der_tag",
        ...     steps=["normalize", "lowercase"],
        ...     parameters={"text_column": "text"}
        ... )
        >>>
        >>> paths = save_preprocessing_results(
        ...     results_df=df,
        ...     metadata=metadata,
        ...     processed_base_dir=config.processed_dir,
        ... )
        >>>
        >>> # Results saved to:
        >>> # data/processed/der_tag/text/normalize_lowercase_20251110_120000/
        >>> #   ├── textblocks.parquet
        >>> #   └── metadata.json
    """
    # Create subdirectory structure: {base}/{source}/text/{preprocessing_id}/
    preprocessing_id = metadata.preprocessing_id or "unknown"
    output_dir = processed_base_dir / metadata.source / "text" / preprocessing_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results parquet
    results_path = output_dir / results_filename
    results_df.write_parquet(results_path, compression="zstd")
    logger.info(f"Saved {len(results_df)} rows to {results_path}")

    # Save metadata
    metadata_path = save_metadata(metadata, results_path)

    logger.info(f"Preprocessing output directory: {output_dir}")
    logger.info(f"Preprocessing ID: {metadata.preprocessing_id}")

    return {
        "output_dir": output_dir,
        "results_path": results_path,
        "metadata_path": metadata_path,
    }
