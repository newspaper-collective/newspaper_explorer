"""
Analysis and preprocessing results I/O utilities.

This module provides functions for saving and loading analysis and preprocessing
results with standardized directory structures and metadata tracking.
"""

import logging
from pathlib import Path
from typing import Optional, cast

import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.metadata import load_metadata, save_metadata
from newspaper_explorer.models.data.metadata import (
    AnalysisMetadata,
    AnalysisType,
    PreprocessingMetadata,
)

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
        >>> from newspaper_explorer.models.data.metadata import AnalysisMetadata
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


def load_analysis_results(
    source: str,
    analysis_type: AnalysisType,
    run_id: Optional[str] = None,
    results_base_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Load analysis results for a source.

    Handles both flat structure (legacy) and timestamped runs (current).
    If run_id is not specified, loads the most recent run.

    Args:
        source: Source name (e.g., "der_tag")
        analysis_type: Type of analysis (entities, emotions, keywords, etc.)
        run_id: Optional specific run ID. If None, loads most recent.
        results_base_dir: Optional base directory. Defaults to config.results_dir

    Returns:
        DataFrame with analysis results

    Raises:
        FileNotFoundError: If no results found for the source/analysis type

    Example:
        >>> from newspaper_explorer.data.utils.results import load_analysis_results
        >>>
        >>> # Load most recent emotions
        >>> df = load_analysis_results("der_tag", "emotions")
        >>>
        >>> # Load specific run
        >>> df = load_analysis_results(
        ...     "der_tag",
        ...     "keywords",
        ...     run_id="keybert_multi_v1_20251109_120000"
        ... )
    """
    if results_base_dir is None:
        config = get_config()
        results_base_dir = Path(config.results_dir)

    analysis_dir = results_base_dir / source / analysis_type
    parquet_filename = f"{analysis_type}.parquet"

    if not analysis_dir.exists():
        raise FileNotFoundError(
            f"No {analysis_type} results found for source '{source}' at {analysis_dir}"
        )

    # Strategy 1: Try flat structure (legacy: results/{source}/{type}/{type}.parquet)
    flat_path = analysis_dir / parquet_filename
    if flat_path.exists() and run_id is None:
        logger.debug(f"Loading {analysis_type} from flat structure: {flat_path}")
        return pl.read_parquet(flat_path)

    # Strategy 2: Look for timestamped run directories
    run_dirs = sorted([d for d in analysis_dir.glob("*/") if d.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(
            f"No {analysis_type} results found for source '{source}'. "
            f"Expected either {flat_path} or run directories in {analysis_dir}"
        )

    # Select run directory
    if run_id:
        target_dir = analysis_dir / run_id
        if not target_dir.exists():
            available = [d.name for d in run_dirs]
            raise FileNotFoundError(
                f"Run '{run_id}' not found. Available runs: {', '.join(available)}"
            )
    else:
        # Use most recent (last in sorted list)
        target_dir = run_dirs[-1]
        logger.debug(f"Loading most recent run: {target_dir.name}")

    results_path = target_dir / parquet_filename
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results file not found: {results_path}")

    logger.debug(f"Loading {analysis_type} from: {results_path}")
    return pl.read_parquet(results_path)


def list_analysis_results(
    source: str,
    analysis_type: AnalysisType,
    results_base_dir: Optional[Path] = None,
) -> list[str]:
    """
    List all available run IDs for a source and analysis type.

    Args:
        source: Source name
        analysis_type: Type of analysis
        results_base_dir: Optional base directory. Defaults to config.results_dir

    Returns:
        List of run IDs (directory names), sorted chronologically (oldest first).
        Returns empty list if no runs found.

    Example:
        >>> from newspaper_explorer.data.utils.results import list_analysis_runs
        >>>
        >>> runs = list_analysis_runs("der_tag", "emotions")
        >>> print(runs)
        ['goemotion_20251109_120000', 'goemotion_20251110_150000']
    """
    if results_base_dir is None:
        config = get_config()
        results_base_dir = Path(config.results_dir)

    analysis_dir = results_base_dir / source / analysis_type

    if not analysis_dir.exists():
        return []

    # Check for flat structure
    flat_path = analysis_dir / f"{analysis_type}.parquet"
    if flat_path.exists():
        return ["default"]  # Legacy flat structure

    # List timestamped runs
    return sorted([d.name for d in analysis_dir.glob("*/") if d.is_dir()])


def load_analysis_metadata(
    source: str,
    analysis_type: AnalysisType,
    run_id: Optional[str] = None,
    results_base_dir: Optional[Path] = None,
) -> AnalysisMetadata:
    """
    Load metadata for an analysis run.

    Args:
        source: Source name
        analysis_type: Type of analysis
        run_id: Optional specific run ID. If None, loads most recent.
        results_base_dir: Optional base directory. Defaults to config.results_dir

    Returns:
        AnalysisMetadata object

    Raises:
        FileNotFoundError: If metadata file not found

    Example:
        >>> from newspaper_explorer.data.utils.results import load_analysis_metadata
        >>>
        >>> metadata = load_analysis_metadata("der_tag", "emotions")
        >>> print(f"Method: {metadata.method_type}, Model: {metadata.model_name}")
    """
    if results_base_dir is None:
        config = get_config()
        results_base_dir = Path(config.results_dir)

    analysis_dir = results_base_dir / source / analysis_type
    parquet_filename = f"{analysis_type}.parquet"

    if not analysis_dir.exists():
        raise FileNotFoundError(f"No {analysis_type} directory found at {analysis_dir}")

    # Strategy 1: Flat structure
    flat_path = analysis_dir / parquet_filename
    if flat_path.exists() and run_id is None:
        return cast("AnalysisMetadata", load_metadata(flat_path))

    # Strategy 2: Timestamped runs
    run_dirs = sorted([d for d in analysis_dir.glob("*/") if d.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(f"No runs found in {analysis_dir}")

    # Select run
    if run_id:
        target_dir = analysis_dir / run_id
        if not target_dir.exists():
            available = [d.name for d in run_dirs]
            raise FileNotFoundError(f"Run '{run_id}' not found. Available: {', '.join(available)}")
    else:
        target_dir = run_dirs[-1]

    parquet_path = target_dir / parquet_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    return cast("AnalysisMetadata", load_metadata(parquet_path))


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
        >>> from newspaper_explorer.models.data.metadata import PreprocessingMetadata
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
