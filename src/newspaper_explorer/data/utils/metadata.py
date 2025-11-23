"""
Unified metadata management for analysis results.

This module provides utilities for creating, saving, and loading metadata
alongside analysis result parquet files. It ensures consistent metadata
structure across all analysis types and enables full reproducibility.

Metadata Pattern:
    - Each analysis result (parquet file) has an accompanying metadata JSON
    - Both files share the same base name: {analysis_id}.parquet + {analysis_id}.json
    - Metadata includes method details, parameters, input data, and provenance

Example:
    >>> from newspaper_explorer.data.models import AnalysisMetadata
    >>> from newspaper_explorer.data.utils.metadata import save_metadata
    >>>
    >>> metadata = AnalysisMetadata(
    ...     analysis_type="entities",
    ...     method_type="gliner",
    ...     model_name="multi-v2.1",
    ...     source="der_tag",
    ...     parameters={"threshold": 0.2, "batch_size": 16},
    ...     input_data={
    ...         "parquet_path": "data/raw/der_tag/text/lines.parquet",
    ...         "row_count": 100000,
    ...         "date_range": ["1901-01-08", "1920-12-31"]
    ...     }
    ... )
    >>>
    >>> # Save alongside parquet file
    >>> save_metadata(metadata, Path("results/entities/analysis.parquet"))
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from newspaper_explorer.data.models import AnalysisMetadata, PreprocessingMetadata

logger = logging.getLogger(__name__)


def save_metadata(
    metadata: Union[AnalysisMetadata, PreprocessingMetadata],
    parquet_path: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save metadata as JSON alongside parquet file.

    Creates a JSON file with the same base name as the parquet file.
    Supports both AnalysisMetadata and PreprocessingMetadata.

    Args:
        metadata: Metadata object to save (AnalysisMetadata or PreprocessingMetadata)
        parquet_path: Path to the parquet file (or its parent directory)
        filename: Optional custom filename for metadata (defaults to matching parquet name)

    Returns:
        Path to saved metadata file

    Example:
        >>> from newspaper_explorer.data.models import AnalysisMetadata
        >>> metadata = AnalysisMetadata(...)
        >>> save_metadata(metadata, Path("results/entities/analysis.parquet"))
        PosixPath('results/entities/analysis.json')
    """
    # Determine output path
    if parquet_path.is_dir():
        # If directory provided, use ID as base name
        if isinstance(metadata, AnalysisMetadata):
            base_name = filename or f"{metadata.analysis_id}.json"
        else:  # PreprocessingMetadata
            base_name = filename or f"{metadata.preprocessing_id}.json"
        json_path = parquet_path / base_name
    elif filename:
        # If file provided with custom filename
        json_path = parquet_path.parent / filename
    else:
        # If file provided, replace extension
        json_path = parquet_path.with_suffix(".json")

    # Ensure directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as formatted JSON
    json_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Saved metadata to {json_path}")
    return json_path


def load_metadata(
    metadata_path: Union[Path, str],
) -> Union[AnalysisMetadata, PreprocessingMetadata]:
    """
    Load metadata from JSON file.

    Automatically detects metadata type based on fields present.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        AnalysisMetadata or PreprocessingMetadata object

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If JSON is invalid

    Example:
        >>> metadata = load_metadata("results/entities/analysis.json")
        >>> print(metadata.model_name)
        'multi-v2.1'
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    # Auto-detect metadata type based on fields
    if "analysis_type" in data:
        return AnalysisMetadata.from_dict(data)
    if "steps" in data:
        return PreprocessingMetadata.from_dict(data)

    # Fallback to AnalysisMetadata for legacy compatibility
    return AnalysisMetadata.from_dict(data)


def find_metadata_for_parquet(parquet_path: Union[Path, str]) -> Optional[Path]:
    """
    Find metadata JSON file for a given parquet file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        Path to metadata file if found, None otherwise

    Example:
        >>> metadata_path = find_metadata_for_parquet("results/entities/analysis.parquet")
        >>> if metadata_path:
        ...     metadata = load_metadata(metadata_path)
    """
    parquet_path = Path(parquet_path)
    json_path = parquet_path.with_suffix(".json")

    if json_path.exists():
        return json_path

    # Also check for metadata.json in same directory (legacy pattern)
    legacy_path = parquet_path.parent / "metadata.json"
    if legacy_path.exists():
        return legacy_path

    return None


def update_metadata_status(
    metadata_path: Path,
    status: str,
    completed_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
    output_data: Optional[dict[str, Any]] = None,
) -> None:
    """
    Update metadata file with status and completion info.

    Useful for long-running processes that want to update metadata
    after completion or failure.

    Args:
        metadata_path: Path to metadata JSON file
        status: New status (completed, failed, in_progress)
        completed_at: ISO timestamp of completion
        duration_seconds: Processing duration
        error_message: Error details if failed
        output_data: Output statistics

    Example:
        >>> save_metadata(metadata, output_dir / "analysis.json")
        >>> # ... processing ...
        >>> update_metadata_status(
        ...     output_dir / "analysis.json",
        ...     status="completed",
        ...     completed_at=datetime.now().isoformat(),
        ...     duration_seconds=123.45,
        ...     output_data={"row_count": 5000}
        ... )
    """
    metadata = load_metadata(metadata_path)

    metadata.status = status
    if completed_at:
        metadata.completed_at = completed_at
    if duration_seconds is not None:
        metadata.duration_seconds = duration_seconds
    if error_message:
        metadata.error_message = error_message
    if output_data:
        metadata.output_data.update(output_data)

    save_metadata(metadata, metadata_path)
    logger.info(f"Updated metadata status to: {status}")


def list_analysis(
    results_dir: Path,
    analysis_type: Optional[str] = None,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    List all analysis with their metadata.

    Args:
        results_dir: Results directory (e.g., "results/")
        analysis_type: Filter by analysis type (entities, topics, etc.)
        source: Filter by source (e.g., "der_tag")

    Returns:
        List of metadata dictionaries

    Example:
        >>> from newspaper_explorer.config.base import get_config
        >>> config = get_config()
        >>> analysis = list_analysis(
        ...     config.results_dir,
        ...     analysis_type="entities",
        ...     source="der_tag"
        ... )
        >>> for item in analysis:
        ...     print(f"{item['analysis_id']}: {item['model_name']}")
    """
    results_dir = Path(results_dir)
    analysis = []

    # Pattern: results/{source}/{analysis_type}/{analysis_id}/metadata.json
    pattern = "**/*.json"
    if source:
        pattern = f"{source}/**/*.json"

    for json_path in results_dir.glob(pattern):
        # Check if it's a metadata file by trying to load it
        try:
            metadata = load_metadata(json_path)
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

        # Apply filters (only for AnalysisMetadata)
        if isinstance(metadata, AnalysisMetadata):
            if analysis_type and metadata.analysis_type != analysis_type:
                continue
            if source and metadata.source != source:
                continue
        elif isinstance(metadata, PreprocessingMetadata):
            # Skip preprocessing metadata in analysis listing
            continue

        # Add path info
        meta_dict = metadata.to_dict()
        meta_dict["metadata_path"] = str(json_path)

        # Try to find associated parquet
        parquet_path = json_path.with_suffix(".parquet")
        if not parquet_path.exists() and isinstance(metadata, AnalysisMetadata):
            parquet_path = json_path.parent / f"{metadata.analysis_id}.parquet"
        if parquet_path.exists():
            meta_dict["parquet_path"] = str(parquet_path)

        analysis.append(meta_dict)

    return analysis
