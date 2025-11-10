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
    >>> from newspaper_explorer.data.utils.metadata import AnalysisMetadata, save_metadata
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class AnalysisMetadata(BaseModel):
    """
    Comprehensive metadata for analysis results.

    This Pydantic model captures all information needed to:
    1. Reproduce the analysis
    2. Understand the method and parameters
    3. Link back to source data via foreign keys
    4. Track provenance and versioning
    """

    # Required core fields
    analysis_type: str = Field(
        ..., description="Type of analysis (entities, topics, emotions, keywords, layout)"
    )
    method_type: str = Field(
        ..., description="Method category (gliner, llm, transformer, yolo, keybert)"
    )
    model_name: str = Field(..., description="Specific model identifier")
    source: str = Field(..., description="Source dataset name (e.g., 'der_tag')")
    parameters: Dict[str, Any] = Field(
        ..., description="All analysis parameters (thresholds, batch sizes, etc.)"
    )

    # Auto-generated/optional fields
    analysis_id: Optional[str] = Field(
        None, description="Unique identifier (auto-generated if not provided)"
    )
    model_version: Optional[str] = Field(None, description="Model version string")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="ISO timestamp of creation"
    )
    completed_at: Optional[str] = Field(None, description="ISO timestamp of completion")
    duration_seconds: Optional[float] = Field(None, description="Processing time in seconds")

    # Input/output tracking
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input dataset information"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Output results information"
    )

    # Status tracking
    status: str = Field(default="completed", description="Status: completed, failed, in_progress")
    error_message: Optional[str] = Field(None, description="Error details if status is failed")

    def model_post_init(self, __context: Any) -> None:
        """Generate analysis_id if not provided."""
        if self.analysis_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.analysis_id = (
                f"{self.method_type}_{self.model_name}_{timestamp}".replace(".", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status field."""
        allowed = ["completed", "failed", "in_progress"]
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}, got: {v}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisMetadata":
        """Create from dictionary."""
        return cls.model_validate(data)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "analysis_type": "entities",
                "method_type": "gliner",
                "model_name": "multi-v2.1",
                "source": "der_tag",
                "parameters": {
                    "threshold": 0.2,
                    "batch_size": 16,
                    "labels": ["Person", "Organisation", "Ort"],
                },
                "analysis_id": "gliner_multi_v2_1_20251109_004150",
                "created_at": "2025-11-09T00:41:50.008260",
                "duration_seconds": 22.33,
                "status": "completed",
                "input_data": {"row_count": 100000, "date_range": ["1901-01-08", "1920-12-31"]},
                "output_data": {"row_count": 5420},
            }
        }


def save_metadata(
    metadata: AnalysisMetadata,
    parquet_path: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save metadata as JSON alongside parquet file.

    Creates a JSON file with the same base name as the parquet file.

    Args:
        metadata: Metadata object to save
        parquet_path: Path to the parquet file (or its parent directory)
        filename: Optional custom filename for metadata (defaults to matching parquet name)

    Returns:
        Path to saved metadata file

    Example:
        >>> metadata = AnalysisMetadata(...)
        >>> save_metadata(metadata, Path("results/entities/analysis.parquet"))
        PosixPath('results/entities/analysis.json')
    """
    # Determine output path
    if parquet_path.is_dir():
        # If directory provided, use analysis_id as base name
        base_name = filename or f"{metadata.analysis_id}.json"
        json_path = parquet_path / base_name
    else:
        # If file provided, replace extension
        if filename:
            json_path = parquet_path.parent / filename
        else:
            json_path = parquet_path.with_suffix(".json")

    # Ensure directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as formatted JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata to {json_path}")
    return json_path


def load_metadata(metadata_path: Union[Path, str]) -> AnalysisMetadata:
    """
    Load metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file

    Returns:
        AnalysisMetadata object

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

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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


def extract_input_stats(
    df: pl.DataFrame,
    id_column: str = "line_id",
    date_column: str = "date",
) -> Dict[str, Any]:
    """
    Extract statistics from input DataFrame for metadata.

    Args:
        df: Input DataFrame
        id_column: Name of ID column
        date_column: Name of date column

    Returns:
        Dictionary with input statistics

    Example:
        >>> df = pl.read_parquet("data/raw/der_tag/text/lines.parquet")
        >>> stats = extract_input_stats(df)
        >>> print(stats["row_count"])
        100000
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

    return stats


def extract_output_stats(df: pl.DataFrame) -> Dict[str, Any]:
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


def update_metadata_status(
    metadata_path: Path,
    status: str,
    completed_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
    output_data: Optional[Dict[str, Any]] = None,
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


def list_analyses(
    results_dir: Path,
    analysis_type: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all analyses with their metadata.

    Args:
        results_dir: Results directory (e.g., "results/")
        analysis_type: Filter by analysis type (entities, topics, etc.)
        source: Filter by source (e.g., "der_tag")

    Returns:
        List of metadata dictionaries

    Example:
        >>> from newspaper_explorer.config.base import get_config
        >>> config = get_config()
        >>> analyses = list_analyses(
        ...     config.results_dir,
        ...     analysis_type="entities",
        ...     source="der_tag"
        ... )
        >>> for analysis in analyses:
        ...     print(f"{analysis['analysis_id']}: {analysis['model_name']}")
    """
    results_dir = Path(results_dir)
    analyses = []

    # Pattern: results/{source}/{analysis_type}/{analysis_id}/metadata.json
    pattern = "**/*.json"
    if source:
        pattern = f"{source}/**/*.json"

    for json_path in results_dir.glob(pattern):
        # Skip non-metadata files
        if json_path.name not in ["metadata.json", "*.json"]:
            # Check if it's a metadata file by trying to load it
            try:
                metadata = load_metadata(json_path)
            except (json.JSONDecodeError, KeyError):
                continue
        else:
            try:
                metadata = load_metadata(json_path)
            except Exception:
                continue

        # Apply filters
        if analysis_type and metadata.analysis_type != analysis_type:
            continue
        if source and metadata.source != source:
            continue

        # Add path info
        meta_dict = metadata.to_dict()
        meta_dict["metadata_path"] = str(json_path)

        # Try to find associated parquet
        parquet_path = json_path.with_suffix(".parquet")
        if not parquet_path.exists():
            parquet_path = json_path.parent / f"{metadata.analysis_id}.parquet"
        if parquet_path.exists():
            meta_dict["parquet_path"] = str(parquet_path)

        analyses.append(meta_dict)

    return analyses


def save_analysis_results(
    results_df: pl.DataFrame,
    metadata: AnalysisMetadata,
    results_base_dir: Path,
    results_filename: Optional[str] = None,
) -> Dict[str, Path]:
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
        >>> from newspaper_explorer.data.utils.metadata import save_analysis_results
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
