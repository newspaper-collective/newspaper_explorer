"""
Processing metadata models.

Models for tracking analysis and preprocessing pipeline execution.
These are "sidecar" metadata models that track how data was processed,
enabling reproducibility and provenance tracking.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Constants for ID generation
_MAX_STEPS_IN_ID = 3  # Maximum number of steps to include in preprocessing ID


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
    parameters: dict[str, Any] = Field(
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
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input dataset information"
    )
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Output results information"
    )

    # Granularity tracking
    granularity: Optional[str] = Field(
        default=None,
        description="Data granularity level: line, textblock, page, issue",
    )

    # Status tracking
    status: str = Field(default="completed", description="Status: completed, failed, in_progress")
    error_message: Optional[str] = Field(None, description="Error details if status is failed")

    def model_post_init(self, __context: object) -> None:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisMetadata":
        """Create from dictionary."""
        return cls.model_validate(data)

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class PreprocessingResult(BaseModel):
    """
    Result from a preprocessing operation.

    Contains the processed DataFrame, metadata, and file paths.
    Used by CLI and FastAPI to receive preprocessing results.

    Example:
        >>> from newspaper_explorer.data.preprocessing.pipeline import run_preprocessing
        >>> result = run_preprocessing(source="der_tag", steps=["normalize-unicode"])
        >>> print(f"Processed {result.input_rows} → {result.output_rows} rows")
        >>> print(f"Output: {result.results_path}")
    """

    # Results (df excluded from serialization - access via results_path)
    metadata: "PreprocessingMetadata"
    output_dir: Path
    results_path: Path
    metadata_path: Path

    # Statistics for display
    input_rows: int = Field(ge=0, description="Number of input rows")
    output_rows: int = Field(ge=0, description="Number of output rows")
    duration_seconds: float = Field(ge=0, description="Processing time in seconds")
    file_size_bytes: int = Field(ge=0, description="Output file size in bytes")

    # Sample for preview (first row original/processed text)
    sample_original: Optional[str] = Field(None, description="Sample original text")
    sample_processed: Optional[str] = Field(None, description="Sample processed text")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (excludes df)."""
        return {
            "metadata": self.metadata.to_dict(),
            "output_dir": str(self.output_dir),
            "results_path": str(self.results_path),
            "metadata_path": str(self.metadata_path),
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "sample_original": self.sample_original,
            "sample_processed": self.sample_processed,
        }


class PreprocessingMetadata(BaseModel):
    """
    Metadata for preprocessing pipeline results.

    Tracks all preprocessing steps applied to text data, enabling:
    1. Reproducibility of preprocessing pipelines
    2. Chaining of preprocessing steps (preprocessed input → new preprocessing)
    3. Full provenance tracking in analysis results

    Example:
        >>> metadata = PreprocessingMetadata(
        ...     source="der_tag",
        ...     steps=["normalize", "lowercase"],
        ...     parameters={"text_column": "text", "output_column": "text_processed"}
        ... )
    """

    # Required core fields
    source: str = Field(..., description="Source dataset name (e.g., 'der_tag')")
    steps: list[str] = Field(..., description="List of preprocessing steps applied in order")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Processing parameters (text_column, batch_size, etc.)",
    )

    # Auto-generated/optional fields
    preprocessing_id: Optional[str] = Field(
        default=None, description="Unique identifier (auto-generated if not provided)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of creation",
    )
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp of completion")
    duration_seconds: Optional[float] = Field(
        default=None, description="Processing time in seconds"
    )

    # Input/output tracking
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input dataset information"
    )
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Output dataset information"
    )

    # Chaining support - track previous preprocessing if input was preprocessed
    previous_preprocessing: Optional[dict[str, Any]] = Field(
        default=None,
        description="Previous preprocessing metadata if input was already preprocessed",
    )

    # Status tracking
    status: str = Field(default="completed", description="Status: completed, failed, in_progress")
    error_message: Optional[str] = Field(
        default=None, description="Error details if status is failed"
    )

    def model_post_init(self, __context: object) -> None:
        """Generate preprocessing_id if not provided."""
        if self.preprocessing_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            steps_str = "_".join(self.steps[:_MAX_STEPS_IN_ID])
            remaining = len(self.steps) - _MAX_STEPS_IN_ID
            if remaining > 0:
                steps_str = f"{steps_str}_plus{remaining}"
            self.preprocessing_id = f"{steps_str}_{timestamp}".replace("-", "_")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status field."""
        allowed = ["completed", "failed", "in_progress"]
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}, got: {v}")
        return v

    def get_all_steps(self) -> list[str]:
        """
        Get complete list of all steps including previous preprocessing.

        Returns:
            Flat list of all steps in chronological order
        """
        all_steps: list[str] = []
        if self.previous_preprocessing:
            prev_steps = self.previous_preprocessing.get("steps", [])
            if isinstance(prev_steps, list):
                all_steps.extend(prev_steps) # type: ignore
        all_steps.extend(self.steps)
        return all_steps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingMetadata":
        """Create from dictionary."""
        return cls.model_validate(data)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "der_tag",
                "steps": ["normalize", "lowercase", "remove-stopwords"],
                "parameters": {
                    "text_column": "text",
                    "output_column": "text_processed",
                    "batch_size": 32,
                },
                "preprocessing_id": "normalize_lowercase_remove_stopwords_20251110_120000",
                "created_at": "2025-11-10T12:00:00.000000",
                "duration_seconds": 45.2,
                "status": "completed",
                "input_data": {"row_count": 100000, "date_range": ["1901-01-08", "1920-12-31"]},
                "output_data": {"row_count": 98500},
            }
        }
    )


class AggregationMetadata(BaseModel):
    """
    Metadata for text aggregation operations.

    Tracks how line-level data was aggregated into higher-level text units
    (textblocks, pages, issues), enabling reproducibility and provenance tracking.

    Example:
        >>> metadata = AggregationMetadata(
        ...     source="der_tag",
        ...     aggregation_type="textblock",
        ...     group_by=["text_block_id"],
        ...     input_parquet="data/raw/der_tag/text/der_tag_lines.parquet",
        ... )
    """

    # Required core fields
    source: str = Field(..., description="Source dataset name (e.g., 'der_tag')")
    aggregation_type: str = Field(
        ...,
        description="Type of aggregation: line, textblock, page, issue",
    )
    group_by: list[str] = Field(
        ..., description="Columns used for grouping (e.g., ['text_block_id'])"
    )
    sort_by: list[str] = Field(
        default_factory=lambda: ["y", "x"],
        description="Columns used for sorting within groups",
    )

    # Input/output tracking
    input_parquet: str = Field(..., description="Path to source parquet file")
    output_parquet: Optional[str] = Field(None, description="Path to output parquet file")
    input_row_count: int = Field(..., ge=0, description="Number of input rows (lines)")
    output_row_count: Optional[int] = Field(
        None, ge=0, description="Number of output rows (aggregated units)"
    )

    # Auto-generated fields
    aggregation_id: Optional[str] = Field(
        None, description="Unique identifier (auto-generated if not provided)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of creation",
    )
    duration_seconds: Optional[float] = Field(None, description="Processing time in seconds")

    # Status tracking
    status: str = Field(default="completed", description="Status: completed, failed, in_progress")
    error_message: Optional[str] = Field(None, description="Error details if status is failed")

    def model_post_init(self, __context: object) -> None:
        """Generate aggregation_id if not provided."""
        if self.aggregation_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.aggregation_id = f"{self.aggregation_type}_{timestamp}"

    @field_validator("aggregation_type")
    @classmethod
    def validate_aggregation_type(cls, v: str) -> str:
        """Validate aggregation_type field."""
        allowed = ["line", "textblock", "page", "issue"]
        if v not in allowed:
            raise ValueError(f"aggregation_type must be one of {allowed}, got: {v}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status field."""
        allowed = ["completed", "failed", "in_progress"]
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}, got: {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AggregationMetadata":
        """Create from dictionary."""
        return cls.model_validate(data)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "der_tag",
                "aggregation_type": "textblock",
                "group_by": ["text_block_id"],
                "sort_by": ["y", "x"],
                "input_parquet": "data/raw/der_tag/text/der_tag_lines.parquet",
                "output_parquet": "data/processed/der_tag/text/textblocks.parquet",
                "input_row_count": 50000000,
                "output_row_count": 23975727,
                "aggregation_id": "textblock_20251207_120000",
                "created_at": "2025-12-07T12:00:00.000000",
                "duration_seconds": 120.5,
                "status": "completed",
            }
        }
    )
