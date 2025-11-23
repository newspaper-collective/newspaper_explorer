"""
Pydantic models for data processing.

Centralized location for data-related models used across the data module.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ImageValidationResult(BaseModel):
    """Result of image validation check."""

    is_valid: bool
    file_path: Path
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None


class ImageReference(BaseModel):
    """Reference to an image in METS XML."""

    file_id: str
    url: str
    extension: str = ".jpg"


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisMetadata":
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

    def model_post_init(self, __context: Any) -> None:
        """Generate preprocessing_id if not provided."""
        if self.preprocessing_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            steps_str = "_".join(self.steps[:3])  # First 3 steps for ID
            if len(self.steps) > 3:
                steps_str += f"_plus{len(self.steps) - 3}"
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
        all_steps = []
        if self.previous_preprocessing:
            prev_steps = self.previous_preprocessing.get("steps", [])
            all_steps.extend(prev_steps)
        all_steps.extend(self.steps)
        return all_steps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingMetadata":
        """Create from dictionary."""
        return cls.model_validate(data)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
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
