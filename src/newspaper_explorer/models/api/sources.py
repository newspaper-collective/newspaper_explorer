"""
API models for source information and statistics.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisResultSummary(BaseModel):
    """Summary of available analysis results for a specific type"""

    count: int  # Total number of files
    parquet: int  # Number of parquet files
    csv: int  # Number of CSV files
    metadata: Optional[dict[str, Any]] = None  # Metadata from JSON files


class SourceInfo(BaseModel):
    """Source configuration information"""

    model_config = {"use_enum_values": True}

    name: str
    dataset_name: str
    data_type: str
    metadata: dict[str, Any]
    loading: dict[str, Any]
    has_text: bool
    has_entities: bool
    has_keywords: bool
    has_layout: bool
    has_topics: bool
    has_emotions: bool
    has_concepts: bool
    has_images: bool
    total_archive_size: Optional[str] = Field(
        default=None, description="Compressed XML archive size"
    )
    image_size: Optional[str] = Field(default=None, description="Total size of downloaded images")
    image_count: Optional[int] = Field(default=None, description="Number of downloaded images")
    analysis_results: dict[str, AnalysisResultSummary] = Field(default_factory=dict)


class SourceStats(BaseModel):
    """Source statistics"""

    total_issues: int
    total_pages: int
    total_lines: int
    total_blocks: int
    total_images: int
    date_range: tuple[str, str]
    years_available: list[int]
