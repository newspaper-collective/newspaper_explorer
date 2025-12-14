"""
Source configuration models.

Pydantic models for validating newspaper source configurations from JSON files.
"""

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourcePart(BaseModel):
    """Configuration for a downloadable dataset part."""

    name: str = Field(description="Part identifier (e.g., 'dertag_1900-1902')")
    url: HttpUrl = Field(description="Download URL")
    years: Optional[str] = Field(default=None, description="Year range for this part")
    md5: str = Field(description="MD5 checksum for validation")
    size: Optional[str] = Field(default=None, description="Human-readable size (e.g., '1.4 GB')")
    description: Optional[str] = Field(default=None, description="Part description")


class SourceMetadata(BaseModel):
    """Metadata about the newspaper source."""

    newspaper_title: str = Field(description="Full newspaper title")
    language: Literal["de", "en"] = Field(description="Primary language code")
    years_available: str = Field(description="Year range (e.g., '1900-1920')")
    zdb_source_id: Optional[str] = Field(
        default=None, description="ZDB (Zeitschriftendatenbank) source identifier"
    )
    publisher: Optional[str] = Field(default=None, description="Publisher name")
    location: Optional[str] = Field(default=None, description="Publication location")
    frequency: Optional[str] = Field(
        default=None, description="Publication frequency (e.g., 'daily')"
    )
    description: Optional[str] = Field(default=None, description="Source description")
    format: Optional[str] = Field(default=None, description="Data format (e.g., 'ALTO XML')")
    info: Optional[str] = Field(
        default=None, description="Additional information about the newspaper"
    )
    citation: Optional[str] = Field(default=None, description="Citation for the dataset")


class LoadingConfig(BaseModel):
    """Configuration for data loading."""

    pattern: str = Field(
        description="Glob pattern for finding XML files (e.g., '**/fulltext/*.xml')"
    )
    compression: Optional[Literal["zstd", "gzip", "snappy", "lz4", "brotli"]] = Field(
        default=None, description="Compression format"
    )
    output_format: Literal["parquet", "csv"] = Field(default="parquet", description="Output format")
    text_encoding: Literal["utf-8", "latin-1", "iso-8859-1"] = Field(
        default="utf-8", description="Text encoding"
    )


class SourceConfig(BaseModel):
    """
    Complete validated configuration for a newspaper source.

    This schema validates source configuration JSON files from data/sources/*.json
    """

    dataset_name: str = Field(description="Dataset identifier (e.g., 'der_tag')")
    data_type: Literal["xml_ocr", "pdf", "txt", "json"] = Field(
        description="Data type (e.g., 'xml_ocr', 'pdf')"
    )
    metadata: SourceMetadata = Field(description="Newspaper metadata")
    loading: LoadingConfig = Field(description="Loading configuration")
    parts: list[SourcePart] = Field(description="Downloadable parts")

    # Optional fields
    collection_id: Optional[str] = Field(default=None, description="Zenodo collection ID")
    collection_url: Optional[HttpUrl] = Field(default=None, description="Zenodo collection URL")
    description: Optional[str] = Field(default=None, description="Dataset description")
    source_provider: Optional[str] = Field(
        default=None, description="Provider/institution of the source"
    )
    license: Optional[str] = Field(default=None, description="License for the dataset")
    fixes: Optional[dict[str, Any]] = Field(default=None, description="Known data issues and fixes")
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @field_validator("parts")
    @classmethod
    def validate_parts_not_empty(cls, v: list[SourcePart]) -> list[SourcePart]:
        """Ensure at least one part exists."""
        if not v:
            raise ValueError("At least one part must be defined")
        return v

    def get_part_by_name(self, part_name: str) -> Optional[SourcePart]:
        """Get a specific part by name."""
        for part in self.parts:
            if part.name == part_name:
                return part
        return None

    def get_total_parts_count(self) -> int:
        """Get total number of parts."""
        return len(self.parts)

    def get_year_range(self) -> tuple[int, int]:
        """
        Get the valid year range from metadata.years_available.

        Returns:
            Tuple of (min_year, max_year) parsed from 'YYYY-YYYY' format.
            Falls back to (1800, 2100) if parsing fails.

        Example:
            >>> config = SourceConfig.from_json_file(path)
            >>> config.get_year_range()
            (1900, 1920)
        """
        try:
            start, end = self.metadata.years_available.split("-")
            return int(start), int(end)
        except (ValueError, AttributeError):
            # Fallback to safe defaults if parsing fails
            return 1800, 2100

    @classmethod
    def from_json_file(cls, path: Path) -> "SourceConfig":
        """Load and validate source config from JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: Path) -> None:
        """Save source config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


class SourceStatus(BaseModel):
    """
    Complete status information for a newspaper source.

    Used by the info command to report on download, extraction, parsing,
    aggregation, and image download status.
    """

    source_name: str = Field(..., description="Source identifier")

    # Raw XML status
    has_raw_xml: bool = Field(..., description="Whether raw XML directory exists with files")
    xml_file_count: int = Field(..., description="Number of XML files found")
    raw_dir: str = Field(..., description="Path to raw XML directory")

    # Parsed data status
    has_parsed_data: bool = Field(..., description="Whether parsed parquet file exists")
    parsed_row_count: int = Field(0, description="Number of rows in parsed data")
    parsed_file_count: int = Field(0, description="Number of unique files parsed")
    parsing_coverage_pct: Optional[float] = Field(
        None, description="Percentage of XML files parsed"
    )
    parsed_date_range: Optional[tuple[str, str]] = Field(
        None, description="Min/max dates in parsed data"
    )
    parsed_size_mb: float = Field(0, description="Size of parsed parquet file in MB")
    output_file: str = Field(..., description="Path to parsed parquet file")

    # Aggregated data status
    has_aggregated_data: bool = Field(..., description="Whether aggregated textblocks exist")
    aggregated_row_count: int = Field(0, description="Number of aggregated text blocks")
    aggregated_size_mb: float = Field(0, description="Size of aggregated parquet file in MB")
    textblocks_path: str = Field(..., description="Path to textblocks parquet file")

    # Image status
    has_images: bool = Field(..., description="Whether images directory exists with files")
    image_count: int = Field(0, description="Number of downloaded images")
    images_expected: int = Field(0, description="Total images expected from METS files")
    image_coverage_pct: Optional[float] = Field(
        None, description="Percentage of expected images downloaded"
    )
    total_size_gb: float = Field(0, description="Total size of images in GB")
    image_year_range: Optional[tuple[int, int]] = Field(
        None, description="Min/max years in image collection"
    )
    images_dir: str = Field(..., description="Path to images directory")
    has_image_index: bool = Field(False, description="Whether image index exists")
