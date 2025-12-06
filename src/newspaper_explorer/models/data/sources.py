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
