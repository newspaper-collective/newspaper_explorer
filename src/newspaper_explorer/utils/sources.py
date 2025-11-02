"""Source configuration management with Pydantic validation.

This module provides validated source configuration management for newspaper data sources.
All source-related operations (download, parse, analyze) rely on these utilities.

Key Classes & Functions:
    - SourceConfig: Pydantic model for validated source configurations
    - list_available_sources(): Get all available source names
    - load_source_config(): Load and validate a source's JSON configuration
    - get_source_paths(): Calculate standard paths for a source's data

Example:
    >>> from newspaper_explorer.utils.sources import load_source_config, SourceConfig
    >>>
    >>> # Load and validate source configuration
    >>> config = load_source_config("der_tag")
    >>> print(config.metadata.newspaper_title)
    'Der Tag'
    >>>
    >>> # Get standard paths
    >>> paths = get_source_paths(config)
    >>> print(paths["output_file"])
    PosixPath('data/raw/der_tag/text/der_tag_lines.parquet')
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from natsort import natsorted
from pydantic import BaseModel, Field, field_validator, HttpUrl

from newspaper_explorer.config.base import get_config
from typing import Literal


# ============================================================================
# Pydantic Models for Source Configuration
# ============================================================================


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
    publisher: Optional[str] = Field(default=None, description="Publisher name")
    location: Optional[str] = Field(default=None, description="Publication location")
    frequency: Optional[str] = Field(
        default=None, description="Publication frequency (e.g., 'daily')"
    )
    description: Optional[str] = Field(default=None, description="Source description")
    format: Optional[str] = Field(default=None, description="Data format (e.g., 'ALTO XML')")


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
    parts: List[SourcePart] = Field(description="Downloadable parts")

    # Optional fields
    collection_id: Optional[str] = Field(default=None, description="Zenodo collection ID")
    collection_url: Optional[HttpUrl] = Field(default=None, description="Zenodo collection URL")
    description: Optional[str] = Field(default=None, description="Dataset description")
    fixes: Optional[Dict[str, Any]] = Field(default=None, description="Known data issues and fixes")
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @field_validator("parts")
    @classmethod
    def validate_parts_not_empty(cls, v: List[SourcePart]) -> List[SourcePart]:
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

    @classmethod
    def from_json_file(cls, path: Path) -> "SourceConfig":
        """Load and validate source config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: Path) -> None:
        """Save source config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


# ============================================================================
# Source Configuration Functions
# ============================================================================


def list_available_sources() -> List[str]:
    """
    List all available sources from the sources directory.

    Returns:
        List[str]: Naturally sorted list of source names

    Example:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['der_tag']
    """
    config = get_config()
    sources = []

    if config.sources_dir.exists():
        for source_file in config.sources_dir.glob("*.json"):
            sources.append(source_file.stem)

    return natsorted(sources)


def load_source_config(source_name: str) -> SourceConfig:
    """
    Load and validate source configuration.

    Args:
        source_name: Name of the source (e.g., 'der_tag')

    Returns:
        SourceConfig: Validated Pydantic model

    Raises:
        ValueError: If source not found or validation fails

    Example:
        >>> config = load_source_config("der_tag")
        >>> print(config.metadata.newspaper_title)
        'Der Tag'
        >>> print(config.loading.pattern)
        '**/fulltext/*.xml'
        >>> print(len(config.parts))
        7
    """
    config = get_config()
    source_file = config.sources_dir / f"{source_name}.json"

    if not source_file.exists():
        available = list_available_sources()
        raise ValueError(
            f"Source '{source_name}' not found. Available sources: {', '.join(available)}"
        )

    try:
        return SourceConfig.from_json_file(source_file)
    except Exception as e:
        raise ValueError(f"Invalid source configuration for '{source_name}': {e}")


def get_source_paths(source_config: SourceConfig) -> Dict[str, Path]:
    """
    Get standard paths for a source's data directories and files.

    Args:
        source_config: Validated SourceConfig object

    Returns:
        Dict[str, Path]: Dictionary with paths:
            - raw_dir: Raw XML/OCR files
            - text_dir: Parsed text data
            - images_dir: Downloaded images
            - output_file: Main parquet output

    Example:
        >>> config = load_source_config("der_tag")
        >>> paths = get_source_paths(config)
        >>> print(paths["raw_dir"])
        PosixPath('data/raw/der_tag/xml_ocr')
    """
    config = get_config()
    dataset_name = source_config.dataset_name
    data_type = source_config.data_type

    raw_dir = config.data_dir / "raw" / dataset_name / data_type
    text_dir = config.data_dir / "raw" / dataset_name / "text"
    images_dir = config.data_dir / "raw" / dataset_name / "images"
    output_file = text_dir / f"{dataset_name}_lines.parquet"

    return {
        "raw_dir": raw_dir,
        "text_dir": text_dir,
        "images_dir": images_dir,
        "output_file": output_file,
    }
