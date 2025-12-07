"""
Canonical schema definitions for Parquet data stores.

Defines the expected columns, types, and constraints for each data store
in the newspaper_explorer system. These schemas serve as:
1. Documentation of data contracts
2. Validation at data boundaries
3. Single source of truth for column naming

Usage:
    from newspaper_explorer.models.data.schema import SourceLinesSchema, validate_schema

    df = pl.read_parquet("source_lines.parquet")
    validate_schema(df, SourceLinesSchema)

ID Hierarchy:
    source_id (der_tag)
    └── issue_id (der_tag_1902-09-05_415_2)
        └── page_id (der_tag_1902-09-05_415_2_005)
            ├── text_block_id (der_tag_1902-09-05_415_2_005_r_1_1)
            │   └── line_id (der_tag_1902-09-05_415_2_005_r_1_1_TL_1)
            └── detection_id (der_tag_1902-09-05_415_2_005_text_ae5fd8)
"""

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


# =============================================================================
# Common Column Definitions
# =============================================================================


class ForeignKeyColumns(BaseModel):
    """
    Common foreign key columns present in all data stores.

    These columns enable cross-dataset joins and must use consistent formats:
    - source_id: Source name (e.g., "der_tag")
    - issue_id: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
    - page_id: {issue_id}_{page:03d}
    """

    source_id: str = Field(..., description="Source identifier (e.g., 'der_tag')")
    issue_id: str = Field(
        ...,
        description="Issue identifier: {source}_{date}_{issue:03d}_{daily}",
    )
    page_id: str = Field(
        ...,
        description="Page identifier: {issue_id}_{page:03d}",
    )


class MetadataColumns(BaseModel):
    """
    Metadata columns from METS/filename parsing.

    These provide context about the newspaper issue and are
    denormalized for query efficiency.
    """

    date: datetime = Field(..., description="Publication date")
    issue_number: Optional[int] = Field(
        None, description="Sequential publication number (from filename)"
    )
    edition: Optional[int] = Field(
        None, description="Edition of the day (1=morning, 2=midday, 3=evening)"
    )
    page_number: Optional[int] = Field(None, description="Page number within issue")
    newspaper_title: Optional[str] = Field(None, description="Newspaper title (from METS)")
    year_volume: Optional[str] = Field(None, description="Year/volume string (from METS)")
    page_count: Optional[int] = Field(None, description="Total pages in issue (from METS)")


class LayoutCoordinates(BaseModel):
    """
    Bounding box coordinates in ALTO coordinate space.

    All coordinates are integers representing pixels in the
    original ALTO coordinate system.
    """

    x: int = Field(..., description="X coordinate (left edge)")
    y: int = Field(..., description="Y coordinate (top edge)")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


# =============================================================================
# Source Data Schemas
# =============================================================================


class SourceLinesSchema(BaseModel):
    """
    Schema for source_lines.parquet (line-level OCR data).

    Output of DataIngester. One row per text line from ALTO XML.

    File: data/raw/{source}/text/{source}_lines.parquet
    """

    # Primary key
    line_id: str = Field(
        ...,
        description="Line identifier: {text_block_id}_TL_{n}",
    )

    # Foreign keys (required)
    source_id: str = Field(..., description="Source identifier (e.g., 'der_tag')")
    issue_id: str = Field(..., description="Issue identifier")
    page_id: str = Field(..., description="Page identifier")
    text_block_id: str = Field(..., description="Text block identifier")

    # Content
    text: str = Field(..., description="OCR text content")
    filename: str = Field(..., description="Source ALTO XML filename")

    date: datetime = Field(..., description="Publication date")

    # Layout coordinates (required)
    x: int = Field(..., description="X coordinate (ALTO units)")
    y: int = Field(..., description="Y coordinate (ALTO units)")
    width: int = Field(..., description="Width (ALTO units)")
    height: int = Field(..., description="Height (ALTO units)")

    # Metadata from filename (optional but typically present)
    issue_number: Optional[int] = Field(None, description="Issue number")
    edition: Optional[int] = Field(None, description="Daily issue count (1, 2, ...)")
    page_number: Optional[int] = Field(None, description="Page number")

    # Metadata from METS (optional)
    newspaper_title: Optional[str] = Field(None, description="Newspaper title")
    year_volume: Optional[str] = Field(None, description="Year/volume string")
    page_count: Optional[int] = Field(None, description="Total pages in issue")

    # Computed columns (may be materialized or computed at query time)
    year: Optional[int] = Field(None, description="Year (from date)")
    month: Optional[int] = Field(None, description="Month (from date)")
    day: Optional[int] = Field(None, description="Day (from date)")


class TextBlocksSchema(BaseModel):
    """
    Schema for textblocks.parquet (aggregated text blocks).

    Output of aggregation.py. One row per text block (lines concatenated).

    File: data/processed/{source}/text/textblocks.parquet
    """

    # Primary key
    text_block_id: str = Field(..., description="Text block identifier")

    # Foreign keys (required)
    source_id: str = Field(..., description="Source identifier (e.g., 'der_tag')")
    issue_id: str = Field(..., description="Issue identifier")
    page_id: str = Field(..., description="Page identifier")

    # Content
    text: str = Field(..., description="Concatenated text from all lines in block")

    date: datetime = Field(..., description="Publication date")

    # Layout coordinates (bounding box of all lines)
    x: int = Field(..., description="Min X coordinate")
    y: int = Field(..., description="Min Y coordinate")
    width: int = Field(..., description="Total width")
    height: int = Field(..., description="Total height")

    issue_number: Optional[int] = Field(None, description="Issue number")
    edition: Optional[int] = Field(None, description="Daily issue count")
    page_number: Optional[int] = Field(None, description="Page number")
    newspaper_title: Optional[str] = Field(None, description="Newspaper title")
    year_volume: Optional[str] = Field(None, description="Year/volume")
    page_count: Optional[int] = Field(None, description="Total pages in issue")

    # Aggregation metadata
    line_count: Optional[int] = Field(None, description="Number of lines in block")


class ImageIndexSchema(BaseModel):
    """
    Schema for image_index.parquet (page image metadata).

    Output of ImageIndexer. One row per page image.

    File: data/raw/{source}/image_index.parquet
    """

    # Primary key (matches text data page_id)
    page_id: str = Field(..., description="Page identifier (matches text data)")

    # Foreign keys (required)
    source_id: str = Field(..., description="Source identifier (e.g., 'der_tag')")
    issue_id: str = Field(..., description="Issue identifier")

    # Image file info
    image_path: str = Field(..., description="Relative path to image file")
    filename: str = Field(..., description="Image filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    file_exists: bool = Field(..., description="Whether file exists on disk")

    # Date components
    date: str = Field(..., description="Date string (YYYY-MM-DD)")
    year: int = Field(..., description="Year")
    month: int = Field(..., description="Month")
    day: int = Field(..., description="Day")

    # Page info
    page_number: int = Field(..., description="Page number")

    # Dimensions (optional - may be null if not extracted)
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    alto_width: Optional[int] = Field(None, description="ALTO coordinate space width")
    alto_height: Optional[int] = Field(None, description="ALTO coordinate space height")

    # Metadata from METS (optional)
    newspaper_title: Optional[str] = Field(None, description="Newspaper title")
    year_volume: Optional[str] = Field(None, description="Year/volume")
    page_count: Optional[int] = Field(None, description="Total pages in issue")
    issue_number: Optional[int] = Field(None, description="Issue number")
    edition: Optional[int] = Field(None, description="Daily issue count")


class PreprocessedSchema(BaseModel):
    """
    Schema for preprocessed text data.

    Output of preprocessing pipeline. Preserves all FK columns from source.
    Can be line-level or block-level depending on input.

    File: data/processed/{source}/text/{preprocessing_id}/preprocessed.parquet
    """

    # Primary key (one of these, depending on input granularity)
    line_id: Optional[str] = Field(None, description="Line ID (if line-level input)")
    text_block_id: Optional[str] = Field(None, description="Block ID (if block-level input)")

    # Foreign keys (always preserved)
    source_id: str = Field(..., description="Source identifier")
    issue_id: str = Field(..., description="Issue identifier")
    page_id: str = Field(..., description="Page identifier")

    # Content
    text: str = Field(..., description="Original text (preserved)")
    text_processed: str = Field(..., description="Preprocessed text (output)")

    date: datetime = Field(..., description="Publication date")

    # All other metadata columns from source are preserved but not required


# =============================================================================
# Analysis Result Schemas
# =============================================================================
# Note: Analysis-specific schemas are in models/analysis/*.py
# They all share the common FK columns (source_id, issue_id, page_id)
# and add analysis-specific columns.


# =============================================================================
# Schema Validation Utilities
# =============================================================================


def get_required_columns(schema_class: type[BaseModel]) -> set[str]:
    """
    Get required column names from a schema class.

    Args:
        schema_class: Pydantic model class

    Returns:
        Set of required column names
    """
    required = set()
    for field_name, field_info in schema_class.model_fields.items():
        if field_info.is_required():
            required.add(field_name)
    return required


def get_optional_columns(schema_class: type[BaseModel]) -> set[str]:
    """
    Get optional column names from a schema class.

    Args:
        schema_class: Pydantic model class

    Returns:
        Set of optional column names
    """
    optional = set()
    for field_name, field_info in schema_class.model_fields.items():
        if not field_info.is_required():
            optional.add(field_name)
    return optional


def get_all_columns(schema_class: type[BaseModel]) -> set[str]:
    """
    Get all column names from a schema class.

    Args:
        schema_class: Pydantic model class

    Returns:
        Set of all column names (required + optional)
    """
    return set(schema_class.model_fields.keys())


def validate_schema(
    df: "pl.DataFrame",
    schema_class: type[BaseModel],
    *,
    strict: bool = False,
) -> list[str]:
    """
    Validate a Polars DataFrame against a schema.

    Checks that required columns exist. Does not validate types or values.

    Args:
        df: Polars DataFrame to validate
        schema_class: Pydantic model class defining expected schema
        strict: If True, raise ValueError on missing required columns.
                If False, return list of warning messages.

    Returns:
        List of warning messages (empty if all required columns present)

    Raises:
        ValueError: If strict=True and required columns are missing

    Example:
        >>> from newspaper_explorer.models.data.schema import (
        ...     SourceLinesSchema, validate_schema
        ... )
        >>> warnings = validate_schema(df, SourceLinesSchema)
        >>> if warnings:
        ...     for w in warnings:
        ...         logger.warning(w)
    """
    warnings = []
    df_columns = set(df.columns)

    required = get_required_columns(schema_class)
    optional = get_optional_columns(schema_class)

    # Check required columns
    missing_required = required - df_columns
    if missing_required:
        msg = f"Missing required columns for {schema_class.__name__}: {sorted(missing_required)}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    # Note missing optional columns (just informational)
    missing_optional = optional - df_columns
    if missing_optional:
        warnings.append(
            f"Missing optional columns for {schema_class.__name__}: {sorted(missing_optional)}"
        )

    # Note unknown columns (not necessarily a problem)
    all_schema_cols = required | optional
    unknown = df_columns - all_schema_cols
    if unknown:
        warnings.append(f"Extra columns not in {schema_class.__name__} schema: {sorted(unknown)}")

    return warnings


def validate_foreign_keys(df: "pl.DataFrame") -> list[str]:
    """
    Validate that common FK columns exist and have consistent format.

    Args:
        df: Polars DataFrame to validate

    Returns:
        List of warning messages
    """
    warnings = []
    df_columns = set(df.columns)

    # Check FK columns exist
    fk_columns = {"source_id", "issue_id", "page_id"}
    missing = fk_columns - df_columns
    if missing:
        warnings.append(f"Missing foreign key columns: {sorted(missing)}")
        return warnings  # Can't validate format without columns

    # Check format consistency (sample first row)
    if df.height > 0:
        sample = df.head(1).to_dicts()[0]

        source_id = sample.get("source_id", "")
        issue_id = sample.get("issue_id", "")
        page_id = sample.get("page_id", "")

        # issue_id should start with source_id
        if issue_id and source_id and not issue_id.startswith(source_id):
            warnings.append(f"issue_id '{issue_id}' does not start with source_id '{source_id}'")

        # page_id should start with issue_id
        if page_id and issue_id and not page_id.startswith(issue_id):
            warnings.append(f"page_id '{page_id}' does not start with issue_id '{issue_id}'")

    return warnings
