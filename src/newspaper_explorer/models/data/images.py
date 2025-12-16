"""
Image-related data models.

Models for image validation, METS image references, and image indexing.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


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


class ImageIndexRecord(BaseModel):
    """
    Metadata record for a single newspaper page image in the index.

    This model represents the complete metadata for an indexed image,
    including file information, layout coordinates, and METS metadata.
    """

    # File information
    image_path: str  # Relative path from images directory
    filename: str  # Image filename
    file_size_bytes: Optional[int] = None
    file_exists: bool = True

    # Foreign keys (linking to source and issue data)
    source_id: str  # FK: Source identifier (e.g., "der_tag")
    issue_id: Optional[str] = None  # FK: {source}_{date}_{issue}_{daily}
    page_id: Optional[str] = None  # FK: {source}_{date}_{issue}_{daily}_{page}

    # Date information
    year: int
    month: int
    day: int
    date: str  # YYYY-MM-DD format

    # Page information
    page_number: Optional[int] = None  # Page number in the issue
    edition: Optional[int] = None  # Edition of the day (1=morning, 2=midday, 3=evening)
    issue_number: Optional[int] = None  # Issue number from METS

    # Image dimensions
    width: Optional[int] = None  # Actual image width in pixels
    height: Optional[int] = None  # Actual image height in pixels
    alto_width: Optional[int] = None  # Image width in ALTO coordinate space
    alto_height: Optional[int] = None  # Image height in ALTO coordinate space

    # METS metadata (denormalized for convenience)
    newspaper_title: Optional[str] = None  # e.g., "Der Tag"
    year_volume: Optional[str] = None  # e.g., "Jahrgang 1902"
    page_count: Optional[int] = None  # Total pages in issue


class ImageStats(BaseModel):
    """Statistics about indexed images for a source."""

    total_images: int
    total_images_expected: int
    total_size_bytes: int
    total_size_gb: float
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    avg_file_size_mb: float
