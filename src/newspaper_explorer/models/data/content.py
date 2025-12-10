"""
Core data models for newspaper content.

Contains Pydantic models for parsed ALTO/METS data structures.
These are the foundational data types used throughout the system.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, computed_field


class IssueMetadata(BaseModel):
    """Metadata for a complete newspaper issue from METS file"""

    filename: str
    date: Optional[datetime] = None
    issue_number: Optional[int] = None
    issue_string: Optional[str] = None  # e.g., "Nr. 415, 05. September 1902"
    edition: Optional[int] = None  # Edition of the day (1=morning, 2=midday, 3=evening)
    year_volume: Optional[str] = None  # e.g., "Jahrgang 1902"
    page_count: Optional[int] = None
    newspaper_title: Optional[str] = None
    newspaper_id: Optional[str] = None  # ZDB ID
    publisher: Optional[str] = None
    language: Optional[str] = None


class TextLine(BaseModel):
    """
    Represents a single text line from ALTO XML with enriched metadata.

    Note: Uses unified ID system with proper foreign key hierarchy:
    source_id -> issue_id -> page_id -> text_block_id -> line_id
    """

    # Primary key and data
    line_id: str  # PRIMARY: {source}_{date}_{issue}_{daily}_{page}_{block}_{line}
    text: str  # OCR text content (raw, as printed with hyphens)
    text_dehyphenated_ocr: Optional[str] = (
        None  # SUBS_CONTENT from ALTO (OCR's dehyphenation suggestion, None if not present)
    )

    # Foreign keys (for linking and querying)
    source_id: str  # FK: Source identifier (e.g., "der_tag")
    issue_id: str  # FK: {source}_{date}_{issue}_{daily}
    page_id: str  # FK: {source}_{date}_{issue}_{daily}_{page}
    text_block_id: str  # FK: {page_id}_{block_id}

    # Original reference (for debugging)
    filename: str  # Source ALTO XML filename

    # Date information
    date: Optional[datetime] = None

    # Layout coordinates
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # From filename parsing (denormalized for convenience)
    issue_number: Optional[int] = None  # Sequential publication number (may differ from METS)
    edition: Optional[int] = None  # Edition of the day (1=morning, 2=midday, 3=evening)
    page_number: Optional[int] = None  # Last number in filename (e.g., 005)

    # From METS metadata (denormalized for convenience)
    year_volume: Optional[str] = None  # e.g., "Jahrgang 1902"
    page_count: Optional[int] = None  # Total pages in issue
    newspaper_title: Optional[str] = None  # e.g., "Der Tag"

    @computed_field
    def year(self) -> Optional[int]:
        """Extract year from date"""
        return self.date.year if self.date else None

    @computed_field
    def month(self) -> Optional[int]:
        """Extract month from date"""
        return self.date.month if self.date else None

    @computed_field
    def day(self) -> Optional[int]:
        """Extract day from date"""
        return self.date.day if self.date else None
