"""
Data schemas for layout analysis results.

Defines the structure for detected elements, matched headlines, and reconstructed articles.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculate width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate height."""
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        """Calculate center X coordinate."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Calculate center Y coordinate."""
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Calculate area."""
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """
        Calculate Intersection over Union with another box.

        Args:
            other: Another bounding box

        Returns:
            IoU value (0.0 to 1.0)
        """
        # Calculate intersection
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


class Detection(BaseModel):
    """A single layout detection."""

    detection_id: str
    class_name: str
    confidence: float
    bbox: BoundingBox
    page_id: str
    image_path: Optional[str] = None

    # ALTO text content
    text_content: Optional[str] = None
    alto_elements: List[str] = Field(default_factory=list)

    # For images: associated caption
    caption: Optional["Detection"] = None
    caption_text: Optional[str] = None


class Headline(BaseModel):
    """A detected headline matched to OCR text."""

    headline_id: str
    detection: Detection
    ocr_text: str
    text_block_ids: List[str]
    confidence: float
    match_score: float  # How well detection bbox matches OCR coordinates

    # Position in document
    page_id: str
    year: int
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None


class Article(BaseModel):
    """A reconstructed newspaper article."""

    article_id: str
    headline: Headline
    text_blocks: List[str]  # Text block IDs
    full_text: str
    page_id: str
    year: int

    # Associated media
    images: List[Detection] = Field(default_factory=list)
    tables: List[Detection] = Field(default_factory=list)
    formulas: List[Detection] = Field(default_factory=list)

    # Metadata
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None

    # Spatial extent
    bbox: Optional[BoundingBox] = None

    # Quality metrics
    completeness_score: float = 1.0  # Estimate of article completeness

    @property
    @computed_field
    def num_images(self) -> int:
        """Number of images in the article."""
        return len(self.images)

    @property
    @computed_field
    def num_tables(self) -> int:
        """Number of tables in the article."""
        return len(self.tables)

    @property
    @computed_field
    def num_formulas(self) -> int:
        """Number of formulas in the article."""
        return len(self.formulas)

    @property
    @computed_field
    def word_count(self) -> int:
        """Word count of the article text."""
        return len(self.full_text.split())


class PageLayout(BaseModel):
    """Complete layout analysis for a page."""

    page_id: str
    image_path: str
    detections: List[Detection]

    # Organized by type
    headlines: List[Detection] = Field(default_factory=list)
    images: List[Detection] = Field(default_factory=list)
    captions: List[Detection] = Field(default_factory=list)
    tables: List[Detection] = Field(default_factory=list)
    text_blocks: List[Detection] = Field(default_factory=list)

    # Metadata
    year: int = 0
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None

    @property
    @computed_field  # type: ignore[misc]
    def total_detections(self) -> int:
        """Total number of detections."""
        return len(self.detections)

    @property
    @computed_field  # type: ignore[misc]
    def counts(self) -> Dict[str, int]:
        """Count of each detection type."""
        return {
            "headlines": len(self.headlines),
            "images": len(self.images),
            "captions": len(self.captions),
            "tables": len(self.tables),
            "text_blocks": len(self.text_blocks),
        }
