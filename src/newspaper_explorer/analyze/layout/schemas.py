"""
Data schemas for layout analysis results.

Defines the structure for detected elements, matched headlines, and reconstructed articles.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

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
    """
    A single layout detection.

    Uses unified ID system with foreign keys for linking to source data.
    """

    # Primary key
    detection_id: str  # Format: {page_id}_{class}_{uuid_short}

    # Detection data
    class_name: str
    confidence: float
    bbox: BoundingBox

    # Foreign keys
    page_id: str  # FK: Links to source page
    source_id: Optional[str] = None  # FK: Source identifier (e.g., "3074409-X")
    issue_id: Optional[str] = None  # FK: Issue identifier
    image_path: Optional[str] = None

    # ALTO text content
    text_content: Optional[str] = None
    alto_elements: List[str] = Field(default_factory=list)

    # For images: associated caption
    caption: Optional["Detection"] = None
    caption_text: Optional[str] = None


class Headline(BaseModel):
    """A detected headline matched to OCR text."""

    # Primary key
    headline_id: str

    # Related detection
    detection: Detection

    # Data
    ocr_text: str
    text_block_ids: List[str]
    confidence: float
    match_score: float  # How well detection bbox matches OCR coordinates

    # Foreign keys
    page_id: str  # FK: Links to page
    source_id: Optional[str] = None  # FK: Source identifier (e.g., "3074409-X")
    issue_id: Optional[str] = None  # FK: Issue identifier
    detection_id: Optional[str] = None  # FK: Link to detection

    # Metadata
    year: int
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None


class Article(BaseModel):
    """A detected article (reconstructed text block)."""

    # Primary key
    article_id: str

    # Related data
    headlines: List[Headline]
    text_blocks: List[str]  # Text block IDs that make up the article
    full_text: str
    detection: Detection

    # Foreign keys
    page_id: str  # FK: Links to page
    source_id: Optional[str] = None  # FK: Source identifier (e.g., "3074409-X")
    issue_id: Optional[str] = None  # FK: Issue identifier
    detection_id: Optional[str] = None  # FK: Link to detection

    # Metadata
    year: int
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None

    # Quality metrics
    confidence: float  # Overall confidence
    completeness: float  # Estimated percentage of article captured


class PageLayout(BaseModel):
    """Complete layout analysis for a page."""

    page_id: str
    image_path: str
    detections: List[Detection]

    # Metadata
    year: int = 0
    date: Optional[datetime] = None
    newspaper_title: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def total_detections(self) -> int:
        """Total number of detections."""
        return len(self.detections)

    @computed_field  # type: ignore[misc]
    @property
    def counts(self) -> Dict[str, int]:
        """Count of each detection type from detections list."""
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def filter_by_class(self, class_names: Union[str, List[str]]) -> List[Detection]:
        """Filter detections by class name(s)."""
        if isinstance(class_names, str):
            class_names = [class_names]
        return [d for d in self.detections if d.class_name in class_names]
