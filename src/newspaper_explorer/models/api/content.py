"""
API models for newspaper content (issues, pages, text blocks, lines).
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel


class Issue(BaseModel):
    """Newspaper issue summary for API"""

    issue_id: str
    date: date
    newspaper_title: str
    year_volume: str
    page_count: int
    has_images: bool = False
    daily_count: Optional[int] = None  # Daily issue number (1, 2, 3 for same day)


class Page(BaseModel):
    """Newspaper page summary for API"""

    page_id: str
    issue_id: str
    date: date
    newspaper_title: str
    page_number: int
    text_preview: Optional[str] = None
    image_url: Optional[str] = None
    has_image: bool = False
    alto_width: Optional[int] = None
    alto_height: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class TextBlock(BaseModel):
    """Text block summary for API"""

    text_block_id: str
    page_id: str
    issue_id: str
    date: date
    text: str
    x: int
    y: int
    width: int
    height: int


class Line(BaseModel):
    """Individual text line from OCR"""

    line_id: str
    text: str
    text_block_id: str
    page_id: str
    issue_id: str
    date: date
    newspaper_title: str
    year_volume: str
    page_number: int
    x: int
    y: int
    width: int
    height: int
