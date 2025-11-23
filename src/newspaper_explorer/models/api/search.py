"""
API models for search functionality.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel

from newspaper_explorer.models.api.filters import DateFilter, PaginationParams


class SearchQuery(BaseModel):
    """Search query parameters"""

    query: str
    date_filter: Optional[DateFilter] = None
    entity_filter: Optional[list[str]] = None
    keyword_filter: Optional[list[str]] = None
    run_id: Optional[str] = None
    pagination: Optional[PaginationParams] = None


class SearchResult(BaseModel):
    """Search result item"""

    text_block_id: str
    page_id: str
    date: date
    text: str
    highlights: list[str]
    score: float
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_path: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response with results"""

    total: int
    results: list[SearchResult]
    page: int
    page_size: int
