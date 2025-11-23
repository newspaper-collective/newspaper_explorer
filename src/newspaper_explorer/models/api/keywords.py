"""
API models for keyword analysis results.
"""

from typing import Optional

from pydantic import BaseModel


class Keyword(BaseModel):
    """Keyword with frequency"""

    keyword: str
    frequency: int
    tfidf_score: float


class KeywordDocument(BaseModel):
    """Document containing a keyword"""

    doc_id: str
    score: float
    date: Optional[str] = None
    page_id: Optional[str] = None


class KeywordCoOccurrence(BaseModel):
    """Co-occurring keyword"""

    keyword: str
    count: int


class KeywordTimeline(BaseModel):
    """Keyword frequency over time"""

    keyword: str
    timeline: dict[str, int]  # date -> count


class PaginatedKeywords(BaseModel):
    """Paginated keyword response"""

    items: list[Keyword]
    total: int
    page: int
    page_size: int
