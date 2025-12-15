"""
Pydantic schema for keyword extraction responses.
"""

from pydantic import BaseModel


class KeywordResponse(BaseModel):
    """Structured response for keyword extraction."""

    keywords: list[str] = []
    scores: list[float] = []