"""
API models for concept extraction results.
"""

from typing import Optional

from pydantic import BaseModel


class Concept(BaseModel):
    """Extracted concept with frequency"""

    concept: str
    frequency: int
    category: Optional[str] = None


class ConceptRelation(BaseModel):
    """Relationship between concepts"""

    source: str
    target: str
    weight: float
    relation_type: Optional[str] = None
