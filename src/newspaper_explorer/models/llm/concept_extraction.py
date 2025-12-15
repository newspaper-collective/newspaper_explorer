"""
Pydantic schemas for concept extraction responses.
"""

from typing import Literal

from pydantic import BaseModel


class ConceptRelationship(BaseModel):
    """Relationship between two concepts."""

    source: str
    target: str
    type: Literal["leads_to", "causes", "contradicts", "supports"]


class ConceptExtractionResponse(BaseModel):
    """Structured response for concept extraction."""

    concepts: list[str]
    relationships: list[ConceptRelationship] = []
