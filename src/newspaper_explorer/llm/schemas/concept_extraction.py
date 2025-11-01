"""
Pydantic schemas for concept extraction responses.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class ConceptRelationship(BaseModel):
    """Relationship between two concepts."""

    source: str = Field(description="Source concept")
    target: str = Field(description="Target concept")
    type: Literal["leads_to", "causes", "contradicts", "supports"] = Field(
        description="Type of relationship"
    )


class ConceptExtractionResponse(BaseModel):
    """Structured response for concept extraction."""

    concepts: List[str] = Field(description="List of key concepts/themes")
    relationships: List[ConceptRelationship] = Field(
        default_factory=list, description="Relationships between concepts"
    )

    @field_validator("concepts", "relationships", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v
