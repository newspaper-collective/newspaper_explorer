"""
Pydantic schema for entity extraction responses.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class EntityResponse(BaseModel):
    """Structured response for named entity extraction."""

    persons: List[str] = Field(
        default_factory=list, description="List of person names found in text"
    )
    locations: List[str] = Field(
        default_factory=list, description="List of location names found in text"
    )
    organizations: List[str] = Field(
        default_factory=list, description="List of organization names found in text"
    )

    @field_validator("persons", "locations", "organizations", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure field is a list even if LLM returns None."""
        if v is None:
            return []
        return v
