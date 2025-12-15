"""
Pydantic schema for entity extraction responses.
"""

from pydantic import BaseModel, Field


class EntityResponse(BaseModel):
    """Structured response for named entity extraction."""

    persons: list[str] = Field(
        default_factory=list, description="List of person names found in text"
    )
    locations: list[str] = Field(
        default_factory=list, description="List of location names found in text"
    )
    organizations: list[str] = Field(
        default_factory=list, description="List of organization names found in text"
    )
