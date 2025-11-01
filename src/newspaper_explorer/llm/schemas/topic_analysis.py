"""
Pydantic schemas for topic analysis responses.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class TopicClassificationResponse(BaseModel):
    """Structured response for topic classification."""

    primary_topic: str = Field(description="Main topic of the text")
    secondary_topics: List[str] = Field(
        default_factory=list, description="Additional relevant topics"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for primary topic (0.0-1.0)"
    )

    @field_validator("secondary_topics", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


class TopicGenerationResponse(BaseModel):
    """Structured response for topic generation."""

    topics: List[str] = Field(description="Generated topic labels")
    main_theme: str = Field(description="One-sentence summary of main theme")

    @field_validator("topics", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v
