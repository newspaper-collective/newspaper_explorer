"""Pydantic schemas for topic analysis responses."""

from typing import Union

from pydantic import BaseModel, Field, field_validator


class TopicClassificationResponse(BaseModel):
    """Structured response for topic classification."""

    primary_topic: str = Field(description="Main topic of the text")
    secondary_topics: Union[list[str], None] = Field(
        default=None, description="Additional relevant topics"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for primary topic (0.0-1.0)"
    )

    @field_validator("secondary_topics", mode="after")
    @classmethod
    def convert_none_to_list(cls, v: Union[list[str], None]) -> list[str]:
        """Convert None to empty list."""
        return v if v is not None else []


class TopicGenerationResponse(BaseModel):
    """Structured response for topic generation."""

    topics: Union[list[str], None] = Field(default=None, description="Generated topic labels")

    @field_validator("topics", mode="after")
    @classmethod
    def convert_none_to_list(cls, v: Union[list[str], None]) -> list[str]:
        """Convert None to empty list."""
        return v if v is not None else []
