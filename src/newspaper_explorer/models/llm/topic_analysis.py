"""Pydantic schemas for topic analysis responses."""

from pydantic import BaseModel, Field


class TopicClassificationResponse(BaseModel):
    """Structured response for topic classification."""

    primary_topic: str = Field(description="Main topic of the text")
    secondary_topics: list[str] = Field(
        default_factory=list, description="Additional relevant topics"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for primary topic (0.0-1.0)"
    )


class TopicGenerationResponse(BaseModel):
    """Structured response for topic generation."""

    topics: list[str] = Field(default_factory=list, description="Generated topic labels")
