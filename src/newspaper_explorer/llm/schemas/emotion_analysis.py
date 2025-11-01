"""
Pydantic schema for emotion and sentiment analysis responses.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class EmotionAnalysisResponse(BaseModel):
    """Structured response for emotion/sentiment analysis."""

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment classification"
    )
    emotions: List[str] = Field(default_factory=list, description="List of detected emotions")
    intensity: float = Field(ge=0.0, le=1.0, description="Emotional intensity score (0.0-1.0)")
    tone: str = Field(description="Descriptive tone of the text")

    @field_validator("emotions", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v
