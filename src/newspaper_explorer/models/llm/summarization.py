"""
Pydantic schema for text summarization responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SummarizationResponse(BaseModel):
    """Structured response for text summarization."""

    summary: str = Field(description="Brief summary of the text")
    key_points: List[str] = Field(description="List of key points (3-5 items)")
    historical_context: Optional[str] = Field(
        default=None, description="Historical significance (if applicable)"
    )

    @field_validator("key_points", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v
