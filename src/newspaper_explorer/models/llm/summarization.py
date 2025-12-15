"""
Pydantic schema for text summarization responses.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SummarizationResponse(BaseModel):
    """Structured response for text summarization."""

    summary: str = Field(description="Brief summary of the text")
    key_points: list[str] = Field(
        default_factory=list, description="List of key points (3-5 items)"
    )
    historical_context: Optional[str] = Field(
        default=None, description="Historical significance (if applicable)"
    )
