"""
Pydantic schema for text quality assessment responses.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class TextQualityResponse(BaseModel):
    """Structured response for text quality assessment."""

    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score (0.0-1.0)")
    ocr_errors: Literal["low", "medium", "high"] = Field(description="Estimated OCR error level")
    coherence: Literal["coherent", "partial", "fragmented"] = Field(
        description="Text coherence level"
    )
    issues: List[str] = Field(default_factory=list, description="List of detected issues")
    readable: bool = Field(description="Whether text is usable for analysis")

    @field_validator("issues", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v
