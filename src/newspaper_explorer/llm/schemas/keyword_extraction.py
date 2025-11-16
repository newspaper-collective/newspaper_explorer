"""
Pydantic schema for keyword extraction responses.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class KeywordResponse(BaseModel):
    """Structured response for keyword extraction."""

    keywords: List[str] = Field(
        default_factory=list,
        description="List of keywords/keyphrases extracted from text, ordered by importance",
    )
    scores: List[float] = Field(
        default_factory=list,
        description="Confidence scores (0.0-1.0) for each keyword indicating importance",
    )

    @field_validator("keywords", "scores", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure field is a list even if LLM returns None."""
        if v is None:
            return []
        return v

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v, info):
        """Ensure scores match keywords length and are in valid range."""
        if not v:
            return v

        # Check all scores are between 0 and 1
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score {score} is not in range [0.0, 1.0]")

        return v

    def model_post_init(self, __context):
        """Ensure keywords and scores have same length."""
        if len(self.keywords) != len(self.scores):
            # If mismatch, pad scores with 0.5 or truncate
            if len(self.scores) < len(self.keywords):
                self.scores.extend([0.5] * (len(self.keywords) - len(self.scores)))
            else:
                self.scores = self.scores[: len(self.keywords)]
