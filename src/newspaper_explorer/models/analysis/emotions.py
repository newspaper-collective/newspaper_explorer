"""
Pydantic schemas for emotion prediction results.

Defines structured data models for emotion classification outputs with foreign key
tracking and type validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class EmotionRecord(BaseModel):
    """
    Single emotion prediction for a text line.

    Contains multi-label emotion predictions (6 emotions) with both binary
    predictions and confidence probabilities. Includes foreign keys for
    traceability back to source data.

    Attributes:
        line_id: Unique identifier for the text line
        source_id: Foreign key to source (e.g., "3074409-X")
        issue_id: Foreign key to issue (e.g., "3074409-X_1902-09-05_415_2")
        page_id: Foreign key to page (e.g., "3074409-X_1902-09-05_415_2_005")
        text_block_id: Foreign key to text block (e.g., "3074409-X_1902-09-05_415_2_005_TB_1")

        # Binary predictions (threshold-based)
        sadness: Whether sadness emotion detected
        love: Whether love emotion detected
        joy: Whether joy emotion detected
        fear: Whether fear emotion detected
        anger: Whether anger emotion detected
        agitation: Whether agitation emotion detected

        # Confidence probabilities (0-1)
        sadness_prob: Confidence score for sadness
        love_prob: Confidence score for love
        joy_prob: Confidence score for joy
        fear_prob: Confidence score for fear
        anger_prob: Confidence score for anger
        agitation_prob: Confidence score for agitation

        # Optional context
        text: Original text (optional, for reference)
    """

    # Identifiers & Foreign Keys
    line_id: str = Field(..., description="Unique line identifier")
    source_id: str = Field(..., description="Source identifier (foreign key)")
    issue_id: str = Field(..., description="Issue identifier (foreign key)")
    page_id: str = Field(..., description="Page identifier (foreign key)")
    text_block_id: str = Field(..., description="Text block identifier (foreign key)")

    # Binary Predictions
    sadness: bool = Field(..., description="Sadness detected (binary)")
    love: bool = Field(..., description="Love detected (binary)")
    joy: bool = Field(..., description="Joy detected (binary)")
    fear: bool = Field(..., description="Fear detected (binary)")
    anger: bool = Field(..., description="Anger detected (binary)")
    agitation: bool = Field(..., description="Agitation detected (binary)")

    # Confidence Probabilities
    sadness_prob: float = Field(..., ge=0.0, le=1.0, description="Sadness confidence")
    love_prob: float = Field(..., ge=0.0, le=1.0, description="Love confidence")
    joy_prob: float = Field(..., ge=0.0, le=1.0, description="Joy confidence")
    fear_prob: float = Field(..., ge=0.0, le=1.0, description="Fear confidence")
    anger_prob: float = Field(..., ge=0.0, le=1.0, description="Anger confidence")
    agitation_prob: float = Field(..., ge=0.0, le=1.0, description="Agitation confidence")

    # Optional Context
    text: Optional[str] = Field(None, description="Original text for reference")


class EmotionPredictionResult(BaseModel):
    """
    Complete emotion prediction result wrapper.

    Wraps emotion predictions with summary statistics for analysis overview.

    Attributes:
        emotions: List of emotion predictions
        statistics: Summary statistics about predictions
    """

    emotions: list[EmotionRecord] = Field(..., description="Emotion predictions")
    statistics: dict[str, Any] = Field(
        ...,
        description="Summary statistics (total_predictions, emotion_counts, etc.)",
    )
