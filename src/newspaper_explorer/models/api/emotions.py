"""
API models for emotion analysis results.
"""

from pydantic import BaseModel


class EmotionTimeline(BaseModel):
    """Emotion scores aggregated over time"""

    emotion: str
    timeline: dict[str, float]  # date -> avg_score
