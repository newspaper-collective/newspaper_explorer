"""
Pydantic schemas for LLM response validation.

Defines structured response formats for different analysis tasks.
Used to validate and parse LLM outputs consistently.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Entity Extraction Schemas
# ============================================================================


class EntityResponse(BaseModel):
    """Structured response for named entity extraction."""

    persons: List[str] = Field(
        default_factory=list, description="List of person names found in text"
    )
    locations: List[str] = Field(
        default_factory=list, description="List of location names found in text"
    )
    organizations: List[str] = Field(
        default_factory=list, description="List of organization names found in text"
    )

    @field_validator("persons", "locations", "organizations", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure field is a list even if LLM returns None."""
        if v is None:
            return []
        return v


# ============================================================================
# Topic Analysis Schemas
# ============================================================================


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


# ============================================================================
# Emotion/Sentiment Analysis Schemas
# ============================================================================


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


# ============================================================================
# Concept Extraction Schemas
# ============================================================================


class ConceptRelationship(BaseModel):
    """Relationship between two concepts."""

    source: str = Field(description="Source concept")
    target: str = Field(description="Target concept")
    type: Literal["leads_to", "causes", "contradicts", "supports"] = Field(
        description="Type of relationship"
    )


class ConceptExtractionResponse(BaseModel):
    """Structured response for concept extraction."""

    concepts: List[str] = Field(description="List of key concepts/themes")
    relationships: List[ConceptRelationship] = Field(
        default_factory=list, description="Relationships between concepts"
    )

    @field_validator("concepts", "relationships", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


# ============================================================================
# Text Summarization Schemas
# ============================================================================


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


# ============================================================================
# Quality Assessment Schemas
# ============================================================================


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


# ============================================================================
# Helper Functions
# ============================================================================


def get_schema(name: str) -> type[BaseModel]:
    """
    Get a response schema by name.

    Args:
        name: Schema name (e.g., "entity_extraction", "topic_classification").

    Returns:
        Pydantic model class.

    Raises:
        KeyError: If schema name not found.
    """
    schemas = {
        "entity_extraction": EntityResponse,
        "topic_classification": TopicClassificationResponse,
        "topic_generation": TopicGenerationResponse,
        "emotion_analysis": EmotionAnalysisResponse,
        "concept_extraction": ConceptExtractionResponse,
        "summarization": SummarizationResponse,
        "text_quality": TextQualityResponse,
    }

    if name not in schemas:
        available = ", ".join(schemas.keys())
        raise KeyError(f"Unknown schema '{name}'. Available: {available}")

    return schemas[name]


def list_schemas() -> List[str]:
    """
    List all available schema names.

    Returns:
        List of schema names.
    """
    return [
        "entity_extraction",
        "topic_classification",
        "topic_generation",
        "emotion_analysis",
        "concept_extraction",
        "summarization",
        "text_quality",
    ]
