"""
Pydantic schemas for entity extraction results.

Defines standardized output schemas for different entity extraction methods
(GLiNER, LLM, etc.) to ensure consistent result structure.
"""

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


class EntityRecord(BaseModel):
    """
    Single entity extraction record.

    This schema represents one detected entity with its context and metadata.
    Used for both per-instance and aggregated results.
    """

    # Entity information
    entity_text: str = Field(..., description="Extracted entity text")
    entity_type: str = Field(
        ...,
        description="Entity type (person, organization, location, event, etc.)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")

    # Foreign keys (required)
    text_block_id: str = Field(..., description="Text block ID where entity was detected")
    source_id: str = Field(..., description="Source identifier (foreign key)")
    issue_id: str = Field(..., description="Issue identifier (foreign key)")
    page_id: str = Field(..., description="Page identifier (foreign key)")

    # Optional context
    text: Optional[str] = Field(None, description="Full text context where entity was found")

    # Position information (if available)
    start_char: Optional[int] = Field(None, description="Start character position in text")
    end_char: Optional[int] = Field(None, description="End character position in text")

    class Config:
        """Pydantic config."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "entity_text": "Berlin",
                "entity_type": "location",
                "confidence": 0.95,
                "text_block_id": "der_tag_1902-09-05_415_2_005_r_1_1",
                "source_id": "der_tag",
                "issue_id": "der_tag_1902-09-05_415_2",
                "page_id": "der_tag_1902-09-05_415_2_005",
                "text": "Der Kaiser besuchte Berlin gestern.",
                "start_char": 21,
                "end_char": 27,
            }
        }


class AggregatedEntityRecord(BaseModel):
    """
    Aggregated entity record across multiple detections.

    Used when entities are deduplicated and aggregated to show
    overall statistics (e.g., how many times "Berlin" appears).
    """

    # Entity information
    entity_text: str = Field(..., description="Extracted entity text (unique)")
    entity_type: str = Field(..., description="Entity type")

    # Aggregated statistics
    detection_count: int = Field(..., ge=1, description="Number of times entity was detected")
    avg_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Average confidence across detections"
    )
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence")
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum confidence")

    # Source references (list of all occurrences)
    text_block_ids: list[str] = Field(
        default_factory=list, description="All text block IDs where entity appears"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "entity_text": "Berlin",
                "entity_type": "location",
                "detection_count": 15,
                "avg_confidence": 0.93,
                "min_confidence": 0.85,
                "max_confidence": 0.98,
                "text_block_ids": [
                    "der_tag_1902-09-05_415_2_005_r_1_1",
                    "der_tag_1902-09-05_415_2_005_r_1_2",
                ],
            }
        }


class EntityExtractionResult(BaseModel):
    """
    Complete entity extraction result with metadata.

    This wraps the entity records with extraction metadata for a complete
    serializable result object.
    """

    # Results
    entities: list[EntityRecord] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list, description="List of extracted entities"
    )

    # Statistics
    total_entities: int = Field(..., description="Total number of entities extracted")
    unique_entities: int = Field(..., description="Number of unique entity texts")
    entity_types: dict[str, int] = Field(
        default_factory=dict, description="Count of entities per type"
    )

    # Processing info
    lines_processed: int = Field(..., description="Number of input lines processed")
    lines_with_entities: int = Field(
        ..., description="Number of lines that had at least one entity"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "entities": [],
                "total_entities": 150,
                "unique_entities": 47,
                "entity_types": {"person": 85, "organization": 35, "location": 30},
                "lines_processed": 1000,
                "lines_with_entities": 320,
            }
        }
