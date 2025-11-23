"""
Pydantic schemas for topic modeling results.

Provides type-safe schemas for LDA and BERTopic document-topic assignments.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TopicRecord(BaseModel):
    """
    Schema for document-level topic assignments.

    Represents a document's topic distribution and associated representative terms.
    Each document is assigned terms from its most probable topics, weighted by
    topic probability.

    Attributes:
        doc_id: Document identifier (page_id, textblock_id, or custom grouping)
        source_id: Source identifier (foreign key)
        issue_id: Issue identifier (foreign key, optional)
        page_id: Page identifier (foreign key, optional)
        text_block_id: Text block identifier (foreign key, optional)
        topic_terms: Representative terms from document's topics
        scores: Weighted scores for each term (topic_prob * word_prob)
        topics: Topic IDs assigned to this document
        topic_probs: Probability of each topic for this document

    Example:
        >>> record = TopicRecord(
        ...     doc_id="3074409-X_1902-09-05_415_2_005",
        ...     source_id="3074409-X",
        ...     issue_id="3074409-X_1902-09-05_415_2",
        ...     page_id="3074409-X_1902-09-05_415_2_005",
        ...     topic_terms=["krieg", "soldaten", "front"],
        ...     scores=[0.85, 0.72, 0.68],
        ...     topics=[5, 12],
        ...     topic_probs=[0.65, 0.30]
        ... )
    """

    doc_id: str = Field(..., description="Document identifier (any level of hierarchy)")

    # Foreign keys (extracted from doc_id)
    source_id: Optional[str] = Field(None, description="Source identifier")
    issue_id: Optional[str] = Field(None, description="Issue identifier")
    page_id: Optional[str] = Field(None, description="Page identifier")
    text_block_id: Optional[str] = Field(None, description="Text block identifier")

    # Topic assignment
    topic_terms: list[str] = Field(..., description="Representative terms from document's topics")
    scores: list[float] = Field(..., description="Weighted scores (topic_prob * word_prob)")
    topics: list[int] = Field(..., description="Topic IDs assigned to this document")
    topic_probs: list[float] = Field(..., description="Probability of each topic for this document")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doc_id": "3074409-X_1902-09-05_415_2_005",
                    "source_id": "3074409-X",
                    "issue_id": "3074409-X_1902-09-05_415_2",
                    "page_id": "3074409-X_1902-09-05_415_2_005",
                    "topic_terms": ["krieg", "soldaten", "front", "kampf", "truppen"],
                    "scores": [0.85, 0.72, 0.68, 0.55, 0.48],
                    "topics": [5, 12],
                    "topic_probs": [0.65, 0.30],
                }
            ]
        }
    }
