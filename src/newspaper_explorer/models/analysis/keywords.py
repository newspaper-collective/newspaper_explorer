"""
Pydantic schemas for keyword extraction results.

Defines structured data models for keyword extraction outputs with foreign key
tracking and type validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class KeywordRecord(BaseModel):
    """
    Single keyword extraction result for a document.

    Contains extracted keywords with relevance scores. Includes foreign keys for
    traceability back to source data.

    Attributes:
        doc_id: Document identifier (text_block_id or line_id)
        source_id: Foreign key to source (e.g., "3074409-X")
        issue_id: Foreign key to issue (e.g., "3074409-X_1902-09-05_415_2")
        page_id: Foreign key to page (e.g., "3074409-X_1902-09-05_415_2_005")
        text_block_id: Foreign key to text block (optional, e.g., "3074409-X_1902-09-05_415_2_005_TB_1")

        keywords: List of extracted keywords/keyphrases
        scores: List of relevance scores (same length as keywords)

        text: Original text (optional, for reference)
    """

    # Identifiers & Foreign Keys
    doc_id: str = Field(..., description="Document identifier (text_block_id or line_id)")
    source_id: str = Field(..., description="Source identifier (foreign key)")
    issue_id: str = Field(..., description="Issue identifier (foreign key)")
    page_id: str = Field(..., description="Page identifier (foreign key)")
    text_block_id: Optional[str] = Field(
        None, description="Text block identifier (foreign key, if applicable)"
    )

    # Keywords and Scores
    keywords: list[str] = Field(..., description="Extracted keywords/keyphrases")
    scores: list[float] = Field(..., description="Relevance scores (0-1, higher = more relevant)")

    # Optional Context
    text: Optional[str] = Field(None, description="Original text for reference")


class KeywordExtractionResult(BaseModel):
    """
    Complete keyword extraction result wrapper.

    Wraps keyword extraction results with summary statistics for analysis overview.

    Attributes:
        keywords: List of keyword extraction results
        statistics: Summary statistics about extraction
    """

    keywords: list[KeywordRecord] = Field(..., description="Keyword extraction results")
    statistics: dict[str, Any] = Field(
        ...,
        description="Summary statistics (total_documents, total_keywords, avg_keywords_per_doc, etc.)",
    )
