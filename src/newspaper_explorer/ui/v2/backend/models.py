"""
Pydantic models for API requests and responses.

This module re-exports models from the centralized models package.
All actual model definitions are in src/newspaper_explorer/models/.
"""

# Re-export core data models
from newspaper_explorer.models.analysis.emotions import EmotionRecord

# Re-export analysis models
from newspaper_explorer.models.analysis.entities import AggregatedEntityRecord, EntityRecord
from newspaper_explorer.models.analysis.layout import BoundingBox, Detection
from newspaper_explorer.models.analysis.topics import TopicRecord
from newspaper_explorer.models.api.concepts import Concept, ConceptRelation
from newspaper_explorer.models.api.content import Issue, Line, Page, TextBlock
from newspaper_explorer.models.api.emotions import EmotionTimeline
from newspaper_explorer.models.api.filters import DateFilter, PaginationParams
from newspaper_explorer.models.api.keywords import (
    Keyword,
    KeywordCoOccurrence,
    KeywordDocument,
    KeywordTimeline,
    PaginatedKeywords,
)
from newspaper_explorer.models.api.search import SearchQuery, SearchResponse, SearchResult

# Re-export API models
from newspaper_explorer.models.api.sources import AnalysisResultSummary, SourceInfo, SourceStats
from newspaper_explorer.models.data.content import IssueMetadata
