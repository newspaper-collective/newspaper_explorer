"""
Pydantic models for API requests and responses.

This module re-exports models from the centralized models package.
All actual model definitions are in src/newspaper_explorer/models/.
"""

# Re-export core data models
from newspaper_explorer.models.core.data import IssueMetadata

# Re-export analysis models
from newspaper_explorer.models.analysis.entities import EntityRecord, AggregatedEntityRecord
from newspaper_explorer.models.analysis.layout import Detection, BoundingBox
from newspaper_explorer.models.analysis.emotions import EmotionRecord
from newspaper_explorer.models.analysis.topics import TopicRecord

# Re-export API models
from newspaper_explorer.models.api.sources import AnalysisResultSummary, SourceInfo, SourceStats
from newspaper_explorer.models.api.filters import DateFilter, PaginationParams
from newspaper_explorer.models.api.content import Issue, Page, TextBlock, Line
from newspaper_explorer.models.api.keywords import (
    Keyword,
    KeywordDocument,
    KeywordCoOccurrence,
    KeywordTimeline,
    PaginatedKeywords,
)
from newspaper_explorer.models.api.emotions import EmotionTimeline
from newspaper_explorer.models.api.concepts import Concept, ConceptRelation
from newspaper_explorer.models.api.search import SearchQuery, SearchResult, SearchResponse
