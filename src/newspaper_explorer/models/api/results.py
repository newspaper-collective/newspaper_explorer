"""
API response models for analysis results endpoints.

These models define the structure of responses from the /results/ API endpoints
for listing and querying analysis runs across all analysis types.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisRunInfo(BaseModel):
    """Information about a single analysis run"""

    run_id: str = Field(..., description="Unique run identifier")
    source: str = Field(..., description="Source dataset name")
    analysis_type: str = Field(..., description="Type of analysis (entities, emotions, etc.)")
    method_type: str = Field(..., description="Method used (gliner, keybert, etc.)")
    model_name: str = Field(..., description="Specific model identifier")
    created_at: str = Field(..., description="ISO timestamp of creation")
    row_count: int = Field(..., description="Number of records in results")
    parameters: dict[str, Any] = Field(..., description="Analysis parameters used")


class AnalysisAvailability(BaseModel):
    """Availability information for an analysis type"""

    available: bool = Field(..., description="Whether any results exist")
    run_count: int = Field(..., description="Number of available runs")
    latest_run: Optional[str] = Field(None, description="ID of most recent run")


class AvailableAnalysis(BaseModel):
    """Information about available analysis for a source"""

    analysis_type: str = Field(..., description="Type of analysis")
    run_count: int = Field(..., description="Number of available runs")
    latest_run: str = Field(..., description="ID of most recent run")
