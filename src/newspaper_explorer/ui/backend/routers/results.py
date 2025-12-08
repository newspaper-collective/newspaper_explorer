"""
Universal results endpoints for all analysis types.

Provides generic endpoints for loading, listing, and querying analysis results
with metadata. Can be used by all analysis modules (entities, emotions, topics, etc.).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from newspaper_explorer.ui.backend.utils.results import ResultsLoader


# Response models
class AnalysisRunInfo(BaseModel):
    """Information about a single analysis run"""

    run_id: str
    display_name: str
    source: str
    analysis_type: str
    created_at: Optional[str]
    duration_seconds: Optional[float]
    row_count: int
    parameters: Dict[str, Any]


class AnalysisMetadata(BaseModel):
    """Complete metadata for an analysis run"""

    source: str
    analysis_type: str
    run_id: str
    display_name: str
    row_count: int
    created_at: Optional[str]
    duration_seconds: Optional[float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]  # Full metadata object


router = APIRouter()


@router.get("/{source}/{analysis_type}/runs", response_model=List[AnalysisRunInfo])
async def list_runs(source: str, analysis_type: str):
    """
    List all available runs for a source and analysis type.

    Returns runs sorted by creation date (newest first).
    """
    try:
        loader = ResultsLoader()
        runs = loader.list_runs(source, analysis_type)

        if not runs:
            return []

        # Load full metadata for each run
        run_infos = []
        for run_id, display_name in runs:
            result = loader.load_result(source, analysis_type, run_id)
            if result:
                run_infos.append(
                    AnalysisRunInfo(
                        run_id=run_id,
                        display_name=display_name,
                        source=source,
                        analysis_type=analysis_type,
                        created_at=result.metadata.get("created_at"),
                        duration_seconds=result.duration_seconds,
                        row_count=result.row_count,
                        parameters=result.parameters,
                    )
                )

        return run_infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source}/{analysis_type}/metadata", response_model=AnalysisMetadata)
async def get_metadata(source: str, analysis_type: str, run_id: Optional[str] = Query(None)):
    """
    Get metadata for a specific run or the most recent run.

    If run_id is not provided, returns metadata for the most recent run.
    """
    try:
        loader = ResultsLoader()
        result = loader.load_result(source, analysis_type, run_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No {analysis_type} results found for {source}",
            )

        return AnalysisMetadata(**result.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source}/{analysis_type}/availability")
async def check_availability(source: str, analysis_type: str):
    """Check if any results exist for the given source and analysis type"""
    try:
        loader = ResultsLoader()
        available = loader.check_availability(source, analysis_type)
        runs = loader.list_runs(source, analysis_type)

        return {
            "available": available,
            "run_count": len(runs),
            "latest_run": runs[0][0] if runs else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source}/available-analyses")
async def get_available_analyses(source: str):
    """Get list of all available analysis types for a source"""
    try:
        loader = ResultsLoader()
        analysis_types = loader.list_analysis_types(source)

        # Get run count for each analysis type
        results = []
        for analysis_type in analysis_types:
            runs = loader.list_runs(source, analysis_type)
            if runs:  # Only include types with actual runs
                results.append(
                    {
                        "analysis_type": analysis_type,
                        "run_count": len(runs),
                        "latest_run": runs[0][0],
                        "latest_display_name": runs[0][1],
                    }
                )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
