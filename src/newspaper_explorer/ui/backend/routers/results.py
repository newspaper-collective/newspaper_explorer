"""
Universal results endpoints for all analysis types.

Provides generic endpoints for loading, listing, and querying analysis results
with metadata. Can be used by all analysis modules (entities, emotions, topics, etc.).
"""

import logging
from pathlib import Path
from typing import Any, Optional, get_args

from fastapi import APIRouter, HTTPException, Query

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.results import (
    list_analysis_results,
    load_analysis_metadata,
    load_analysis_results,
)
from newspaper_explorer.models.api.results import (
    AnalysisAvailability,
    AnalysisRunInfo,
    AvailableAnalysis,
)
from newspaper_explorer.models.data.metadata import AnalysisType

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{source}/{analysis_type}/runs", response_model=list[AnalysisRunInfo])
async def list_runs(source: str, analysis_type: AnalysisType) -> list[AnalysisRunInfo]:
    """
    List all available runs for a source and analysis type.

    Returns runs sorted chronologically (oldest first).
    """
    try:
        run_ids = list_analysis_results(source, analysis_type)

        if not run_ids:
            return []

        # Load metadata for each run
        run_infos: list[AnalysisRunInfo] = []
        for run_id in run_ids:
            try:
                metadata = load_analysis_metadata(source, analysis_type, run_id)
                df = load_analysis_results(source, analysis_type, run_id)

                run_infos.append(
                    AnalysisRunInfo(
                        run_id=run_id,
                        source=source,
                        analysis_type=analysis_type,
                        method_type=metadata.method_type,
                        model_name=metadata.model_name,
                        created_at=metadata.created_at,
                        row_count=len(df),
                        parameters=metadata.parameters,
                    )
                )
            except (FileNotFoundError, ValueError, KeyError) as e:
                logger.warning(f"Error loading run {run_id}: {e}")
                continue

        return run_infos
    except FileNotFoundError:
        return []  # No runs found
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source}/{analysis_type}/metadata")
async def get_metadata(
    source: str, analysis_type: AnalysisType, run_id: Optional[str] = Query(None)
) -> dict[str, Any]:
    """
    Get metadata for a specific run or the most recent run.

    If run_id is not provided, returns metadata for the most recent run.
    """
    try:
        metadata = load_analysis_metadata(source, analysis_type, run_id)
        return metadata.model_dump()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source}/{analysis_type}/availability", response_model=AnalysisAvailability)
async def check_availability(source: str, analysis_type: AnalysisType) -> AnalysisAvailability:
    """Check if any results exist for the given source and analysis type"""
    try:
        runs = list_analysis_results(source, analysis_type)

        return AnalysisAvailability(
            available=len(runs) > 0,
            run_count=len(runs),
            latest_run=runs[-1] if runs else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source}/available-analyses", response_model=list[AvailableAnalysis])
async def get_available_analyses(source: str) -> list[AvailableAnalysis]:
    """Get list of all available analysis types for a source"""
    try:
        config = get_config()
        results_dir = Path(config.results_dir) / source

        if not results_dir.exists():
            return []

        # Check each known analysis type
        analysis_types = get_args(AnalysisType)

        results: list[AvailableAnalysis] = []
        for analysis_type in analysis_types:
            runs = list_analysis_results(source, analysis_type)
            if runs:
                results.append(
                    AvailableAnalysis(
                        analysis_type=analysis_type,
                        run_count=len(runs),
                        latest_run=runs[-1],
                    )
                )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
