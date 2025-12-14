"""
Emotion classification endpoints
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from newspaper_explorer.analyze.emotions import queries as emotion_queries
from newspaper_explorer.data.utils.results import load_analysis_results

router = APIRouter()


@router.get("/{source_name}/statistics")
async def get_overall_statistics(source_name: str) -> dict[str, Any]:
    """Get overall emotion statistics and era analysis"""
    try:
        df = load_analysis_results(source_name, "emotions")
        return emotion_queries.get_statistics(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source_name}/timeline")
async def get_emotion_timeline(
    source_name: str, granularity: str = Query("year", regex="^(year|month)$")
) -> list[dict[str, Any]]:
    """Get emotion timeline aggregated by year or month"""
    try:
        df = load_analysis_results(source_name, "emotions")
        return emotion_queries.get_timeline(df, granularity)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source_name}/peaks")
async def get_emotion_peaks(
    source_name: str,
    emotion: str = Query(..., regex="^(sadness|love|joy|fear|anger|agitation)$"),
    limit: int = Query(10, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get top peaks for a specific emotion"""
    try:
        df = load_analysis_results(source_name, "emotions")
        return emotion_queries.get_peaks(df, emotion, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
