"""
Entity extraction endpoints
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from newspaper_explorer.analyze.entities import queries as entity_queries
from newspaper_explorer.data.utils.results import load_analysis_results
from newspaper_explorer.models.analysis.entities import AggregatedEntityRecord, EntityRecord

router = APIRouter()


@router.get("/{source_name}/", response_model=list[AggregatedEntityRecord])
async def get_entities(
    source_name: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(default=None),
):
    """Get list of aggregated entities with optional filtering"""
    try:
        df = load_analysis_results(source_name, "entities", run_id)
        aggregated_df = entity_queries.aggregate_entities(df, entity_type, limit)

        # Convert to response model
        return [AggregatedEntityRecord(**row) for row in aggregated_df.to_dicts()]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source_name}/types", response_model=list[str])
async def get_entity_types(source_name: str, run_id: Optional[str] = Query(None)):
    """Get list of unique entity types"""
    try:
        df = load_analysis_results(source_name, "entities", run_id)
        return entity_queries.get_entity_types(df)
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source_name}/occurrences", response_model=list[EntityRecord])
async def get_entity_occurrences(
    source_name: str,
    entity_text: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(default=100),
):
    """Get individual occurrences of a specific entity"""
    try:
        df = load_analysis_results(source_name, "entities", run_id)
        occurrences_df = entity_queries.get_entity_occurrences(df, entity_text, entity_type, limit)

        return [EntityRecord(**row) for row in occurrences_df.to_dicts()]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{source_name}/timeline")
async def get_entity_timeline(
    source_name: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    aggregation: str = Query(default="month", regex="^(day|month|year)$"),
) -> dict[str, list[dict[str, Any]]]:
    """Get entity counts over time, aggregated by day/month/year"""
    try:
        df = load_analysis_results(source_name, "entities", run_id)
        return entity_queries.get_timeline(df, entity_type, aggregation)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
