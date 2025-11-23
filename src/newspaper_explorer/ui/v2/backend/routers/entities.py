"""
Entity extraction endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import date
import polars as pl
from pathlib import Path

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.analysis.entities import EntityRecord, AggregatedEntityRecord
from newspaper_explorer.ui.v2.backend.utils.results import ResultsLoader

router = APIRouter()


@router.get("/{source_name}/", response_model=List[AggregatedEntityRecord])
async def get_entities(
    source_name: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(default=None),
):
    """Get list of aggregated entities with optional filtering"""
    try:
        loader = ResultsLoader()
        result = loader.load_result(source_name, "entities", run_id)

        if not result:
            raise HTTPException(status_code=404, detail="No entity data available")

        df = result.df

        # Apply filters
        if entity_type:
            df = df.filter(df["entity_type"] == entity_type)

        # Aggregate by entity text and type
        entities_df = (
            df.group_by(["entity_text", "entity_type"])
            .agg(
                [
                    pl.count().alias("detection_count"),
                    pl.col("confidence").mean().alias("avg_confidence"),
                    pl.col("confidence").min().alias("min_confidence"),
                    pl.col("confidence").max().alias("max_confidence"),
                    pl.col("line_id").unique().alias("line_ids"),
                ]
            )
            .sort("detection_count", descending=True)
        )

        # Apply limit only if specified
        if limit is not None:
            entities_df = entities_df.head(limit)

        # Convert to response model
        entities = []
        for row in entities_df.iter_rows(named=True):
            entities.append(
                AggregatedEntityRecord(
                    entity_text=row["entity_text"],
                    entity_type=row["entity_type"],
                    detection_count=row["detection_count"],
                    avg_confidence=row["avg_confidence"],
                    min_confidence=row["min_confidence"],
                    max_confidence=row["max_confidence"],
                    line_ids=row["line_ids"],
                )
            )

        return entities
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No entity data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/types", response_model=List[str])
async def get_entity_types(source_name: str, run_id: Optional[str] = Query(None)):
    """Get list of unique entity types"""
    try:
        loader = ResultsLoader()
        result = loader.load_result(source_name, "entities", run_id)

        if not result:
            return []

        df = result.df
        types = df["entity_type"].unique().to_list()
        return sorted(types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/occurrences", response_model=List[EntityRecord])
async def get_entity_occurrences(
    source_name: str,
    entity_text: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(default=100),
):
    """Get individual occurrences of a specific entity"""
    try:
        loader = ResultsLoader()
        result = loader.load_result(source_name, "entities", run_id)

        if not result:
            raise HTTPException(status_code=404, detail="No entity data available")

        df = result.df

        # Filter by entity text
        df = df.filter(df["entity_text"] == entity_text)
        if entity_type:
            df = df.filter(df["entity_type"] == entity_type)

        # Apply limit only if specified
        if limit is not None:
            df = df.head(limit)

        # Convert to response model using existing schema
        return [EntityRecord(**row) for row in df.to_dicts()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/timeline")
async def get_entity_timeline(
    source_name: str,
    entity_type: Optional[str] = None,
    run_id: Optional[str] = Query(None),
    aggregation: str = Query(default="month", regex="^(day|month|year)$"),
):
    """Get entity counts over time, aggregated by day/month/year"""
    try:
        loader = ResultsLoader()
        result = loader.load_result(source_name, "entities", run_id)

        if not result:
            raise HTTPException(status_code=404, detail="No entity data available")

        df = result.df

        # Extract date from line_id (format: source_YYYY-MM-DD_volume_page_...)
        # line_id example: "3074409-X_1902-09-05_415_2_005_TB_1_TL_1"
        if "line_id" not in df.columns:
            return {}

        # Parse date from line_id structure
        df = df.with_columns(
            [
                pl.col("line_id")
                .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 1)
                .cast(pl.Int32)
                .alias("year"),
                pl.col("line_id")
                .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 2)
                .cast(pl.Int32)
                .alias("month"),
                pl.col("line_id")
                .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 3)
                .cast(pl.Int32)
                .alias("day"),
            ]
        )

        # Drop rows where date extraction failed
        df = df.filter(pl.col("year").is_not_null())

        # Filter by entity type if specified
        if entity_type:
            df = df.filter(df["entity_type"] == entity_type)

        # Create date column based on aggregation
        if aggregation == "year":
            df = df.with_columns(pl.col("year").cast(pl.Utf8).alias("date"))
        elif aggregation == "month":
            df = df.with_columns(
                (
                    pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)
                ).alias("date")
            )
        else:  # day
            if "day" in df.columns:
                df = df.with_columns(
                    (
                        pl.col("year").cast(pl.Utf8)
                        + "-"
                        + pl.col("month").cast(pl.Utf8).str.zfill(2)
                        + "-"
                        + pl.col("day").cast(pl.Utf8).str.zfill(2)
                    ).alias("date")
                )
            else:
                # Fall back to month if day not available
                df = df.with_columns(
                    (
                        pl.col("year").cast(pl.Utf8)
                        + "-"
                        + pl.col("month").cast(pl.Utf8).str.zfill(2)
                    ).alias("date")
                )

        # Group by date and entity type
        timeline_df = (
            df.group_by(["date", "entity_type"]).agg(pl.count().alias("count")).sort("date")
        )

        # Convert to dictionary format for frontend
        result_data = {}
        for row in timeline_df.iter_rows(named=True):
            entity_type = row["entity_type"]
            if entity_type not in result_data:
                result_data[entity_type] = []
            result_data[entity_type].append({"date": row["date"], "value": row["count"]})

        return result_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
