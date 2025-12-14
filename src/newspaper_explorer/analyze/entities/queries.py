"""
Centralized entity analysis query functions.

Provides reusable query/aggregation logic for entity statistics, filtering,
and timeline generation. Used by both CLI and FastAPI backend.
"""

from typing import Any, Optional

import polars as pl


def aggregate_entities(
    df: pl.DataFrame,
    entity_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Aggregate entities by text and type with detection statistics.

    Args:
        df: DataFrame with entity detection records
        entity_type: Optional filter for specific entity type
        limit: Optional limit on number of results

    Returns:
        DataFrame with aggregated entity statistics
    """
    if df.is_empty():
        return pl.DataFrame()

    # Filter by entity type if specified
    if entity_type:
        df = df.filter(pl.col("entity_type") == entity_type)

    # Aggregate by entity text and type
    aggregated = (
        df.group_by(["entity_text", "entity_type"])
        .agg(
            [
                pl.count().alias("detection_count"),
                pl.col("confidence").mean().alias("avg_confidence"),
                pl.col("confidence").min().alias("min_confidence"),
                pl.col("confidence").max().alias("max_confidence"),
                pl.col("text_block_id").n_unique().alias("unique_blocks"),
                pl.col("text_block_id").unique().alias("text_block_ids"),
            ]
        )
        .sort("detection_count", descending=True)
    )

    # Apply limit if specified
    if limit is not None:
        aggregated = aggregated.head(limit)

    return aggregated


def get_entity_types(df: pl.DataFrame) -> list[str]:
    """
    Get sorted list of unique entity types.

    Args:
        df: DataFrame with entity detection records

    Returns:
        Sorted list of entity type strings
    """
    if df.is_empty():
        return []

    types = df["entity_type"].unique().to_list()
    return sorted(types)


def get_entity_occurrences(
    df: pl.DataFrame,
    entity_text: str,
    entity_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Get individual occurrences of a specific entity.

    Args:
        df: DataFrame with entity detection records
        entity_text: Exact entity text to search for
        entity_type: Optional filter for specific entity type
        limit: Optional limit on number of results

    Returns:
        DataFrame with filtered occurrences
    """
    if df.is_empty():
        return pl.DataFrame()

    # Filter by entity text (exact match)
    filtered = df.filter(pl.col("entity_text") == entity_text)

    # Filter by entity type if specified
    if entity_type:
        filtered = filtered.filter(pl.col("entity_type") == entity_type)

    # Apply limit if specified
    if limit is not None:
        filtered = filtered.head(limit)

    return filtered


def get_timeline(
    df: pl.DataFrame,
    entity_type: Optional[str] = None,
    aggregation: str = "month",
) -> dict[str, list[dict[str, Any]]]:
    """
    Get entity counts over time, aggregated by day/month/year.

    Extracts dates from text_block_id format: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}_{block}

    Args:
        df: DataFrame with entity detection records
        entity_type: Optional filter for specific entity type
        aggregation: Time granularity - "day", "month", or "year"

    Returns:
        Dictionary mapping entity types to timeline data:
        {
            "PERSON": [{"date": "2020-01", "value": 42}, ...],
            "LOCATION": [{"date": "2020-01", "value": 15}, ...]
        }
    """
    if df.is_empty() or "text_block_id" not in df.columns:
        return {}

    # Extract date components from text_block_id
    df = df.with_columns(
        [
            pl.col("text_block_id")
            .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 1)
            .cast(pl.Int32)
            .alias("year"),
            pl.col("text_block_id")
            .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 2)
            .cast(pl.Int32)
            .alias("month"),
            pl.col("text_block_id")
            .str.extract(r"_(\d{4})-(\d{2})-(\d{2})_", 3)
            .cast(pl.Int32)
            .alias("day"),
        ]
    )

    # Drop rows where date extraction failed
    df = df.filter(pl.col("year").is_not_null())

    # Filter by entity type if specified
    if entity_type:
        df = df.filter(pl.col("entity_type") == entity_type)

    # Create date column based on aggregation level
    if aggregation == "year":
        df = df.with_columns(pl.col("year").cast(pl.Utf8).alias("date"))
    elif aggregation == "month":
        df = df.with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.zfill(2)).alias(
                "date"
            )
        )
    else:  # day
        df = df.with_columns(
            (
                pl.col("year").cast(pl.Utf8)
                + "-"
                + pl.col("month").cast(pl.Utf8).str.zfill(2)
                + "-"
                + pl.col("day").cast(pl.Utf8).str.zfill(2)
            ).alias("date")
        )

    # Group by date and entity type
    timeline_df = df.group_by(["date", "entity_type"]).agg(pl.count().alias("count")).sort("date")

    # Convert to nested dictionary format
    result: dict[str, list[dict[str, Any]]] = {}
    for row in timeline_df.iter_rows(named=True):
        ent_type = row["entity_type"]
        if ent_type not in result:
            result[ent_type] = []
        result[ent_type].append({"date": row["date"], "value": row["count"]})

    return result
