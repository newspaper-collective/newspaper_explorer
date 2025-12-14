"""
Query and aggregation functions for emotion analysis results.

These functions operate on emotion DataFrames to compute statistics,
timelines, peaks, and era comparisons. Used by both CLI and API.
"""

from typing import Any

import polars as pl

from newspaper_explorer.models.analysis.emotions import (
    EMOTION_MAP,
    EMOTIONS,
    ERA_PRE_WAR_END,
    ERA_WAR_END,
)


def get_statistics(df: pl.DataFrame) -> dict[str, Any]:
    """
    Calculate overall emotion statistics and era analysis.

    Args:
        df: Emotion predictions DataFrame with emotion probability columns

    Returns:
        Dictionary with overall_means, era_statistics, and total_records
    """
    # Calculate overall means (return lowercase keys for frontend)
    means = {emo.lower(): df.select(pl.col(f"{emo}_prob")).mean().item() for emo in EMOTIONS}

    # Extract year from date column if available, otherwise try to parse from issue_id
    if "date" not in df.columns:
        # Try to extract date from issue_id format: {source}_{YYYY-MM-DD}_...
        df = df.with_columns(
            pl.col("issue_id")
            .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date")
        )

    df = df.with_columns(pl.col("date").dt.year().alias("year"))

    # Era analysis: Pre-war < 1914, War 1914-1918, Post-war > 1918
    eras = {
        "pre_war": df.filter(pl.col("year") < ERA_PRE_WAR_END),
        "war": df.filter((pl.col("year") >= ERA_PRE_WAR_END) & (pl.col("year") <= ERA_WAR_END)),
        "post_war": df.filter(pl.col("year") > ERA_WAR_END),
    }

    era_stats = {}
    for era_name, era_df in eras.items():
        if era_df.height == 0:
            continue

        stats = {}
        for emo in EMOTIONS:
            col = f"{emo}_prob"
            stats[emo] = {
                "mean": era_df.select(pl.col(col)).mean().item(),
                "std": era_df.select(pl.col(col)).std().item(),
                "min": era_df.select(pl.col(col)).min().item(),
                "max": era_df.select(pl.col(col)).max().item(),
                # For box plots
                "q1": era_df.select(pl.col(col)).quantile(0.25).item(),
                "median": era_df.select(pl.col(col)).median().item(),
                "q3": era_df.select(pl.col(col)).quantile(0.75).item(),
            }
        era_stats[era_name] = stats

    return {"overall_means": means, "era_statistics": era_stats, "total_records": df.height}


def get_timeline(df: pl.DataFrame, granularity: str = "year") -> list[dict[str, Any]]:
    """
    Get emotion values aggregated over time.

    Args:
        df: Emotion predictions DataFrame
        granularity: Time granularity - "year" or "month"

    Returns:
        List of dictionaries with time_key and emotion values
    """
    if "date" not in df.columns:
        df = df.with_columns(
            pl.col("issue_id")
            .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date")
        )

    # Group by date
    if granularity == "year":
        df_agg = df.with_columns(pl.col("date").dt.year().alias("time_key"))
    else:
        df_agg = df.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("time_key"))

    # Calculate means for each emotion (use lowercase for frontend)
    agg_exprs = [pl.col(f"{emo}_prob").mean().alias(emo.lower()) for emo in EMOTIONS]
    result = df_agg.group_by("time_key").agg(agg_exprs).sort("time_key")

    return result.to_dicts()


def get_peaks(df: pl.DataFrame, emotion: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Get top peaks for a specific emotion.

    Args:
        df: Emotion predictions DataFrame
        emotion: Emotion name (lowercase: sadness, love, joy, fear, anger, agitation)
        limit: Maximum number of peaks to return

    Returns:
        List of peak records with emotion scores and era information
    """
    if "date" not in df.columns:
        df = df.with_columns(
            pl.col("issue_id")
            .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
            .str.strptime(pl.Date, "%Y-%m-%d")
            .alias("date")
        )

    # Map lowercase param to capitalized column
    emotion_capitalized = EMOTION_MAP[emotion]
    col = f"{emotion_capitalized}_prob"

    # Get top N records
    peaks = df.sort(col, descending=True).head(limit)

    # Add era info
    peaks = peaks.with_columns(pl.col("date").dt.year().alias("year")).with_columns(
        pl.when(pl.col("year") < ERA_PRE_WAR_END)
        .then(pl.lit("pre_war"))
        .when(pl.col("year") <= ERA_WAR_END)
        .then(pl.lit("war"))
        .otherwise(pl.lit("post_war"))
        .alias("era")
    )

    return peaks.to_dicts()
