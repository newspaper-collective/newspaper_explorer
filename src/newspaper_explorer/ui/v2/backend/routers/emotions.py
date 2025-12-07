"""
Emotion classification endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import date
import polars as pl
from pathlib import Path
import numpy as np

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.analysis.emotions import EmotionRecord
from newspaper_explorer.models.api.emotions import EmotionTimeline

router = APIRouter()

EMOTIONS = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
EMOTION_COLS = [f"{emo}_prob" for emo in EMOTIONS]
EMOTION_MAP = {emo.lower(): emo for emo in EMOTIONS}  # For API params


def get_emotions_df(source_name: str) -> pl.DataFrame:
    config = get_config()
    emotions_dir = Path(config.results_dir) / source_name / "emotions"

    # Try direct path first (flat structure)
    emotions_path = emotions_dir / "emotions.parquet"

    if not emotions_path.exists():
        # Find the most recent run directory (same pattern as keywords, entities, layout)
        run_dirs = [d for d in emotions_dir.glob("*/") if d.is_dir()]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            emotions_path = latest_run / "emotions.parquet"

    if not emotions_path.exists():
        raise HTTPException(status_code=404, detail="No emotion data available")

    return pl.read_parquet(emotions_path)


@router.get("/{source_name}/statistics")
async def get_overall_statistics(source_name: str):
    """Get overall emotion statistics and era analysis"""
    try:
        df = get_emotions_df(source_name)

        # Calculate overall means (return lowercase keys for frontend)
        means = {emo.lower(): df.select(pl.col(f"{emo}_prob")).mean().item() for emo in EMOTIONS}

        # Era analysis
        # Pre-war: < 1914
        # War: 1914-1918
        # Post-war: > 1918

        # Extract year from date column if available, otherwise try to parse from issue_id
        # Assuming issue_id format: {source}_{YYYY-MM-DD}_...
        if "date" not in df.columns:
            # Try to extract date from issue_id
            df = df.with_columns(
                pl.col("issue_id")
                .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
                .str.strptime(pl.Date, "%Y-%m-%d")
                .alias("date")
            )

        df = df.with_columns(pl.col("date").dt.year().alias("year"))

        eras = {
            "pre_war": df.filter(pl.col("year") < 1914),
            "war": df.filter((pl.col("year") >= 1914) & (pl.col("year") <= 1918)),
            "post_war": df.filter(pl.col("year") > 1918),
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/timeline")
async def get_emotion_timeline(
    source_name: str, granularity: str = Query("year", regex="^(year|month)$")
):
    """Get emotion timeline aggregated by year or month"""
    try:
        df = get_emotions_df(source_name)

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/peaks")
async def get_emotion_peaks(
    source_name: str,
    emotion: str = Query(..., regex="^(sadness|love|joy|fear|anger|agitation)$"),
    limit: int = Query(10, ge=1, le=100),
):
    """Get top peaks for a specific emotion"""
    try:
        df = get_emotions_df(source_name)

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
            pl.when(pl.col("year") < 1914)
            .then(pl.lit("pre_war"))
            .when(pl.col("year") <= 1918)
            .then(pl.lit("war"))
            .otherwise(pl.lit("post_war"))
            .alias("era")
        )

        return peaks.to_dicts()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
