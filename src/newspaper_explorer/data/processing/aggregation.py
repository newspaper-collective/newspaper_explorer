"""
Text aggregation utilities for newspaper data.

Functions for aggregating line-level data into text blocks and other
higher-level text units. These operate on already-loaded parquet files.
"""

import json
import logging
from pathlib import Path
import time
from typing import List, Optional, Union

import polars as pl

from newspaper_explorer.models.data.metadata import AggregationMetadata

logger = logging.getLogger(__name__)


def load_and_aggregate_textblocks(
    parquet_path: Union[str, Path],
    group_by: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    auto_save: bool = True,
) -> pl.DataFrame:
    """
    Load parquet file and aggregate text lines by text blocks.

    This function loads a parquet file containing line-level newspaper data
    and aggregates the text lines by text blocks (or other grouping criteria).

    The output will be saved to data/processed/{source_name}/text/textblocks.parquet
    (unless save_path is explicitly provided or auto_save is False).

    All metadata (text_block_id, page_id, date, filename,
    newspaper_title, year, month, day, and spatial information) is preserved.


    Args:
        parquet_path: Path to the parquet file
        group_by: Columns to group by. Default is ["text_block_id"]
                  Since text_block_id is globally unique and contains the date,
                  no additional grouping columns are needed.
        sort_by: Columns to sort by within each group before aggregating.
                 Default is ["y", "x"] to maintain reading order
        save_path: If provided, save the result to this parquet file (overrides auto_save)
        auto_save: If True, automatically save to
                   data/processed/{source}/text/textblocks.parquet

    Returns:
        Polars DataFrame with aggregated text blocks containing:
        - All grouping columns
        - text: concatenated text from all lines in the block
        - line_count: number of lines in the block
        - avg_x, avg_y: average coordinates
        - min_x, min_y, max_x, max_y: bounding box coordinates

    Example:
        >>> from newspaper_explorer.data.loading.aggregation import load_and_aggregate_textblocks
        >>> # Load and auto-save text blocks
        >>> df = load_and_aggregate_textblocks("data/raw/der_tag/text/lines.parquet")
        >>> # Saved to: data/processed/der_tag/text/textblocks.parquet
        >>>
        >>> # Get text blocks for a specific date
        >>> filtered = df.filter(pl.col("date") == "1901-01-08")
        >>>
        >>> # Group by text_block_id alone (now safe!)
        >>> df = load_and_aggregate_textblocks(..., group_by=["text_block_id"])
    """
    parquet_path = Path(parquet_path)
    start_time = time.time()

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logger.info(f"Loading parquet file: {parquet_path}")

    # Load the parquet file
    df = pl.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} lines from parquet")

    # Set default grouping and sorting
    # Note: text_block_id is now globally unique (includes page_id prefix),
    # so no additional grouping columns are needed for uniqueness.
    if group_by is None:
        group_by = ["text_block_id"]

    if sort_by is None:
        sort_by = ["y", "x"]

    # Check if required columns exist
    required_cols = set(group_by + sort_by + ["text"])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Aggregating text blocks by: {group_by}")

    # Sort by reading order within each block
    df = df.sort(group_by + sort_by)

    # Determine which metadata columns to preserve
    metadata_columns = [
        "filename",
        "newspaper_title",
        "year",
        "month",
        "day",
        "date",  # datetime column
        "year_volume",
        "page_count",
        "source_id",
        "issue_id",
        "page_id",
        "page_number",
        "issue_number",
        "daily_issue_number",
    ]

    # Only include columns that actually exist in the DataFrame
    available_metadata = [col for col in metadata_columns if col in df.columns]

    # Aggregate text blocks
    agg_exprs = [
        # Concatenate text with space separator
        pl.col("text").str.concat(" ").alias("text"),
        # Count lines in block
        pl.count().alias("line_count"),
        # Average coordinates
        pl.col("x").mean().alias("avg_x"),
        pl.col("y").mean().alias("avg_y"),
        # Bounding box
        pl.col("x").min().alias("min_x"),
        pl.col("y").min().alias("min_y"),
        pl.col("x").max().alias("max_x"),
        pl.col("y").max().alias("max_y"),
    ]

    # Add metadata columns (take first value from group since they're the same within a block)
    for col in available_metadata:
        agg_exprs.append(pl.col(col).first().alias(col))

    aggregated = df.group_by(group_by, maintain_order=True).agg(agg_exprs)

    logger.info(f"Aggregated into {len(aggregated)} text blocks")

    # Determine save path
    if save_path:
        final_save_path = Path(save_path)
    elif auto_save:
        # Extract source name from path (e.g., "der_tag" from "data/raw/der_tag/text/lines.parquet")
        parts = parquet_path.parts
        try:
            # Find "raw" in path and get the next part as source name
            raw_idx = parts.index("raw")
            source_name = parts[raw_idx + 1]
        except (ValueError, IndexError):
            # Fallback: try to infer from path structure
            logger.warning("Could not extract source name from path, using 'unknown'")
            source_name = "unknown"

        # Construct output path
        final_save_path = Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
        logger.info(f"Auto-save enabled: will save to {final_save_path}")
    else:
        final_save_path = None

    # Save if path is determined
    if final_save_path:
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {final_save_path}")
        aggregated.write_parquet(final_save_path, compression="zstd")
        logger.info(f"Saved {len(aggregated)} text blocks to {final_save_path}")

        # Save aggregation metadata
        end_time = time.time()
        metadata = AggregationMetadata(
            source=source_name,
            aggregation_type="textblock",
            group_by=group_by,
            sort_by=sort_by,
            input_parquet=str(parquet_path),
            output_parquet=str(final_save_path),
            input_row_count=len(df),
            output_row_count=len(aggregated),
            duration_seconds=end_time - start_time,
        )
        metadata_path = final_save_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved aggregation metadata to {metadata_path}")

    return aggregated
