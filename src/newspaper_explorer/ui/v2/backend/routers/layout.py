"""
Layout analysis endpoints
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from newspaper_explorer.config.base import get_config

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{source_name}/results")
async def list_layout_results(source_name: str):
    """List available layout detection results"""
    try:
        config = get_config()
        layout_dir = Path(config.results_dir) / source_name / "layout"

        if not layout_dir.exists():
            return []

        results = []

        # Look for result directories (timestamp-based or named)
        for result_path in layout_dir.iterdir():
            if not result_path.is_dir():
                continue

            detections_file = result_path / "layout.parquet"
            metadata_file = result_path / "layout.json"

            if not detections_file.exists():
                continue  # Load metadata if available
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except:
                    pass

            # Get basic stats from data
            try:
                df = pl.read_parquet(detections_file)
                stats: Dict[str, Any] = {
                    "total_detections": len(df),
                    "unique_pages": df["page_id"].n_unique(),
                    "unique_classes": df["class_name"].n_unique(),
                }

                # Extract date from page_id if available
                if "page_id" in df.columns:
                    try:
                        # page_id format: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}
                        dates = df["page_id"].str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
                        if dates is not None:
                            stats["date_range"] = {
                                "min": str(dates.min()),
                                "max": str(dates.max()),
                            }
                    except:
                        pass
            except:
                stats = {}

            results.append(
                {
                    "path": str(detections_file),
                    "name": result_path.name,
                    "metadata": metadata,
                    "stats": stats,
                }
            )

        # Sort by creation time (newest first)
        def sort_key(x: Dict[str, Any]) -> str:
            return x["metadata"].get("created_at", "") if isinstance(x["metadata"], dict) else ""

        results.sort(key=sort_key, reverse=True)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/pages")
async def get_layout_pages(
    source_name: str,
    run_id: Optional[str] = None,
    result_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    class_filter: Optional[List[str]] = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    """Get pages with layout detections, grouped and paginated"""
    try:
        config = get_config()

        # Determine which results file to use
        if result_path:
            layout_path = Path(result_path)
        elif run_id:
            # Use run_id to find specific result
            layout_path = (
                Path(config.results_dir) / source_name / "layout" / run_id / "layout.parquet"
            )
        else:
            # Try to find the most recent result
            layout_dir = Path(config.results_dir) / source_name / "layout"
            if layout_dir.exists():
                result_dirs = [
                    d
                    for d in layout_dir.iterdir()
                    if d.is_dir() and (d / "layout.parquet").exists()
                ]
                if result_dirs:
                    # Sort by name (which includes timestamp) and get most recent
                    result_dirs.sort(reverse=True)
                    layout_path = result_dirs[0] / "layout.parquet"
                else:
                    raise HTTPException(status_code=404, detail="No layout data available")
            else:
                raise HTTPException(status_code=404, detail="No layout data available")

        if not layout_path.exists():
            raise HTTPException(status_code=404, detail="No layout data available")

        df = pl.read_parquet(layout_path)

        # Apply filters
        if start_date:
            df = df.filter(pl.col("date") >= date.fromisoformat(start_date))
        if end_date:
            df = df.filter(pl.col("date") <= date.fromisoformat(end_date))
        if class_filter:
            df = df.filter(pl.col("class_name").is_in(class_filter))
        if min_confidence > 0:
            df = df.filter(pl.col("confidence") >= min_confidence)

        # Group by page
        pages = []
        unique_pages = df["page_id"].unique().sort().to_list()
        total_pages = len(unique_pages)

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_pages)
        page_ids = unique_pages[start_idx:end_idx]

        for page_id in page_ids:
            page_detections = df.filter(pl.col("page_id") == page_id)

            if len(page_detections) == 0:
                continue

            # Get page metadata from first detection
            first_row = page_detections.row(0, named=True)

            # Parse page_id for metadata: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}
            parts = page_id.split("_")
            page_metadata = {}
            if len(parts) >= 5:
                page_metadata = {
                    "date": parts[1],
                    "issue_number": parts[2],
                    "daily_count": parts[3],
                    "page_number": parts[4],
                }

            # Get image path and convert to relative if needed
            raw_image_path = (
                first_row.get("image_path")
                if isinstance(first_row, dict)
                else first_row["image_path"]
            )

            # Convert absolute path to relative (extract path after /images/)
            # Format: /full/path/data/raw/{source}/images/YYYY/MM/DD/...
            # Convert to: YYYY/MM/DD/...
            if raw_image_path and "/images/" in str(raw_image_path):
                relative_path = str(raw_image_path).split("/images/", 1)[1]
            else:
                relative_path = str(raw_image_path) if raw_image_path else ""

            pages.append(
                {
                    "page_id": page_id,
                    "image_path": relative_path,
                    "detection_count": len(page_detections),
                    "metadata": page_metadata,
                    "detections": [
                        {
                            "detection_id": row["detection_id"],
                            "class_name": row["class_name"],
                            "confidence": row["confidence"],
                            "bbox": {
                                "x1": row["bbox_x1"],
                                "y1": row["bbox_y1"],
                                "x2": row["bbox_x2"],
                                "y2": row["bbox_y2"],
                            },
                            "text_content": (
                                row.get("text_content") if isinstance(row, dict) else None
                            ),
                        }
                        for row in page_detections.iter_rows(named=True)
                    ],
                }
            )

        return {
            "total": total_pages,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/stats")
async def get_layout_stats(
    source_name: str,
    run_id: Optional[str] = None,
    label: Optional[str] = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Get aggregated statistics for layout detections"""
    try:
        config = get_config()
        layout_dir = Path(config.results_dir) / source_name / "layout"

        if not layout_dir.exists():
            return {"total": 0, "timeline": [], "counts": {}}

        # Determine which layout file to use
        if run_id:
            layout_path = layout_dir / run_id / "layout.parquet"
            enriched_path = layout_dir / run_id / "layout_enriched.parquet"
        else:
            result_dirs = [
                d for d in layout_dir.iterdir() if d.is_dir() and (d / "layout.parquet").exists()
            ]
            if not result_dirs:
                return {"total": 0, "timeline": [], "counts": {}}
            result_dirs.sort(reverse=True)
            layout_path = result_dirs[0] / "layout.parquet"
            enriched_path = result_dirs[0] / "layout_enriched.parquet"

        # Prefer enriched path
        data_path = enriched_path if enriched_path.exists() else layout_path

        if not data_path.exists():
            return {"total": 0, "timeline": [], "counts": {}}

        # Scan parquet for efficient aggregation
        lf = pl.scan_parquet(data_path)

        # Apply filters
        if label:
            lf = lf.filter(pl.col("class_name") == label)
        if min_confidence > 0:
            lf = lf.filter(pl.col("confidence") >= min_confidence)

        # Extract date from page_id
        lf = lf.with_columns(
            pl.col("page_id").str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1).alias("date")
        )

        # 1. Total count
        total_count = lf.select(pl.len()).collect().item()

        # 2. Timeline (aggregated by month)
        # Filter out null dates
        timeline_lf = lf.filter(pl.col("date").is_not_null())

        # Convert to date and truncate to month
        timeline_lf = timeline_lf.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d").dt.truncate("1mo").alias("period")
        )

        # Group by period
        timeline_df = (
            timeline_lf.group_by("period").agg(pl.len().alias("count")).sort("period").collect()
        )

        timeline = [
            {"date": str(row["period"]), "value": row["count"]}
            for row in timeline_df.iter_rows(named=True)
        ]

        # 3. Counts by class (if no label filter)
        counts = {}
        if not label:
            counts_df = lf.group_by("class_name").agg(pl.len().alias("count")).collect()
            counts = {row["class_name"]: row["count"] for row in counts_df.iter_rows(named=True)}

        # 3.5. Aggregate statistics (unique pages, averages)
        # Calculate unique page count
        unique_pages_count = lf.select(pl.col("page_id").n_unique()).collect().item()

        # Calculate average confidence
        avg_conf = lf.select(pl.col("confidence").mean()).collect().item()

        # Calculate average width and height
        avg_stats_df = lf.select(
            [
                (pl.col("bbox_x2") - pl.col("bbox_x1")).mean().alias("avg_width"),
                (pl.col("bbox_y2") - pl.col("bbox_y1")).mean().alias("avg_height"),
            ]
        ).collect()

        avg_width = avg_stats_df["avg_width"][0] if len(avg_stats_df) > 0 else 0
        avg_height = avg_stats_df["avg_height"][0] if len(avg_stats_df) > 0 else 0

        # 4. Confidence Distribution (Histogram)
        # Use simple binning: multiply by 20 and round to get bin index (0-19)
        conf_dist_df = (
            lf.select((pl.col("confidence") * 20).round(0).cast(pl.Int32).alias("bin_idx"))
            .group_by("bin_idx")
            .agg(pl.len().alias("count"))
            .collect()
        )

        # Format confidence histogram
        confidence_dist = {"bins": [f"{(i / 20):.2f}" for i in range(20)], "counts": [0] * 20}

        for row in conf_dist_df.iter_rows(named=True):
            idx = row["bin_idx"]
            if 0 <= idx < 20:
                confidence_dist["counts"][idx] = row["count"]

        # 5. Size Distribution (all data points)
        size_df = lf.select(
            [
                (pl.col("bbox_x2") - pl.col("bbox_x1")).alias("width"),
                (pl.col("bbox_y2") - pl.col("bbox_y1")).alias("height"),
            ]
        ).filter((pl.col("width") > 0) & (pl.col("height") > 0))

        size_data = size_df.collect()

        size_distribution = [
            [row["width"], row["height"]] for row in size_data.iter_rows(named=True)
        ]

        # 6. Position Distribution (Vertical position on page)
        # Normalize Y center by page height?
        # We don't have page height easily available in detections without joining with image size
        # But we can approximate using the max Y on the page or just raw Y if pages are similar
        # Or we can just return raw Y center distribution
        # Let's return raw Y center for now, or maybe skip if too complex without page info
        # Actually, user wants "normalized position".
        # Without page height, we can't normalize accurately.
        # Let's skip position distribution for now or use a heuristic (e.g. assume A4 height ~2000-3000px)
        # Or we can try to get max Y per page from the data itself

        # Estimate page height per page as max(y2) of any detection on that page
        # This is expensive to compute on the fly.
        # Let's omit position distribution for now to avoid performance hit,
        # or just return raw Y centers.
        # Let's return raw Y centers binned

        # Bin Y centers into 20 bins (0-4000px?)
        # Let's just use raw Y center histogram
        y_center_df = (
            lf.select(((pl.col("bbox_y1") + pl.col("bbox_y2")) / 2).alias("y_center"))
            .select(
                (pl.col("y_center") / 200).round(0).cast(pl.Int32).alias("bin_idx")
            )  # 200px bins
            .group_by("bin_idx")
            .agg(pl.len().alias("count"))
            .collect()
        )

        position_dist = []
        for row in y_center_df.sort("bin_idx").iter_rows(named=True):
            position_dist.append({"bin_start": row["bin_idx"] * 200, "count": row["count"]})

        return {
            "total": total_count,
            "unique_pages": unique_pages_count,
            "avg_confidence": avg_conf,
            "avg_width": avg_width,
            "avg_height": avg_height,
            "timeline": timeline,
            "counts": counts,
            "confidence_distribution": confidence_dist,
            "size_distribution": size_distribution,
            "position_distribution": position_dist,
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/detections")
async def get_layout_detections(
    source_name: str,
    page_id: Optional[str] = None,
    label: Optional[str] = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: Optional[int] = Query(default=None, ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    run_id: Optional[str] = None,
    include_captions: bool = Query(default=False),
    exclude_headers_footers: bool = Query(default=False),
    header_footer_threshold: float = Query(default=10.0, ge=0.0, le=50.0),
    only_with_captions: bool = Query(default=False),
    min_height: float = Query(default=0.0, ge=0.0),
    search: Optional[str] = Query(default=None),
):
    """Get detected layout regions with optional filtering and pagination"""
    try:
        config = get_config()
        layout_dir = Path(config.results_dir) / source_name / "layout"
        if not layout_dir.exists():
            raise HTTPException(status_code=404, detail="No layout data available")

        # Determine which layout file to use
        if run_id:
            # Use run_id to find specific result
            layout_path = layout_dir / run_id / "layout.parquet"
            enriched_path = layout_dir / run_id / "layout_enriched.parquet"
        else:
            # Try to find the most recent result
            result_dirs = [
                d for d in layout_dir.iterdir() if d.is_dir() and (d / "layout.parquet").exists()
            ]
            if not result_dirs:
                raise HTTPException(status_code=404, detail="No layout data available")

            # Sort by name (which includes timestamp) and get most recent
            result_dirs.sort(reverse=True)
            layout_path = result_dirs[0] / "layout.parquet"
            enriched_path = result_dirs[0] / "layout_enriched.parquet"

        # Prefer enriched path if available
        # But REQUIRE it if only_with_captions filter is active
        if only_with_captions and not enriched_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Caption data not available. Run caption enrichment first: 'newspaper-explorer analyze captions enrich' and 'newspaper-explorer analyze captions match'",
            )

        data_path = enriched_path if enriched_path.exists() else layout_path

        if not data_path.exists():
            raise HTTPException(status_code=404, detail="No layout data available")

        # Load data
        # We use lazy loading for better performance with large files
        lf = pl.scan_parquet(data_path)

        # Apply filters
        if page_id:
            lf = lf.filter(pl.col("page_id") == page_id)
        if label:
            lf = lf.filter(pl.col("class_name") == label)
        if min_confidence > 0:
            lf = lf.filter(pl.col("confidence") >= min_confidence)

        # Apply additional filters
        if min_height > 0:
            # Filter by minimum height
            lf = lf.filter((pl.col("bbox_y2") - pl.col("bbox_y1")) >= min_height)

        if exclude_headers_footers:
            # Exclude detections in top/bottom threshold% of page
            # But exempt large images (>30% of page) which may legitimately span into H/F zones

            # Calculate image dimensions and center Y
            lf = lf.with_columns(
                [
                    ((pl.col("bbox_y1") + pl.col("bbox_y2")) / 2).alias("center_y"),
                    (pl.col("bbox_y2") - pl.col("bbox_y1")).alias("detection_height"),
                ]
            )

            # Try to load image index for accurate page heights
            page_heights_available = False
            try:
                from newspaper_explorer.data.indexing.image_index import ImageIndexer

                indexer = ImageIndexer(source_name)
                image_index_df = indexer.load_index()

                if (
                    image_index_df is not None
                    and "page_id" in image_index_df.columns
                    and "height" in image_index_df.columns
                ):
                    # Join with image index to get actual page heights
                    image_index_lf = pl.LazyFrame(image_index_df).select(
                        [pl.col("page_id"), pl.col("height").alias("page_height")]
                    )

                    lf = lf.join(image_index_lf, on="page_id", how="left")
                    page_heights_available = True
                    logger.info(
                        "Using actual page heights from image index for header/footer filtering"
                    )
                else:
                    logger.warning("Image index exists but missing required columns")
            except Exception as e:
                logger.warning(f"Could not load image index: {e}")

            # Apply filtering
            threshold_fraction = header_footer_threshold / 100

            if page_heights_available:
                # Use actual page heights with fallback
                lf = lf.with_columns(
                    [
                        pl.when(pl.col("page_height").is_null())
                        .then(pl.lit(3000))
                        .otherwise(pl.col("page_height"))
                        .alias("page_height_final"),
                    ]
                )

                # Calculate if detection is large (>30% of page) and thresholds
                lf = lf.with_columns(
                    [
                        ((pl.col("detection_height") / pl.col("page_height_final")) > 0.3).alias(
                            "is_large"
                        ),
                        (pl.col("page_height_final") * threshold_fraction).alias("top_threshold"),
                        (pl.col("page_height_final") * (1 - threshold_fraction)).alias(
                            "bottom_threshold"
                        ),
                    ]
                )

                # Keep if: large detection OR center not in H/F zone
                lf = lf.filter(
                    pl.col("is_large")
                    | (
                        (pl.col("center_y") > pl.col("top_threshold"))
                        & (pl.col("center_y") < pl.col("bottom_threshold"))
                    )
                )
            else:
                # Fallback to estimated page height
                est_height = 3000
                lf = lf.with_columns(
                    ((pl.col("detection_height") / est_height) > 0.3).alias("is_large")
                )

                top_thresh = est_height * threshold_fraction
                bot_thresh = est_height * (1 - threshold_fraction)

                lf = lf.filter(
                    pl.col("is_large")
                    | ((pl.col("center_y") > top_thresh) & (pl.col("center_y") < bot_thresh))
                )

        if only_with_captions:
            # Filter to only include pictures with captions
            # Check if caption_id column exists (only in enriched files)
            schema = lf.collect_schema()
            has_caption_id = "caption_id" in schema.names()
            has_text_content = "text_content" in schema.names()

            if has_caption_id:
                # Primary method: Use caption_id (most reliable)
                lf = lf.filter(pl.col("caption_id").is_not_null())
            elif has_text_content:
                # Fallback: Use text_content (may include other enriched text)
                lf = lf.filter(
                    pl.col("text_content").is_not_null() & (pl.col("text_content") != "")
                )
            else:
                # This should not happen due to the check above, but log it anyway
                logger.error(
                    "Cannot filter by captions: columns not found despite enriched file check"
                )
                raise HTTPException(
                    status_code=500, detail="Caption data columns not found in enriched file"
                )

        # Apply search filter (searches in page_id and text_content)
        if search:
            schema = lf.collect_schema()
            has_text_content = "text_content" in schema.names()

            search_lower = search.lower()

            # Build search condition
            if has_text_content:
                # Search in both page_id and text_content
                lf = lf.filter(
                    pl.col("page_id").str.to_lowercase().str.contains(search_lower)
                    | (
                        pl.col("text_content").is_not_null()
                        & pl.col("text_content").str.to_lowercase().str.contains(search_lower)
                    )
                )
            else:
                # Only search in page_id
                lf = lf.filter(pl.col("page_id").str.to_lowercase().str.contains(search_lower))

        # Sort by date (extracted from page_id) and then by page_id
        # Extract date for sorting if not present
        # Note: This regex extraction in scan mode might be tricky if schema doesn't match
        # For now, let's sort by page_id which usually contains date
        lf = lf.sort("page_id")

        # Pagination
        # Calculate total count first
        total_count = lf.select(pl.len()).collect().item()

        # Apply limit/pagination
        offset = (page - 1) * page_size

        if limit:
            # If explicit limit is set, override pagination
            lf = lf.head(limit)
        else:
            lf = lf.slice(offset, page_size)

        # Collect results
        df = lf.collect()

        # Convert to dict format with bbox structure
        results = []
        for row in df.iter_rows(named=True):
            result = {
                "detection_id": row["detection_id"],
                "page_id": row["page_id"],
                "class_name": row["class_name"],
                "confidence": row["confidence"],
                "bbox": {
                    "x1": row["bbox_x1"],
                    "y1": row["bbox_y1"],
                    "x2": row["bbox_x2"],
                    "y2": row["bbox_y2"],
                },
                "image_path": row.get("image_path"),
            }

            # Include date if available (might need extraction if not in parquet)
            if "date" in row and row["date"] is not None:
                result["date"] = row["date"]
            elif "page_id" in row:
                # Try simple extraction from page_id: {source}_{YYYY-MM-DD}_...
                parts = row["page_id"].split("_")
                if len(parts) > 1 and len(parts[1]) == 10:  # Simple check for YYYY-MM-DD
                    result["date"] = parts[1]

            # Include text_content if available
            if "text_content" in row and row["text_content"] is not None:
                result["text_content"] = row["text_content"]

            # Include caption info if available (pre-calculated)
            if "caption_id" in row and row["caption_id"] is not None:
                result["caption_id"] = row["caption_id"]

            if "caption_bbox" in row and row["caption_bbox"] is not None:
                result["caption_bbox"] = row["caption_bbox"]

            # Fallback for legacy 'caption_text' column
            if "caption_text" in row and row["caption_text"] is not None:
                result["caption_text"] = row["caption_text"]

            results.append(result)

        return {"total": total_count, "page": page, "page_size": page_size, "items": results}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No layout data available")
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/labels")
async def get_layout_labels(
    source_name: str,
    run_id: Optional[str] = None,
    result_path: Optional[str] = None,
):
    """Get list of unique layout region labels"""
    try:
        config = get_config()

        # Determine which results file to use
        if result_path:
            layout_path = Path(result_path)
        elif run_id:
            # Use run_id to find specific result
            layout_path = (
                Path(config.results_dir) / source_name / "layout" / run_id / "layout.parquet"
            )
        else:
            # Try to find the most recent result
            layout_dir = Path(config.results_dir) / source_name / "layout"
            if not layout_dir.exists():
                return []

            result_dirs = [
                d for d in layout_dir.iterdir() if d.is_dir() and (d / "layout.parquet").exists()
            ]
            if not result_dirs:
                return []

            # Sort by name (which includes timestamp) and get most recent
            result_dirs.sort(reverse=True)
            layout_path = result_dirs[0] / "layout.parquet"

        if not layout_path.exists():
            return []

        df = pl.read_parquet(layout_path)
        labels = df["class_name"].unique().to_list()
        return sorted(labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/timeline")
async def get_layout_timeline(
    source_name: str,
    run_id: Optional[str] = None,
    result_path: Optional[str] = None,
    aggregation: str = Query(default="month", regex="^(day|week|month|year)$"),
):
    """Get timeline of layout detections by class"""
    try:
        config = get_config()

        # Determine which results file to use
        if result_path:
            layout_path = Path(result_path)
        elif run_id:
            # Use run_id to find specific result
            layout_path = (
                Path(config.results_dir) / source_name / "layout" / run_id / "layout.parquet"
            )
        else:
            # Try to find the most recent result
            layout_dir = Path(config.results_dir) / source_name / "layout"
            if not layout_dir.exists():
                return {}

            result_dirs = [
                d for d in layout_dir.iterdir() if d.is_dir() and (d / "layout.parquet").exists()
            ]
            if not result_dirs:
                return {}

            # Sort by name (which includes timestamp) and get most recent
            result_dirs.sort(reverse=True)
            layout_path = result_dirs[0] / "layout.parquet"

        if not layout_path.exists():
            return {}

        df = pl.read_parquet(layout_path)

        # Extract date from page_id
        df = df.with_columns(
            pl.col("page_id").str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1).alias("date")
        )

        # Filter out rows without dates
        df = df.filter(pl.col("date").is_not_null())

        # Convert to date type
        df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

        # Aggregate by time period and class
        if aggregation == "day":
            df = df.with_columns(pl.col("date").alias("period"))
        elif aggregation == "week":
            df = df.with_columns(pl.col("date").dt.truncate("1w").alias("period"))
        elif aggregation == "month":
            df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("period"))
        elif aggregation == "year":
            df = df.with_columns(pl.col("date").dt.truncate("1y").alias("period"))

        # Group by period and class_name
        timeline = (
            df.group_by(["period", "class_name"])
            .agg(pl.count("detection_id").alias("count"))
            .sort("period")
        )

        # Convert to nested dict structure {class_name: [{date, value}]}
        result = {}
        for row in timeline.iter_rows(named=True):
            class_name = row["class_name"]
            if class_name not in result:
                result[class_name] = []
            result[class_name].append({"date": str(row["period"]), "value": row["count"]})

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
