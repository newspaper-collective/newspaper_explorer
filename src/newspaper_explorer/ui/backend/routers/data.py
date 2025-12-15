"""
Data access endpoints for text, issues, and pages
"""

from datetime import date
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.indexing.image_index import ImageIndexer
from newspaper_explorer.models.api.content import Issue, Line, Page, TextBlock
from newspaper_explorer.models.api.filters import DateFilter, PaginationParams

router = APIRouter()


@router.get("/{source_name}/pages", response_model=list[Page])
async def get_pages(
    source_name: str,
    issue_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=1000),
):
    """Get list of pages with optional filtering"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        parquet_files = list(text_path.glob("*.parquet"))
        if not parquet_files:
            return []

        df = pl.read_parquet(parquet_files[0])

        if df is None or df.height == 0:
            return []

        # Apply filters
        if issue_id:
            df = df.filter(pl.col("issue_id") == issue_id)
        if start_date:
            df = df.filter(pl.col("date") >= start_date)
        if end_date:
            df = df.filter(pl.col("date") <= end_date)

        # Group by page
        pages_df = (
            df.group_by("page_id")
            .agg(
                [
                    pl.col("issue_id").first().alias("issue_id"),
                    pl.col("date").first().alias("date"),
                    pl.col("newspaper_title").first().alias("newspaper_title"),
                    pl.col("page_number").first().alias("page_number"),
                    pl.col("text").str.concat(" ").str.slice(0, 200).alias("text_preview"),
                ]
            )
            .sort("date", descending=True)
        )

        # Pagination
        offset = (page - 1) * page_size
        pages_df = pages_df.slice(offset, page_size)

        # Load image index for this source
        image_index = None
        try:
            image_indexer = ImageIndexer(source_name)
            image_index = image_indexer.load_index()
        except (FileNotFoundError, ValueError) as e:
            logging.debug(f"Image index not available for {source_name}: {e}")

        # Convert to response model
        pages = []
        for row in pages_df.iter_rows(named=True):
            page_id = str(row["page_id"])
            issue_id = str(row["issue_id"])
            page_number = int(row["page_number"])
            image_url = None
            has_image = False

            # Try to find image in index
            alto_width, alto_height = None, None
            image_width, image_height = None, None

            if image_index is not None and len(image_index) > 0:
                try:
                    # Filter by issue_id and page_number
                    img_row = image_index.filter(
                        (pl.col("issue_id") == issue_id) & (pl.col("page_number") == page_number)
                    )

                    if len(img_row) > 0:
                        image_path = img_row["image_path"][0]
                        image_url = f"/static/{source_name}/images/{image_path}"
                        has_image = True
                        # Get dimensions for coordinate scaling
                        alto_width = img_row["alto_width"][0] if img_row["alto_width"][0] else None
                        alto_height = (
                            img_row["alto_height"][0] if img_row["alto_height"][0] else None
                        )
                        image_width = img_row["width"][0] if img_row["width"][0] else None
                        image_height = img_row["height"][0] if img_row["height"][0] else None
                except Exception:
                    pass  # Image not found, keep None

            pages.append(
                Page(
                    page_id=page_id,
                    issue_id=issue_id,
                    date=row["date"],
                    newspaper_title=str(row["newspaper_title"]),
                    page_number=page_number,
                    text_preview=row["text_preview"],
                    image_url=image_url,
                    has_image=has_image,
                    alto_width=alto_width,
                    alto_height=alto_height,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        return pages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/text-blocks", response_model=list[TextBlock])
async def get_text_blocks(
    source_name: str,
    page_id: Optional[str] = None,
    issue_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=1000),
):
    """Get aggregated text blocks with optional filtering"""
    try:
        config = get_config()
        # Try processed directory first (aggregated blocks)
        processed_path = Path(config.data_dir) / "processed" / source_name / "text"
        raw_path = Path(config.data_dir) / "raw" / source_name / "text"

        # Look for textblocks.parquet in processed directory
        textblocks_file = processed_path / "textblocks.parquet"
        if textblocks_file.exists():
            df = pl.read_parquet(textblocks_file)
        else:
            # Fall back to raw lines and aggregate on the fly
            lines_file = raw_path / f"{source_name}_lines.parquet"
            if not lines_file.exists():
                return []

            df = pl.read_parquet(lines_file)

            # Apply filters first for efficiency
            if page_id:
                df = df.filter(pl.col("page_id") == page_id)
            if issue_id:
                df = df.filter(pl.col("issue_id") == issue_id)

            # Aggregate lines into blocks
            df = df.sort(["text_block_id", "y", "x"])
            df = df.group_by("text_block_id", maintain_order=True).agg(
                [
                    pl.col("text").str.join(delimiter=" ").alias("text"),
                    pl.col("page_id").first(),
                    pl.col("issue_id").first(),
                    pl.col("date").first(),
                    pl.col("x").min().alias("x"),
                    pl.col("y").min().alias("y"),
                    (pl.col("x").max() - pl.col("x").min() + pl.col("width").first()).alias(
                        "width"
                    ),
                    (pl.col("y").max() - pl.col("y").min() + pl.col("height").first()).alias(
                        "height"
                    ),
                ]
            )

        if df is None or df.height == 0:
            return []

        # Apply filters if using pre-aggregated data
        if textblocks_file.exists():
            if page_id:
                df = df.filter(pl.col("page_id") == page_id)
            if issue_id:
                df = df.filter(pl.col("issue_id") == issue_id)

        # Pagination
        offset = (page - 1) * page_size
        df = df.slice(offset, page_size)

        # Convert to response model
        blocks = []
        for row in df.iter_rows(named=True):
            blocks.append(
                TextBlock(
                    text_block_id=row["text_block_id"],
                    page_id=row["page_id"],
                    issue_id=row["issue_id"],
                    date=row["date"],
                    text=row["text"],
                    x=int(row.get("x", 0)),
                    y=int(row.get("y", 0)),
                    width=int(row.get("width", 0)),
                    height=int(row.get("height", 0)),
                )
            )

        return blocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/lines", response_model=list[Line])
async def get_lines(
    source_name: str,
    page_id: Optional[str] = None,
    issue_id: Optional[str] = None,
    text_block_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=1000, ge=1, le=5000),
):
    """Get individual text lines with bounding boxes"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        lines_file = text_path / f"{source_name}_lines.parquet"
        if not lines_file.exists():
            return []

        df = pl.read_parquet(lines_file)

        if df is None or df.height == 0:
            return []

        # Apply filters
        if page_id:
            df = df.filter(pl.col("page_id") == page_id)
        if issue_id:
            df = df.filter(pl.col("issue_id") == issue_id)
        if text_block_id:
            df = df.filter(pl.col("text_block_id") == text_block_id)

        # Sort by reading order
        df = df.sort(["y", "x"])

        # Pagination
        offset = (page - 1) * page_size
        df = df.slice(offset, page_size)

        # Convert to response model
        lines = []
        for row in df.iter_rows(named=True):
            lines.append(
                Line(
                    line_id=row["line_id"],
                    text=row["text"],
                    text_block_id=row["text_block_id"],
                    page_id=row["page_id"],
                    issue_id=row["issue_id"],
                    date=row["date"],
                    newspaper_title=row["newspaper_title"],
                    year_volume=row["year_volume"],
                    page_number=row["page_number"],
                    x=int(row["x"]),
                    y=int(row["y"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )

        return lines
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/random-lines", response_model=list[dict])
async def get_random_lines(
    source_name: str,
    count: int = Query(default=10, ge=1, le=100, description="Number of random lines to return"),
):
    """Get random text lines from the data as dictionaries"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        # Find parquet files (look for lines file first, then any parquet)
        lines_file = text_path / f"{source_name}_lines.parquet"
        if lines_file.exists():
            df = pl.read_parquet(lines_file)
        else:
            parquet_files = list(text_path.glob("*.parquet"))
            if not parquet_files:
                return []
            df = pl.read_parquet(parquet_files[0])

        if df is None or df.height == 0:
            return []

        # Sample random lines
        sample_size = min(count, df.height)
        sample_df = df.sample(n=sample_size)

        # Convert to list of dicts
        return sample_df.to_dicts()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/random-images", response_model=list[dict])
async def get_random_images(
    source_name: str,
    count: int = 10,
) -> list[dict]:
    """Get random image URLs with metadata from the specified source"""
    try:
        config = get_config()
        index_path = Path(config.data_dir) / "raw" / source_name / "image_index.parquet"

        if not index_path.exists():
            return []

        # Load image index
        df = pl.read_parquet(index_path)

        # Filter to existing files only
        df = df.filter(pl.col("file_exists") == True)

        if df.height == 0:
            return []

        # Sample random images
        sample_size = min(count, df.height)
        sample_df = df.sample(n=sample_size)

        # Convert to list of dicts with URLs
        results = []
        for row in sample_df.iter_rows(named=True):
            results.append(
                {
                    "url": f"/static/{source_name}/images/{row['image_path']}",
                    "date": row["date"],
                    "page_number": row["page_number"],
                    "newspaper_title": row.get("newspaper_title"),
                    "issue_id": row.get("issue_id"),
                }
            )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/browse/years")
async def browse_years(
    source_name: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort_order: str = Query(default="asc", regex="^(asc|desc)$"),
):
    """Browse newspapers by year with aggregated statistics"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        parquet_files = list(text_path.glob("*.parquet"))
        if not parquet_files:
            return []

        df = pl.read_parquet(parquet_files[0])

        if df is None or df.height == 0:
            return []

        # Get year range from data if not specified
        if year_from is None or year_to is None:
            date_stats = df.select(
                [pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
            ).to_dicts()[0]
            if year_from is None and date_stats["min_date"]:
                year_from = date_stats["min_date"].year
            if year_to is None and date_stats["max_date"]:
                year_to = date_stats["max_date"].year

        # Apply year filter
        if year_from:
            df = df.filter(pl.col("year") >= year_from)
        if year_to:
            df = df.filter(pl.col("year") <= year_to)

        # Group by year
        years_df = (
            df.group_by("year")
            .agg(
                [
                    pl.len().alias("line_count"),
                    pl.col("date").n_unique().alias("unique_dates"),
                    pl.col("issue_id").n_unique().alias("issue_count"),
                ]
            )
            .sort("year", descending=(sort_order == "desc"))
        )

        # Load image index for thumbnails
        image_index_path = Path(config.data_dir) / "raw" / source_name / "image_index.parquet"
        year_images = {}
        if image_index_path.exists():
            img_df = pl.read_parquet(image_index_path)
            img_df = img_df.filter(pl.col("file_exists") == True)

            for row in years_df.iter_rows(named=True):
                year = row["year"]
                year_imgs = img_df.filter(pl.col("year") == year)
                if year_imgs.height > 0:
                    sample = year_imgs.sample(n=1)
                    year_images[year] = sample["image_path"][0]

        # Convert to response
        results = []
        for row in years_df.iter_rows(named=True):
            results.append(
                {
                    "year": row["year"],
                    "line_count": row["line_count"],
                    "unique_dates": row["unique_dates"],
                    "issue_count": row["issue_count"],
                    "image_path": year_images.get(row["year"]),
                }
            )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/browse/months")
async def browse_months(
    source_name: str,
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort_order: str = Query(default="asc", regex="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Browse newspapers by month with aggregated statistics"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        parquet_files = list(text_path.glob("*.parquet"))
        if not parquet_files:
            return {"results": [], "total": 0, "page": page, "page_size": page_size}

        df = pl.read_parquet(parquet_files[0])

        if df is None or df.height == 0:
            return {"results": [], "total": 0, "page": page, "page_size": page_size}

        # Apply year filters
        if year:
            df = df.filter(pl.col("year") == year)
        else:
            if year_from:
                df = df.filter(pl.col("year") >= year_from)
            if year_to:
                df = df.filter(pl.col("year") <= year_to)

        # Group by year and month
        months_df = (
            df.group_by(["year", "month"])
            .agg(
                [
                    pl.len().alias("line_count"),
                    pl.col("date").n_unique().alias("unique_dates"),
                    pl.col("issue_id").n_unique().alias("issue_count"),
                ]
            )
            .sort(["year", "month"], descending=[sort_order == "desc", sort_order == "desc"])
        )

        # Get total count for pagination
        total = months_df.height

        # Apply pagination
        offset = (page - 1) * page_size
        months_df = months_df.slice(offset, page_size)

        # Convert to response
        results = []
        for row in months_df.iter_rows(named=True):
            results.append(
                {
                    "year": row["year"],
                    "month": row["month"],
                    "line_count": row["line_count"],
                    "unique_dates": row["unique_dates"],
                    "issue_count": row["issue_count"],
                }
            )

        return {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/browse/issues")
async def browse_issues(
    source_name: str,
    year: Optional[int] = None,
    month: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort_order: str = Query(default="asc", regex="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Browse individual newspaper issues with metadata and images"""
    try:
        config = get_config()
        text_path = Path(config.data_dir) / "raw" / source_name / "text"

        parquet_files = list(text_path.glob("*.parquet"))
        if not parquet_files:
            return {"results": [], "total": 0, "page": page, "page_size": page_size}

        df = pl.read_parquet(parquet_files[0])

        if df is None or df.height == 0:
            return {"results": [], "total": 0, "page": page, "page_size": page_size}

        # Apply filters
        if year:
            df = df.filter(pl.col("year") == year)
        else:
            if year_from:
                df = df.filter(pl.col("year") >= year_from)
            if year_to:
                df = df.filter(pl.col("year") <= year_to)

        if month:
            df = df.filter(pl.col("month") == month)

        # Group by issue
        issues_df = (
            df.group_by("issue_id")
            .agg(
                [
                    pl.col("date").min().alias("date"),
                    pl.col("newspaper_title").first().alias("newspaper_title"),
                    pl.col("year_volume").first().alias("year_volume"),
                    pl.len().alias("line_count"),
                    pl.col("text_block_id").n_unique().alias("block_count"),
                    pl.col("page_count").first().alias("page_count"),
                ]
            )
            # Extract edition for proper sorting within same day (1=morning, 2=midday, 3=evening)
            .with_columns(
                pl.col("issue_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("edition")
            )
            .sort(["date", "edition"], descending=[sort_order == "desc", False])
        )

        # Get total count for pagination
        total = issues_df.height

        # Apply pagination
        offset = (page - 1) * page_size
        issues_df = issues_df.slice(offset, page_size)

        # Load image index for thumbnails (first page of each issue)
        image_index_path = Path(config.data_dir) / "raw" / source_name / "image_index.parquet"
        issue_images = {}
        if image_index_path.exists():
            img_df = pl.read_parquet(image_index_path)
            img_df = img_df.filter(pl.col("file_exists") == True)

            for row in issues_df.iter_rows(named=True):
                issue_id = row["issue_id"]
                issue_imgs = img_df.filter(
                    (pl.col("issue_id") == issue_id) & (pl.col("page_number") == 1)
                )
                if issue_imgs.height > 0:
                    issue_images[issue_id] = issue_imgs["image_path"][0]

        # Convert to response
        results = []
        for row in issues_df.iter_rows(named=True):
            issue_id = row["issue_id"]
            # Extract daily count from issue_id (format: {source}_{YYYY-MM-DD}_{issue:03d}_{daily})
            issue_parts = issue_id.split("_")
            daily_count = issue_parts[-1] if len(issue_parts) >= 4 else None
            issue_number = issue_parts[-2] if len(issue_parts) >= 3 else None

            results.append(
                {
                    "issue_id": issue_id,
                    "date": row["date"],
                    "newspaper_title": row["newspaper_title"],
                    "year_volume": row["year_volume"],
                    "line_count": row["line_count"],
                    "block_count": row["block_count"],
                    "page_count": row["page_count"],
                    "image_path": issue_images.get(issue_id),
                    "issue_number": issue_number,
                    "daily_count": daily_count,
                }
            )

        return {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/page-analysis/{page_id}")
async def get_page_analysis(source_name: str, page_id: str):
    """
    Get all available analysis results for a specific page.
    Returns all result sets grouped by type, not just the latest.
    """
    try:
        config = get_config()
        results_path = Path(config.data_dir).parent / "results" / source_name

        # Use explicit dict typing to avoid type checker issues
        keywords_dict = {}
        emotions_dict = {}
        layout_dict = {}
        entities_dict = {}

        # Helper to find all result directories for a type
        def find_all_results(analysis_type: str) -> list:
            type_path = results_path / analysis_type
            if not type_path.exists():
                return []
            # Get all directories sorted by name (most recent first)
            dirs = sorted([d for d in type_path.iterdir() if d.is_dir()], reverse=True)
            return dirs

        # Load keywords from all result sets
        for keywords_dir in find_all_results("keywords"):
            keywords_file = keywords_dir / "keywords.parquet"
            if keywords_file.exists():
                try:
                    kw_df = pl.read_parquet(keywords_file)
                    kw_df = kw_df.filter(pl.col("page_id") == page_id)
                    if kw_df.height > 0:
                        result_name = keywords_dir.name
                        # Unpack keywords list into individual objects
                        keywords_list = []
                        for row in kw_df.to_dicts():
                            keywords = row.get("keywords", [])
                            scores = row.get("scores", [])
                            # Zip keywords and scores together
                            for kw, score in zip(keywords, scores):
                                keywords_list.append(
                                    {
                                        "keyword": kw,
                                        "score": score,
                                        "text_block_id": row.get("text_block_id"),
                                    }
                                )
                        keywords_dict[result_name] = keywords_list
                except Exception:
                    continue

        # Load emotions from all result sets
        for emotions_dir in find_all_results("emotions"):
            emotions_file = emotions_dir / "emotions.parquet"
            if emotions_file.exists():
                try:
                    em_df = pl.read_parquet(emotions_file)
                    em_df = em_df.filter(pl.col("page_id") == page_id)
                    if em_df.height > 0:
                        result_name = emotions_dir.name
                        # Transform emotion records into frontend-friendly format
                        # Each record has boolean flags and probabilities for 6 emotions
                        # Note: Column names are Capitalized (Sadness, Love, Joy, Fear, Anger, Agitation)
                        emotions_list = []
                        emotion_types = [
                            ("Sadness", "sadness"),
                            ("Love", "love"),
                            ("Joy", "joy"),
                            ("Fear", "fear"),
                            ("Anger", "anger"),
                            ("Agitation", "agitation"),
                        ]

                        # Check if we need to join with text data
                        has_text = "text" in em_df.columns

                        for row in em_df.iter_rows(named=True):
                            # For each row, extract detected emotions (where boolean flag is True/1)
                            for cap_name, lower_name in emotion_types:
                                if row.get(cap_name, 0):  # Check boolean flag (0 or 1)
                                    emotions_list.append(
                                        {
                                            "label": cap_name,
                                            "emotion": cap_name,
                                            "score": row.get(f"{cap_name}_prob", 0.0),
                                            "text": row.get("text") if has_text else None,
                                            "line_id": row.get("line_id"),
                                            "text_block_id": row.get("text_block_id"),
                                        }
                                    )

                        emotions_dict[result_name] = emotions_list
                except Exception as e:
                    import logging

                    logging.error(f"Error loading emotions from {emotions_dir}: {e}")
                    continue

        # Load layout from all result sets
        for layout_dir in find_all_results("layout"):
            # Try enriched first, then regular
            layout_file = layout_dir / "layout_enriched.parquet"
            if not layout_file.exists():
                layout_file = layout_dir / "layout.parquet"
            if layout_file.exists():
                try:
                    layout_df = pl.read_parquet(layout_file)
                    layout_df = layout_df.filter(pl.col("page_id") == page_id)
                    if layout_df.height > 0:
                        result_name = layout_dir.name
                        # Convert bbox struct columns to nested dicts
                        layout_dict[result_name] = [
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
                                "page_id": row["page_id"],
                                "source_id": row.get("source_id"),
                                "issue_id": row.get("issue_id"),
                                "text_content": row.get("text_content"),
                            }
                            for row in layout_df.iter_rows(named=True)
                        ]
                except (FileNotFoundError, ValueError, OSError, pl.ComputeError) as e:
                    logging.warning(
                        f"Error loading layout from {layout_dir}: {e.__class__.__name__}: {e}"
                    )
                    continue

        # Load entities from all result sets
        for entities_dir in find_all_results("entities"):
            entities_file = entities_dir / "entities.parquet"
            if entities_file.exists():
                try:
                    ent_df = pl.read_parquet(entities_file)
                    ent_df = ent_df.filter(pl.col("page_id") == page_id)
                    if ent_df.height > 0:
                        result_name = entities_dir.name
                        entities_dict[result_name] = ent_df.to_dicts()
                except (FileNotFoundError, ValueError, OSError, pl.ComputeError) as e:
                    logging.warning(
                        f"Error loading entities from {entities_dir}: {e.__class__.__name__}: {e}"
                    )
                    continue

        return {
            "page_id": page_id,
            "keywords": keywords_dict,
            "emotions": emotions_dict,
            "layout": layout_dict,
            "entities": entities_dict,
        }

    except (FileNotFoundError, ValueError, OSError, KeyError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
