"""
Keyword extraction endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Union
from datetime import date
import polars as pl
from pathlib import Path

from newspaper_explorer.config.base import get_config
from ..models import Keyword, KeywordDocument, KeywordCoOccurrence, PaginatedKeywords

router = APIRouter()


def _get_keyword_data(source_name: str, run_id: Optional[str] = None) -> pl.DataFrame:
    """Helper to load keyword data for a specific source and run"""
    config = get_config()
    keywords_dir = Path(config.results_dir) / source_name / "keywords"

    if run_id:
        # Load specific run - run_id is the directory name
        keywords_path = keywords_dir / run_id / "keywords.parquet"
    else:
        # Default: try to load keywords.parquet or latest run
        keywords_path = keywords_dir / "keywords.parquet"

        if not keywords_path.exists():
            # Try to find the most recent run directory
            run_dirs = [d for d in keywords_dir.glob("*/") if d.is_dir()]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                keywords_path = latest_run / "keywords.parquet"

    if not keywords_path.exists():
        raise HTTPException(status_code=404, detail="No keyword data available")

    df = pl.read_parquet(keywords_path)

    # Extract date from issue_id if not present
    if "date" not in df.columns and "issue_id" in df.columns:
        # issue_id format: 3074409-X_1905-09-05_437_1
        # Extract date part (YYYY-MM-DD)
        df = df.with_columns(
            pl.col("issue_id")
            .str.extract(r"_(\d{4}-\d{2}-\d{2})_", 1)
            .str.to_date("%Y-%m-%d")
            .alias("date")
        )

    return df


@router.get("/{source_name}/stats")
async def get_keyword_stats(
    source_name: str,
    run_id: Optional[str] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Get aggregated statistics for keywords"""
    try:
        df = _get_keyword_data(source_name, run_id)

        # Check which columns are available
        has_tfidf = "tfidf_score" in df.columns
        has_scores = "scores" in df.columns
        has_keywords = "keywords" in df.columns

        # Handle different data formats
        if has_keywords and has_scores:
            # Exploded format: keywords and scores are lists per document
            df = df.explode(["keywords", "scores"])
            df = df.rename({"keywords": "keyword", "scores": "score"})
        elif "keyword" not in df.columns:
            raise HTTPException(status_code=500, detail="Unexpected keyword data format")

        # Ensure we have a score column
        if "score" not in df.columns:
            if has_tfidf:
                df = df.rename({"tfidf_score": "score"})
            else:
                df = df.with_columns(pl.lit(1.0).alias("score"))

        # Use LazyFrame for efficient aggregation
        lf = df.lazy()

        # Apply date filter if present
        if "date" in df.columns:
            if start_date:
                lf = lf.filter(pl.col("date") >= start_date)
            if end_date:
                lf = lf.filter(pl.col("date") <= end_date)

        # Apply score filter
        lf = lf.filter(pl.col("score") >= min_score)

        # Calculate total keywords and occurrences
        total_df = lf.select(
            [
                pl.col("keyword").n_unique().alias("unique_keywords"),
                pl.len().alias("total_occurrences"),
                pl.col("score").mean().alias("avg_score"),
            ]
        ).collect()

        # Calculate unique documents
        unique_docs = 0
        if "doc_id" in df.columns:
            unique_docs = lf.select(pl.col("doc_id").n_unique()).collect().item()
        elif "issue_id" in df.columns:
            unique_docs = lf.select(pl.col("issue_id").n_unique()).collect().item()
        else:
            # Estimate from keyword counts
            unique_docs = lf.select(pl.len()).collect().item()

        # Calculate average frequency (occurrences per keyword)
        avg_frequency = (
            total_df["total_occurrences"][0] / total_df["unique_keywords"][0]
            if total_df["unique_keywords"][0] > 0
            else 0
        )

        # Score distribution histogram (20 bins)
        score_dist_df = (
            lf.select((pl.col("score") * 20).round(0).cast(pl.Int32).alias("bin_idx"))
            .group_by("bin_idx")
            .agg(pl.len().alias("count"))
            .collect()
        )

        score_distribution = {"bins": [f"{(i/20):.2f}" for i in range(20)], "counts": [0] * 20}
        for row in score_dist_df.iter_rows(named=True):
            idx = row["bin_idx"]
            if 0 <= idx < 20:
                score_distribution["counts"][idx] = int(row["count"])

        # Top keywords by frequency (return up to 200 for wordcloud support)
        top_keywords_df = (
            lf.group_by("keyword")
            .agg(
                [
                    pl.len().alias("frequency"),
                    pl.col("score").mean().alias("avg_score"),
                ]
            )
            .sort("frequency", descending=True)
            .head(200)
            .collect()
        )

        top_keywords = [
            {"keyword": row["keyword"], "frequency": row["frequency"], "score": row["avg_score"]}
            for row in top_keywords_df.iter_rows(named=True)
        ]

        # Keywords per document distribution
        keywords_per_doc = []
        if "doc_id" in df.columns:
            kw_per_doc_df = (
                lf.group_by("doc_id").agg(pl.len().alias("count")).select("count").collect()
            )
            keywords_per_doc = kw_per_doc_df["count"].to_list()

        return {
            "total": total_df["unique_keywords"][0],
            "total_occurrences": total_df["total_occurrences"][0],
            "avg_score": total_df["avg_score"][0],
            "avg_frequency": avg_frequency,
            "unique_documents": unique_docs,
            "score_distribution": score_distribution,
            "top_keywords": top_keywords,
            "keywords_per_doc": keywords_per_doc,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/")
async def get_keywords(
    source_name: str,
    run_id: Optional[str] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: Optional[int] = Query(default=None),
    page: Optional[int] = Query(default=1, ge=1),
    page_size: Optional[int] = Query(default=100, ge=1, le=1000),
) -> Union[PaginatedKeywords, List[Keyword]]:
    """Get list of keywords with optional filtering and pagination"""
    try:
        df = _get_keyword_data(source_name, run_id)

        # Check which columns are available
        has_tfidf = "tfidf_score" in df.columns
        has_scores = "scores" in df.columns
        has_keywords = "keywords" in df.columns

        # Handle different data formats
        if has_keywords and has_scores:
            # Exploded format: keywords and scores are lists per document
            # Need to explode them first
            df = df.explode(["keywords", "scores"])
            df = df.rename({"keywords": "keyword", "scores": "score"})
        elif "keyword" not in df.columns:
            raise HTTPException(status_code=500, detail="Unexpected keyword data format")

        # Ensure we have a score column
        if "score" not in df.columns:
            if has_tfidf:
                df = df.rename({"tfidf_score": "score"})
            else:
                df = df.with_columns(pl.lit(1.0).alias("score"))

        # Apply date filter if present
        if "date" in df.columns:
            if start_date:
                df = df.filter(pl.col("date") >= start_date)
            if end_date:
                df = df.filter(pl.col("date") <= end_date)

        # Apply score filter
        df = df.filter(pl.col("score") >= min_score)

        # Aggregate by keyword
        keywords_df = (
            df.group_by("keyword")
            .agg(
                [
                    pl.len().alias("frequency"),
                    pl.col("score").mean().alias("avg_score"),
                    pl.col("score").min().alias("min_score"),
                    pl.col("score").max().alias("max_score"),
                ]
            )
            .sort("frequency", descending=True)
        )

        # Calculate total count before pagination
        total_count = len(keywords_df)

        # Apply pagination
        _page = page or 1
        _page_size = page_size or 100

        if limit is not None:
            # Legacy behavior: use limit without pagination
            keywords_df = keywords_df.head(limit)
            total = total_count
        else:
            # New pagination behavior
            offset = (_page - 1) * _page_size
            keywords_df = keywords_df.slice(offset, _page_size)
            total = total_count

        # Convert to response model
        keywords = []
        for row in keywords_df.iter_rows(named=True):
            keywords.append(
                Keyword(
                    keyword=row["keyword"],
                    frequency=row["frequency"],
                    tfidf_score=row["avg_score"],  # Use avg_score as tfidf_score
                )
            )

        # Return paginated response if using pagination
        if limit is None:
            return PaginatedKeywords(
                items=keywords,
                total=total,
                page=_page,
                page_size=_page_size,
            )
        else:
            return keywords
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/timeline")
async def get_keywords_timeline(
    source_name: str,
    run_id: Optional[str] = Query(None),
    aggregation: str = Query(default="month", regex="^(day|week|month|year)$"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(default=10),
):
    """
    Get keyword frequency timeline aggregated by time period.
    Returns top N keywords over time.
    """
    try:
        df = _get_keyword_data(source_name, run_id)

        # Check if date column exists
        if "date" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Timeline not available - data does not contain date information",
            )

        # Handle different data formats (explode if needed)
        if "keywords" in df.columns and "scores" in df.columns:
            df = df.explode(["keywords", "scores"])
            df = df.rename({"keywords": "keyword", "scores": "score"})

        # Apply date filter
        if start_date:
            df = df.filter(pl.col("date") >= start_date)
        if end_date:
            df = df.filter(pl.col("date") <= end_date)

        # Get top N keywords overall
        top_keywords = (
            df.group_by("keyword")
            .agg(pl.len().alias("total_count"))
            .sort("total_count", descending=True)
            .head(limit)
            .select("keyword")
        )

        # Filter to top keywords only
        df = df.join(top_keywords, on="keyword", how="inner")

        # Aggregate by time period
        if aggregation == "day":
            time_col = pl.col("date")
        elif aggregation == "week":
            time_col = pl.col("date").dt.truncate("1w")
        elif aggregation == "month":
            time_col = pl.col("date").dt.truncate("1mo")
        else:  # year
            time_col = pl.col("date").dt.truncate("1y")

        df = df.with_columns(time_col.alias("period"))

        # Group by keyword and period
        timeline_df = df.group_by(["keyword", "period"]).agg(pl.len().alias("value")).sort("period")

        # Convert to response format
        result: Dict = {}
        for row in timeline_df.iter_rows(named=True):
            keyword = row["keyword"]
            if keyword not in result:
                result[keyword] = []
            result[keyword].append(
                {"date": row["period"].strftime("%Y-%m-%d"), "value": row["value"]}
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/documents", response_model=List[KeywordDocument])
async def get_keyword_documents(
    source_name: str,
    keyword: str = Query(..., description="Keyword to search for"),
    run_id: Optional[str] = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Get list of documents containing a specific keyword.
    Returns documents sorted by relevance score.
    """
    try:
        df = _get_keyword_data(source_name, run_id)

        # Handle different data formats
        if "keywords" in df.columns and "scores" in df.columns:
            # Exploded format: keywords and scores are lists per document
            df = df.explode(["keywords", "scores"])
            df = df.rename({"keywords": "keyword", "scores": "score"})
        elif "keyword" not in df.columns:
            raise HTTPException(status_code=500, detail="Unexpected keyword data format")

        # Ensure we have a score column
        if "score" not in df.columns:
            if "tfidf_score" in df.columns:
                df = df.rename({"tfidf_score": "score"})
            else:
                df = df.with_columns(pl.lit(1.0).alias("score"))

        # Filter to documents containing the specific keyword
        docs_df = df.filter(pl.col("keyword") == keyword).sort("score", descending=True).head(limit)

        if len(docs_df) == 0:
            return []

        # Convert to response model
        documents = []
        for row in docs_df.iter_rows(named=True):
            doc_id = row.get("doc_id") or row.get("issue_id", "unknown")

            # Try to extract page_id if available
            page_id = row.get("page_id")
            if not page_id and "doc_id" in row:
                page_id = row["doc_id"]

            # Format date if available
            date_str = None
            if "date" in row and row["date"]:
                date_str = (
                    row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"])
                )

            documents.append(
                KeywordDocument(
                    doc_id=doc_id,
                    score=row["score"],
                    date=date_str,
                    page_id=page_id,
                )
            )

        return documents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/cooccurrence", response_model=List[KeywordCoOccurrence])
async def get_keyword_cooccurrence(
    source_name: str,
    keyword: str = Query(..., description="Keyword to find co-occurrences for"),
    run_id: Optional[str] = Query(None),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Get keywords that frequently co-occur with the specified keyword.
    Returns keywords that appear in the same documents as the target keyword.
    """
    try:
        df = _get_keyword_data(source_name, run_id)

        # Handle different data formats
        if "keywords" in df.columns and "scores" in df.columns:
            # In exploded format, we need the original grouped data
            # Group back by document to find co-occurrences
            docs_with_keyword = df.filter(pl.col("keywords").list.contains(keyword))

            if len(docs_with_keyword) == 0:
                return []

            # Explode all keywords from these documents
            all_keywords = docs_with_keyword.select("keywords").explode("keywords")

            # Count occurrences, excluding the target keyword
            cooccur_df = (
                all_keywords.filter(pl.col("keywords") != keyword)
                .group_by("keywords")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(limit)
            )

        else:
            # Data is already exploded - need to reconstruct documents
            # Get document IDs containing the target keyword
            doc_ids_with_keyword = df.filter(pl.col("keyword") == keyword).select("doc_id").unique()

            if len(doc_ids_with_keyword) == 0:
                return []

            # Find all keywords in those documents
            docs_df = df.join(doc_ids_with_keyword, on="doc_id", how="inner")

            # Count co-occurrences, excluding the target keyword
            cooccur_df = (
                docs_df.filter(pl.col("keyword") != keyword)
                .group_by("keyword")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .head(limit)
            )

        # Convert to response model
        cooccurrences = []
        for row in cooccur_df.iter_rows(named=True):
            kw = row.get("keyword") or row.get("keywords") or "unknown"
            cooccurrences.append(
                KeywordCoOccurrence(
                    keyword=kw,
                    count=row["count"],
                )
            )

        return cooccurrences
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
