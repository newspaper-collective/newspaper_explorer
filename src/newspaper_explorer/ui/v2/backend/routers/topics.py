"""
Topic modeling endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import date
import polars as pl
from pathlib import Path

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.analysis.topics import TopicRecord

router = APIRouter()


@router.get("/{source_name}/", response_model=List[TopicRecord])
async def get_topic_records(
    source_name: str,
    doc_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get topic modeling results (document-topic assignments)"""
    try:
        config = get_config()
        topics_path = Path(config.results_dir) / source_name / "topics" / "topics.parquet"

        if not topics_path.exists():
            raise HTTPException(status_code=404, detail="No topic data available")

        df = pl.read_parquet(topics_path)

        # Filter by doc_id if provided
        if doc_id:
            df = df.filter(df["doc_id"] == doc_id)

        # Limit results
        df = df.head(limit)

        # Convert to response model using existing schema
        return [TopicRecord(**row) for row in df.to_dicts()]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No topic data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
