"""
Text search endpoints
"""

from typing import List

from fastapi import APIRouter, Body, HTTPException

from newspaper_explorer.analyze.query.engine import QueryEngine
from newspaper_explorer.models.api.search import SearchQuery, SearchResponse, SearchResult

router = APIRouter()


@router.post("/{source_name}/", response_model=SearchResponse)
async def search_text(
    source_name: str,
    query: SearchQuery = Body(...),
):
    """Search text content with filters"""
    try:
        # Determine parquet path from run_id if provided
        # For now, we assume run_id is the filename relative to the text directory
        # or a special identifier. If it's "default" or None, we use the default.
        parquet_path = None
        if query.run_id and query.run_id != "default":
            # Check if it's a full path or relative
            # This logic depends on how ResultsViewer sends the ID.
            # If ResultsViewer sends a full path, use it.
            # If it sends a filename, construct the path.
            import os

            if os.path.isabs(query.run_id):
                parquet_path = query.run_id
            else:
                # Assume it's in the text directory
                from pathlib import Path

                from newspaper_explorer.config.base import get_config

                config = get_config()
                parquet_path = str(
                    Path(config.parsed_dir) / source_name / query.run_id
                )

        qe = QueryEngine(source=source_name, in_memory=True)

        # Pagination
        page = query.pagination.page if query.pagination else 1
        page_size = query.pagination.page_size if query.pagination else 50
        offset = (page - 1) * page_size

        # Extract date filters
        start_date = query.date_filter.start_date if query.date_filter else None
        end_date = query.date_filter.end_date if query.date_filter else None

        # Get total count
        total = qe.search_text_count(
            query=query.query,
            start_date=start_date,
            end_date=end_date,
        )

        if total == 0:
            return SearchResponse(total=0, results=[], page=page, page_size=page_size)

        # Get results
        rows = qe.search_text(
            query=query.query,
            limit=page_size,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
        )

        # Search terms for highlighting
        search_terms = query.query.lower().split()

        # Convert to response model
        results = []
        for row in rows:
            # Extract highlights (simple substring extraction)
            text = row["text"]
            highlights = []
            for term in search_terms:
                if term in text.lower():
                    # Find occurrences and extract context
                    start = text.lower().find(term)
                    context_start = max(0, start - 50)
                    context_end = min(len(text), start + len(term) + 50)
                    highlights.append(text[context_start:context_end])

            results.append(
                SearchResult(
                    text_block_id=row.get("text_block_id", "unknown"),
                    page_id=row.get("page_id", "unknown"),
                    date=row["date"],
                    text=text[:500],  # Limit text length
                    highlights=highlights[:3],  # Limit highlights
                    score=1.0,  # Simple scoring for now
                    x=row.get("x"),
                    y=row.get("y"),
                    width=row.get("width"),
                    height=row.get("height"),
                    image_path=row.get("image_path"),
                )
            )

        return SearchResponse(
            total=total,
            results=results,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
