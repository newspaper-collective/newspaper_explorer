"""
Image access endpoints
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.indexing.image_index import ImageIndexer

router = APIRouter()


@router.get("/{source_name}/list")
async def list_images(
    source_name: str,
    issue_id: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    """List available images with optional filtering"""
    try:
        indexer = ImageIndexer(source_name)

        if issue_id:
            images = indexer.get_issue_images(issue_id)
        else:
            # Get all images
            images = []
            for issue in indexer.list_issues():
                issue_images = indexer.get_issue_images(issue)
                images.extend(issue_images)

        # Pagination
        total = len(images)
        offset = (page - 1) * page_size
        images = images[offset : offset + page_size]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "images": images,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/url")
async def get_image_url(
    source_name: str,
    page_id: str,
):
    """Get URL for a specific page image"""
    try:
        indexer = ImageIndexer(source_name)
        image_path = indexer.get_page_image_path(page_id)

        if not image_path or not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # Construct URL (relative to static mount point)
        relative_path = image_path.relative_to(
            Path(get_config().data_dir) / "raw" / source_name / "images"
        )
        url = f"/static/{source_name}/images/{relative_path}"

        return {"url": url}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_name}/stats")
async def get_image_stats(source_name: str):
    """Get image availability statistics"""
    try:
        indexer = ImageIndexer(source_name)

        issues = indexer.list_issues()
        total_images = sum(len(indexer.get_issue_images(issue)) for issue in issues)

        return {
            "total_issues_with_images": len(issues),
            "total_images": total_images,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
