"""Worker functions for parallel data processing."""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from newspaper_explorer.data.utils.ids import generate_page_id

logger = logging.getLogger(__name__)


def extract_image_metadata_worker(
    img_path: Path,
    images_dir: Path,
    source_id: str,
    mets_cache: dict,
    alto_cache: dict,
) -> Optional[dict]:
    """
    Worker function for parallel image metadata extraction.

    This is a module-level function so it can be pickled for multiprocessing.

    Args:
        img_path: Path to image file
        images_dir: Base images directory
        source_id: Source identifier
        mets_cache: Dictionary mapping path keys to METS metadata
        alto_cache: Dictionary mapping page keys to (width, height) tuples

    Returns:
        Dictionary with image metadata, or None if extraction failed
    """
    try:
        # Get relative path from images directory
        rel_path = img_path.relative_to(images_dir)
        rel_path_str = str(rel_path)

        # Parse path structure: YYYY/MM/DD/issue_number/filename.jpg
        parts = rel_path.parts
        if len(parts) < 5:
            return None

        year, month, day, issue_num, filename = parts[0], parts[1], parts[2], parts[3], parts[4]

        # Extract page number from filename (e.g., "max_7.jpg" -> 7)
        page_number = None
        if "max_" in filename:
            try:
                page_number = int(filename.split("max_")[1].split(".")[0])
            except (IndexError, ValueError):
                pass

        # Create path-based key for METS lookup
        path_key = f"{year}/{month}/{day}/{issue_num}"

        # Get METS metadata if available
        mets_data = mets_cache.get(path_key, {})

        # Use the proper issue_id from METS cache if available
        issue_id = mets_data.get("issue_id", path_key)

        # Get file size in bytes
        file_size = img_path.stat().st_size if img_path.exists() else None

        # Get ALTO dimensions from cache
        alto_width, alto_height = None, None
        if page_number is not None:
            # Try different zero-padding lengths (3, 4, 5 digits are common)
            for padding in [3, 4, 5]:
                padded_page = str(page_number).zfill(padding)
                page_key = f"{year}/{month}/{day}/{issue_num}/{padded_page}"
                dims = alto_cache.get(page_key)
                if dims:
                    alto_width, alto_height = dims
                    break

        # Get actual image dimensions
        width, height = None, None
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            pass

        # Generate page_id using the standard function if we have all required data
        page_id = None
        if (
            page_number is not None
            and mets_data.get("date")
            and mets_data.get("issue_number") is not None
            and mets_data.get("daily_issue_number") is not None
        ):
            from datetime import datetime

            date_obj = datetime.fromisoformat(mets_data["date"])
            page_id = generate_page_id(
                source_id,
                date_obj,
                mets_data["issue_number"],
                mets_data["daily_issue_number"],
                page_number,
            )

        # Build record
        record = {
            "image_path": rel_path_str,
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "date": f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            "issue_id": issue_id,
            "page_id": page_id,
            "page_number": page_number,
            "filename": filename,
            "file_size_bytes": file_size,
            "alto_width": alto_width,
            "alto_height": alto_height,
            "width": width,
            "height": height,
            "newspaper_title": mets_data.get("newspaper_title"),
            "year_volume": mets_data.get("year_volume"),
            "page_count": mets_data.get("page_count"),
            "issue_number": mets_data.get("issue_number"),
            "daily_issue_number": mets_data.get("daily_issue_number"),
            "file_exists": True,
        }

        return record

    except Exception:
        return None
