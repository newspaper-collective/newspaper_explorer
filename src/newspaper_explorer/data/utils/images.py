"""
Image index utilities for managing newspaper page images and their metadata.

This module provides functionality to:
1. Create and maintain an index of downloaded images with METS metadata
2. Query image metadata efficiently without scanning the filesystem
3. Support incremental index updates for new downloads
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.parser.mets import METSParser
from newspaper_explorer.data.utils.ids import generate_issue_id
from newspaper_explorer.utils.sources import load_source_config

logger = logging.getLogger(__name__)


class ImageIndexer:
    """
    Creates and manages a Parquet index of newspaper page images with metadata.

    The index includes:
    - image_path: Relative path to the image file
    - year, month, day: Date components from path
    - date: Full date (YYYY-MM-DD format)
    - page_number: Page number extracted from filename
    - issue_id: Issue identifier in format {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
    - filename: Image filename
    - file_size_bytes: File size in bytes
    - newspaper_title: From METS
    - year_volume: From METS
    - page_count: Total pages in issue from METS
    - issue_number: Issue number from METS
    - daily_issue_number: Daily issue number (1, 2, 3, etc.)
    - file_exists: Whether the image file exists
    """

    def __init__(self, source_name: str):
        """
        Initialize the image indexer for a source.

        Args:
            source_name: Name of the newspaper source
        """
        self.source_name = source_name
        self.config = get_config()

        # Load source config to get the ZDB source ID
        source_config = load_source_config(source_name)
        self.source_id = (
            source_config.metadata.zdb_source_id
            if (source_config and source_config.metadata.zdb_source_id)
            else source_name
        )

        self.images_dir = Path(self.config.data_dir) / "raw" / source_name / "images"
        self.index_path = Path(self.config.data_dir) / "raw" / source_name / "image_index.parquet"

        # METS files are in the xml_ocr directory
        self.xml_dir = Path(self.config.data_dir) / "raw" / source_name / "xml_ocr"

    def create_index(self, force_rebuild: bool = False) -> pl.DataFrame:
        """
        Create or update the image index.

        Args:
            force_rebuild: If True, rebuild entire index. If False, update incrementally.

        Returns:
            Polars DataFrame with image metadata
        """
        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return pl.DataFrame()

        # Load existing index if available
        existing_index = None
        if not force_rebuild and self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            existing_index = pl.read_parquet(self.index_path)
            existing_paths = set(existing_index["image_path"].to_list())
        else:
            existing_paths = set()

        # Scan for images
        logger.info(f"Scanning for images in {self.images_dir}")
        image_files: list[Path] = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(self.images_dir.rglob(ext))

        logger.info(f"Found {len(image_files)} total image files")

        # Filter to new images if doing incremental update
        if existing_index is not None:
            image_files = [
                img
                for img in image_files
                if str(img.relative_to(self.images_dir)) not in existing_paths
            ]
            logger.info(f"Found {len(image_files)} new images to index")

        if not image_files:
            logger.info("No new images to index")
            return existing_index if existing_index is not None else pl.DataFrame()

        # Build METS cache for metadata
        logger.info("Building METS cache for metadata enrichment")
        mets_cache = self._build_mets_cache()

        # Extract metadata from each image
        records = []
        for img_path in image_files:
            record = self._extract_image_metadata(img_path, mets_cache)
            if record:
                records.append(record)

        if not records:
            logger.warning("No image metadata extracted")
            return existing_index if existing_index is not None else pl.DataFrame()

        # Create new index DataFrame
        new_index = pl.DataFrame(records)

        # Merge with existing index if available
        if existing_index is not None:
            full_index = pl.concat([existing_index, new_index])
        else:
            full_index = new_index

        # Save index
        logger.info(f"Saving image index to {self.index_path}")
        full_index.write_parquet(self.index_path, compression="zstd")

        logger.info(f"Image index complete: {len(full_index)} total images")
        return full_index

    def _build_mets_cache(self) -> dict[str, dict]:
        """
        Build a cache of METS metadata keyed by issue identifier.

        Returns:
            Dictionary mapping issue_id to METS metadata
        """
        mets_cache: dict[str, dict] = {}

        if not self.xml_dir.exists():
            logger.warning(f"XML directory not found: {self.xml_dir}")
            return mets_cache

        # METS files are *.xml files NOT in fulltext directories
        mets_files = [f for f in self.xml_dir.rglob("*.xml") if "fulltext" not in str(f)]
        logger.info(f"Processing {len(mets_files)} METS files")

        parser = METSParser()
        for mets_path in mets_files:
            try:
                metadata = parser.parse_file(mets_path)

                if metadata:
                    # Generate proper issue_id using the standard format
                    if metadata.date and metadata.issue_number is not None:
                        # Extract daily issue number from path (folder name)
                        rel_path = mets_path.relative_to(self.xml_dir)
                        parts = rel_path.parts
                        if len(parts) >= 5:  # Year/Month/Day/Issue/filename.xml
                            daily_issue_num = int(parts[3])  # Folder name (e.g., "01" -> 1)

                            # Generate proper issue_id: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
                            issue_id = generate_issue_id(
                                self.source_id,  # Use ZDB source ID, not source_name
                                metadata.date,
                                metadata.issue_number,
                                daily_issue_num,
                            )

                            # Also store the path-based key for lookup
                            path_key = f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"

                            # Convert IssueMetadata to dict for caching
                            cache_entry = {
                                "newspaper_title": metadata.newspaper_title,
                                "year_volume": metadata.year_volume,
                                "page_count": metadata.page_count,
                                "date": metadata.date.isoformat() if metadata.date else None,
                                "issue_number": metadata.issue_number,
                                "issue_id": issue_id,
                                "daily_issue_number": daily_issue_num,
                            }

                            # Store with both keys for compatibility
                            mets_cache[path_key] = cache_entry
                            mets_cache[issue_id] = cache_entry
            except Exception as e:
                logger.warning(f"Failed to parse METS file {mets_path}: {e}")

        logger.info(f"Built METS cache with {len(mets_cache)} entries")
        return mets_cache

    def _extract_image_metadata(self, img_path: Path, mets_cache: dict) -> Optional[dict]:
        """
        Extract metadata from an image path and enrich with METS data.

        Args:
            img_path: Path to image file
            mets_cache: METS metadata cache

        Returns:
            Dictionary with image metadata, or None if extraction fails
        """
        try:
            # Get relative path from images directory
            rel_path = img_path.relative_to(self.images_dir)
            rel_path_str = str(rel_path)

            # Parse path structure: YYYY/MM/DD/issue_number/filename.jpg
            parts = rel_path.parts
            if len(parts) < 5:
                logger.warning(f"Unexpected path structure: {rel_path}")
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

            # Build record
            record = {
                "image_path": rel_path_str,
                "year": int(year),
                "month": int(month),
                "day": int(day),
                "date": f"{year}-{month.zfill(2)}-{day.zfill(2)}",
                "issue_id": issue_id,
                "page_number": page_number,
                "filename": filename,
                "file_size_bytes": file_size,
                "newspaper_title": mets_data.get("newspaper_title"),
                "year_volume": mets_data.get("year_volume"),
                "page_count": mets_data.get("page_count"),
                "issue_number": mets_data.get("issue_number"),
                "daily_issue_number": mets_data.get("daily_issue_number"),
                "file_exists": True,
            }

            return record

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {img_path}: {e}")
            return None

    def load_index(self) -> Optional[pl.DataFrame]:
        """
        Load the existing image index.

        Returns:
            Polars DataFrame with image metadata, or None if not found
        """
        if not self.index_path.exists():
            logger.warning(f"Image index not found: {self.index_path}")
            return None

        return pl.read_parquet(self.index_path)

    def get_stats(self) -> dict:
        """
        Get statistics about indexed images.

        Returns:
            Dictionary with image statistics
        """
        index = self.load_index()
        if index is None or len(index) == 0:
            return {
                "total_images": 0,
                "total_size_bytes": 0,
                "total_size_gb": 0.0,
                "years": 0,
                "min_year": None,
                "max_year": None,
                "avg_file_size_mb": 0.0,
            }

        total_size = index["file_size_bytes"].sum() if "file_size_bytes" in index.columns else 0
        years = sorted(index["year"].unique().to_list())

        return {
            "total_images": len(index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "years": len(years),
            "min_year": min(years) if years else None,
            "max_year": max(years) if years else None,
            "avg_file_size_mb": (total_size / len(index) / (1024**2)) if len(index) > 0 else 0.0,
        }

    def get_sample_images(
        self,
        limit: int = 6,
        spread_years: bool = True,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Get sample images from the index.

        Args:
            limit: Number of images to return
            spread_years: If True, spread samples across different years
            min_year: Minimum year filter
            max_year: Maximum year filter

        Returns:
            Polars DataFrame with sample image metadata
        """
        index = self.load_index()
        if index is None or len(index) == 0:
            return pl.DataFrame()

        # Apply year filters
        if min_year:
            index = index.filter(pl.col("year") >= min_year)
        if max_year:
            index = index.filter(pl.col("year") <= max_year)

        if len(index) == 0:
            return pl.DataFrame()

        if spread_years:
            # Get one image per year, spreading across available years
            years = sorted(index["year"].unique().to_list())
            step = max(1, len(years) // limit)
            selected_years = years[::step][:limit]

            samples = []
            for i, year in enumerate(selected_years):
                year_images = index.filter(pl.col("year") == year)
                if len(year_images) > 0:
                    # Pick different image based on index for variety
                    idx = i % len(year_images)
                    samples.append(year_images[idx : idx + 1])

            if samples:
                return pl.concat(samples)
            return pl.DataFrame()
        else:
            # Random sample
            return index.sample(n=min(limit, len(index)))
