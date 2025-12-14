"""
Image index management for newspaper page images.

Creates and maintains a Parquet index of downloaded images with metadata
from METS and ALTO files.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Optional, Union

from lxml import etree
import polars as pl
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.indexing.image_metadata_worker import extract_image_metadata_worker
from newspaper_explorer.data.parser.mets import METSParser
from newspaper_explorer.data.utils.ids import extract_edition, generate_issue_id
from newspaper_explorer.data.utils.files import find_mets_files

logger = logging.getLogger(__name__)


class ImageIndexer:
    """
    Creates and manages a Parquet index of newspaper page images with metadata.

    The index includes:
    - image_path: Relative path to the image file
    - source_id: Source identifier (foreign key, e.g., "der_tag")
    - year, month, day: Date components from path
    - date: Full date (YYYY-MM-DD format)
    - page_number: Page number extracted from filename
    - issue_id: Issue identifier in format {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
    - page_id: Page identifier in format {source}_{YYYY-MM-DD}_{issue:03d}_{daily}_{page:03d}
    - filename: Image filename
    - file_size_bytes: File size in bytes
    - alto_width: Image width in ALTO coordinate space
    - alto_height: Image height in ALTO coordinate space
    - width: Actual image width in pixels
    - height: Actual image height in pixels
    - newspaper_title: From METS
    - year_volume: From METS
    - page_count: Total pages in issue from METS
    - issue_number: Issue number from METS
    - edition: Daily issue number (1, 2, 3, etc.)
    - file_exists: Whether the image file exists
    """

    def __init__(self, source_name: str) -> None:
        """
        Initialize the image indexer for a source.

        Args:
            source_name: Name of the newspaper source
        """
        self.source_name = source_name
        self.config = get_config()

        # Always use source_name for consistent ID generation across the codebase
        # (ZDB source ID is kept in source config for provenance but not used in IDs)
        self.source_id = source_name

        self.images_dir = Path(self.config.data_dir) / "raw" / source_name / "images"
        self.index_path = Path(self.config.data_dir) / "raw" / source_name / "image_index.parquet"

        # METS files are in the xml_ocr directory
        self.xml_dir = Path(self.config.data_dir) / "raw" / source_name / "xml_ocr"

    def create_index(self, *, force_rebuild: bool = False) -> pl.DataFrame:
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
        existing_paths: set[str] = set()
        if not force_rebuild and self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            existing_index = pl.read_parquet(self.index_path)
            existing_paths = set(existing_index["image_path"].to_list())

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

        # Build ALTO cache for image dimensions
        logger.info("Building ALTO cache for image dimensions")
        alto_cache = self._build_alto_dimension_cache()

        # Extract metadata from each image (parallel processing)
        records = self._extract_image_metadata_parallel(image_files, mets_cache, alto_cache)

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

    def _extract_image_metadata_parallel(
        self,
        image_files: list[Path],
        mets_cache: dict[str, dict[str, Union[str, int, None]]],
        alto_cache: dict[str, tuple[int, int]],
    ) -> list[dict[str, Union[str, int, float, bool, None]]]:
        """Extract metadata from images using parallel processing.

        Args:
            image_files: List of image file paths to process
            mets_cache: Pre-built METS metadata cache
            alto_cache: Pre-built ALTO dimension cache

        Returns:
            List of metadata records for each successfully processed image
        """
        records: list[dict[str, Union[str, int, float, bool, None]]] = []
        max_workers = max(1, len(image_files) // 1000)  # Scale workers with dataset size
        max_workers = min(max_workers, 16)  # Cap at 16 workers

        logger.info(f"Processing {len(image_files)} images with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(
                    extract_image_metadata_worker,
                    img_path,
                    self.images_dir,
                    self.source_id,
                    mets_cache,
                    alto_cache,
                ): img_path
                for img_path in image_files
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_path),
                total=len(image_files),
                desc="Indexing images",
                unit="img",
            ):
                try:
                    record = future.result()
                    if record:
                        records.append(record)
                except (OSError, ValueError, KeyError, IndexError) as e:
                    img_path = future_to_path[future]
                    logger.warning(f"Failed to process {img_path}: {e}")

        return records

    def _build_mets_cache(self) -> dict[str, dict[str, Union[str, int, None]]]:
        """
        Build a cache of METS metadata keyed by issue identifier.

        Returns:
            Dictionary mapping issue_id to METS metadata
        """
        mets_cache: dict[str, dict[str, Union[str, int, None]]] = {}

        mets_files = find_mets_files(self.xml_dir)
        if not mets_files:
            logger.warning(f"XML directory not found or no METS files: {self.xml_dir}")
            return mets_cache

        logger.info(f"Processing {len(mets_files)} METS files")

        parser = METSParser()
        for mets_path in tqdm(mets_files, desc="Building METS cache", unit="file"):
            try:
                metadata = parser.parse_file(mets_path)

                # Generate proper issue_id using the standard format
                if metadata and metadata.date and metadata.issue_number is not None:
                    # Use edition from METS parser (extracted from filename)
                    # Fallback to path extraction if filename parsing failed
                    rel_path = mets_path.relative_to(self.xml_dir)
                    edition = metadata.edition
                    if edition is None:
                        edition = extract_edition(folder_path=str(rel_path))

                    if edition is not None:
                        # Generate proper issue_id: {source}_{YYYY-MM-DD}_{issue:03d}_{edition}
                        issue_id = generate_issue_id(
                            self.source_id,
                            metadata.date,
                            metadata.issue_number,
                            edition,
                        )

                        # Also store the path-based key for lookup
                        parts = rel_path.parts
                        try:
                            year, month, day, issue_folder = parts[:4]
                        except ValueError:
                            logger.debug(f"Unexpected METS path structure: {rel_path}")
                            continue

                        path_key = f"{year}/{month}/{day}/{issue_folder}"

                        # Convert IssueMetadata to dict for caching
                        cache_entry: dict[str, Union[str, int, None]] = {
                            "newspaper_title": metadata.newspaper_title,
                            "year_volume": metadata.year_volume,
                            "page_count": metadata.page_count,
                            "date": metadata.date.isoformat() if metadata.date else None,
                            "issue_number": metadata.issue_number,
                            "issue_id": issue_id,
                            "edition": edition,
                        }

                        # Store with path-based key (used by image_metadata_worker for lookup)
                        mets_cache[path_key] = cache_entry

            except (etree.XMLSyntaxError, OSError, ValueError, IndexError) as e:
                logger.warning(f"Failed to parse METS file {mets_path}: {e}")

        logger.info(f"Built METS cache with {len(mets_cache)} entries")
        return mets_cache

    def _build_alto_dimension_cache(self) -> dict[str, tuple[int, int]]:
        """Build a cache of image dimensions from ALTO files.

        Returns:
            Dictionary mapping page_id to (width, height) tuple
        """
        alto_cache: dict[str, tuple[int, int]] = {}

        if not self.xml_dir.exists():
            logger.warning(f"XML directory not found: {self.xml_dir}")
            return alto_cache

        # Find all ALTO files in fulltext directories
        alto_files = list(self.xml_dir.rglob("fulltext/*.xml"))
        logger.info(f"Processing {len(alto_files)} ALTO files for image dimensions")

        for alto_path in tqdm(alto_files, desc="Building ALTO cache", unit="file"):
            try:
                tree = etree.parse(str(alto_path))
                root = tree.getroot()

                # Detect namespace
                ns = None
                if root.tag.startswith("{"):
                    ns_url = root.tag.split("}")[0][1:]
                    ns = {"alto": ns_url}

                # Find Page element with dimensions
                page_elem = root.find(".//alto:Page", ns) if ns else root.find(".//Page")

                if page_elem is not None:
                    width = page_elem.get("WIDTH")
                    height = page_elem.get("HEIGHT")

                    if width and height:
                        # Parse path to create page identifier: YYYY/MM/DD/issue_num/page_num
                        rel_path = alto_path.relative_to(self.xml_dir)
                        parts = rel_path.parts
                        # Expected year/month/day/issue/fulltext/filename.xml
                        try:
                            year, month, day, issue_num, _fulltext, filename = parts[:6]
                        except ValueError:
                            logger.debug(f"Unexpected ALTO path structure: {rel_path}")
                            continue

                        # Extract page number from filename (last number before .xml)
                        page_match = filename.split("_")[-1].replace(".xml", "")
                        page_key = f"{year}/{month}/{day}/{issue_num}/{page_match}"
                        alto_cache[page_key] = (int(width), int(height))
            except (etree.XMLSyntaxError, OSError, ValueError, KeyError, IndexError) as e:
                logger.debug(f"Failed to parse ALTO file {alto_path}: {e}")

        logger.info(f"Built ALTO dimension cache with {len(alto_cache)} entries")
        return alto_cache

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

    def get_stats(self) -> dict[str, Union[int, float, None]]:
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
        *,
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

            samples: list[pl.DataFrame] = []
            for i, selected_year in enumerate(selected_years):
                year_images = index.filter(pl.col("year") == selected_year)
                if len(year_images) > 0:
                    # Pick different image based on index for variety
                    idx = i % len(year_images)
                    samples.append(year_images[idx : idx + 1])

            if samples:
                return pl.concat(samples)
            return pl.DataFrame()

        # Random sample
        return index.sample(n=min(limit, len(index)))
