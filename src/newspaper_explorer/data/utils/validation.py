"""
Data validation utilities for newspaper data quality assessment.

Functions for checking data integrity, finding missing or empty files,
and validating data completeness, including image validation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
from natsort import natsorted
from PIL import Image

from newspaper_explorer.utils.sources import get_source_paths, load_source_config

logger = logging.getLogger(__name__)

# Default glob pattern for finding ALTO XML files
DEFAULT_ALTO_PATTERN = "**/fulltext/*.xml"

# Minimum expected file size for images (in bytes)
# Files smaller than this are likely corrupted or incomplete
MIN_IMAGE_SIZE_BYTES = 1024  # 1 KB


@dataclass
class ImageValidationResult:
    """Result of image validation check."""

    is_valid: bool
    file_path: Path
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "error": self.error,
        }


def find_empty_xml_files(source_name: str) -> Dict[str, Any]:
    """
    Find XML files without OCR transcription (no text content).

    Compares all XML files in the source directory with processed files
    in the parquet output to identify files that were skipped due to
    having no text content (e.g., pages with only images/graphics).

    Args:
        source_name: Name of the source to check (e.g., 'der_tag')

    Returns:
        Dictionary with:
        - total_xml_files: Total number of XML files found
        - processed_files: Number of files with text content
        - empty_files: Number of files without text
        - empty_file_list: List of paths to empty files

    Example:
        >>> result = find_empty_xml_files("der_tag")
        >>> print(f"Found {result['empty_files']} empty files")
    """
    # Load source configuration
    config = load_source_config(source_name)
    paths = get_source_paths(config)
    raw_dir = paths["raw_dir"]
    output_file = paths["output_file"]

    # Get loading config
    loading_config = config.get("loading", {})
    pattern = loading_config.get("pattern", DEFAULT_ALTO_PATTERN)

    logger.info(f"Scanning for XML files in {raw_dir}")
    all_files = natsorted([str(f.relative_to(raw_dir)) for f in raw_dir.glob(pattern)])

    logger.info(f"Found {len(all_files)} XML files")

    # Get processed files from parquet
    if not output_file.exists():
        logger.warning("No parquet file found - run data load first")
        return {
            "total_xml_files": len(all_files),
            "processed_files": 0,
            "empty_files": len(all_files),
            "empty_file_list": all_files,
        }

    logger.info("Loading processed files from parquet")
    df = pl.read_parquet(output_file)
    processed_filenames = set(df["filename"].unique())

    logger.info(f"Found {len(processed_filenames)} processed files")

    # Find files that weren't processed (empty)
    empty_files = [f for f in all_files if Path(f).name not in processed_filenames]

    return {
        "total_xml_files": len(all_files),
        "processed_files": len(processed_filenames),
        "empty_files": len(empty_files),
        "empty_file_list": empty_files,
    }


def validate_image_file(
    image_path: Path, min_size_bytes: int = MIN_IMAGE_SIZE_BYTES
) -> ImageValidationResult:
    """
    Validate a downloaded image file.

    Checks if the image file:
    1. Exists and has minimum size (not empty/truncated)
    2. Is a valid image format that can be opened
    3. Has reasonable dimensions

    Args:
        image_path: Path to image file
        min_size_bytes: Minimum expected file size (default: 1KB)

    Returns:
        ImageValidationResult with validation details

    Example:
        >>> result = validate_image_file(Path("image.jpg"))
        >>> if not result.is_valid:
        ...     print(f"Invalid: {result.error}")
    """
    # Check file exists
    if not image_path.exists():
        return ImageValidationResult(
            is_valid=False, file_path=image_path, error="File does not exist"
        )

    # Check file size
    file_size: Optional[int] = None
    try:
        file_size = image_path.stat().st_size

        if file_size < min_size_bytes:
            return ImageValidationResult(
                is_valid=False,
                file_path=image_path,
                file_size=file_size,
                error=f"File too small ({file_size} bytes < {min_size_bytes} bytes)",
            )

        # Try to open and validate image
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format

            # Check for reasonable dimensions
            if width == 0 or height == 0:
                return ImageValidationResult(
                    is_valid=False,
                    file_path=image_path,
                    file_size=file_size,
                    width=width,
                    height=height,
                    format=format_name,
                    error="Image has zero width or height",
                )

            return ImageValidationResult(
                is_valid=True,
                file_path=image_path,
                file_size=file_size,
                width=width,
                height=height,
                format=format_name,
            )

    except Exception as e:
        return ImageValidationResult(
            is_valid=False,
            file_path=image_path,
            file_size=file_size,
            error=f"Failed to validate image: {str(e)}",
        )


def check_image_size(image_path: Path, min_size_bytes: int = MIN_IMAGE_SIZE_BYTES) -> bool:
    """
    Quick check if image file meets minimum size requirement.

    Args:
        image_path: Path to image file
        min_size_bytes: Minimum expected file size (default: 1KB)

    Returns:
        True if file exists and meets size requirement

    Example:
        >>> if not check_image_size(Path("image.jpg"), min_size_bytes=5000):
        ...     print("Image too small or missing")
    """
    if not image_path.exists():
        return False

    try:
        return image_path.stat().st_size >= min_size_bytes
    except Exception:
        return False


def verify_mets_completeness(source_name: str) -> Dict[str, Any]:
    """
    Verify completeness of downloaded files against METS XML references.

    Checks both images and ALTO fulltext files referenced in METS to ensure
    all expected files have been downloaded.

    Args:
        source_name: Name of the source to check (e.g., 'der_tag')

    Returns:
        Dictionary with completeness statistics:
        - mets_files_checked: Number of METS files processed
        - images_expected: Total images referenced in METS
        - images_found: Number of images actually present
        - images_missing: Number of missing images
        - alto_expected: Total ALTO files referenced in METS
        - alto_found: Number of ALTO files present
        - alto_missing: Number of missing ALTO files
        - missing_images_list: List of missing image paths
        - missing_alto_list: List of missing ALTO paths

    Example:
        >>> result = verify_mets_completeness("der_tag")
        >>> print(f"Missing {result['images_missing']} images")
    """
    from lxml import etree

    # Load source configuration
    config = load_source_config(source_name)
    paths = get_source_paths(config)
    raw_dir = paths["raw_dir"]

    # Get paths
    dataset_name = config["dataset_name"]
    from newspaper_explorer.config.base import get_config

    base_config = get_config()
    data_dir = Path(base_config.data_dir)
    images_dir = data_dir / "raw" / dataset_name / "images"
    xml_dir = data_dir / "raw" / dataset_name / config["data_type"]

    logger.info(f"Checking completeness for source: {source_name}")
    logger.info(f"XML directory: {xml_dir}")
    logger.info(f"Images directory: {images_dir}")

    # Find all METS files (excluding fulltext subdirectories)
    mets_files = natsorted([f for f in xml_dir.rglob("*.xml") if "fulltext" not in str(f)])
    logger.info(f"Found {len(mets_files)} METS files to check")

    NAMESPACES = {
        "mets": "http://www.loc.gov/METS/",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    images_expected = 0
    images_found = 0
    images_missing_list = []

    alto_expected = 0
    alto_found = 0
    alto_missing_list = []

    for mets_file in mets_files:
        try:
            tree = etree.parse(str(mets_file))
            root = tree.getroot()

            # Get relative path from xml_dir for directory structure
            try:
                relative_path = mets_file.parent.relative_to(xml_dir)
            except ValueError:
                relative_path = Path(".")

            # Check MAX images
            max_file_grp = root.find('.//mets:fileGrp[@USE="MAX"]', NAMESPACES)
            if max_file_grp is not None:
                for file_elem in max_file_grp.findall(".//mets:file", NAMESPACES):
                    file_id = file_elem.get("ID", "unknown")
                    images_expected += 1

                    # Determine extension from MIMETYPE
                    mimetype = file_elem.get("MIMETYPE", "image/jpg")
                    ext = ".jpg" if "jpg" in mimetype or "jpeg" in mimetype else ".tif"

                    # Check if image exists
                    image_path = images_dir / relative_path / f"{file_id}{ext}"
                    if image_path.exists():
                        images_found += 1
                    else:
                        images_missing_list.append(str(image_path.relative_to(images_dir)))

            # Check FULLTEXT (ALTO) files
            fulltext_grp = root.find('.//mets:fileGrp[@USE="FULLTEXT"]', NAMESPACES)
            if fulltext_grp is not None:
                for file_elem in fulltext_grp.findall(".//mets:file", NAMESPACES):
                    file_id = file_elem.get("ID", "unknown")
                    alto_expected += 1

                    # ALTO files are in fulltext/ subdirectory
                    # Extract page number from file_id (e.g., "fulltext_1" -> "001")
                    page_match = file_id.replace("fulltext_", "")
                    if page_match.isdigit():
                        page_num = f"{int(page_match):03d}"

                        # Build expected ALTO filename
                        # Pattern: {newspaper_id}_{date}_{volume}_{issue}_{edition}_{subedition}_{page}.xml
                        mets_stem = mets_file.stem
                        alto_filename = f"{mets_stem}_{page_num}.xml"
                        alto_path = xml_dir / relative_path / "fulltext" / alto_filename

                        if alto_path.exists():
                            alto_found += 1
                        else:
                            alto_missing_list.append(str(alto_path.relative_to(xml_dir)))

        except Exception as e:
            logger.warning(f"Error processing {mets_file.name}: {e}")
            continue

    images_missing = images_expected - images_found
    alto_missing = alto_expected - alto_found

    logger.info("=" * 60)
    logger.info("Completeness Check Results")
    logger.info("=" * 60)
    logger.info(f"METS files checked: {len(mets_files)}")
    logger.info(f"\nImages:")
    logger.info(f"  Expected: {images_expected}")
    logger.info(f"  Found:    {images_found}")
    logger.info(f"  Missing:  {images_missing}")
    logger.info(f"\nALTO files:")
    logger.info(f"  Expected: {alto_expected}")
    logger.info(f"  Found:    {alto_found}")
    logger.info(f"  Missing:  {alto_missing}")
    logger.info("=" * 60)

    return {
        "mets_files_checked": len(mets_files),
        "images_expected": images_expected,
        "images_found": images_found,
        "images_missing": images_missing,
        "alto_expected": alto_expected,
        "alto_found": alto_found,
        "alto_missing": alto_missing,
        "missing_images_list": images_missing_list,
        "missing_alto_list": alto_missing_list,
    }
