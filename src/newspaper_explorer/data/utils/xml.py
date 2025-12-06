"""
XML and METS file utilities.

General-purpose utilities for working with XML and METS files
in the newspaper data pipeline.
"""

import logging
from pathlib import Path

from natsort import natsorted


def find_xml_files(directory: Path, pattern: str = "**/*.xml") -> list[Path]:
    """
    Find XML files matching a pattern in a directory.

    Generic utility for finding XML files with custom patterns.
    Results are naturally sorted for consistent ordering.

    Args:
        directory: Directory to search
        pattern: Glob pattern for matching files (default: "**/*.xml")

    Returns:
        Naturally sorted list of matching file paths

    Example:
        >>> from pathlib import Path
        >>> # Find all ALTO files in fulltext directories
        >>> alto_files = find_xml_files(Path("data/raw"), "**/fulltext/*.xml")
        >>> len(alto_files)
        5678
    """
    if not directory.exists():
        return []

    return natsorted(list(directory.glob(pattern)))


def find_mets_files(xml_dir: Path) -> list[Path]:
    """
    Find all METS XML files, excluding fulltext subdirectories.

    METS files are typically in the root of each issue directory,
    while ALTO fulltext files are in fulltext/ subdirectories.

    Args:
        xml_dir: Directory to search for METS files

    Returns:
        Naturally sorted list of METS file paths

    Example:
        >>> from pathlib import Path
        >>> mets_files = find_mets_files(Path("data/raw/der_tag/xml_ocr"))
        >>> len(mets_files)
        1234
    """
    if not xml_dir.exists():
        return []

    logger = logging.getLogger(__name__)

    mets_files = [f for f in xml_dir.rglob("*.xml") if "fulltext" not in str(f)]
    mets_files = natsorted(mets_files)

    if mets_files:
        logger.info(f"Found {len(mets_files)} METS files")
    else:
        logger.warning(f"XML directory not found or empty: {xml_dir}")

    return mets_files


def get_file_extension_from_mimetype(mimetype: str) -> str:
    """
    Determine file extension from METS MIMETYPE attribute.

    Used when extracting image references from METS XML to determine
    the correct file extension for downloaded images.

    Args:
        mimetype: MIME type string (e.g., "image/jpeg", "image/tiff")

    Returns:
        File extension including dot (e.g., ".jpg", ".tif")

    Example:
        >>> get_file_extension_from_mimetype("image/jpeg")
        '.jpg'
        >>> get_file_extension_from_mimetype("image/tiff")
        '.tif'
    """
    # JPEG variants
    if "jpg" in mimetype or "jpeg" in mimetype:
        return ".jpg"

    # TIFF variants (default)
    return ".tif"
