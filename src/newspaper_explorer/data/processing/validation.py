"""
Data processing validation for newspaper data quality assessment.

Functions for validating processed data, finding empty files,
and checking completeness against METS references.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import polars as pl
from lxml import etree

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.parser.mets import METSParser
from newspaper_explorer.data.utils.sources import get_source_paths, load_source_config
from newspaper_explorer.data.utils.xml import (
    find_mets_files,
    find_xml_files,
    get_file_extension_from_mimetype,
)

logger = logging.getLogger(__name__)

# Module-level constants
METS_NAMESPACES = {
    "mets": "http://www.loc.gov/METS/",
    "xlink": "http://www.w3.org/1999/xlink",
}


def find_empty_xml_files(source_name: str) -> dict[str, Any]:
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
    base_config = get_config()
    pattern = config.loading.pattern if config.loading else base_config.default_alto_pattern

    logger.info(f"Scanning for XML files in {raw_dir}")
    xml_files = find_xml_files(raw_dir, pattern)
    all_files = [str(f.relative_to(raw_dir)) for f in xml_files]

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


def _check_alto_mets_pairing(
    alto_file: Path, raw_dir: Path, parser: METSParser
) -> tuple[bool, bool, Optional[str]]:
    """Check if ALTO file has parent METS and is listed in it.

    Returns:
        Tuple of (has_mets, is_in_mets, relative_path_or_none)
    """
    # Find parent METS file
    mets_file = parser.find_mets_for_alto(alto_file)

    if mets_file is None or not mets_file.exists():
        return False, False, str(alto_file.relative_to(raw_dir))

    # Check if ALTO is referenced in the METS file
    try:
        tree = etree.parse(str(mets_file))
        root = tree.getroot()

        # Get expected page number from ALTO filename
        alto_stem = alto_file.stem
        page_match = re.search(r"_(\d{3})$", alto_stem)

        if not page_match:
            logger.warning(f"Could not extract page number from ALTO: {alto_file.name}")
            return True, False, str(alto_file.relative_to(raw_dir))

        page_num = int(page_match.group(1))

        # Look for corresponding fulltext file reference
        fulltext_grp = root.find('.//mets:fileGrp[@USE="FULLTEXT"]', METS_NAMESPACES)
        if fulltext_grp is None:
            return True, False, str(alto_file.relative_to(raw_dir))

        # Check if this page is referenced
        for file_elem in fulltext_grp.findall(".//mets:file", METS_NAMESPACES):
            file_id = file_elem.get("ID", "")
            if file_id == f"fulltext_{page_num}":
                return True, True, None

        # Not found in references
        logger.debug(f"ALTO not referenced in METS: {alto_file.name} (page {page_num})")
        return True, False, str(alto_file.relative_to(raw_dir))

    except (OSError, etree.XMLSyntaxError) as e:
        logger.error(f"Error checking METS reference for {alto_file.name}: {e}")
        return True, False, str(alto_file.relative_to(raw_dir))


def validate_alto_mets_relationship(source_name: str) -> dict[str, Any]:
    """
    Validate that each ALTO file has a parent METS file and is listed in it.

    Checks that:
    1. Each ALTO file can find its parent METS file
    2. The ALTO file is actually referenced in that METS file

    Args:
        source_name: Name of the source to check (e.g., 'der_tag')

    Returns:
        Dictionary with validation statistics:
        - total_alto_files: Total number of ALTO files found
        - alto_with_mets: Number of ALTO files with parent METS
        - alto_without_mets: Number of ALTO files without parent METS
        - alto_not_in_mets: Number of ALTO files not listed in their parent METS
        - orphaned_alto_list: List of ALTO files without parent METS
        - unlisted_alto_list: List of ALTO files not referenced in METS

    Example:
        >>> result = validate_alto_mets_relationship("der_tag")
        >>> print(f"Found {result['alto_without_mets']} orphaned ALTO files")
    """
    logger.info(f"Validating ALTO-METS relationships for source: {source_name}")

    # Load source configuration and get paths
    config = load_source_config(source_name)
    paths = get_source_paths(config)
    raw_dir = paths["raw_dir"]

    # Get loading pattern for ALTO files
    base_config = get_config()
    pattern = config.loading.pattern if config.loading else base_config.default_alto_pattern

    # Find all ALTO files
    logger.info(f"Scanning for ALTO files in {raw_dir}")
    alto_files = find_xml_files(raw_dir, pattern)
    logger.info(f"Found {len(alto_files)} ALTO files")
    total_alto = len(alto_files)

    parser = METSParser()

    alto_with_mets = 0
    alto_without_mets = 0
    alto_not_in_mets = 0
    orphaned_alto_list: list[str] = []
    unlisted_alto_list: list[str] = []

    # Validate each ALTO file
    for alto_file in alto_files:
        has_mets, is_in_mets, relative_path = _check_alto_mets_pairing(alto_file, raw_dir, parser)

        if not has_mets:
            alto_without_mets += 1
            if relative_path:
                orphaned_alto_list.append(relative_path)
            logger.debug(f"No METS found for ALTO: {alto_file.name}")
        elif not is_in_mets:
            alto_with_mets += 1
            alto_not_in_mets += 1
            if relative_path:
                unlisted_alto_list.append(relative_path)
        else:
            alto_with_mets += 1

    logger.info("=" * 60)
    logger.info("ALTO-METS Relationship Validation Results")
    logger.info("=" * 60)
    logger.info(f"Total ALTO files:              {total_alto}")
    logger.info(f"ALTO with valid METS:          {alto_with_mets}")
    logger.info(f"ALTO without parent METS:      {alto_without_mets}")
    logger.info(f"ALTO not listed in METS:       {alto_not_in_mets}")
    logger.info("=" * 60)

    return {
        "total_alto_files": total_alto,
        "alto_with_mets": alto_with_mets,
        "alto_without_mets": alto_without_mets,
        "alto_not_in_mets": alto_not_in_mets,
        "orphaned_alto_list": orphaned_alto_list,
        "unlisted_alto_list": unlisted_alto_list,
    }


def _check_images_in_mets(
    mets_file: Path,
    xml_dir: Path,
    images_dir: Path,
) -> tuple[int, int, list[str]]:
    """Check MAX images referenced in a METS file.

    Returns:
        Tuple of (expected_count, found_count, missing_list)
    """
    tree = etree.parse(str(mets_file))
    root = tree.getroot()

    # Get relative path for directory structure
    try:
        relative_path = mets_file.parent.relative_to(xml_dir)
    except ValueError:
        relative_path = Path()

    expected = 0
    found = 0
    missing = []

    # Check MAX images
    max_file_grp = root.find('.//mets:fileGrp[@USE="MAX"]', METS_NAMESPACES)
    if max_file_grp is not None:
        for file_elem in max_file_grp.findall(".//mets:file", METS_NAMESPACES):
            file_id = file_elem.get("ID", "unknown")
            expected += 1

            # Determine extension from MIMETYPE
            mimetype = file_elem.get("MIMETYPE", "image/jpg")
            ext = get_file_extension_from_mimetype(mimetype)

            # Check if image exists
            image_path = images_dir / relative_path / f"{file_id}{ext}"
            if image_path.exists():
                found += 1
            else:
                missing.append(str(image_path.relative_to(images_dir)))

    return expected, found, missing


def _check_alto_in_mets(
    mets_file: Path,
    xml_dir: Path,
) -> tuple[int, int, list[str]]:
    """Check FULLTEXT (ALTO) files referenced in a METS file.

    Returns:
        Tuple of (expected_count, found_count, missing_list)
    """
    tree = etree.parse(str(mets_file))
    root = tree.getroot()

    # Get relative path for directory structure
    try:
        relative_path = mets_file.parent.relative_to(xml_dir)
    except ValueError:
        relative_path = Path()

    expected = 0
    found = 0
    missing = []

    # Check FULLTEXT (ALTO) files
    fulltext_grp = root.find('.//mets:fileGrp[@USE="FULLTEXT"]', METS_NAMESPACES)
    if fulltext_grp is not None:
        for file_elem in fulltext_grp.findall(".//mets:file", METS_NAMESPACES):
            file_id = file_elem.get("ID", "unknown")
            expected += 1

            # ALTO files are in fulltext/ subdirectory
            # Extract page number from file_id (e.g., "fulltext_1" -> "001")
            page_match = file_id.replace("fulltext_", "")
            if page_match.isdigit():
                page_num = f"{int(page_match):03d}"

                # Build expected ALTO filename
                mets_stem = mets_file.stem
                alto_filename = f"{mets_stem}_{page_num}.xml"
                alto_path = xml_dir / relative_path / "fulltext" / alto_filename

                if alto_path.exists():
                    found += 1
                else:
                    missing.append(str(alto_path.relative_to(xml_dir)))

    return expected, found, missing


def verify_mets_completeness(source_name: str) -> dict[str, Any]:
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

    # Load source configuration
    source_config = load_source_config(source_name)
    dataset_name = source_config.dataset_name

    base_config = get_config()
    data_dir = Path(base_config.data_dir)
    images_dir = data_dir / "raw" / dataset_name / "images"
    xml_dir = data_dir / "raw" / dataset_name / source_config.data_type

    logger.info(f"Checking completeness for source: {source_name}")
    logger.info(f"XML directory: {xml_dir}")
    logger.info(f"Images directory: {images_dir}")

    # Find all METS files
    mets_files = find_mets_files(xml_dir)
    logger.info(f"Found {len(mets_files)} METS files to check")

    # Track totals
    images_expected = 0
    images_found = 0
    images_missing_list: list[str] = []

    alto_expected = 0
    alto_found = 0
    alto_missing_list: list[str] = []

    # Process each METS file
    for mets_file in mets_files:
        try:
            # Check images
            img_exp, img_fnd, img_miss = _check_images_in_mets(mets_file, xml_dir, images_dir)
            images_expected += img_exp
            images_found += img_fnd
            images_missing_list.extend(img_miss)

            # Check ALTO files
            alto_exp, alto_fnd, alto_miss = _check_alto_in_mets(mets_file, xml_dir)
            alto_expected += alto_exp
            alto_found += alto_fnd
            alto_missing_list.extend(alto_miss)

        except (OSError, etree.XMLSyntaxError) as e:
            logger.warning(f"Error processing {mets_file.name}: {e}")
            continue

    images_missing = images_expected - images_found
    alto_missing = alto_expected - alto_found

    logger.info("=" * 60)
    logger.info("Completeness Check Results")
    logger.info("=" * 60)
    logger.info(f"METS files checked: {len(mets_files)}")
    logger.info("\nImages:")
    logger.info(f"  Expected: {images_expected}")
    logger.info(f"  Found:    {images_found}")
    logger.info(f"  Missing:  {images_missing}")
    logger.info("\nALTO files:")
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
