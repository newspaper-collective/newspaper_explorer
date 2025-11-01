"""
Worker functions for parallel ALTO XML processing.

These functions are designed to run in separate processes for parallel
file processing. They handle ALTO parsing with METS metadata enrichment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from newspaper_explorer.data.parser.alto import ALTOParser
from newspaper_explorer.data.parser.mets import METSParser

logger = logging.getLogger(__name__)


def parse_file_worker(
    filepath: str, mets_cache: Optional[Dict[str, Dict]] = None
) -> tuple[List[Dict], bool]:
    """
    Worker function for parallel processing with METS enrichment.

    Finds corresponding METS file, parses metadata, and enriches ALTO lines.
    Uses METS cache to avoid re-parsing the same METS file multiple times.

    Args:
        filepath: Path to ALTO XML file
        mets_cache: Optional cache of already-parsed METS metadata

    Returns:
        Tuple of (list of line dicts, success flag)
    """
    try:
        filepath_obj = Path(filepath)

        # Find METS file for this ALTO file
        mets_parser = METSParser()
        mets_file = mets_parser.find_mets_for_alto(filepath_obj)

        mets_metadata = None
        if mets_file and mets_file.exists():
            mets_path_str = str(mets_file)

            # Check cache first (within this worker process)
            if mets_cache is not None and mets_path_str in mets_cache:
                mets_metadata = mets_cache[mets_path_str]
            else:
                # Parse METS file and cache it
                issue_metadata = mets_parser.parse_file(mets_file)
                if issue_metadata:
                    mets_metadata = {
                        "year_volume": issue_metadata.year_volume,
                        "page_count": issue_metadata.page_count,
                        "newspaper_title": issue_metadata.newspaper_title,
                        "newspaper_subtitle": issue_metadata.newspaper_subtitle,
                    }
                    # Cache for this worker
                    if mets_cache is not None:
                        mets_cache[mets_path_str] = mets_metadata

        # Parse ALTO file with METS metadata
        alto_parser = ALTOParser()
        lines = alto_parser.parse_file(filepath_obj, mets_metadata=mets_metadata)

        return [line.to_dict() for line in lines], True

    except Exception as e:
        logger.error(f"Error in worker for {filepath}: {e}")
        return [], False


def parse_mets_worker(mets_path: str) -> tuple[str, Optional[Dict]]:
    """
    Worker function for parallel METS parsing.

    Args:
        mets_path: Path to METS file

    Returns:
        Tuple of (mets_path, metadata_dict or None)
    """
    try:
        mets_parser = METSParser()
        mets_file = Path(mets_path)
        issue_metadata = mets_parser.parse_file(mets_file)

        if issue_metadata:
            metadata = {
                "year_volume": issue_metadata.year_volume,
                "page_count": issue_metadata.page_count,
                "newspaper_title": issue_metadata.newspaper_title,
                "newspaper_subtitle": issue_metadata.newspaper_subtitle,
            }
            return mets_path, metadata
        return mets_path, None

    except Exception as e:
        logger.error(f"Error parsing METS file {mets_path}: {e}")
        return mets_path, None
