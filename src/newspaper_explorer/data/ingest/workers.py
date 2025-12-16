"""
Worker functions for parallel ALTO XML processing.

These functions are designed to run in separate processes for parallel
file processing. They handle ALTO parsing with METS metadata enrichment.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from newspaper_explorer.data.parser.alto import ALTOParser
from newspaper_explorer.data.parser.mets import METSParser
from newspaper_explorer.models.data.content import IssueMetadata

logger = logging.getLogger(__name__)


def parse_file_worker(
    filepath: str, source_name: str, mets_cache: Optional[dict[str, IssueMetadata]] = None
) -> tuple[list[dict[str, Any]], bool]:
    """
    Worker function for parallel processing with METS enrichment.

    Finds corresponding METS file, parses metadata, and enriches ALTO lines.
    Uses METS cache to avoid re-parsing the same METS file multiple times.

    Args:
        filepath: Path to ALTO XML file
        source_name: Source identifier (e.g., "der_tag")
        mets_cache: Optional cache of already-parsed METS metadata (IssueMetadata objects)

    Returns:
        Tuple of (list of line dicts, success flag)
    """
    try:
        filepath_obj = Path(filepath)

        # Find METS file for this ALTO file
        mets_parser = METSParser()
        mets_file = mets_parser.find_mets_for_alto(filepath_obj)

        issue_metadata: Optional[IssueMetadata] = None
        if mets_file and mets_file.exists():
            mets_path_str = str(mets_file)

            # Check cache first (within this worker process)
            if mets_cache is not None and mets_path_str in mets_cache:
                issue_metadata = mets_cache[mets_path_str]
            else:
                # Parse METS file and cache it
                parsed_metadata = mets_parser.parse_file(mets_file)
                if parsed_metadata:
                    issue_metadata = parsed_metadata
                    # Cache for this worker
                    if mets_cache is not None:
                        mets_cache[mets_path_str] = issue_metadata

        # Convert IssueMetadata to dict format expected by ALTOParser
        mets_dict: Optional[dict[str, Union[str, int, None]]] = None
        if issue_metadata:
            mets_dict = {
                "year_volume": issue_metadata.year_volume,
                "page_count": issue_metadata.page_count,
                "newspaper_title": issue_metadata.newspaper_title,
            }

        # Parse ALTO file with METS metadata
        alto_parser = ALTOParser()
        lines = alto_parser.parse_file(
            filepath_obj, source_name=source_name, mets_metadata=mets_dict
        )

        # Handle empty pages: create a sentinel row to preserve page structure
        if len(lines) == 0:
            empty_line = alto_parser.create_empty_page_line(
                filepath_obj, source_name, mets_metadata=mets_dict
            )
            if empty_line:
                return [empty_line.model_dump()], True

        return [line.model_dump() for line in lines], True

    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Error in worker for {filepath}: {e}")
        return [], False


def parse_mets_worker(mets_path: str) -> tuple[str, Optional[IssueMetadata]]:
    """
    Worker function for parallel METS parsing.

    Args:
        mets_path: Path to METS file

    Returns:
        Tuple of (mets_path, IssueMetadata or None)
    """
    try:
        mets_parser = METSParser()
        mets_file = Path(mets_path)
        issue_metadata = mets_parser.parse_file(mets_file)
        return mets_path, issue_metadata

    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Error parsing METS file {mets_path}: {e}")
        return mets_path, None
