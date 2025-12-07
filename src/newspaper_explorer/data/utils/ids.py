"""
Unified ID generation and parsing utilities for newspaper data.

This module provides consistent ID generation and parsing across all data processing stages:
- Source data (ALTO/METS parsing)
- Preprocessing
- Analysis (layout detection, entity extraction, etc.)

All IDs follow a hierarchical structure to enable efficient linking and querying.

ID Hierarchy:
    source_id -> issue_id -> page_id -> text_block_id -> line_id
                          -> page_id -> detection_id
                                     -> article_id

Canonical source_id format: source_name (e.g., "der_tag")
- Human-readable and matches directory structure
- ZDB ID stored separately in config for provenance
"""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import logging
import re
from typing import TYPE_CHECKING, NamedTuple, Optional
import uuid

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


# ============================================================================
# Filename Parsing (ALTO/METS source files)
# ============================================================================


@dataclass
class ParsedFilename:
    """
    Components parsed from an ALTO/METS filename.

    ALTO filename format: 3074409X_1902-09-05_000_415_H_2_005.xml
    Components:
    - zdb_prefix: "3074409X" (ZDB ID without hyphen, NOT used for ID generation)
    - date: datetime(1902, 9, 5)
    - unknown_field: "000" (always 000, purpose unknown)
    - issue_number: 415 (sequential publication number, can repeat across editions)
    - separator: "H" (means "Heft" = issue)
    - edition: 2 (edition of the day: 1=morning, 2=midday, 3=evening)
    - page_number: 5

    Note: The zdb_prefix from filenames is NOT used for ID generation.
    We use source_name (e.g., "der_tag") from config instead.

    Important: `issue_number` and `edition` are NOT independent!
    Multiple editions can share the same issue_number (e.g., morning and midday
    editions both labeled as issue 37, while evening edition is issue 38).
    """

    zdb_prefix: str
    date: datetime
    unknown_field: str
    issue_number: int
    separator: str
    edition: int
    page_number: int


# Regex pattern for ALTO filenames
# Format: 3074409X_1902-09-05_000_415_H_2_005.xml
_ALTO_FILENAME_PATTERN = re.compile(
    r"^([A-Z0-9]+)_(\d{4})-(\d{2})-(\d{2})_(\d{3})_(\d+)_([A-Z])_(\d+)_(\d+)"
)

# Regex pattern for METS filenames (no page suffix)
# Format: 3074409X_1902-09-05_000_415_H_2.xml
_METS_FILENAME_PATTERN = re.compile(
    r"^([A-Z0-9]+)_(\d{4})-(\d{2})-(\d{2})_(\d{3})_(\d+)_([A-Z])_(\d+)\.xml$"
)

# Simpler pattern to just extract edition from _H_N_ part
_EDITION_PATTERN = re.compile(r"_([A-Z])_(\d+)(?:_\d+)?\.xml$")


def parse_alto_filename(filename: str) -> Optional[ParsedFilename]:
    """
    Parse an ALTO/METS filename into its components.

    This is the single source of truth for parsing ALTO filenames.
    Use this instead of duplicating regex patterns.

    Args:
        filename: ALTO filename (e.g., "3074409X_1902-09-05_000_415_H_2_005.xml")

    Returns:
        ParsedFilename with all components, or None if format doesn't match

    Example:
        >>> parsed = parse_alto_filename("3074409X_1902-09-05_000_415_H_2_005.xml")
        >>> parsed.date
        datetime.datetime(1902, 9, 5, 0, 0)
        >>> parsed.issue_number
        415
        >>> parsed.page_number
        5
    """
    match = _ALTO_FILENAME_PATTERN.match(filename)
    if not match:
        return None

    zdb_prefix = match.group(1)
    year = int(match.group(2))
    month = int(match.group(3))
    day = int(match.group(4))
    unknown_field = match.group(5)
    issue_number = int(match.group(6))
    separator = match.group(7)
    edition = int(match.group(8))
    page_number = int(match.group(9))

    try:
        date = datetime(year, month, day)
    except ValueError:
        return None

    return ParsedFilename(
        zdb_prefix=zdb_prefix,
        date=date,
        unknown_field=unknown_field,
        issue_number=issue_number,
        separator=separator,
        edition=edition,
        page_number=page_number,
    )


# Maximum reasonable edition number per day
_MAX_EDITION = 9


def extract_edition(
    *,
    filename: Optional[str] = None,
    folder_path: Optional[str] = None,
) -> Optional[int]:
    """
    Extract the numeric edition (1=morning, 2=midday, 3=evening) from available sources.

    This is the **authoritative** function for edition extraction.
    Priority order:
    1. ALTO/METS filename (e.g., "..._H_2_..." → 2)
    2. Folder path (e.g., "/1920/03/03/02/" → 2)

    Args:
        filename: ALTO or METS filename (e.g., "3074409X_1902-09-05_000_415_H_2_005.xml")
        folder_path: Path containing edition folder (e.g., "1920/03/03/02" or Path object)

    Returns:
        Numeric edition (1, 2, or 3), or None if extraction fails

    Example:
        >>> extract_edition(filename="3074409X_1902-09-05_000_415_H_2_005.xml")
        2
        >>> extract_edition(filename="3074409X_1920-03-03_000_53_H_1.xml")  # METS
        1
        >>> extract_edition(folder_path="1920/03/03/02")
        2
    """
    # Priority 1: Extract from filename using simple pattern (works for ALTO and METS)
    if filename:
        # First try full ALTO parsing
        parsed = parse_alto_filename(filename)
        if parsed:
            return parsed.edition
        # Fallback: use simpler pattern for METS filenames (no page suffix)
        match = _EDITION_PATTERN.search(filename)
        if match:
            return int(match.group(2))

    # Priority 2: Extract from folder path
    if folder_path:
        edition = _extract_edition_from_path(folder_path)
        if edition:
            return edition

    return None


def _extract_edition_from_path(path: str) -> Optional[int]:
    """
    Extract edition number from a path like '1920/03/03/02/...'.

    Path structure expected: YYYY/MM/DD/EDITION/[filename or fulltext/...]
    The edition is the 4th component (index 3) after the year.
    """
    path_str = str(path).replace("\\", "/").strip("/")
    parts = path_str.split("/")

    # Look for YYYY/MM/DD/ED pattern
    # Find the year component (4 digits starting with 18 or 19 or 20)
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit() and part[:2] in ("18", "19", "20"):
            # Found year at index i, edition should be at index i+3 (YYYY/MM/DD/ED)
            edition_idx = i + 3
            if edition_idx < len(parts):
                edition_part = parts[edition_idx]
                if edition_part.isdigit() and len(edition_part) <= 2:
                    edition = int(edition_part)
                    if 1 <= edition <= _MAX_EDITION:
                        return edition
            break  # Only check first year found

    return None


# ============================================================================
# ID Generation Functions
# ============================================================================


def generate_source_id(source_name: str) -> str:
    """
    Generate a source identifier.

    Args:
        source_name: Source name (e.g., "der_tag")

    Returns:
        Source ID (same as source_name for simplicity)

    Example:
        >>> generate_source_id("der_tag")
        'der_tag'
    """
    return source_name


def source_id_to_filename_prefix(source_id: str) -> str:
    """
    Convert a source_id to its filename prefix format.

    The source_id uses hyphens (e.g., "3074409-X") for consistency,
    but ALTO/METS filenames use the format without hyphens (e.g., "3074409X").

    Args:
        source_id: Source identifier (e.g., "3074409-X" or "der_tag")

    Returns:
        Filename prefix (e.g., "3074409X" or "der_tag")

    Example:
        >>> source_id_to_filename_prefix("3074409-X")
        '3074409X'
        >>> source_id_to_filename_prefix("der_tag")
        'der_tag'

    Note:
        Only removes hyphens from ZDB-style IDs (ending in -X or similar).
        For regular source names like "der_tag", returns unchanged.
    """
    return source_id.replace("-", "")


def generate_issue_id(
    source: str,
    date: datetime,
    issue_number: int,
    edition: int,
) -> str:
    """
    Generate a unique issue identifier.

    Args:
        source: Source name (e.g., "der_tag")
        date: Publication date
        issue_number: Sequential publication number (e.g., 415)
        edition: Edition of the day (1=morning, 2=midday, 3=evening)

    Returns:
        Issue ID in format: {source}_{YYYY-MM-DD}_{issue:03d}_{edition}

    Example:
        >>> from datetime import datetime
        >>> generate_issue_id("der_tag", datetime(1902, 9, 5), 415, 2)
        'der_tag_1902-09-05_415_2'

    Note:
        Multiple editions can share the same issue_number. For example,
        morning (edition=1) and midday (edition=2) might both be issue 37,
        while evening (edition=3) is issue 38.
    """
    date_str = date.strftime("%Y-%m-%d")
    return f"{source}_{date_str}_{issue_number:03d}_{edition}"


def generate_page_id(
    source: str,
    date: datetime,
    issue_number: int,
    edition: int,
    page_number: int,
) -> str:
    """
    Generate a unique page identifier.

    Args:
        source: Source name (e.g., "der_tag")
        date: Publication date
        issue_number: Sequential publication number (e.g., 415)
        edition: Edition of the day (1=morning, 2=midday, 3=evening)
        page_number: Page number (e.g., 5)

    Returns:
        Page ID in format: {source}_{YYYY-MM-DD}_{issue:03d}_{edition}_{page:03d}

    Example:
        >>> from datetime import datetime
        >>> generate_page_id("der_tag", datetime(1902, 9, 5), 415, 2, 5)
        'der_tag_1902-09-05_415_2_005'
    """
    issue_id = generate_issue_id(source, date, issue_number, edition)
    return f"{issue_id}_{page_number:03d}"


def generate_text_block_id(page_id: str, block_id: str) -> str:
    """
    Generate a unique text block identifier.

    Args:
        page_id: Page ID (from generate_page_id)
        block_id: Block ID from ALTO XML (e.g., "TB_1")

    Returns:
        Text block ID in format: {page_id}_{block_id}

    Example:
        >>> generate_text_block_id("der_tag_1902-09-05_415_2_005", "TB_1")
        'der_tag_1902-09-05_415_2_005_TB_1'
    """
    return f"{page_id}_{block_id}"


def generate_line_id(text_block_id: str, line_id: str) -> str:
    """
    Generate a unique line identifier.

    Args:
        text_block_id: Text block ID (from generate_text_block_id)
        line_id: Line ID from ALTO XML (e.g., "TL_1")

    Returns:
        Line ID in format: {text_block_id}_{line_id}

    Example:
        >>> generate_line_id("der_tag_1902-09-05_415_2_005_TB_1", "TL_1")
        'der_tag_1902-09-05_415_2_005_TB_1_TL_1'
    """
    return f"{text_block_id}_{line_id}"


def generate_detection_id(page_id: str, class_name: str) -> str:
    """
    Generate a stable unique identifier for layout detection.

    Uses short UUID to ensure uniqueness while remaining readable.
    UUID-based IDs are stable across re-runs (unlike index-based).

    Args:
        page_id: Page ID (from generate_page_id)
        class_name: Detection class (e.g., "headline", "text-region")

    Returns:
        Detection ID in format: {page_id}_{class}_{uuid_short}

    Example:
        >>> generate_detection_id("der_tag_1902-09-05_415_2_005", "headline")
        'der_tag_1902-09-05_415_2_005_headline_a3f9c2'
    """
    short_uuid = str(uuid.uuid4())[:6]
    clean_class = class_name.lower().replace("-", "_").replace(" ", "_")
    return f"{page_id}_{clean_class}_{short_uuid}"


def generate_article_id(page_id: str) -> str:
    """
    Generate a unique identifier for a reconstructed article.

    Args:
        page_id: Page ID where article starts (from generate_page_id)

    Returns:
        Article ID in format: {page_id}_art_{uuid_short}

    Example:
        >>> generate_article_id("der_tag_1902-09-05_415_2_005")
        'der_tag_1902-09-05_415_2_005_art_b7e4d1'
    """
    short_uuid = str(uuid.uuid4())[:6]
    return f"{page_id}_art_{short_uuid}"


def generate_entity_id(line_id: str) -> str:
    """
    Generate a unique identifier for an extracted entity.

    Args:
        line_id: Line ID where entity was found (from generate_line_id)

    Returns:
        Entity ID in format: {line_id}_ent_{uuid_short}

    Example:
        >>> generate_entity_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
        'der_tag_1902-09-05_415_2_005_TB_1_TL_1_ent_c5a8b3'
    """
    short_uuid = str(uuid.uuid4())[:6]
    return f"{line_id}_ent_{short_uuid}"


# ============================================================================
# ID Parsing Functions
# ============================================================================


# Compiled regex for date pattern matching (YYYY-MM-DD)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _find_date_index(parts: list[str]) -> Optional[int]:
    """
    Find the index of the date part (YYYY-MM-DD) in a list of ID components.

    This is used internally to handle multi-part source names like "der_tag".
    Uses compiled regex for efficient matching.

    Args:
        parts: List of ID components split by underscore

    Returns:
        Index of the date part, or None if not found

    Example:
        >>> _find_date_index(["der", "tag", "1902-09-05", "415", "2"])
        2
    """
    for i, part in enumerate(parts):
        if _DATE_PATTERN.match(part):
            return i
    return None


class PageIdComponents(NamedTuple):
    """Components extracted from a page_id."""

    source: str
    date: str
    issue_number: int
    edition: int
    page_number: int


class IssueIdComponents(NamedTuple):
    """Components extracted from an issue_id."""

    source: str
    date: str
    issue_number: int
    edition: int


class LineIdComponents(NamedTuple):
    """Components extracted from a line_id."""

    source: str
    date: str
    issue_number: int
    edition: int
    page_number: int
    block_id: str
    line_id: str
    full_page_id: str
    full_text_block_id: str


def parse_page_id(page_id: str) -> PageIdComponents:
    """
    Parse a page ID into its components.

    Args:
        page_id: Page ID (e.g., "der_tag_1902-09-05_415_2_005")

    Returns:
        PageIdComponents with extracted fields

    Raises:
        ValueError: If page_id format is invalid

    Example:
        >>> components = parse_page_id("der_tag_1902-09-05_415_2_005")
        >>> components.source
        'der_tag'
        >>> components.page_number
        5
    """
    parts = page_id.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Invalid page_id format: {page_id}. "
            f"Expected: {{source}}_{{date}}_{{issue}}_{{daily}}_{{page}}"
        )

    # Handle multi-part source names (e.g., "der_tag")
    date_idx = _find_date_index(parts)
    if date_idx is None:
        raise ValueError(f"Could not find date in page_id: {page_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    edition = int(parts[date_idx + 2])
    page_number = int(parts[date_idx + 3])

    return PageIdComponents(
        source=source,
        date=date_str,
        issue_number=issue_number,
        edition=edition,
        page_number=page_number,
    )


def parse_issue_id(issue_id: str) -> IssueIdComponents:
    """
    Parse an issue ID into its components.

    Args:
        issue_id: Issue ID (e.g., "der_tag_1902-09-05_415_2")

    Returns:
        IssueIdComponents with extracted fields

    Raises:
        ValueError: If issue_id format is invalid

    Example:
        >>> components = parse_issue_id("der_tag_1902-09-05_415_2")
        >>> components.source
        'der_tag'
        >>> components.issue_number
        415
    """
    parts = issue_id.split("_")
    if len(parts) < 4:
        raise ValueError(
            f"Invalid issue_id format: {issue_id}. "
            f"Expected: {{source}}_{{date}}_{{issue}}_{{daily}}"
        )

    # Handle multi-part source names (e.g., "der_tag")
    date_idx = _find_date_index(parts)
    if date_idx is None:
        raise ValueError(f"Could not find date in issue_id: {issue_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    edition = int(parts[date_idx + 2])

    return IssueIdComponents(
        source=source,
        date=date_str,
        issue_number=issue_number,
        edition=edition,
    )


def parse_line_id(line_id: str) -> LineIdComponents:
    """
    Parse a line ID into its components.

    Args:
        line_id: Line ID (e.g., "der_tag_1902-09-05_415_2_005_TB_1_TL_1")

    Returns:
        LineIdComponents with extracted fields

    Raises:
        ValueError: If line_id format is invalid

    Example:
            >>> components = parse_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
            >>> components.source
            'der_tag'
            >>> components.line_id
            'TL_1'
    """
    parts = line_id.split("_")
    if len(parts) < 7:
        raise ValueError(
            f"Invalid line_id format: {line_id}. "
            f"Expected: {{source}}_{{date}}_{{issue}}_{{daily}}_{{page}}_{{block}}_{{line}}"
        )

    # Handle multi-part source names (e.g., "der_tag")
    date_idx = _find_date_index(parts)
    if date_idx is None:
        raise ValueError(f"Could not find date in line_id: {line_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    edition = int(parts[date_idx + 2])
    page_number = int(parts[date_idx + 3])

    # Find line ID marker (TL = TextLine) using tuple for O(1) lookup
    # The rest between page and line ID is the block ID
    parts_tuple = tuple(parts)
    try:
        # Try to find TL marker starting from after page number
        tl_idx = parts_tuple.index("TL", date_idx + 4)
    except ValueError:
        tl_idx = None

    if tl_idx is None:
        # No TL marker found, assume everything after page is block_id + line_id
        block_parts = parts[date_idx + 4 :]
        if len(block_parts) >= 2:
            # Assume last part is line number, rest is block
            block_id = "_".join(block_parts[:-1])
            line_id_str = block_parts[-1]
        else:
            block_id = "_".join(block_parts)
            line_id_str = ""
    else:
        # Block ID is between page and TL marker
        block_id = "_".join(parts[date_idx + 4 : tl_idx])
        line_id_str = "_".join(parts[tl_idx:])

    full_page_id = f"{source}_{date_str}_{issue_number:03d}_{edition}_{page_number:03d}"
    full_text_block_id = f"{full_page_id}_{block_id}"

    return LineIdComponents(
        source=source,
        date=date_str,
        issue_number=issue_number,
        edition=edition,
        page_number=page_number,
        block_id=block_id,
        line_id=line_id_str,
        full_page_id=full_page_id,
        full_text_block_id=full_text_block_id,
    )


# ============================================================================
# ID Extraction Functions (from existing data)
# ============================================================================


def extract_issue_id_from_page_id(page_id: str) -> str:
    """
    Extract issue_id from a page_id.

    Args:
        page_id: Page ID (e.g., "der_tag_1902-09-05_415_2_005")

    Returns:
        Issue ID (e.g., "der_tag_1902-09-05_415_2")

    Example:
        >>> extract_issue_id_from_page_id("der_tag_1902-09-05_415_2_005")
        'der_tag_1902-09-05_415_2'
    """
    # Use parsing to handle multi-part source names
    components = parse_page_id(page_id)
    return (
        f"{components.source}_{components.date}_{components.issue_number:03d}_{components.edition}"
    )


def extract_page_id_from_text_block_id(text_block_id: str) -> str:
    """
    Extract page_id from a text_block_id.

    Args:
        text_block_id: Text block ID (e.g., "der_tag_1902-09-05_415_2_005_TB_1")

    Returns:
        Page ID (e.g., "der_tag_1902-09-05_415_2_005")

    Example:
        >>> extract_page_id_from_text_block_id("der_tag_1902-09-05_415_2_005_TB_1")
        'der_tag_1902-09-05_415_2_005'
    """
    parts = text_block_id.split("_")
    date_idx = _find_date_index(parts)
    if date_idx is None:
        raise ValueError(f"Could not find date in text_block_id: {text_block_id}")

    # Page ID includes: source + date + issue + daily + page (5 parts after source)
    return "_".join(parts[: date_idx + 4])


def extract_text_block_id_from_line_id(line_id: str) -> str:
    """
    Extract text_block_id from a line_id.

    Args:
        line_id: Line ID (e.g., "der_tag_1902-09-05_415_2_005_TB_1_TL_1")

    Returns:
        Text block ID (e.g., "der_tag_1902-09-05_415_2_005_TB_1")

    Example:
        >>> extract_text_block_id_from_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
        'der_tag_1902-09-05_415_2_005_TB_1'
    """
    parts = line_id.split("_")

    # Find the TL (TextLine) marker
    tl_idx = None
    for i, part in enumerate(parts):
        if part == "TL":
            tl_idx = i
            break

    if tl_idx is None:
        raise ValueError(f"Could not find TL marker in line_id: {line_id}")

    # Text block ID is everything before TL marker
    return "_".join(parts[:tl_idx])


# ============================================================================
# ID Type Identification
# ============================================================================


def identify_id_type(id_string: str) -> str:
    """
    Identify the type of ID from its structure.

    This utility function examines an ID string and determines what type
    of identifier it represents. Useful for generic processing where the
    ID type is unknown.

    Args:
        id_string: Any ID string from the system

    Returns:
        ID type: "source_id", "issue_id", "page_id", "text_block_id",
                "line_id", "detection_id", "article_id", "entity_id",
                "doc_id", or "unknown"

    Example:
        >>> identify_id_type("der_tag")
        'source_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2")
        'issue_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005")
        'page_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005_TB_1")
        'text_block_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005_TB_1_TL_1")
        'line_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005_headline_a3f9c2")
        'detection_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005_art_b7e4d1")
        'article_id'
        >>> identify_id_type("3074409-X_1902-09-05_415_2_005_TB_1_TL_1_ent_c5a8b3")
        'entity_id'
    """
    parts = id_string.split("_")

    # Check for special suffixes
    if len(parts) >= 2:
        # Entity ID: ends with _ent_{uuid}
        if len(parts) >= 2 and parts[-2] == "ent":
            return "entity_id"

        # Article ID: ends with _art_{uuid}
        if len(parts) >= 2 and parts[-2] == "art":
            return "article_id"

        # Detection ID: has element type before UUID (headline, image, caption, etc.)
        # Check if second-to-last part looks like a detection class
        detection_classes = [
            "headline",
            "image",
            "caption",
            "advertisement",
            "table",
            "paragraph",
            "title",
        ]
        if parts[-2] in detection_classes or (
            len(parts[-1]) == 6 and all(c in "0123456789abcdef" for c in parts[-1])
        ):
            # Last part is a short UUID (6 hex chars)
            return "detection_id"

    # Find date marker (YYYY-MM-DD format) using compiled regex
    date_idx = None
    for i, part in enumerate(parts):
        if _DATE_PATTERN.match(part):
            date_idx = i
            break

    # No date found - must be source_id or unknown
    if date_idx is None:
        # Simple source names don't contain dates
        if len(parts) <= 2:
            return "source_id"
        return "unknown"

    # Count parts after date to determine ID type
    # Structure: source + date + issue + daily + page + [block] + [line]
    # date_idx points to date, so we have:
    # - date_idx + 1 = issue number
    # - date_idx + 2 = daily issue number
    # - date_idx + 3 = page number
    # - date_idx + 4+ = block/line parts

    parts_after_date = len(parts) - date_idx - 1

    # Issue ID: date + issue + daily (2 parts after date)
    if parts_after_date == 2:
        return "issue_id"

    # Page ID: date + issue + daily + page (3 parts after date)
    if parts_after_date == 3:
        return "page_id"

    # Text block or line ID (4+ parts after date)
    if parts_after_date >= 4:
        # Check for TL (TextLine) marker
        if "TL" in parts[date_idx + 4 :]:
            return "line_id"
        return "text_block_id"

    return "unknown"


@lru_cache(maxsize=10000)
def _extract_foreign_keys_cached(
    id_string: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Internal cached implementation of extract_foreign_keys.

    Returns a tuple instead of dict for hashability/caching.
    Order: (source_id, issue_id, page_id, text_block_id, line_id)
    """
    id_type = identify_id_type(id_string)

    source_id: Optional[str] = None
    issue_id: Optional[str] = None
    page_id: Optional[str] = None
    text_block_id: Optional[str] = None
    line_id: Optional[str] = None

    try:
        if id_type == "line_id" or id_type == "entity_id":
            # Extract line components
            if id_type == "entity_id":
                # Remove _ent_{uuid} suffix
                parts = id_string.split("_")
                line_id_str = "_".join(parts[:-2])
            else:
                line_id_str = id_string

            components = parse_line_id(line_id_str)
            source_id = components.source
            issue_id = (
                f"{components.source}_{components.date}_"
                f"{components.issue_number:03d}_{components.edition}"
            )
            page_id = components.full_page_id
            text_block_id = components.full_text_block_id
            line_id = line_id_str

        elif id_type == "text_block_id":
            # Extract text block components
            page_id_extracted = extract_page_id_from_text_block_id(id_string)
            page_components = parse_page_id(page_id_extracted)
            source_id = page_components.source
            issue_id = extract_issue_id_from_page_id(page_id_extracted)
            page_id = page_id_extracted
            text_block_id = id_string

        elif id_type == "page_id" or id_type == "detection_id" or id_type == "article_id":
            # Extract page components
            if id_type in ["detection_id", "article_id"]:
                # Remove suffix to get page_id
                page_id_extracted = extract_page_id_from_detection_or_article_id(id_string)
            else:
                page_id_extracted = id_string

            page_components = parse_page_id(page_id_extracted)
            source_id = page_components.source
            issue_id = extract_issue_id_from_page_id(page_id_extracted)
            page_id = page_id_extracted

        elif id_type == "issue_id":
            # Extract issue components
            issue_components = parse_issue_id(id_string)
            source_id = issue_components.source
            issue_id = id_string

        elif id_type == "source_id":
            source_id = id_string

    except (ValueError, IndexError) as e:
        logger.warning(f"Could not extract foreign keys from {id_string}: {e}")

    return (source_id, issue_id, page_id, text_block_id, line_id)


def extract_foreign_keys(id_string: str) -> dict[str, Optional[str]]:
    """
    Extract all foreign key IDs from any ID type.

    Given any ID in the system, extract all parent IDs (foreign keys)
    that link this entity to higher levels in the hierarchy.

    Results are cached for performance - repeated calls with the same
    ID string return cached results without re-parsing.

    Args:
        id_string: Any ID string from the system

    Returns:
        Dictionary with all extractable foreign keys:
        - source_id: Source identifier
        - issue_id: Issue identifier (if extractable)
        - page_id: Page identifier (if extractable)
        - text_block_id: Text block identifier (if extractable)
        - line_id: Line identifier (if applicable)

    Example:
        >>> fks = extract_foreign_keys("3074409-X_1902-09-05_415_2_005_TB_1_TL_1")
        >>> fks["source_id"]
        '3074409-X'
        >>> fks["issue_id"]
        '3074409-X_1902-09-05_415_2'
        >>> fks["page_id"]
        '3074409-X_1902-09-05_415_2_005'
        >>> fks["text_block_id"]
        '3074409-X_1902-09-05_415_2_005_TB_1'
    """
    cached = _extract_foreign_keys_cached(id_string)
    return {
        "source_id": cached[0],
        "issue_id": cached[1],
        "page_id": cached[2],
        "text_block_id": cached[3],
        "line_id": cached[4],
    }

    try:
        if id_type == "line_id" or id_type == "entity_id":
            # Extract line components
            if id_type == "entity_id":
                # Remove _ent_{uuid} suffix
                parts = id_string.split("_")
                line_id = "_".join(parts[:-2])
            else:
                line_id = id_string

            components = parse_line_id(line_id)
            result["source_id"] = components.source
            result["issue_id"] = (
                f"{components.source}_{components.date}_"
                f"{components.issue_number:03d}_{components.edition}"
            )
            result["page_id"] = components.full_page_id
            result["text_block_id"] = components.full_text_block_id
            result["line_id"] = line_id

        elif id_type == "text_block_id":
            # Extract text block components
            page_id = extract_page_id_from_text_block_id(id_string)
            page_components = parse_page_id(page_id)
            result["source_id"] = page_components.source
            result["issue_id"] = extract_issue_id_from_page_id(page_id)
            result["page_id"] = page_id
            result["text_block_id"] = id_string

        elif id_type == "page_id" or id_type == "detection_id" or id_type == "article_id":
            # Extract page components
            if id_type in ["detection_id", "article_id"]:
                # Remove suffix to get page_id
                page_id = extract_page_id_from_detection_or_article_id(id_string)
            else:
                page_id = id_string

            page_components = parse_page_id(page_id)
            result["source_id"] = page_components.source
            result["issue_id"] = extract_issue_id_from_page_id(page_id)
            result["page_id"] = page_id

        elif id_type == "issue_id":
            # Extract issue components
            issue_components = parse_issue_id(id_string)
            result["source_id"] = issue_components.source
            result["issue_id"] = id_string

        elif id_type == "source_id":
            result["source_id"] = id_string

    except (ValueError, IndexError) as e:
        logger.warning(f"Could not extract foreign keys from {id_string}: {e}")

    return result


def extract_page_id_from_detection_or_article_id(id_string: str) -> str:
    """
    Extract page_id from a detection_id or article_id.

    Args:
        id_string: Detection or article ID

    Returns:
        Page ID

    Example:
        >>> extract_page_id_from_detection_or_article_id(
        ...     "3074409-X_1902-09-05_415_2_005_headline_a3f9c2"
        ... )
        '3074409-X_1902-09-05_415_2_005'
    """
    parts = id_string.split("_")

    # Find the date part using compiled regex
    date_idx = _find_date_index(parts)

    if date_idx is None:
        raise ValueError(f"Could not find date in ID: {id_string}")

    # Page ID is: source + date + issue + daily + page
    return "_".join(parts[: date_idx + 4])


def add_foreign_key_columns(
    df: "pl.DataFrame",
    id_column: str = "text_block_id",
    *,
    source_df: Optional["pl.DataFrame"] = None,
) -> "pl.DataFrame":
    """
    Add foreign key columns to a DataFrame.

    Efficiently adds source_id, issue_id, page_id, and text_block_id columns.
    If source_df is provided and has these columns, joins from there.
    Otherwise, parses FK columns from the id_column.

    This is the preferred method for adding FK columns to analysis results
    as it avoids redundant parsing when the source DataFrame already has
    the FK columns available.

    Args:
        df: DataFrame to add FK columns to
        id_column: Column containing the document/block ID
        source_df: Optional source DataFrame with FK columns to join from

    Returns:
        DataFrame with FK columns added

    Example:
        >>> # From source DataFrame (preferred - no parsing)
        >>> results_df = add_foreign_key_columns(
        ...     results_df,
        ...     id_column="text_block_id",
        ...     source_df=input_df
        ... )

        >>> # Without source (falls back to parsing)
        >>> results_df = add_foreign_key_columns(results_df, id_column="doc_id")
    """
    # Import here to avoid circular dependency and keep ids.py lightweight
    import polars as pl_impl

    fk_columns = ["source_id", "issue_id", "page_id", "text_block_id"]

    # Check if we can join from source_df
    if source_df is not None:
        available_fks = [col for col in fk_columns if col in source_df.columns]
        if available_fks and id_column in source_df.columns:
            # Exclude id_column from FK columns if it's already a FK (to avoid duplicate)
            cols_to_select = [id_column] + [c for c in available_fks if c != id_column]
            fk_df = source_df.select(cols_to_select).unique(subset=[id_column])
            df = df.join(fk_df, on=id_column, how="left")

            # Fill any missing FK columns with None
            for col in fk_columns:
                if col not in df.columns:
                    df = df.with_columns(pl_impl.lit(None).alias(col))

            return df

    # Fall back to parsing from ID column
    logger.info(f"Parsing foreign keys from {id_column} column...")
    id_list = df[id_column].to_list()
    foreign_keys = [extract_foreign_keys(id_str) if id_str else {} for id_str in id_list]

    # Add FK columns
    return df.with_columns(
        [
            pl_impl.Series("source_id", [fk.get("source_id") for fk in foreign_keys]),
            pl_impl.Series("issue_id", [fk.get("issue_id") for fk in foreign_keys]),
            pl_impl.Series("page_id", [fk.get("page_id") for fk in foreign_keys]),
            pl_impl.Series(
                "text_block_id",
                [fk.get("text_block_id") for fk in foreign_keys],
            ),
        ]
    )
