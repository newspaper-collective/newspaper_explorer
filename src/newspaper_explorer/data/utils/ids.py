"""
Unified ID generation utilities for newspaper data.

This module provides consistent ID generation across all data processing stages:
- Source data (ALTO/METS parsing)
- Preprocessing
- Analysis (layout detection, entity extraction, etc.)

All IDs follow a hierarchical structure to enable efficient linking and querying.

ID Hierarchy:
    source_id -> issue_id -> page_id -> text_block_id -> line_id
                          -> page_id -> detection_id
                                     -> article_id
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, NamedTuple, Optional, Union

logger = logging.getLogger(__name__)


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
    daily_issue_number: int,
) -> str:
    """
    Generate a unique issue identifier.

    Args:
        source: Source name (e.g., "der_tag")
        date: Publication date
        issue_number: Issue number (e.g., 415)
        daily_issue_number: Daily issue number (e.g., 2)

    Returns:
        Issue ID in format: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}

    Example:
        >>> from datetime import datetime
        >>> generate_issue_id("der_tag", datetime(1902, 9, 5), 415, 2)
        'der_tag_1902-09-05_415_2'
    """
    date_str = date.strftime("%Y-%m-%d")
    return f"{source}_{date_str}_{issue_number:03d}_{daily_issue_number}"


def generate_page_id(
    source: str,
    date: datetime,
    issue_number: int,
    daily_issue_number: int,
    page_number: int,
) -> str:
    """
    Generate a unique page identifier.

    Args:
        source: Source name (e.g., "der_tag")
        date: Publication date
        issue_number: Issue number (e.g., 415)
        daily_issue_number: Daily issue number (e.g., 2)
        page_number: Page number (e.g., 5)

    Returns:
        Page ID in format: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}_{page:03d}

    Example:
        >>> from datetime import datetime
        >>> generate_page_id("der_tag", datetime(1902, 9, 5), 415, 2, 5)
        'der_tag_1902-09-05_415_2_005'
    """
    issue_id = generate_issue_id(source, date, issue_number, daily_issue_number)
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


class PageIdComponents(NamedTuple):
    """Components extracted from a page_id."""

    source: str
    date: str
    issue_number: int
    daily_issue_number: int
    page_number: int


class IssueIdComponents(NamedTuple):
    """Components extracted from an issue_id."""

    source: str
    date: str
    issue_number: int
    daily_issue_number: int


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
    # Find the date part (format: YYYY-MM-DD)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            date_idx = i
            break

    if date_idx is None:
        raise ValueError(f"Could not find date in page_id: {page_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    daily_issue_number = int(parts[date_idx + 2])
    page_number = int(parts[date_idx + 3])

    return PageIdComponents(
        source=source,
        date=date_str,
        issue_number=issue_number,
        daily_issue_number=daily_issue_number,
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
    # Find the date part (format: YYYY-MM-DD)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            date_idx = i
            break

    if date_idx is None:
        raise ValueError(f"Could not find date in issue_id: {issue_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    daily_issue_number = int(parts[date_idx + 2])

    return IssueIdComponents(
        source=source,
        date=date_str,
        issue_number=issue_number,
        daily_issue_number=daily_issue_number,
    )


def parse_line_id(line_id: str) -> Dict[str, Union[str, int]]:
    """
    Parse a line ID into its components.

    Args:
        line_id: Line ID (e.g., "der_tag_1902-09-05_415_2_005_TB_1_TL_1")

    Returns:
        Dictionary with extracted components:
        - source: Source identifier
        - date: Date string (YYYY-MM-DD)
        - issue_number: Issue number (int)
        - daily_issue_number: Daily issue number (int)
        - page_number: Page number (int)
        - block_id: Text block ID
        - line_id: Line ID within block
        - full_page_id: Complete page ID
        - full_text_block_id: Complete text block ID

    Raises:
        ValueError: If line_id format is invalid

    Example:
            >>> components = parse_line_id("der_tag_1902-09-05_415_2_005_TB_1_TL_1")
            >>> components["source"]
            'der_tag'
            >>> components["line_id"]
            'TL_1'
    """
    parts = line_id.split("_")
    if len(parts) < 7:
        raise ValueError(
            f"Invalid line_id format: {line_id}. "
            f"Expected: {{source}}_{{date}}_{{issue}}_{{daily}}_{{page}}_{{block}}_{{line}}"
        )

    # Find the date part (format: YYYY-MM-DD)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            date_idx = i
            break

    if date_idx is None:
        raise ValueError(f"Could not find date in line_id: {line_id}")

    source = "_".join(parts[:date_idx])
    date_str = parts[date_idx]
    issue_number = int(parts[date_idx + 1])
    daily_issue_number = int(parts[date_idx + 2])
    page_number = int(parts[date_idx + 3])

    # Find line ID marker (TL = TextLine)
    # The rest between page and line ID is the block ID
    tl_idx = None
    for i in range(date_idx + 4, len(parts)):
        if parts[i] == "TL":
            tl_idx = i
            break

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

    page_id = f"{source}_{date_str}_{issue_number:03d}_{daily_issue_number}_{page_number:03d}"
    text_block_id = f"{page_id}_{block_id}"

    return {
        "source": source,
        "date": date_str,
        "issue_number": issue_number,
        "daily_issue_number": daily_issue_number,
        "page_number": page_number,
        "block_id": block_id,
        "line_id": line_id_str,
        "full_page_id": page_id,
        "full_text_block_id": text_block_id,
    }


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
    return f"{components.source}_{components.date}_{components.issue_number:03d}_{components.daily_issue_number}"


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

    # Find the date part (format: YYYY-MM-DD)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            date_idx = i
            break

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

    # Find date marker (YYYY-MM-DD format)
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part.count("-") == 2:
            try:
                # Verify it looks like a date
                if part[4] == "-" and part[7] == "-":
                    int(part[:4])  # Year
                    int(part[5:7])  # Month
                    int(part[8:10])  # Day
                    date_idx = i
                    break
            except ValueError:
                continue

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
        else:
            return "text_block_id"

    return "unknown"


def extract_foreign_keys(id_string: str) -> Dict[str, Optional[str]]:
    """
    Extract all foreign key IDs from any ID type.

    Given any ID in the system, extract all parent IDs (foreign keys)
    that link this entity to higher levels in the hierarchy.

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
    id_type = identify_id_type(id_string)

    result: Dict[str, Optional[str]] = {
        "source_id": None,
        "issue_id": None,
        "page_id": None,
        "text_block_id": None,
        "line_id": None,
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
            result["source_id"] = str(components["source"])
            result["issue_id"] = (
                f"{components['source']}_{components['date']}_"
                f"{components['issue_number']:03d}_{components['daily_issue_number']}"
            )
            result["page_id"] = str(components["full_page_id"])
            result["text_block_id"] = str(components["full_text_block_id"])
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

    # Find the date part
    date_idx = None
    for i, part in enumerate(parts):
        if len(part) == 10 and part.count("-") == 2:
            date_idx = i
            break

    if date_idx is None:
        raise ValueError(f"Could not find date in ID: {id_string}")

    # Page ID is: source + date + issue + daily + page
    return "_".join(parts[: date_idx + 4])
