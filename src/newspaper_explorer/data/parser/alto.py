"""
ALTO XML parser for newspaper fulltext.
Extracts text lines with coordinates and metadata from ALTO format.
Integrates with METS metadata for rich issue-level information.
"""

from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Optional, Union

from lxml import etree
from lxml.etree import _Element

from newspaper_explorer.data.utils.ids import (
    generate_issue_id,
    generate_line_id,
    generate_page_id,
    generate_source_id,
    generate_text_block_id,
)
from newspaper_explorer.models.data.content import TextLine

logger = logging.getLogger(__name__)


class ALTOParser:
    """
    Fast ALTO XML parser with automatic namespace detection.
    Returns list of TextLine objects with enriched metadata.
    """

    def __init__(self) -> None:
        self.namespace_cache: dict[str, Optional[dict[str, str]]] = {}

    def _detect_namespace(self, root: _Element) -> Optional[dict[str, str]]:
        """
        Detect ALTO namespace from root element with caching.

        Since all ALTO files typically use the same namespace,
        caching avoids redundant string operations across thousands of files.
        """
        # Use root tag as cache key
        cache_key = root.tag

        # Check cache first
        if cache_key in self.namespace_cache:
            return self.namespace_cache[cache_key]

        # Detect namespace
        if root.tag.startswith("{"):
            ns = root.tag[1 : root.tag.rindex("}")]
            result = {"alto": ns}
        else:
            result = None

        # Cache the result
        self.namespace_cache[cache_key] = result
        return result

    def _parse_filename(
        self, filename: str
    ) -> tuple[
        Optional[datetime],  # date
        Optional[int],  # issue_number
        Optional[int],  # daily_issue_number
        Optional[int],  # page_number
    ]:
        """
        Parse all metadata from ALTO filename in a single pass.

        Format: 3074409X_1902-09-05_000_415_H_2_005.xml
        Components:
        - 3074409X: newspaper ID (ignored, we use source_id from config instead)
        - 1902-09-05: date (YYYY-MM-DD)
        - 000: unknown field (always 000)
        - 415: issue number (may differ from METS)
        - H: separator meaning "Heft" (issue)
        - 2: daily issue number (1st, 2nd, 3rd issue that day)
        - 005: page number

        Returns:
            (date, issue_number, daily_issue_number, page_number)
        """
        # Single regex to capture all components
        pattern = r"^([A-Z0-9]+)_(\d{4})-(\d{2})-(\d{2})_\d{3}_(\d+)_H_(\d+)_(\d+)"
        match = re.match(pattern, filename)

        if not match:
            return None, None, None, None

        # Skip newspaper_id (group 1) - we use source_id from config instead
        year = int(match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        issue_number = int(match.group(5))
        daily_issue_number = int(match.group(6))
        page_number = int(match.group(7))

        # Try to create datetime object
        try:
            date = datetime(year, month, day)
        except ValueError:
            date = None

        return date, issue_number, daily_issue_number, page_number

    def parse_file(
        self,
        filepath: Path,
        source_name: str,
        mets_metadata: Optional[dict[str, Union[str, int, None]]] = None,
    ) -> list[TextLine]:
        """
        Parse a single ALTO XML file and extract all text lines.

        Args:
            filepath: Path to ALTO XML file
            source_name: Source identifier (e.g., "der_tag")
            mets_metadata: Optional METS metadata dict to enrich lines

        Returns:
            List of TextLine objects with unified IDs
        """
        try:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

            # Detect namespace
            ns = self._detect_namespace(root)

            # Parse all filename metadata in one pass
            filename = filepath.name
            (
                date,
                issue_number,
                daily_issue_number,
                page_number,
            ) = self._parse_filename(filename)

            # Extract METS metadata if provided
            year_volume = mets_metadata.get("year_volume") if mets_metadata else None
            page_count = mets_metadata.get("page_count") if mets_metadata else None
            newspaper_title = mets_metadata.get("newspaper_title") if mets_metadata else None

            lines = []

            # Generate source_id
            source_id_str = generate_source_id(source_name)

            # Generate IDs using unified ID system
            # Skip if we don't have the required components
            if not (date and issue_number and daily_issue_number and page_number):
                logger.warning(
                    f"Skipping {filename}: Missing required ID components "
                    f"(date={date}, issue={issue_number}, daily={daily_issue_number}, page={page_number})"
                )
                return []

            # Generate hierarchical IDs
            issue_id_str = generate_issue_id(source_name, date, issue_number, daily_issue_number)
            page_id_str = generate_page_id(
                source_name, date, issue_number, daily_issue_number, page_number
            )

            # Find all TextBlocks
            for text_block in root.findall(".//alto:TextBlock", ns):
                block_id = text_block.get("ID", "")
                if not block_id:
                    continue

                # Generate globally unique text_block_id using unified system
                unique_text_block_id = generate_text_block_id(page_id_str, block_id)

                # Parse each TextLine
                for text_line_elem in text_block.findall(".//alto:TextLine", ns):
                    alto_line_id = text_line_elem.get("ID", "")
                    if not alto_line_id:
                        continue

                    # Get position
                    x = text_line_elem.get("HPOS")
                    y = text_line_elem.get("VPOS")
                    width = text_line_elem.get("WIDTH")
                    height = text_line_elem.get("HEIGHT")

                    # Extract text from String elements
                    words = []
                    for string_elem in text_line_elem.findall(".//alto:String", ns):
                        content = string_elem.get("CONTENT", "")
                        subs_content = string_elem.get("SUBS_CONTENT", "")
                        word = subs_content if subs_content else content
                        if word:
                            words.append(word)

                    if not words:
                        continue

                    # Normalize whitespace inline
                    text = re.sub(r"\s+", " ", " ".join(words)).strip()
                    if not text:
                        continue

                    # Generate unique line_id using unified system
                    unique_line_id = generate_line_id(unique_text_block_id, alto_line_id)

                    # Helper function to safely convert coordinates (handles scientific notation)
                    def safe_int(value: Optional[str]) -> Optional[int]:
                        if not value:
                            return None
                        try:
                            # Convert via float to handle scientific notation like '9.999999E6'
                            return int(float(value))
                        except (ValueError, TypeError):
                            return None

                    lines.append(
                        TextLine(
                            # Primary key
                            line_id=unique_line_id,
                            # Data
                            text=text,
                            # Foreign keys
                            source_id=source_id_str,
                            issue_id=issue_id_str,
                            page_id=page_id_str,
                            text_block_id=unique_text_block_id,
                            # Original reference
                            filename=filename,
                            # Date & coordinates
                            date=date,
                            x=safe_int(x),
                            y=safe_int(y),
                            width=safe_int(width),
                            height=safe_int(height),
                            # Denormalized Metadata
                            issue_number=issue_number,
                            daily_issue_number=daily_issue_number,
                            page_number=page_number,
                            year_volume=year_volume,
                            page_count=page_count,
                            newspaper_title=newspaper_title,
                        )
                    )

            return lines

        except etree.XMLSyntaxError as e:
            logger.error(f"XML syntax error parsing {filepath}: {e}")
            return []
        except OSError as e:
            logger.error(f"File I/O error reading {filepath}: {e}")
            return []
