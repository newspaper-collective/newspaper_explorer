"""
ALTO XML parser for newspaper fulltext.
Extracts text lines with coordinates and metadata from ALTO format.
Integrates with METS metadata for rich issue-level information.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

from lxml import etree

from newspaper_explorer.data.utils.ids import (
    generate_issue_id,
    generate_line_id,
    generate_page_id,
    generate_source_id,
    generate_text_block_id,
    parse_alto_filename,
)
from newspaper_explorer.models.data.content import TextLine

logger = logging.getLogger(__name__)
# Type alias to avoid using private _Element
ElementType = Any  # lxml.etree._Element


class ALTOParser:
    """
    Fast ALTO XML parser with automatic namespace detection.
    Returns list of TextLine objects with enriched metadata.
    """

    def __init__(self) -> None:
        self.namespace_cache: dict[str, dict[str, str] | None] = {}

    def _detect_namespace(self, root: ElementType) -> dict[str, str] | None:
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
        datetime | None,  # date
        int | None,  # issue_number
        int | None,  # edition
        int | None,  # page_number
    ]:
        """
        Parse all metadata from ALTO filename in a single pass.

        Uses the unified parse_alto_filename() function from ids.py.

        Format: 3074409X_1902-09-05_000_415_H_2_005.xml

        Returns:
            (date, issue_number, edition, page_number)
        """
        parsed = parse_alto_filename(filename)
        if parsed is None:
            return None, None, None, None

        return (
            parsed.date,
            parsed.issue_number,
            parsed.edition,
            parsed.page_number,
        )

    def parse_file(
        self,
        filepath: Path,
        source_name: str,
        mets_metadata: dict[str, str | int | None] | None = None,
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
                edition,
                page_number,
            ) = self._parse_filename(filename)

            # Extract and normalize METS metadata types
            year_volume: str | None = None
            page_count: int | None = None
            newspaper_title: str | None = None

            if mets_metadata:
                if val := mets_metadata.get("year_volume"):
                    year_volume = str(val)
                if val := mets_metadata.get("page_count"):
                    page_count = int(val)
                if val := mets_metadata.get("newspaper_title"):
                    newspaper_title = str(val)

            lines: list[TextLine] = []

            # Generate source_id
            source_id_str = generate_source_id(source_name)

            # Generate IDs using unified ID system
            # Skip if we don't have the required components
            if not (date and issue_number and edition and page_number):
                logger.warning(
                    f"Skipping {filename}: Missing required ID components "
                    f"(date={date}, issue={issue_number}, daily={edition}, page={page_number})"
                )
                return []

            # Generate hierarchical IDs
            issue_id_str = generate_issue_id(source_name, date, issue_number, edition)
            page_id_str = generate_page_id(source_name, date, issue_number, edition, page_number)

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

                    # Extract text from String and HYP elements
                    # Use CONTENT (raw/fragmented) as primary, SUBS_CONTENT for dehyphenation
                    words: list[str] = []
                    dehyphenated_words: list[str] = []

                    # Process all child elements in order (String and HYP)
                    for child_elem in text_line_elem:
                        # Skip namespace prefix if present
                        tag = (
                            child_elem.tag.split("}")[-1]
                            if "}" in child_elem.tag
                            else child_elem.tag
                        )

                        if tag == "String":
                            content = child_elem.get("CONTENT", "")
                            subs_content = child_elem.get("SUBS_CONTENT", "")
                            subs_type = child_elem.get("SUBS_TYPE", "")

                            # Primary text uses CONTENT (as printed)
                            if content:
                                words.append(content)

                            # Handle dehyphenated version based on SUBS_TYPE
                            if subs_content:
                                # HypPart1: first part of hyphenated word
                                # HypPart2: second part of hyphenated word
                                # Only add SUBS_CONTENT to dehyphenated for HypPart1
                                if subs_type == "HypPart1":
                                    dehyphenated_words.append(subs_content)
                                # For HypPart2, skip adding (word already in previous line)
                            elif content:
                                dehyphenated_words.append(content)

                        elif tag == "HYP":
                            # Add hyphen character to primary text (as printed)
                            hyp_content = child_elem.get("CONTENT", "-")
                            if words:  # Append to last word
                                words[-1] = words[-1] + hyp_content

                    if not words:
                        continue

                    # Normalize whitespace inline
                    text = re.sub(r"\s+", " ", " ".join(words)).strip()
                    if not text:
                        continue

                    # Build dehyphenated version if we collected any SUBS_CONTENT
                    # (dehyphenated_words will differ from words when SUBS_CONTENT was present)
                    text_dehyphenated_ocr = None
                    if dehyphenated_words != words:  # Only set if different
                        text_dehyphenated_ocr = re.sub(
                            r"\s+", " ", " ".join(dehyphenated_words)
                        ).strip()

                    # Generate unique line_id using unified system
                    unique_line_id = generate_line_id(unique_text_block_id, alto_line_id)

                    # Helper function to safely convert coordinates (handles scientific notation)
                    def safe_int(value: str | None) -> int | None:
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
                            text_dehyphenated_ocr=text_dehyphenated_ocr,
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
                            edition=edition,
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

    def create_empty_page_line(
        self,
        filepath: Path,
        source_name: str,
        mets_metadata: dict[str, str | int | None] | None = None,
    ) -> TextLine | None:
        """
        Create a sentinel TextLine for empty pages (ALTO files with no text content).

        Empty pages are represented with a single row using sentinel values:
        - text: "" (empty string)
        - x, y, width, height: 0
        - line_id: "{page_id}_EMPTY_LINE"
        - text_block_id: "{page_id}_EMPTY_BLOCK"
        - is_empty: True
        - All metadata preserved (date, page_number, etc.)

        Args:
            filepath: Path to ALTO XML file (used for filename parsing)
            source_name: Source identifier (e.g., "der_tag")
            mets_metadata: Optional METS metadata dict to enrich the line

        Returns:
            TextLine object representing empty page, or None if filename can't be parsed
        """
        # Parse filename to get page metadata
        filename = filepath.name
        date, issue_number, edition, page_number = self._parse_filename(filename)

        if not (date and issue_number and edition and page_number):
            logger.warning(
                f"Cannot create empty page line for {filename}: Missing required ID components"
            )
            return None

        # Extract and normalize METS metadata types
        year_volume: str | None = None
        page_count: int | None = None
        newspaper_title: str | None = None

        if mets_metadata:
            if val := mets_metadata.get("year_volume"):
                year_volume = str(val)
            if val := mets_metadata.get("page_count"):
                page_count = int(val)
            if val := mets_metadata.get("newspaper_title"):
                newspaper_title = str(val)

        # Generate hierarchical IDs
        source_id_str = generate_source_id(source_name)
        issue_id_str = generate_issue_id(source_name, date, issue_number, edition)
        page_id_str = generate_page_id(source_name, date, issue_number, edition, page_number)

        # Create sentinel IDs for empty page
        text_block_id = f"{page_id_str}_EMPTY_BLOCK"
        line_id = f"{page_id_str}_EMPTY_LINE"

        return TextLine(
            # Primary key
            line_id=line_id,
            # Empty text data
            text="",
            text_dehyphenated_ocr=None,
            # Foreign keys
            source_id=source_id_str,
            issue_id=issue_id_str,
            page_id=page_id_str,
            text_block_id=text_block_id,
            # Original reference
            filename=filename,
            # Date & coordinates (zeros for empty)
            date=date,
            x=0,
            y=0,
            width=0,
            height=0,
            # Denormalized metadata
            issue_number=issue_number,
            edition=edition,
            page_number=page_number,
            year_volume=year_volume,
            page_count=page_count,
            newspaper_title=newspaper_title,
            # Empty page flag
            is_empty=True,
        )
