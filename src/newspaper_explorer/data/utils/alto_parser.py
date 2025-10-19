"""
ALTO XML parser for newspaper fulltext.
Extracts text lines with coordinates and metadata from ALTO format.
Integrates with METS metadata for rich issue-level information.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree
from lxml.etree import _Element

logger = logging.getLogger(__name__)


@dataclass
class TextLine:
    """Represents a single text line from ALTO XML with enriched metadata"""

    # Core identifiers
    line_id: str
    text: str
    text_block_id: str
    filename: str

    # Date information
    date: Optional[datetime] = None

    # Layout coordinates
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # From filename parsing
    newspaper_id: Optional[str] = None  # e.g., "3074409X"
    issue_number: Optional[int] = None  # from filename (may differ from METS)
    daily_issue_number: Optional[int] = None  # Number after H (2nd issue that day)
    page_number: Optional[int] = None  # Last number in filename (e.g., 005)

    # From METS metadata
    year_volume: Optional[str] = None  # e.g., "Jahrgang 1902"
    page_count: Optional[int] = None  # Total pages in issue
    newspaper_title: Optional[str] = None  # e.g., "Der Tag"
    newspaper_subtitle: Optional[str] = None

    @property
    def year(self) -> Optional[int]:
        """Extract year from date"""
        return self.date.year if self.date else None

    @property
    def month(self) -> Optional[int]:
        """Extract month from date"""
        return self.date.month if self.date else None

    @property
    def day(self) -> Optional[int]:
        """Extract day from date"""
        return self.date.day if self.date else None

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction"""
        return {
            "line_id": self.line_id,
            "text": self.text,
            "text_block_id": self.text_block_id,
            "filename": self.filename,
            "date": self.date,
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "newspaper_id": self.newspaper_id,
            "issue_number": self.issue_number,
            "daily_issue_number": self.daily_issue_number,
            "page_number": self.page_number,
            "year_volume": self.year_volume,
            "page_count": self.page_count,
            "newspaper_title": self.newspaper_title,
            "newspaper_subtitle": self.newspaper_subtitle,
        }


class ALTOParser:
    """
    Fast ALTO XML parser with automatic namespace detection.
    Returns list of TextLine objects with enriched metadata.
    """

    def __init__(self) -> None:
        self.namespace_cache: Dict[str, Optional[Dict[str, str]]] = {}

    def _detect_namespace(self, root: _Element) -> Optional[Dict[str, str]]:
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

    def _parse_filename(self, filename: str) -> Tuple[
        Optional[str],  # newspaper_id
        Optional[datetime],  # date
        Optional[int],  # issue_number
        Optional[int],  # daily_issue_number
        Optional[int],  # page_number
    ]:
        """
        Parse all metadata from ALTO filename in a single pass.

        Format: 3074409X_1902-09-05_000_415_H_2_005.xml
        Components:
        - 3074409X: newspaper ID
        - 1902-09-05: date (YYYY-MM-DD)
        - 000: unknown field (always 000)
        - 415: issue number (may differ from METS)
        - H: separator meaning "Heft" (issue)
        - 2: daily issue number (1st, 2nd, 3rd issue that day)
        - 005: page number

        Returns:
            (newspaper_id, date, issue_number, daily_issue_number, page_number)
        """
        # Single regex to capture all components
        pattern = r"^([A-Z0-9]+)_(\d{4})-(\d{2})-(\d{2})_\d{3}_(\d+)_H_(\d+)_(\d+)"
        match = re.match(pattern, filename)

        if not match:
            return None, None, None, None, None

        newspaper_id = match.group(1)
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

        return newspaper_id, date, issue_number, daily_issue_number, page_number

    def parse_file(
        self,
        filepath: Path,
        mets_metadata: Optional[Dict] = None,
    ) -> List[TextLine]:
        """
        Parse a single ALTO XML file and extract all text lines.

        Args:
            filepath: Path to ALTO XML file
            mets_metadata: Optional METS metadata dict to enrich lines

        Returns:
            List of TextLine objects
        """
        try:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

            # Detect namespace
            ns = self._detect_namespace(root)

            # Parse all filename metadata in one pass
            filename = filepath.name
            (
                newspaper_id,
                date,
                issue_number,
                daily_issue_number,
                page_number,
            ) = self._parse_filename(filename)

            # Extract METS metadata if provided
            year_volume = mets_metadata.get("year_volume") if mets_metadata else None
            page_count = mets_metadata.get("page_count") if mets_metadata else None
            newspaper_title = mets_metadata.get("newspaper_title") if mets_metadata else None
            newspaper_subtitle = mets_metadata.get("newspaper_subtitle") if mets_metadata else None

            lines = []

            # Find all TextBlocks
            for text_block in root.findall(".//alto:TextBlock", ns):
                block_id = text_block.get("ID", "")
                if not block_id:
                    continue

                # Parse each TextLine
                for text_line_elem in text_block.findall(".//alto:TextLine", ns):
                    line_id = text_line_elem.get("ID", "")
                    if not line_id:
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

                    # Create unique line_id
                    unique_line_id = f"{filename}_{block_id}_{line_id}"

                    lines.append(
                        TextLine(
                            line_id=unique_line_id,
                            text=text,
                            text_block_id=block_id,
                            filename=filename,
                            date=date,
                            x=int(x) if x else None,
                            y=int(y) if y else None,
                            width=int(width) if width else None,
                            height=int(height) if height else None,
                            newspaper_id=newspaper_id,
                            issue_number=issue_number,
                            daily_issue_number=daily_issue_number,
                            page_number=page_number,
                            year_volume=year_volume,
                            page_count=page_count,
                            newspaper_title=newspaper_title,
                            newspaper_subtitle=newspaper_subtitle,
                        )
                    )

            return lines

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return []
