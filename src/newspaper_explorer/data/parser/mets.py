"""
METS XML metadata parser for newspaper issues.
Extracts rich metadata from METS files that describe complete issues.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class IssueMetadata:
    """Metadata for a complete newspaper issue from METS file"""

    filename: str
    date: Optional[datetime] = None
    issue_number: Optional[int] = None
    issue_string: Optional[str] = None  # e.g., "Nr. 415, 05. September 1902"
    edition: Optional[str] = None  # e.g., "Ausgabe A"
    year_volume: Optional[str] = None  # e.g., "Jahrgang 1902"
    page_count: Optional[int] = None
    newspaper_title: Optional[str] = None
    newspaper_subtitle: Optional[str] = None
    newspaper_id: Optional[str] = None  # ZDB ID
    publisher: Optional[str] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "filename": self.filename,
            "date": self.date,
            "issue_number": self.issue_number,
            "issue_string": self.issue_string,
            "edition": self.edition,
            "year_volume": self.year_volume,
            "page_count": self.page_count,
            "newspaper_title": self.newspaper_title,
            "newspaper_id": self.newspaper_id,
            "language": self.language,
        }


class METSParser:
    """Parser for METS XML metadata files"""

    NAMESPACES = {
        "mets": "http://www.loc.gov/METS/",
        "mods": "http://www.loc.gov/mods/v3",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    # Pre-compiled regex patterns
    _ISSUE_NUMBER_RE = re.compile(r"Nr\.\s*(\d+)")
    _PAGE_COUNT_RE = re.compile(r"(\d+)\s*Seiten?")

    def _get_text(self, root, xpath: str) -> Optional[str]:
        """Helper to extract and strip text from an element"""
        elem = root.find(xpath, self.NAMESPACES)
        return elem.text.strip() if elem is not None and elem.text else None

    def parse_file(self, filepath: Path) -> Optional[IssueMetadata]:
        """
        Parse a METS XML file and extract issue metadata.

        Args:
            filepath: Path to METS XML file

        Returns:
            IssueMetadata object or None if parsing fails
        """
        try:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

            filename = filepath.name
            metadata = IssueMetadata(filename=filename)

            # Extract date
            date_text = self._get_text(root, ".//mods:dateIssued[@encoding='iso8601']")
            if date_text:
                try:
                    metadata.date = datetime.fromisoformat(date_text)
                except ValueError:
                    pass

            # Extract issue number and string
            metadata.issue_string = self._get_text(
                root, ".//mods:detail[@type='issue']/mods:number"
            )
            if metadata.issue_string:
                match = self._ISSUE_NUMBER_RE.search(metadata.issue_string)
                if match:
                    metadata.issue_number = int(match.group(1))

            # Extract edition (Ausgabe)
            metadata.edition = self._get_text(root, ".//mods:partNumber")

            # Extract year/volume
            metadata.year_volume = self._get_text(
                root, ".//mods:detail[@type='volume']/mods:number"
            )

            # Extract page count
            extent_text = self._get_text(root, ".//mods:physicalDescription/mods:extent")
            if extent_text:
                match = self._PAGE_COUNT_RE.search(extent_text)
                if match:
                    metadata.page_count = int(match.group(1))

            # Extract newspaper title and subtitle
            metadata.newspaper_title = self._get_text(
                root, ".//mods:relatedItem[@type='host']//mods:title"
            )


            # Extract ZDB ID
            metadata.newspaper_id = self._get_text(
                root, ".//mods:relatedItem[@type='host']/mods:identifier[@type='zdb']"
            )



            # Extract language
            metadata.language = self._get_text(root, ".//mods:languageTerm[@type='code']")

            logger.debug(f"Parsed METS metadata from {filename}")
            return metadata

        except Exception as e:
            logger.error(f"Error parsing METS file {filepath}: {e}")
            return None

    def find_mets_for_alto(self, alto_path: Path) -> Optional[Path]:
        """
        Find the corresponding METS file for an ALTO file.

        ALTO: .../1902/09/05/02/fulltext/3074409X_1902-09-05_000_415_H_2_001.xml
        METS: .../1902/09/05/02/3074409X_1902-09-05_000_415_H_2.xml

        Args:
            alto_path: Path to ALTO XML file

        Returns:
            Path to METS file or None if not found
        """
        try:
            # ALTO files are in fulltext/ subdirectory
            if alto_path.parent.name == "fulltext":
                issue_dir = alto_path.parent.parent

                # Extract issue identifier from ALTO filename
                # 3074409X_1902-09-05_000_415_H_2_001.xml -> 3074409X_1902-09-05_000_415_H_2
                alto_filename = alto_path.stem
                match = re.match(r"(.+?)_\d{3}$", alto_filename)
                if match:
                    issue_id = match.group(1)
                    mets_file = issue_dir / f"{issue_id}.xml"

                    if mets_file.exists():
                        return mets_file
        except Exception as e:
            logger.debug(f"Could not find METS for {alto_path}: {e}")

        return None
