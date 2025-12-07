"""
METS XML metadata parser for newspaper issues.
Extracts rich metadata from METS files that describe complete issues.
"""

from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Optional

from lxml import etree

from newspaper_explorer.data.utils.ids import extract_edition
from newspaper_explorer.models.data.content import IssueMetadata

logger = logging.getLogger(__name__)


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

            # Extract edition from filename (not from <mods:partNumber> which is always "Ausgabe A")
            # Edition is the numeric value (1=morning, 2=midday, 3=evening) from filename pattern _H_N_
            metadata.edition = extract_edition(filename=filename)

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
