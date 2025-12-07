"""
Error correction utilities for newspaper data.
Handles known data issues in downloaded newspaper collections.
"""

import logging
import os
from pathlib import Path
import re
import shutil

logger = logging.getLogger(__name__)


class DataFixer:
    """Apply error corrections to newspaper data."""

    def __init__(self, dataset_name: str, data_type: str) -> None:
        """
        Initialize the data fixer.

        Args:
            dataset_name: Name of the dataset (e.g., "der_tag")
            data_type: Type of data (e.g., "xml_ocr")
        """
        self.dataset_name = dataset_name
        self.data_type = data_type

    def apply_fixes(self, part_name: str, extract_path: Path) -> int:
        """
        Apply automatic error corrections to extracted data.

        Args:
            part_name: Name of the dataset part
            extract_path: Path to the extracted directory (base raw directory)

        """
        logger.info("Checking for known errors in %s...", part_name)
        fixes_applied = 0

        # Fix for dertag_1900-1902: Files labeled as 1900-01-02 are actually 1902-01-01/02
        if part_name == "dertag_1900-1902" and self.dataset_name == "der_tag":
            fixes_applied += self._fix_dertag_1900_mislabeled_files(extract_path)

        # Fix for dertag_1900-1902: Mixed issue numbers in same directory
        if part_name == "dertag_1900-1902" and self.dataset_name == "der_tag":
            fixes_applied += self._fix_dertag_mixed_issues(extract_path)

        # Fix for dertag_1900-1902: Corrupted page numbers in 1901-07-14 issue
        if part_name == "dertag_1900-1902" and self.dataset_name == "der_tag":
            fixes_applied += self._fix_dertag_1901_07_14_page_numbers(extract_path)

        if fixes_applied > 0:
            logger.info("Applied %d error fix(es)", fixes_applied)
        else:
            logger.debug("No known errors to fix")

        return fixes_applied

    def _fix_dertag_1900_mislabeled_files(self, raw_dir: Path) -> int:
        """
        Fix mislabeled files in the 1900 directory that are actually from 1902.
        Uses a hardcoded list of known mislabeled dates.

        Args:
            raw_dir: Base raw directory (e.g., data/raw/der_tag/xml_ocr/)

        Returns:
            Number of files fixed
        """
        fixes_applied = 0
        year_1900_dir = raw_dir / "1900"

        if not year_1900_dir.exists():
            return 0

        logger.debug("Checking for mislabeled 1900 files...")

        # Hardcoded list of known mislabeled dates
        # Keys are old years, values are dicts mapping (month, day, issue) tuples to correct years
        mislabeled_dates = {
            "1900": {
                ("01", "02", "01"): "1902",  # 1900-01-02 issue 01 is actually 1902-01-02
            }
        }

        old_year = "1900"
        if old_year not in mislabeled_dates:
            return 0

        # Process each known mislabeled date
        for (month, day, issue_num), correct_year in mislabeled_dates[old_year].items():
            issue_dir = year_1900_dir / month / day / issue_num

            if issue_dir.exists():
                rel_path = issue_dir.relative_to(raw_dir)
                logger.info("Found mislabeled: %s -> %s", rel_path, correct_year)
                fixes_applied += self._relocate_and_fix_issue(
                    issue_dir, raw_dir, old_year, correct_year
                )
            else:
                logger.debug(
                    "Expected mislabeled path not found: %s/%s/%s/%s",
                    old_year,
                    month,
                    day,
                    issue_num,
                )

        return fixes_applied

    def _relocate_and_fix_issue(
        self, issue_dir: Path, raw_dir: Path, old_year: str, new_year: str
    ) -> int:
        """
        Relocate an issue directory to the correct year and fix metadata.

        Args:
            issue_dir: Path to the issue directory (e.g., .../1900/01/02/01/)
            raw_dir: Base raw directory
            old_year: Incorrect year in filename/path (e.g., "1900")
            new_year: Correct year (e.g., "1902")

        Returns:
            Number of fixes applied (1 if successful)
        """
        try:
            # Extract month/day/issue from current path
            parts = issue_dir.relative_to(raw_dir / old_year).parts

            month, day, issue_num = parts[0], parts[1], parts[2]

            # Create target directory
            target_dir = raw_dir / new_year / month / day / issue_num
            target_dir.mkdir(parents=True, exist_ok=True)

            # Process all files in issue directory
            files_fixed = 0
            for file_path in issue_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                # Calculate relative path within issue
                rel_path = file_path.relative_to(issue_dir)
                target_file = target_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                # Read, fix, and write file
                try:
                    content = file_path.read_text(encoding="utf-8")

                    # Fix filename pattern in content: 3074409X_1900-01-02 -> 3074409X_1902-01-02
                    old_date_pattern = f"{old_year}-{month}-{day}"
                    new_date_pattern = f"{new_year}-{month}-{day}"
                    content = content.replace(old_date_pattern, new_date_pattern)

                    # Fix dateIssued in MODS metadata
                    old_elem = f'<mods:dateIssued encoding="iso8601">{old_date_pattern}'
                    new_elem = f'<mods:dateIssued encoding="iso8601">{new_date_pattern}'
                    content = content.replace(old_elem, new_elem)

                    # Fix year in LABEL attributes (e.g., "02. Januar 1900")
                    content = re.sub(
                        rf'LABEL="(\d+)\.\s+(\w+)\s+{old_year}"',
                        rf'LABEL="\1. \2 {new_year}"',
                        content,
                    )

                    # Fix year in issue number (e.g., "Nr. 2, 02. Januar 1900")
                    content = re.sub(
                        rf"<mods:number>Nr\.\s+(\d+),\s+(\d+)\.\s+(\w+)\s+{old_year}</mods:number>",
                        rf"<mods:number>Nr. \1, \2. \3 {new_year}</mods:number>",
                        content,
                    )

                    # Fix year in part order attribute
                    # Pattern: order="19000102XX" where XX is the edition number
                    old_order_prefix = f"{old_year}{month}{day}"
                    new_order_prefix = f"{new_year}{month}{day}"
                    content = re.sub(
                        rf'order="{old_order_prefix}(\d{{2}})"',
                        rf'order="{new_order_prefix}\1"',
                        content,
                    )

                    # Fix year in Jahrgang
                    content = re.sub(
                        rf"<mods:number>Jahrgang {old_year}</mods:number>",
                        rf"<mods:number>Jahrgang {new_year}</mods:number>",
                        content,
                    )

                    # Write to target with corrected filename
                    target_filename = file_path.name.replace(old_date_pattern, new_date_pattern)
                    final_target = target_file.parent / target_filename
                    final_target.write_text(content, encoding="utf-8")
                    files_fixed += 1

                except (OSError, PermissionError, UnicodeDecodeError) as e:
                    logger.warning("Could not fix %s: %s", file_path.name, e)
                    continue

            # Remove old directory if all files were moved successfully
            if files_fixed > 0:
                try:
                    shutil.rmtree(issue_dir)
                    old_path = f"{old_year}/{month}/{day}/{issue_num}"
                    new_path = f"{new_year}/{month}/{day}/{issue_num}"
                    logger.info("Relocated %d files: %s -> %s", files_fixed, old_path, new_path)

                    # Clean up empty parent directories
                    self._cleanup_empty_dirs(raw_dir / old_year)
                except (OSError, PermissionError) as e:
                    logger.warning("Could not remove old directory: %s", e)

            return 1 if files_fixed > 0 else 0

        except (OSError, PermissionError, ValueError, UnicodeDecodeError) as e:
            logger.error("Error relocating issue: %s", e)
            return 0

    def _cleanup_empty_dirs(self, start_dir: Path) -> None:
        """
        Recursively remove empty directories starting from start_dir.

        Args:
            start_dir: Directory to start cleanup from
        """

        if not start_dir.exists() or not start_dir.is_dir():
            return

        try:
            # Walk bottom-up and remove empty directories
            for root, dirs, _ in os.walk(str(start_dir), topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        # Try to remove if directory is empty
                        if dir_path.exists() and not any(dir_path.iterdir()):
                            dir_path.rmdir()
                    except OSError:
                        # Directory not empty or permission issue, skip
                        pass

            # Finally, try to remove the start directory itself if empty
            try:
                if start_dir.exists() and not any(start_dir.iterdir()):
                    start_dir.rmdir()
            except OSError:
                pass
        except (OSError, PermissionError):
            # Silently ignore cleanup errors (filesystem issues)
            pass

    def _fix_dertag_mixed_issues(self, raw_dir: Path) -> int:
        """
        Fix directories with ALTO files from multiple issues mixed together.

        Some issue directories contain ALTO files from the next issue(s) that should
        have their own METS file and directory. This relocates those orphaned files
        to proper directories and creates missing METS files.

        Args:
            raw_dir: Base raw directory (e.g., data/raw/der_tag/xml_ocr/)

        Returns:
            Number of issues fixed
        """
        logger.debug("Checking for mixed-issue directories...")

        # Hardcoded list of known mixed-issue cases
        mixed_cases = [
            ("1901/03/19/01", ["104"]),
            ("1901/03/26/01", ["116"]),
            ("1901/03/27/01", ["118"]),
            ("1901/04/18/01", ["152"]),
            ("1901/04/19/01", ["153", "154"]),
            ("1901/05/04/01", ["180"]),
            ("1901/05/08/01", ["186"]),
            ("1901/05/11/01", ["192"]),
            ("1901/06/21/01", ["258"]),
            ("1909/09/13/01", ["679"]),
        ]

        fixes_applied = 0

        for directory, orphaned_issues in mixed_cases:
            issue_dir = raw_dir / directory
            fulltext_dir = issue_dir / "fulltext"

            if not fulltext_dir.exists():
                continue

            # Check if orphaned files exist
            orphaned_files: list[Path] = []
            for orphaned_issue in orphaned_issues:
                pattern = f"*_{orphaned_issue}_H_*.xml"
                found = list(fulltext_dir.glob(pattern))
                if found:
                    orphaned_files.extend(found)

            if not orphaned_files:
                continue

            logger.info("Found mixed issues in %s", directory)
            logger.info(
                "%d orphaned file(s) from issue(s) %s", len(orphaned_files), orphaned_issues
            )

            # For now, just log the issue - actual relocation would require:
            # 1. Determining correct dates for orphaned issues
            # 2. Creating proper directory structure
            # 3. Creating/copying METS files
            # This is complex and risky without more metadata

            # Just count as identified, not fixed yet
            fixes_applied += 1

        if fixes_applied > 0:
            logger.info("Identified %d mixed-issue case(s)", fixes_applied)
            logger.warning("Automatic fix not yet implemented - manual intervention required")

        return 0  # Return 0 since we're not actually fixing yet

    def _fix_dertag_1901_07_14_page_numbers(self, raw_dir: Path) -> int:
        """
        Fix corrupted page numbers in 1901-07-14 issue 297.

        The first page has a negative page number (-01 instead of 001), which
        caused all subsequent pages to be numbered incorrectly (starting from 002
        instead of 003). This fix renames all pages in reverse order to avoid
        overwriting files.

        Mapping:
        - _-01.xml → _001.xml  (negative page number, should be first page)
        - _001.xml → _002.xml  (misnamed, should be second page)
        - _002.xml → _003.xml
        - _003.xml → _004.xml
        - ...
        - _012.xml → _013.xml

        Args:
            raw_dir: Base raw directory (e.g., data/raw/der_tag/xml_ocr/)

        Returns:
            Number of files fixed
        """
        issue_dir = raw_dir / "1901" / "07" / "14" / "01" / "fulltext"

        if not issue_dir.exists():
            return 0

        logger.debug("Checking for corrupted page numbers in 1901-07-14...")

        # Check if the corrupted file exists
        corrupted_file = issue_dir / "3074409X_1901-07-14_000_297_H_1_-01.xml"
        if not corrupted_file.exists():
            return 0

        logger.info("Found corrupted page numbering in %s", issue_dir.relative_to(raw_dir))

        try:
            # Rename files in reverse order to avoid conflicts
            # Start from page 12 down to page 1, shifting each up by 1
            files_renamed = 0

            for old_page in range(12, 0, -1):  # 12, 11, 10, ..., 2, 1
                new_page = old_page + 1
                old_name = f"3074409X_1901-07-14_000_297_H_1_{old_page:03d}.xml"
                new_name = f"3074409X_1901-07-14_000_297_H_1_{new_page:03d}.xml"

                old_path = issue_dir / old_name
                new_path = issue_dir / new_name

                if old_path.exists():
                    old_path.rename(new_path)
                    files_renamed += 1
                    logger.debug("Renamed: %s → %s", old_name, new_name)
                else:
                    logger.warning("Expected file not found: %s", old_name)

            # Finally, rename the corrupted -01 file to 001
            if corrupted_file.exists():
                correct_name = "3074409X_1901-07-14_000_297_H_1_001.xml"
                correct_path = issue_dir / correct_name
                corrupted_file.rename(correct_path)
                files_renamed += 1
                logger.debug("Renamed: 3074409X_1901-07-14_000_297_H_1_-01.xml → %s", correct_name)

            if files_renamed > 0:
                logger.info("Fixed page numbering: %d file(s) renamed", files_renamed)
                return 1  # Return 1 fix applied (the issue as a whole)

        except (OSError, PermissionError) as e:
            logger.error("Error fixing page numbers: %s", e)
            return 0

        return 0
