"""
Error correction utilities for newspaper data.
Handles known data issues in downloaded newspaper collections.
"""

import re
import shutil
from pathlib import Path


class DataFixer:
    """Apply error corrections to newspaper data."""

    def __init__(self, dataset_name: str, data_type: str):
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

        Returns:
            Number of fixes applied
        """
        print(f"Checking for known errors in {part_name}...")
        fixes_applied = 0

        # Fix for dertag_1900-1902: Files labeled as 1900-01-02 are actually 1902-01-01/02
        if part_name == "dertag_1900-1902" and self.dataset_name == "der_tag":
            fixes_applied += self._fix_dertag_1900_mislabeled_files(extract_path)

        if fixes_applied > 0:
            print(f"Applied {fixes_applied} error fix(es)")
        else:
            print("No known errors to fix")

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

        print("  Checking for mislabeled 1900 files...")

        # Hardcoded list of known mislabeled dates
        # Format: {old_year: {(month, day, issue): correct_year}}
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
                print(f"    Found mislabeled: {rel_path} -> {correct_year}")
                fixes_applied += self._relocate_and_fix_issue(
                    issue_dir, raw_dir, old_year, correct_year
                )
            else:
                print(
                    f"    Note: Expected mislabeled path not found: "
                    f"{old_year}/{month}/{day}/{issue_num}"
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
            if len(parts) < 3:
                return 0

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

                except Exception as e:
                    print(f"      Warning: Could not fix {file_path.name}: {e}")
                    continue

            # Remove old directory if all files were moved successfully
            if files_fixed > 0:
                try:
                    shutil.rmtree(issue_dir)
                    old_path = f"{old_year}/{month}/{day}/{issue_num}"
                    new_path = f"{new_year}/{month}/{day}/{issue_num}"
                    print(f"    Relocated {files_fixed} files: {old_path} -> {new_path}")

                    # Clean up empty parent directories
                    self._cleanup_empty_dirs(raw_dir / old_year)
                except Exception as e:
                    print(f"      Warning: Could not remove old directory: {e}")

            return 1 if files_fixed > 0 else 0

        except Exception as e:
            print(f"    Error relocating issue: {e}")
            return 0

    def _cleanup_empty_dirs(self, start_dir: Path):
        """
        Recursively remove empty directories starting from start_dir.

        Args:
            start_dir: Directory to start cleanup from
        """
        import os

        if not start_dir.exists() or not start_dir.is_dir():
            return

        try:
            # Walk bottom-up and remove empty directories
            for root, dirs, files in os.walk(str(start_dir), topdown=False):
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
        except Exception:
            # Silently ignore cleanup errors
            pass
