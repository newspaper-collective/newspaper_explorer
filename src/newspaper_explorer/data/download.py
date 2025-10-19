"""
Data gathering utilities for newspaper explorer.
Handles downloading and extraction of Zenodo newspaper collections.
"""

import hashlib
import logging
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, cast

import requests
from tqdm import tqdm

from newspaper_explorer.data.fixes import DataFixer
from newspaper_explorer.utils.config import get_config
from newspaper_explorer.utils.sources import load_source_config

logger = logging.getLogger(__name__)


class ZenodoDownloader:
    """Download and extract newspaper data from Zenodo collections."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the Zenodo downloader.

        Args:
            data_dir: Directory to store downloaded and extracted data.
                     Defaults to config or workspace data/ directory.
        """
        config = get_config()

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = Path(data_dir)
        self.download_dir: Path = Path(config.download_dir)
        self.extracted_dir: Path = Path(config.extracted_dir)

        # Ensure directories exist
        config.ensure_directories()

        # Load Zenodo links configuration from sources directory
        # TODO: Make this configurable instead of hardcoded to 'der_tag'
        self.config: dict[str, Any] = load_source_config("der_tag")

        # Get dataset metadata
        self.dataset_name: str = str(self.config.get("dataset_name", "unknown"))
        self.data_type: str = str(self.config.get("data_type", "data"))

    def list_available_parts(self) -> List[dict]:
        """
        List all available dataset parts from the configuration.

        Returns:
            List of dictionaries containing part information.
        """
        return cast(List[dict[str, Any]], self.config["parts"])

    def _calculate_md5(self, filepath: Path) -> str:
        """
        Calculate MD5 checksum of a file.

        Args:
            filepath: Path to the file

        Returns:
            MD5 checksum as hex string
        """
        md5_hash = hashlib.md5()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _verify_checksum(self, filepath: Path, expected_md5: str) -> bool:
        """
        Verify file checksum matches expected value.

        Args:
            filepath: Path to the file to verify
            expected_md5: Expected MD5 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        logger.info("Verifying checksum...")
        actual_md5 = self._calculate_md5(filepath)

        if actual_md5 == expected_md5:
            logger.info(f"Checksum verified: {actual_md5}")
            return True
        else:
            logger.warning("Checksum mismatch!")
            logger.warning(f"  Expected: {expected_md5}")
            logger.warning(f"  Got:      {actual_md5}")
            return False

    def download_part(self, part_name: str, force_redownload: bool = False) -> Path:
        """
        Download a specific dataset part.

        Args:
            part_name: Name of the part to download (e.g., 'dertag_1900-1902')
            force_redownload: If True, redownload even if file exists

        Returns:
            Path to the downloaded file

        Raises:
            ValueError: If part_name is not found in configuration
        """
        # Find the part in configuration
        part_info = None
        for part in self.config["parts"]:
            if part["name"] == part_name:
                part_info = part
                break

        if part_info is None:
            available = [p["name"] for p in self.config["parts"]]
            raise ValueError(f"Part '{part_name}' not found. Available parts: {available}")

        url = part_info["url"]
        filename = f"{part_name}.tar.gz"

        # Create dataset-specific download directory
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        dataset_download_dir.mkdir(parents=True, exist_ok=True)
        filepath = dataset_download_dir / filename

        # Check if file already exists
        if filepath.exists() and not force_redownload:
            logger.info(f"File {filename} already exists")
            # Verify checksum if available
            if "md5" in part_info:
                if self._verify_checksum(filepath, part_info["md5"]):
                    logger.info("Skipping download - file verified")
                    return filepath
                else:
                    logger.warning("Checksum failed - will re-download")
            else:
                logger.info("Skipping download (no checksum available)")
                return filepath

        logger.info(f"Downloading {part_name}...")

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        logger.info(f"Downloaded {filename}")

        # Verify checksum if available
        if "md5" in part_info:
            if not self._verify_checksum(filepath, part_info["md5"]):
                logger.warning("Downloaded file checksum does not match!")
                logger.warning("File may be corrupted. Consider re-downloading.")

        return filepath

    def extract_part(self, part_name: str, fix_errors: bool = True) -> Path:
        """
        Extract a downloaded dataset part.

        Args:
            part_name: Name of the part to extract
            fix_errors: If True, apply automatic error corrections

        Returns:
            Path to the extracted directory

        Raises:
            FileNotFoundError: If the tar.gz file doesn't exist
        """
        filename = f"{part_name}.tar.gz"

        # Look for file in dataset-specific download directory
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        filepath = dataset_download_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} not found. Please download it first.")

        # Use dataset-specific extracted directory structure
        dataset_extracted_dir = self.extracted_dir / self.dataset_name / self.data_type
        dataset_extracted_dir.mkdir(parents=True, exist_ok=True)

        # Use raw directory for organized data - years go directly under data_type
        config = get_config()
        raw_dir = Path(config.data_dir) / "raw" / self.dataset_name / self.data_type
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary extraction path specific to this part
        temp_extract_path = dataset_extracted_dir / part_name

        # Extract with progress bar to dataset-specific temporary directory
        with tarfile.open(filepath, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for member in members:
                    tar.extract(member, path=temp_extract_path)
                    pbar.update(1)

        # Handle the dertagcopy directory structure (specific to this part)
        # The archives extract to a "dertagcopy" directory containing year folders
        dertagcopy_path = temp_extract_path / "dertagcopy"
        if dertagcopy_path.exists():
            # Move year directories from dertagcopy directly to raw/dataset_name/data_type/
            logger.info(f"Organizing data into raw/{self.dataset_name}/{self.data_type}/")

            years_processed = []
            for year_dir in dertagcopy_path.iterdir():
                if year_dir.is_dir():
                    year_name = year_dir.name
                    dest = raw_dir / year_name

                    # If year directory already exists, merge contents
                    if dest.exists():
                        logger.info(f"Merging {year_name} data...")
                        # Move contents of year_dir into existing dest
                        for item in year_dir.iterdir():
                            item_dest = dest / item.name
                            if not item_dest.exists():
                                shutil.move(str(item), str(item_dest))
                    else:
                        logger.info(f"Moving {year_name} data...")
                        shutil.move(str(year_dir), str(dest))

                    years_processed.append(year_name)

            # Clean up temporary extraction directory
            shutil.rmtree(temp_extract_path)

            # Clean up empty parent directories in extracted dir
            self._cleanup_empty_parent_dirs(dataset_extracted_dir)

            logger.info(f"Extracted and organized years: {', '.join(years_processed)}")

            # Apply error corrections if needed
            if fix_errors:
                self._apply_error_fixes(part_name, raw_dir)

            return raw_dir
        else:
            logger.info(f"Extracted to {temp_extract_path}")

            # Apply error corrections if needed
            if fix_errors:
                self._apply_error_fixes(part_name, temp_extract_path)

            return temp_extract_path

    def _apply_error_fixes(self, part_name: str, extract_path: Path):
        """
        Apply automatic error corrections to extracted data.

        Args:
            part_name: Name of the dataset part
            extract_path: Path to the extracted directory (base raw directory)
        """
        fixer = DataFixer(self.dataset_name, self.data_type)
        fixer.apply_fixes(part_name, extract_path)

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

    def _cleanup_empty_parent_dirs(self, start_dir: Path):
        """
        Remove empty parent directories up to and including extracted_dir.

        Args:
            start_dir: Starting directory to clean up from (e.g., extracted/der_tag/xml_ocr)
        """
        try:
            # Remove empty directories from start_dir up to and including self.extracted_dir
            current = start_dir
            while current.exists() and current >= self.extracted_dir:
                # Check if directory is empty
                if current.is_dir() and not any(current.iterdir()):
                    current.rmdir()
                    rel_path = current.relative_to(self.extracted_dir.parent)
                    logger.debug(f"Cleaned up empty directory: {rel_path}")
                    # Move up to parent
                    current = current.parent
                else:
                    # Directory not empty or doesn't exist, stop
                    break
        except OSError:
            # Silently ignore cleanup errors
            pass

    def download_parts_parallel(
        self, part_names: List[str], force_redownload: bool = False, max_workers: int = 3
    ) -> List[Path]:
        """
        Download multiple dataset parts in parallel.

        Args:
            part_names: List of part names to download
            force_redownload: If True, redownload even if files exist
            max_workers: Maximum number of parallel downloads

        Returns:
            List of paths to downloaded files
        """
        downloaded_paths = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_part = {
                executor.submit(self.download_part, part_name, force_redownload): part_name
                for part_name in part_names
            }

            # Process completed downloads
            for future in as_completed(future_to_part):
                part_name = future_to_part[future]
                try:
                    filepath = future.result()
                    downloaded_paths.append(filepath)
                except Exception as e:
                    logger.error(f"Error downloading {part_name}: {e}")

        return downloaded_paths

    def download_and_extract(
        self,
        part_names: Optional[List[str]] = None,
        fix_errors: bool = True,
        parallel: bool = False,
        max_workers: int = 3,
    ) -> List[Path]:
        """
        Download and extract one or more dataset parts.

        Args:
            part_names: List of part names to process. If None, downloads all parts.
            fix_errors: If True, apply automatic error corrections
            parallel: If True, download parts in parallel (extraction is still sequential)
            max_workers: Maximum number of parallel downloads when parallel=True

        Returns:
            List of paths to extracted directories
        """
        if part_names is None:
            part_names = [part["name"] for part in self.config["parts"]]

        extracted_paths = []

        if parallel and len(part_names) > 1:
            # Download all parts in parallel
            logger.info(
                f"Downloading {len(part_names)} parts in parallel (max {max_workers} workers)"
            )
            self.download_parts_parallel(part_names, max_workers=max_workers)

            # Extract sequentially (extraction is I/O bound and can conflict)
            for part_name in part_names:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Extracting {part_name}")
                    logger.info(f"{'='*60}")
                    extract_path = self.extract_part(part_name, fix_errors=fix_errors)
                    extracted_paths.append(extract_path)
                except Exception as e:
                    logger.error(f"Error extracting {part_name}: {e}")
                    continue
        else:
            # Sequential download and extract
            for part_name in part_names:
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing {part_name}")
                    logger.info(f"{'='*60}")

                    # Download
                    self.download_part(part_name)

                    # Extract
                    extract_path = self.extract_part(part_name, fix_errors=fix_errors)
                    extracted_paths.append(extract_path)

                except Exception as e:
                    logger.error(f"Error processing {part_name}: {e}")
                    continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Successfully processed {len(extracted_paths)}/{len(part_names)} parts")
        logger.info(f"{'='*60}\n")

        return extracted_paths

    def get_extraction_status(self) -> dict:
        """
        Get the status of all dataset parts (downloaded/extracted).

        Returns:
            Dictionary with status information for each part
        """
        status = {}

        # Get dataset-specific paths
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        config = get_config()
        raw_dir = config.data_dir / "raw" / self.dataset_name / self.data_type

        for part in self.config["parts"]:
            part_name = part["name"]
            download_file = dataset_download_dir / f"{part_name}.tar.gz"

            # Check if years from this part are extracted
            # Years are stored directly under raw/dataset_name/data_type/
            years = part.get("years", "").split("-")
            extracted = False
            extracted_years = []

            if len(years) == 2:
                try:
                    start_year = int(years[0])
                    end_year = int(years[1])
                    # Check if all years in range exist
                    for year in range(start_year, end_year + 1):
                        year_dir = raw_dir / str(year)
                        if year_dir.exists():
                            extracted_years.append(str(year))

                    # Consider extracted if at least one year is found
                    extracted = len(extracted_years) > 0
                except ValueError:
                    pass

            status[part_name] = {
                "years": part["years"],
                "size": part.get("size", "unknown"),
                "md5": part.get("md5", None),
                "downloaded": download_file.exists(),
                "extracted": extracted,
                "download_path": str(download_file) if download_file.exists() else None,
                "extract_path": str(raw_dir) if extracted else None,
                "extracted_years": extracted_years if extracted else [],
            }

        return status

    def print_status_summary(self):
        """
        Print a summary of download and extraction status.

        Note: This method uses print() for formatted table output, which is
        appropriate for this display-only utility method called from CLI.
        """
        status = self.get_extraction_status()

        print("\n" + "=" * 90)
        print("DATASET STATUS SUMMARY")
        print("=" * 90)
        print(f"{'Part Name':<25} {'Years':<12} {'Size':<12} {'Downloaded':<13} {'Extracted':<13}")
        print("-" * 90)

        for part_name, info in status.items():
            downloaded = "Yes" if info["downloaded"] else "No"
            extracted = "Yes" if info["extracted"] else "No"
            size = info.get("size", "unknown")
            print(
                f"{part_name:<25} {info['years']:<12} {size:<12} "
                f"{downloaded:<13} {extracted:<13}"
            )

        print("=" * 90 + "\n")
