"""
Data gathering utilities for newspaper explorer.
Handles downloading and extraction of Zenodo newspaper collections.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import shutil
import tarfile
from typing import Any, Optional

import click
from natsort import natsorted
import requests
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.checksums import verify_md5_checksum
from newspaper_explorer.data.utils.fixes import DataFixer
from newspaper_explorer.data.utils.sources import load_source_config

logger = logging.getLogger(__name__)


class ZenodoDownloader:
    """Download and extract newspaper data from Zenodo collections."""

    def __init__(self, source_name: str = "der_tag", data_dir: Optional[Path] = None) -> None:
        """
        Initialize the Zenodo downloader.

        Args:
            source_name: Name of the source to download (e.g., 'der_tag')
            data_dir: Directory to store downloaded and extracted data.
                     Defaults to config or workspace data/ directory.
        """
        config = get_config()

        if data_dir is None:
            data_dir = config.data_dir

        self.data_dir = Path(data_dir)
        self.download_dir: Path = Path(config.archives_dir)
        self.extracted_dir: Path = Path(config.extracted_dir)

        # Ensure directories exist
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        # Load source configuration
        self.config = load_source_config(source_name)

        # Get dataset metadata
        self.dataset_name: str = self.config.dataset_name
        self.data_type: str = self.config.data_type

        # Get valid year range from source config
        self.min_year, self.max_year = self.config.get_year_range()

    def list_available_parts(self) -> list[dict[str, Any]]:
        """
        List all available dataset parts from the configuration.

        Returns:
            List of dictionaries containing part information.
        """
        return [part.model_dump() for part in self.config.parts]

    def download_part(self, part_name: str, *, force_redownload: bool = False) -> Path:
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
        for part in self.config.parts:
            if part.name == part_name:
                part_info = part
                break

        if part_info is None:
            available = [p.name for p in self.config.parts]
            raise ValueError(f"Part '{part_name}' not found. Available parts: {available}")

        url = part_info.url
        filename = f"{part_name}.tar.gz"

        # Create dataset-specific download directory
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        dataset_download_dir.mkdir(parents=True, exist_ok=True)
        filepath = dataset_download_dir / filename

        # Check if file already exists
        if filepath.exists() and not force_redownload:
            logger.info(f"File {filename} already exists")
            # Verify checksum if available
            if part_info.md5:
                if verify_md5_checksum(filepath, part_info.md5):
                    logger.info("Skipping download - file verified")
                    return filepath
                logger.warning("Checksum failed - will re-download")
            else:
                logger.info("Skipping download (no checksum available)")
                return filepath

        logger.info(f"Downloading {part_name}...")

        # Download with progress bar (30 second timeout)
        response = requests.get(str(url), stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            filepath.open("wb") as f,
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
        if part_info.md5 and not verify_md5_checksum(filepath, part_info.md5):
            logger.warning("Downloaded file checksum does not match!")
            logger.warning("File may be corrupted. Consider re-downloading.")

        return filepath

    def extract_part(self, part_name: str, *, fix_errors: bool = True) -> Path:
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
        filepath = self._get_archive_path(part_name)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath.name} not found. Please download it first.")

        # Set up directories
        dataset_extracted_dir = self.extracted_dir / self.dataset_name / self.data_type
        dataset_extracted_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = Path(get_config().data_dir) / "raw" / self.dataset_name / self.data_type
        raw_dir.mkdir(parents=True, exist_ok=True)

        temp_extract_path = dataset_extracted_dir / part_name

        # Extract archive
        self._extract_archive(filepath, temp_extract_path)

        # Organize extracted files and get result path
        result_path = self._organize_extracted_years(
            temp_extract_path, raw_dir, dataset_extracted_dir, part_name
        )

        # Apply error corrections if needed
        if fix_errors:
            fixer = DataFixer(self.dataset_name, self.data_type)
            fixer.apply_fixes(part_name, result_path)

        return result_path

    def _get_archive_path(self, part_name: str) -> Path:
        """Get path to the downloaded archive file for a part."""
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        return dataset_download_dir / f"{part_name}.tar.gz"

    def _extract_archive(self, filepath: Path, dest_path: Path) -> None:
        """Extract tar.gz archive with progress bar."""
        with tarfile.open(filepath, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for member in members:
                    tar.extract(member, path=dest_path)
                    pbar.update(1)

    def _find_year_source_path(self, temp_extract_path: Path, part_name: str) -> Optional[Path]:
        """
        Find the source path containing year directories.

        Archives may have a 'dertagcopy' prefix or contain years directly.

        Returns:
            Path to directory containing year folders, or None if not found.
        """
        dertagcopy_path = temp_extract_path / "dertagcopy"

        if not dertagcopy_path.exists():
            return None

        # Check if there are also direct year folders (takes precedence)
        year_dirs = [
            d
            for d in temp_extract_path.iterdir()
            if d.is_dir() and d.name.isdigit() and self.min_year <= int(d.name) <= self.max_year
        ]

        if year_dirs:
            logger.info(f"Found direct year structure in {part_name} (no dertagcopy prefix)")
            return temp_extract_path

        return dertagcopy_path

    def _organize_extracted_years(
        self,
        temp_extract_path: Path,
        raw_dir: Path,
        dataset_extracted_dir: Path,
        part_name: str,
    ) -> Path:
        """
        Organize extracted year directories into the raw data structure.

        Returns:
            Path to the organized data (raw_dir if years found, else temp path).
        """
        source_path = self._find_year_source_path(temp_extract_path, part_name)

        if not source_path:
            logger.info(f"Extracted to {temp_extract_path}")
            return temp_extract_path

        logger.info(f"Organizing data into raw/{self.dataset_name}/{self.data_type}/")
        years_processed = self._move_year_directories(source_path, raw_dir)

        # Clean up
        shutil.rmtree(temp_extract_path)
        self._cleanup_empty_parent_dirs(dataset_extracted_dir)

        logger.info(f"Extracted and organized years: {', '.join(years_processed)}")
        return raw_dir

    def _move_year_directories(self, source_path: Path, raw_dir: Path) -> list[str]:
        """
        Move year directories from source to raw directory, merging if needed.

        Returns:
            List of year names that were processed.
        """
        years_processed: list[str] = []

        for year_dir in natsorted(source_path.iterdir()):
            # Check if path is a valid year directory within the source's year range
            is_year_dir = (
                year_dir.is_dir()
                and year_dir.name.isdigit()
                and self.min_year <= int(year_dir.name) <= self.max_year
            )
            if not is_year_dir:
                continue

            year_name = year_dir.name
            dest = raw_dir / year_name

            if dest.exists():
                self._merge_directory(year_dir, dest)
            else:
                logger.info(f"Moving {year_name} data...")
                shutil.move(str(year_dir), str(dest))

            years_processed.append(year_name)

        return years_processed

    def _merge_directory(self, source: Path, dest: Path) -> None:
        """
        Merge contents of source directory into destination.

        Handles two levels of nesting (for year/month structures).
        """
        logger.info(f"Merging {source.name} data...")

        for item in natsorted(source.iterdir()):
            item_dest = dest / item.name

            if not item_dest.exists():
                shutil.move(str(item), str(item_dest))
            elif item.is_dir() and item_dest.is_dir():
                # Merge one level deeper (for month directories)
                for subitem in natsorted(item.iterdir()):
                    subitem_dest = item_dest / subitem.name
                    if not subitem_dest.exists():
                        shutil.move(str(subitem), str(subitem_dest))

    def _cleanup_empty_parent_dirs(self, start_dir: Path) -> None:
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
        self, part_names: list[str], *, force_redownload: bool = False, max_workers: int = 3
    ) -> list[Path]:
        """
        Download multiple dataset parts in parallel.

        Args:
            part_names: List of part names to download
            force_redownload: If True, redownload even if files exist
            max_workers: Maximum number of parallel downloads

        Returns:
            List of paths to downloaded files
        """
        downloaded_paths: list[Path] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_part = {
                executor.submit(
                    self.download_part, part_name, force_redownload=force_redownload
                ): part_name
                for part_name in part_names
            }

            # Process completed downloads
            for future in as_completed(future_to_part):
                part_name = future_to_part[future]
                try:
                    filepath = future.result()
                    downloaded_paths.append(filepath)
                except (
                    FileNotFoundError,
                    OSError,
                    PermissionError,
                    ValueError,
                    requests.RequestException,
                ) as e:
                    logger.error(f"Error downloading {part_name}: {e}")

        return downloaded_paths

    def download_and_extract(
        self,
        part_names: Optional[list[str]] = None,
        *,
        fix_errors: bool = True,
        parallel: bool = False,
        max_workers: int = 3,
    ) -> list[Path]:
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
            part_names = [part.name for part in self.config.parts]

        extracted_paths: list[Path] = []

        if parallel and len(part_names) > 1:
            # Download all parts in parallel first
            logger.info(
                f"Downloading {len(part_names)} parts in parallel (max {max_workers} workers)"
            )
            self.download_parts_parallel(part_names, max_workers=max_workers)

            # Extract sequentially (extraction is I/O bound and can conflict)
            for part_name in part_names:
                try:
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"Extracting {part_name}")
                    logger.info(f"{'=' * 60}")
                    extract_path = self.extract_part(part_name, fix_errors=fix_errors)
                    extracted_paths.append(extract_path)
                except (FileNotFoundError, tarfile.TarError, OSError, PermissionError) as e:
                    logger.error(f"Error extracting {part_name}: {e}")
                    continue
        else:
            # Sequential download and extract
            for part_name in part_names:
                try:
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"Processing {part_name}")
                    logger.info(f"{'=' * 60}")

                    # Download
                    self.download_part(part_name)

                    # Extract
                    extract_path = self.extract_part(part_name, fix_errors=fix_errors)
                    extracted_paths.append(extract_path)

                except (
                    FileNotFoundError,
                    tarfile.TarError,
                    OSError,
                    PermissionError,
                    ValueError,
                    requests.RequestException,
                ) as e:
                    logger.error(f"Error processing {part_name}: {e}")
                    continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Successfully processed {len(extracted_paths)}/{len(part_names)} parts")
        logger.info(f"{'=' * 60}\n")

        return extracted_paths

    def get_extraction_status(self) -> dict[str, Any]:
        """
        Get the status of all dataset parts (downloaded/extracted).

        Returns:
            Dictionary with status information for each part
        """
        status: dict[str, Any] = {}

        # Get dataset-specific paths
        dataset_download_dir = self.download_dir / self.dataset_name / self.data_type
        config = get_config()
        raw_dir = config.data_dir / "raw" / self.dataset_name / self.data_type

        for part in self.config.parts:
            part_name = part.name
            download_file = dataset_download_dir / f"{part_name}.tar.gz"

            # Check if years from this part are extracted
            # Years are stored directly under raw/dataset_name/data_type/
            extracted_years: list[str] = []

            try:
                start_year, end_year = part.years.split("-") if part.years else ("", "")
                extracted_years = [
                    str(year)
                    for year in range(int(start_year), int(end_year) + 1)
                    if (raw_dir / str(year)).exists()
                ]
            except ValueError:
                pass  # Invalid year range format

            status[part_name] = {
                "years": part.years or "unknown",
                "size": part.size or "unknown",
                "md5": part.md5,
                "downloaded": download_file.exists(),
                "extracted": bool(extracted_years),
                "download_path": str(download_file) if download_file.exists() else None,
                "extract_path": str(raw_dir) if extracted_years else None,
                "extracted_years": extracted_years,
            }

        return status

    def print_status_summary(self) -> None:
        """
        Print a summary of download and extraction status.

        Note: This method uses click.echo() for formatted table output, which is
        appropriate for this display-only utility method called from CLI.
        """
        status = self.get_extraction_status()

        click.echo("\n" + "=" * 90)
        click.echo("DATASET STATUS SUMMARY")
        click.echo("=" * 90)
        click.echo(
            f"{'Part Name':<25} {'Years':<12} {'Size':<12} {'Downloaded':<13} {'Extracted':<13}"
        )
        click.echo("-" * 90)

        for part_name, info in status.items():
            downloaded = "Yes" if info["downloaded"] else "No"
            extracted = "Yes" if info["extracted"] else "No"
            size = info.get("size", "unknown")
            click.echo(
                f"{part_name:<25} {info['years']:<12} {size:<12} {downloaded:<13} {extracted:<13}"
            )

        click.echo("=" * 90 + "\n")
