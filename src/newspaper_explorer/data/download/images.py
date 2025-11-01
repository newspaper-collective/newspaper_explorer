"""
Image downloader for newspaper page scans.
Downloads high-resolution images from METS XML references with validation.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from lxml import etree
from natsort import natsorted
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.validation import validate_image_file
from newspaper_explorer.utils.sources import load_source_config

logger = logging.getLogger(__name__)


@dataclass
class ImageReference:
    """Reference to an image in METS XML."""

    file_id: str
    url: str
    extension: str = ".jpg"


class ImageDownloader:
    """Download newspaper page images from METS XML files."""

    NAMESPACES = {
        "mets": "http://www.loc.gov/METS/",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    def __init__(
        self,
        source_name: str,
        max_workers: int = 8,
        max_retries: int = 3,
        timeout: int = 30,
        validate: bool = True,
        min_image_size: int = 1024,
    ):
        """
        Initialize image downloader.

        Args:
            source_name: Name of the source (e.g., 'der_tag')
            max_workers: Maximum parallel download threads
            max_retries: Maximum retry attempts for failed downloads
            timeout: Request timeout in seconds
            validate: Whether to validate downloaded images (default: True)
            min_image_size: Minimum expected image size in bytes (default: 1KB)
        """
        self.source_name = source_name
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.validate = validate
        self.min_image_size = min_image_size

        # Load source configuration
        self.config = load_source_config(source_name)
        config = get_config()
        data_dir = Path(config.data_dir)

        # Setup paths following data/raw/{source}/images structure
        self.dataset_name = str(self.config["dataset_name"])
        self.data_type = str(self.config["data_type"])
        self.xml_dir: Path = data_dir / "raw" / self.dataset_name / self.data_type
        self.images_dir: Path = data_dir / "raw" / self.dataset_name / "images"

        logger.info(f"Initialized ImageDownloader for '{source_name}'")
        logger.info(f"XML directory: {self.xml_dir}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Validation: {'enabled' if self.validate else 'disabled'}")

    def find_mets_files(self) -> List[Path]:
        """
        Find all METS XML files (excluding fulltext).

        Returns:
            List of METS XML file paths
        """
        if not self.xml_dir.exists():
            logger.warning(f"XML directory not found: {self.xml_dir}")
            return []

        mets_files = []
        for xml_file in natsorted(self.xml_dir.rglob("*.xml")):
            # Skip fulltext directory
            if "fulltext" not in str(xml_file):
                mets_files.append(xml_file)

        logger.info(f"Found {len(mets_files)} METS files")
        return mets_files

    def extract_image_references(self, mets_file: Path) -> List[ImageReference]:
        """
        Extract image URLs from METS XML MAX fileGrp.

        Args:
            mets_file: Path to METS XML file

        Returns:
            List of ImageReference objects
        """
        try:
            tree = etree.parse(str(mets_file))
            root = tree.getroot()

            # Find the fileGrp with USE="MAX" (maximum resolution)
            max_file_grp = root.find('.//mets:fileGrp[@USE="MAX"]', self.NAMESPACES)

            if max_file_grp is None:
                logger.debug(f"No MAX fileGrp in {mets_file.name}")
                return []

            references = []
            for file_elem in max_file_grp.findall(".//mets:file", self.NAMESPACES):
                file_id = file_elem.get("ID", "unknown")

                # Find FLocat with URL
                flocat = file_elem.find(".//mets:FLocat", self.NAMESPACES)
                if flocat is not None:
                    url = flocat.get("{http://www.w3.org/1999/xlink}href")
                    if url:
                        # Extract extension from URL
                        parsed_url = urlparse(url)
                        extension = Path(parsed_url.path).suffix or ".jpg"
                        references.append(
                            ImageReference(file_id=file_id, url=url, extension=extension)
                        )

            return references

        except Exception as e:
            logger.error(f"Error parsing {mets_file}: {e}")
            return []

    def _get_image_path(self, mets_file: Path, image_ref: ImageReference) -> Path:
        """
        Calculate target path for image, mirroring XML directory structure.

        Args:
            mets_file: Source METS file
            image_ref: Image reference

        Returns:
            Target path for downloaded image
        """
        # Get relative path from xml_dir to maintain year/month/day structure
        try:
            relative_path = mets_file.parent.relative_to(self.xml_dir)
        except ValueError:
            # Fallback if not in expected directory
            relative_path = Path(".")
        # Create target directory: data/raw/{source}/images/{year}/{month}/{day}/
        target_dir: Path = self.images_dir / relative_path
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Filename: {file_id}{extension}
        filename = f"{image_ref.file_id}{image_ref.extension}"
        return target_dir / filename

    def _download_single_image(self, url: str, save_path: Path, img_id: str) -> Dict[str, Any]:
        """
        Download a single image with retry logic and validation.

        Args:
            url: Image URL
            save_path: Target file path
            img_id: Image identifier

        Returns:
            Result dictionary with status and validation info
        """
        # Skip if already exists and is valid
        if save_path.exists():
            # Optionally validate existing file
            if self.validate:
                validation = validate_image_file(save_path, self.min_image_size)
                if not validation.is_valid:
                    logger.warning(
                        f"Existing file {save_path.name} failed validation: {validation.error}"
                    )
                    # Remove invalid file and re-download
                    save_path.unlink()
                else:
                    return {
                        "success": True,
                        "skipped": True,
                        "filename": save_path.name,
                        "id": img_id,
                        "validated": True,
                    }
            else:
                return {
                    "success": True,
                    "skipped": True,
                    "filename": save_path.name,
                    "id": img_id,
                    "validated": False,
                }

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Atomic write
                temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                with open(temp_path, "wb") as f:
                    f.write(response.content)

                temp_path.rename(save_path)

                # Validate downloaded image if enabled
                if self.validate:
                    validation = validate_image_file(save_path, self.min_image_size)
                    if not validation.is_valid:
                        # Remove invalid file
                        save_path.unlink()
                        last_error = f"Validation failed: {validation.error}"
                        if attempt < self.max_retries - 1:
                            time.sleep(1 * (attempt + 1))
                            continue
                        # Final attempt failed
                        return {
                            "success": False,
                            "skipped": False,
                            "filename": save_path.name,
                            "id": img_id,
                            "error": last_error,
                            "validated": True,
                        }

                return {
                    "success": True,
                    "skipped": False,
                    "filename": save_path.name,
                    "id": img_id,
                    "validated": self.validate,
                }

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue

        return {
            "success": False,
            "skipped": False,
            "filename": save_path.name,
            "id": img_id,
            "error": last_error,
            "validated": False,
        }

    def download_images(self, mets_files: Optional[List[Path]] = None) -> Dict[str, int]:
        """
        Download images from METS files.

        Args:
            mets_files: Optional list of METS files. If None, finds all files.

        Returns:
            Statistics dictionary
        """
        if mets_files is None:
            mets_files = self.find_mets_files()

        if not mets_files:
            logger.warning("No METS files to process")
            return {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        stats = {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0}

        # Process each METS file
        for mets_file in tqdm(mets_files, desc="Processing METS files", unit="file"):
            # Extract image references
            images = self.extract_image_references(mets_file)

            if not images:
                continue

            stats["total"] += len(images)

            # Prepare download tasks
            download_tasks = []
            for img_ref in images:
                save_path = self._get_image_path(mets_file, img_ref)
                download_tasks.append((img_ref.url, save_path, img_ref.file_id))

            # Download in parallel with progress bar
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._download_single_image, url, save_path, img_id): (
                        url,
                        save_path,
                        img_id,
                    )
                    for url, save_path, img_id in download_tasks
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  {mets_file.name}",
                    unit="img",
                    leave=False,
                ):
                    try:
                        result = future.result()
                        if result["success"]:
                            if result["skipped"]:
                                stats["skipped"] += 1
                            else:
                                stats["downloaded"] += 1
                        else:
                            stats["failed"] += 1
                            logger.warning(
                                f"Failed {result['filename']}: {result.get('error', 'Unknown')}"
                            )
                    except Exception as e:
                        stats["failed"] += 1
                        logger.error(f"Exception during download: {e}")

        # Log summary
        logger.info("=" * 60)
        logger.info("Download complete!")
        logger.info(f"Total images: {stats['total']}")
        logger.info(f"Downloaded: {stats['downloaded']}")
        logger.info(f"Skipped (exist): {stats['skipped']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("=" * 60)

        return stats

    def get_download_status(self) -> Dict[str, Any]:
        """
        Get image download status without downloading.

        Returns:
            Dictionary with download status information
        """
        mets_files = self.find_mets_files()

        if not mets_files:
            return {
                "images_dir": self.images_dir,
                "images_dir_exists": False,
                "mets_files": 0,
                "total_images_expected": 0,
                "images_downloaded": 0,
                "coverage_pct": 0.0,
            }

        # Count expected images from METS
        total_expected = 0
        for mets_file in mets_files:
            images = self.extract_image_references(mets_file)
            total_expected += len(images)

        # Count downloaded images
        downloaded = 0
        if self.images_dir.exists():
            # Count all image files in the images directory
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                downloaded += len(list(self.images_dir.rglob(f"*{ext}")))

        coverage_pct = (downloaded / total_expected * 100) if total_expected > 0 else 0.0

        return {
            "images_dir": self.images_dir,
            "images_dir_exists": self.images_dir.exists(),
            "mets_files": len(mets_files),
            "total_images_expected": total_expected,
            "images_downloaded": downloaded,
            "coverage_pct": coverage_pct,
        }

    def validate_downloaded_images(self, min_size_bytes: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate all downloaded images in the images directory.

        Args:
            min_size_bytes: Minimum expected image size (uses self.min_image_size if None)

        Returns:
            Dictionary with validation statistics:
            - total: Total images checked
            - valid: Number of valid images
            - invalid: Number of invalid images
            - invalid_list: List of (path, error) tuples for invalid images
        """
        if min_size_bytes is None:
            min_size_bytes = self.min_image_size

        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "invalid_list": [],
            }

        # Find all image files
        logger.info(f"Scanning for images in {self.images_dir}")
        image_files: List[Path] = []
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            image_files.extend(self.images_dir.rglob(f"*{ext}"))

        if not image_files:
            logger.warning("No image files found")
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "invalid_list": [],
            }

        logger.info(f"Found {len(image_files)} images to validate")

        valid_count = 0
        invalid_count = 0
        invalid_list = []

        # Validate images with progress bar
        for img_path in tqdm(image_files, desc="Validating images", unit="img"):
            result = validate_image_file(img_path, min_size_bytes)

            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                relative_path = img_path.relative_to(self.images_dir)
                invalid_list.append((str(relative_path), result.error))
                logger.debug(f"Invalid image: {relative_path} - {result.error}")

        # Log summary
        logger.info("=" * 60)
        logger.info("Validation complete!")
        logger.info(f"Total images: {len(image_files)}")
        logger.info(f"Valid: {valid_count}")
        logger.info(f"Invalid: {invalid_count}")
        logger.info("=" * 60)

        return {
            "total": len(image_files),
            "valid": valid_count,
            "invalid": invalid_count,
            "invalid_list": invalid_list,
        }
