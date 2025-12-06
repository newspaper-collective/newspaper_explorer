"""
Image validation utilities for newspaper data.

General-purpose utilities for validating image files.
For source-specific validation (empty files, ALTO-METS relationships),
see newspaper_explorer.data.processing.validation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.models.data.images import ImageValidationResult

logger = logging.getLogger(__name__)


def validate_image_file(
    image_path: Path, min_size_bytes: Optional[int] = None
) -> ImageValidationResult:
    """
    Validate a downloaded image file.

    Checks if the image file:
    1. Exists and has minimum size (not empty/truncated)
    2. Is a valid image format that can be opened
    3. Has reasonable dimensions

    Args:
        image_path: Path to image file
        min_size_bytes: Minimum expected file size (default: from config)

    Returns:
        ImageValidationResult with validation details

    Example:
        >>> result = validate_image_file(Path("image.jpg"))
        >>> if not result.is_valid:
        ...     print(f"Invalid: {result.error}")
    """
    if min_size_bytes is None:
        min_size_bytes = get_config().min_image_size_bytes
    # Check file exists
    if not image_path.exists():
        return ImageValidationResult(
            is_valid=False, file_path=image_path, error="File does not exist"
        )

    # Check file size
    file_size: Optional[int] = None
    try:
        file_size = image_path.stat().st_size

        if file_size < min_size_bytes:
            return ImageValidationResult(
                is_valid=False,
                file_path=image_path,
                file_size=file_size,
                error=f"File too small ({file_size} bytes < {min_size_bytes} bytes)",
            )

        # Try to open and validate image
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format

            # Check for reasonable dimensions
            if width == 0 or height == 0:
                return ImageValidationResult(
                    is_valid=False,
                    file_path=image_path,
                    file_size=file_size,
                    width=width,
                    height=height,
                    format=format_name,
                    error="Image has zero width or height",
                )

            return ImageValidationResult(
                is_valid=True,
                file_path=image_path,
                file_size=file_size,
                width=width,
                height=height,
                format=format_name,
            )

    except OSError as e:
        # OSError covers PIL.UnidentifiedImageError and file I/O errors
        return ImageValidationResult(
            is_valid=False,
            file_path=image_path,
            file_size=file_size,
            error=f"Failed to validate image: {e!s}",
        )


def check_image_size(image_path: Path, min_size_bytes: Optional[int] = None) -> bool:
    """
    Quick check if image file meets minimum size requirement.

    Args:
        image_path: Path to image file
        min_size_bytes: Minimum expected file size (default: from config)

    Returns:
        True if file exists and meets size requirement

    Example:
        >>> if not check_image_size(Path("image.jpg"), min_size_bytes=5000):
        ...     print("Image too small or missing")
    """
    if min_size_bytes is None:
        min_size_bytes = get_config().min_image_size_bytes

    if not image_path.exists():
        return False
    try:
        return image_path.stat().st_size >= min_size_bytes

    except OSError:
        # Catches permission errors, file access issues
        return False


def validate_images_in_directory(
    images_dir: Path,
    min_size_bytes: Optional[int] = None,
    *,
    extensions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Validate all images in a directory.

    Args:
        images_dir: Directory containing images to validate
        min_size_bytes: Minimum expected image size (default: from config)
        extensions: List of extensions to check (default: common image formats)

    Returns:
        Dictionary with validation statistics:
        - total: Total images checked
        - valid: Number of valid images
        - invalid: Number of invalid images
        - invalid_list: List of (relative_path, error) tuples for invalid images

    Example:
        >>> result = validate_images_in_directory(Path("data/images"))
        >>> print(f"Valid: {result['valid']}/{result['total']}")
    """

    if min_size_bytes is None:
        min_size_bytes = get_config().min_image_size_bytes

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "invalid_list": [],
        }

    # Find all image files
    logger.info(f"Scanning for images in {images_dir}")
    image_files: list[Path] = []
    for ext in extensions:
        image_files.extend(images_dir.rglob(f"*{ext}"))

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
    invalid_list: list[tuple[str, str]] = []

    # Validate images with progress bar
    for img_path in tqdm(image_files, desc="Validating images", unit="img"):
        result = validate_image_file(img_path, min_size_bytes)

        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            relative_path = img_path.relative_to(images_dir)
            error_msg = result.error or "Unknown validation error"
            invalid_list.append((str(relative_path), error_msg))
            logger.debug(f"Invalid image: {relative_path} - {error_msg}")

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
