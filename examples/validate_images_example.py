#!/usr/bin/env python
"""
Example script demonstrating image validation.

Shows how to use the validation utilities to check downloaded images.
"""

from pathlib import Path

from newspaper_explorer.data.utils.validation import (
    check_image_size,
    validate_image_file,
)


def validate_images_in_directory(directory: Path, min_size: int = 1024):
    """
    Validate all images in a directory.

    Args:
        directory: Directory containing images
        min_size: Minimum expected image size in bytes
    """
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.rglob(f"*{ext}"))

    if not image_files:
        print(f"No image files found in {directory}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Validating with minimum size: {min_size} bytes\n")

    valid_count = 0
    invalid_count = 0
    invalid_images = []

    for img_path in image_files:
        # Quick size check first
        if not check_image_size(img_path, min_size):
            invalid_count += 1
            invalid_images.append((img_path, "File too small"))
            continue

        # Full validation
        result = validate_image_file(img_path, min_size)
        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_images.append((img_path, result.error))

    print(f"Valid images: {valid_count}")
    print(f"Invalid images: {invalid_count}\n")

    if invalid_images:
        print("Invalid images:")
        for img_path, error in invalid_images[:10]:  # Show first 10
            print(f"  - {img_path.name}: {error}")
        if len(invalid_images) > 10:
            print(f"  ... and {len(invalid_images) - 10} more")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_images_example.py <directory> [min_size]")
        print("\nExample:")
        print("  python validate_images_example.py data/raw/der_tag/images")
        print("  python validate_images_example.py data/raw/der_tag/images 5000")
        sys.exit(1)

    directory = Path(sys.argv[1])
    min_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    validate_images_in_directory(directory, min_size)


if __name__ == "__main__":
    main()
