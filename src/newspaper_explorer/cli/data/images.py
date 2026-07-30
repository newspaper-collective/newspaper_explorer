"""CLI commands for downloading images."""

import logging

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    force_option,
    max_retries_option,
    min_image_size_option,
    num_workers_option,
    source_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.download.images import ImageDownloader
from newspaper_explorer.data.indexing.image_index import ImageIndexer


@click.group(name="images")
def images_group() -> None:
    """Image download and indexing commands."""
    pass


@images_group.command(name="download")
@source_option()
@num_workers_option(default=8)
@max_retries_option(default=3)
@min_image_size_option(default=1024)
@click.option(
    "--no-validate",
    is_flag=True,
    help="Skip image validation after download",
)
def download_images(
    source: str, *, num_workers: int, max_retries: int, no_validate: bool, min_size: int
) -> None:
    """
    Download high-resolution newspaper page images from METS XML.

    Images are stored in data/raw/{source}/images/ with the same
    directory structure as the XML files (year/month/day).

    Downloaded images are validated by default to ensure they are:
    - Valid image files that can be opened
    - Meet minimum size requirements (not corrupted/truncated)

    Use --no-validate to skip validation (faster but risky).

    \b
    Examples:
      newspaper-explorer data images download --source der_tag
      newspaper-explorer data images download --source der_tag --max-workers 16
      newspaper-explorer data images download --source der_tag --no-validate
      newspaper-explorer data images download --source der_tag --min-size 5000
    """

    # Configure logging so user sees download progress
    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        output.header(f"DOWNLOAD IMAGES: {source.upper()}")

        # Show configuration
        output.section("CONFIGURATION")
        output.key_value("Parallel workers", num_workers)
        output.key_value("Max retries", max_retries)
        output.key_value("Validation", "Enabled" if not no_validate else "Disabled")
        if not no_validate:
            output.key_value("Minimum image size", f"{min_size} bytes")

        output.section("DOWNLOADING")
        downloader = ImageDownloader(
            source_name=source,
            max_workers=num_workers,
            max_retries=max_retries,
            validate=not no_validate,
            min_image_size=min_size,
        )

        stats = downloader.download_images()

        output.section("RESULTS")
        output.key_value("Total images found", f"{stats['total']:,}")
        output.key_value("Successfully downloaded", f"{stats['downloaded']:,}")
        output.key_value("Skipped (already exist)", f"{stats['skipped']:,}")
        output.key_value("Failed", f"{stats['failed']:,}")

        click.echo()
        if stats["failed"] > 0:
            output.warning("Some images failed to download. Check logs for details.")
        else:
            output.success("All images downloaded successfully!")

        output.info(
            "Tip: Validate images with 'newspaper-explorer data validation images'",
            muted=True,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@images_group.command(name="index")
@source_option()
@force_option()
def index_images(source: str, *, force: bool) -> None:
    """
    Create an image index with dimensions and metadata.

    This command creates a comprehensive parquet index of all downloaded images,
    enriched with:
    - Original image dimensions from ALTO XML files
    - Issue metadata from METS XML files (title, date, volume, page count)
    - File paths and sizes

    The index enables fast lookups and is required for accurate coordinate
    scaling when matching layout detections with OCR text.

    \b
    Examples:
      newspaper-explorer data images index --source der_tag
      newspaper-explorer data images index --source der_tag --force
    """

    # Configure logging
    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        output.header(f"CREATE IMAGE INDEX: {source.upper()}")

        indexer = ImageIndexer(source)

        # Check if index already exists
        existing_index = indexer.load_index()
        if existing_index is not None and not force:
            output.section("INDEX STATUS")
            output.success(f"Image index already exists with {len(existing_index):,} images")
            output.key_value("Location", str(indexer.index_path))
            click.echo()
            output.info("Use --force to recreate it", muted=True)
            return

        # Show configuration
        output.section("CONFIGURATION")
        output.key_value("Force rebuild", "Yes" if force else "No")

        output.section("BUILDING INDEX")
        output.info("Extracting dimensions from ALTO XML files...")
        output.info("Loading metadata from METS XML files...", muted=True)
        output.info("Scanning image directory...", muted=True)

        # Create index
        image_index = indexer.create_index(force_rebuild=force)

        output.section("RESULTS")
        output.key_value("Total images indexed", f"{len(image_index):,}")
        output.key_value("Index location", str(indexer.index_path))

        # Show statistics
        stats = indexer.get_stats()
        output.divider()
        output.key_value("Total size", f"{stats.total_size_gb:.2f} GB")
        output.key_value("Average file size", f"{stats.avg_file_size_mb:.2f} MB")

        # Calculate year span
        year_span = (
            (stats.max_year - stats.min_year + 1) if stats.min_year and stats.max_year else 0
        )
        output.key_value(
            "Year range",
            f"{stats.min_year} - {stats.max_year} ({year_span} years)",
        )

        # Show completeness
        with_real_dims = image_index.filter(image_index["width"].is_not_null())
        with_alto_dims = image_index.filter(image_index["alto_width"].is_not_null())
        with_mets = image_index.filter(image_index["newspaper_title"].is_not_null())

        output.section("DATA COMPLETENESS")
        output.key_value(
            "Images with real dimensions",
            f"{len(with_real_dims):,} ({len(with_real_dims) / len(image_index) * 100:.1f}%)",
        )
        output.key_value(
            "Images with ALTO dimensions",
            f"{len(with_alto_dims):,} ({len(with_alto_dims) / len(image_index) * 100:.1f}%)",
        )
        output.key_value(
            "Images with METS data",
            f"{len(with_mets):,} ({len(with_mets) / len(image_index) * 100:.1f}%)",
        )

        click.echo()
        if len(with_alto_dims) < len(image_index):
            missing_dims = len(image_index) - len(with_alto_dims)
            output.warning(
                f"{missing_dims:,} images missing ALTO dimensions (ALTO files not found)"
            )

        if len(with_mets) < len(image_index):
            missing_mets = len(image_index) - len(with_mets)
            output.warning(
                f"{missing_mets:,} images missing METS metadata (METS files not available)"
            )

        output.success("Image index created successfully!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)
