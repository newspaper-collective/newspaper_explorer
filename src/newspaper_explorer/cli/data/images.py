"""
CLI commands for downloading images.
"""

import click

from newspaper_explorer.config.base import get_config


def register_image_commands(data_group):
    """Register image commands to the data group."""

    @data_group.command("download-images")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel download threads",
        show_default=True,
    )
    @click.option(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed downloads",
        show_default=True,
    )
    @click.option(
        "--no-validate",
        is_flag=True,
        help="Skip image validation after download",
    )
    @click.option(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum expected image size in bytes",
        show_default=True,
    )
    def download_images(source, max_workers, max_retries, no_validate, min_size):
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
          newspaper-explorer data download-images --source der_tag
          newspaper-explorer data download-images --source der_tag --max-workers 16
          newspaper-explorer data download-images --source der_tag --no-validate
          newspaper-explorer data download-images --source der_tag --min-size 5000
        """
        import logging

        from newspaper_explorer.data.download.images import ImageDownloader

        # Configure logging so user sees download progress
        config = get_config()
        logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

        try:
            click.echo(f"Downloading images for source: {source}")
            click.echo(f"Using {max_workers} parallel workers")
            click.echo(f"Validation: {'disabled' if no_validate else 'enabled'}")
            if not no_validate:
                click.echo(f"Minimum image size: {min_size} bytes")
            click.echo()

            downloader = ImageDownloader(
                source_name=source,
                max_workers=max_workers,
                max_retries=max_retries,
                validate=not no_validate,
                min_image_size=min_size,
            )

            stats = downloader.download_images()

            click.echo("\n" + "=" * 60)
            click.echo("Image Download Summary")
            click.echo("=" * 60)
            click.echo(f"Total images found:      {stats['total']}")
            click.echo(f"Successfully downloaded: {stats['downloaded']}")
            click.echo(f"Skipped (already exist): {stats['skipped']}")
            click.echo(f"Failed:                  {stats['failed']}")
            click.echo("=" * 60)

            if stats["failed"] > 0:
                click.echo(
                    "\nWarning: Some images failed to download. Check logs for details.",
                    err=True,
                )

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()

    @data_group.command("validate-images")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum expected image size in bytes",
        show_default=True,
    )
    @click.option(
        "--save-report",
        type=click.Path(),
        default=None,
        help="Save list of invalid images to file",
    )
    def validate_images(source, min_size, save_report):
        """
        Validate already downloaded images for a source.

        Checks all downloaded images in data/raw/{source}/images/ to ensure:
        - Files are valid image formats that can be opened
        - Files meet minimum size requirements (not corrupted/truncated)

        Invalid images are reported and can be saved to a file for review.

        \b
        Examples:
          newspaper-explorer data validate-images --source der_tag
          newspaper-explorer data validate-images --source der_tag --min-size 5000
          newspaper-explorer data validate-images --source der_tag --save-report invalid_images.txt
        """
        import logging

        from newspaper_explorer.data.download.images import ImageDownloader

        # Configure logging
        config = get_config()
        logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

        try:
            click.echo(f"Validating images for source: {source}")
            click.echo(f"Minimum image size: {min_size} bytes\n")

            downloader = ImageDownloader(source_name=source, validate=True, min_image_size=min_size)

            result = downloader.validate_downloaded_images(min_size_bytes=min_size)

            click.echo("\n" + "=" * 60)
            click.echo("Image Validation Summary")
            click.echo("=" * 60)
            click.echo(f"Total images checked: {result['total']}")
            click.echo(f"Valid images:         {result['valid']}")
            click.echo(f"Invalid images:       {result['invalid']}")
            click.echo("=" * 60)

            if result["invalid"] > 0:
                click.echo("\nInvalid images found:")
                for path, error in result["invalid_list"][:10]:  # Show first 10
                    click.echo(f"  - {path}: {error}")

                if len(result["invalid_list"]) > 10:
                    click.echo(f"  ... and {len(result['invalid_list']) - 10} more")

                # Save report if requested
                if save_report:
                    with open(save_report, "w") as f:
                        f.write("# Invalid Images Report\n")
                        f.write(f"# Source: {source}\n")
                        f.write(f"# Total invalid: {result['invalid']}\n\n")
                        for path, error in result["invalid_list"]:
                            f.write(f"{path}\t{error}\n")
                    click.echo(f"\nInvalid images list saved to: {save_report}")

                click.echo(
                    "\nWarning: Some images are invalid. Consider re-downloading them.",
                    err=True,
                )
            else:
                click.echo("\nAll images are valid! [OK]")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()

    @data_group.command("index-images")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--force-rebuild",
        is_flag=True,
        help="Force rebuild of index even if it exists",
    )
    def index_images(source, force_rebuild):
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
          newspaper-explorer data index-images --source der_tag
          newspaper-explorer data index-images --source der_tag --force-rebuild
        """
        import logging

        from newspaper_explorer.config.base import get_config
        from newspaper_explorer.data.indexing.image_index import ImageIndexer

        # Configure logging
        config = get_config()
        logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

        try:
            click.echo(f"Creating image index for source: {source}")
            if force_rebuild:
                click.echo("Force rebuild: existing index will be overwritten")
            click.echo()

            indexer = ImageIndexer(source)

            # Check if index already exists
            existing_index = indexer.load_index()
            if existing_index is not None and not force_rebuild:
                click.echo(f"Image index already exists with {len(existing_index)} images")
                click.echo(f"Location: {indexer.index_path}")
                click.echo("\nUse --force-rebuild to recreate it")
                return

            click.echo("Building image index...")
            click.echo("- Extracting dimensions from ALTO XML files")
            click.echo("- Loading metadata from METS XML files")
            click.echo("- Scanning image directory\n")

            # Create index
            image_index = indexer.create_index(force_rebuild=force_rebuild)

            click.echo("\n" + "=" * 60)
            click.echo("Image Index Created")
            click.echo("=" * 60)
            click.echo(f"Total images indexed: {len(image_index):,}")
            click.echo(f"Location: {indexer.index_path}")

            # Show statistics
            stats = indexer.get_stats()
            click.echo(f"\nStatistics:")
            click.echo(f"  Total size: {stats['total_size_gb']:.2f} GB")
            click.echo(f"  Average file size: {stats['avg_file_size_mb']:.2f} MB")
            click.echo(
                f"  Year range: {stats['min_year']} - {stats['max_year']} ({stats['years']} years)"
            )

            # Show completeness
            with_real_dims = image_index.filter(image_index["width"].is_not_null())
            with_alto_dims = image_index.filter(image_index["alto_width"].is_not_null())
            with_mets = image_index.filter(image_index["newspaper_title"].is_not_null())

            click.echo(f"\nData completeness:")
            click.echo(
                f"  Images with real dimensions: {len(with_real_dims):,} ({len(with_real_dims) / len(image_index) * 100:.1f}%)"
            )
            click.echo(
                f"  Images with ALTO dimensions: {len(with_alto_dims):,} ({len(with_alto_dims) / len(image_index) * 100:.1f}%)"
            )
            click.echo(
                f"  Images with METS data:      {len(with_mets):,} ({len(with_mets) / len(image_index) * 100:.1f}%)"
            )

            if len(with_alto_dims) < len(image_index):
                missing_dims = len(image_index) - len(with_alto_dims)
                click.echo(
                    f"\n[WARNING] {missing_dims:,} images missing ALTO dimensions (ALTO files not found)"
                )

            if len(with_mets) < len(image_index):
                missing_mets = len(image_index) - len(with_mets)
                click.echo(
                    f"[WARNING] {missing_mets:,} images missing METS metadata (METS files not available)"
                )

            click.echo("\n[OK] Image index created successfully!")
            click.echo("=" * 60)

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()
