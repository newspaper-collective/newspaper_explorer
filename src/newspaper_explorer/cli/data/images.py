"""
CLI commands for downloading images.
"""

import click

from .common import CLI_LOG_FORMAT


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
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

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
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

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
                click.echo("\nAll images are valid! ✓")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()
