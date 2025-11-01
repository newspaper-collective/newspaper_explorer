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
    def download_images(source, max_workers, max_retries):
        """
        Download high-resolution newspaper page images from METS XML.

        Images are stored in data/raw/{source}/images/ with the same
        directory structure as the XML files (year/month/day).

        \b
        Examples:
          newspaper-explorer data download-images --source der_tag
          newspaper-explorer data download-images --source der_tag --max-workers 16
        """
        import logging

        from newspaper_explorer.data.download.images import ImageDownloader

        # Configure logging so user sees download progress
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

        try:
            click.echo(f"Downloading images for source: {source}")
            click.echo(f"Using {max_workers} parallel workers\n")

            downloader = ImageDownloader(
                source_name=source, max_workers=max_workers, max_retries=max_retries
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
