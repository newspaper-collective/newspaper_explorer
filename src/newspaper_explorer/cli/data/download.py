"""
CLI commands for downloading and extracting data.
"""

import csv
import io

import click

from newspaper_explorer.data.download.text import ZenodoDownloader

from .common import CLI_LOG_FORMAT


def register_download_commands(data_group):
    """Register download commands to the data group."""

    @data_group.command()
    @click.option("--part", type=str, help="Single dataset part to download")
    @click.option(
        "--parts",
        type=str,
        help="Comma-separated list of parts (e.g., part1,part2)",
    )
    @click.option("--all", "download_all", is_flag=True, help="Download all available parts")
    @click.option("--force", is_flag=True, help="Force re-download even if files exist")
    @click.option("--no-extract", is_flag=True, help="Download only, skip extraction")
    @click.option("--no-fix", is_flag=True, help="Skip automatic error corrections")
    @click.option(
        "--parallel",
        is_flag=True,
        help="Download multiple parts in parallel (faster)",
    )
    @click.option(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel downloads (default: 3)",
    )
    def download(part, parts, download_all, force, no_extract, no_fix, parallel, max_workers):
        """
        Download newspaper data parts.

        \b
        Examples:
          newspaper-explorer data download --part dertag_1900-1902
          newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905
          newspaper-explorer data download --all
        """
        import logging

        # Configure logging so user sees download progress
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

        downloader = ZenodoDownloader()

        # Determine which parts to download
        if download_all:
            part_names = None  # None means all parts
            click.echo("Downloading ALL dataset parts...")
        elif part or parts:
            # Combine single part and multiple parts
            part_names = []
            if part:
                part_names.append(part)
            if parts:
                # Parse comma-separated list, supporting quoted values
                reader = csv.reader(io.StringIO(parts))
                for row in reader:
                    part_names.extend([p.strip() for p in row if p.strip()])
            count = len(part_names)
            part_word = "part" if count == 1 else "parts"
            click.echo(f"Downloading {count} {part_word}...")
        else:
            click.echo("Error: Please specify parts to download with --part, --parts, or use --all")
            click.echo("\nAvailable parts:")
            for part_info in downloader.list_available_parts():
                size_info = f" ({part_info.get('size', 'unknown')})" if "size" in part_info else ""
                click.echo(f"  • {part_info['name']}{size_info} - {part_info['years']}")
            return

        try:
            if no_extract:
                # Download only
                if part_names is None:
                    part_names = [p["name"] for p in downloader.list_available_parts()]

                for part_name in part_names:
                    downloader.download_part(part_name, force_redownload=force)
            else:
                # Download and extract
                downloader.download_and_extract(
                    part_names=part_names,
                    fix_errors=not no_fix,
                    parallel=parallel,
                    max_workers=max_workers,
                )

            click.echo("\nDone!")

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()

    @data_group.command()
    @click.option("--source", "-s", type=str, required=True, help="Source name (e.g., der_tag)")
    @click.option("--part", type=str, help="Single dataset part to unpack")
    @click.option(
        "--parts",
        type=str,
        help="Comma-separated list of parts (e.g., part1,part2)",
    )
    @click.option("--fix/--no-fix", default=True, help="Apply automatic error corrections")
    def unpack(source, part, parts, fix):
        """
        Unpack (extract) already downloaded data archives.

        \b
        Examples:
          newspaper-explorer data unpack --source der_tag
          newspaper-explorer data unpack --source der_tag --part dertag_1900-1902
          newspaper-explorer data unpack --source der_tag --parts dertag_1900-1902,dertag_1903-1905
        """
        import logging

        from newspaper_explorer.utils.sources import load_source_config

        # Configure logging so user sees extraction progress
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

        try:
            # Load source config
            config = load_source_config(source)
            source_name = config.get("dataset_name", source)

            # Combine single part and multiple parts
            part_names = []
            if part:
                part_names.append(part)
            if parts:
                # Parse comma-separated list, supporting quoted values
                reader = csv.reader(io.StringIO(parts))
                for row in reader:
                    part_names.extend([p.strip() for p in row if p.strip()])

            # If no specific parts, unpack all parts from source
            if not part_names:
                all_parts = config.get("parts", [])
                part_names = [p.get("name") for p in all_parts if p.get("name")]
                click.echo(f"Unpacking all {len(part_names)} parts for {source_name}...")

            downloader = ZenodoDownloader(source_name)

            for part_name in part_names:
                click.echo(f"Unpacking {part_name}...")
                downloader.extract_part(part_name, fix_errors=fix)

            click.echo("\n✓ Unpacking complete!")

        except FileNotFoundError as e:
            click.echo(f"\nError: {e}", err=True)
            click.echo(
                f"Tip: Download first with 'newspaper-explorer data download --source {source}'"
            )
            raise click.Abort()
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()

    @data_group.command()
    @click.option("--part", type=str, help="Single dataset part to verify")
    @click.option(
        "--parts",
        type=str,
        help="Comma-separated list of parts (e.g., part1,part2)",
    )
    @click.confirmation_option(
        prompt="Are you sure you want to verify checksums? This may take a while."
    )
    def verify(part, parts):
        """
        Verify MD5 checksums of downloaded files.

        \b
        Examples:
          newspaper-explorer data verify --part dertag_1900-1902
          newspaper-explorer data verify --parts dertag_1900-1902,dertag_1903-1905
        """
        # Combine single part and multiple parts
        part_names = []
        if part:
            part_names.append(part)
        if parts:
            # Parse comma-separated list, supporting quoted values
            reader = csv.reader(io.StringIO(parts))
            for row in reader:
                part_names.extend([p.strip() for p in row if p.strip()])

        if not part_names:
            click.echo("Error: Please specify parts to verify with --part or --parts")
            return

        downloader = ZenodoDownloader()

        for part_name in part_names:
            try:
                # Find the part info
                part_info = None
                for part in downloader.list_available_parts():
                    if part["name"] == part_name:
                        part_info = part
                        break

                if not part_info:
                    click.echo(f"Part '{part_name}' not found")
                    continue

                if "md5" not in part_info:
                    click.echo(f"Warning: {part_name}: No checksum available")
                    continue

                # Check if file exists
                filepath = downloader.download_dir / f"{part_name}.tar.gz"
                if not filepath.exists():
                    click.echo(f"Error: {part_name}: File not downloaded")
                    continue

                # Verify checksum
                click.echo(f"Verifying {part_name}...")
                if downloader._verify_checksum(filepath, part_info["md5"]):
                    click.echo(f"Success: {part_name}: Checksum verified!")
                else:
                    click.echo(f"Error: {part_name}: Checksum mismatch!")

            except Exception as e:
                click.echo(f"Error verifying {part_name}: {e}", err=True)

        click.echo("\nVerification complete!")
