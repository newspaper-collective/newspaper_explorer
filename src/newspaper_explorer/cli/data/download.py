"""CLI commands for downloading and extracting data."""

import logging
from typing import Optional

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    force_option,
    max_workers_option,
    source_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.download.text import ZenodoDownloader
from newspaper_explorer.data.utils.checksums import verify_md5_checksum
from newspaper_explorer.data.utils.sources import load_source_config


@click.group(name="download")
def download_group() -> None:
    """Download and extraction commands."""
    pass


@download_group.command(name="download")
@click.option("--part", type=str, help="Single dataset part to download")
@click.option(
    "--parts",
    type=str,
    help="Comma-separated list of parts (e.g., part1,part2)",
)
@click.option("--all", "download_all", is_flag=True, help="Download all available parts")
@force_option()
@click.option("--no-extract", is_flag=True, help="Download only, skip extraction")
@click.option("--no-fix", is_flag=True, help="Skip automatic error corrections")
@click.option(
    "--parallel",
    is_flag=True,
    help="Download multiple parts in parallel (faster)",
)
@max_workers_option(default=3)
def download_cmd(
    part: str,
    parts: str,
    *,
    download_all: bool,
    force: bool,
    no_extract: bool,
    no_fix: bool,
    parallel: bool,
    max_workers: int,
) -> None:
    """
    Download newspaper data archives from Zenodo.

    Downloads dataset parts and optionally extracts them. Use --all to download
    all available parts, or specify individual parts with --part or --parts.

    \b
    Examples:
      newspaper-explorer data download --part dertag_1900-1902
      newspaper-explorer data download --parts dertag_1900-1902,dertag_1903-1905
      newspaper-explorer data download --all
      newspaper-explorer data download --all --parallel
    """

    # Configure logging so user sees download progress
    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    downloader = ZenodoDownloader()
    part_names: Optional[list[str]] = None
    # Determine which parts to download
    if download_all:
        output.header("DOWNLOAD ALL DATASET PARTS")
    elif part or parts:
        # Combine single part and multiple parts
        part_names = []
        if part:
            part_names.append(part)
        if parts:
            part_names.extend([p.strip() for p in parts.split(",") if p.strip()])
        count = len(part_names)
        part_word = "part" if count == 1 else "parts"
        output.header(f"DOWNLOAD {count} {part_word.upper()}")
    else:
        output.error("Please specify parts to download with --part, --parts, or use --all")
        output.section("AVAILABLE PARTS")
        available_parts = [
            f"{p['name']}{' (' + p.get('size', 'unknown') + ')' if 'size' in p else ''} - {p['years']}"
            for p in downloader.list_available_parts()
        ]
        for part_info in available_parts:
            output.info(part_info, muted=True)
        return

    try:
        if no_extract:
            # Download only
            if part_names is None:
                part_names = [p["name"] for p in downloader.list_available_parts()]

            output.section("DOWNLOADING")
            for part_name in part_names:
                output.info(f"Downloading {part_name}...")
                downloader.download_part(part_name, force_redownload=force)
        else:
            # Download and extract
            output.section("DOWNLOADING & EXTRACTING")
            downloader.download_and_extract(
                part_names=part_names,
                fix_errors=not no_fix,
                parallel=parallel,
                max_workers=max_workers,
            )

        click.echo()
        output.success("Download complete!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@download_group.command(name="unpack")
@source_option()
@click.option("--part", type=str, help="Single dataset part to unpack")
@click.option(
    "--parts",
    type=str,
    help="Comma-separated list of parts (e.g., part1,part2)",
)
@click.option("--fix/--no-fix", default=True, help="Apply automatic error corrections")
def unpack(source: str, part: str, parts: str, *, fix: bool) -> None:
    """
    Extract already downloaded data archives.

    Unpacks compressed archives to the raw data directory. Optionally applies
    automatic error corrections for known data issues.

    \b
    Examples:
      newspaper-explorer data unpack --source der_tag
      newspaper-explorer data unpack --source der_tag --part dertag_1900-1902
      newspaper-explorer data unpack --source der_tag --parts dertag_1900-1902,dertag_1903-1905
      newspaper-explorer data unpack --source der_tag --no-fix
    """

    # Configure logging so user sees extraction progress
    cfg = get_config()
    logging.basicConfig(level=logging.INFO, format=cfg.cli_log_format)

    try:
        # Load source config
        config = load_source_config(source)
        source_name = config.dataset_name

        # Combine single part and multiple parts
        part_names: list[str] = []
        if part:
            part_names.append(part)
        if parts:
            part_names.extend([p.strip() for p in parts.split(",") if p.strip()])

        # If no specific parts, unpack all parts from source
        if not part_names:
            part_names = [p.name for p in config.parts]
            output.header(f"UNPACK {source_name.upper()}")
            output.key_value("Parts to unpack", len(part_names))
        else:
            output.header(f"UNPACK {source_name.upper()}")
            output.key_value("Parts to unpack", len(part_names))

        downloader = ZenodoDownloader()

        output.section("EXTRACTING")
        for i, part_name in enumerate(part_names, 1):
            output.info(f"[{i}/{len(part_names)}] Unpacking {part_name}...")
            downloader.extract_part(part_name, fix_errors=fix)

        click.echo()
        output.success("Unpacking complete!")

    except FileNotFoundError as e:
        errors.handle_error(e)
        output.warning(
            f"Tip: Download first with 'newspaper-explorer data download --source {source}'"
        )
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@download_group.command(name="verify")
@click.option("--part", type=str, help="Single dataset part to verify")
@click.option(
    "--parts",
    type=str,
    help="Comma-separated list of parts (e.g., part1,part2)",
)
@click.confirmation_option(
    prompt="Are you sure you want to verify checksums? This may take a while."
)
def verify(part: str, parts: str) -> None:
    """
    Verify MD5 checksums of downloaded archives.

    Checks that downloaded files match their expected checksums to ensure
    data integrity. Useful for detecting download corruption.

    \b
    Examples:
      newspaper-explorer data verify --part dertag_1900-1902
      newspaper-explorer data verify --parts dertag_1900-1902,dertag_1903-1905
    """
    # Combine single part and multiple parts
    part_names: list[str] = []
    if part:
        part_names.append(part)
    if parts:
        part_names.extend([p.strip() for p in parts.split(",") if p.strip()])

    if not part_names:
        output.error("Please specify parts to verify with --part or --parts")
        return

    output.header("VERIFY CHECKSUMS")
    output.key_value("Parts to verify", len(part_names))

    downloader = ZenodoDownloader()

    output.section("VERIFICATION")
    verified = 0
    failed = 0

    for i, part_name in enumerate(part_names, 1):
        try:
            # Find the part info
            part_info = None
            for part_data in downloader.list_available_parts():
                if part_data["name"] == part_name:
                    part_info = part_data
                    break

            if not part_info:
                output.error(f"[{i}/{len(part_names)}] {part_name}: Part not found")
                failed += 1
                continue

            if "md5" not in part_info:
                output.warning(f"[{i}/{len(part_names)}] {part_name}: No checksum available")
                continue

            # Check if file exists
            filepath = downloader.download_dir / f"{part_name}.tar.gz"
            if not filepath.exists():
                output.error(f"[{i}/{len(part_names)}] {part_name}: File not downloaded")
                failed += 1
                continue

            # Verify checksum
            output.info(f"[{i}/{len(part_names)}] Verifying {part_name}...")
            if verify_md5_checksum(filepath, part_info["md5"]):
                output.success(f"[{i}/{len(part_names)}] {part_name}: Checksum verified")
                verified += 1
            else:
                output.error(f"[{i}/{len(part_names)}] {part_name}: Checksum mismatch")
                failed += 1

        except (OSError, ValueError) as e:
            errors.handle_error(e, show_traceback=True)
            failed += 1

    click.echo()
    output.section("SUMMARY")
    output.key_value("Verified", verified)
    if failed > 0:
        output.key_value("Failed", failed)
        output.error("Some checksums failed verification")
    else:
        output.success("All checksums verified successfully!")
