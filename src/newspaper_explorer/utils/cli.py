"""
CLI commands for data management.
Handles downloading, extracting, and managing newspaper data.
"""

import csv
import io

import click

from newspaper_explorer.data.download import ZenodoDownloader
from newspaper_explorer.utils.sources import (
    get_source_paths,
    list_available_sources,
    load_source_config,
)

# Logging format for CLI - simple messages without timestamps
CLI_LOG_FORMAT = "%(message)s"


@click.group()
def data():
    """Manage newspaper data (download, extract, status)."""
    pass


@data.command()
@click.option("--part", type=str, help="Single dataset part to download")
@click.option("--parts", type=str, help="Comma-separated list of parts (e.g., part1,part2)")
@click.option("--all", "download_all", is_flag=True, help="Download all available parts")
@click.option("--force", is_flag=True, help="Force re-download even if files exist")
@click.option("--no-extract", is_flag=True, help="Download only, skip extraction")
@click.option("--no-fix", is_flag=True, help="Skip automatic error corrections")
@click.option(
    "--parallel",
    is_flag=True,
    help="Download multiple parts in parallel (faster)",
)
@click.option("--max-workers", type=int, default=3, help="Maximum parallel downloads (default: 3)")
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


@data.command()
@click.option("--part", type=str, help="Single dataset part to extract")
@click.option("--parts", type=str, help="Comma-separated list of parts (e.g., part1,part2)")
@click.option("--fix/--no-fix", default=True, help="Apply automatic error corrections")
def extract(part, parts, fix):
    """
    Extract already downloaded data parts.

    \b
    Examples:
      newspaper-explorer data extract --part dertag_1900-1902
      newspaper-explorer data extract --parts dertag_1900-1902,dertag_1903-1905
    """
    import logging

    # Configure logging so user sees extraction progress
    logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

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
        click.echo("Error: Please specify parts to extract with --part or --parts")
        return

    downloader = ZenodoDownloader()

    try:
        for part_name in part_names:
            click.echo(f"Extracting {part_name}...")
            downloader.extract_part(part_name, fix_errors=fix)

        click.echo("\nDone!")

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("Tip: Download the file first with 'data download'")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@data.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def status(verbose):
    """
    Show status of all dataset parts (downloaded/extracted).

    \b
    Examples:
      newspaper-explorer data status
      newspaper-explorer data status --verbose
    """
    downloader = ZenodoDownloader()
    downloader.print_status_summary()

    if verbose:
        click.echo("\nDetailed Information:")
        click.echo("=" * 90)
        status_dict = downloader.get_extraction_status()

        for part_name, info in status_dict.items():
            click.echo(f"\n{part_name}")
            click.echo(f"   Years: {info['years']}")
            if info.get("size"):
                click.echo(f"   Size: {info['size']}")
            if info.get("md5"):
                click.echo(f"   MD5: {info['md5']}")
            if info["downloaded"]:
                click.echo(f"   Download path: {info['download_path']}")
            if info["extracted"]:
                click.echo(f"   Extract path: {info['extract_path']}")


@data.command()
def list():
    """
    List all available dataset parts.

    \b
    Examples:
      newspaper-explorer data list
    """
    downloader = ZenodoDownloader()

    click.echo("\nAvailable Dataset Parts")
    click.echo("=" * 90)
    click.echo(f"{'Name':<25} {'Years':<12} {'Size':<12} {'MD5 Checksum':<35}")
    click.echo("-" * 90)

    for part in downloader.list_available_parts():
        name = part["name"]
        years = part["years"]
        size = part.get("size", "unknown")
        md5 = (
            part.get("md5", "not available")[:32] + "..."
            if part.get("md5") and len(part.get("md5", "")) > 35
            else part.get("md5", "not available")
        )

        click.echo(f"{name:<25} {years:<12} {size:<12} {md5:<35}")

    click.echo("=" * 90)
    click.echo("\nTip: Use 'newspaper-explorer data download --part <part-name>' to download")
    click.echo("Tip: Or use --parts to download multiple parts at once")


@data.command()
@click.option("--part", type=str, help="Single dataset part to verify")
@click.option("--parts", type=str, help="Comma-separated list of parts (e.g., part1,part2)")
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


@data.command()
@click.option("--source", "-s", type=str, required=True, help="Source name (e.g., 'der_tag')")
@click.option("--max-files", type=int, help="Limit number of files to process (for testing)")
@click.option(
    "--workers",
    type=int,
    help="Number of parallel workers (default: CPU count - 1)",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Disable resume functionality (reprocess all files)",
)
def load(source, max_files, workers, no_resume):
    """
    Load ALTO XML files into a Parquet DataFrame.

    Parses ALTO XML newspaper files in parallel and creates a line-level
    DataFrame with text, coordinates, dates, and metadata.

    Configuration-driven: loads source configuration from data/sources/{source}.json

    RESUME FUNCTIONALITY: By default, loads existing parquet and skips already
    processed files. Use --no-resume to reprocess everything.

    \b
    Examples:
      # Load der_tag source with resume (skips already processed files)
      newspaper-explorer data load --source der_tag

      # Force reprocess all files
      newspaper-explorer data load --source der_tag --no-resume

      # Test with 100 files using 4 workers
      newspaper-explorer data load --source der_tag --max-files 100 --workers 4
    """
    import logging

    from newspaper_explorer.data.loading import DataLoader

    # Setup logging with simple format (no timestamps/module names for cleaner CLI output)
    logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

    try:
        # Initialize loader with source
        click.echo(f"\nLoading source: {source}")
        loader = DataLoader(source_name=source, max_workers=workers)

        # Show status before loading
        status = loader.get_loading_status()
        click.echo("\nSource Status:")
        click.echo(f"  Raw directory: {status['raw_dir']}")
        click.echo(f"  XML files found: {status.get('xml_files_count', 0):,}")
        if status.get("parquet_exists"):
            click.echo(f"  Parquet exists: Yes ({status.get('parquet_rows', 0):,} rows)")
        else:
            click.echo("  Parquet exists: No")

        if max_files:
            click.echo(f"\nWarning: Limiting to {max_files} files (testing mode)")

        if no_resume:
            click.echo("Resume disabled - will reprocess all files")
        else:
            click.echo("Resume enabled - will skip already processed files")

        # Load source
        click.echo("\nStarting load process...")
        df = loader.load_source(
            max_files=max_files,
            skip_processed=not no_resume,
        )

        # Show statistics
        click.echo("\n" + "=" * 60)
        click.echo("DATA LOADED SUCCESSFULLY")
        click.echo("=" * 60)
        click.echo(f"Total lines: {len(df):,}")
        if len(df) > 0:
            click.echo(f"Date range: {df['date'].min()} to {df['date'].max()}")
            click.echo(f"Unique files: {df['filename'].n_unique()}")
            click.echo(f"Unique pages: {df['page_id'].n_unique()}")
            click.echo(f"Unique text blocks: {df['text_block_id'].n_unique()}")
            click.echo(f"Average text length: {df['text'].str.len_chars().mean():.1f} chars")

        # Show file info
        config_data = load_source_config(source)
        paths = get_source_paths(config_data)
        output_file = paths["output_file"]
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            click.echo(f"\nSaved to: {output_file} ({size_mb:.1f} MB)")
            click.echo("\nTip: Load the data with:")
            click.echo("   from newspaper_explorer.data.loading import DataLoader")
            click.echo(f"   df = DataLoader.load_parquet('{output_file}')")

    except ValueError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nTip: Use 'newspaper-explorer data sources' to list available sources")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort()


@data.command()
def sources():
    """
    List all available data sources.

    Shows sources that can be loaded from data/sources/ directory.

    \b
    Examples:
      newspaper-explorer data sources
    """
    sources = list_available_sources()

    if not sources:
        click.echo("No sources found in data/sources/ directory")
        return

    click.echo("\nAvailable Data Sources")
    click.echo("=" * 90)

    for source_name in sources:
        try:
            info = load_source_config(source_name)
            metadata = info.get("metadata", {})

            click.echo(f"\n{source_name}")
            click.echo(f"  Description: {info.get('description', 'N/A')}")

            if metadata:
                if "newspaper_title" in metadata:
                    click.echo(f"  Newspaper: {metadata['newspaper_title']}")
                if "years_available" in metadata:
                    click.echo(f"  Years: {metadata['years_available']}")
                if "language" in metadata:
                    click.echo(f"  Language: {metadata['language']}")
                if "location" in metadata:
                    click.echo(f"  Location: {metadata['location']}")

            # Show parts info
            parts = info.get("parts", [])
            if parts:
                total_size_gb = 0.0
                for p in parts:
                    if "size" in p and p["size"] != "unknown":
                        size_parts = p["size"].split()
                        if len(size_parts) == 2:
                            value = float(size_parts[0])
                            unit = size_parts[1].upper()
                            if unit == "GB":
                                total_size_gb += value
                            elif unit == "MB":
                                total_size_gb += value / 1024
                            elif unit == "TB":
                                total_size_gb += value * 1024
                click.echo(
                    f"  Parts: {len(parts)} (total size: ~{total_size_gb:.1f} GB (compressed))"
                )

        except Exception as e:
            click.echo(f"{source_name} (error reading metadata: {e})")

    click.echo("\n" + "=" * 90)
    click.echo("\nTip: Use 'newspaper-explorer data load --source <name>' to load a source")


@data.command()
@click.option("--source", "-s", type=str, help="Specific source to check (optional)")
def load_status(source):
    """
    Show loading status for data sources.

    Displays information about XML files found and parquet output status.

    \b
    Examples:
      # Show status for all sources
      newspaper-explorer data load-status

      # Show status for specific source
      newspaper-explorer data load-status --source der_tag
    """
    from newspaper_explorer.data.loading import DataLoader

    if source:
        # Show status for specific source
        sources_to_check = [source]
    else:
        # Show status for all sources
        sources_to_check = list_available_sources()

    if not sources_to_check:
        click.echo("No sources found")
        return

    click.echo("\nData Loading Status")
    click.echo("=" * 90)

    for source_name in sources_to_check:
        try:
            loader = DataLoader(source_name=source_name)
            status = loader.get_loading_status()

            click.echo(f"\n{source_name}")
            click.echo(f"  Raw directory: {status['raw_dir']}")
            click.echo(f"  XML files: {status.get('xml_files_count', 0):,}")

            if status.get("parquet_exists"):
                click.echo("  Parquet: ✓ Exists")
                click.echo(f"    Rows: {status.get('parquet_rows', 0):,}")
                click.echo(f"    Files processed: {status.get('parquet_files', 0):,}")
                click.echo(f"    Size: {status.get('parquet_size_mb', 0):.1f} MB")
                if "date_range" in status:
                    click.echo(f"    Date range: {status['date_range']}")

                # Show completion percentage
                xml_count = status.get("xml_files_count", 0)
                parquet_files = status.get("parquet_files", 0)
                if xml_count > 0:
                    pct = (parquet_files / xml_count) * 100
                    click.echo(f"    Progress: {pct:.1f}% ({parquet_files}/{xml_count} files)")
            else:
                click.echo("  Parquet: ✗ Not yet processed")

        except Exception as e:
            click.echo(f"\n{source_name}: Error - {e}")

    click.echo("\n" + "=" * 90)
