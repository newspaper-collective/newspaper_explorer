"""
CLI commands for data management.
Handles downloading, extracting, and managing newspaper data.
"""

import csv
import io

import click

from newspaper_explorer.data.download import ZenodoDownloader


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
    downloader = ZenodoDownloader()

    # Determine which parts to download
    if download_all:
        part_names = None  # None means all parts
        click.echo("📦 Downloading ALL dataset parts...")
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
        click.echo(f"📦 Downloading {count} {part_word}...")
    else:
        click.echo("❌ Error: Please specify parts to download with --part, --parts, or use --all")
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

        click.echo("\n✅ Done!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
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
        click.echo("❌ Error: Please specify parts to extract with --part or --parts")
        return

    downloader = ZenodoDownloader()

    try:
        for part_name in part_names:
            click.echo(f"📂 Extracting {part_name}...")
            downloader.extract_part(part_name, fix_errors=fix)

        click.echo("\n✅ Done!")

    except FileNotFoundError as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        click.echo("💡 Tip: Download the file first with 'data download'")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
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
        click.echo("\n📊 Detailed Information:")
        click.echo("=" * 90)
        status_dict = downloader.get_extraction_status()

        for part_name, info in status_dict.items():
            click.echo(f"\n📦 {part_name}")
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

    click.echo("\n📰 Available Dataset Parts")
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
    click.echo("\n💡 Use 'newspaper-explorer data download --part <part-name>' to download")
    click.echo("💡 Or use --parts to download multiple parts at once")


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
        click.echo("❌ Error: Please specify parts to verify with --part or --parts")
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
                click.echo(f"❌ Part '{part_name}' not found")
                continue

            if "md5" not in part_info:
                click.echo(f"⚠️  {part_name}: No checksum available")
                continue

            # Check if file exists
            filepath = downloader.download_dir / f"{part_name}.tar.gz"
            if not filepath.exists():
                click.echo(f"❌ {part_name}: File not downloaded")
                continue

            # Verify checksum
            click.echo(f"🔍 Verifying {part_name}...")
            if downloader._verify_checksum(filepath, part_info["md5"]):
                click.echo(f"✅ {part_name}: Checksum verified!")
            else:
                click.echo(f"❌ {part_name}: Checksum mismatch!")

        except Exception as e:
            click.echo(f"❌ Error verifying {part_name}: {e}", err=True)

    click.echo("\n✅ Verification complete!")
