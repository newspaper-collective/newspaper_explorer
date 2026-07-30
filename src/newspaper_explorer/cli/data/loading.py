"""CLI commands for data loading and aggregation."""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.cli.utils import errors
from newspaper_explorer.cli.utils import output as out
from newspaper_explorer.cli.utils.options import (
    force_option,
    force_resume_option,
    input_file_option,
    limit_option,
    output_path_option,
    source_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.processing.aggregation import load_and_aggregate_textblocks
from newspaper_explorer.data.utils.sources import get_source_paths, load_source_config


@click.group(name="loading")
def loading_group() -> None:
    """Data loading and aggregation commands."""
    pass


@loading_group.command(name="parse")
@source_option()
@force_resume_option()
@limit_option()
def parse_cmd(source: str, *, force: bool, limit: Optional[int]) -> None:
    """
    Parse XML files to Parquet format.

    Reads ALTO XML files and extracts line-level text data with coordinates
    and metadata from METS files. Output is saved to a compressed Parquet file
    in data/raw/{source}/text/{source}_lines.parquet.

    By default, resumes from where it left off by skipping already processed
    files. Use --force to reprocess all files from scratch.

    \b
    Examples:
      newspaper-explorer data text parse --source der_tag
      newspaper-explorer data text parse --source der_tag --force
      newspaper-explorer data text parse --source der_tag --limit 100
    """
    # Setup logging - WARNING for library, INFO for CLI
    config = get_config()
    logging.basicConfig(level=logging.WARNING, format=config.cli_log_format)

    try:
        out.header(f"PARSE XML: {source.upper()}")

        # Show configuration
        out.section("CONFIGURATION")
        out.key_value("Resume mode", "Disabled" if force else "Enabled")
        if limit:
            out.key_value("File limit", f"{limit:,}")

        out.section("PARSING")
        ingester = DataIngester(source_name=source)

        # Load the source with optional limit
        skip_processed = not force
        df = ingester.load_source(skip_processed=skip_processed, max_files=limit)

        if len(df) == 0:
            out.error("No data loaded. Check if files exist and are valid.")
            return

        # Construct output path from source config
        source_config = load_source_config(source)
        paths = get_source_paths(source_config)
        output_path = paths["parsed_dir"] / "lines.parquet"

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

        out.section("RESULTS")
        out.key_value("Total rows", f"{len(df):,}")
        out.key_value("Columns", len(df.columns))
        out.key_value("File size", f"{file_size_mb:.1f} MB")
        out.key_value("Output file", str(output_path))

        # Show sample
        out.section("SAMPLE DATA")
        click.echo(df.head(3))

        click.echo()
        out.success("Parsing completed successfully!")

    except FileNotFoundError as e:
        errors.handle_error(
            e,
            tip=f"Run 'newspaper-explorer data text download --source {source}' first",
            show_traceback=False,
        )
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@loading_group.command(name="aggregate")
@source_option()
@input_file_option(
    help_text="Input parquet file (default: data/parsed/{source}/lines.parquet)"
)
@output_path_option(
    help_text="Output parquet file (default: data/parsed/{source}/textblocks.parquet)"
)
@force_option()
def aggregate_cmd(
    source: str, input_file: Optional[str], output: Optional[str], *, force: bool
) -> None:
    """
    Aggregate line-level data into text blocks.

    Combines individual text lines from ALTO XML into logical text blocks
    based on text_block_id. Each block represents a coherent text region
    (paragraph, column, etc.) with concatenated text and bounding box.

    Output is automatically saved to data/parsed/{source}/textblocks.parquet
    unless a custom output path is specified.

    \b
    Examples:
      newspaper-explorer data text aggregate --source der_tag
      newspaper-explorer data text aggregate --source der_tag --force
    """
    output_file = output  # Avoid shadowing the 'out' module alias
    try:
        # Load config
        source_config = load_source_config(source)
        source_name = source_config.dataset_name

        # Get paths
        paths = get_source_paths(source_config)

        # Determine input path
        input_path = Path(input_file) if input_file else paths["parsed_dir"] / "lines.parquet"

        # Determine output path
        output_path = (
            Path(output_file) if output_file else paths["parsed_dir"] / "textblocks.parquet"
        )

        # Check input exists
        errors.require_file(
            input_path,
            error_message=f"Input file not found: {input_path}",
            tip=f"Run 'newspaper-explorer data text parse --source {source}' first",
        )

        # Check output - use confirm_overwrite utility
        if not errors.confirm_overwrite(output_path, force=force):
            out.info("Skipping (file exists, use --force to overwrite)")
            return

        out.header(f"AGGREGATE TEXT BLOCKS: {source_name.upper()}")

        # Show configuration
        out.section("CONFIGURATION")
        out.key_value("Input file", str(input_path))
        out.key_value("Output file", str(output_path))

        # Aggregate
        out.section("PROCESSING")
        out.info("Aggregating text blocks...")
        df = load_and_aggregate_textblocks(str(input_path))

        if len(df) == 0:
            out.error("No data after aggregation")
            return

        # Save output
        out.info("Saving aggregated data...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd")

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        out.section("RESULTS")
        out.key_value("Text blocks", f"{len(df):,}")
        out.key_value("Columns", len(df.columns))
        out.key_value("File size", f"{file_size_mb:.1f} MB")
        out.key_value("Output", str(output_path))

        # Show sample
        out.section("SAMPLE DATA")
        click.echo(df.select(["text_block_id", "text", "year", "month", "day"]).head(3))

        click.echo()
        out.success("Aggregation completed successfully!")

    except FileNotFoundError as e:
        errors.handle_error(e, show_traceback=False)
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)
