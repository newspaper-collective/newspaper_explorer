"""CLI commands for data loading and aggregation."""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    force_option,
    input_file_option,
    limit_option,
    output_path_option,
    resume_option,
    source_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.processing.aggregation import load_and_aggregate_textblocks
from newspaper_explorer.data.processing.validation import find_empty_xml_files
from newspaper_explorer.data.utils.sources import get_source_paths, load_source_config

# Display limit for empty file lists
MAX_EMPTY_FILES_TO_DISPLAY = 10


@click.group(name="loading")
def loading_group() -> None:
    """Data loading and aggregation commands."""
    pass


@loading_group.command(name="parse")
@source_option()
@resume_option()
@limit_option()
def parse_cmd(source: str, *, resume: bool, limit: Optional[int]) -> None:
    """
    Parse XML files to Parquet format.

    Reads ALTO XML files and extracts line-level text data with coordinates
    and metadata from METS files. Output is saved to a compressed Parquet file
    in data/raw/{source}/text/{source}_lines.parquet.

    By default, resumes from where it left off by skipping already processed
    files. Use --no-resume to force reprocessing all files.

    \b
    Examples:
      newspaper-explorer data loading parse --source der_tag
      newspaper-explorer data loading parse --source der_tag --no-resume
      newspaper-explorer data loading parse --source der_tag --limit 100
    """
    # Setup logging
    config = get_config()
    logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

    try:
        output.header(f"PARSE XML: {source.upper()}")

        # Show configuration
        output.section("CONFIGURATION")
        output.key_value("Resume mode", "Enabled" if resume else "Disabled")
        if limit:
            output.key_value("File limit", f"{limit:,}")

        output.section("PARSING")
        ingester = DataIngester(source_name=source)

        # Load the source with optional limit
        df = ingester.load_source(skip_processed=resume, max_files=limit)

        if len(df) == 0:
            output.error("No data loaded. Check if files exist and are valid.")
            return

        # Construct output path from source config
        source_config = load_source_config(source)
        paths = get_source_paths(source_config)
        source_name = source_config.dataset_name
        output_path = paths["text_dir"] / f"{source_name}_lines.parquet"

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

        output.section("RESULTS")
        output.key_value("Total rows", f"{len(df):,}")
        output.key_value("Columns", len(df.columns))
        output.key_value("File size", f"{file_size_mb:.1f} MB")
        output.key_value("Output file", str(output_path))

        # Show sample
        output.section("SAMPLE DATA")
        click.echo(df.head(3))

        click.echo()
        output.success("Parsing completed successfully!")

    except FileNotFoundError as e:
        errors.handle_error(
            e,
            tip=f"Run 'newspaper-explorer data download --source {source}' first",
            show_traceback=False,
        )
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@loading_group.command(name="aggregate")
@source_option()
@input_file_option(
    help_text="Input parquet file (default: data/raw/{source}/text/{source}_lines.parquet)"
)
@output_path_option(
    help_text="Output parquet file (default: data/processed/{source}/text/textblocks.parquet)"
)
@force_option()
def aggregate_cmd(
    source: str, input_file: Optional[str], output_file: Optional[str], *, force: bool
) -> None:
    """
    Aggregate line-level data into text blocks.

    Combines individual text lines from ALTO XML into logical text blocks
    based on text_block_id. Each block represents a coherent text region
    (paragraph, column, etc.) with concatenated text and bounding box.

    Output is automatically saved to data/processed/{source}/text/textblocks.parquet
    unless a custom output path is specified.

    \b
    Examples:
      newspaper-explorer data loading aggregate --source der_tag
      newspaper-explorer data loading aggregate --source der_tag --force
    """
    try:
        # Load config
        source_config = load_source_config(source)
        source_name = source_config.dataset_name

        # Get paths
        paths = get_source_paths(source_config)

        # Determine input path
        input_path = (
            Path(input_file) if input_file else paths["text_dir"] / f"{source_name}_lines.parquet"
        )

        # Determine output path
        output_path = (
            Path(output_file)
            if output_file
            else Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
        )

        # Check input exists
        errors.require_file(
            input_path,
            error_message=f"Input file not found: {input_path}",
            tip=f"Run 'newspaper-explorer data loading parse --source {source}' first",
        )

        # Check output - use confirm_overwrite utility
        if not errors.confirm_overwrite(output_path, force=force):
            output.info("Skipping (file exists, use --force to overwrite)")
            return

        output.header(f"AGGREGATE TEXT BLOCKS: {source_name.upper()}")

        # Show configuration
        output.section("CONFIGURATION")
        output.key_value("Input file", str(input_path))
        output.key_value("Output file", str(output_path))

        # Aggregate
        output.section("PROCESSING")
        output.info("Aggregating text blocks...")
        df = load_and_aggregate_textblocks(str(input_path))

        if len(df) == 0:
            output.error("No data after aggregation")
            return

        # Save output
        output.info("Saving aggregated data...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd")

        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        output.section("RESULTS")
        output.key_value("Text blocks", f"{len(df):,}")
        output.key_value("Columns", len(df.columns))
        output.key_value("File size", f"{file_size_mb:.1f} MB")
        output.key_value("Output", str(output_path))

        # Show sample
        output.section("SAMPLE DATA")
        click.echo(df.select(["text_block_id", "text", "year", "month", "day"]).head(3))

        click.echo()
        output.success("Aggregation completed successfully!")

    except FileNotFoundError as e:
        errors.handle_error(e, show_traceback=False)
    except (ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)


@loading_group.command(name="find-empty")
@source_option()
@click.option(
    "--show",
    type=int,
    default=MAX_EMPTY_FILES_TO_DISPLAY,
    help=f"Number of empty files to display (default: {MAX_EMPTY_FILES_TO_DISPLAY})",
)
def find_empty_cmd(source: str, show: int) -> None:
    """
    Find XML files without OCR text content.

    Identifies XML files that were skipped during loading due to
    having no extractable text. Uses the processed parquet file
    to determine which files have text content.

    \b
    Examples:
      newspaper-explorer data loading find-empty --source der_tag
      newspaper-explorer data loading find-empty --source der_tag --show 20
    """
    try:
        output.header(f"FIND EMPTY FILES: {source.upper()}")

        # Use validation utility
        output.section("SCANNING")
        output.info("Analyzing XML files...")
        result = find_empty_xml_files(source)

        # Display results
        output.section("RESULTS")
        output.key_value("Total XML files", f"{result['total_xml_files']:,}")
        output.key_value("Processed files", f"{result['processed_files']:,}")
        output.key_value("Empty files", f"{result['empty_files']:,}")

        if result["empty_files"] > 0:
            empty_pct = (result["empty_files"] / result["total_xml_files"]) * 100
            output.key_value("Empty rate", f"{empty_pct:.2f}%")

            # Show sample
            empty_list = result["empty_file_list"]
            output.section(
                f"EMPTY FILES (showing {min(show, len(empty_list))} of {len(empty_list)})"
            )

            for i, path in enumerate(empty_list[:show], 1):
                output.info(f"{i}. {path}", muted=True)

            if len(empty_list) > show:
                remaining = len(empty_list) - show
                output.info(f"... and {remaining:,} more", muted=True)

            click.echo()
            output.warning("Some files have no extractable text content")
        else:
            click.echo()
            output.success("No empty files found!")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        errors.handle_error(e, show_traceback=True)
