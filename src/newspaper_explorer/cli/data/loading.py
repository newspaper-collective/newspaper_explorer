"""Loading commands for the data CLI."""

import logging
from pathlib import Path

import click
from natsort import natsorted

from .common import CLI_LOG_FORMAT


def register_loading_commands(data_group):
    """Register all loading-related commands to the data group."""

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--resume/--no-resume",
        default=True,
        help="Skip already processed files (default: True)",
    )
    @click.option(
        "--limit",
        type=int,
        help="Process only N files (for testing)",
    )
    def parse(source, resume, limit):
        """
        Parse XML files to Parquet format.

        Reads ALTO XML files and extracts line-level text data with coordinates
        and metadata from METS files. Output is saved to a compressed Parquet file
        in data/raw/{source}/text/{source}_lines.parquet.

        By default, resumes from where it left off by skipping already processed
        files. Use --no-resume to force reprocessing all files.

        \b
        Examples:
          newspaper-explorer data parse --source der_tag
          newspaper-explorer data parse --source der_tag --no-resume
          newspaper-explorer data parse --source der_tag --limit 100
        """
        from newspaper_explorer.data.loading.loader import DataLoader

        # Setup logging with simple format
        logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)

        try:
            click.echo(f"\nParsing source: {source}")
            loader = DataLoader(source_name=source)

            # Load the source with optional limit
            df = loader.load_source(skip_processed=resume, max_files=limit)

            if df is None or len(df) == 0:
                click.echo("\nNo data loaded. Check if files exist and are valid.")
                raise click.Abort()

            # Show statistics
            click.echo("\n" + "=" * 60)
            click.echo("PARSING COMPLETE")
            click.echo("=" * 60)
            click.echo(f"Total rows: {len(df):,}")

            # Construct output path from source config
            from newspaper_explorer.utils.sources import get_source_paths, load_source_config

            config = load_source_config(source)
            paths = get_source_paths(config)
            source_name = config.get("dataset_name", source)
            output_path = paths["text_dir"] / f"{source_name}_lines.parquet"
            click.echo(f"Output: {output_path}")

            # Show sample
            click.echo("\nSample data:")
            click.echo(df.head(3))

            click.echo("\n" + "=" * 60)

        except FileNotFoundError as e:
            click.echo(f"\nError: {e}", err=True)
            click.echo(f"\nTip: Run 'newspaper-explorer data download --source {source}' first")
            raise click.Abort()
        except Exception as e:
            click.echo(f"\nError during parsing: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    @data_group.command()
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--input",
        "-i",
        type=click.Path(exists=True),
        help="Input parquet file (default: data/raw/{source}/text/{source}_lines.parquet)",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output parquet file (default: data/processed/{source}/text/textblocks.parquet)",
    )
    @click.option(
        "--force/--no-force",
        default=False,
        help="Overwrite existing output file",
    )
    def aggregate(source, input, output, force):
        """
        Aggregate line-level data into text blocks.

        Combines individual text lines from ALTO XML into logical text blocks
        based on text_block_id. Each block represents a coherent text region
        (paragraph, column, etc.) with concatenated text and bounding box.

        Output is automatically saved to data/processed/{source}/text/textblocks.parquet
        unless a custom output path is specified.

        \b
        Examples:
          newspaper-explorer data aggregate --source der_tag
          newspaper-explorer data aggregate --source der_tag --force
        """
        import polars as pl

        from newspaper_explorer.data.loading.aggregation import load_and_aggregate_textblocks
        from newspaper_explorer.utils.sources import get_source_paths, load_source_config

        try:
            # Load config
            config = load_source_config(source)
            source_name = config.get("dataset_name", source)

            # Get paths
            paths = get_source_paths(config)

            # Determine input path
            if input:
                input_path = Path(input)
            else:
                input_path = paths["text_dir"] / f"{source_name}_lines.parquet"

            # Determine output path
            if output:
                output_path = Path(output)
            else:
                output_path = (
                    Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
                )

            # Check input exists
            if not input_path.exists():
                click.echo(f"Error: Input file not found: {input_path}", err=True)
                click.echo(f"\nTip: Run 'newspaper-explorer data parse --source {source}' first")
                raise click.Abort()

            # Check output
            if output_path.exists() and not force:
                click.echo(f"Output file already exists: {output_path}")
                click.echo("Use --force to overwrite")
                raise click.Abort()

            click.echo(f"\nAggregating text blocks for: {source_name}")
            click.echo(f"Input:  {input_path}")
            click.echo(f"Output: {output_path}")

            # Aggregate
            df = load_and_aggregate_textblocks(str(input_path))

            if df is None or len(df) == 0:
                click.echo("\nError: No data after aggregation", err=True)
                raise click.Abort()

            # Save output
            click.echo(f"\nSaving aggregated data...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path, compression="zstd")

            # Show statistics
            click.echo("\n" + "=" * 60)
            click.echo("AGGREGATION COMPLETE")
            click.echo("=" * 60)
            click.echo(f"Text blocks: {len(df):,}")
            click.echo(f"Output: {output_path}")

            # File size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            click.echo(f"File size: {file_size_mb:.1f} MB")

            # Show sample
            click.echo("\nSample data:")
            click.echo(df.select(["text_block_id", "text", "date"]).head(3))

            click.echo("\n" + "=" * 60)

        except FileNotFoundError as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()
        except Exception as e:
            click.echo(f"\nError during aggregation: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()

    @data_group.command("find-empty")
    @click.option(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source name (e.g., der_tag)",
    )
    @click.option(
        "--show",
        type=int,
        default=10,
        help="Number of empty files to display (default: 10)",
    )
    def find_empty(source, show):
        """
        Find XML files without OCR text content.

        Identifies XML files that were skipped during loading due to
        having no extractable text. Uses the processed parquet file
        to determine which files have text content.

        \b
        Examples:
          newspaper-explorer data find-empty --source der_tag
          newspaper-explorer data find-empty --source der_tag --show 20
        """
        from newspaper_explorer.data.utils.validation import find_empty_xml_files

        try:
            # Use validation utility
            result = find_empty_xml_files(source)

            # Display results
            click.echo("\n" + "=" * 60)
            click.echo("EMPTY FILE SCAN RESULTS")
            click.echo("=" * 60)
            click.echo(f"Total XML files: {result['total_xml_files']:,}")
            click.echo(f"Processed files: {result['processed_files']:,}")
            click.echo(f"Empty files: {result['empty_files']:,}")

            if result["empty_files"] > 0:
                empty_pct = (result["empty_files"] / result["total_xml_files"]) * 100
                click.echo(f"Empty rate: {empty_pct:.2f}%")

                # Show sample
                empty_list = result["empty_file_list"]
                click.echo(f"\nShowing first {min(show, len(empty_list))} empty files:")
                for path in empty_list[:show]:
                    click.echo(f"  {path}")

                if len(empty_list) > show:
                    remaining = len(empty_list) - show
                    click.echo(f"  ... and {remaining:,} more")
            else:
                click.echo("\n✓ No empty files found!")

            click.echo("\n" + "=" * 60)

        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()
