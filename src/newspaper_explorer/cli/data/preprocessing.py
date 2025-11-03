"""Preprocessing commands for the data CLI."""

import logging
from pathlib import Path

import click

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor


def register_preprocessing_commands(data_group):
    """Register all preprocessing-related commands to the data group."""

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
        help="Input parquet file (default: data/processed/{source}/text/textblocks.parquet)",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output parquet file (default: auto-generated based on steps)",
    )
    @click.option(
        "--steps",
        type=str,
        required=True,
        help="Comma-separated preprocessing steps (e.g., normalize,lowercase,remove-stopwords)",
    )
    @click.option(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column to process (default: text)",
    )
    @click.option(
        "--output-column",
        type=str,
        default="text_processed",
        help="Name for processed text column (default: text_processed)",
    )
    @click.option(
        "--sample",
        type=int,
        help="Process only first N rows (for testing)",
    )
    @click.option(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for transnormer inference (default: 32, try 64/128 for faster processing)",
    )
    @click.option(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for transnormer (default: 4, try 1-2 for faster processing)",
    )
    @click.option(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for transnormer (default: 1, multi-GPU uses parallel processing)",
    )
    @click.option(
        "--no-cache",
        is_flag=True,
        help="Disable caching for transnormer (default: caching enabled for resume capability)",
    )
    def preprocess(
        source,
        input,
        output,
        steps,
        text_column,
        output_column,
        sample,
        batch_size,
        num_beams,
        num_gpus,
        no_cache,
    ):
        """
        Preprocess text data with configurable pipeline.

        Apply a series of preprocessing steps to text data. Steps are applied
        in the order specified and results are saved to a new parquet file.

        \b
        Available steps:
          normalize              - Normalize historical German (ſ→s, ẞ→SS) - FAST
          normalize-transnormer  - Transformer-based normalization - HIGH QUALITY
          normalize-dtacab       - DTA-CAB API normalization - SLOW but best
          lowercase              - Convert to lowercase
          remove-punctuation     - Remove punctuation marks
          remove-numbers         - Remove numeric digits
          remove-stopwords       - Remove German stopwords (requires spaCy)
          dehyphenate            - Remove line-break hyphens (requires pyphen)
          lemmatize-spacy        - Lemmatize with spaCy (FAST, context-aware)
          lemmatize              - Lemmatize with GermaLemma (SLOW but thorough)

        \b
        Examples:
          # Basic normalization
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize,lowercase

          # Full cleaning pipeline
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize,lowercase,remove-punctuation,remove-stopwords

          # Test on sample
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize,lowercase --sample 1000

          # Transnormer with custom batch size for speed
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize-transnormer --batch-size 128 --num-beams 2
          
          # Use all 4 GPUs for maximum speed
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize-transnormer --num-gpus 4 --batch-size 128
        """
        import polars as pl

        from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
        from newspaper_explorer.utils.sources import get_source_paths, load_source_config

        # Setup logging
        config = get_config()
        logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

        try:
            # Parse steps
            step_list = [s.strip() for s in steps.split(",")]

            click.echo(f"\nPreprocessing source: {source}")
            click.echo(f"Steps: {', '.join(step_list)}")

            # Get input path
            if input:
                input_path = Path(input)
            else:
                # Auto-detect from source
                config_data = load_source_config(source)
                source_name = config_data.dataset_name
                input_path = (
                    Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
                )

            if not input_path.exists():
                click.echo(f"Error: Input file not found: {input_path}", err=True)
                click.echo(
                    f"\nTip: Run 'newspaper-explorer data aggregate --source {source}' first"
                )
                raise click.Abort()

            # Determine output path
            if output:
                output_path = Path(output)
            else:
                # Auto-generate based on steps
                config_data = load_source_config(source)
                source_name = config_data.dataset_name

                # Create descriptive suffix
                if len(step_list) <= 2:
                    suffix = "_".join(step_list)
                else:
                    suffix = "processed"

                output_path = (
                    Path("data")
                    / "processed"
                    / source_name
                    / "text"
                    / f"textblocks_{suffix}.parquet"
                )

            click.echo(f"\nInput:  {input_path}")
            click.echo(f"Output: {output_path}")

            # Load data
            click.echo(f"\nLoading data...")
            df = pl.read_parquet(input_path)
            click.echo(f"Loaded {len(df):,} rows")

            # Sample if requested
            if sample and sample < len(df):
                click.echo(f"Sampling first {sample:,} rows for testing")
                df = df.head(sample)

            # Check text column exists
            if text_column not in df.columns:
                click.echo(f"Error: Text column '{text_column}' not found", err=True)
                click.echo(f"Available columns: {', '.join(df.columns)}")
                raise click.Abort()

            # Run preprocessing
            click.echo("\nStarting preprocessing pipeline...")
            preprocessor = TextPreprocessor(text_column=text_column)

            df = preprocessor.pipeline(
                df,
                steps=step_list,
                output_column=output_column,
                batch_size=batch_size,
                num_beams=num_beams,
                num_gpus=num_gpus,
                use_cache=not no_cache,
            )

            # Show sample output
            click.echo("\n" + "=" * 60)
            click.echo("SAMPLE OUTPUT")
            click.echo("=" * 60)

            sample_row = df.head(1).to_dicts()[0]
            original_text = sample_row.get(text_column, "")
            processed_text = sample_row.get(output_column, "")

            click.echo(f"Original:  {original_text[:200]}...")
            click.echo(f"Processed: {processed_text[:200]}...")

            # Save output
            click.echo(f"\nSaving to: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path, compression="zstd")

            # Show statistics
            click.echo("\n" + "=" * 60)
            click.echo("PREPROCESSING COMPLETE")
            click.echo("=" * 60)
            click.echo(f"Total rows: {len(df):,}")
            click.echo(f"Input column: {text_column}")
            click.echo(f"Output column: {output_column}")
            click.echo(f"Steps applied: {len(step_list)}")

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            click.echo(f"\nFile saved: {output_path}")
            click.echo(f"File size: {file_size_mb:.1f} MB")

            click.echo("\nTip: Load the preprocessed data with:")
            click.echo("   import polars as pl")
            click.echo(f"   df = pl.read_parquet('{output_path}')")

            click.echo("\n" + "=" * 60)

        except ValueError as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort()
        except ImportError as e:
            click.echo(f"\nError: {e}", err=True)
            click.echo("\nSome preprocessing steps require optional dependencies.")
            click.echo("Install with: pip install -e '.[nlp]'")
            raise click.Abort()
        except Exception as e:
            click.echo(f"\nError: {e}", err=True)
            import traceback

            traceback.print_exc()
            raise click.Abort()
