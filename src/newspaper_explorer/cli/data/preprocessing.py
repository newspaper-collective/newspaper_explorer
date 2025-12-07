"""Preprocessing commands for the data CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
from newspaper_explorer.data.preprocessing.presets import get_preset, list_presets

if TYPE_CHECKING:
    from newspaper_explorer.models.data.metadata import PreprocessingResult


def register_preprocessing_commands(data_group: click.Group) -> None:
    """Register all preprocessing commands to the data group."""

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
        "input_path",
        type=click.Path(exists=True, path_type=Path),
        help="Input parquet file (default: data/processed/{source}/text/textblocks.parquet)",
    )
    @click.option(
        "--steps",
        type=str,
        help="Comma-separated preprocessing steps (e.g., normalize,lowercase,remove-stopwords)",
    )
    @click.option(
        "--pipeline",
        "-p",
        type=click.Choice(
            [
                "minimal",
                "basic",
                "standard",
                "search",
                "analysis",
                "entities",
                "topics",
                "emotions",
                "keywords",
                "embeddings",
                "concepts",
            ],
            case_sensitive=False,
        ),
        help="Use a recommended pipeline preset (alternative to --steps)",
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
        source: str,
        input_path: Path | None,
        steps: str | None,
        pipeline: str | None,
        text_column: str,
        output_column: str,
        sample: int | None,
        batch_size: int,
        num_beams: int,
        num_gpus: int,
        *,
        no_cache: bool,
    ) -> None:
        """
        Preprocess text data with configurable pipeline.

        Apply a series of preprocessing steps to text data. Steps are applied
        in the order specified and results are saved to a new parquet file.

        Use either --steps for custom pipeline or --pipeline for recommended presets.

        \b
        Recommended Pipelines (--pipeline):
          minimal   - Minimal processing, preserves original text
          basic     - Basic OCR cleanup without heavy processing (recommended start)
          standard  - General text analysis, topic modeling, embeddings (default choice)
          search    - Optimized for search, matching, entity extraction
          analysis  - Word frequency, keyword extraction with filtering

        \b
        Available steps (--steps):
          normalize_unicode       - Unicode normalization (NFC + ftfy) - RECOMMENDED FIRST
          normalize_hyphens       - Normalize hyphen variants (U+2E17, U+2013, etc.) - RUN BEFORE DEHYPHENATE
          normalize_whitespace    - Normalize whitespace
          normalize_casing        - Convert to lowercase
          normalize_umlauts       - Normalize umlauts (ae→ä, etc.)
          normalize_long_s        - Normalize historical long s (long-s to s)
          dehyphenate             - Remove line-break hyphens (auto-detects line/block level)
          transnormer             - Transformer-based normalization - HIGH QUALITY
          dta_cab                 - DTA-CAB API normalization - SLOW but best
          remove_diacritics       - Remove diacritics (ä→a, ö→o, ü→u)
          remove_punctuation      - Remove punctuation marks
          remove_numbers          - Remove numeric digits
          remove_stopwords        - Remove German stopwords (requires spaCy)
          only_keep_allowed_chars - Remove OCR artifacts
          lemmatize_spacy         - Lemmatize with spaCy (FAST, context-aware)
          lemmatize_germalemma    - Lemmatize with GermaLemma (SLOW but thorough)
          remove_garbage_words    - Remove OCR garbage words (ssss, jjjj)
          filter_by_total_character_length - Filter by character count
          filter_by_word_count    - Filter by word count
          filter_empty_lines      - Remove empty lines
          filter_number_only_lines - Remove number-only lines
          calculate_quality_metrics - Calculate quality scores
          filter_by_quality_score - Filter by quality score

        \b
        Examples:
          # Use recommended pipeline (easiest)
          newspaper-explorer data preprocess --source der_tag --pipeline standard

          # Test recommended pipeline on sample
          newspaper-explorer data preprocess --source der_tag --pipeline basic --sample 1000

          # Custom pipeline with specific steps
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize_unicode,normalize_long_s,normalize_casing

          # Full cleaning pipeline
          newspaper-explorer data preprocess --source der_tag \\
              --steps normalize_unicode,normalize_long_s,normalize_casing,remove_punctuation,remove_stopwords

          # Transnormer with custom batch size for speed
          newspaper-explorer data preprocess --source der_tag \\
              --steps transnormer --batch-size 128 --num-beams 2

          # Use all 4 GPUs for maximum speed
          newspaper-explorer data preprocess --source der_tag \\
              --steps transnormer --num-gpus 4 --batch-size 128
        """

        # Setup logging
        config = get_config()
        logging.basicConfig(level=logging.INFO, format=config.cli_log_format)

        # Validate options
        if not steps and not pipeline:
            click.echo("Error: Must specify either --steps or --pipeline", err=True)
            click.echo("\nAvailable pipelines:")
            for name, preset_config in list_presets().items():
                click.echo(f"  {name:10s} - {preset_config['description']}")
            raise click.Abort()

        if steps and pipeline:
            click.echo("Error: Cannot specify both --steps and --pipeline", err=True)
            raise click.Abort()

        # Get step list
        if pipeline:
            step_list = get_preset(pipeline)
            click.echo(f"\nUsing recommended pipeline: {pipeline}")
        else:
            step_list = [s.strip() for s in steps.split(",")]  # type: ignore[union-attr]

        click.echo(f"Preprocessing source: {source}")
        step_names = [s if isinstance(s, str) else s["name"] for s in step_list]
        click.echo(f"Steps: {', '.join(step_names)}")

        try:
            # Run preprocessing using TextPreprocessor class
            preprocessor = TextPreprocessor(source=source, text_column=text_column)
            result: PreprocessingResult = preprocessor.run(
                steps=step_list,
                input_path=input_path,
                output_column=output_column,
                sample=sample,
                batch_size=batch_size,
                num_beams=num_beams,
                num_gpus=num_gpus,
                use_cache=not no_cache,
            )

            # Display results
            click.echo("\n" + "=" * 60)
            click.echo("SAMPLE OUTPUT")
            click.echo("=" * 60)

            if result.sample_original:
                click.echo(f"Original:  {result.sample_original[:200]}...")
            if result.sample_processed:
                click.echo(f"Processed: {result.sample_processed[:200]}...")

            click.echo("\n" + "=" * 60)
            click.echo("PREPROCESSING COMPLETE")
            click.echo("=" * 60)
            click.echo(f"Total rows: {result.output_rows:,}")
            click.echo(f"Input column: {text_column}")
            click.echo(f"Output column: {output_column}")
            click.echo(f"Steps applied: {len(step_list)}")

            all_steps = result.metadata.get_all_steps()
            if len(all_steps) > len(step_list):
                click.echo(f"Total steps (including previous): {len(all_steps)}")

            click.echo(f"Processing time: {result.duration_seconds:.1f}s")

            file_size_mb = result.file_size_bytes / (1024 * 1024)
            click.echo(f"\nPreprocessing ID: {result.metadata.preprocessing_id}")
            click.echo(f"\nOutput directory: {result.output_dir}")
            click.echo(f"  Data:     {result.results_path.name}")
            click.echo(f"  Metadata: {result.metadata_path.name}")
            click.echo(f"  File size: {file_size_mb:.1f} MB")

            click.echo("\nTip: Load the preprocessed data with:")
            click.echo("   import polars as pl")
            click.echo(f"   df = pl.read_parquet('{result.results_path}')")

            click.echo("\n" + "=" * 60)

        except FileNotFoundError as e:
            click.echo(f"\nError: {e}", err=True)
            click.echo(f"\nTip: Run 'newspaper-explorer data aggregate --source {source}' first")
            raise click.Abort() from None
        except ValueError as e:
            click.echo(f"\nError: {e}", err=True)
            raise click.Abort() from None
        except ImportError as e:
            click.echo(f"\nError: {e}", err=True)
            click.echo("\nSome preprocessing steps require optional dependencies.")
            click.echo("Install with: pip install -e '.[nlp]'")
            raise click.Abort() from None

    @data_group.command(name="list-pipelines")
    def list_preprocessing_pipelines() -> None:
        """
        List available recommended preprocessing pipelines.

        Shows all preset pipelines with their descriptions and steps.

        Example:
          newspaper-explorer data list-pipelines
        """

        pipelines = list_presets()

        click.echo("\n" + "=" * 80)
        click.echo("RECOMMENDED PREPROCESSING PIPELINES")
        click.echo("=" * 80)

        for name, preset_config in pipelines.items():
            click.echo(f"\n{name.upper()}")
            click.echo("-" * 80)
            click.echo(f"Description: {preset_config['description']}")
            click.echo(f"Use case:    {preset_config['use_case']}")
            click.echo(f"\nSteps ({len(preset_config['steps'])}):")
            for i, step in enumerate(preset_config["steps"], 1):
                click.echo(f"  {i}. {step}")

        click.echo("\n" + "=" * 80)
        click.echo("Usage:")
        click.echo("  newspaper-explorer data preprocess --source SOURCE --pipeline PIPELINE_NAME")
        click.echo("\nExample:")
        click.echo("  newspaper-explorer data preprocess --source der_tag --pipeline standard")
        click.echo("=" * 80 + "\n")
