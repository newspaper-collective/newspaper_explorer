"""Preprocessing commands for the data CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    batch_size_option,
    input_file_option,
    limit_option,
    source_option,
    text_column_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
from newspaper_explorer.data.preprocessing.presets import get_preset, list_presets

if TYPE_CHECKING:
    from newspaper_explorer.data.preprocessing.presets import StepType
    from newspaper_explorer.models.data.metadata import PreprocessingResult


@click.group(name="preprocessing")
def preprocessing_group() -> None:
    """Text preprocessing commands."""
    pass


@preprocessing_group.command(name="preprocess")
@source_option()
@input_file_option(
    help_text="Input parquet file (default: data/processed/{source}/text/textblocks.parquet)"
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
@text_column_option()
@click.option(
    "--output-column",
    type=str,
    default="text_processed",
    help="Name for processed text column (default: text_processed)",
)
@limit_option(help_text="Limit number of rows to process (for testing)")
@batch_size_option(default=32)
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
def preprocess_cmd(
    source: str,
    input_file: str | None,
    steps: str | None,
    pipeline: str | None,
    text_column: str,
    output_column: str,
    limit: int | None,
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
          newspaper-explorer data preprocess --source der_tag --pipeline basic --limit 1000

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
        output.error("Must specify either --steps or --pipeline")
        output.section("AVAILABLE PIPELINES")
        for name, preset_config in list_presets().items():
            output.key_value(name, str(preset_config["description"]))
        click.echo()
        output.info(
            "Tip: Use --pipeline <name> or view all: newspaper-explorer data preprocessing list-pipelines",
            muted=True,
        )
        raise click.Abort()

    if steps and pipeline:
        output.error("Cannot specify both --steps and --pipeline")
        raise click.Abort()

    # Get step list
    step_list: list[StepType]
    if pipeline:
        step_list = get_preset(pipeline)
        output.info(f"Using recommended pipeline: {pipeline}")
    else:
        step_list = [s.strip() for s in steps.split(",")]  # type: ignore[union-attr]

    output.header(f"PREPROCESS TEXT: {source.upper()}")
    step_names = [s if isinstance(s, str) else s["name"] for s in step_list]
    output.key_value("Steps", f"{len(step_list)} steps")
    output.info(f"Pipeline: {', '.join(step_names)}", muted=True)

    try:
        # Run preprocessing using TextPreprocessor class
        preprocessor = TextPreprocessor(source=source, text_column=text_column)
        result: PreprocessingResult = preprocessor.run(
            steps=step_list,
            input_path=Path(input_file) if input_file else None,
            output_column=output_column,
            sample=limit,
            batch_size=batch_size,
            num_beams=num_beams,
            num_gpus=num_gpus,
            use_cache=not no_cache,
        )

        # Display sample output
        if result.sample_original or result.sample_processed:
            output.section("SAMPLE OUTPUT")
            if result.sample_original:
                output.info(f"Original:  {result.sample_original[:200]}...", muted=True)
            if result.sample_processed:
                output.info(f"Processed: {result.sample_processed[:200]}...", muted=True)

        # Display results
        output.section("PREPROCESSING COMPLETE")
        output.key_value("Total rows", f"{result.output_rows:,}")
        output.key_value("Input column", text_column)
        output.key_value("Output column", output_column)
        output.key_value("Steps applied", len(step_list))

        all_steps = result.metadata.get_all_steps()
        if len(all_steps) > len(step_list):
            output.key_value("Total steps (including previous)", len(all_steps))

        output.key_value("Processing time", f"{result.duration_seconds:.1f}s")

        file_size_mb = result.file_size_bytes / (1024 * 1024)
        output.section("OUTPUT FILES")
        if result.metadata.preprocessing_id:
            output.key_value("Preprocessing ID", result.metadata.preprocessing_id)
        output.key_value("Directory", str(result.output_dir))
        output.key_value("Data file", result.results_path.name, indent=2)
        output.key_value("Metadata file", result.metadata_path.name, indent=2)
        output.key_value("File size", f"{file_size_mb:.1f} MB", indent=2)

        output.section("USAGE")
        output.info("Load the preprocessed data with:", muted=True)
        output.code_block(["import polars as pl", f"df = pl.read_parquet('{result.results_path}')"])

        click.echo()
        output.success("Preprocessing completed successfully!")

    except FileNotFoundError as e:
        errors.handle_error(
            e, tip=f"Run 'newspaper-explorer data loading aggregate --source {source}' first"
        )
    except ValueError as e:
        errors.handle_error(e, show_traceback=True)
    except ImportError as e:
        errors.handle_error(
            e,
            tip="Some preprocessing steps require optional dependencies. Install with: pip install -e '.[nlp]'",
        )


@preprocessing_group.command(name="list-pipelines")
def list_pipelines_cmd() -> None:
    """
    List available recommended preprocessing pipelines.

    Shows all preset pipelines with their descriptions and steps.

    \b
    Example:
      newspaper-explorer data preprocessing list-pipelines
    """

    pipelines = list_presets()

    output.header("RECOMMENDED PREPROCESSING PIPELINES")

    for name, preset_config in pipelines.items():
        output.section(name.upper())
        output.key_value("Description", str(preset_config["description"]))
        output.key_value("Use case", str(preset_config["use_case"]))
        output.key_value("Steps", f"{len(preset_config['steps'])} steps")
        output.divider()

        # Show steps as a list
        steps = [f"{i}. {step}" for i, step in enumerate(preset_config["steps"], 1)]
        for step in steps:
            output.info(step, muted=True)

        click.echo()

    output.section("USAGE")
    output.code_block(
        [
            "newspaper-explorer data preprocessing preprocess --source SOURCE --pipeline PIPELINE_NAME"
        ]
    )

    output.section("EXAMPLE")
    output.code_block(
        ["newspaper-explorer data preprocessing preprocess --source der_tag --pipeline standard"]
    )
