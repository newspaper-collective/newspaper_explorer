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
    input_type_option,
    limit_option,
    source_option,
    text_column_option,
)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
from newspaper_explorer.data.preprocessing.presets import get_extra_steps, get_preset, list_presets
from newspaper_explorer.data.utils.sources import load_source_config

if TYPE_CHECKING:
    from newspaper_explorer.data.preprocessing.presets import StepType
    from newspaper_explorer.models.data.metadata import PreprocessingResult


@click.group(name="preprocessing")
def preprocessing_group() -> None:
    """Text preprocessing commands."""
    pass


@preprocessing_group.command(name="preprocess")
@source_option()
@input_type_option()
@input_file_option(help_text="Input parquet file (overrides --input-type auto-resolution)")
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
            "advanced",
            "full",
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
    input_type: str,
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
          minimal    - Minimal processing, preserves original text
          basic      - Basic OCR cleanup without heavy processing (recommended start)
          standard   - General text analysis, topic modeling, embeddings (default choice)
          advanced   - Aggressive cleanup with quality filtering
          full       - Maximum cleaning for bag-of-words analysis
          entities   - Optimized for NER (preserves case and punctuation)
          topics     - Optimized for topic modeling (lowercase, no stopwords)
          emotions   - Optimized for sentiment analysis (preserves emphasis)
          keywords   - Optimized for TF-IDF and keyword extraction
          embeddings - Optimized for generating text embeddings
          concepts   - Optimized for concept extraction

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
            "Tip: Use --pipeline <name> or view all: newspaper-explorer data list-pipelines",
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
        # Resolve input path based on --input-type (unless --input-file overrides)
        results_filename = f"{input_type}.parquet"
        resolved_input = Path(input_file) if input_file else _resolve_input_path(source, input_type)

        output.key_value("Input type", input_type)
        output.key_value("Input file", str(resolved_input))

        # Run preprocessing using TextPreprocessor class
        preprocessor = TextPreprocessor(source=source, text_column=text_column)
        result: PreprocessingResult = preprocessor.run(
            steps=step_list,
            input_path=resolved_input,
            output_column=output_column,
            sample=limit,
            batch_size=batch_size,
            num_beams=num_beams,
            num_gpus=num_gpus,
            use_cache=not no_cache,
            results_filename=results_filename,
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
            e, tip=f"Run 'newspaper-explorer data text aggregate --source {source}' first"
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
      newspaper-explorer data list-pipelines
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
        ["newspaper-explorer data preprocess --source SOURCE --pipeline PIPELINE_NAME"]
    )

    output.section("EXAMPLE")
    output.code_block(["newspaper-explorer data preprocess --source der_tag --pipeline standard"])


# Analysis presets that add steps beyond standard
_ANALYSIS_PRESETS = ["topics", "keywords", "concepts"]


def _resolve_input_path(source: str, input_type: str) -> Path:
    """Resolve input parquet path for a source and input type."""
    config = get_config()
    source_config = load_source_config(source)
    dataset = source_config.dataset_name
    if input_type == "lines":
        return config.parsed_dir / dataset / "lines.parquet"
    return config.parsed_dir / dataset / "textblocks.parquet"


@preprocessing_group.command(name="preprocess-all")
@source_option()
@click.option(
    "--input-types",
    type=str,
    default="textblocks,lines",
    help="Comma-separated input types to process (default: textblocks,lines)",
    show_default=True,
)
@click.option(
    "--presets",
    type=str,
    default=None,
    help=(
        "Comma-separated analysis presets to run on top of standard "
        f"(default: {','.join(_ANALYSIS_PRESETS)})"
    ),
)
@limit_option(help_text="Limit number of rows per run (for testing)")
@batch_size_option(default=32)
@click.option("--num-beams", type=int, default=4, help="Beams for transnormer (default: 4)")
@click.option("--num-gpus", type=int, default=1, help="GPUs for transnormer (default: 1)")
@click.option("--no-cache", is_flag=True, help="Disable transnormer caching")
def preprocess_all_cmd(
    source: str,
    input_types: str,
    presets: str | None,
    limit: int | None,
    batch_size: int,
    num_beams: int,
    num_gpus: int,
    *,
    no_cache: bool,
) -> None:
    """
    Run standard + method-specific preprocessing for all input types.

    First runs the 'standard' preset, then runs only the extra steps
    from each analysis preset (topics, keywords, concepts) on top of
    the standard output. Presets identical to standard (entities,
    emotions, embeddings) are skipped.

    Processes both textblocks and lines by default.

    \b
    Examples:
      # Full batch: standard + all analysis presets, both input types
      newspaper-explorer data preprocess-all --source der_tag

      # Test with small sample first
      newspaper-explorer data preprocess-all --source der_tag --limit 10000

      # Only textblocks, only topics preset
      newspaper-explorer data preprocess-all --source der_tag \\
          --input-types textblocks --presets topics

      # Only lines
      newspaper-explorer data preprocess-all --source der_tag --input-types lines
    """
    logging.basicConfig(level=logging.INFO, format=get_config().cli_log_format)

    # Parse options
    type_list = [t.strip() for t in input_types.split(",")]
    preset_list = [p.strip() for p in presets.split(",")] if presets else list(_ANALYSIS_PRESETS)

    # Validate presets have extra steps
    preset_extras: dict[str, list] = {}
    for preset_name in preset_list:
        extra = get_extra_steps(preset_name, base_preset="standard")
        if extra:
            preset_extras[preset_name] = extra
        else:
            output.info(f"Skipping '{preset_name}' (identical to standard)", muted=True)

    # Plan summary
    total_runs = len(type_list) * (1 + len(preset_extras))
    output.header("BATCH PREPROCESSING")
    output.key_value("Source", source)
    output.key_value("Input types", ", ".join(type_list))
    output.key_value("Base preset", "standard")
    output.key_value(
        "Analysis presets", ", ".join(preset_extras.keys()) if preset_extras else "(none)"
    )
    output.key_value("Total runs", total_runs)
    if limit:
        output.key_value("Row limit", f"{limit:,}")
    click.echo()

    completed: list[str] = []
    failed: list[str] = []

    for input_type in type_list:
        results_filename = f"{input_type}.parquet"
        input_path = _resolve_input_path(source, input_type)

        # --- Run standard preset ---
        run_label = f"standard/{input_type}"
        output.section(f"RUN: {run_label}")
        output.key_value("Input", str(input_path))

        try:
            preprocessor = TextPreprocessor(source=source, text_column="text")
            standard_result: PreprocessingResult = preprocessor.run(
                steps=get_preset("standard"),
                input_path=input_path,
                sample=limit,
                batch_size=batch_size,
                num_beams=num_beams,
                num_gpus=num_gpus,
                use_cache=not no_cache,
                results_filename=results_filename,
                preprocessing_id=f"standard_{input_type}",
            )
            output.success(
                f"{run_label}: {standard_result.output_rows:,} rows "
                f"({standard_result.duration_seconds:.1f}s)"
            )
            completed.append(run_label)
        except (FileNotFoundError, ValueError, ImportError) as e:
            output.error(f"{run_label}: {e}")
            failed.append(run_label)
            continue  # Skip analysis presets if standard failed

        standard_output = standard_result.results_path

        # --- Run analysis presets on top of standard ---
        for preset_name, extra_steps in preset_extras.items():
            run_label = f"{preset_name}/{input_type}"
            step_names = [s if isinstance(s, str) else s["name"] for s in extra_steps]
            output.section(f"RUN: {run_label}")
            output.key_value("Base", str(standard_output))
            output.key_value("Extra steps", ", ".join(step_names))

            try:
                chained_preprocessor = TextPreprocessor(source=source, text_column="text_processed")
                chained_result: PreprocessingResult = chained_preprocessor.run(
                    steps=extra_steps,
                    input_path=standard_output,
                    sample=limit,
                    batch_size=batch_size,
                    num_beams=num_beams,
                    num_gpus=num_gpus,
                    use_cache=not no_cache,
                    results_filename=results_filename,
                    preprocessing_id=f"{preset_name}_{input_type}",
                )
                output.success(
                    f"{run_label}: {chained_result.output_rows:,} rows "
                    f"({chained_result.duration_seconds:.1f}s)"
                )
                completed.append(run_label)
            except (FileNotFoundError, ValueError, ImportError) as e:
                output.error(f"{run_label}: {e}")
                failed.append(run_label)

    # --- Summary ---
    output.section("BATCH COMPLETE")
    output.key_value("Completed", f"{len(completed)}/{total_runs}")
    for label in completed:
        output.info(f"  OK  {label}")
    if failed:
        for label in failed:
            output.info(f"  FAIL  {label}")
        output.warning(f"{len(failed)} run(s) failed")
    else:
        output.success("All preprocessing runs completed successfully!")
