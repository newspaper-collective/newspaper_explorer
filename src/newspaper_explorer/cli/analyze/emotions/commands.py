"""
CLI commands for emotion analysis.

Provides commands for predicting emotions using pre-trained BERT models.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.analyze.emotions.bert import EmotionPredictor
from newspaper_explorer.analyze.emotions.utils import get_model_status
from newspaper_explorer.cli.utils import errors, output
from newspaper_explorer.cli.utils.options import (
    batch_size_option,
    input_file_option,
    limit_option,
    model_dir_option,
    source_option,
    text_column_option,
)
from newspaper_explorer.models.analysis.emotions import EMOTIONS


@click.group(name="emotions")
def emotions_group() -> None:
    """Predict emotions using pre-trained BERT models."""
    pass


@emotions_group.command(name="predict")
@source_option()
@input_file_option(help_text="Custom input parquet file (default: textblocks or lines)")
@text_column_option()
@batch_size_option(default=64)
@limit_option()
@model_dir_option(default="models/emotions")
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Number of rows per chunk",
    show_default=True,
)
@click.option(
    "--fp16",
    is_flag=True,
    help="Use FP16 mixed precision (~30% faster, recommended for modern GPUs)",
)
@click.option(
    "--compile/--no-compile",
    default=True,
    help="Use torch.compile for ~20-30% additional speedup (default: enabled)",
)
@click.option(
    "--single-gpu",
    is_flag=True,
    help="Force single GPU mode (default: use all GPUs in parallel)",
)
def predict(
    source: str,
    input_file: Optional[str],
    text_column: str,
    batch_size: int,
    limit: Optional[int],
    model_dir: str,
    chunk_size: int,
    *,
    fp16: bool,
    compile: bool,
    single_gpu: bool,
) -> None:
    """
    Predict emotions for newspaper texts using pre-trained BERT models.

    Uses the Shaver emotion model with 6 categories:
    - Sadness (Traurigkeit)
    - Love (Liebe)
    - Joy (Freude)
    - Fear (Angst)
    - Anger (Wut/Ärger)
    - Agitation (Unruhe/Erregung)

    Each emotion is predicted independently (binary classification).
    A text can have multiple emotions or none.    Performance:
    - Single GPU: ~1,000-2,000 texts/second
    - 4x L40S GPUs: ~4,000-8,000 texts/second
    - For 61M texts: ~2-4 hours on 4x L40S

    Examples:

    \b
    Predict emotions for source (auto-detect input):
        newspaper-explorer analyze emotions predict --source der_tag

    \b
    Use custom input file with large batches (for L40S GPUs):
        newspaper-explorer analyze emotions predict --source der_tag \\
            --input-file data/processed/der_tag/textblocks_normalized.parquet \\
            --batch-size 192 --fp16

    \b
    Process with custom text column:
        newspaper-explorer analyze emotions predict --source der_tag \\
            --text-column text_normalized

    \b
    Single GPU mode with smaller batches:
        newspaper-explorer analyze emotions predict --source der_tag \\
            --single-gpu --batch-size 32

    \b
    Output:
        Saves to: results/{source}/emotions/{output_name}.parquet
        Columns: original columns + for each emotion:
            - {Emotion}: Binary prediction (0/1)
            - {Emotion}_prob: Confidence score (0.0-1.0)
        Example: Sadness, Sadness_prob, Love, Love_prob, Joy, Joy_prob, etc.
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output.header("Emotion Prediction")

    output.key_value("Source", source)
    if input_file:
        output.key_value("Input file", input_file)
    output.key_value("Text column", text_column)
    output.key_value("Batch size", batch_size)
    if limit:
        output.key_value("Limit", f"{limit:,} rows")
    output.key_value("Chunk size", f"{chunk_size:,}")
    output.key_value("Model directory", model_dir)
    output.key_value("FP16", fp16)
    output.key_value("Model compilation", compile)
    output.key_value("Multi-GPU parallel", not single_gpu)
    click.echo()

    try:
        # Initialize predictor
        predictor = EmotionPredictor(
            source_name=source,
            model_dir=Path(model_dir),
            batch_size=batch_size,
            chunk_size=chunk_size,
            use_fp16=fp16,
            use_compile=compile,
            multi_gpu=not single_gpu,
        )

        # Run prediction
        if input_file:
            output_file = predictor.predict(
                input_file=Path(input_file),
                text_column=text_column,
                limit=limit,
            )
        else:
            output_file = predictor.predict_from_source(
                text_column=text_column,
                limit=limit,
            )

        click.echo()
        output.success("Emotion prediction complete!")
        output.info(f"Results saved to: {output_file}")

        # Build column lists from centralized constants
        prob_cols = [f"{e}_prob" for e in EMOTIONS]

        click.echo("\nTo analyze results:")
        output.code_block(
            [
                "import polars as pl",
                f"df = pl.read_parquet('{output_file}')",
                "# Binary counts",
                f"print(df[{EMOTIONS}].sum())",
                "# Average confidence scores",
                f"print(df[{prob_cols}].mean())",
            ]
        )
        click.echo()

    except FileNotFoundError as e:
        click.echo()
        errors.handle_validation_error(
            f"File not found: {e}",
            issues=[
                f"Run parsing first: newspaper-explorer data parse --source {source}",
                f"Download emotion models to: {model_dir}",
                "Specify custom input: --input-file path/to/file.parquet",
            ],
        )
    except (RuntimeError, ValueError, OSError) as e:
        click.echo()
        errors.handle_error(e, show_traceback=True)


@emotions_group.command(name="models")
@model_dir_option(default="models/emotions")
def models(model_dir: str) -> None:
    """
    Check emotion model availability.

    Lists all required emotion models and their status (found/missing).
    Useful for verifying that models are properly installed.

    Example:

    \b
    Check model availability:
        newspaper-explorer analyze emotions models
    """
    output.header("Emotion Model Status")

    # Get complete status
    status = get_model_status(Path(model_dir))
    cuda_info = status["cuda_info"]
    model_info = status["model_info"]

    # Display model directory info
    output.key_value("Model directory", model_info["model_dir"])
    output.key_value("Directory exists", model_info["exists"])
    click.echo()

    # Display CUDA info
    output.section("CUDA Status")
    if cuda_info.get("error"):
        output.error(f"CUDA error: {cuda_info['error']}", symbol=False)
    elif cuda_info["cuda_available"]:
        output.info(f"CUDA available: {cuda_info['device_count']} GPU(s)")
        for gpu in cuda_info["gpu_info"]:
            output.info(
                f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)", muted=True
            )
    else:
        output.warning("CUDA not available (will use CPU)")

    # Display model status
    output.section("Model Files")
    for emotion, model_item in model_info["models"].items():
        if model_item["found"]:
            output.success(f"{emotion:12s} ({model_item['size_mb']:.1f} MB)")
        else:
            output.error(f"{emotion:12s} (NOT FOUND)")

    click.echo()

    # Summary and instructions
    if model_info["all_found"]:
        output.success("All models found! Ready to predict.")
    else:
        output.error("Some models are missing.")
        click.echo("\nTo download models:")
        output.bullet_list(
            [
                "Download from your source",
                f"Extract to: {model_info['model_dir']}",
                "Ensure files are named: Sadness.pt, Love.pt, etc.",
            ]
        )

    click.echo()
