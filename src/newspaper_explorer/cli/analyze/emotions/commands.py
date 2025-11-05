"""
CLI commands for emotion analysis.

Provides commands for predicting emotions using pre-trained BERT models.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from newspaper_explorer.config.base import get_config


@click.group(name="emotions")
def emotions_group():
    """Predict emotions using pre-trained BERT models."""
    pass


@emotions_group.command(name="predict")
@click.option(
    "--source",
    type=str,
    required=True,
    help="Source name (e.g., 'der_tag')",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="Custom input parquet file (default: textblocks or lines)",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Name of text column to process",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for inference (increase for GPUs with more memory)",
    show_default=True,
)
@click.option(
    "--chunk-size",
    type=int,
    default=100000,
    help="Number of rows per chunk",
    show_default=True,
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True),
    default="models/emotions",
    help="Directory containing emotion model files",
    show_default=True,
)
@click.option(
    "--fp16",
    is_flag=True,
    help="Use FP16 mixed precision (~30% faster, recommended for modern GPUs)",
)
@click.option(
    "--compile",
    "use_compile",
    is_flag=True,
    default=True,
    help="Use torch.compile for ~20-30% additional speedup (default: enabled)",
)
@click.option(
    "--no-compile",
    "no_compile",
    is_flag=True,
    help="Disable torch.compile (use if you have PyTorch < 2.0)",
)
@click.option(
    "--single-gpu",
    is_flag=True,
    help="Force single GPU mode (default: use all GPUs in parallel)",
)
@click.option(
    "--output-name",
    type=str,
    default="emotion_predictions",
    help="Base name for output file",
    show_default=True,
)
def predict(
    source: str,
    input_file: Optional[str],
    text_column: str,
    batch_size: int,
    chunk_size: int,
    model_dir: str,
    fp16: bool,
    use_compile: bool,
    no_compile: bool,
    single_gpu: bool,
    output_name: str,
):
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
    from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    click.echo(f"\n{'='*80}")
    click.echo(f"Emotion Prediction")
    click.echo(f"{'='*80}\n")

    # Handle compile flags
    compile_enabled = use_compile and not no_compile

    click.echo(f"Source: {source}")
    if input_file:
        click.echo(f"Input file: {input_file}")
    click.echo(f"Text column: {text_column}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Chunk size: {chunk_size:,}")
    click.echo(f"Model directory: {model_dir}")
    click.echo(f"FP16: {'enabled' if fp16 else 'disabled'}")
    click.echo(f"Model compilation: {'enabled' if compile_enabled else 'disabled'}")
    click.echo(f"Multi-GPU parallel: {'disabled' if single_gpu else 'enabled'}")
    click.echo()

    try:
        # Initialize predictor
        predictor = EmotionPredictor(
            source_name=source,
            model_dir=Path(model_dir),
            batch_size=batch_size,
            chunk_size=chunk_size,
            use_fp16=fp16,
            use_compile=compile_enabled,
            multi_gpu=not single_gpu,
        )

        # Run prediction
        if input_file:
            output_file = predictor.predict(
                input_file=Path(input_file),
                text_column=text_column,
                output_name=output_name,
            )
        else:
            output_file = predictor.predict_from_source(
                text_column=text_column,
                output_name=output_name,
            )

        click.echo(f"\n{'='*80}")
        click.echo(f"✓ Emotion prediction complete!")
        click.echo(f"{'='*80}\n")
        click.echo(f"Results saved to: {output_file}")
        click.echo(f"\nTo analyze results:")
        click.echo(f"  import polars as pl")
        click.echo(f"  df = pl.read_parquet('{output_file}')")
        click.echo(f"  # Binary counts")
        click.echo(f"  print(df[['Sadness', 'Love', 'Joy', 'Fear', 'Anger', 'Agitation']].sum())")
        click.echo(f"  # Average confidence scores")
        click.echo(
            f"  print(df[['Sadness_prob', 'Love_prob', 'Joy_prob', 'Fear_prob', 'Anger_prob', 'Agitation_prob']].mean())"
        )
        click.echo()

    except FileNotFoundError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nPossible solutions:", err=True)
        click.echo(
            f"  1. Run parsing first: newspaper-explorer data parse --source {source}", err=True
        )
        click.echo(f"  2. Download emotion models to: {model_dir}", err=True)
        click.echo(f"  3. Specify custom input: --input-file path/to/file.parquet", err=True)
        return
    except Exception as e:
        click.echo(f"\nError during emotion prediction: {e}", err=True)
        logging.exception("Full error details:")
        return


@emotions_group.command(name="models")
@click.option(
    "--model-dir",
    type=click.Path(),
    default="models/emotions",
    help="Directory to check for model files",
    show_default=True,
)
def models(model_dir: str):
    """
    Check emotion model availability.

    Lists all required emotion models and their status (found/missing).
    Useful for verifying that models are properly installed.

    Example:

    \b
    Check model availability:
        newspaper-explorer analyze emotions models
    """
    import torch

    click.echo(f"\n{'='*80}")
    click.echo(f"Emotion Model Status")
    click.echo(f"{'='*80}\n")

    model_dir_path = Path(model_dir)
    click.echo(f"Model directory: {model_dir_path.absolute()}")
    click.echo(f"Directory exists: {'Yes' if model_dir_path.exists() else 'No'}")
    click.echo()

    # Check CUDA
    if torch.cuda.is_available():
        click.echo(f"CUDA available: Yes ({torch.cuda.device_count()} GPU(s))")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            click.echo(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        click.echo(f"CUDA available: No (will use CPU)")
    click.echo()

    # Check models
    click.echo("Required models:")
    emotions = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
    all_found = True

    for emotion in emotions:
        model_file = model_dir_path / f"{emotion}.pt"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024**2)
            click.echo(f"  ✓ {emotion:12s} ({size_mb:.1f} MB)")
        else:
            click.echo(f"  ✗ {emotion:12s} (NOT FOUND)")
            all_found = False

    click.echo()

    if all_found:
        click.echo("✓ All models found! Ready to predict.")
    else:
        click.echo("✗ Some models are missing.")
        click.echo("\nTo download models:")
        click.echo("  1. Download from your source")
        click.echo(f"  2. Extract to: {model_dir_path.absolute()}")
        click.echo("  3. Ensure files are named: Sadness.pt, Love.pt, etc.")

    click.echo()
