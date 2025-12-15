"""
Reusable CLI option decorators.

Provides standard option decorators for common CLI parameters to ensure
consistency across all commands.
"""

from typing import Any, Callable, Optional, TypeVar

import click

F = TypeVar("F", bound=Callable[..., Any])


def source_option(*, required: bool = True) -> Callable[[F], F]:
    """
    Add --source option to a Click command.

    Args:
        required: Whether the source option is required (default: True)

    Returns:
        Decorator function that adds the option

    Example:
        @click.command()
        @source_option()
        def my_command(source):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--source",
            "-s",
            type=str,
            required=required,
            help="Source name (e.g., 'der_tag')",
        )(func)

    return decorator


def limit_option(
    default: Optional[int] = None, help_text: Optional[str] = None
) -> Callable[[F], F]:
    """
    Standard --limit option for row limiting.

    Replaces inconsistent --sample and --sample-size options across commands.

    Args:
        default: Default limit value (default: None for no limit)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @limit_option(default=1000)
        def my_command(limit):
            pass
    """
    return click.option(
        "--limit",
        type=int,
        default=default,
        help=help_text or "Limit number of rows to process (for testing)",
        show_default=bool(default),
    )


def year_option(*, required: bool = False) -> Callable[[F], F]:
    """
    Standard --year option for filtering by year.

    Args:
        required: Whether the year option is required (default: False)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @year_option()
        def my_command(year):
            pass
    """
    return click.option(
        "--year",
        type=int,
        required=required,
        help="Filter by year",
    )


def input_type_option(default: str = "textblocks") -> Callable[[F], F]:
    """
    Standard --input-type option.

    Args:
        default: Default input type (default: "textblocks")

    Returns:
        Click option decorator

    Example:
        @click.command()
        @input_type_option()
        def my_command(input_type):
            pass
    """
    return click.option(
        "--input-type",
        type=click.Choice(["textblocks", "lines"], case_sensitive=False),
        default=default,
        help="Input data type",
        show_default=True,
    )


def text_column_option(default: str = "text") -> Callable[[F], F]:
    """
    Standard --text-column option.

    Args:
        default: Default column name (default: "text")

    Returns:
        Click option decorator

    Example:
        @click.command()
        @text_column_option()
        def my_command(text_column):
            pass
    """
    return click.option(
        "--text-column",
        type=str,
        default=default,
        help="Name of text column to process",
        show_default=True,
    )


def resume_option(*, default: bool = True) -> Callable[[F], F]:
    """
    Standard --resume option for skipping already processed items.

    Args:
        default: Default resume behavior (default: True)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @resume_option()
        def my_command(resume):
            pass
    """
    return click.option(
        "--resume/--no-resume",
        default=default,
        help="Skip already processed items",
        show_default=True,
    )


def batch_size_option(default: int = 32) -> Callable[[F], F]:
    """
    Standard --batch-size option for ML/processing operations.

    Args:
        default: Default batch size (default: 32)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @batch_size_option(default=64)
        def my_command(batch_size):
            pass
    """
    return click.option(
        "--batch-size",
        type=int,
        default=default,
        help="Batch size for processing",
        show_default=True,
    )


def min_length_option(default: int = 10) -> Callable[[F], F]:
    """
    Standard --min-length option for text filtering.

    Args:
        default: Default minimum text length in characters (default: 10)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @min_length_option(default=100)
        def my_command(min_length):
            pass
    """
    return click.option(
        "--min-length",
        type=int,
        default=default,
        help="Minimum text length in characters",
        show_default=True,
    )


def max_length_option(default: int = 1000, help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --max-length option for text chunking.

    Args:
        default: Default maximum text length in characters (default: 1000)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @max_length_option(default=8000, help_text="Max chars before chunking")
        def my_command(max_length):
            pass
    """
    return click.option(
        "--max-length",
        type=int,
        default=default,
        help=help_text or "Maximum text length in characters",
        show_default=True,
    )


def threshold_option(default: float = 0.5, help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --threshold option for confidence/probability thresholds.

    Args:
        default: Default threshold value (default: 0.5)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @threshold_option(default=0.2)
        def my_command(threshold):
            pass
    """
    return click.option(
        "--threshold",
        type=float,
        default=default,
        help=help_text or "Confidence threshold (0-1)",
        show_default=True,
    )


def model_option(default: str, help_text: str) -> Callable[[F], F]:
    """
    Standard --model option for specifying ML model names.

    Args:
        default: Default model name or identifier
        help_text: Help text describing available models

    Returns:
        Click option decorator

    Example:
        @click.command()
        @model_option(default="gpt-4o-mini", help_text="LLM model to use")
        def my_command(model):
            pass
    """
    return click.option(
        "--model",
        type=str,
        default=default,
        help=help_text,
        show_default=True,
    )


def num_gpus_option(default: int = 1) -> Callable[[F], F]:
    """
    Standard --num-gpus option for GPU parallel processing.

    Args:
        default: Default number of GPUs to use (default: 1)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @num_gpus_option(default=1)
        def my_command(num_gpus):
            pass
    """
    return click.option(
        "--num-gpus",
        type=int,
        default=default,
        help="Number of GPUs to use for parallel processing (auto-detects available GPUs)",
        show_default=True,
    )


def num_topics_option(
    default: Optional[int] = None, help_text: Optional[str] = None
) -> Callable[[F], F]:
    """
    Standard --num-topics option for topic modeling.

    Args:
        default: Default number of topics (default: None for automatic)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @num_topics_option(default=20)
        def my_command(num_topics):
            pass
    """
    return click.option(
        "--num-topics",
        type=int,
        default=default,
        help=help_text or "Number of topics to extract",
        show_default=bool(default),
    )


def temperature_option(default: float = 0.3) -> Callable[[F], F]:
    """
    Standard --temperature option for LLM sampling temperature.

    Args:
        default: Default temperature value (default: 0.3)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @temperature_option(default=0.5)
        def my_command(temperature):
            pass
    """
    return click.option(
        "--temperature",
        type=float,
        default=default,
        help="Sampling temperature (0.0 = deterministic, higher = more random)",
        show_default=True,
    )


def max_tokens_option(default: int = 2000) -> Callable[[F], F]:
    """
    Standard --max-tokens option for LLM maximum output tokens.

    Args:
        default: Default maximum tokens (default: 2000)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @max_tokens_option(default=4000)
        def my_command(max_tokens):
            pass
    """
    return click.option(
        "--max-tokens",
        type=int,
        default=default,
        help="Maximum tokens in LLM response",
        show_default=True,
    )


def output_path_option(help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --output option for specifying output file path.

    Args:
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @output_path_option()
        def my_command(output):
            pass
    """
    return click.option(
        "--output",
        "-o",
        type=click.Path(),
        help=help_text or "Output file path",
    )


def force_option() -> Callable[[F], F]:
    """
    Standard --force option for overwriting existing files.

    Returns:
        Click option decorator

    Example:
        @click.command()
        @force_option()
        def my_command(force):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--force",
            is_flag=True,
            help="Overwrite existing output files",
        )(func)

    return decorator


def input_file_option(help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --input-file option for custom input parquet files.

    Args:
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @input_file_option()
        def my_command(input_file):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--input-file",
            type=click.Path(exists=True),
            help=help_text or "Custom input parquet file (overrides default source path)",
        )(func)

    return decorator


def output_name_option(default: str = "results") -> Callable[[F], F]:
    """
    Standard --output-name option for result file naming.

    Args:
        default: Default output name (default: "results")

    Returns:
        Click option decorator

    Example:
        @click.command()
        @output_name_option(default="predictions")
        def my_command(output_name):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--output-name",
            type=str,
            default=default,
            help="Base name for output file",
            show_default=True,
        )(func)

    return decorator


def model_dir_option(default: str = "models") -> Callable[[F], F]:
    """
    Standard --model-dir option for ML model directories.

    Args:
        default: Default model directory (default: "models")

    Returns:
        Click option decorator

    Example:
        @click.command()
        @model_dir_option(default="models/emotions")
        def my_command(model_dir):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--model-dir",
            type=click.Path(),
            default=default,
            help="Directory containing model files",
            show_default=True,
        )(func)

    return decorator


def num_workers_option(
    default: Optional[int] = None, help_text: Optional[str] = None
) -> Callable[[F], F]:
    """
    Standard --num-workers option for parallel processing.

    Args:
        default: Default number of workers (default: None for auto-detection)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @num_workers_option()  # Auto-detect
        def my_command(num_workers):
            pass

        @click.command()
        @num_workers_option(default=8)  # Fixed default
        def my_command(num_workers):
            pass
    """

    def decorator(func: F) -> F:
        if default is not None:
            return click.option(
                "--num-workers",
                type=int,
                default=default,
                help=help_text or "Number of parallel workers",
                show_default=True,
            )(func)
        return click.option(
            "--num-workers",
            type=int,
            help=help_text or "Number of parallel workers (default: auto-detect)",
        )(func)

    return decorator


def max_retries_option(default: int = 3) -> Callable[[F], F]:
    """
    Standard --max-retries option for retry logic.

    Args:
        default: Default number of retries (default: 3)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @max_retries_option(default=5)
        def my_command(max_retries):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--max-retries",
            type=int,
            default=default,
            help="Maximum retry attempts for failed operations",
            show_default=True,
        )(func)

    return decorator


def min_image_size_option(default: int = 1024) -> Callable[[F], F]:
    """
    Standard --min-size option for minimum image file size validation.

    Args:
        default: Minimum expected image size in bytes (default: 1024)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @min_image_size_option(default=5000)
        def my_command(min_size):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--min-size",
            type=int,
            default=default,
            help="Minimum expected image size in bytes",
            show_default=True,
        )(func)

    return decorator


def top_k_option(default: int = 10, help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --top-k option for limiting top results.

    Args:
        default: Default number of top items to return (default: 10)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @top_k_option(default=15)
        def my_command(top_k):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--top-k",
            type=int,
            default=default,
            help=help_text or "Number of top items to extract",
            show_default=True,
        )(func)

    return decorator


def group_by_option(help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --group-by option for grouping columns.

    Args:
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @group_by_option()
        def my_command(group_by):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--group-by",
            type=str,
            multiple=True,
            default=None,
            help=help_text or "Column(s) to group by (can specify multiple times)",
        )(func)

    return decorator


def device_option(help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --device option for ML device selection.

    Args:
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @device_option()
        def my_command(device):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--device",
            type=str,
            help=help_text or "Device to use ('cuda', 'cpu', or None for auto-detect)",
        )(func)

    return decorator


def use_chunking_option(
    *, default: bool = True, help_text: Optional[str] = None
) -> Callable[[F], F]:
    """
    Standard --use-chunking option for text chunking.

    Args:
        default: Default chunking behavior (default: True)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @use_chunking_option()
        def my_command(use_chunking):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--use-chunking/--no-chunking",
            default=default,
            help=help_text or "Split long texts into chunks",
            show_default=True,
        )(func)

    return decorator


def chunk_size_option(default: int = 400, help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --chunk-size option for text chunk size.

    Args:
        default: Default chunk size in tokens (default: 400)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @chunk_size_option(default=512)
        def my_command(chunk_size):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--chunk-size",
            type=int,
            default=default,
            help=help_text or "Token size for each chunk",
            show_default=True,
        )(func)

    return decorator


def chunk_overlap_option(default: int = 50, help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --chunk-overlap option for chunk overlap.

    Args:
        default: Default overlap in tokens (default: 50)
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @chunk_overlap_option(default=100)
        def my_command(chunk_overlap):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--chunk-overlap",
            type=int,
            default=default,
            help=help_text or "Overlap between chunks in tokens",
            show_default=True,
        )(func)

    return decorator


def compile_model_option(help_text: Optional[str] = None) -> Callable[[F], F]:
    """
    Standard --compile-model option for PyTorch model compilation.

    Args:
        help_text: Custom help text (default: standard help)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @compile_model_option()
        def my_command(compile_model):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--compile-model",
            is_flag=True,
            help=help_text or "Use torch.compile for model optimization (requires PyTorch 2.0+)",
        )(func)

    return decorator
