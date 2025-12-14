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


def max_workers_option(default: int = 8) -> Callable[[F], F]:
    """
    Standard --max-workers option for parallel processing.

    Args:
        default: Default number of workers (default: 8)

    Returns:
        Click option decorator

    Example:
        @click.command()
        @max_workers_option(default=16)
        def my_command(max_workers):
            pass
    """

    def decorator(func: F) -> F:
        return click.option(
            "--max-workers",
            type=int,
            default=default,
            help="Maximum parallel workers",
            show_default=True,
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
