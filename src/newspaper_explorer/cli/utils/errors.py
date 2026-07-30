"""
Standard error handling utilities for CLI commands.

Provides consistent error handling with proper output formatting and
exit behavior.
"""

import logging
from pathlib import Path
import traceback
from typing import Optional, Union

import click

from newspaper_explorer.cli.utils import output

logger = logging.getLogger(__name__)


def handle_error(
    error: Exception,
    *,
    tip: Optional[str] = None,
    show_traceback: bool = False,
) -> None:
    """
    Handle CLI errors with consistent formatting and abort behavior.

    Args:
        error: The exception that was raised
        tip: Optional recovery tip for the user
        show_traceback: Whether to show full traceback (default: False)

    Raises:
        click.Abort: Always raised to exit cleanly

    Example:
        try:
            df = load_data(path)
        except FileNotFoundError as e:
            handle_error(
                e,
                tip="Run 'newspaper-explorer data parse --source der_tag' first"
            )
    """
    # Print error message
    output.error(str(error))

    # Print recovery tip if provided
    if tip:
        click.echo()
        output.info(f"Tip: {tip}", muted=True)

    # Log full traceback if requested
    if show_traceback:
        click.echo()
        logger.exception("Full error details:")
        traceback.print_exc()

    # Abort with proper exit code
    raise click.Abort()


def handle_validation_error(
    message: str,
    *,
    issues: Optional[list[str]] = None,
    tip: Optional[str] = None,
) -> None:
    """
    Handle validation errors with detailed issue list.

    Args:
        message: Main error message
        issues: List of specific validation issues
        tip: Optional recovery tip

    Raises:
        click.Abort: Always raised to exit cleanly

    Example:
        handle_validation_error(
            "Invalid configuration",
            issues=[
                "Missing required field: 'source'",
                "Invalid value for 'limit': must be positive"
            ],
            tip="Check your command options"
        )
    """
    output.error(message)

    if issues:
        click.echo()
        output.bullet_list(issues, color="red")

    if tip:
        click.echo()
        output.info(f"Tip: {tip}", muted=True)

    raise click.Abort()


def require_file(
    path: Union[Path, str],
    *,
    error_message: Optional[str] = None,
    tip: Optional[str] = None,
) -> None:
    """
    Require that a file exists, abort if not.

    Args:
        path: File path to check
        error_message: Custom error message (default: auto-generated)
        tip: Recovery tip for user

    Raises:
        click.Abort: If file doesn't exist

    Example:
        require_file(
            input_path,
            error_message="Input file not found",
            tip="Run 'newspaper-explorer data parse --source der_tag' first"
        )
    """
    path = Path(path)

    if not path.exists():
        message = error_message or f"File not found: {path}"
        handle_error(FileNotFoundError(message), tip=tip)


def require_directory(
    path: Union[Path, str],
    *,
    error_message: Optional[str] = None,
    tip: Optional[str] = None,
) -> None:
    """
    Require that a directory exists, abort if not.

    Args:
        path: Directory path to check
        error_message: Custom error message (default: auto-generated)
        tip: Recovery tip for user

    Raises:
        click.Abort: If directory doesn't exist

    Example:
        require_directory(
            images_dir,
            error_message="Images directory not found",
            tip="Run 'newspaper-explorer data images download --source der_tag' first"
        )
    """
    path = Path(path)

    if not path.exists() or not path.is_dir():
        message = error_message or f"Directory not found: {path}"
        handle_error(FileNotFoundError(message), tip=tip)


def warn_if_missing(path: Union[Path, str], *, message: Optional[str] = None) -> bool:
    """
    Warn if a file is missing but don't abort.

    Args:
        path: File path to check
        message: Custom warning message (default: auto-generated)

    Returns:
        True if file exists, False if missing

    Example:
        if not warn_if_missing(config_path, message="Config file not found"):
            click.echo("Using default configuration")
    """

    path = Path(path)

    if not path.exists():
        message = message or f"File not found: {path}"
        output.warning(message)
        return False

    return True


def confirm_overwrite(path: Union[Path, str], *, force: bool = False) -> bool:
    """
    Confirm before overwriting an existing file.

    Args:
        path: File path to check
        force: If True, skip confirmation

    Returns:
        True if should proceed, False if should skip

    Example:
        if not confirm_overwrite(output_path, force=force):
            click.echo("Skipping (file exists)")
            return
    """

    path = Path(path)

    if not path.exists():
        return True

    if force:
        return True

    return click.confirm(
        f"File exists: {path}\nOverwrite?",
        default=False,
    )
