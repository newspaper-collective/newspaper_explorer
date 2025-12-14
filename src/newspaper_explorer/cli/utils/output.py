"""
CLI utilities for consistent output styling.

Provides helper functions for formatting CLI output with consistent styling,
colors, and symbols across all commands.
"""

from typing import Optional, Union

import click


# Color scheme
class Colors:
    """ANSI color codes for CLI output."""

    SUCCESS = "green"
    ERROR = "red"
    WARNING = "yellow"
    INFO = "cyan"
    MUTED = "bright_black"
    HEADER = "bright_cyan"


# Symbols
class Symbols:
    """Unicode symbols for CLI output."""

    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"  # noqa: RUF001  # Unicode info symbol is intentional
    ARROW = "→"
    BULLET = "•"


def header(text: str, char: str = "=", width: int = 80) -> None:
    """
    Print a header with consistent styling.

    Args:
        text: Header text
        char: Character to use for border (default: '=')
        width: Total width of header (default: 80)

    Example:
        >>> header("Processing Data")
        ================================================================================
        Processing Data
        ================================================================================
    """
    border = char * width
    click.echo()
    click.secho(border, fg=Colors.HEADER)
    click.secho(text, fg=Colors.HEADER, bold=True)
    click.secho(border, fg=Colors.HEADER)
    click.echo()


def success(text: str, *, symbol: bool = True) -> None:
    """
    Print a success message.

    Args:
        text: Message text
        symbol: Whether to include success symbol (default: True)

    Example:
        >>> success("Operation completed")
        ✓ Operation completed
    """
    prefix = f"{Symbols.SUCCESS} " if symbol else ""
    click.secho(f"{prefix}{text}", fg=Colors.SUCCESS)


def error(text: str, *, symbol: bool = True, err: bool = True) -> None:
    """
    Print an error message.

    Args:
        text: Error message
        symbol: Whether to include error symbol (default: True)
        err: Whether to write to stderr (default: True)

    Example:
        >>> error("File not found")
        ✗ File not found
    """
    prefix = f"{Symbols.ERROR} " if symbol else ""
    click.secho(f"{prefix}{text}", fg=Colors.ERROR, err=err)


def warning(text: str, *, symbol: bool = True) -> None:
    """
    Print a warning message.

    Args:
        text: Warning message
        symbol: Whether to include warning symbol (default: True)

    Example:
        >>> warning("This may take a while")
        ⚠ This may take a while
    """
    prefix = f"{Symbols.WARNING} " if symbol else ""
    click.secho(f"{prefix}{text}", fg=Colors.WARNING)


def info(text: str, *, symbol: bool = False, muted: bool = False) -> None:
    """
    Print an informational message.

    Args:
        text: Info message
        symbol: Whether to include info symbol (default: False)
        muted: Whether to use muted color (default: False)

    Example:
        >>> info("Processing 100 files...")
        Processing 100 files...
    """
    prefix = f"{Symbols.INFO} " if symbol else ""
    color = Colors.MUTED if muted else Colors.INFO
    click.secho(f"{prefix}{text}", fg=color)


def bullet_list(items: list[str], color: Optional[str] = None) -> None:
    """
    Print a bulleted list.

    Args:
        items: List of items to print
        color: Optional color for bullets

    Example:
        >>> bullet_list(["Item 1", "Item 2", "Item 3"])
        • Item 1
        • Item 2
        • Item 3
    """
    for item in items:
        click.secho(f"  {Symbols.BULLET} {item}", fg=color)


def key_value(
    key: str,
    value: Union[str, int],
    *,
    indent: int = 0,
    is_enabled: Optional[bool] = None,
) -> None:
    """
    Print a key-value pair with consistent formatting.

    Args:
        key: Key name
        value: Value to display
        indent: Number of spaces to indent (default: 0)
        is_enabled: If set, formats value as enabled/disabled (default: None)

    Example:
        >>> key_value("Source", "der_tag")
        Source: der_tag
        >>> key_value("FP16", "", is_enabled=True)
        FP16: enabled
    """
    spaces = " " * indent

    # Format enabled/disabled values
    if is_enabled is not None:
        value_str = (
            click.style("enabled", fg=Colors.SUCCESS)
            if is_enabled
            else click.style("disabled", fg=Colors.MUTED)
        )
    else:
        value_str = str(value)

    click.echo(f"{spaces}{click.style(key + ':', bold=True)} {value_str}")


def section(title: str) -> None:
    """
    Print a section title.

    Args:
        title: Section title

    Example:
        >>> section("Configuration")

        Configuration
        ─────────────
    """
    click.echo()
    click.secho(title, bold=True)
    click.secho("─" * len(title), fg=Colors.MUTED)


def divider(char: str = "─", width: int = 80) -> None:
    """
    Print a horizontal divider.

    Args:
        char: Character to use (default: '─')
        width: Width of divider (default: 80)

    Example:
        >>> divider()
        ────────────────────────────────────────────────────────────────────────────────
    """
    click.secho(char * width, fg=Colors.MUTED)


def code_block(lines: list[str]) -> None:
    """
    Print a code block.

    Args:
        lines: Lines of code

    Example:
        >>> code_block(["import polars as pl", "df = pl.read_parquet('file.parquet')"])
          import polars as pl
          df = pl.read_parquet('file.parquet')
    """
    for line in lines:
        click.echo(f"  {line}")


def progress_info(
    current: int,
    total: int,
    item_name: str = "items",
    extra: Optional[str] = None,
) -> None:
    """
    Print progress information.

    Args:
        current: Current count
        total: Total count
        item_name: Name of items being processed
        extra: Optional extra information

    Example:
        >>> progress_info(50, 100, "files", "~2 min remaining")
        Processing 50/100 files (~2 min remaining)
    """
    extra_str = f" ({extra})" if extra else ""
    click.echo(f"Processing {current:,}/{total:,} {item_name}{extra_str}")
