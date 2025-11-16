"""
Text cleaning operations.

Provides text cleaning methods:
- Whitespace normalization
- Lowercase conversion
"""

import logging
import re
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def normalize_whitespace(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    keep_newlines: bool = False,
) -> pl.DataFrame:
    """
    Normalize whitespace characters in text.

    Two modes available:

    **Default mode (keep_newlines=False):**
    - Collapses ALL whitespace (spaces, tabs, newlines) to single space
    - Good for: aggregated text blocks, NLP tasks, topic modeling
    - Example: "Hello    world\\n\\tNext" → "Hello world Next"

    **Newline-preserving mode (keep_newlines=True):**
    - Collapses multiple spaces/tabs to single space
    - KEEPS newlines intact, removes spaces around them
    - Good for: line-by-line processing, preserving text structure
    - Example: "Hello    world\\n\\tNext" → "Hello world\\nNext"

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_whitespace)
        keep_newlines: If True, preserves newlines (default: False)

    Returns:
        DataFrame with normalized whitespace

    Example:
        >>> # Default: collapse all whitespace
        >>> df = normalize_whitespace(df)
        >>> # "Hello    world\\n\\ttab  " → "Hello world tab"
        >>>
        >>> # Preserve newlines
        >>> df = normalize_whitespace(df, keep_newlines=True)
        >>> # "Hello    world\\n\\ttab  " → "Hello world\\ntab"
    """
    if output_column is None:
        output_column = f"{input_column}_whitespace"

    mode = "preserve newlines" if keep_newlines else "collapse all"
    logger.info(f"Normalizing whitespace ({mode}): {input_column} → {output_column}")

    if keep_newlines:
        # Preserve newlines, collapse only spaces/tabs
        def normalize_with_newlines(text: str) -> str:
            if not text:
                return text
            # Normalize line breaks to \n
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Collapse multiple spaces/tabs to single space
            text = re.sub(r"[ \t]+", " ", text)
            # Remove spaces/tabs around newlines
            text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
            return text.strip()

        df = df.with_columns(
            [
                pl.col(input_column)
                .map_elements(normalize_with_newlines, return_dtype=pl.Utf8)
                .alias(output_column)
            ]
        )
    else:
        # Default: collapse all whitespace to single space
        df = df.with_columns(
            [
                pl.col(input_column)
                .str.replace_all(r"\s+", " ")  # All whitespace → single space
                .str.strip_chars()  # Remove leading/trailing
                .alias(output_column)
            ]
        )

    logger.info(f"Whitespace normalized for {len(df):,} rows")
    return df


def lowercase(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Convert text to lowercase.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_lower)

    Returns:
        DataFrame with lowercased text column
    """
    if output_column is None:
        output_column = f"{input_column}_lower"

    logger.info(f"Converting to lowercase: {input_column} → {output_column}")

    df = df.with_columns([pl.col(input_column).str.to_lowercase().alias(output_column)])

    logger.info(f"Lowercased {len(df):,} rows")
    return df
