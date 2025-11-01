"""
Basic text cleaning operations.

Provides fundamental text cleaning methods:
- Whitespace normalization
- Lowercase conversion
- Diacritic removal
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def normalize_whitespace(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Normalize all whitespace characters in text.

    Replaces all consecutive whitespace characters (spaces, tabs, newlines,
    carriage returns, etc.) with a single space and removes leading/trailing
    whitespace.

    This handles:
    - Multiple spaces → single space
    - Tabs (\\t) → single space
    - Newlines (\\n) → single space
    - Carriage returns (\\r) → single space
    - Mixed whitespace → single space

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_whitespace)

    Returns:
        DataFrame with normalized whitespace

    Example:
        >>> df = normalize_whitespace(df)
        >>> # "Hello    world\\n\\ttab  " → "Hello world tab"
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_whitespace"

    logger.info(f"Normalizing whitespace: {input_column} → {output_column}")

    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all(
                r"\s+", " "
            )  # Replace all whitespace (spaces, tabs, newlines, etc.) with single space
            .str.strip_chars()  # Remove leading/trailing whitespace
            .alias(output_column)
        ]
    )

    logger.info(f"Whitespace normalized for {len(df):,} rows")
    return df


def lowercase(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Convert text to lowercase.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_lower)

    Returns:
        DataFrame with lowercased text column
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_lower"

    logger.info(f"Converting to lowercase: {input_column} → {output_column}")

    df = df.with_columns([pl.col(input_column).str.to_lowercase().alias(output_column)])

    logger.info(f"Lowercased {len(df):,} rows")
    return df


def remove_diacritics(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Remove diacritics from text using unidecode.

    Converts accented characters to their ASCII equivalents:
    - ä → a, ö → o, ü → u
    - é → e, à → a, etc.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_no_diacritics)

    Returns:
        DataFrame with diacritics removed

    Example:
        >>> df = remove_diacritics(df)
        >>> # "Münchner Straße" → "Munchner Strasse"

    Note:
        Requires unidecode package: pip install unidecode
    """
    try:
        from unidecode import unidecode
    except ImportError:
        raise ImportError(
            "unidecode is required for diacritic removal. " "Install with: pip install unidecode"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_no_diacritics"

    logger.info(f"Removing diacritics: {input_column} → {output_column}")

    # Apply unidecode to each text
    texts = df[input_column].to_list()
    processed = [unidecode(str(text)) if text else "" for text in texts]

    df = df.with_columns([pl.Series(output_column, processed)])

    logger.info(f"Removed diacritics from {len(df):,} rows")
    return df
