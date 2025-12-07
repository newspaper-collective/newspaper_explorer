"""
Content filtering and removal operations.

Provides methods to remove unwanted content from text:
- Punctuation removal
- Number removal
- Stopword removal
- Length filtering
- OCR artifact cleanup
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def filter_by_total_character_length(
    df: pl.DataFrame,
    input_column: str = "text",
    min_length: int = 10,
    max_length: Optional[int] = None,
) -> pl.DataFrame:
    """
    Filter out texts that are too short or too long.

    Removes rows where text length is outside the specified range.
    Useful for removing OCR artifacts, headers, footers, or malformed entries.

    Args:
        df: Input DataFrame
        input_column: Column to check length (default: "text")
        min_length: Minimum text length in characters (default: 10)
        max_length: Maximum text length in characters (default: None = no limit)

    Returns:
        DataFrame with short/long texts filtered out

    Example:
        >>> # Remove very short texts (likely artifacts)
        >>> df = filter_by_total_character_length(df, min_length=20)
        >>> # Remove both short and very long texts
        >>> df = filter_by_total_character_length(df, min_length=10, max_length=10000)
    """
    logger.info(f"Filtering by length: {input_column} (min={min_length}, max={max_length})")

    original_count = len(df)

    # Calculate text lengths
    lengths = df[input_column].str.len_chars()

    # Apply filters
    mask = lengths >= min_length
    if max_length is not None:
        mask = mask & (lengths <= max_length)

    df = df.filter(mask)

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_by_word_count(
    df: pl.DataFrame,
    input_column: str = "text",
    min_words: int = 2,
    max_words: Optional[int] = None,
) -> pl.DataFrame:
    """
    Filter out texts based on word count.

    Removes rows where word count is outside the specified range.
    Useful for removing single-word artifacts, headers, or excessively long malformed entries.
    More meaningful than character count for content filtering.

    Args:
        df: Input DataFrame
        input_column: Column to check word count (default: "text")
        min_words: Minimum number of words (default: 2)
        max_words: Maximum number of words (default: None = no limit)

    Returns:
        DataFrame with texts filtered by word count

    Example:
        >>> # Remove single-word lines (likely artifacts)
        >>> df = filter_by_word_count(df, min_words=2)
        >>> # Keep only lines with 3-50 words
        >>> df = filter_by_word_count(df, min_words=3, max_words=50)
        >>> # Remove very long lines (possible OCR errors)
        >>> df = filter_by_word_count(df, max_words=100)
    """
    logger.info(f"Filtering by word count: {input_column} (min={min_words}, max={max_words})")

    original_count = len(df)

    # Count words by splitting on whitespace
    word_counts = df[input_column].str.split(" ").list.len()

    # Apply filters
    mask = word_counts >= min_words
    if max_words is not None:
        mask = mask & (word_counts <= max_words)

    df = df.filter(mask)

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_number_only_lines(
    df: pl.DataFrame,
    input_column: str = "text",
    *,
    allow_separators: bool = True,
) -> pl.DataFrame:
    """
    Filter out lines containing only numbers and optional separators.

    These are typically page numbers, dates, or OCR artifacts:
    - "123" (page numbers)
    - "1.234" (numbers with periods)
    - "12-34-56" (dates or reference numbers)
    - "1,000" (formatted numbers)
    - "3⁴" (superscripts)
    - "½" or "3 ½" (fractions)

    Uses Unicode \\p{N} pattern to match all numeric characters including
    superscripts (⁴, ³, ²), subscripts (₁, ₂), and fractions (½, ¼, ¾).

    Args:
        df: Input DataFrame
        input_column: Column to check (default: "text")
        allow_separators: If True, allows common separators (., -, /, :, ,, space) with numbers
                         If False, only pure numeric strings are filtered

    Returns:
        DataFrame with number-only lines filtered out

    Example:
        >>> # These get REMOVED:
        >>> # "123", "45.67", "1-2-3", "1,000", "3⁴", "½", "3¾", "3 ½"
        >>>
        >>> # These are KEPT:
        >>> # "Seite 123" (has text)
        >>> # "Am 12. März" (has text)
        >>> # "Die 3 Musketiere" (has text)
        >>>
        >>> df = filter_number_only_lines(df)

    Note:
        Recommended for line-level OCR cleanup.
        Removes page numbers and date artifacts before text analysis.
    """
    logger.info(f"Filtering number-only lines: {input_column}")

    original_count = len(df)

    # Pattern: Unicode numbers (\p{N}) and common separators (incl. space), or strict numbers only
    # \p{N} matches all Unicode number categories: digits, superscripts, subscripts, fractions
    pattern = r"^[\p{N}.,\-/: ]+$" if allow_separators else r"^\p{N}+$"

    # Use native Polars regex for efficiency
    # Filter: keep rows that do NOT match number-only pattern
    df = df.filter(
        ~(
            pl.col(input_column).str.strip_chars().str.len_chars().gt(0)
            & pl.col(input_column).str.strip_chars().str.contains(pattern)
        )
    )

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_lines_without_alphabetic_chars(
    df: pl.DataFrame,
    input_column: str = "text",
) -> pl.DataFrame:
    """
    Filter out lines that do not contain any alphabetic characters.

    Removes lines that are purely numeric, punctuation, or symbols.
    Useful for cleaning OCR output that may include non-text artifacts.

    Args:
        df: Input DataFrame
        input_column: Column to check (default: "text")

    Returns:
        DataFrame with non-alphabetic lines filtered out

    Example:
        >>> # These get REMOVED:
        >>> # "12345", "!!!", "-----"
        >>>
        >>> # These are KEPT:
        >>> # "Die Zeitung", "Am 12. März", "Hello, World!"
        >>>
        >>> df = filter_lines_without_alphabetic_chars(df)
    """
    logger.info(f"Filtering lines without alphabetic chars: {input_column}")

    original_count = len(df)

    # Use \p{L} to match any Unicode letter (more efficient than map_elements)
    df = df.filter(pl.col(input_column).str.contains(r"\p{L}"))

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_empty_lines(
    df: pl.DataFrame,
    input_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Filter out empty lines from the DataFrame.

    Args:
        df: Input DataFrame
        input_column: Column to check (default: text_column)

    Returns:
        DataFrame with empty lines filtered out
    """
    if input_column is None:
        input_column = "text"

    logger.info(f"Filtering empty lines: {input_column}")

    original_count = len(df)

    df = df.filter(pl.col(input_column).str.strip_chars() != "")

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df
