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
import re
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def remove_punctuation(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    keep_chars: str = "",
) -> pl.DataFrame:
    """
    Remove punctuation from text.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_nopunct)
        keep_chars: Characters to keep (e.g., "-'" to keep hyphens and apostrophes)

    Returns:
        DataFrame with punctuation removed
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_nopunct"

    logger.info(f"Removing punctuation: {input_column} → {output_column}")

    # Build regex pattern: remove all non-alphanumeric except spaces and keep_chars
    if keep_chars:
        pattern = f"[^a-zA-ZäöüÄÖÜß0-9\\s{re.escape(keep_chars)}]"
    else:
        pattern = "[^a-zA-ZäöüÄÖÜß0-9\\s]"

    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all(pattern, "")
            .str.replace_all(r"\s+", " ")  # Normalize whitespace
            .str.strip_chars()
            .alias(output_column)
        ]
    )

    logger.info(f"Removed punctuation from {len(df):,} rows")
    return df


def remove_numbers(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Remove numbers from text.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_nonum)

    Returns:
        DataFrame with numbers removed
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_nonum"

    logger.info(f"Removing numbers: {input_column} → {output_column}")

    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all(r"\d+", "")
            .str.replace_all(r"\s+", " ")  # Normalize whitespace
            .str.strip_chars()
            .alias(output_column)
        ]
    )

    logger.info(f"Removed numbers from {len(df):,} rows")
    return df


def remove_stopwords(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    language: str = "de",
) -> pl.DataFrame:
    """
    Remove stopwords using spaCy.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_nostop)
        language: Language code (default: "de" for German)

    Returns:
        DataFrame with stopwords removed

    Raises:
        ImportError: If spaCy is not installed
    """
    try:
        import spacy
        from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
        from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
    except ImportError:
        raise ImportError(
            "spaCy is required for stopword removal. " "Install with: pip install -e '.[nlp]'"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_nostop"

    logger.info(f"Removing stopwords ({language}): {input_column} → {output_column}")

    # Select stopwords
    if language == "de":
        stopwords = DE_STOP_WORDS
    elif language == "en":
        stopwords = EN_STOP_WORDS
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Create blank language model for tokenization
    nlp = spacy.blank(language)

    # Process in batches for efficiency
    texts = df[input_column].to_list()
    cleaned_texts = []

    batch_size = 10000
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for doc in nlp.pipe(batch, batch_size=1000):
            tokens = [token.text for token in doc if token.text.lower() not in stopwords]
            cleaned_texts.append(" ".join(tokens))

        if (i + batch_size) % 50000 == 0:
            logger.info(f"Processed {i + batch_size:,} / {len(texts):,} texts")

    # Add cleaned column
    df = df.with_columns([pl.Series(output_column, cleaned_texts)])

    logger.info(f"Removed stopwords from {len(df):,} rows")
    return df


def filter_by_length(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    min_length: int = 10,
    max_length: Optional[int] = None,
) -> pl.DataFrame:
    """
    Filter out texts that are too short or too long.

    Removes rows where text length is outside the specified range.
    Useful for removing OCR artifacts, headers, footers, or malformed entries.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to check length (default: text_column)
        min_length: Minimum text length in characters (default: 10)
        max_length: Maximum text length in characters (default: None = no limit)

    Returns:
        DataFrame with short/long texts filtered out

    Example:
        >>> # Remove very short texts (likely artifacts)
        >>> df = filter_by_length(df, min_length=20)
        >>> # Remove both short and very long texts
        >>> df = filter_by_length(df, min_length=10, max_length=10000)
    """
    if input_column is None:
        input_column = text_column

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
    logger.info(f"Filtered out {filtered_count:,} rows ({filtered_count/original_count*100:.1f}%)")
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def clean_ocr_artifacts(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    allowed_chars: Optional[str] = None,
    replace_with: str = "",
) -> pl.DataFrame:
    """
    Remove OCR artifacts and invalid characters from text.

    Removes characters that are likely OCR errors or encoding issues.
    By default, keeps only common German text characters, spaces, and basic punctuation.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_clean)
        allowed_chars: Regex pattern of allowed characters (default: German letters, digits, common punctuation)
        replace_with: What to replace invalid characters with (default: empty string)

    Returns:
        DataFrame with cleaned text

    Example:
        >>> # Use default pattern (keeps German text + common punctuation)
        >>> df = clean_ocr_artifacts(df)
        >>> # Custom pattern to be more restrictive
        >>> df = clean_ocr_artifacts(df, allowed_chars=r"[a-zA-ZäöüÄÖÜß0-9\\s.,!?-]")
        >>> # Replace invalid chars with space instead of removing
        >>> df = clean_ocr_artifacts(df, replace_with=" ")
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_clean"

    logger.info(f"Cleaning OCR artifacts: {input_column} → {output_column}")

    # Default pattern: German letters, digits, common punctuation, whitespace
    # This handles most legitimate text while removing OCR garbage
    if allowed_chars is None:
        # Include:
        # - German letters (with umlauts and ß)
        # - ASCII letters and digits
        # - Common punctuation: . , ! ? ; : - ( ) " ' /
        # - Whitespace
        allowed_chars = r"[a-zA-ZäöüÄÖÜßẞ0-9\s.,!?;:\-()\"'/]"

    # Create pattern to match anything NOT in allowed set
    pattern = f"[^{allowed_chars[1:-1]}]"

    # Apply cleaning
    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all(pattern, replace_with)
            .str.replace_all(r"\s+", " ")  # Normalize whitespace after cleanup
            .str.strip_chars()  # Remove leading/trailing whitespace
            .alias(output_column)
        ]
    )

    logger.info(f"Cleaned OCR artifacts from {len(df):,} rows")
    return df
