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

    By default, preserves % (common for percentages in newspaper text).

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_nopunct)
        keep_chars: Additional characters to keep (e.g., "-'" to keep hyphens and apostrophes)

    Returns:
        DataFrame with punctuation removed

    Example:
        >>> # Remove all punctuation except %
        >>> df = remove_punctuation(df)
        >>> # Keep hyphens and apostrophes too
        >>> df = remove_punctuation(df, keep_chars="-'")
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_nopunct"

    logger.info(f"Removing punctuation: {input_column} → {output_column}")

    # Build regex pattern: remove all non-alphanumeric except spaces and keep_chars
    # Default: keep % for percentages (common in newspaper text)
    if keep_chars:
        pattern = f"[^a-zA-ZäöüÄÖÜß0-9\\s%{re.escape(keep_chars)}]"
    else:
        pattern = "[^a-zA-ZäöüÄÖÜß0-9\\s%]"

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
            "spaCy is required for stopword removal. Install with: pip install -e '.[nlp]'"
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
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_by_word_count(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
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
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to check word count (default: text_column)
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
    if input_column is None:
        input_column = text_column

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


def filter_repeating_chars(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    min_unique_chars: int = 3,
    max_repetition_ratio: float = 0.3,
    remove_lines: bool = True,
) -> pl.DataFrame:
    """
    Filter or remove words with excessive character repetition (OCR garbage).

    Detects OCR artifacts like "ssss", "jjjj", "sssuuusss" that have low
    character vocabulary (many repetitions of few characters).

    Two detection methods:
    1. Absolute: word has fewer than min_unique_chars unique characters
    2. Ratio: unique_chars / word_length ≤ max_repetition_ratio

    Examples:
        - "sssuuusss" → filtered (2 unique / 9 total = 0.22 ratio)
        - "jjjj" → filtered (1 unique / 4 total = 0.25 ratio)
        - "|||" → filtered (1 unique / 3 total = 0.33 ratio)
        - "Die Zeitung" → kept (normal text)
        - "Schifffahrt" → kept (valid German compound with fff)

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_filtered)
        min_unique_chars: Minimum unique characters per word (default: 3)
        max_repetition_ratio: Maximum ratio of unique to total chars (default: 0.3)
        remove_lines: If True, remove lines where all words are garbage.
                     If False, just remove garbage words from lines.

    Returns:
        DataFrame with repeating character patterns filtered out

    Example:
        >>> # Remove garbage words, keep valid words
        >>> df = filter_repeating_chars(df)
        >>> # "Die ssss Zeitung" → "Die Zeitung"
        >>>
        >>> # Remove entire lines if all words are garbage
        >>> df = filter_repeating_chars(df, remove_lines=True)
        >>> # "ssss jjjj" → (line removed)

    Note:
        Preserves valid German compounds like "Schifffahrt" (contains fff)
        which have legitimate character repetition in context.
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_filtered"

    logger.info(
        f"Filtering repeating chars: {input_column} "
        f"(min_unique={min_unique_chars}, max_ratio={max_repetition_ratio})"
    )

    original_count = len(df)

    def is_garbage_word(word: str) -> bool:
        """Check if a word is OCR garbage based on character repetition."""
        if not word or len(word) < 2:
            return False

        # Count unique characters
        unique_chars = len(set(word))

        # Method 1: Absolute minimum unique characters
        if unique_chars < min_unique_chars:
            return True

        # Method 2: Ratio of unique to total characters
        ratio = unique_chars / len(word)
        if ratio <= max_repetition_ratio:
            return True

        return False

    def filter_line(text: str) -> str:
        """Filter garbage words from a line of text."""
        if not text or not text.strip():
            return text

        words = text.split()
        filtered_words = [word for word in words if not is_garbage_word(word)]

        return " ".join(filtered_words)

    # Apply filtering to each line
    texts = df[input_column].to_list()
    filtered_texts = [filter_line(text) for text in texts]

    # Add filtered column
    df = df.with_columns([pl.Series(output_column, filtered_texts)])

    if remove_lines:
        # Remove lines that became empty after filtering
        df = df.filter(pl.col(output_column).str.strip_chars() != "")
        removed_count = original_count - len(df)
        logger.info(
            f"Removed {removed_count:,} lines with only garbage words "
            f"({removed_count / original_count * 100:.1f}%)"
        )
        logger.info(f"Remaining: {len(df):,} rows")
    else:
        logger.info(f"Filtered garbage words from {len(df):,} rows")

    return df


def filter_number_only_lines(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    allow_separators: bool = True,
) -> pl.DataFrame:
    """
    Filter out lines containing only numbers and optional separators.

    These are typically page numbers, dates, or OCR artifacts:
    - "123" (page numbers)
    - "1.234" (numbers with periods)
    - "12-34-56" (dates or reference numbers)
    - "1,000" (formatted numbers)

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to check (default: text_column)
        allow_separators: If True, allows common separators (., -, /, :, ,) with numbers
                         If False, only pure digit strings are filtered

    Returns:
        DataFrame with number-only lines filtered out

    Example:
        >>> # These get REMOVED:
        >>> # "123", "45.67", "1-2-3", "1,000"
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
    if input_column is None:
        input_column = text_column

    logger.info(f"Filtering number-only lines: {input_column}")

    original_count = len(df)

    if allow_separators:
        # Pattern: only digits and common separators (., -, /, :, ,)
        pattern = r"^[\d.,\-/:]+$"
    else:
        # Pattern: only digits (strict)
        pattern = r"^\d+$"

    def is_number_only(text: str) -> bool:
        if not text or not text.strip():
            return False  # Empty text, don't filter

        text = text.strip()

        # Check pattern
        if re.match(pattern, text):
            # Additional check: must contain at least one digit
            return any(c.isdigit() for c in text)
        return False

    # Filter: keep rows that are NOT number-only
    mask = ~pl.col(input_column).map_elements(is_number_only, return_dtype=pl.Boolean)
    df = df.filter(mask)

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_by_char_token_ratio(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    max_ratio: float = 8.0,
    min_tokens: int = 2,
) -> pl.DataFrame:
    """
    Filter lines with suspiciously high character-to-token ratio.

    Clean German text averages 5-7 characters per token.
    Ratios >8:1 indicate OCR fragmentation or artifacts.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to check (default: text_column)
        max_ratio: Maximum allowed char/token ratio (default: 8.0)
        min_tokens: Minimum tokens required to check ratio (default: 2)

    Returns:
        DataFrame with high-ratio lines filtered out

    Example:
        >>> # Clean German: 5-7 chars/token
        >>> # "Die Zeitung erscheint täglich"
        >>> # 30 chars / 4 tokens = 7.5 (KEPT)
        >>>
        >>> # OCR fragmentation: >8 chars/token
        >>> # "DieZeitungerscheinttäglich"
        >>> # 27 chars / 1 token = 27 (REMOVED)
        >>>
        >>> df = filter_by_char_token_ratio(df, max_ratio=8.0)

    Note:
        Based on normalize.md validation recommendations.
        Token-to-character ratio is a quality indicator for OCR text.
    """
    if input_column is None:
        input_column = text_column

    logger.info(f"Filtering by char/token ratio (max={max_ratio}): {input_column}")

    original_count = len(df)

    def has_valid_ratio(text: str) -> bool:
        if not text:
            return True  # Keep empty

        tokens = text.split()
        if len(tokens) < min_tokens:
            return True  # Skip very short texts

        char_count = len(text)
        token_count = len(tokens)
        ratio = char_count / token_count

        return ratio <= max_ratio

    mask = pl.col(input_column).map_elements(has_valid_ratio, return_dtype=pl.Boolean)
    df = df.filter(mask)

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def filter_excessive_word_length(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    max_word_length: int = 45,
) -> pl.DataFrame:
    """
    Filter lines containing words exceeding typical German compound length.

    German compound words can be long (Donaudampfschifffahrtsgesellschaft = 36 chars)
    but rarely exceed 45 characters. Longer strings are likely OCR errors where
    multiple words were incorrectly merged.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to check (default: text_column)
        max_word_length: Maximum allowed word length (default: 45)

    Returns:
        DataFrame with excessive-word-length lines filtered out

    Example:
        >>> # Valid long German compounds (KEPT):
        >>> # "Donaudampfschifffahrtsgesellschaft" (36 chars)
        >>>
        >>> # OCR errors (REMOVED):
        >>> # "DieZeitungerscheinttäglichundberichtetüberdiePolitik" (merged words)
        >>>
        >>> df = filter_excessive_word_length(df, max_word_length=45)

    Note:
        Based on normalize.md: "Words >45 characters likely artifacts
        (German compounds rarely exceed this)"
    """
    if input_column is None:
        input_column = text_column

    logger.info(f"Filtering lines with words >{max_word_length} chars: {input_column}")

    original_count = len(df)

    def has_valid_word_lengths(text: str) -> bool:
        if not text:
            return True

        words = text.split()
        return all(len(word) <= max_word_length for word in words)

    mask = pl.col(input_column).map_elements(has_valid_word_lengths, return_dtype=pl.Boolean)
    df = df.filter(mask)

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df
