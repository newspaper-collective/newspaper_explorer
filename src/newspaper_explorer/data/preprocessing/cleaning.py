"""
Text cleaning operations.

Provides text transformation methods that modify content while keeping all rows:
- Whitespace normalization
- Lowercase conversion
- Punctuation removal
- Number removal
- Stopword removal
- Character allowlist filtering
"""

import logging
import re
from typing import Optional

import polars as pl
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from unidecode import unidecode

logger = logging.getLogger(__name__)


def only_keep_allowed_chars(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    allowed_chars: Optional[str] = None,
    replace_with: str = "",
) -> pl.DataFrame:
    """
    Keep only characters matching an allowlist pattern, removing all others.

    This is a whitelist-based character filter. Characters not in the allowed
    set are removed (or replaced with a specified string).

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_filtered)
        allowed_chars: Regex character class of allowed characters
                      (default: German letters, digits, common punctuation, whitespace)
        replace_with: What to replace disallowed characters with (default: empty string)

    Returns:
        DataFrame with filtered text

    Example:
        >>> # Use default allowlist (German text + common punctuation)
        >>> df = only_keep_allowed_chars(df)
        >>> # Custom pattern to be more restrictive
        >>> df = only_keep_allowed_chars(df, allowed_chars=r"[a-zA-ZäöüÄÖÜß0-9\\s.,!?-]")
        >>> # Replace disallowed chars with space instead of removing
        >>> df = only_keep_allowed_chars(df, replace_with=" ")
    """
    if output_column is None:
        output_column = f"{input_column}_filtered"

    logger.info(f"Filtering to allowed chars: {input_column} → {output_column}")

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

    logger.info(f"Filtered to allowed chars for {len(df):,} rows")
    return df


def remove_long_words(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    max_word_length: int = 45,
) -> pl.DataFrame:
    """
    Remove words exceeding typical German compound length (likely OCR merge errors).

    German compound words can be long (Donaudampfschifffahrtsgesellschaft = 36 chars)
    but rarely exceed 45 characters. Longer strings are likely OCR errors where
    multiple words were incorrectly merged.

    This is a cleaning function - it removes problematic words while preserving
    valid content. Lines that become empty after cleaning will be caught by
    filter_empty_lines().

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: overwrites input_column)
        max_word_length: Maximum allowed word length (default: 45)

    Returns:
        DataFrame with long words removed from text

    Example:
        >>> # Valid long German compounds (KEPT):
        >>> # "Donaudampfschifffahrtsgesellschaft" (36 chars)
        >>>
        >>> # OCR merge errors (word REMOVED, line KEPT):
        >>> # "Die DieZeitungerscheinttäglich Zeitung" → "Die Zeitung"
        >>>
        >>> df = remove_long_words(df, max_word_length=45)

    Note:
        Based on normalize.md: "Words >45 characters likely artifacts
        (German compounds rarely exceed this)"

        Use filter_empty_lines() after this to remove lines that became empty.
    """
    if output_column is None:
        output_column = input_column

    logger.info(f"Removing words >{max_word_length} chars: {input_column}")

    def remove_long(text: str) -> str:
        if not text:
            return text

        words = text.split()
        filtered_words = [word for word in words if len(word) <= max_word_length]
        return " ".join(filtered_words)

    # Count words that will be removed for logging
    total_words = 0
    removed_words = 0
    for text in df[input_column].to_list():
        if text:
            words = text.split()
            total_words += len(words)
            removed_words += sum(1 for w in words if len(w) > max_word_length)

    df = df.with_columns(
        pl.col(input_column).map_elements(remove_long, return_dtype=pl.Utf8).alias(output_column)
    )

    if total_words > 0:
        logger.info(
            f"Removed {removed_words:,} words ({removed_words / total_words * 100:.2f}%) "
            f"exceeding {max_word_length} chars"
        )

    return df


def remove_garbage_words(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    min_unique_chars: int = 3,
    max_repetition_ratio: float = 0.3,
    min_word_length: int = 2,
) -> pl.DataFrame:
    """
    Remove words with excessive character repetition (OCR garbage).

    Detects OCR artifacts like "ssss", "jjjj", "sssuuusss" that have low
    character vocabulary (many repetitions of few characters).

    Two detection methods:
    1. Absolute: word has fewer than min_unique_chars unique characters
    2. Ratio: unique_chars / word_length ≤ max_repetition_ratio

    Examples:
        - "sssuuusss" → removed (2 unique / 9 total = 0.22 ratio)
        - "jjjj" → removed (1 unique / 4 total = 0.25 ratio)
        - "|||" → removed (1 unique / 3 total = 0.33 ratio)
        - "Die Zeitung" → kept (normal text)
        - "Schifffahrt" → kept (valid German compound with fff)

    This is a cleaning function - it removes problematic words while preserving
    valid content. Lines that become empty after cleaning will be caught by
    filter_empty_lines().

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: overwrites input_column)
        min_unique_chars: Minimum unique characters per word (default: 3)
        max_repetition_ratio: Maximum ratio of unique to total chars (default: 0.3)
        min_word_length: Minimum word length to check for garbage (default: 2).
                        Words shorter than this are not checked.

    Returns:
        DataFrame with garbage words removed from text

    Example:
        >>> # "Die ssss Zeitung" → "Die Zeitung"
        >>> df = remove_garbage_words(df)
        >>>
        >>> # Use filter_empty_lines() after to remove lines that became empty
        >>> df = filter_empty_lines(df)

    Note:
        Preserves valid German compounds like "Schifffahrt" (contains fff)
        which have legitimate character repetition in context.
    """
    if output_column is None:
        output_column = input_column

    logger.info(
        f"Removing garbage words: {input_column} "
        f"(min_unique={min_unique_chars}, max_ratio={max_repetition_ratio})"
    )

    def is_garbage_word(word: str) -> bool:
        """Check if a word is OCR garbage based on character repetition."""
        if not word or len(word) < min_word_length:
            return False

        # Count unique characters
        unique_chars = len(set(word))

        # Method 1: Absolute minimum unique characters
        if unique_chars < min_unique_chars:
            return True

        # Method 2: Ratio of unique to total characters
        ratio = unique_chars / len(word)
        return ratio <= max_repetition_ratio

    def clean_line(text: str) -> str:
        """Remove garbage words from a line of text."""
        if not text or not text.strip():
            return text

        words = text.split()
        cleaned_words = [word for word in words if not is_garbage_word(word)]

        return " ".join(cleaned_words)

    # Count words that will be removed for logging
    total_words = 0
    removed_words = 0
    for text in df[input_column].to_list():
        if text:
            words = text.split()
            total_words += len(words)
            removed_words += sum(1 for w in words if is_garbage_word(w))

    df = df.with_columns(
        pl.col(input_column).map_elements(clean_line, return_dtype=pl.Utf8).alias(output_column)
    )

    if total_words > 0:
        logger.info(
            f"Removed {removed_words:,} garbage words ({removed_words / total_words * 100:.2f}%)"
        )

    return df


def remove_diacritics(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Remove diacritics from text using unidecode.

    Converts accented characters to their ASCII equivalents:
    - ä → a, ö → o, ü → u
    - é → e, à → a, etc.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_no_diacritics)

    Returns:
        DataFrame with diacritics removed

    Example:
        >>> df = remove_diacritics(df)
        >>> # "Münchner Straße" → "Munchner Strasse"

    """

    if output_column is None:
        output_column = f"{input_column}_no_diacritics"

    logger.info(f"Removing diacritics: {input_column} → {output_column}")

    # Apply unidecode via map_elements (no native Polars equivalent exists)
    df = df.with_columns(
        pl.col(input_column)
        .map_elements(lambda x: unidecode(str(x)) if x else "", return_dtype=pl.Utf8)
        .alias(output_column)
    )

    logger.info(f"Removed diacritics from {len(df):,} rows")
    return df


def remove_punctuation(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    keep_chars: str = "",
) -> pl.DataFrame:
    """
    Remove punctuation from text using Unicode categories.

    Uses \\p{P} (Unicode punctuation) to remove all punctuation marks while
    preserving letters from any language (German umlauts, French accents, etc.).

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_nopunct)
        keep_chars: Punctuation characters to preserve (e.g., "-'" to keep hyphens and apostrophes)

    Returns:
        DataFrame with punctuation removed

    Example:
        >>> # Remove all punctuation
        >>> df = remove_punctuation(df)
        >>> # Keep hyphens and apostrophes
        >>> df = remove_punctuation(df, keep_chars="-'")
    """
    if output_column is None:
        output_column = f"{input_column}_nopunct"

    logger.info(f"Removing punctuation: {input_column} → {output_column}")

    # Use Unicode punctuation category \p{P}
    # This correctly handles all languages without hardcoding character sets
    if keep_chars:
        # Use character class subtraction: [\p{P}--[chars_to_keep]]
        # This removes all punctuation EXCEPT the specified characters
        escaped = re.escape(keep_chars)
        pattern = f"[\\p{{P}}--[{escaped}]]"
    else:
        pattern = r"\p{P}"

    df = df.with_columns([pl.col(input_column).str.replace_all(pattern, "").alias(output_column)])

    logger.info(f"Removed punctuation from {len(df):,} rows")
    return df


def remove_numbers(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Remove numbers from text.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_nonum)

    Returns:
        DataFrame with numbers removed
    """
    if output_column is None:
        output_column = f"{input_column}_nonum"

    logger.info(f"Removing numbers: {input_column} → {output_column}")

    # Use Unicode number category \p{N} to match all numeric characters
    # This includes ASCII digits, subscripts, superscripts, fractions, etc.
    df = df.with_columns([pl.col(input_column).str.replace_all(r"\p{N}+", "").alias(output_column)])

    logger.info(f"Removed numbers from {len(df):,} rows")
    return df


def remove_stopwords(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    language: str = "de",
) -> pl.DataFrame:
    """
    Remove stopwords from text.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_nostop)
        language: Language code (default: "de" for German)

    Returns:
        DataFrame with stopwords removed
    """
    if output_column is None:
        output_column = f"{input_column}_nostop"

    logger.info(f"Removing stopwords ({language}): {input_column} → {output_column}")

    # Get stopwords set
    if language == "de":
        stopwords = DE_STOP_WORDS
    elif language == "en":
        stopwords = EN_STOP_WORDS
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Convert to list for Polars is_in()
    stopwords_list = list(stopwords)

    # Native Polars: split, filter stopwords, rejoin
    df = df.with_columns(
        pl.col(input_column)
        .str.split(" ")
        .list.eval(pl.element().filter(~pl.element().str.to_lowercase().is_in(stopwords_list)))
        .list.join(" ")
        .alias(output_column)
    )

    logger.info(f"Removed stopwords from {len(df):,} rows")
    return df
