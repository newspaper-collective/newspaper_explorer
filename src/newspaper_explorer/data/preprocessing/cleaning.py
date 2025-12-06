"""
Text cleaning operations.

Provides text transformation methods that modify content while keeping all rows:
- Whitespace normalization
- Lowercase conversion
- Punctuation removal
- Number removal
- Stopword removal
- OCR artifact cleanup
"""

import logging
import re
from typing import Optional

import polars as pl
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from unidecode import unidecode

logger = logging.getLogger(__name__)


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

    Note:
        Requires unidecode package: pip install unidecode
    """

    if output_column is None:
        output_column = f"{input_column}_no_diacritics"

    logger.info(f"Removing diacritics: {input_column} → {output_column}")

    # Apply unidecode to each text
    texts = df[input_column].to_list()
    processed = [unidecode(str(text)) if text else "" for text in texts]

    df = df.with_columns([pl.Series(output_column, processed)])

    logger.info(f"Removed diacritics from {len(df):,} rows")
    return df


def normalize_whitespace(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    *,
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


def remove_punctuation(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    keep_chars: str = "",
) -> pl.DataFrame:
    """
    Remove punctuation from text using Unicode categories.

    Uses \\p{P} (Unicode punctuation) to remove all punctuation marks while
    preserving letters from any language (German umlauts, French accents, etc.).

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
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
    if input_column is None:
        input_column = text_column
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

    # Use Unicode number category \p{N} to match all numeric characters
    # This includes ASCII digits, subscripts, superscripts, fractions, etc.
    df = df.with_columns([pl.col(input_column).str.replace_all(r"\p{N}+", "").alias(output_column)])

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
    Remove stopwords from text.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_nostop)
        language: Language code (default: "de" for German)

    Returns:
        DataFrame with stopwords removed
    """
    if input_column is None:
        input_column = text_column
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
