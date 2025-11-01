"""
Linguistic processing operations.

Provides advanced linguistic processing methods:
- Dehyphenation (line-break hyphen removal)
- Lemmatization (spaCy and GermaLemma)
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def dehyphenate(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    language: str = "de_DE",
) -> pl.DataFrame:
    """
    Remove hyphenation from text using pyphen.

    Newspapers often split words across line breaks with hyphens.
    This method intelligently removes those hyphens while preserving
    legitimate hyphens in compound words.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_dehyphen)
        language: Language code for pyphen (default: de_DE)

    Returns:
        DataFrame with dehyphenated text column

    Raises:
        ImportError: If pyphen is not installed

    Example:
        >>> # "Zeitung-\nspapier" → "Zeitungspapier"
        >>> # But "Nord-Süd" stays "Nord-Süd"
    """
    try:
        import pyphen
    except ImportError:
        raise ImportError(
            "pyphen is required for dehyphenation. " "Install with: pip install pyphen"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_dehyphen"

    logger.info(f"Dehyphenating text: {input_column} → {output_column}")

    dic = pyphen.Pyphen(lang=language)

    def dehyphenate_text(text: str) -> str:
        """Remove line-break hyphens from text."""
        if not text:
            return text

        # Pattern: word-hyphen-newline-word
        # Replace "Wort-\nTeil" with "WortTeil"
        import re

        # Remove hyphen + newline/space combinations
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"-\s{2,}", "", text)  # Multiple spaces after hyphen

        return text

    df = df.with_columns(
        [
            pl.col(input_column)
            .map_elements(dehyphenate_text, return_dtype=pl.Utf8)
            .alias(output_column)
        ]
    )

    logger.info(f"Dehyphenated {len(df):,} rows")
    return df


def lemmatize_spacy(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    model: str = "de_core_news_sm",
    batch_size: int = 1000,
) -> pl.DataFrame:
    """
    Lemmatize German text using spaCy (FAST).

    Much faster than GermaLemma (100x) and context-aware.
    Uses part-of-speech information for better lemmatization.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_lemma)
        model: spaCy model to use (default: de_core_news_sm)
        batch_size: Batch size for processing (default: 1000)

    Returns:
        DataFrame with lemmatized text column

    Raises:
        ImportError: If spaCy is not installed
        OSError: If spaCy model is not downloaded

    Example:
        >>> # First download model: python -m spacy download de_core_news_sm
        >>> df = lemmatize_spacy(df, batch_size=5000)
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for lemmatization. " "Install with: pip install -e '.[nlp]'"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing with spaCy: {input_column} → {output_column}")

    try:
        nlp = spacy.load(model, disable=["ner", "parser"])  # Faster: only need lemmatizer
    except OSError:
        logger.error(f"spaCy model '{model}' not found!")
        logger.error(f"Download it with: python -m spacy download {model}")
        raise

    from tqdm import tqdm

    texts = df[input_column].to_list()
    lemmatized_texts = []

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="spaCy lemmatization"):
        batch = texts[i : i + batch_size]

        # Process batch
        for doc in nlp.pipe(batch, batch_size=batch_size):
            lemmas = [token.lemma_ for token in doc]
            lemmatized_texts.append(" ".join(lemmas))

    df = df.with_columns([pl.Series(name=output_column, values=lemmatized_texts)])

    logger.info(f"Lemmatized {len(df):,} rows with spaCy")
    return df


def lemmatize_germalemma(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Lemmatize German text using GermaLemma.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_lemma)

    Returns:
        DataFrame with lemmatized text column

    Raises:
        ImportError: If germalemma is not installed
    """
    try:
        from germalemma import GermaLemma
    except ImportError:
        raise ImportError(
            "germalemma is required for lemmatization. " "Install with: pip install -e '.[nlp]'"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing text: {input_column} → {output_column}")
    logger.warning("Lemmatization is slow and may take considerable time!")

    lemmatizer = GermaLemma()

    texts = df[input_column].to_list()
    lemmatized_texts = []

    for i, text in enumerate(texts):
        if text:
            tokens = str(text).split()
            # Assume NOUN for all tokens (simplification)
            lemmas = [lemmatizer.find_lemma(token, "NOUN") for token in tokens]
            lemmatized_texts.append(" ".join(lemmas))
        else:
            lemmatized_texts.append("")

        if (i + 1) % 10000 == 0:
            logger.info(f"Lemmatized {i + 1:,} / {len(texts):,} texts")

    df = df.with_columns([pl.Series(output_column, lemmatized_texts)])

    logger.info(f"Lemmatized {len(df):,} rows")
    return df
