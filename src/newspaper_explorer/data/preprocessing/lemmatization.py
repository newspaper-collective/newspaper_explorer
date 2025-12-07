"""
Linguistic processing operations.

Provides advanced linguistic processing methods:
- Dehyphenation (line-break hyphen removal)
- Line-level dehyphenation (preserves line structure)
- Lemmatization (spaCy and GermaLemma)
"""

import logging
from typing import Optional

from germalemma import GermaLemma  # type: ignore
import polars as pl
import spacy
from spacy.tokens import Token
from tqdm import tqdm

logger = logging.getLogger(__name__)


def lemmatize_spacy(
    df: pl.DataFrame,
    input_column: str = "text",
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
        input_column: Column to process (default: "text")
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

    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing with spaCy: {input_column} → {output_column}")

    try:
        nlp = spacy.load(model, disable=["ner", "parser"])  # Faster: only need lemmatizer
    except OSError:
        logger.error(f"spaCy model '{model}' not found!")
        logger.error(f"Download it with: python -m spacy download {model}")
        raise

    texts = df[input_column].to_list()
    lemmatized_texts: list[str] = []

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
    input_column: str = "text",
    output_column: Optional[str] = None,
    spacy_model: str = "de_core_news_sm",
    batch_size: int = 1000,
) -> pl.DataFrame:
    """
    Lemmatize German text using GermaLemma with spaCy POS tagging.

    GermaLemma requires Part-of-Speech tagged words. This function uses spaCy
    for POS tagging and then applies GermaLemma for lemmatization. GermaLemma
    achieves 99.43% accuracy (vs spaCy's ~95%) but is slower.

    Valid POS tags for GermaLemma (STTS tagset):
    - 'N...' (nouns) → e.g., NN, NE
    - 'V...' (verbs) → e.g., VVFIN, VVINF, VVPP
    - 'ADJ...' (adjectives) → e.g., ADJA, ADJD
    - 'ADV...' (adverbs) → e.g., ADV

    For tokens with unsupported POS tags, the original token is returned.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_lemma)
        spacy_model: spaCy model for POS tagging (default: de_core_news_sm)
        batch_size: Batch size for spaCy processing (default: 1000)

    Returns:
        DataFrame with lemmatized text column

    Example:
        >>> # GermaLemma uses STTS tags from spaCy
        >>> # "Feinstaubbelastungen" (NN) → "Feinstaubbelastung"
        >>> # "ging" (VVFIN) → "gehen"
        >>> df = lemmatize_germalemma(df)
    """
    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing text: {input_column} → {output_column}")
    logger.info("Using spaCy for POS tagging + GermaLemma for lemmatization")

    # Load spaCy for POS tagging
    try:
        nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    except OSError:
        logger.error(f"spaCy model '{spacy_model}' not found!")
        logger.error(f"Download it with: python -m spacy download {spacy_model}")
        raise

    lemmatizer = GermaLemma()

    texts = df[input_column].to_list()
    lemmatized_texts: list[str] = []

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    def get_lemma(token: Token) -> str:
        """Get lemma for a token, preserving original if unsupported POS."""
        """Get lemma for a token, preserving original if unsupported POS."""
        tag = token.tag_
        if tag.startswith(("NN", "NE", "V", "ADJ", "ADV")):
            try:
                return str(lemmatizer.find_lemma(token.text, tag))  # type: ignore
            except ValueError:
                return token.text
        return token.text

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="GermaLemma lemmatization"):
        batch = texts[i : i + batch_size]

        for doc in nlp.pipe(batch, batch_size=batch_size):
            # Reconstruct text preserving original whitespace
            # token.whitespace_ contains the whitespace AFTER the token
            result = "".join(get_lemma(token) + token.whitespace_ for token in doc)
            lemmatized_texts.append(result)

    df = df.with_columns([pl.Series(name=output_column, values=lemmatized_texts)])

    logger.info(f"Lemmatized {len(df):,} rows with GermaLemma")
    return df
