"""
Text processing utilities for newspaper data.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def split_into_sentences(
    df: pl.DataFrame,
    text_column: str = "text",
    model: str = "de_core_news_sm",
    batch_size: int = 1000,
) -> pl.DataFrame:
    """
    Split text into sentences using spaCy for German language processing.

    This function uses spaCy's German language model to split text into sentences.
    It processes the data in batches for efficiency and expands each row into
    multiple rows (one per sentence).

    Args:
        df: Polars DataFrame containing text to split
        text_column: Name of the column containing text to split (default: "text")
        model: spaCy model to use (default: "de_core_news_sm")
               You need to install it first: python -m spacy download de_core_news_sm
        batch_size: Number of texts to process at once (default: 1000)

    Returns:
        Polars DataFrame with one row per sentence, containing:
        - All original columns
        - sentence: the extracted sentence text
        - sentence_id: index of sentence within the original text (0-based)
        - sentence_count: total number of sentences in the original text

    Example:
        >>> # First aggregate text blocks
        >>> blocks_df = load_and_aggregate_textblocks("lines.parquet")
        >>> # Then split into sentences
        >>> sentences_df = split_into_sentences(blocks_df, text_column="text")
        >>> print(f"Expanded {len(blocks_df)} blocks into {len(sentences_df)} sentences")

    Raises:
        ImportError: If spaCy is not installed
        OSError: If the specified spaCy model is not installed
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for sentence splitting. Install it with: pip install spacy"
        )

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    logger.info(f"Loading spaCy model: {model}")

    try:
        nlp = spacy.load(model)
    except OSError:
        raise OSError(
            f"spaCy model '{model}' not found. Install it with: "
            f"python -m spacy download {model}"
        )

    # Disable unnecessary pipeline components for speed
    # We only need sentence segmentation
    nlp.select_pipes(enable=["tok2vec", "morphologizer", "parser", "lemmatizer", "attribute_ruler"])

    logger.info(f"Processing {len(df)} texts in batches of {batch_size}")

    # Convert to pandas for easier processing (spaCy works better with pandas)
    df_pandas = df.to_pandas()

    # Prepare list to collect results
    results = []

    # Process in batches
    texts = df_pandas[text_column].tolist()

    for doc_idx, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        # Get original row data
        row_data = df_pandas.iloc[doc_idx].to_dict()

        # Split into sentences
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Create a row for each sentence
        for sent_idx, sentence in enumerate(sentences):
            result_row = row_data.copy()
            result_row["sentence"] = sentence
            result_row["sentence_id"] = sent_idx
            result_row["sentence_count"] = len(sentences)
            results.append(result_row)

        # Log progress every 1000 documents
        if (doc_idx + 1) % 1000 == 0:
            logger.info(f"Processed {doc_idx + 1}/{len(texts)} texts")

    logger.info(f"Split {len(df)} texts into {len(results)} sentences")

    # Convert back to Polars DataFrame
    result_df = pl.DataFrame(results)

    return result_df
