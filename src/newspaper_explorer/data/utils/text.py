"""
Text processing utilities for newspaper data.
Provides functions for loading, aggregating, and processing text data.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

# Supported Transnormer models for historical German text normalization
TRANSNORMER_MODELS = {
    "19c": "ybracke/transnormer-19c-beta-v02",  # 1780-1899
    "18-19c": "ybracke/transnormer-18-19c-beta-v01",  # 1700-1899
}


def load_and_aggregate_textblocks(
    parquet_path: Union[str, Path],
    group_by: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    auto_save: bool = True,
) -> pl.DataFrame:
    """
    Load parquet file and aggregate text lines by text blocks.

    This function loads a parquet file containing line-level newspaper data
    and aggregates the text lines by text blocks (or other grouping criteria).

    The output will be saved to data/processed/{source_name}/text/textblocks.parquet
    (unless save_path is explicitly provided or auto_save is False).

    All metadata (text_block_id, page_id, date, filename, newspaper_id,
    newspaper_title, year, month, day, and spatial information) is preserved.

    Args:
        parquet_path: Path to the parquet file
        group_by: Columns to group by. Default is ["text_block_id", "page_id", "date"]
        sort_by: Columns to sort by within each group before aggregating.
                 Default is ["y", "x"] to maintain reading order
        save_path: If provided, save the result to this parquet file (overrides auto_save)
        auto_save: If True, automatically save to
                   data/processed/{source}/text/textblocks.parquet

    Returns:
        Polars DataFrame with aggregated text blocks containing:
        - All grouping columns
        - text: concatenated text from all lines in the block
        - line_count: number of lines in the block
        - avg_x, avg_y: average coordinates
        - min_x, min_y, max_x, max_y: bounding box coordinates

    Example:
        >>> # Load and auto-save text blocks
        >>> df = load_and_aggregate_textblocks("data/raw/der_tag/text/lines.parquet")
        >>> # Saved to: data/processed/der_tag/text/textblocks.parquet
        >>>
        >>> # Get text blocks for a specific date
        >>> filtered = df.filter(pl.col("date") == "1901-01-08")
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logger.info(f"Loading parquet file: {parquet_path}")

    # Load the parquet file
    df = pl.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} lines from parquet")

    # Set default grouping and sorting
    if group_by is None:
        group_by = ["text_block_id", "page_id", "date"]

    if sort_by is None:
        sort_by = ["y", "x"]

    # Check if required columns exist
    required_cols = set(group_by + sort_by + ["text"])
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Aggregating text blocks by: {group_by}")

    # Sort by reading order within each block
    df = df.sort(group_by + sort_by)

    # Aggregate text blocks
    aggregated = df.group_by(group_by, maintain_order=True).agg(
        [
            # Concatenate text with space separator
            pl.col("text").str.concat(" ").alias("text"),
            # Count lines in block
            pl.count().alias("line_count"),
            # Average coordinates
            pl.col("x").mean().alias("avg_x"),
            pl.col("y").mean().alias("avg_y"),
            # Bounding box
            pl.col("x").min().alias("min_x"),
            pl.col("y").min().alias("min_y"),
            pl.col("x").max().alias("max_x"),
            pl.col("y").max().alias("max_y"),
            # Keep other metadata (take first value from group)
            pl.col("filename").first().alias("filename"),
            pl.col("newspaper_id").first().alias("newspaper_id"),
            pl.col("newspaper_title").first().alias("newspaper_title"),
            pl.col("year").first().alias("year"),
            pl.col("month").first().alias("month"),
            pl.col("day").first().alias("day"),
        ]
    )

    logger.info(f"Aggregated into {len(aggregated)} text blocks")

    # Determine save path
    if save_path:
        final_save_path = Path(save_path)
    elif auto_save:
        # Extract source name from path (e.g., "der_tag" from "data/raw/der_tag/text/lines.parquet")
        parts = parquet_path.parts
        try:
            # Find "raw" in path and get the next part as source name
            raw_idx = parts.index("raw")
            source_name = parts[raw_idx + 1]
        except (ValueError, IndexError):
            # Fallback: try to infer from path structure
            logger.warning("Could not extract source name from path, using 'unknown'")
            source_name = "unknown"

        # Construct output path
        final_save_path = Path("data") / "processed" / source_name / "text" / "textblocks.parquet"
        logger.info(f"Auto-save enabled: will save to {final_save_path}")
    else:
        final_save_path = None

    # Save if path is determined
    if final_save_path:
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {final_save_path}")
        aggregated.write_parquet(final_save_path, compression="zstd")
        logger.info(f"Saved {len(aggregated)} text blocks to {final_save_path}")

    return aggregated


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


def normalize_text(
    df: pl.DataFrame,
    text_column: str = "sentence",
    model: str = "19c",
    batch_size: int = 32,
    num_beams: int = 4,
    max_length: int = 128,
    output_column: str = "normalized_text",
) -> pl.DataFrame:
    """
    Normalize historical German text using Transnormer models.

    This function uses pre-trained Transnormer models to normalize historical German text
    to modern German orthography. This is particularly useful for 18th and 19th century
    newspaper text which uses outdated spelling conventions.

    The normalization handles:
    - Outdated characters (ſ -> s, aͤ -> ä, etc.)
    - Historical spelling conventions
    - Grammatical modernization

    Args:
        df: Polars DataFrame containing text to normalize
        text_column: Name of the column containing text to normalize (default: "sentence")
        model: Model to use. Options:
               - "19c": For text from 1780-1899 (transnormer-19c-beta-v02)
               - "18-19c": For text from 1700-1899 (transnormer-18-19c-beta-v01)
               - Or provide full model name from Hugging Face Hub
        batch_size: Number of texts to process at once (default: 32)
        num_beams: Number of beams for beam search (default: 4, higher = better quality)
        max_length: Maximum length of normalized text (default: 128)
        output_column: Name for the output column (default: "normalized_text")

    Returns:
        Polars DataFrame with original columns plus the normalized text column

    Example:
        >>> # Normalize sentences from 19th century newspaper
        >>> sentences_df = split_into_sentences(blocks_df)
        >>> normalized_df = normalize_text(
        ...     sentences_df,
        ...     text_column="sentence",
        ...     model="19c"
        ... )
        >>> # Compare original and normalized
        >>> print(normalized_df.select(["sentence", "normalized_text"]).head())

    Raises:
        ImportError: If transformers is not installed
        ValueError: If specified column not found or model not recognized

    Note:
        First time use will download the model (~500MB). Subsequent uses will
        use the cached model. Install with: pip install newspaper-explorer[normalize]
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Transformers is required for text normalization. "
            "Install it with: pip install newspaper-explorer[normalize] "
            "or: pip install transformers torch"
        )

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Resolve model name
    if model in TRANSNORMER_MODELS:
        model_name = TRANSNORMER_MODELS[model]
        logger.info(f"Using Transnormer model '{model}': {model_name}")
    else:
        model_name = model
        logger.info(f"Using custom model: {model_name}")

    logger.info("Loading normalization model (this may take a while on first run)")

    try:
        # Initialize the pipeline
        transnormer = pipeline(model=model_name, device=-1)  # -1 = CPU, 0 = GPU
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    logger.info(f"Normalizing {len(df)} texts in batches of {batch_size}")

    # Convert to pandas for easier processing
    df_pandas = df.to_pandas()

    # Get texts to normalize
    texts = df_pandas[text_column].tolist()

    # Process in batches
    normalized_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Skip empty texts
        batch_processed = []
        for text in batch:
            if text and text.strip():
                result = transnormer(
                    text, num_beams=num_beams, max_length=max_length, truncation=True
                )
                # Extract the generated text from the result
                normalized = result[0]["generated_text"] if result else text
                batch_processed.append(normalized)
            else:
                batch_processed.append(text)  # Keep empty texts as-is

        normalized_texts.extend(batch_processed)

        # Log progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(texts):
            logger.info(f"Normalized {min(i + batch_size, len(texts))}/{len(texts)} texts")

    # Add normalized column to dataframe
    df_pandas[output_column] = normalized_texts

    # Convert back to Polars
    result_df = pl.DataFrame(df_pandas)

    logger.info("Normalization complete")

    return result_df


def load_aggregate_and_split(
    parquet_path: Union[str, Path],
    group_by: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None,
    spacy_model: str = "de_core_news_sm",
    batch_size: int = 1000,
    normalize: bool = False,
    normalize_model: str = "19c",
    normalize_batch_size: int = 32,
    save_path: Optional[Union[str, Path]] = None,
    auto_save: bool = True,
) -> pl.DataFrame:
    """
    Convenience function to load, aggregate, split, and optionally normalize text.

    This combines load_and_aggregate_textblocks(), split_into_sentences(),
    and optionally normalize_text() into a single function call.

    The output will be saved to data/processed/{source_name}/text/sentences.parquet
    or sentences_normalized.parquet if normalization is enabled (unless save_path
    is explicitly provided or auto_save is False).

    All metadata from text_block_id upwards (text_block_id, page_id, date, filename,
    newspaper_id, newspaper_title, year, month, day, and spatial information) is preserved.

    Args:
        parquet_path: Path to the input parquet file with line-level data
        group_by: Columns to group by for aggregation
        sort_by: Columns to sort by within groups
        spacy_model: spaCy model to use for sentence splitting
        batch_size: Batch size for spaCy processing
        normalize: If True, normalize text using Transnormer (default: False)
        normalize_model: Transnormer model to use ("19c" or "18-19c")
        normalize_batch_size: Batch size for normalization
        save_path: If provided, save the result to this parquet file (overrides auto_save)
        auto_save: If True, automatically save to
                   data/processed/{source}/text/sentences[_normalized].parquet

    Returns:
        Polars DataFrame with sentence-level data (and normalized text if requested)

    Example:
        >>> # Process and auto-save with normalized text
        >>> df = load_aggregate_and_split(
        ...     "data/raw/der_tag/text/lines.parquet",
        ...     normalize=True,
        ...     normalize_model="19c"
        ... )
        >>> # Saved to: data/processed/der_tag/text/sentences_normalized.parquet
        >>>
        >>> # Process without normalization
        >>> df = load_aggregate_and_split(
        ...     "data/raw/der_tag/text/lines.parquet"
        ... )
        >>> # Saved to: data/processed/der_tag/text/sentences.parquet
    """
    parquet_path = Path(parquet_path)

    logger.info("Step 1: Loading and aggregating text blocks")
    blocks_df = load_and_aggregate_textblocks(
        parquet_path, group_by=group_by, sort_by=sort_by, auto_save=False
    )

    logger.info("Step 2: Splitting into sentences")
    sentences_df = split_into_sentences(
        blocks_df, text_column="text", model=spacy_model, batch_size=batch_size
    )

    if normalize:
        logger.info("Step 3: Normalizing text")
        sentences_df = normalize_text(
            sentences_df,
            text_column="sentence",
            model=normalize_model,
            batch_size=normalize_batch_size,
        )

    # Determine save path
    if save_path:
        final_save_path = Path(save_path)
    elif auto_save:
        # Extract source name from path (e.g., "der_tag" from "data/raw/der_tag/text/lines.parquet")
        parts = parquet_path.parts
        try:
            # Find "raw" in path and get the next part as source name
            raw_idx = parts.index("raw")
            source_name = parts[raw_idx + 1]
        except (ValueError, IndexError):
            # Fallback: try to infer from path structure
            logger.warning("Could not extract source name from path, using 'unknown'")
            source_name = "unknown"

        # Construct output path
        filename = "sentences_normalized.parquet" if normalize else "sentences.parquet"
        final_save_path = Path("data") / "processed" / source_name / "text" / filename
        logger.info(f"Auto-save enabled: will save to {final_save_path}")
    else:
        final_save_path = None

    # Save if path is determined
    if final_save_path:
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to {final_save_path}")
        sentences_df.write_parquet(final_save_path, compression="zstd")
        logger.info(f"Saved {len(sentences_df)} sentences to {final_save_path}")

    return sentences_df


def remove_stopwords(
    df: pl.DataFrame,
    text_column: str = "sentence",
    output_column: str = "cleaned_text",
    lang: str = "de",
) -> pl.DataFrame:
    """
    Remove stopwords from the specified text column in the DataFrame using spacy.

    Args:
        df: Polars DataFrame containing text
        text_column: Name of the column containing text to process (default: "sentence")
        output_column: Name for the output column (default: "cleaned_text")
        lang: Language code for stopwords (default: "de" for German)

    Returns:
        Polars DataFrame with an additional column containing text without stopwords
    """

    try:
        import spacy
        from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
        from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
    except ImportError:
        raise ImportError(
            "spaCy is required for stopword removal. Install it with: pip install spacy"
        )

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    logger.info(f"Loading spaCy model for stopword removal: {lang}")

    try:
        nlp = spacy.blank(lang)
    except Exception as e:
        raise RuntimeError(f"Failed to load spaCy language '{lang}': {e}")

    # Select appropriate stopwords
    if lang == "de":
        stopwords = DE_STOP_WORDS
    elif lang == "en":
        stopwords = EN_STOP_WORDS
    else:
        raise ValueError(f"Unsupported language for stopwords: {lang}")

    logger.info(f"Removing stopwords from {len(df)} texts")

    # Convert to pandas for easier processing
    df_pandas = df.to_pandas()

    cleaned_texts = []

    for text in df_pandas[text_column].tolist():
        if text and text.strip():
            doc = nlp(text)
            tokens = [token.text for token in doc if token.text.lower() not in stopwords]
            cleaned_text = " ".join(tokens)
            cleaned_texts.append(cleaned_text)
        else:
            cleaned_texts.append(text)  # Keep empty texts as-is

    # Add cleaned column to dataframe
    df_pandas[output_column] = cleaned_texts

    # Convert back to Polars
    result_df = pl.DataFrame(df_pandas)

    logger.info("Stopword removal complete")

    return result_df
