"""
Text processing utilities for newspaper data.
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    max_length: int = 512,
    split_symbols: Optional[list[str]] = None,
    split_margin: int = 50,
) -> list[str]:
    """
    Split text into chunks at safe boundaries (spaces, punctuation).

    Prevents splitting mid-word by looking for split symbols within a margin
    before max_length. This is useful for processing long texts with models
    that have maximum input length constraints.

    Args:
        text: Text to chunk
        max_length: Maximum chunk length in characters (default: 512)
        split_symbols: Symbols to split at (default: [" ", ",", ".", ";", ":", "!", "?"])
        split_margin: Look for split points within this margin before max_length

    Returns:
        List of text chunks. If text is shorter than max_length, returns single-item list.

    Example:
        >>> text = "Very long text that needs to be split..."
        >>> chunks = chunk_text(text, max_length=512, split_margin=50)
        >>> print(f"Split into {len(chunks)} chunks")
    """
    if split_symbols is None:
        split_symbols = [" ", ",", ".", ";", ":", "!", "?"]

    text_length = len(text)
    if text_length <= max_length:
        return [text]

    chunks = []
    prev_index = 0
    start = max_length - split_margin
    end = max_length

    while start < text_length:
        sym_index = -1

        # Find last occurrence of any split symbol in range
        for sym in split_symbols:
            idx = text.rfind(sym, start, min(end, text_length))
            if idx != -1 and (sym_index == -1 or idx > sym_index):
                sym_index = idx

        # No split symbol found, force split at max_length
        if sym_index == -1:
            sym_index = min(end, text_length) - 1

        # Add chunk
        chunks.append(text[prev_index : sym_index + 1])

        # Move to next chunk
        prev_index = sym_index + 1
        start = prev_index + max_length - split_margin
        end = prev_index + max_length

    # Add remaining text
    if prev_index < text_length:
        chunks.append(text[prev_index:])

    return chunks


def split_text_into_sentences(
    text: str,
    model: str = "de_core_news_sm",
) -> list[str]:
    """
    Split a single text into sentences using spaCy for German language processing.

    This is a lightweight function for splitting individual texts. For batch processing
    of DataFrames, use split_into_sentences_df() instead.

    Args:
        text: Text to split into sentences
        model: spaCy model to use (default: "de_core_news_sm")
               Install it with: python -m spacy download de_core_news_sm

    Returns:
        List of sentences (strings)

    Example:
        >>> text = "Das ist der erste Satz. Hier ist der zweite. Und der dritte!"
        >>> sentences = split_text_into_sentences(text)
        >>> print(f"Split into {len(sentences)} sentences")
        >>> # Split into 3 sentences

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

    try:
        nlp = spacy.load(model)
    except OSError:
        raise OSError(
            f"spaCy model '{model}' not found. Install it with: "
            f"python -m spacy download {model}"
        )

    # Process text
    doc = nlp(text)

    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return sentences


def split_df_into_sentences(
    df: pl.DataFrame,
    text_column: str = "text",
    model: str = "de_core_news_sm",
    batch_size: int = 1000,
) -> pl.DataFrame:
    """
    Split text in a DataFrame into sentences using spaCy for German language processing.

    This function processes DataFrames in batches for efficiency and expands each row into
    multiple rows (one per sentence). For splitting individual texts, use split_text_into_sentences().

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
        >>> sentences_df = split_df_into_sentences(blocks_df, text_column="text")
        >>> print(f"Expanded {len(blocks_df)} blocks into {len(sentences_df)} sentences")

    Raises:
        ImportError: If spaCy is not installed
        OSError: If the specified spaCy model is not installed

    See Also:
        split_text_into_sentences: For splitting individual texts (not DataFrames)
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


def get_longest_lines(
    df: pl.DataFrame,
    text_column: str = "text",
    top_n: int = 10,
) -> pl.DataFrame:
    """
    Retrieve the longest lines from the DataFrame.

    Args:
        df: Polars DataFrame containing text lines
        text_column: Name of the column containing text (default: "text")
        top_n: Number of longest lines to retrieve (default: 10)
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    # Get lengths of all lines
    length_series = df[text_column].str.len_chars().fill_null(0)

    # Get indices of longest lines
    longest_indices = length_series.arg_sort(descending=True)[:top_n]

    # Return longest lines
    return df[longest_indices]


def analyze_character_lengths(
    df: pl.DataFrame,
    text_column: str = "text",
    sample_size: int = 50000,
) -> dict:
    """
    Analyze character lengths in text data.

    This function samples texts from the DataFrame and computes comprehensive
    statistics about character lengths. Useful for:
    - Understanding text length distribution
    - Comparing with token lengths
    - Data quality checks

    Args:
        df: Polars DataFrame containing text to analyze
        text_column: Name of the column containing text (default: "text")
        sample_size: Number of rows to sample for analysis (default: 50000)

    Returns:
        Dictionary containing:
        - total_rows: Total number of rows in DataFrame
        - sample_size: Number of rows analyzed
        - min_chars: Minimum character count
        - max_chars: Maximum character count
        - mean_chars: Average character count
        - median_chars: Median character count
        - p90_chars: 90th percentile character count
        - p95_chars: 95th percentile character count
        - p99_chars: 99th percentile character count
        - distribution: Dict with counts for different character ranges
        - longest_examples: List of (char_count, text) tuples for longest texts

    Example:
        >>> from newspaper_explorer.data.loading.loader import DataLoader
        >>> df = DataLoader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
        >>> stats = analyze_character_lengths(df, sample_size=50000)
        >>> print(f"Average characters: {stats['mean_chars']:.1f}")
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    total_rows = len(df)
    logger.info(f"Analyzing character lengths for {total_rows:,} rows")

    # Sample evenly distributed rows if needed
    if total_rows > sample_size:
        logger.info(f"Sampling {sample_size:,} rows evenly distributed across dataset")
        step = max(1, total_rows // sample_size)
        df_sample = (
            df.lazy()
            .with_row_index()
            .filter(pl.col("index") % step == 0)
            .select(pl.col(text_column))
            .limit(sample_size)
            .collect()
        )
    else:
        logger.info(f"Using all {total_rows:,} rows")
        df_sample = df.select(text_column)

    texts = df_sample[text_column].to_list()
    actual_sample_size = len(texts)

    # Get character lengths
    logger.info("Computing character lengths...")
    lengths = [len(text) for text in texts]

    # Compute statistics
    sorted_lengths = sorted(lengths)
    min_chars = min(lengths)
    max_chars = max(lengths)
    mean_chars = sum(lengths) / len(lengths)
    median_chars = sorted_lengths[len(sorted_lengths) // 2]

    # Percentiles
    p90 = sorted_lengths[int(0.90 * len(sorted_lengths))]
    p95 = sorted_lengths[int(0.95 * len(sorted_lengths))]
    p99 = sorted_lengths[int(0.99 * len(sorted_lengths))]

    # Distribution
    under_50 = sum(1 for l in lengths if l <= 50)
    under_100 = sum(1 for l in lengths if l <= 100)
    under_200 = sum(1 for l in lengths if l <= 200)
    under_500 = sum(1 for l in lengths if l <= 500)
    under_1000 = sum(1 for l in lengths if l <= 1000)

    distribution = {
        "under_50": under_50,
        "under_100": under_100,
        "under_200": under_200,
        "under_500": under_500,
        "under_1000": under_1000,
    }

    # Get longest examples
    longest_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:5]
    longest_examples = [(lengths[idx], texts[idx]) for idx in longest_indices]

    logger.info(f"Character analysis complete for {actual_sample_size:,} texts")

    return {
        "total_rows": total_rows,
        "sample_size": actual_sample_size,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "mean_chars": mean_chars,
        "median_chars": median_chars,
        "p90_chars": p90,
        "p95_chars": p95,
        "p99_chars": p99,
        "distribution": distribution,
        "longest_examples": longest_examples,
    }


def analyze_token_lengths(
    df: pl.DataFrame,
    text_column: str = "text",
    tokenizer_name: str = "deepset/gbert-large",
    sample_size: int = 50000,
    max_length: int = 512,
) -> dict:
    """
    Analyze token lengths in text data using BERT tokenizer.

    This function samples texts from the DataFrame and computes comprehensive
    statistics about token lengths. Useful for:
    - Understanding padding requirements for BERT models
    - Estimating speedup potential from dynamic padding
    - Identifying truncation issues

    Args:
        df: Polars DataFrame containing text to analyze
        text_column: Name of the column containing text (default: "text")
        tokenizer_name: HuggingFace tokenizer to use (default: "deepset/gbert-large")
        sample_size: Number of rows to sample for analysis (default: 50000)
        max_length: Maximum sequence length for tokenization (default: 512)

    Returns:
        Dictionary containing:
        - total_rows: Total number of rows in DataFrame
        - sample_size: Number of rows analyzed
        - min_tokens: Minimum token count
        - max_tokens: Maximum token count
        - mean_tokens: Average token count
        - median_tokens: Median token count
        - p90_tokens: 90th percentile token count
        - p95_tokens: 95th percentile token count
        - p99_tokens: 99th percentile token count
        - distribution: Dict with counts for different token ranges
        - truncated_count: Number of texts truncated to max_length
        - truncated_percent: Percentage of texts truncated
        - wasted_padding_percent: Percentage of compute wasted with static padding
        - expected_speedup: Expected speedup factor with dynamic padding
        - longest_examples: List of (token_count, text) tuples for longest texts

    Example:
        >>> from newspaper_explorer.data.loading.loader import DataLoader
        >>> df = DataLoader.load_parquet("data/raw/der_tag/text/der_tag_lines.parquet")
        >>> stats = analyze_token_lengths(df, sample_size=50000)
        >>> print(f"Average tokens: {stats['mean_tokens']:.1f}")
        >>> print(f"Expected speedup: {stats['expected_speedup']:.1f}x")

    Raises:
        ImportError: If transformers is not installed
        ValueError: If text_column not found in DataFrame
    """
    try:
        from transformers.models.bert import BertTokenizerFast
    except ImportError:
        raise ImportError(
            "transformers is required for token analysis. "
            "Install it with: pip install transformers"
        )

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    total_rows = len(df)
    logger.info(f"Analyzing token lengths for {total_rows:,} rows")

    # Sample evenly distributed rows if needed
    if total_rows > sample_size:
        logger.info(f"Sampling {sample_size:,} rows evenly distributed across dataset")
        step = max(1, total_rows // sample_size)
        df_sample = (
            df.lazy()
            .with_row_index()
            .filter(pl.col("index") % step == 0)
            .select(pl.col(text_column))
            .limit(sample_size)
            .collect()
        )
    else:
        logger.info(f"Using all {total_rows:,} rows")
        df_sample = df.select(text_column)

    texts = df_sample[text_column].to_list()
    actual_sample_size = len(texts)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    # Tokenize and get lengths
    logger.info("Tokenizing texts (this may take a minute)...")
    lengths = [
        len(tokenizer.encode(text, truncation=True, max_length=max_length)) for text in texts
    ]

    # Compute statistics
    sorted_lengths = sorted(lengths)
    min_tokens = min(lengths)
    max_tokens = max(lengths)
    mean_tokens = sum(lengths) / len(lengths)
    median_tokens = sorted_lengths[len(sorted_lengths) // 2]

    # Percentiles
    p90 = sorted_lengths[int(0.90 * len(sorted_lengths))]
    p95 = sorted_lengths[int(0.95 * len(sorted_lengths))]
    p99 = sorted_lengths[int(0.99 * len(sorted_lengths))]

    # Distribution
    under_50 = sum(1 for l in lengths if l <= 50)
    under_100 = sum(1 for l in lengths if l <= 100)
    under_200 = sum(1 for l in lengths if l <= 200)
    under_300 = sum(1 for l in lengths if l <= 300)
    at_max = sum(1 for l in lengths if l == max_length)

    distribution = {
        "under_50": under_50,
        "under_100": under_100,
        "under_200": under_200,
        "under_300": under_300,
        "at_max_length": at_max,
    }

    # Truncation analysis
    truncated_count = at_max
    truncated_percent = 100 * truncated_count / len(lengths)

    # Speedup estimate
    wasted_padding_percent = 100 * (max_length - mean_tokens) / max_length
    expected_speedup = max_length / mean_tokens

    # Get longest examples
    longest_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:5]
    longest_examples = [(lengths[idx], texts[idx]) for idx in longest_indices]

    logger.info(f"Token analysis complete for {actual_sample_size:,} texts")

    return {
        "total_rows": total_rows,
        "sample_size": actual_sample_size,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "mean_tokens": mean_tokens,
        "median_tokens": median_tokens,
        "p90_tokens": p90,
        "p95_tokens": p95,
        "p99_tokens": p99,
        "distribution": distribution,
        "truncated_count": truncated_count,
        "truncated_percent": truncated_percent,
        "wasted_padding_percent": wasted_padding_percent,
        "expected_speedup": expected_speedup,
        "longest_examples": longest_examples,
    }


def get_longest_lines_by_tokens(
    df: pl.DataFrame,
    text_column: str = "text",
    tokenizer_name: str = "deepset/gbert-large",
    top_n: int = 10,
) -> pl.DataFrame:
    """
    Retrieve the longest lines based on token count (not character count).

    Uses a tokenizer to determine actual token lengths, which is more accurate
    than character count for understanding model input size. Useful for:
    - Finding texts that will be truncated
    - Understanding the distribution of long texts
    - Testing with edge cases

    Args:
        df: Polars DataFrame containing text lines
        text_column: Name of the column containing text (default: "text")
        tokenizer_name: HuggingFace tokenizer to use (default: "deepset/gbert-large")
        top_n: Number of longest lines to retrieve (default: 10)

    Returns:
        Polars DataFrame with longest lines, including a "token_count" column

    Example:
        >>> longest = get_longest_lines_by_tokens(df, top_n=10)
        >>> print(longest[["token_count", "text"]])

        >>> # Use with different tokenizer
        >>> longest = get_longest_lines_by_tokens(df, tokenizer_name="bert-base-german-cased")

        >>> # Get more results
        >>> longest = get_longest_lines_by_tokens(df, top_n=50)
    """
    try:
        from transformers.models.bert import BertTokenizerFast
    except ImportError:
        raise ImportError("transformers is required. Install it with: pip install transformers")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    # Get top 100 longest by character count first (optimization)
    longest_by_chars = get_longest_lines(df, text_column=text_column, top_n=100)

    # Tokenize and get actual token counts
    logger.info("Tokenizing texts to get token counts...")
    texts = longest_by_chars[text_column].to_list()
    token_counts = [len(tokenizer.encode(text, truncation=True, max_length=512)) for text in texts]

    # Add token counts to DataFrame
    result_df = longest_by_chars.with_columns(pl.Series("token_count", token_counts))

    # Sort by token count and take top N
    result_df = result_df.sort("token_count", descending=True).head(top_n)

    return result_df
