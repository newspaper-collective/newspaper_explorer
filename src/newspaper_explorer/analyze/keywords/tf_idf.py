"""
TF-IDF keyword extraction for newspaper texts.

Implements Term Frequency-Inverse Document Frequency (TF-IDF) to extract
the most relevant keywords from newspaper articles. TF-IDF identifies words
that are important to a document by considering both their frequency within
that document and their rarity across the entire corpus.

Example:
    >>> from newspaper_explorer.analyze.keywords.tf_idf import TFIDFExtractor
    >>> extractor = TFIDFExtractor(source_name="der_tag")
    >>> keywords_df = extractor.extract_keywords(
    ...     group_by="date",
    ...     top_k=10,
    ...     min_df=2,
    ...     max_df=0.8
    ... )
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from multiprocessing import Pool, cpu_count

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


def _extract_keywords_batch(args):
    """
    Worker function to extract keywords from a batch of documents.

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple of (batch_start, batch_end, batch_dense, feature_names, top_k)

    Returns:
        List of dicts with doc_idx, keywords, scores
    """
    batch_start, batch_end, batch_dense, feature_names, top_k = args

    results = []
    num_docs = batch_dense.shape[0]

    # Vectorized extraction of top-k indices for entire batch
    # argpartition is faster than argsort for finding top-k
    top_indices_batch = np.argpartition(batch_dense, -top_k, axis=1)[:, -top_k:]

    # For each document in batch, sort the top-k to get correct order
    for i in range(num_docs):
        doc_idx = batch_start + i
        top_idx = top_indices_batch[i]
        scores_top = batch_dense[i, top_idx]

        # Sort by score (descending)
        sorted_order = scores_top.argsort()[::-1]
        top_idx_sorted = top_idx[sorted_order]
        scores_sorted = scores_top[sorted_order]

        # Extract keywords and scores
        keywords = [feature_names[j] for j in top_idx_sorted]
        scores = [round(float(s), 4) for s in scores_sorted]

        # Filter out zero scores
        filtered = [(k, s) for k, s in zip(keywords, scores) if s > 0]
        if filtered:
            keywords, scores = zip(*filtered)
        else:
            keywords, scores = [], []

        results.append(
            {
                "doc_idx": doc_idx,
                "keywords": list(keywords),
                "scores": list(scores),
            }
        )

    return results


class TFIDFExtractor:
    """
    Extract keywords from newspaper texts using TF-IDF.

    TF-IDF (Term Frequency-Inverse Document Frequency) identifies words that
    are important to documents by:
    1. Term Frequency (TF): How often a word appears in a document
    2. Inverse Document Frequency (IDF): How rare/unique a word is across all documents

    Words with high TF-IDF scores are both frequent in a document AND rare
    across the corpus, making them good keywords.
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
    ):
        """
        Initialize TF-IDF extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            use_stopwords: Whether to remove common stopwords
            custom_stopwords: Additional stopwords to exclude
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to lines.parquet for page-level analysis
            # Falls back to textblocks.parquet if lines don't exist
            source_dir = self.config.data_dir / "raw" / source_name / "text"
            lines_file = source_dir / f"{source_name}_lines.parquet"
            textblocks_file = source_dir / f"{source_name}_textblocks.parquet"

            if lines_file.exists():
                self.input_file = lines_file
                logger.info("Using lines.parquet (line-level data)")
            elif textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info("Using textblocks.parquet (aggregated text blocks)")
            else:
                # Set to lines even if doesn't exist - will error later with helpful message
                self.input_file = lines_file

        # Setup stopwords
        self.stopwords = self._get_stopwords(use_stopwords, custom_stopwords)

        logger.info(f"Initialized TF-IDF extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Using {len(self.stopwords)} stopwords")

    def _get_stopwords(
        self, use_stopwords: bool, custom_stopwords: Optional[List[str]]
    ) -> List[str]:
        """
        Get stopwords list using SpaCy's German stopwords.

        Uses the same stopword list as the preprocessing pipeline for consistency.
        """
        stopwords = []

        if use_stopwords:
            try:
                from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

                # Convert SpaCy's set to list
                stopwords = list(DE_STOP_WORDS)
                logger.info(f"Loaded {len(stopwords)} German stopwords from SpaCy")

            except ImportError:
                logger.warning(
                    "SpaCy not installed, using basic German stopwords. "
                    "Install with: pip install -e '.[nlp]' for better stopword list"
                )
                # Fallback to basic stopwords if SpaCy not available
                stopwords = [
                    "der",
                    "die",
                    "das",
                    "und",
                    "in",
                    "zu",
                    "den",
                    "ist",
                    "von",
                    "mit",
                    "auf",
                    "für",
                    "als",
                    "an",
                    "im",
                    "dem",
                    "ein",
                    "eine",
                    "nicht",
                    "auch",
                    "sich",
                    "wird",
                    "oder",
                    "aus",
                    "werden",
                    "bei",
                    "nach",
                    "aber",
                    "noch",
                    "wie",
                    "sind",
                    "hat",
                    "zum",
                    "zur",
                    "war",
                    "durch",
                    "nur",
                    "über",
                    "vor",
                    "es",
                    "so",
                    "am",
                    "bis",
                    "dass",
                    "daß",
                    "wenn",
                    "sein",
                    "kann",
                    "mehr",
                    "diese",
                    "dieser",
                    "einem",
                    "einen",
                    "einer",
                    "eines",
                    "haben",
                    "gegen",
                    "doch",
                    "alle",
                    "schon",
                    "was",
                    "wir",
                    "ihm",
                    "ihr",
                    "sie",
                    "sie",
                ]

        if custom_stopwords:
            stopwords.extend(custom_stopwords)  # type: ignore

        # Remove duplicates and return
        return list(set(stopwords))

    def _apply_preprocessing(
        self,
        df: pl.DataFrame,
        preprocessing_steps: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Apply preprocessing pipeline to text.

        Uses the existing preprocessing infrastructure from data.preprocessing.

        Args:
            df: DataFrame with text column
            preprocessing_steps: List of preprocessing steps to apply.
                               If None, applies default cleaning:
                               ['normalize-whitespace', 'lowercase', 'remove-punctuation']

        Returns:
            DataFrame with 'clean_text' column
        """
        if preprocessing_steps is None:
            # Default cleaning for TF-IDF
            preprocessing_steps = ["normalize-whitespace", "lowercase", "remove-punctuation"]

        if not preprocessing_steps:
            # No preprocessing - just copy text
            return df.with_columns(pl.col(self.text_column).alias("clean_text"))

        logger.info(f"Applying preprocessing steps: {', '.join(preprocessing_steps)}")

        from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

        preprocessor = TextPreprocessor(text_column=self.text_column)
        df = preprocessor.pipeline(
            df,
            steps=preprocessing_steps,
            output_column="clean_text",
        )

        return df

    def extract_keywords(
        self,
        group_by: Optional[Union[str, List[str]]] = None,
        document_level: str = "page",
        top_k: int = 10,
        min_df: int = 2,
        max_df: float = 0.8,
        ngram_range: tuple = (1, 1),
        preprocessing_steps: Optional[List[str]] = None,
        num_workers: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract top keywords using TF-IDF.

        Args:
            group_by: Column(s) to group documents by. If specified, overrides document_level.
                     Examples: "date", "year", ["year", "month"], "text_block_id"
            document_level: Defines what constitutes a "document" (ignored if group_by is set):
                          - "page": One document per newspaper page (default, recommended)
                          - "textblock": One document per text block
                          - "file": One document per XML file (entire issue)
                          - "date": One document per publication date
            top_k: Number of top keywords to extract per document/group
            min_df: Minimum document frequency (ignore terms appearing in fewer documents)
            max_df: Maximum document frequency (ignore terms appearing in more than this
                    fraction of documents, e.g., 0.8 = 80%)
            ngram_range: Range of n-grams to extract (1,1)=unigrams, (1,2)=unigrams+bigrams
            preprocessing_steps: List of preprocessing steps to apply to text.
                               If None, uses default: ['normalize-whitespace', 'lowercase', 'remove-punctuation']
                               Pass empty list [] to skip preprocessing.
                               Available: 'normalize', 'lowercase', 'normalize-whitespace',
                                        'remove-punctuation', 'remove-stopwords', etc.
                               See preprocessing.pipeline for full list.
            num_workers: Number of CPU workers for parallel keyword extraction.
                        Default: cpu_count() - 1 (auto-detect)
            limit: Limit number of rows to process (for testing)

        Returns:
            DataFrame with columns:
            - group_id: Document/group identifier
            - keywords: List of top keywords
            - scores: List of corresponding TF-IDF scores
            - [original grouping columns if group_by used]

        Example:
            >>> # Extract keywords per page (recommended)
            >>> extractor = TFIDFExtractor("der_tag")
            >>> df = extractor.extract_keywords(document_level="page", top_k=10)
            >>>
            >>> # Extract keywords per date
            >>> df = extractor.extract_keywords(document_level="date", top_k=15)
            >>>
            >>> # Custom grouping by year and month
            >>> df = extractor.extract_keywords(group_by=["year", "month"], top_k=20)
        """
        logger.info(f"Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)

        if limit:
            logger.info(f"Limiting to {limit:,} rows for testing")
            df = df.head(limit)

        logger.info(f"Loaded {len(df):,} rows")

        # Filter out null/empty texts
        df = df.filter(pl.col(self.text_column).is_not_null())
        df = df.filter(pl.col(self.text_column).str.len_chars() > 0)
        logger.info(f"After filtering: {len(df):,} rows with non-empty text")

        # Apply preprocessing
        df = self._apply_preprocessing(df, preprocessing_steps)

        # Determine document grouping
        if group_by:
            # Use explicit grouping
            group_cols = [group_by] if isinstance(group_by, str) else group_by
            logger.info(f"Using explicit grouping: {group_cols}")
        else:
            # Use document_level to determine grouping
            if document_level == "page":
                # Group by page_id (includes filename + page number)
                group_cols = ["filename", "page_number"]
                logger.info(f"Document level: PAGE (one document per newspaper page)")
            elif document_level == "textblock":
                # Each text block is a document
                group_cols = ["text_block_id"]
                logger.info(f"Document level: TEXTBLOCK (one document per text block)")
            elif document_level == "file":
                # Entire file (issue) is one document
                group_cols = ["filename"]
                logger.info(f"Document level: FILE (one document per XML file/issue)")
            elif document_level == "date":
                # All content from same date is one document
                group_cols = ["year", "month", "day"]
                logger.info(f"Document level: DATE (one document per publication date)")
            else:
                raise ValueError(
                    f"Invalid document_level: {document_level}. "
                    f"Choose from: 'page', 'textblock', 'file', 'date'"
                )

        # Check if grouping columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Grouping columns not found in data: {missing_cols}. "
                f"Available columns: {df.columns}"
            )

        # Group documents
        logger.info(f"Grouping by: {', '.join(group_cols)}")

        df_grouped = df.group_by(group_cols).agg(
            [
                pl.col("clean_text").str.concat(" ").alias("document"),
            ]
        )

        documents = df_grouped["document"].to_list()
        num_documents = len(documents)
        logger.info(f"Created {num_documents:,} documents")

        # Memory warning for large document counts
        if num_documents > 5_000_000:
            logger.warning(
                f"\n{'='*80}\n"
                f"⚠️  MEMORY WARNING: {num_documents:,} documents will create a VERY LARGE TF-IDF matrix\n"
                f"This may cause out-of-memory errors.\n\n"
                f"RECOMMENDATIONS:\n"
                f"  1. Use --document-level page (fewer documents, better context)\n"
                f"  2. Use --document-level date (daily keyword trends)\n"
                f"  3. Use --group-by year (yearly keyword trends)\n"
                f"  4. Test with --limit 500000 first\n"
                f"{'='*80}\n"
            )
            import time

            time.sleep(3)  # Give user time to read warning        # Initialize TF-IDF vectorizer
        logger.info("Initializing TF-IDF vectorizer...")
        logger.info(f"  min_df={min_df}, max_df={max_df}, ngram_range={ngram_range}")
        logger.info(f"  Using {len(self.stopwords)} stopwords")

        vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            smooth_idf=True,
            use_idf=True,
        )

        # Fit and transform
        # Note: TF-IDF computation is CPU-only (scikit-learn sparse matrix operations)
        # This is very fast for this algorithm - GPU wouldn't provide benefit
        logger.info("Computing TF-IDF scores (CPU-based sparse matrix operations)...")
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(feature_names)}")
        logger.info(
            "Note: TF-IDF uses CPU (not GPU) - this is expected and optimal for this algorithm"
        )

        # Extract top keywords for each document (streaming, memory-efficient)
        logger.info(f"Extracting top {top_k} keywords per document...")
        logger.info(f"Processing {tfidf_matrix.shape[0]:,} documents...")

        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        logger.info(f"Using {num_workers} CPU workers for parallel extraction")

        # CRITICAL: For large datasets, we need to avoid loading dense matrices into memory
        # Process in smaller batches and use sparse matrix operations
        batch_size = 5000  # Smaller batches to avoid OOM
        num_docs = tfidf_matrix.shape[0]
        num_batches = (num_docs + batch_size - 1) // batch_size

        logger.info(f"Processing {num_batches:,} batches of ~{batch_size:,} documents each")
        logger.info(f"Memory-efficient streaming mode (sparse matrices)")

        # Process batches in parallel with streaming (no pre-loading all batches)
        results = []

        def batch_generator():
            """Generator to create batches on-demand (memory efficient)"""
            for batch_start in range(0, num_docs, batch_size):
                batch_end = min(batch_start + batch_size, num_docs)

                # Get batch of documents and convert to dense only when needed
                batch_matrix = tfidf_matrix[batch_start:batch_end]  # type: ignore
                batch_dense = batch_matrix.toarray()

                yield (batch_start, batch_end, batch_dense, feature_names, top_k)

        # Process with multiprocessing pool, using imap for streaming
        with Pool(processes=num_workers) as pool:
            for batch_results in tqdm(
                pool.imap(_extract_keywords_batch, batch_generator(), chunksize=1),
                total=num_batches,
                desc="Extracting keywords",
                unit="batch",
            ):
                results.extend(batch_results)

        # Create results DataFrame
        logger.info("Creating results DataFrame...")
        results_df = pl.DataFrame(results)

        # Join with original grouping data
        df_grouped = df_grouped.with_row_count("doc_idx")
        results_df = results_df.join(df_grouped, on="doc_idx", how="left")

        # Select output columns: group columns + keywords + scores
        output_cols = group_cols + ["keywords", "scores"]
        results_df = results_df.select(output_cols)

        logger.info(f"Extracted keywords for {len(results_df):,} documents")

        # Log some statistics
        total_keywords = results_df["keywords"].map_elements(len, return_dtype=pl.Int64).sum()
        avg_keywords = total_keywords / len(results_df) if len(results_df) > 0 else 0
        logger.info(f"Average keywords per document: {avg_keywords:.1f}")

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "tfidf_keywords",
    ) -> Path:
        """
        Save extracted keywords to parquet file.

        Args:
            results_df: DataFrame with keywords (from extract_keywords)
            output_name: Base name for output file (without extension)

        Returns:
            Path to saved file
        """
        output_dir = self.config.results_dir / self.source_name / "keywords"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{output_name}.parquet"

        logger.info(f"Saving keywords to {output_file}")
        results_df.write_parquet(output_file)
        logger.info(f"Saved {len(results_df):,} keyword results")

        return output_file


def extract_keywords_from_document(
    document: str,
    corpus: List[str],
    top_k: int = 10,
    stopwords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract top keywords from a single document given a corpus.

    Utility function for extracting keywords from one document when you
    already have a corpus for IDF calculation.

    Args:
        document: Text document to extract keywords from
        corpus: List of all documents in corpus (for IDF calculation)
        top_k: Number of top keywords to return
        stopwords: List of stopwords to exclude

    Returns:
        Dict with 'keywords' (list) and 'scores' (list)

    Example:
        >>> corpus = ["article 1 text", "article 2 text", ...]
        >>> doc = "article 3 text with important keywords"
        >>> result = extract_keywords_from_document(doc, corpus, top_k=5)
        >>> print(result['keywords'])
        ['important', 'keywords', ...]
    """
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        smooth_idf=True,
        use_idf=True,
    )

    # Fit on entire corpus
    vectorizer.fit(corpus)

    # Transform single document
    tfidf_vector = vectorizer.transform([document])
    feature_names = vectorizer.get_feature_names_out()

    # Get scores
    scores = tfidf_vector.toarray().flatten()  # type: ignore
    top_indices = scores.argsort()[-top_k:][::-1]

    keywords = [feature_names[i] for i in top_indices]
    keyword_scores = [round(float(scores[i]), 4) for i in top_indices]

    # Filter zero scores
    filtered = [(k, s) for k, s in zip(keywords, keyword_scores) if s > 0]
    if filtered:
        keywords, keyword_scores = zip(*filtered)

    return {
        "keywords": list(keywords),
        "scores": list(keyword_scores),
    }
