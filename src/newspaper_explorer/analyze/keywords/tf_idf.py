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

from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import ForeignKeys, extract_foreign_keys
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


# Global variable for feature names (shared across workers to avoid pickling)
_FEATURE_NAMES = None


def _init_worker(feature_names):
    """Initialize worker process with shared feature names."""
    global _FEATURE_NAMES
    _FEATURE_NAMES = feature_names


def _extract_keywords_batch(args):
    """
    Worker function to extract keywords from a batch of documents.

    This function is designed to be called by multiprocessing workers.

    Args:
        args: Tuple of (batch_start, batch_end, batch_dense, top_k)

    Returns:
        List of dicts with doc_idx, keywords, scores
    """
    batch_start, batch_end, batch_dense, top_k = args
    feature_names = _FEATURE_NAMES  # Use shared feature names

    if feature_names is None:
        raise RuntimeError("Worker not properly initialized - feature_names is None")

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
            from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

            # Convert SpaCy's set to list
            stopwords = list(DE_STOP_WORDS)
            logger.info(f"Loaded {len(stopwords)} German stopwords from SpaCy")

        if custom_stopwords:
            stopwords.extend(custom_stopwords)  # type: ignore

        # Remove duplicates and return
        return list(set(stopwords))

    def extract_keywords(
        self,
        group_by: Optional[Union[str, List[str]]] = None,
        document_level: str = "page",
        top_k: int = 10,
        min_df: int = 2,
        max_df: float = 0.8,
        ngram_range: tuple = (1, 1),
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
            num_workers: Number of CPU workers for parallel keyword extraction.
                        Default: cpu_count() - 1 (auto-detect)
            limit: Limit number of rows to process (for testing)

        Returns:
            DataFrame with columns:
            - group_id: Document/group identifier
            - source_id, issue_id, page_id, text_block_id: Foreign keys
            - keywords: List of top keywords
            - scores: List of corresponding TF-IDF scores
            - [original grouping columns if group_by used]

        Note:
            No preprocessing is applied. Users should provide either:
            1. Raw text in the text column (will include OCR artifacts)
            2. Preprocessed text (e.g., from `newspaper-explorer data preprocess`)
               Use --text-column to specify the preprocessed column name.

        Example:
            >>> # Extract keywords from raw text
            >>> extractor = TFIDFExtractor("der_tag")
            >>> df = extractor.extract_keywords(document_level="page", top_k=10)
            >>>
            >>> # Extract keywords from preprocessed text
            >>> extractor = TFIDFExtractor(
            ...     "der_tag",
            ...     input_file="data/processed/der_tag/text/preprocessed.parquet",
            ...     text_column="text_processed"
            ... )
            >>> df = extractor.extract_keywords(document_level="page", top_k=10)
        """
        start_time = time.time()

        logger.info(f"Loading data from {self.input_file}")
        print("Loading data...", flush=True)
        df = pl.read_parquet(self.input_file)

        if limit:
            logger.info(f"Limiting to {limit:,} rows for testing")
            df = df.head(limit)

        logger.info(f"Loaded {len(df):,} rows")
        print(f"[OK] Loaded {len(df):,} rows", flush=True)

        # Filter out null/empty texts
        df = df.filter(pl.col(self.text_column).is_not_null())
        df = df.filter(pl.col(self.text_column).str.len_chars() > 0)
        logger.info(f"After filtering: {len(df):,} rows with non-empty text")

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

        # Group documents and aggregate foreign keys
        logger.info(f"Grouping by: {', '.join(group_cols)}")
        print(f"Grouping {len(df):,} rows by {', '.join(group_cols)}...", flush=True)

        # Determine which foreign key columns exist in the data
        fk_cols = []
        for col in ["line_id", "text_block_id", "page_id", "issue_id", "source_id"]:
            if col in df.columns:
                fk_cols.append(col)

        # Build aggregation: text + first non-null foreign key value from each group
        agg_exprs = [pl.col(self.text_column).str.concat(" ").alias("document")]
        for col in fk_cols:
            agg_exprs.append(pl.col(col).drop_nulls().first().alias(col))

        df_grouped = df.group_by(group_cols).agg(agg_exprs)

        documents = df_grouped["document"].to_list()
        num_documents = len(documents)
        logger.info(f"Created {num_documents:,} documents")
        print(f"[OK] Created {num_documents:,} documents", flush=True)

        # Memory warning for large document counts
        if num_documents > 5_000_000:
            logger.warning(
                f"\n{'=' * 80}\n"
                f"MEMORY WARNING: {num_documents:,} documents will create a VERY LARGE TF-IDF matrix\n"
                f"This may cause out-of-memory errors.\n\n"
                f"RECOMMENDATIONS:\n"
                f"  1. Use --document-level page (fewer documents, better context)\n"
                f"  2. Use --document-level date (daily keyword trends)\n"
                f"  3. Use --group-by year (yearly keyword trends)\n"
                f"  4. Test with --limit 500000 first\n"
                f"{'=' * 80}\n"
            )

            time.sleep(3)  # Give user time to read warning

        # Initialize TF-IDF vectorizer
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
        print(f"Computing TF-IDF matrix for {num_documents:,} documents...", flush=True)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(feature_names)}")
        print(
            f"[OK] TF-IDF matrix: {tfidf_matrix.shape[0]:,} documents × {tfidf_matrix.shape[1]:,} terms",
            flush=True,
        )
        logger.info(
            "Note: TF-IDF uses CPU (not GPU) - this is expected and optimal for this algorithm"
        )

        # Extract top keywords for each document (streaming, memory-efficient)
        logger.info(f"Extracting top {top_k} keywords per document...")
        logger.info(f"Processing {tfidf_matrix.shape[0]:,} documents...")

        # Determine number of workers
        if num_workers is None:
            # Reduce workers for large vocabulary to avoid OOM (each worker gets feature_names)
            # With 2.2M vocab, even shared initialization can cause issues with many workers
            num_workers = min(8, max(1, cpu_count() - 1))
        logger.info(f"Using {num_workers} CPU workers for parallel extraction")

        # CRITICAL: For large datasets, we need to avoid loading dense matrices into memory
        # Process in smaller batches and use sparse matrix operations
        # With 2.2M vocabulary, even 5000 docs × 2.2M = 44 billion floats is too much
        batch_size = 1000  # Very small batches for large vocabulary to avoid OOM
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

                # Don't pass feature_names - workers get it via _init_worker
                yield (batch_start, batch_end, batch_dense, top_k)

        # Process with multiprocessing pool, initializing workers with feature_names
        # Use imap_unordered for better memory efficiency (doesn't need to maintain order)
        with Pool(
            processes=num_workers, initializer=_init_worker, initargs=(feature_names,)
        ) as pool:
            for batch_results in tqdm(
                pool.imap_unordered(_extract_keywords_batch, batch_generator(), chunksize=1),
                total=num_batches,
                desc="Extracting keywords",
                unit="batch",
            ):
                results.extend(batch_results)
                # Force garbage collection after each batch to free memory
                import gc

                gc.collect()

        # Create results DataFrame
        logger.info("Creating results DataFrame...")
        results_df = pl.DataFrame(results)

        # Join with original grouping data
        df_grouped = df_grouped.with_row_count("doc_idx")
        results_df = results_df.join(df_grouped, on="doc_idx", how="left")

        # Create meaningful doc_id (composite from grouping columns)
        # Don't just copy a single column - create a unique identifier
        if group_cols:
            if len(group_cols) == 1:
                # Single column: create ID with prefix
                col_name = group_cols[0]
                results_df = results_df.with_columns(
                    (pl.lit(f"{col_name}=") + pl.col(col_name).cast(pl.Utf8)).alias("doc_id")
                )
            else:
                # Multiple columns: create composite ID
                doc_ids = [
                    "_".join(f"{col}={val}" for col, val in zip(group_cols, row))
                    for row in results_df.select(group_cols).iter_rows()
                ]
                results_df = results_df.with_columns(pl.Series("doc_id", doc_ids))

        # Foreign keys: use aggregated values from original data, not extracted from doc_id
        # If foreign keys were aggregated, they're already in df_grouped
        # If not, extract from line_id if available
        logger.info("Adding foreign key columns...")

        if "source_id" in results_df.columns and "issue_id" in results_df.columns:
            # Foreign keys already aggregated from source data
            logger.info("Using aggregated foreign keys from source data")
            # Ensure they exist, fill nulls if needed
            for col in ["source_id", "issue_id", "page_id", "text_block_id"]:
                if col not in results_df.columns:
                    results_df = results_df.with_columns(pl.lit(None).alias(col))
        elif "line_id" in results_df.columns:
            # Extract from line_id
            logger.info("Extracting foreign keys from line_id...")
            line_ids_list = results_df["line_id"].to_list()
            foreign_keys = [
                extract_foreign_keys(line_id)
                if line_id
                else ForeignKeys(None, None, None, None, None)
                for line_id in line_ids_list
            ]

            results_df = results_df.with_columns(
                [
                    pl.Series("source_id", [fk.source_id for fk in foreign_keys]),
                    pl.Series("issue_id", [fk.issue_id for fk in foreign_keys]),
                    pl.Series("page_id", [fk.page_id for fk in foreign_keys]),
                    pl.Series(
                        "text_block_id",
                        [fk.text_block_id or "" for fk in foreign_keys],
                    ),
                ]
            )
        else:
            # No foreign keys available - add null columns
            logger.warning("No foreign key columns (source_id, line_id) found in data")
            for col in ["source_id", "issue_id", "page_id", "text_block_id"]:
                results_df = results_df.with_columns(pl.lit(None).alias(col))

        logger.info(f"Extracted keywords for {len(results_df):,} documents")

        # Log some statistics
        total_keywords = results_df["keywords"].map_elements(len, return_dtype=pl.Int64).sum()
        avg_keywords = total_keywords / len(results_df) if len(results_df) > 0 else 0
        logger.info(f"Average keywords per document: {avg_keywords:.1f}")

        # Remove internal processing columns from output
        # Keep: doc_id, foreign keys, grouping columns, keywords, scores
        columns_to_drop = ["doc_idx", "document"]
        results_df = results_df.drop([col for col in columns_to_drop if col in results_df.columns])

        # Store timing info for save_results
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = df
        self._last_params = {
            "document_level": document_level,
            "group_by": group_by,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "top_k": top_k,
            "input": {
                "parquet": str(self.input_file),
                "metadata": str(self.input_file).replace(".parquet", ".json"),
            },
        }

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "tfidf_keywords",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save extracted keywords to parquet file with metadata.

        Args:
            results_df: DataFrame with keywords (from extract_keywords)
            output_name: Base name for output file (without extension, default: "tfidf_keywords")
            top_k: Number of keywords (for metadata, optional)

        Returns:
            Path to saved file
        """
        logger.info("Creating metadata...")

        # Calculate statistics
        total_keywords = sum(len(kw_list) for kw_list in results_df["keywords"].to_list())
        avg_keywords = total_keywords / len(results_df) if len(results_df) > 0 else 0
        avg_score = (
            sum(
                sum(scores) / len(scores) if scores else 0
                for scores in results_df["scores"].to_list()
            )
            / len(results_df)
            if len(results_df) > 0
            else 0
        )

        # Create output statistics
        output_stats = extract_output_stats(results_df)
        output_stats.update(
            {
                "total_documents": len(results_df),
                "total_keywords": total_keywords,
                "avg_keywords_per_doc": round(avg_keywords, 2),
                "avg_score": round(avg_score, 4),
            }
        )

        # Get duration if available from last extraction
        duration = getattr(self, "_last_extraction_time", None)

        # Get input stats if available
        input_stats = {}
        if hasattr(self, "_last_input_df"):
            input_stats = extract_input_stats(self._last_input_df)

        # Get extraction parameters if available
        params = getattr(self, "_last_params", {})
        params.update(
            {
                "algorithm": "TF-IDF (Term Frequency-Inverse Document Frequency)",
                "top_k": top_k if top_k is not None else params.get("top_k"),
                "text_column": self.text_column,
            }
        )

        # Create metadata with properly formatted timestamps
        completed_at = datetime.now().isoformat()

        metadata = AnalysisMetadata(
            analysis_id=None,  # Will be auto-generated
            analysis_type="keywords",
            method_type="tfidf",
            model_name="sklearn_tfidfvectorizer",
            model_version="1.3.0",  # sklearn version
            source=self.source_name,
            parameters=params,
            input_data=input_stats,
            output_data=output_stats,
            granularity="textblock",  # TF-IDF keyword extraction runs on textblock level
            status="completed",
            duration_seconds=duration,
            completed_at=completed_at,
            error_message=None,
        )

        # Save using unified helper
        results_base = self.config.results_dir / self.source_name
        paths = save_analysis_results(
            results_df=results_df,
            metadata=metadata,
            results_base_dir=results_base,
            results_filename="keywords.parquet",
        )

        logger.info(f"Saved {len(results_df):,} keyword results to {paths['results_path']}")
        logger.info(f"Metadata saved to: {paths['metadata_path']}")
        logger.info(f"Analysis ID: {metadata.analysis_id}")

        return paths["results_path"]


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
        keywords_tuple, scores_tuple = zip(*filtered)
        keywords = list(keywords_tuple)
        keyword_scores = list(scores_tuple)

    return {
        "keywords": list(keywords),
        "scores": list(keyword_scores),
    }
