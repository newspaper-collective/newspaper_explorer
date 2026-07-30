"""
YAKE keyword extraction for newspaper texts.

Implements YAKE (Yet Another Keyword Extractor) to extract keywords and keyphrases
from newspaper articles. YAKE uses statistical features (term frequency, casing,
position, etc.) to identify important keywords without requiring training data
or external knowledge.

YAKE is particularly good for:
- Unsupervised keyword extraction
- Multi-lingual support (works with German)
- Extracting both single words and multi-word phrases
- Handling various text lengths

Example:
    >>> from newspaper_explorer.analyze.keywords.yake import YAKEExtractor
    >>> extractor = YAKEExtractor(source_name="der_tag")
    >>> keywords = extractor.extract_keywords(top_k=10)
"""

from datetime import datetime
import logging
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
from tqdm import tqdm
import yake

from newspaper_explorer.cli.utils.options import resolve_text_column
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


from typing import Sequence, Union


def _filter_overlapping_keywords(
    keywords: List[str], scores: List[float]
) -> Tuple[List[str], List[float]]:
    """
    Filter out keywords that are substrings of other keywords.

    When multiple n-grams overlap (e.g., "Präsident Theodore" and "Theodore Roosevelt"),
    keeps only the longest/most specific version with the best score.

    Args:
        keywords: List of keyword strings
        scores: Corresponding YAKE scores (lower is better)

    Returns:
        Tuple of (filtered_keywords, filtered_scores)
    """
    if not keywords:
        return keywords, scores

    # Create list of (keyword, score, index) tuples
    kw_data = [(kw, score, i) for i, (kw, score) in enumerate(zip(keywords, scores))]

    # Sort by length (descending) then by score (ascending, since lower is better)
    kw_data.sort(key=lambda x: (-len(x[0]), x[1]))

    filtered = []
    filtered_indices = set()

    for kw, score, idx in kw_data:
        # Check if this keyword is a substring of any already-selected keyword
        is_substring = False
        for selected_kw, _, _ in filtered:
            if kw in selected_kw and kw != selected_kw:
                is_substring = True
                break

        if not is_substring:
            # Check if any previously seen keyword is a substring of this one
            # Remove those if this has a better score
            to_remove = []
            for i, (prev_kw, prev_score, prev_idx) in enumerate(filtered):
                if prev_kw in kw and prev_kw != kw:
                    to_remove.append(i)

            # Remove substrings (in reverse order to preserve indices)
            for i in reversed(to_remove):
                filtered_indices.discard(filtered[i][2])
                filtered.pop(i)

            filtered.append((kw, score, idx))
            filtered_indices.add(idx)

    # Return in original order
    result = [(kw, score) for kw, score, idx in sorted(filtered, key=lambda x: x[2])]
    return [kw for kw, _ in result], [score for _, score in result]


def _worker_extract_yake(
    args: Tuple[List[Tuple[str, str]], int, str, int, float, str, int],
) -> List[Dict[str, Union[str, List[str], List[float]]]]:
    """
    Worker function for multiprocessing YAKE extraction.

    Args:
        args: Tuple of (batch_data, top_k, language, max_ngram_size,
                       deduplication_threshold, deduplication_algo, window_size)

    Returns:
        List of result dictionaries
    """
    (
        batch_data,
        top_k,
        language,
        max_ngram_size,
        dedup_threshold,
        dedup_algo,
        window_size,
    ) = args

    # Initialize YAKE for this worker
    kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=dedup_threshold,
        dedupFunc=dedup_algo,
        windowsSize=window_size,
        top=top_k,
    )

    results = []

    for doc_id, text in batch_data:
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            results.append({"doc_id": doc_id, "keywords": [], "scores": []})
            continue

        try:
            # Extract keywords
            extracted = kw_extractor.extract_keywords(text)

            if extracted:
                keywords = [kw for kw, score in extracted]
                scores = [float(score) for kw, score in extracted]

                # Filter overlapping keywords
                keywords, scores = _filter_overlapping_keywords(keywords, scores)

                results.append({"doc_id": doc_id, "keywords": keywords, "scores": scores})
            else:
                results.append({"doc_id": doc_id, "keywords": [], "scores": []})
        except Exception:
            results.append({"doc_id": doc_id, "keywords": [], "scores": []})

    return results


class YAKEExtractor:
    """
    Extract keywords from newspaper texts using YAKE.

    YAKE (Yet Another Keyword Extractor) uses statistical text features to
    identify important keywords without external knowledge:
    - Term frequency within document
    - Term casing (capitalization patterns)
    - Term position (where it appears)
    - Term context (surrounding words)
    - Term relatedness to context

    Key advantages:
    - Unsupervised (no training needed)
    - Multi-lingual (works with German text)
    - Extracts meaningful phrases
    - Fast and efficient

    Lower scores = more important keywords (YAKE's scoring inverted from others)
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        language: str = "de",
        max_ngram_size: int = 3,
        deduplication_threshold: float = 0.9,
        deduplication_algo: str = "seqm",
        window_size: int = 1,
    ):
        """
        Initialize YAKE extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            language: Language code ("de" for German)
            max_ngram_size: Maximum n-gram size (1=words, 2=bigrams, 3=trigrams)
            deduplication_threshold: Threshold for removing duplicate keywords (0-1)
            deduplication_algo: Algorithm for deduplication ("seqm", "leve", "jaro")
            window_size: Window size for co-occurrence statistics
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.deduplication_algo = deduplication_algo
        self.window_size = window_size

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better keyword extraction
            source_dir = self.config.parsed_dir / source_name
            textblocks_file = source_dir / "textblocks.parquet"
            lines_file = source_dir / "lines.parquet"

            if textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info("Using textblocks.parquet (aggregated text blocks)")
            elif lines_file.exists():
                self.input_file = lines_file
                logger.info("Using lines.parquet (line-level data)")
            else:
                self.input_file = textblocks_file

        # Auto-resolve text column (prefer text_processed if available)
        self.text_column = resolve_text_column(
            self.text_column, file_path=str(self.input_file)
        )

        logger.info(f"Initialized YAKE extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Language: {language}")
        logger.info(f"Max n-gram size: {max_ngram_size}")

    def extract_keywords(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        batch_size: int = 1000,
        num_workers: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract keywords from documents with batching and multiprocessing.

        Args:
            top_k: Number of top keywords per document
            limit: Limit number of documents to process
            group_by: Columns to group by (aggregates text)
            batch_size: Number of documents per batch (default: 1000)
            num_workers: Number of worker processes (default: CPU count - 1)

        Returns:
            DataFrame with columns: doc_id, source_id, issue_id, page_id, text_block_id, keywords, scores

        Note:
            YAKE scores are inverted: lower score = more important keyword
        """
        start_time = time.time()

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}\n"
                f"Run parsing first: newspaper-explorer data parse --source {self.source_name}"
            )

        logger.info(f"Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)

        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Auto-aggregate by page if no grouping specified and using textblocks
        if group_by is None and "page_id" in df.columns:
            logger.info("Auto-aggregating text blocks by page for keyword extraction")
            group_by = ["page_id"]

        # Group if requested
        if group_by:
            logger.info(f"Grouping by: {', '.join(group_by)}")
            df = df.group_by(group_by).agg(pl.col(self.text_column).str.concat(" "))

        # Extract document IDs and texts
        texts = df[self.text_column].to_list()

        # Create document IDs
        if "text_block_id" in df.columns:
            doc_ids = df["text_block_id"].to_list()
        elif "filename" in df.columns:
            doc_ids = df["filename"].to_list()
        elif group_by:
            # For grouped data, create ID from group columns
            doc_ids = [
                "_".join([str(row[col]) for col in group_by])
                for row in df.select(group_by).iter_rows(named=True)
            ]
        else:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]

        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        logger.info(f"Extracting keywords from {len(texts)} documents...")
        logger.info(f"Using {num_workers} workers with batch size {batch_size}")

        # Prepare data for processing
        doc_text_pairs = list(zip(doc_ids, texts))

        # Split into batches
        batches = [
            doc_text_pairs[i : i + batch_size] for i in range(0, len(doc_text_pairs), batch_size)
        ]

        logger.info(f"Processing {len(batches)} batches...")

        # Process batches
        results = []

        if num_workers == 1:
            # Single-process mode
            kw_extractor = yake.KeywordExtractor(
                lan=self.language,
                n=self.max_ngram_size,
                dedupLim=self.deduplication_threshold,
                dedupFunc=self.deduplication_algo,
                windowsSize=self.window_size,
                top=top_k,
            )

            for batch in tqdm(batches, desc="Processing batches"):
                for doc_id, text in batch:
                    if not text or not isinstance(text, str) or len(text.strip()) < 10:
                        results.append({"doc_id": doc_id, "keywords": [], "scores": []})
                        continue

                    try:
                        extracted = kw_extractor.extract_keywords(text)
                        if extracted:
                            keywords = [kw for kw, score in extracted]
                            scores = [float(score) for kw, score in extracted]

                            # Filter overlapping keywords
                            keywords, scores = _filter_overlapping_keywords(keywords, scores)

                            results.append(
                                {"doc_id": doc_id, "keywords": keywords, "scores": scores}
                            )
                        else:
                            results.append({"doc_id": doc_id, "keywords": [], "scores": []})
                    except Exception as e:
                        logger.warning(f"Failed for {doc_id}: {e}")
                        results.append({"doc_id": doc_id, "keywords": [], "scores": []})
        else:
            # Multi-process mode
            with mp.Pool(processes=num_workers) as pool:
                # Prepare arguments for each batch
                worker_args = [
                    (
                        batch,
                        top_k,
                        self.language,
                        self.max_ngram_size,
                        self.deduplication_threshold,
                        self.deduplication_algo,
                        self.window_size,
                    )
                    for batch in batches
                ]

                # Process batches in parallel with progress bar
                for batch_results in tqdm(
                    pool.imap(_worker_extract_yake, worker_args),
                    total=len(batches),
                    desc="Processing batches",
                ):
                    results.extend(batch_results)

        # Convert to DataFrame
        results_df = pl.DataFrame(results)

        # Extract foreign keys from doc_ids
        logger.info("Extracting foreign keys from doc_ids...")
        doc_ids_list = results_df["doc_id"].to_list()
        foreign_keys = [extract_foreign_keys(doc_id) for doc_id in doc_ids_list]

        # Add foreign key columns
        results_df = results_df.with_columns(
            [
                pl.Series("source_id", [fk.source_id for fk in foreign_keys]),
                pl.Series("issue_id", [fk.issue_id for fk in foreign_keys]),
                pl.Series("page_id", [fk.page_id for fk in foreign_keys]),
                pl.Series(
                    "text_block_id",
                    [fk.text_block_id for fk in foreign_keys],
                ),
            ]
        )

        logger.info(f"Extracted keywords for {len(results_df)} documents")

        # Add grouping columns if they exist (avoid duplicates with foreign keys)
        if group_by:
            # Merge back grouping columns, but skip if they're already in foreign keys
            fk_columns = {"source_id", "issue_id", "page_id", "text_block_id"}
            new_group_cols = [col for col in group_by if col not in fk_columns]
            if new_group_cols:
                group_data = df.select(new_group_cols)
                group_data = group_data.with_columns(pl.Series("doc_id", doc_ids))
                results_df = results_df.join(group_data, on="doc_id", how="left")

        # Output structure: doc_id, source_id, issue_id, page_id, text_block_id,
        # keywords, scores, + any user-specified grouping columns

        # Store parameters and timing info for save_results
        input_metadata_file = str(self.input_file).replace(".parquet", ".json")
        self._last_params = {
            "top_k": top_k,
            "limit": limit,
            "group_by": group_by,
            "language": self.language,
            "max_ngram_size": self.max_ngram_size,
            "deduplication_threshold": self.deduplication_threshold,
            "deduplication_algo": self.deduplication_algo,
            "window_size": self.window_size,
            "input": {
                "parquet": str(self.input_file),
                "metadata": input_metadata_file,
            },
        }
        self._last_extraction_time = time.time() - start_time
        self._last_input_df = df

        return results_df

    def save_results(
        self,
        results_df: pl.DataFrame,
        output_name: str = "yake_keywords",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame
            output_name: Output filename (without extension, default: "keywords")
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
                "algorithm": "YAKE (Yet Another Keyword Extractor)",
                "language": self.language,
                "max_ngram_size": self.max_ngram_size,
                "deduplication_threshold": self.deduplication_threshold,
                "deduplication_algo": self.deduplication_algo,
                "window_size": self.window_size,
                "top_k": top_k,
                "text_column": self.text_column,
            }
        )

        # Create metadata with properly formatted timestamps
        completed_at = datetime.now().isoformat()

        metadata = AnalysisMetadata(
            analysis_id=None,  # Will be auto-generated
            analysis_type="keywords",
            method_type="yake",
            model_name="yake",
            model_version="3.0.0",  # yake package version
            source=self.source_name,
            parameters=params,
            input_data=input_stats,
            output_data=output_stats,
            granularity="textblock",  # Keyword extraction runs on textblock level
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

        logger.info(f"Saved results to {paths['results_path']}")
        logger.info(f"Metadata saved to: {paths['metadata_path']}")
        logger.info(f"Analysis ID: {metadata.analysis_id}")

        return paths["results_path"]


def extract_keywords_simple(
    source_name: str,
    top_k: int = 10,
    limit: Optional[int] = None,
    max_ngram_size: int = 2,
) -> pl.DataFrame:
    """
    Simple one-shot function to extract keywords using YAKE.

    Args:
        source_name: Source name (e.g., "der_tag")
        top_k: Keywords per document
        limit: Limit documents
        max_ngram_size: Maximum n-gram size

    Returns:
        DataFrame with keywords

    Example:
        >>> keywords_df = extract_keywords_simple("der_tag", top_k=10)
        >>> print(keywords_df)
    """
    extractor = YAKEExtractor(
        source_name=source_name,
        max_ngram_size=max_ngram_size,
    )
    return extractor.extract_keywords(top_k=top_k, limit=limit)
