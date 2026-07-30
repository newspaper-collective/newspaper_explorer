"""
RAKE keyword extraction for newspaper texts.

Implements RAKE (Rapid Automatic Keyword Extraction) to extract multi-word
keyphrases from newspaper articles. RAKE identifies candidate keywords based
on word co-occurrence and word frequency, making it excellent for extracting
meaningful phrases rather than just single words.

Unlike TF-IDF (which finds statistically distinctive words) or LDA (which finds
topic-based terms), RAKE extracts semantically meaningful keyphrases based on
linguistic patterns.

Example:
    >>> from newspaper_explorer.analyze.keywords.rake import RAKEExtractor
    >>> extractor = RAKEExtractor(source_name="der_tag")
    >>> keywords = extractor.extract_keyphrases(top_k=10)
"""

from datetime import datetime
import logging
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
from rake_nltk import Rake
from tqdm import tqdm

from newspaper_explorer.cli.utils.options import resolve_text_column
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.models.data.metadata import AnalysisMetadata

logger = logging.getLogger(__name__)


def _worker_extract_batch(
    args: Tuple[List[Tuple[str, str]], int, List[str], int, int],
) -> List[Dict[str, Any]]:
    """
    Worker function for multiprocessing RAKE extraction.

    Args:
        args: Tuple of (batch_data, top_k, stopwords, min_length, max_length)

    Returns:
        List of result dictionaries
    """
    batch_data, top_k, stopwords, min_length, max_length = args

    # Initialize RAKE for this worker
    rake = Rake(
        stopwords=set(stopwords) if stopwords else None,
        min_length=min_length,
        max_length=max_length,
    )

    results = []

    for doc_id, text in batch_data:
        if not text or not isinstance(text, str):
            results.append(
                {
                    "doc_id": doc_id,
                    "keywords": [],
                    "scores": [],
                }
            )
            continue

        # Extract keyphrases
        rake.extract_keywords_from_text(text)
        ranked_phrases = rake.get_ranked_phrases_with_scores()

        # Take top k
        top_phrases = ranked_phrases[:top_k]

        if top_phrases:
            scores, phrases = zip(*top_phrases)
            results.append(
                {
                    "doc_id": doc_id,
                    "keywords": list(phrases),
                    "scores": list(scores),  # RAKE scores are already numeric
                }
            )
        else:
            results.append(
                {
                    "doc_id": doc_id,
                    "keywords": [],
                    "scores": [],
                }
            )

    return results


class RAKEExtractor:
    """
    Extract keyphrases from newspaper texts using RAKE.

    RAKE (Rapid Automatic Keyword Extraction) identifies candidate keyphrases
    by analyzing word co-occurrence patterns and word frequencies. It's
    particularly good at extracting multi-word expressions like "erste weltkrieg"
    or "sozialdemokratische partei".

    Key advantages:
    - Extracts meaningful multi-word phrases
    - Language-independent (works with any language)
    - Fast and efficient
    - No training required

    Methodology:
    1. Splits text on delimiters (punctuation, stopwords)
    2. Identifies candidate keyphrases (sequences of content words)
    3. Scores phrases based on word frequency and co-occurrence
    4. Returns top-ranked phrases
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        use_stopwords: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        min_phrase_length: int = 1,
        max_phrase_length: int = 3,
    ):
        """
        Initialize RAKE extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            use_stopwords: Whether to use German stopwords
            custom_stopwords: Additional stopwords to exclude
            min_phrase_length: Minimum words per keyphrase
            max_phrase_length: Maximum words per keyphrase
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better phrase extraction
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
        self.text_column = resolve_text_column(self.text_column, file_path=str(self.input_file))

        # Setup stopwords for RAKE
        stopwords = self._get_stopwords(use_stopwords, custom_stopwords)

        # Initialize RAKE with German stopwords
        self.rake = Rake(
            stopwords=set(stopwords) if stopwords else None,
            min_length=min_phrase_length,
            max_length=max_phrase_length,
        )

        logger.info(f"Initialized RAKE extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Phrase length: {min_phrase_length}-{max_phrase_length} words")
        logger.info(f"Using {len(stopwords)} stopwords")

    def _get_stopwords(
        self, use_stopwords: bool, custom_stopwords: Optional[List[str]]
    ) -> List[str]:
        """Get German stopwords list from SpaCy."""
        stopwords: List[str] = []

        if use_stopwords:
            from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS

            stopwords = list(DE_STOP_WORDS)
            logger.info(f"Loaded {len(stopwords)} German stopwords from SpaCy")

        # Add custom stopwords
        if custom_stopwords:
            stopwords.extend(custom_stopwords)
            logger.info(f"Added {len(custom_stopwords)} custom stopwords")

        return stopwords

    def _extract_batch(
        self,
        batch_data: List[Tuple[str, str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Extract keyphrases from a batch of documents.

        Args:
            batch_data: List of (doc_id, text) tuples
            top_k: Number of top keyphrases per document

        Returns:
            List of result dictionaries
        """
        results = []

        for doc_id, text in batch_data:
            if not text or not isinstance(text, str):
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )
                continue

            # Extract keyphrases
            self.rake.extract_keywords_from_text(text)
            ranked_phrases = self.rake.get_ranked_phrases_with_scores()

            # Take top k
            top_phrases = ranked_phrases[:top_k]

            if top_phrases:
                scores, phrases = zip(*top_phrases)
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": list(phrases),
                        "scores": list(scores),
                    }
                )
            else:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )

        return results

    def extract_keyphrases(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        batch_size: int = 1000,
        num_workers: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract keyphrases from documents with batching and multiprocessing.

        Args:
            top_k: Number of top keyphrases per document
            limit: Limit number of documents to process
            group_by: Columns to group by (aggregates text)
            batch_size: Number of documents per batch (default: 1000)
            num_workers: Number of worker processes (default: CPU count - 1)

        Returns:
            DataFrame with columns: doc_id, source_id, issue_id, page_id, text_block_id, keywords, scores
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

        logger.info(f"Extracting keyphrases from {len(texts)} documents...")
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
            for batch in tqdm(batches, desc="Processing batches"):
                batch_results = self._extract_batch(batch, top_k)
                results.extend(batch_results)
        else:
            # Multi-process mode
            with mp.Pool(processes=num_workers) as pool:
                # Create worker function with fixed parameters
                worker_fn = _worker_extract_batch

                # Get stopwords once for all workers
                worker_stopwords = self._get_stopwords(True, None)

                # Prepare arguments for each batch
                worker_args = [
                    (
                        batch,
                        top_k,
                        worker_stopwords,
                        self.min_phrase_length,
                        self.max_phrase_length,
                    )
                    for batch in batches
                ]

                # Process batches in parallel with progress bar
                for batch_results in tqdm(
                    pool.imap(worker_fn, worker_args),
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

        logger.info(f"Extracted keyphrases for {len(results_df)} documents")

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
            "batch_size": batch_size,
            "num_workers": num_workers,
            "min_phrase_length": self.min_phrase_length,
            "max_phrase_length": self.max_phrase_length,
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
        output_name: str = "rake_keywords",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame
            output_name: Output filename (without extension, default: "keywords")
            top_k: Number of keyphrases (for metadata, optional)

        Returns:
            Path to saved file
        """
        logger.info("Creating metadata...")

        # Calculate statistics
        total_keyphrases = sum(len(kp_list) for kp_list in results_df["keywords"].to_list())
        avg_keyphrases = total_keyphrases / len(results_df) if len(results_df) > 0 else 0
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
                "total_keyphrases": total_keyphrases,
                "avg_keyphrases_per_doc": round(avg_keyphrases, 2),
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
                "algorithm": "RAKE (Rapid Automatic Keyword Extraction)",
                "min_phrase_length": self.min_phrase_length,
                "max_phrase_length": self.max_phrase_length,
                "top_k": top_k,
                "text_column": self.text_column,
            }
        )

        # Create metadata with properly formatted timestamps
        completed_at = datetime.now().isoformat()

        metadata = AnalysisMetadata(
            analysis_id=None,  # Will be auto-generated
            analysis_type="keywords",
            method_type="rake",
            model_name="rake_nltk",
            model_version="1.0.0",  # rake_nltk package
            source=self.source_name,
            parameters=params,
            input_data=input_stats,
            output_data=output_stats,
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


def extract_keyphrases_simple(
    source_name: str,
    top_k: int = 10,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Simple one-shot function to extract keyphrases using RAKE.

    Args:
        source_name: Source name (e.g., "der_tag")
        top_k: Keyphrases per document
        limit: Limit documents

    Returns:
        DataFrame with keyphrases

    Example:
        >>> keyphrases_df = extract_keyphrases_simple("der_tag", top_k=10)
        >>> print(keyphrases_df)
    """
    extractor = RAKEExtractor(source_name=source_name)
    return extractor.extract_keyphrases(top_k=top_k, limit=limit)
