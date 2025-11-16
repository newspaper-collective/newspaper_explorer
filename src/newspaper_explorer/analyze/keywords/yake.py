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

import logging
import time
from pathlib import Path
from typing import List, Optional

import polars as pl
import yake
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.metadata import (
    AnalysisMetadata,
    save_metadata,
    save_analysis_results,
    extract_input_stats,
    extract_output_stats,
)

logger = logging.getLogger(__name__)


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
            source_dir = self.config.data_dir / "raw" / source_name / "text"
            textblocks_file = source_dir / f"{source_name}_textblocks.parquet"
            lines_file = source_dir / f"{source_name}_lines.parquet"

            if textblocks_file.exists():
                self.input_file = textblocks_file
                logger.info("Using textblocks.parquet (aggregated text blocks)")
            elif lines_file.exists():
                self.input_file = lines_file
                logger.info("Using lines.parquet (line-level data)")
            else:
                self.input_file = textblocks_file

        logger.info(f"Initialized YAKE extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Language: {language}")
        logger.info(f"Max n-gram size: {max_ngram_size}")

    def extract_keywords(
        self,
        top_k: int = 10,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Extract keywords from documents.

        Args:
            top_k: Number of top keywords per document
            limit: Limit number of documents to process
            group_by: Columns to group by (aggregates text)

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

        # Initialize YAKE keyword extractor
        kw_extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            dedupFunc=self.deduplication_algo,
            windowsSize=self.window_size,
            top=top_k,
        )

        # Extract keywords
        logger.info(f"Extracting keywords from {len(texts)} documents...")
        results = []

        for doc_id, text in tqdm(zip(doc_ids, texts), total=len(texts), desc="Extracting keywords"):
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )
                continue

            try:
                # Extract keywords (returns list of (keyword, score) tuples)
                extracted = kw_extractor.extract_keywords(text)

                if extracted:
                    keywords = [kw for kw, score in extracted]
                    scores = [float(score) for kw, score in extracted]

                    results.append(
                        {
                            "doc_id": doc_id,
                            "keywords": keywords,
                            "scores": scores,
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
            except Exception as e:
                logger.warning(f"Failed to extract keywords for {doc_id}: {e}")
                results.append(
                    {
                        "doc_id": doc_id,
                        "keywords": [],
                        "scores": [],
                    }
                )

        # Convert to DataFrame
        results_df = pl.DataFrame(results)

        # Extract foreign keys from doc_ids
        logger.info("Extracting foreign keys from doc_ids...")
        doc_ids_list = results_df["doc_id"].to_list()
        foreign_keys = [extract_foreign_keys(doc_id) for doc_id in doc_ids_list]

        # Add foreign key columns
        results_df = results_df.with_columns(
            [
                pl.Series("source_id", [fk["source_id"] for fk in foreign_keys]),
                pl.Series("issue_id", [fk["issue_id"] for fk in foreign_keys]),
                pl.Series("page_id", [fk["page_id"] for fk in foreign_keys]),
                pl.Series(
                    "text_block_id",
                    [fk.get("text_block_id", fk.get("line_id", "")) for fk in foreign_keys],
                ),
            ]
        )

        logger.info(f"Extracted keywords for {len(results_df)} documents")

        # Add grouping columns if they exist
        if group_by:
            # Merge back grouping columns
            group_data = df.select(group_by)
            group_data = group_data.with_columns(pl.Series("doc_id", doc_ids))
            results_df = results_df.join(group_data, on="doc_id", how="left")

        # Output structure: doc_id, source_id, issue_id, page_id, text_block_id,
        # keywords, scores, + any user-specified grouping columns

        # Store timing info for save_results
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

        # Create metadata
        metadata = AnalysisMetadata(
            analysis_type="keywords",
            method_type="yake",
            model_name="yake",
            source=self.source_name,
            parameters={
                "algorithm": "YAKE (Yet Another Keyword Extractor)",
                "language": self.language,
                "max_ngram_size": self.max_ngram_size,
                "deduplication_threshold": self.deduplication_threshold,
                "deduplication_algo": self.deduplication_algo,
                "window_size": self.window_size,
                "top_k": top_k,
                "text_column": self.text_column,
            },
            input_data=input_stats,
            output_data=output_stats,
            status="completed",
            duration_seconds=duration,
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
