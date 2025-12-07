"""
LLM-based keyword extraction for newspaper texts.

Uses large language models to extract semantically meaningful keywords and keyphrases
from newspaper articles. Unlike statistical methods (TF-IDF, YAKE, RAKE), this approach
leverages LLM understanding of language, context, and semantics to identify truly
important keywords.

LLM extraction is particularly good for:
- Understanding context and semantic importance
- Handling historical German language variations
- Extracting both explicit and implicit key concepts
- Balancing single words and meaningful phrases
- Understanding domain-specific terminology

Example:
    >>> from newspaper_explorer.analyze.keywords.llm import LLMKeywordExtractor
    >>> extractor = LLMKeywordExtractor(source_name="der_tag")
    >>> keywords = extractor.extract_keywords(top_k=15, batch_size=10)
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import List, Optional

import polars as pl
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.ids import extract_foreign_keys
from newspaper_explorer.data.utils.results import save_analysis_results
from newspaper_explorer.data.utils.stats import extract_input_stats, extract_output_stats
from newspaper_explorer.llm.client import LLMClient
from newspaper_explorer.llm.prompts.keyword_extraction import KEYWORD_EXTRACTION
from newspaper_explorer.models.data.metadata import AnalysisMetadata
from newspaper_explorer.models.llm.keyword_extraction import KeywordResponse

logger = logging.getLogger(__name__)


class LLMKeywordExtractor:
    """
    Extract keywords from newspaper texts using LLMs.

    Uses large language models to identify semantically important keywords
    and keyphrases based on understanding of content, context, and language.
    Provides confidence scores for each keyword.
    """

    def __init__(
        self,
        source_name: str,
        input_file: Optional[Path] = None,
        text_column: str = "text",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        """
        Initialize LLM keyword extractor.

        Args:
            source_name: Name of the source (e.g., "der_tag")
            input_file: Custom input parquet file (default: textblocks.parquet)
            text_column: Name of column containing text
            model_name: LLM model to use (default: gpt-4o-mini)
            temperature: LLM temperature (0.0-1.0, lower = more focused)
            max_tokens: Maximum tokens in response
        """
        self.source_name = source_name
        self.text_column = text_column
        self.config = get_config()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Determine input file
        if input_file:
            self.input_file = Path(input_file)
        else:
            # Default to textblocks for better context
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

        logger.info(f"Initialized LLM keyword extractor for {source_name}")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Model: {model_name}, Temperature: {temperature}")

    def extract_keywords(
        self,
        top_k: int = 15,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        batch_size: int = 10,
        include_metadata: bool = True,
    ) -> pl.DataFrame:
        """
        Extract keywords from documents using LLM.

        Args:
            top_k: Number of top keywords per document (default: 15)
            limit: Limit number of documents to process (for testing)
            group_by: Columns to group by (aggregates text)
            batch_size: Number of documents to process before saving progress
            include_metadata: Include source metadata in prompts for context

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
            doc_ids = [
                "_".join([str(row[col]) for col in group_by])
                for row in df.select(group_by).iter_rows(named=True)
            ]
        else:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]

        # Prepare metadata if available
        metadata_list = []
        if include_metadata:
            for i, row in enumerate(df.iter_rows(named=True)):
                metadata = {
                    "source": self.source_name,
                    "date": str(row.get("date", "unknown")),
                }
                # Add any other available metadata
                if "newspaper_title" in df.columns:
                    metadata["newspaper_title"] = row.get("newspaper_title", "")
                if "year_volume" in df.columns:
                    metadata["year_volume"] = row.get("year_volume", "")
                if "page_number" in df.columns:
                    metadata["page_number"] = row.get("page_number", "")
                metadata_list.append(metadata)
        else:
            metadata_list = [{}] * len(texts)

        logger.info(f"Extracting keywords from {len(texts)} documents using {self.model_name}...")
        logger.info(f"Batch size: {batch_size}")

        # Initialize LLM client
        client = LLMClient(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Process documents with progress bar
        results = []
        for i in tqdm(range(len(texts)), desc="Processing documents"):
            doc_id = doc_ids[i]
            text = texts[i]
            metadata = metadata_list[i]

            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                results.append({"doc_id": doc_id, "keywords": [], "scores": []})
                continue

            try:
                # Format prompt with text and metadata
                prompts = KEYWORD_EXTRACTION.format(text=text, metadata=metadata)

                # Get LLM response
                response = client.complete(
                    prompt=prompts["user"],
                    system_prompt=prompts["system"],
                    response_schema=KeywordResponse,
                )

                # Validate response type
                if not isinstance(response, KeywordResponse):
                    logger.warning(f"Unexpected response type for doc {doc_id}: {type(response)}")
                    results.append({"doc_id": doc_id, "keywords": [], "scores": []})
                    continue

                # Limit to top_k keywords
                keywords = response.keywords[:top_k] if response.keywords else []
                scores = response.scores[:top_k] if response.scores else []

                results.append({"doc_id": doc_id, "keywords": keywords, "scores": scores})

            except Exception as e:
                logger.warning(f"Failed to extract keywords for doc {doc_id}: {e}")
                results.append({"doc_id": doc_id, "keywords": [], "scores": []})

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

        # Add grouping columns if they exist (avoid duplicates with foreign keys)
        if group_by:
            # Merge back grouping columns, but skip if they're already in foreign keys
            fk_columns = {"source_id", "issue_id", "page_id", "text_block_id"}
            new_group_cols = [col for col in group_by if col not in fk_columns]
            if new_group_cols:
                group_data = df.select(new_group_cols)
                group_data = group_data.with_columns(pl.Series("doc_id", doc_ids))
                results_df = results_df.join(group_data, on="doc_id", how="left")

        # Store parameters and timing info for save_results
        input_metadata_file = str(self.input_file).replace(".parquet", ".json")
        self._last_params = {
            "top_k": top_k,
            "limit": limit,
            "group_by": group_by,
            "batch_size": batch_size,
            "include_metadata": include_metadata,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        output_name: str = "llm_keywords",
        top_k: Optional[int] = None,
    ) -> Path:
        """
        Save results to parquet file with metadata.

        Args:
            results_df: Results DataFrame
            output_name: Output filename (without extension)
            top_k: Number of keywords (for metadata)

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
                "algorithm": "LLM-based keyword extraction",
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_k": top_k,
                "text_column": self.text_column,
            }
        )

        # Create metadata with properly formatted timestamps
        completed_at = datetime.now().isoformat()

        metadata = AnalysisMetadata(
            analysis_id=None,  # Will be auto-generated
            analysis_type="keywords",
            method_type="llm",
            model_name=self.model_name,
            model_version="1.0.0",  # LLM keyword extraction version
            source=self.source_name,
            parameters=params,
            input_data=input_stats,
            output_data=output_stats,
            granularity="textblock",  # LLM keyword extraction runs on textblock level
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
