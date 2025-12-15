"""
LLM-based entity extraction for newspaper text.

Extracts named entities (persons, locations, organizations) using LLM with
structured response validation and proper result storage following the
new data architecture pattern.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

import polars as pl
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.utils.metadata import create_result_metadata
from newspaper_explorer.data.utils.text import chunk_text
from newspaper_explorer.llm.client import LLMClient, LLMRetryError, LLMValidationError
from newspaper_explorer.llm.prompts.entity_extraction import ENTITY_EXTRACTION
from newspaper_explorer.models.llm.entity_extraction import EntityResponse

logger = logging.getLogger(__name__)


class LLMEntityExtractor:
    """
    Extract named entities using LLM with structured validation.

    Uses the new data architecture:
    - Saves results as Parquet with source_id foreign keys
    - Creates metadata.json for reproducibility
    - Follows results/{source}/entities/{method_id}/ structure

    Note: source_id column contains IDs from input (line_id, text_block_id, etc.)
    """

    def __init__(
        self,
        source_name: str = "der_tag",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 3,
        batch_size: int = 10,
        min_text_length: int = 100,
        max_text_length: int = 8000,
    ):
        """
        Initialize LLM entity extractor.

        Args:
            source_name: Source dataset name (e.g., "der_tag").
            model_name: LLM model to use.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens per response.
            max_retries: Number of retry attempts on failure.
            batch_size: Process N lines before saving checkpoint.
            min_text_length: Minimum text length to process (chars).
            max_text_length: Maximum text length before chunking (chars).
        """
        self.source_name = source_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        # Setup paths following new architecture
        config = get_config()
        self.config = config

        # Get prompt template
        self.prompt_template = ENTITY_EXTRACTION

        logger.info(f"Initialized LLMEntityExtractor for '{source_name}'")
        logger.info(f"Model: {model_name}, Temperature: {temperature}")
        logger.info(f"Text length: min={min_text_length}, max={max_text_length} chars")

    def _prepare_text(self, text: str) -> List[str]:
        """
        Prepare text for extraction by chunking if needed.

        Args:
            text: Input text to process.

        Returns:
            List of text chunks (single item if no chunking needed).
        """
        if len(text) <= self.max_text_length:
            return [text]

        # Chunk text at sentence boundaries
        chunks = chunk_text(
            text,
            max_length=self.max_text_length,
            split_margin=200,  # 200 char overlap for context
        )

        return chunks

    def extract_from_text(
        self, text: str, line_id: str, metadata: Optional[Dict] = None
    ) -> Optional[List[Dict]]:
        """
        Extract entities from a single text (with chunking if needed).

        Args:
            text: Text content to analyze.
            line_id: Unique line identifier.
            metadata: Optional metadata dict (source, date, newspaper_title, etc.)

        Returns:
            List of entity dictionaries with line_id, or None on failure.
        """
        # Chunk text if needed
        chunks = self._prepare_text(text)

        all_records = []

        # Process each chunk
        for chunk in chunks:
            # Format prompt with optional metadata
            prompts = self.prompt_template.format(text=chunk, metadata=metadata)

            # Make LLM request with validation
            with LLMClient(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
            ) as client:
                try:
                    response = client.complete(
                        prompt=prompts["user"],
                        system_prompt=prompts["system"],
                        response_schema=EntityResponse,
                    )

                    # Ensure response is EntityResponse object (handle string response)
                    if isinstance(response, str):
                        response = EntityResponse.model_validate_json(response)

                    # Convert to flat records (one row per entity)
                    for person in response.persons:
                        all_records.append(
                            {
                                "source_id": line_id,
                                "entity_text": person,
                                "entity_type": "person",
                            }
                        )

                    for location in response.locations:
                        all_records.append(
                            {
                                "source_id": line_id,
                                "entity_text": location,
                                "entity_type": "location",
                            }
                        )

                    for org in response.organizations:
                        all_records.append(
                            {
                                "source_id": line_id,
                                "entity_text": org,
                                "entity_type": "organization",
                            }
                        )

                except (LLMRetryError, LLMValidationError) as e:
                    logger.warning(f"Failed to extract entities for {line_id} (chunk): {e}")
                    # Continue with next chunk instead of returning None
                    continue

        # Deduplicate entities found across chunks (keep all instances since LLM has no confidence)
        if all_records:
            # Use set to deduplicate based on (source_id, entity_text, entity_type)
            seen = set()
            unique_records = []
            for record in all_records:
                key = (record["source_id"], record["entity_text"], record["entity_type"])
                if key not in seen:
                    seen.add(key)
                    unique_records.append(record)

            return unique_records if unique_records else None

        return None

    def extract_from_dataframe(
        self,
        df: pl.DataFrame,
        text_column: str = "text",
        id_column: str = "line_id",
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Extract entities from a Polars DataFrame.

        Args:
            df: DataFrame with text data.
            text_column: Column containing text content.
            id_column: Column containing unique identifiers.
            limit: Optional limit on number of rows to process.

        Returns:
            DataFrame with extracted entities (source_id, entity_text, entity_type).
        """
        logger.info(f"Extracting entities from {len(df)} rows")

        # Filter by minimum text length
        df_filtered = df.filter(pl.col(text_column).str.len_chars() >= self.min_text_length)
        logger.info(f"Filtered to {len(df_filtered)} texts (min length: {self.min_text_length})")

        if limit:
            df_filtered = df_filtered.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Prepare texts and check for chunking
        logger.info("Preparing texts and chunking long texts...")
        chunks_needed = 0
        for row in df_filtered.iter_rows(named=True):
            text = row[text_column]
            if len(text) > self.max_text_length:
                chunks_needed += 1

        total_texts = len(df_filtered)
        logger.info(f"Prepared {total_texts} texts")
        logger.info(f"  - {chunks_needed} texts will be split into chunks")

        all_records = []
        processed = 0
        failed = 0

        # Process each row
        for row in tqdm(
            df_filtered.iter_rows(named=True), total=len(df_filtered), desc="Extracting entities"
        ):
            text = row[text_column]
            line_id = row[id_column]

            # Build metadata from row if available
            metadata = {}
            for field in ["source", "newspaper_title", "date", "year_volume", "page_number"]:
                if field in row and row[field]:
                    metadata[field] = row[field]

            # Extract entities with metadata (handles chunking internally)
            records = self.extract_from_text(text, line_id, metadata=metadata if metadata else None)

            if records:
                all_records.extend(records)
                processed += 1
            else:
                failed += 1

            # Small delay to avoid rate limits
            time.sleep(0.1)

        logger.info(f"Processed: {processed}, Failed: {failed}")
        logger.info(f"Total entities extracted: {len(all_records)}")

        # Convert to DataFrame
        if all_records:
            results_df = pl.DataFrame(all_records)
        else:
            # Empty DataFrame with correct schema
            results_df = pl.DataFrame(
                schema={
                    "source_id": pl.Utf8,
                    "entity_text": pl.Utf8,
                    "entity_type": pl.Utf8,
                }
            )

        return results_df

    def extract_and_save(
        self,
        source_parquet: Optional[Path] = None,
        limit: Optional[int] = None,
        text_column: str = "text",
        id_column: str = "line_id",
    ) -> Dict:
        """
        Complete extraction pipeline: load, extract, save with metadata.

        Args:
            source_parquet: Path to source parquet file. If None, loads from DataLoader.
            limit: Optional limit on rows to process (for testing).
            text_column: Column containing text.
            id_column: Column containing line IDs.

        Returns:
            Dictionary with extraction statistics and output paths.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("LLM Entity Extraction Pipeline")
        logger.info("=" * 60)

        # Load data
        if source_parquet:
            logger.info(f"Loading data from {source_parquet}")
            df = pl.read_parquet(source_parquet)
        else:
            logger.info(f"Loading data using DataIngester for '{self.source_name}'")
            ingester = DataIngester(source_name=self.source_name)
            df = ingester.load_source()

        logger.info(f"Loaded {len(df)} lines")

        # Extract entities
        results_df = self.extract_from_dataframe(
            df, text_column=text_column, id_column=id_column, limit=limit
        )

        # Calculate duration
        duration = time.time() - start_time

        # Create metadata
        metadata = create_result_metadata(
            analysis_type="entities",
            method_type="llm",
            model_name=self.model_name,
            source=self.source_name,
            parameters={
                "model": self.model_name,  # Full model name
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "prompt_template": "entity_extraction",
                "text_column": text_column,
                "source_id_column": id_column,  # Track what source_id references
            },
            line_count=limit if limit else len(df),
            duration_seconds=duration,
        )

        # Setup output directory: results/{source}/entities/{method_id}/
        output_dir = (
            self.config.results_dir / self.source_name / "entities" / metadata["analysis_id"]
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_path = output_dir / "entities.parquet"
        results_df.write_parquet(results_path, compression="zstd")
        logger.info(f"Saved {len(results_df)} entities to {results_path}")

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")

        # Summary
        logger.info("=" * 60)
        logger.info("Extraction Complete!")
        logger.info(f"Method ID: {metadata['analysis_id']}")
        logger.info(f"Total entities: {len(results_df)}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 60)

        return {
            "metadata": metadata,
            "results_df": results_df,
            "output_dir": output_dir,
            "results_path": results_path,
            "metadata_path": metadata_path,
        }


def extract_entities_llm(
    source_name: str = "der_tag",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    limit: Optional[int] = None,
) -> Dict:
    """
    Convenience function for LLM-based entity extraction.

    Args:
        source_name: Source dataset name.
        model_name: LLM model to use.
        temperature: Sampling temperature.
        limit: Optional limit on rows (for testing).

    Returns:
        Extraction results dictionary.

    Example:
        ```python
        from newspaper_explorer.analyzeities.llm_extraction import extract_entities_llm

        # Extract from first 100 lines (testing)
        results = extract_entities_llm(
            source_name="der_tag",
            model_name="gpt-4o-mini",
            limit=100
        )

        # Full extraction
        results = extract_entities_llm(source_name="der_tag")
        ```
    """
    extractor = LLMEntityExtractor(
        source_name=source_name, model_name=model_name, temperature=temperature
    )

    return extractor.extract_and_save(limit=limit)
