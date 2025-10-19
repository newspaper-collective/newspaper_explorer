"""
LLM-based entity extraction for newspaper text.

Extracts named entities (persons, locations, organizations) using LLM with
structured response validation and proper result storage following the
new data architecture pattern.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from tqdm import tqdm

from newspaper_explorer.data.loading import DataLoader
from newspaper_explorer.utils.config import get_config
from newspaper_explorer.utils.llm import LLMClient, LLMRetryError, LLMValidationError
from newspaper_explorer.utils.prompts import get_prompt
from newspaper_explorer.utils.queries import create_result_metadata
from newspaper_explorer.utils.schemas import EntityResponse

logger = logging.getLogger(__name__)


class LLMEntityExtractor:
    """
    Extract named entities using LLM with structured validation.

    Uses the new data architecture:
    - Saves results as Parquet with line_id foreign keys
    - Creates metadata.json for reproducibility
    - Follows results/{source}/entities/{method_id}/ structure
    """

    def __init__(
        self,
        source_name: str = "der_tag",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        max_retries: int = 3,
        batch_size: int = 10,
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
        """
        self.source_name = source_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Setup paths following new architecture
        config = get_config()
        self.config = config

        # Get prompt template
        self.prompt_template = get_prompt("entity_extraction")

        logger.info(f"Initialized LLMEntityExtractor for '{source_name}'")
        logger.info(f"Model: {model_name}, Temperature: {temperature}")

    def extract_from_text(self, text: str, line_id: str) -> Optional[Dict]:
        """
        Extract entities from a single text.

        Args:
            text: Text content to analyze.
            line_id: Unique line identifier.

        Returns:
            Dictionary with line_id and extracted entities, or None on failure.
        """
        # Format prompt
        prompts = self.prompt_template.format(text=text)

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

                # Convert to flat records (one row per entity)
                records = []

                for person in response.persons:
                    records.append(
                        {
                            "line_id": line_id,
                            "entity_text": person,
                            "entity_type": "person",
                        }
                    )

                for location in response.locations:
                    records.append(
                        {
                            "line_id": line_id,
                            "entity_text": location,
                            "entity_type": "location",
                        }
                    )

                for org in response.organizations:
                    records.append(
                        {
                            "line_id": line_id,
                            "entity_text": org,
                            "entity_type": "organization",
                        }
                    )

                return records

            except (LLMRetryError, LLMValidationError) as e:
                logger.warning(f"Failed to extract entities for {line_id}: {e}")
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
            DataFrame with extracted entities (line_id, entity_text, entity_type).
        """
        logger.info(f"Extracting entities from {len(df)} rows")

        if limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} rows")

        all_records = []
        processed = 0
        failed = 0

        # Process each row
        for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Extracting entities"):
            text = row[text_column]
            line_id = row[id_column]

            # Skip empty or very short texts
            if not text or len(text.strip()) < 50:
                continue

            # Extract entities
            records = self.extract_from_text(text, line_id)

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
                    "line_id": pl.Utf8,
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
            logger.info(f"Loading data using DataLoader for '{self.source_name}'")
            loader = DataLoader(source_name=self.source_name)
            df = loader.load_source()

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
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "prompt_template": "entity_extraction",
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
        from newspaper_explorer.analysis.entities.llm_extraction import extract_entities_llm

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
