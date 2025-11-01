"""
GLiNER-based entity extraction for newspaper text.

Extracts named entities (persons, organizations, locations) using GLiNER model
with structured response and proper result storage following the new data
architecture pattern.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch
from gliner import GLiNER
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.loading.loader import DataLoader
from newspaper_explorer.data.utils.text import normalize_german_text
from newspaper_explorer.analysis.query.engine import create_result_metadata

logger = logging.getLogger(__name__)


class GLiNEREntityExtractor:
    """
    Extract named entities using GLiNER model.

    Uses the new data architecture:
    - Saves results as Parquet with line_id foreign keys
    - Creates metadata.json for reproducibility
    - Follows results/{source}/entities/{method_id}/ structure
    """

    # Map GLiNER labels to standard types
    LABEL_MAPPING = {
        "Person": "person",
        "Organisation": "organization",
        "Ereignis": "event",
        "Ort": "location",
    }

    def __init__(
        self,
        source_name: str = "der_tag",
        model_name: str = "urchade/gliner_multi-v2.1",
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        min_text_length: int = 100,
        max_text_length: int = 500,
        normalize: bool = True,
    ):
        """
        Initialize GLiNER entity extractor.

        Args:
            source_name: Source dataset name (e.g., "der_tag").
            model_name: GLiNER model from Hugging Face Hub.
            labels: Entity labels to extract (default: Person, Organisation, Ereignis, Ort).
            threshold: Confidence threshold for entity extraction (0-1).
            batch_size: Batch size for processing (adjust based on GPU memory).
            min_text_length: Minimum text length to process (skip shorter texts).
            max_text_length: Maximum text length to use (truncate longer texts).
            normalize: Whether to normalize German text before extraction.
        """
        self.source_name = source_name
        self.model_name = model_name
        self.labels = labels or ["Person", "Organisation", "Ereignis", "Ort"]
        self.threshold = threshold
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.normalize = normalize

        # Setup paths following new architecture
        self.config = get_config()

        # Model will be loaded on first use
        self.model = None
        self.device = None

        logger.info(f"Initialized GLiNEREntityExtractor for '{source_name}'")
        logger.info(f"Model: {model_name}")
        logger.info(f"Labels: {self.labels}")

    def _load_model(self):
        """Load GLiNER model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self.model = GLiNER.from_pretrained(self.model_name)

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")

    def _prepare_text(self, text: str) -> str:
        """
        Prepare a single text for extraction.

        Args:
            text: Raw text.

        Returns:
            Prepared text (normalized and truncated).
        """
        if self.normalize:
            text = normalize_german_text(text)

        # Truncate to max length
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        return text

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
            DataFrame with extracted entities (line_id, entity_text, entity_type, confidence).
        """
        self._load_model()

        logger.info(f"Extracting entities from {len(df)} rows")

        # Filter by text length
        df_filtered = df.filter(pl.col(text_column).str.len_chars() >= self.min_text_length)
        logger.info(f"Filtered to {len(df_filtered)} texts (min length: {self.min_text_length})")

        if limit:
            df_filtered = df_filtered.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Prepare texts
        logger.info("Preparing texts...")
        texts = []
        line_ids = []

        for row in df_filtered.iter_rows(named=True):
            text = self._prepare_text(row[text_column])
            texts.append(text)
            line_ids.append(row[id_column])

        # Extract entities in batches
        logger.info(f"Batch size: {self.batch_size}, Threshold: {self.threshold}")

        all_records = []

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Extracting entities (batched)",
        ):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = line_ids[i : i + self.batch_size]

            # Batch inference
            batch_entities = self.model.batch_predict_entities(
                batch_texts, self.labels, threshold=self.threshold
            )

            # Convert to records
            for line_id, entities in zip(batch_ids, batch_entities):
                for entity in entities:
                    # Map label to standard type
                    entity_type = self.LABEL_MAPPING.get(entity["label"], entity["label"].lower())

                    all_records.append(
                        {
                            "line_id": line_id,
                            "entity_text": entity["text"],
                            "entity_type": entity_type,
                            "confidence": float(entity.get("score", 0.0)),
                        }
                    )

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
                    "confidence": pl.Float64,
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
        logger.info("GLiNER Entity Extraction Pipeline")
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
            method_type="gliner",
            model_name=self.model_name.replace("/", "_").replace(".", "_"),
            source=self.source_name,
            parameters={
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "labels": self.labels,
                "normalize": self.normalize,
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
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


def extract_entities_gliner(
    source_name: str = "der_tag",
    model_name: str = "urchade/gliner_multi-v2.1",
    threshold: float = 0.5,
    batch_size: int = 32,
    limit: Optional[int] = None,
    normalize: bool = True,
) -> Dict:
    """
    Convenience function for GLiNER-based entity extraction.

    Args:
        source_name: Source dataset name.
        model_name: GLiNER model from Hugging Face Hub.
        threshold: Confidence threshold (0-1).
        batch_size: Batch size for processing.
        limit: Optional limit on rows (for testing).
        normalize: Whether to normalize text.

    Returns:
        Extraction results dictionary.

    Example:
        ```python
        from newspaper_explorer.analysis.entities.gliner_extraction import extract_entities_gliner

        # Extract from first 100 lines (testing)
        results = extract_entities_gliner(
            source_name="der_tag",
            limit=100
        )

        # Full extraction
        results = extract_entities_gliner(source_name="der_tag")
        ```
    """
    extractor = GLiNEREntityExtractor(
        source_name=source_name,
        model_name=model_name,
        threshold=threshold,
        batch_size=batch_size,
        normalize=normalize,
    )

    return extractor.extract_and_save(limit=limit)
