"""
GLiNER2-based entity extraction for newspaper text.

Extracts named entities using GLiNER2 unified model (205M parameters) with
efficient CPU-based inference, structured response and proper result storage
following the new data architecture pattern.

GLiNER2 provides:
- Faster CPU inference than GLiNER v1
- Unified API for entity extraction, classification, and structured data
- 100% local processing without external dependencies
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
from gliner2 import GLiNER2
from tqdm import tqdm

from newspaper_explorer.config.base import get_config, get_project_root
from newspaper_explorer.data.ingest.loader import DataIngester
from newspaper_explorer.data.utils.text import chunk_text
from newspaper_explorer.analyze.query.engine import create_result_metadata

logger = logging.getLogger(__name__)

# Set cache directory for Hugging Face models
_project_root = get_project_root()
os.environ["HF_HOME"] = str(_project_root / ".cache" / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(_project_root / ".cache" / "huggingface" / "transformers")


class GLiNER2EntityExtractor:
    """
    Extract named entities using GLiNER2 model.

    Uses the new data architecture:
    - Saves results as Parquet with source_id foreign keys
    - Creates metadata.json for reproducibility
    - Follows results/{source}/entities/{method_id}/ structure

    Note: source_id column contains IDs from input (line_id, text_block_id, etc.)

    GLiNER2 advantages:
    - Optimized CPU inference (no GPU required)
    - Support for label descriptions for better accuracy
    - Single 205M parameter model
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
        model_name: str = "fastino/gliner2-base-0207",
        labels: Optional[Union[List[str], Dict[str, Union[str, Dict]]]] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        min_text_length: int = 100,
        max_text_length: int = 2500,
    ):
        """
        Initialize GLiNER2 entity extractor.

        Args:
            source_name: Source dataset name (e.g., "der_tag").
            model_name: GLiNER2 model from Hugging Face Hub.
                       Default: "fastino/gliner2-base-0207" (205M, fast CPU inference)
                       Alternative: "fastino/gliner2-large-1006" (340M, higher accuracy)
            labels: Entity labels to extract. Can be:
                   - List of strings: ["Person", "Organisation", "Ort", "Ereignis"]
                   - Dict with descriptions: {"Person": "Names of people", ...}
            threshold: Confidence threshold for entity extraction (0-1).
            batch_size: Batch size for processing.
            min_text_length: Minimum text length to process (skip shorter texts).
            max_text_length: Maximum text length for chunking (~2500 chars optimal).
        """
        self.source_name = source_name
        self.model_name = model_name

        # Setup labels with optional descriptions
        if labels is None:
            # Default German newspaper labels with descriptions
            self.labels: Union[List[str], Dict[str, Any]] = {
                "Person": "Namen von Personen, historischen Figuren oder Individuen",
                "Organisation": "Organisationen, Unternehmen, Institutionen oder Behörden",
                "Ort": "Geografische Orte, Städte, Länder oder Regionen",
                "Ereignis": "Historische Ereignisse, Vorgänge oder Geschehnisse",
            }
        elif isinstance(labels, dict):
            self.labels = labels
        else:
            # Convert list to dict without descriptions
            self.labels = {label: label for label in labels}

        self.threshold = threshold
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        # Setup paths following new architecture
        self.config = get_config()

        # Model will be loaded on first use
        self.model = None

        logger.info(f"Initialized GLiNER2EntityExtractor for '{source_name}'")
        logger.info(f"Model: {model_name}")
        logger.info(f"Labels: {list(self.labels.keys())}")
        logger.info(f"Using label descriptions: {isinstance(labels, dict) or labels is None}")

    def _load_model(self):
        """Load GLiNER2 model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading GLiNER2 model: {self.model_name}")
            logger.info(f"Cache directory: {os.environ.get('HF_HOME')}")

            self.model = GLiNER2.from_pretrained(self.model_name)
            logger.info("GLiNER2 model loaded (optimized for CPU inference)")

    def _prepare_text(self, text: str) -> List[str]:
        """
        Prepare text for extraction by chunking if needed.

        Args:
            text: Raw text.

        Returns:
            List of text chunks. Single-item list if text is short enough.
        """
        # Use chunk_text utility for smart chunking at sentence boundaries
        chunks = chunk_text(
            text,
            max_length=self.max_text_length,
            split_margin=100,  # Look for split points within 100 chars of limit
        )
        return chunks

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
            id_column: Column containing unique identifiers (line_id, text_block_id, etc.).
            limit: Optional limit on number of rows to process.

        Returns:
            DataFrame with extracted entities (source_id, entity_text, entity_type, confidence, detection_count).
            Note: source_id contains values from the id_column (maintains foreign key relationship).
                  detection_count tracks how many times the same entity was detected (useful for chunked texts).
        """
        self._load_model()
        assert self.model is not None, "Model failed to load"

        logger.info(f"Extracting entities from {len(df)} rows")

        # Filter by text length
        df_filtered = df.filter(pl.col(text_column).str.len_chars() >= self.min_text_length)
        logger.info(f"Filtered to {len(df_filtered)} texts (min length: {self.min_text_length})")

        if limit:
            df_filtered = df_filtered.head(limit)
            logger.info(f"Limited to {limit} rows")

        # Prepare texts with chunking
        logger.info("Preparing texts and chunking long texts...")
        texts = []
        line_ids = []
        chunk_counts = []

        for row in df_filtered.iter_rows(named=True):
            chunks = self._prepare_text(row[text_column])
            chunk_counts.append(len(chunks))
            # Add all chunks with same ID
            for chunk in chunks:
                texts.append(chunk)
                line_ids.append(row[id_column])

        total_chunks = len(texts)
        chunked_texts = sum(1 for c in chunk_counts if c > 1)
        logger.info(f"Prepared {total_chunks} text chunks from {len(df_filtered)} rows")
        logger.info(f"  - {chunked_texts} texts were split into multiple chunks")

        # Extract entities in batches
        logger.info(f"Batch size: {self.batch_size}, Threshold: {self.threshold}")

        all_records = []

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Extracting entities (batched)",
        ):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = line_ids[i : i + self.batch_size]

            # Batch inference using GLiNER2's extract_entities
            for text, line_id in zip(batch_texts, batch_ids):
                result = self.model.extract_entities(text, self.labels)

                # GLiNER2 returns: {'entities': {'label': ['entity1', 'entity2', ...]}}
                for label, entities in result.get("entities", {}).items():
                    # Map label to standard type
                    entity_type = self.LABEL_MAPPING.get(label, label.lower())

                    for entity_text in entities:
                        all_records.append(
                            {
                                "source_id": line_id,
                                "entity_text": entity_text,
                                "entity_type": entity_type,
                                "confidence": self.threshold,  # GLiNER2 doesn't return scores in basic mode
                            }
                        )

        # Deduplicate entities per ID (same entity found in multiple chunks)
        logger.info(f"Raw entities extracted: {len(all_records)}")
        if all_records:
            results_df_raw = pl.DataFrame(all_records)
            # Deduplicate: keep highest confidence for each (source_id, entity_text, entity_type)
            # Count detections to track how many times same entity was found across chunks
            results_df = (
                results_df_raw.group_by(["source_id", "entity_text", "entity_type"])
                .agg(
                    [
                        pl.col("confidence").max().alias("confidence"),
                        pl.col("confidence").count().alias("detection_count"),
                    ]
                )
                .sort(["source_id", "entity_type", "confidence"], descending=[False, False, True])
            )
            logger.info(f"After deduplication: {len(results_df)} unique entities")
            all_records = results_df.to_dicts()
        else:
            logger.info("No entities extracted")

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
                    "confidence": pl.Float64,
                    "detection_count": pl.UInt32,
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
            id_column: Column containing source IDs (line_id, text_block_id, etc.).

        Returns:
            Dictionary with extraction statistics and output paths.
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("GLiNER2 Entity Extraction Pipeline")
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
            method_type="gliner2",
            model_name=self.model_name.replace("/", "_").replace(".", "_"),
            source=self.source_name,
            parameters={
                "model": self.model_name,  # Full model name
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "labels": (
                    list(self.labels.keys()) if isinstance(self.labels, dict) else self.labels
                ),
                "label_descriptions": isinstance(self.labels, dict),
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "chunking": "enabled",
                "text_column": text_column,
                "source_id_column": id_column,  # Track what source_id references
                "inference_device": "cpu",  # GLiNER2 optimized for CPU
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


def extract_entities_gliner2(
    source_name: str = "der_tag",
    model_name: str = "fastino/gliner2-base-0207",
    labels: Optional[Union[List[str], Dict[str, Union[str, Dict]]]] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    limit: Optional[int] = None,
) -> Dict:
    """
    Convenience function for GLiNER2-based entity extraction.

    Args:
        source_name: Source dataset name.
        model_name: GLiNER2 model from Hugging Face Hub.
        labels: Entity labels (list or dict with descriptions).
        threshold: Confidence threshold (0-1).
        batch_size: Batch size for processing.
        limit: Optional limit on rows (for testing).

    Returns:
        Extraction results dictionary.

    Example:
        ```python
        from newspaper_explorer.analyze.entities.gliner2_extraction import extract_entities_gliner2

        # With label descriptions for better accuracy
        results = extract_entities_gliner2(
            source_name="der_tag",
            labels={
                "Person": "Namen von Personen oder historischen Figuren",
                "Organisation": "Organisationen, Unternehmen oder Institutionen",
                "Ort": "Städte, Länder oder geografische Orte"
            },
            limit=100
        )

        # Simple list of labels
        results = extract_entities_gliner2(
            source_name="der_tag",
            labels=["Person", "Organisation", "Ort", "Ereignis"]
        )
        ```
    """
    extractor = GLiNER2EntityExtractor(
        source_name=source_name,
        model_name=model_name,
        labels=labels,
        threshold=threshold,
        batch_size=batch_size,
    )

    return extractor.extract_and_save(limit=limit)
