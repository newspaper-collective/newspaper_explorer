"""
Entity extraction for newspaper text using GLiNER.
Extracts named entities (persons, organizations, events, locations) from text.

NOTE: Current implementation includes inline text normalization during extraction.
This is a temporary solution for prototyping. Future design should use preprocessed
datasets where normalization happens once during preprocessing, not during analysis.
See __scrap/PREPROCESSING_DESIGN.md for future architecture.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import torch
from gliner import GLiNER
from tqdm import tqdm

from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.text import normalize_german_text
from newspaper_explorer.utils.sources import load_source_config

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract named entities from newspaper text using GLiNER."""

    # Default entity labels for German historical newspapers
    DEFAULT_LABELS = ["Person", "Organisation", "Ereignis", "Ort"]

    def __init__(
        self,
        source_name: str,
        model_name: str = "urchade/gliner_multi-v2.1",
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        batch_size: int = 32,
        min_text_length: int = 100,
        max_text_length: int = 500,
    ):
        """
        Initialize entity extractor.

        Args:
            source_name: Name of the source (e.g., 'der_tag')
            model_name: GLiNER model from Hugging Face Hub
            labels: Entity labels to extract (default: Person, Organisation, Ereignis, Ort)
            threshold: Confidence threshold for entity extraction (0-1)
            batch_size: Batch size for processing (adjust based on GPU memory)
            min_text_length: Minimum text length to process (skip shorter texts)
            max_text_length: Maximum text length to use (truncate longer texts)
        """
        self.source_name = source_name
        self.model_name = model_name
        self.labels = labels or self.DEFAULT_LABELS
        self.threshold = threshold
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        # Load source configuration
        self.config = load_source_config(source_name)
        config = get_config()
        self.dataset_name = self.config["dataset_name"]

        # Setup paths: results/{source}/entities/
        self.output_dir = config.results_dir / self.dataset_name / "entities"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized EntityExtractor for '{source_name}'")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Labels: {self.labels}")

        # Model will be loaded on first use
        self.model = None
        self.device = None

    def _load_model(self):
        """Load GLiNER model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self.model = GLiNER.from_pretrained(self.model_name)

            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")

    def prepare_texts(
        self,
        df: pl.DataFrame,
        text_column: str = "text",
        normalize: bool = True,
    ) -> pl.DataFrame:
        """
        Prepare texts for entity extraction.

        Filters texts by length and optionally normalizes them.

        Args:
            df: Polars DataFrame with text data
            text_column: Name of column containing text
            normalize: If True, normalize text using normalize_german_text()

        Returns:
            Filtered DataFrame with prepared texts
        """
        logger.info(f"Preparing {len(df)} texts for extraction")

        # Filter by length
        df_filtered = df.filter(pl.col(text_column).str.len_chars() >= self.min_text_length)
        logger.info(f"Filtered to {len(df_filtered)} texts (min length: {self.min_text_length})")

        if normalize:
            logger.info("Normalizing texts...")
            # Apply normalization to each text
            df_filtered = df_filtered.with_columns(
                pl.col(text_column)
                .map_elements(normalize_german_text, return_dtype=pl.String)
                .alias("normalized_text")
            )
            text_column = "normalized_text"

        # Truncate to max length
        df_filtered = df_filtered.with_columns(
            pl.col(text_column).str.slice(0, self.max_text_length).alias("prepared_text")
        )

        return df_filtered

    def extract_entities(
        self,
        df: pl.DataFrame,
        text_column: str = "prepared_text",
        id_column: str = "text_block_id",
    ) -> pl.DataFrame:
        """
        Extract entities from texts using GLiNER.

        Args:
            df: DataFrame with prepared texts
            text_column: Column containing text to process
            id_column: Column to use as identifier

        Returns:
            DataFrame with extracted entities
        """
        self._load_model()

        logger.info(f"Extracting entities from {len(df)} texts")
        logger.info(f"Batch size: {self.batch_size}, Threshold: {self.threshold}")

        # Prepare texts and IDs
        texts_to_process = df[text_column].to_list()
        ids_to_process = df[id_column].to_list()

        # Results storage
        all_ids = []
        all_entities = []

        # Process in batches
        for i in tqdm(
            range(0, len(texts_to_process), self.batch_size),
            desc="Entity extraction (batched)",
        ):
            batch_texts = texts_to_process[i : i + self.batch_size]
            batch_ids = ids_to_process[i : i + self.batch_size]

            # Batch inference (much faster!)
            batch_entities = self.model.batch_predict_entities(
                batch_texts, self.labels, threshold=self.threshold
            )

            # Store results
            for text_id, entities in zip(batch_ids, batch_entities):
                for entity in entities:
                    all_ids.append(text_id)
                    all_entities.append(
                        {
                            "text": entity["text"],
                            "label": entity["label"],
                            "score": entity.get("score", 0.0),
                        }
                    )

        logger.info(f"Extracted {len(all_entities)} entities")

        # Create results DataFrame
        results_df = pl.DataFrame({"id": all_ids, "entity": all_entities})

        return results_df

    def serialize_entities(self, entities_df: pl.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """
        Serialize entities into a structured format.

        Groups entities by ID and label, removing duplicates.

        Args:
            entities_df: DataFrame with id and entity columns

        Returns:
            Dictionary mapping IDs to entity dictionaries
            Format: {id: {label: [entity_texts]}}
        """
        logger.info("Serializing entities...")

        result = {}

        # Convert to pandas for easier processing
        df_pandas = entities_df.to_pandas()

        # Group by ID
        grouped = df_pandas.groupby("id")

        for text_id, group in tqdm(grouped, desc="Serializing"):
            # Initialize empty lists for each label
            entity_dict = {label: [] for label in self.labels}

            # Process each entity
            for _, row in group.iterrows():
                entity = row["entity"]
                label = entity["label"]
                text = entity["text"].lower()  # Lowercase for deduplication

                # Add if not duplicate and label is valid
                if label in entity_dict and text not in entity_dict[label]:
                    entity_dict[label].append(text)

            result[text_id] = entity_dict

        logger.info(f"Serialized entities for {len(result)} texts")

        return result

    def save_results(
        self,
        entities_df: pl.DataFrame,
        serialized: Optional[Dict] = None,
        format: str = "parquet",
    ):
        """
        Save extraction results.

        Args:
            entities_df: DataFrame with raw entity extractions
            serialized: Optional pre-computed serialized entities
            format: Output format ('parquet', 'json', or 'both')

        Saves:
            - results/{source}/entities/entities_raw.parquet - Raw extractions
            - results/{source}/entities/entities_grouped.json - Grouped by ID and label
        """
        logger.info(f"Saving results to {self.output_dir}")

        # Save raw extractions
        if format in ["parquet", "both"]:
            raw_path = self.output_dir / "entities_raw.parquet"
            entities_df.write_parquet(raw_path, compression="zstd")
            logger.info(f"Saved raw entities to {raw_path}")

        # Save serialized/grouped entities
        if format in ["json", "both"]:
            if serialized is None:
                serialized = self.serialize_entities(entities_df)

            json_path = self.output_dir / "entities_grouped.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved grouped entities to {json_path}")

    def extract_and_save(
        self,
        input_path: Union[str, Path],
        text_column: str = "text",
        id_column: str = "text_block_id",
        normalize: bool = True,
        output_format: str = "both",
    ) -> Dict[str, pl.DataFrame]:
        """
        Complete pipeline: load, prepare, extract, serialize, and save.

        Args:
            input_path: Path to input parquet file (textblocks or sentences)
            text_column: Column containing text
            id_column: Column to use as identifier
            normalize: Whether to normalize text before extraction
            output_format: 'parquet', 'json', or 'both'

        Returns:
            Dictionary with 'entities' DataFrame and optionally 'serialized' dict
        """
        logger.info("=" * 60)
        logger.info("Entity Extraction Pipeline")
        logger.info("=" * 60)

        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pl.read_parquet(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Prepare texts
        df_prepared = self.prepare_texts(df, text_column=text_column, normalize=normalize)

        # Extract entities
        entities_df = self.extract_entities(
            df_prepared, text_column="prepared_text", id_column=id_column
        )

        # Serialize
        serialized = self.serialize_entities(entities_df)

        # Save
        self.save_results(entities_df, serialized=serialized, format=output_format)

        logger.info("=" * 60)
        logger.info("Extraction Complete!")
        logger.info(f"Total entities: {len(entities_df)}")
        logger.info(f"Unique texts: {len(serialized)}")
        logger.info("=" * 60)

        return {"entities": entities_df, "serialized": serialized}
