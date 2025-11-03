"""
GLiNER-based entity extraction for newspaper text.

Extracts named entities (persons, organizations, locations) using GLiNER model
with structured response and proper result storage following the new data
architecture pattern.
"""

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import polars as pl
import torch
from gliner import GLiNER
from tqdm import tqdm

from newspaper_explorer.analyze.query.engine import create_result_metadata
from newspaper_explorer.config.base import get_config, get_project_root
from newspaper_explorer.data.loading.loader import DataLoader
from newspaper_explorer.data.utils.text import chunk_text

logger = logging.getLogger(__name__)

# Set cache directory for Hugging Face models (GLiNER uses HF)
_project_root = get_project_root()
os.environ["HF_HOME"] = str(_project_root / ".cache" / "huggingface")

# Suppress SentencePiece tokenizer conversion warning
# The fast tokenizer works fine for our use case and is much faster
warnings.filterwarnings(
    "ignore",
    message=".*byte fallback.*",
    category=UserWarning,
    module="transformers.convert_slow_tokenizer",
)


# Available GLiNER models with metadata
GLINER_MODELS: Dict[str, Dict[str, Any]] = {
    "multi-v2.1": {
        "model_id": "urchade/gliner_multi-v2.1",
        "size": "~250MB",
        "max_tokens": 384,
        "description": "Multilingual model, balanced speed/quality",
        "default": True,
    },
    "xxl-v2.5": {
        "model_id": "gliner-community/gliner_xxl-v2.5",
        "size": "~1.2GB",
        "max_tokens": 512,
        "description": "Extra large model, highest quality",
        "default": False,
    },
    "small-v2.1": {
        "model_id": "urchade/gliner_small-v2.1",
        "size": "~150MB",
        "max_tokens": 384,
        "description": "Smaller model, faster inference",
        "default": False,
    },
    "large-v2.1": {
        "model_id": "urchade/gliner_large-v2.1",
        "size": "~650MB",
        "max_tokens": 384,
        "description": "Larger model, better accuracy",
        "default": False,
    },
}


def get_gliner_model_id(model_name: str) -> str:
    """
    Get full model ID from short name or full ID.

    Args:
        model_name: Short name (e.g., "xxl-v2.5") or full ID (e.g., "gliner-community/gliner_xxl-v2.5")

    Returns:
        Full model ID for Hugging Face Hub

    Raises:
        ValueError: If model name is not found
    """
    # If it looks like a full model ID (contains /), return as-is
    if "/" in model_name:
        return model_name

    # Look up short name
    if model_name in GLINER_MODELS:
        model_id: str = GLINER_MODELS[model_name]["model_id"]
        return model_id

    # Not found
    available = ", ".join(GLINER_MODELS.keys())
    raise ValueError(
        f"Unknown GLiNER model: {model_name}. "
        f"Available short names: {available}. "
        f"Or provide full model ID (e.g., 'urchade/gliner_multi-v2.1')"
    )


def _process_texts_on_gpu(
    device: str,
    texts: List[str],
    text_ids: List[str],
    text_indices: List[int],
    model_id: str,
    labels: List[str],
    threshold: float,
    batch_size: int,
    result_queue=None,
    barrier=None,
    gpu_id: Optional[int] = None,
) -> Optional[List[Tuple[int, List[Dict]]]]:
    """
    Worker function to process texts on a specific GPU device.

    Can be used in two modes:
    1. Direct call (single-GPU): result_queue=None, barrier=None - returns results
    2. Spawned process (multi-GPU): result_queue and barrier provided - sends to queue

    Args:
        device: Device string (e.g., "cuda:0", "cuda", "cpu")
        texts: List of text chunks to process
        text_ids: List of source IDs corresponding to texts
        text_indices: Original indices for reassembly
        model_id: HuggingFace GLiNER model identifier
        labels: Entity labels to extract
        threshold: Confidence threshold
        batch_size: Batch size for inference
        result_queue: Optional multiprocessing queue for results (multi-GPU mode)
        barrier: Optional multiprocessing barrier for synchronization (multi-GPU mode)
        gpu_id: Optional GPU ID for progress bar labeling (multi-GPU mode)

    Returns:
        List of (index, entities_for_text) tuples if result_queue is None, else None
    """
    # In multi-GPU mode, suppress output during model loading
    if gpu_id is not None:
        # Disable all HF Hub progress bars and messages
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
        # Set transformers verbosity to error level
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        # Suppress stderr during model loading to hide tqdm output
        import sys
        import contextlib

        @contextlib.contextmanager
        def suppress_output():
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr

        # Load model with suppressed output
        with suppress_output():
            model = GLiNER.from_pretrained(model_id)
    else:
        # Single GPU mode - load normally
        model = GLiNER.from_pretrained(model_id)

    model = model.to(device)

    # Set tokenizer max_length
    if hasattr(model, "data_processor") and hasattr(model.data_processor, "transformer_tokenizer"):
        max_tokens = model.config.max_len if hasattr(model, "config") else 384
        model.data_processor.transformer_tokenizer.model_max_length = max_tokens

    # Compile model for faster inference (PyTorch 2.0+)
    if device.startswith("cuda") and hasattr(torch, "compile"):
        try:
            logger.info(f"Compiling model on {device} with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info(f"Model compilation complete on {device}")
        except Exception as e:
            logger.warning(
                f"Could not compile model on {device}: {e}. Continuing without compilation."
            )

    # Wait for all GPUs to finish loading (multi-GPU mode only)
    if barrier is not None:
        barrier.wait()

    # Process in batches
    results = []

    # Configure progress bar based on mode
    if gpu_id is not None:
        # Multi-GPU mode: labeled progress bar
        pbar_desc = f"GPU {gpu_id}"
        pbar_position = gpu_id
    else:
        # Single-GPU mode: simple progress bar
        pbar_desc = "Extracting entities"
        pbar_position = 0

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=pbar_desc,
        position=pbar_position,
        leave=True,
    ):
        batch_texts = texts[i : i + batch_size]
        batch_ids = text_ids[i : i + batch_size]
        batch_indices = text_indices[i : i + batch_size]

        # Batch inference
        batch_entities = model.run(
            batch_texts,
            labels=labels,
            threshold=threshold,
            flat_ner=True,
            batch_size=len(batch_texts),
        )

        # Store results with original indices
        for idx, entities in zip(batch_indices, batch_entities):
            results.append((idx, entities))

    # Return or send to queue
    if result_queue is not None:
        result_queue.put(results)
        return None
    else:
        return results


class GLiNEREntityExtractor:
    """
    Extract named entities using GLiNER model.

    Uses the new data architecture:
    - Saves results as Parquet with source_id foreign keys
    - Creates metadata.json for reproducibility
    - Follows results/{source}/entities/{method_id}/ structure

    Note: source_id column contains IDs from input (line_id, text_block_id, etc.)
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
        model_name: str = "multi-v2.1",
        labels: Optional[List[str]] = None,
        threshold: float = 0.2,
        batch_size: int = 32,
        min_text_length: int = 10,
        max_text_length: int = 1000,
    ):
        """
        Initialize GLiNER entity extractor.

        Args:
            source_name: Source dataset name (e.g., "der_tag").
            model_name: GLiNER model name. Can be:
                       - Short name: "multi-v2.1", "xxl-v2.5", "small-v2.1", "large-v2.1"
                       - Full HF model ID: "urchade/gliner_multi-v2.1"
            labels: Entity labels to extract (default: Person, Organisation, Ereignis, Ort).
            threshold: Confidence threshold for entity extraction (0-1).
            batch_size: Batch size for processing (adjust based on GPU memory).
            min_text_length: Minimum text length to process (skip shorter texts).
            max_text_length: Maximum text length for chunking (~1800 chars = ~360 tokens for German).
        """
        self.source_name = source_name

        # Resolve model name to full model ID
        self.model_id = get_gliner_model_id(model_name)
        self.model_name = model_name  # Keep original name for metadata

        self.labels = labels or ["Person", "Organisation", "Ereignis", "Ort"]
        self.threshold = threshold
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        # Setup paths following new architecture
        self.config = get_config()

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
        num_gpus: int = 1,
    ) -> pl.DataFrame:
        """
        Extract entities from a Polars DataFrame.

        Args:
            df: DataFrame with text data.
            text_column: Column containing text content.
            id_column: Column containing unique identifiers (line_id, text_block_id, etc.).
            limit: Optional limit on number of rows to process.
            num_gpus: Number of GPUs to use (default: 1). Set to >1 for multi-GPU processing.

        Returns:
            DataFrame with extracted entities (source_id, entity_text, entity_type, confidence, detection_count).
            Note: source_id contains values from the id_column (maintains foreign key relationship).
                  detection_count tracks how many times the same entity was detected (useful for chunked texts).
        """
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

        # Auto-detect available GPUs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and num_gpus > 1:
            available_gpus = torch.cuda.device_count()
            if available_gpus < num_gpus:
                logger.warning(
                    f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                    f"Using {available_gpus} GPUs."
                )
                num_gpus = available_gpus
            logger.info(f"Using {num_gpus} GPUs for parallel processing")
        else:
            num_gpus = 1
            logger.info(f"Using device: {device}")

        # Extract entities in batches
        logger.info(f"Batch size: {self.batch_size}, Threshold: {self.threshold}")

        # Create indices for tracking
        text_indices = list(range(len(texts)))

        # Multi-GPU processing
        if num_gpus > 1:
            logger.info(f"Distributing {len(texts)} chunks across {num_gpus} GPUs")

            import multiprocessing as mp

            # Set start method to 'spawn' for CUDA compatibility
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                # Already set, ignore
                pass

            from multiprocessing import Process, Queue, Barrier

            # Create barrier for synchronization
            barrier = Barrier(num_gpus)

            # Split chunks across GPUs
            chunks_per_gpu = (len(texts) + num_gpus - 1) // num_gpus
            processes = []
            result_queue = Queue()

            for gpu_id in range(num_gpus):
                start_idx = gpu_id * chunks_per_gpu
                end_idx = min(start_idx + chunks_per_gpu, len(texts))

                if start_idx >= len(texts):
                    break

                gpu_texts = texts[start_idx:end_idx]
                gpu_ids = line_ids[start_idx:end_idx]
                gpu_indices = text_indices[start_idx:end_idx]

                logger.info(f"GPU {gpu_id}: will process {len(gpu_texts)} chunks")

                p = Process(
                    target=_process_texts_on_gpu,
                    kwargs={
                        "device": f"cuda:{gpu_id}",
                        "texts": gpu_texts,
                        "text_ids": gpu_ids,
                        "text_indices": gpu_indices,
                        "model_id": self.model_id,
                        "labels": self.labels,
                        "threshold": self.threshold,
                        "batch_size": self.batch_size,
                        "result_queue": result_queue,
                        "barrier": barrier,
                        "gpu_id": gpu_id,
                    },
                )
                p.start()
                processes.append(p)

            # Collect results from all GPUs
            logger.info("All GPUs are loading models and will start processing together...")
            logger.info("Waiting for GPU processes to complete...")
            processing_results = []
            for _ in range(len(processes)):
                processing_results.extend(result_queue.get())

            # Wait for all processes to finish
            for p in processes:
                p.join()

            logger.info("Multi-GPU processing complete")

        else:
            # Single GPU/CPU processing
            logger.info("Loading GLiNER model (this may take a while on first run)...")
            processing_results = _process_texts_on_gpu(
                device=device,
                texts=texts,
                text_ids=line_ids,
                text_indices=text_indices,
                model_id=self.model_id,
                labels=self.labels,
                threshold=self.threshold,
                batch_size=self.batch_size,
                result_queue=None,
                barrier=None,
                gpu_id=None,
            )

        # Convert results to records
        all_records = []
        for idx, entities in processing_results:
            line_id = line_ids[idx]
            for entity in entities:
                # Map label to standard type
                entity_type = self.LABEL_MAPPING.get(entity["label"], entity["label"].lower())

                all_records.append(
                    {
                        "source_id": line_id,
                        "entity_text": entity["text"],
                        "entity_type": entity_type,
                        "confidence": float(entity.get("score", 0.0)),
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
        num_gpus: int = 1,
    ) -> Dict:
        """
        Complete extraction pipeline: load, extract, save with metadata.

        Args:
            source_parquet: Path to source parquet file. If None, loads from DataLoader.
            limit: Optional limit on rows to process (for testing).
            text_column: Column containing text.
            id_column: Column containing source IDs (line_id, text_block_id, etc.).
            num_gpus: Number of GPUs to use for parallel processing (default: 1).

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
            df, text_column=text_column, id_column=id_column, limit=limit, num_gpus=num_gpus
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
                "model": self.model_name,  # Full model name
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "labels": self.labels,
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "chunking": "enabled",
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
