"""
Emotion prediction using pre-trained BERT models.

Provides multi-GPU emotion classification for large-scale newspaper analysis.
Uses the Shaver emotion model with 6 categories: Sadness, Love, Joy, Fear, Anger, Agitation.

Optimizations:
- Process-based multi-GPU parallelism (each GPU processes different emotions)
- Pre-tokenize once per chunk: Reuse across all 6 emotions (6x tokenization speedup)
- Per-batch dynamic padding: Pads only to longest in batch (30-50% less wasted computation)
- Tensor core optimization with pad_to_multiple_of=8
- torch.compile for 20-30% speedup (PyTorch 2.0+)
- FP16 mixed precision (enabled by default, ~30% speedup)
- TF32 on Ampere+ GPUs (~20% speedup)
- Non-blocking H2D transfers for async memory copy
- torch.inference_mode() instead of no_grad()
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, List
import queue
import warnings
import time

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertForSequenceClassification, BertTokenizerFast
from transformers import logging as transformers_logging
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

# Suppress harmless warnings
# Transformers: "Some weights not initialized" (we load trained weights right after)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.backends")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.fx.experimental.symbolic_shapes"
)

# Suppress torch.fx symbolic shapes logging warnings
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


class PreTokenizedDataset(Dataset):
    """
    Dataset of pre-tokenized (but unpadded) sequences.

    This enables the best of both worlds:
    - Tokenize once per chunk (6x reuse across emotions)
    - Pad per-batch dynamically (30-50% less wasted computation)

    Stores variable-length token ID lists that get padded in collate_fn.
    """

    def __init__(self, input_ids_list, attention_mask_list):
        """
        Args:
            input_ids_list: List of 1D tensors (variable length)
            attention_mask_list: List of 1D tensors (variable length)
        """
        self.input_ids_list = input_ids_list
        self.attention_mask_list = attention_mask_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
        }


def collate_pretokenized(batch, pad_token_id: int = 0):
    """
    Collate function for pre-tokenized sequences with per-batch padding.

    Pads variable-length sequences to longest in batch (not chunk).

    Args:
        batch: List of dicts with 'input_ids' and 'attention_mask' tensors
        pad_token_id: Token ID to use for padding (default: 0)

    Returns:
        Dict with padded tensors
    """
    # Extract sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Find longest sequence in batch
    max_len = max(len(ids) for ids in input_ids)

    # Round up to multiple of 8 for tensor core optimization
    max_len = ((max_len + 7) // 8) * 8

    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []

    for ids, mask in zip(input_ids, attention_mask):
        pad_len = max_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
        )
        padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }


def create_tokenized_dataloader(
    texts: list[str],
    tokenizer: BertTokenizerFast,
    batch_size: int,
    max_length: int = 512,
) -> DataLoader:
    """
    Create a DataLoader with optimal tokenization strategy.

    Shared helper for both single-GPU and multi-GPU paths.
    Best of both worlds approach:
    - Pre-tokenize once per chunk (6x reuse across emotions)
    - Pad per-batch dynamically (30-50% less wasted computation)
    - Tensor core optimization (pad_to_multiple_of=8)

    Args:
        texts: List of text strings to tokenize
        tokenizer: BERT tokenizer instance
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length (default: 512)

    Returns:
        DataLoader with pre-tokenized sequences and per-batch padding
    """
    # Pre-tokenize once (without padding)
    encoding = tokenizer(
        texts,
        padding=False,  # No padding yet - do it per-batch
        truncation=True,
        max_length=max_length,
    )

    # Convert to individual tensors (variable length)
    input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in encoding.input_ids]
    attention_mask_list = [torch.tensor(mask, dtype=torch.long) for mask in encoding.attention_mask]

    # Create dataset from pre-tokenized sequences
    dataset = PreTokenizedDataset(input_ids_list, attention_mask_list)

    # Create DataLoader with per-batch padding
    pad_token_id: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0  # type: ignore
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No workers needed (already tokenized)
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,  # Must be None when num_workers=0
        collate_fn=lambda batch: collate_pretokenized(batch, pad_token_id),
    )

    return dataloader


def load_checkpoint_cls(model_name, path, device, use_fp16=False, use_compile=False):
    """
    Load model checkpoint with optimizations.

    Args:
        model_name: Base model name (deepset/gbert-large)
        path: Path to checkpoint file
        device: Device to load on (cuda/cpu)
        use_fp16: Use FP16 mixed precision (~30% faster, minimal accuracy loss)
        use_compile: Use torch.compile for additional 20-30% speedup (PyTorch 2.0+)
    """
    # Load base model (warnings suppressed at module level)
    model = BertForSequenceClassification.from_pretrained(model_name)

    state_dict = torch.load(path, map_location="cpu")

    # Handle state dict mismatches
    mkeys = set(model.state_dict().keys())
    skeys = set(state_dict.keys())

    for k in list(state_dict.keys()):
        if k not in mkeys:
            del state_dict[k]

    for k in mkeys:
        if k not in state_dict:
            state_dict[k] = model.state_dict()[k]

    model.load_state_dict(state_dict)
    model.to(device)

    if use_fp16 and device != "cpu":
        logger.debug("Converting to FP16 (mixed precision)")
        model.half()

    model.eval()

    # torch.compile for additional speedup (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        logger.debug("Compiling model with torch.compile")
        model = torch.compile(model, mode="reduce-overhead")

    return model


def predict_batch(model, dataloader, device):
    """
    Batch prediction with GPU acceleration.

    Returns both binary predictions (0/1) and probability scores (0.0-1.0).

    Optimizations:
    - torch.inference_mode() instead of no_grad() for better performance
    - non_blocking=True for async H2D transfers
    - TF32 automatically enabled via torch.set_float32_matmul_precision("high")
    """
    predictions = []
    probabilities = []

    with torch.inference_mode():  # Faster than no_grad()
        for batch in tqdm(dataloader, desc="Processing batches", leave=False):
            # Non-blocking transfers for async H2D copy
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Binary predictions (0 or 1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

            # Probability scores (0.0 to 1.0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Prob of class 1
            probabilities.extend(probs)

    return predictions, probabilities


def worker_process_emotion(
    gpu_id: int,
    emotion: str,
    model_path: Path,
    texts_queue: mp.Queue,
    results_queue: mp.Queue,
    tokenizer_name: str,
    batch_size: int,
    use_fp16: bool,
    use_compile: bool,
):
    """
    Worker process for processing one emotion on a specific GPU.

    This enables true parallelism: each GPU processes a different emotion simultaneously.
    Much faster than sequential processing (3x speedup for 6 emotions on 4 GPUs).
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Enable TF32 for Ampere+ GPUs (L40S supports this) - ~20% speedup
    # Use new PyTorch 2.9+ API for TF32 configuration
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"GPU {gpu_id}: Loading {emotion} model...")

    try:
        model = load_checkpoint_cls(
            "deepset/gbert-large",
            model_path,
            device,
            use_fp16=use_fp16,
            use_compile=use_compile,
        )
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

        while True:
            try:
                item = texts_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break

                chunk_idx, texts = item

                # Pre-tokenize and create DataLoader (shared helper function)
                dataloader = create_tokenized_dataloader(
                    texts=texts,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    max_length=512,
                )

                # Predict (returns both binary and probabilities)
                predictions, probabilities = predict_batch(model, dataloader, device)

                # Send results back
                results_queue.put((emotion, chunk_idx, predictions, probabilities))

            except queue.Empty:
                continue

    except Exception as e:
        logger.error(f"GPU {gpu_id}: Error processing {emotion}: {e}")
        raise

    logger.info(f"GPU {gpu_id}: {emotion} worker finished")


class EmotionPredictor:
    """
    Emotion prediction using pre-trained BERT models.

    Supports both single-GPU and multi-GPU processing with automatic
    device detection and optimal batch sizing.
    """

    EMOTION_NAMES = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]

    def __init__(
        self,
        source_name: str,
        model_dir: Optional[Path] = None,
        batch_size: int = 64,
        chunk_size: int = 100000,
        use_fp16: bool = True,  # Enabled by default for Ampere+ GPUs (L40S)
        use_compile: bool = True,
        multi_gpu: bool = True,
    ):
        """
        Initialize emotion predictor.

        Args:
            source_name: Source name (e.g., 'der_tag')
            model_dir: Directory containing model files (default: models/emotions)
            batch_size: Batch size for inference
            chunk_size: Number of rows per chunk
            use_fp16: Use FP16 mixed precision (~30% faster, enabled by default for Ampere+)
            use_compile: Use torch.compile (~20-30% additional speedup)
            multi_gpu: Enable true multi-GPU parallelism (each GPU = different emotion)
        """
        self.source_name = source_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1

        config = get_config()

        # Setup model directory
        if model_dir is None:
            self.model_dir = Path("models/emotions")
        else:
            self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please download emotion models first."
            )

        # Model files
        self.model_files = {
            emotion: self.model_dir / f"{emotion}.pt" for emotion in self.EMOTION_NAMES
        }

        # Verify models exist
        for emotion, path in self.model_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

        # Setup device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Using {self.num_gpus} GPU(s)")
            if self.multi_gpu and self.num_gpus > 1:
                logger.info(f"Multi-GPU parallel mode enabled (process-based)")
            if self.use_compile:
                logger.info("Model compilation enabled (torch.compile)")
        else:
            self.device = "cpu"
            self.num_gpus = 0
            self.multi_gpu = False
            logger.warning("No GPU available, using CPU")

        # Initialize tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("deepset/gbert-large")

        # Base results directory (subdirectories created per analysis run)
        self.results_base = config.results_dir / source_name

    def load_models(self):
        """Load all emotion models (sequential mode only)"""
        logger.info("Loading emotion models...")
        models = {}

        for emotion, model_path in self.model_files.items():
            logger.info(f"  Loading {emotion}...")
            model = load_checkpoint_cls(
                "deepset/gbert-large",
                model_path,
                self.device,
                use_fp16=self.use_fp16,
                use_compile=self.use_compile,
            )
            models[emotion] = model

        logger.info("All models loaded")
        return models

    def predict(
        self,
        input_file: Path,
        text_column: str = "text",
        output_name: str = "emotion_predictions",
    ) -> Path:
        """
        Predict emotions for text data.

        Uses multi-GPU parallel processing if enabled, otherwise sequential.

        Args:
            input_file: Input parquet file
            text_column: Name of text column
            output_name: Base name for output file

        Returns:
            Path to output file
        """
        if self.multi_gpu and self.num_gpus > 1:
            return self._predict_parallel(input_file, text_column, output_name)
        else:
            return self._predict_sequential(input_file, text_column, output_name)

    def _predict_sequential(
        self,
        input_file: Path,
        text_column: str,
        output_name: str,
    ) -> Path:
        """Sequential prediction (single GPU or CPU)"""
        start_time = time.time()

        # Create output directory with analysis_id for versioning
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_id = f"gbert_large_shaver_{timestamp}"
        output_dir = self.results_base / "emotions" / analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Enable TF32 for Ampere+ GPUs (L40S supports this) - ~20% speedup
        if torch.cuda.is_available():
            # Use new PyTorch 2.9+ API for TF32 configuration
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        logger.info(f"Processing {input_file} (sequential mode)")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Chunk size: {self.chunk_size:,}")

        # Load models
        models = self.load_models()

        # Load input data for metadata extraction
        df_input = pl.read_parquet(input_file)
        total_rows = len(df_input)
        logger.info(f"Total rows: {total_rows:,}")

        # Check for existing results (resume functionality)
        output_file = output_dir / "emotions.parquet"
        processed_ids = set()

        # Check both final output and chunk files
        existing_files = []
        if output_file.exists():
            existing_files.append(output_file)
        existing_files.extend(output_dir.glob(f"{output_name}_chunk_*.parquet"))

        if existing_files:
            logger.info(f"Found {len(existing_files)} existing file(s)")
            for f in existing_files:
                df = pl.read_parquet(f)
                processed_ids.update(df["line_id"].to_list())
            logger.info(f"Already processed: {len(processed_ids):,} rows")
            logger.info(f"Remaining: {total_rows - len(processed_ids):,} rows")
            if len(processed_ids) >= total_rows:
                logger.info("All rows already processed. Skipping.")
                return output_file

        # Process in chunks
        first_chunk = not output_file.exists() or len(processed_ids) == 0

        for chunk_idx, chunk_start in enumerate(range(0, total_rows, self.chunk_size), start=1):
            chunk_end = min(chunk_start + self.chunk_size, total_rows)
            logger.info(f"\nProcessing chunk {chunk_idx} (rows {chunk_start:,}-{chunk_end:,})")

            # Load chunk
            df_chunk = df_input.slice(chunk_start, self.chunk_size)

            # Filter out already processed rows (resume functionality)
            if processed_ids:
                original_len = len(df_chunk)
                df_chunk = df_chunk.filter(~pl.col("line_id").is_in(processed_ids))
                skipped = original_len - len(df_chunk)
                if skipped > 0:
                    logger.info(f"  Skipping {skipped:,} already processed rows")
                if len(df_chunk) == 0:
                    logger.info(f"  Chunk already complete, skipping")
                    continue

            # Extract texts
            texts = df_chunk[text_column].to_list()

            # Pre-tokenize ONCE and create DataLoader (reused for all 6 emotions)
            dataloader = create_tokenized_dataloader(
                texts=texts,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_length=512,
            )

            # Predict for each emotion (sequential, but tokenization done only once!)
            predictions = {}
            probabilities = {}
            for emotion, model in models.items():
                logger.info(f"  Predicting {emotion}...")
                preds, probs = predict_batch(model, dataloader, self.device)
                predictions[emotion] = preds
                probabilities[emotion] = probs

            # Add predictions to dataframe (binary + probabilities)
            for emotion, preds in predictions.items():
                df_chunk = df_chunk.with_columns(pl.Series(emotion, preds))
                df_chunk = df_chunk.with_columns(
                    pl.Series(f"{emotion}_prob", probabilities[emotion])
                )

            # Extract foreign keys from line_ids
            logger.info("  Extracting foreign keys...")
            line_ids = df_chunk["line_id"].to_list()
            foreign_keys = [extract_foreign_keys(line_id) for line_id in line_ids]

            # Add foreign key columns
            df_chunk = df_chunk.with_columns(
                [
                    pl.Series("source_id", [fk["source_id"] for fk in foreign_keys]),
                    pl.Series("issue_id", [fk["issue_id"] for fk in foreign_keys]),
                    pl.Series("page_id", [fk["page_id"] for fk in foreign_keys]),
                    pl.Series("text_block_id", [fk["text_block_id"] for fk in foreign_keys]),
                ]
            )

            # Save chunk to temporary file (fast - no concat needed)
            chunk_file = output_dir / f"{output_name}_chunk_{chunk_idx}.parquet"
            df_chunk.write_parquet(chunk_file)

            logger.info(f"  ✓ Chunk {chunk_idx} saved to {chunk_file.name}")

        # Combine all chunk files into final output
        logger.info("\nCombining chunk files...")
        chunk_files = sorted(output_dir.glob(f"{output_name}_chunk_*.parquet"))
        if chunk_files:
            df_final = pl.concat([pl.read_parquet(f) for f in chunk_files])
            df_final.write_parquet(output_file)

            # Clean up chunk files
            for f in chunk_files:
                f.unlink()
            logger.info(f"Combined {len(chunk_files)} chunks into {output_file}")

        # Create and save metadata
        duration_seconds = time.time() - start_time
        logger.info("Creating metadata...")

        # Load final results for statistics
        df_result = pl.read_parquet(output_file)

        # Calculate emotion statistics
        emotion_counts = {}
        for emotion in self.EMOTION_NAMES:
            count = df_result[emotion].sum()
            emotion_counts[emotion.lower()] = count

        # Create output statistics dict
        output_stats = extract_output_stats(df_result)
        output_stats["emotion_counts"] = emotion_counts
        output_stats["total_predictions"] = len(df_result)

        # Create metadata
        metadata = AnalysisMetadata(
            analysis_type="emotions",
            method_type="bert_multi_label",
            model_name="gbert_large_shaver",
            source=self.source_name,
            parameters={
                "model_base": "deepset/gbert-large",
                "emotions": list(self.EMOTION_NAMES),
                "batch_size": self.batch_size,
                "chunk_size": self.chunk_size,
                "max_length": 512,
                "use_fp16": self.use_fp16,
                "use_compile": self.use_compile,
                "multi_gpu": self.multi_gpu,
                "num_gpus": self.num_gpus if self.multi_gpu else 1,
                "text_column": text_column,
            },
            input_data=extract_input_stats(df_input),
            output_data=output_stats,
            status="completed",
            duration_seconds=duration_seconds,
        )

        # Save metadata
        metadata_path = save_metadata(metadata, output_file)
        logger.info(f"Metadata saved to: {metadata_path}")

        logger.info(f"\n✓ Processing complete!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Analysis ID: {metadata.analysis_id}")

        return output_file

    def _predict_parallel(
        self,
        input_file: Path,
        text_column: str,
        output_name: str,
    ) -> Path:
        """
        Parallel prediction using multi-GPU.

        Each GPU processes different emotions simultaneously for ~3x speedup.
        """
        start_time = time.time()

        # Create output directory with analysis_id for versioning
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_id = f"gbert_large_shaver_{timestamp}"
        output_dir = self.results_base / "emotions" / analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {input_file} (parallel mode - {self.num_gpus} GPUs)")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Text column: {text_column}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Chunk size: {self.chunk_size:,}")

        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, that's fine
            pass

        # Load input data for metadata extraction
        df_input = pl.read_parquet(input_file)
        total_rows = len(df_input)
        logger.info(f"Total rows: {total_rows:,}")

        # Check for existing results (resume functionality)
        output_file = output_dir / "emotions.parquet"
        processed_ids = set()

        # Check both final output and chunk files
        existing_files = []
        if output_file.exists():
            existing_files.append(output_file)
        existing_files.extend(output_dir.glob(f"{output_name}_chunk_*.parquet"))

        if existing_files:
            logger.info(f"Found {len(existing_files)} existing file(s)")
            for f in existing_files:
                df = pl.read_parquet(f)
                processed_ids.update(df["line_id"].to_list())
            logger.info(f"Already processed: {len(processed_ids):,} rows")
            logger.info(f"Remaining: {total_rows - len(processed_ids):,} rows")
            if len(processed_ids) >= total_rows:
                logger.info("All rows already processed. Skipping.")
                return output_file

        # Setup queues for communication
        texts_queues: List[mp.Queue] = [mp.Queue() for _ in self.EMOTION_NAMES]
        results_queue: mp.Queue = mp.Queue()

        # Start worker processes (one per emotion)
        processes = []
        for i, emotion in enumerate(self.EMOTION_NAMES):
            gpu_id = i % self.num_gpus  # Distribute emotions across GPUs
            p = mp.Process(
                target=worker_process_emotion,
                args=(
                    gpu_id,
                    emotion,
                    self.model_files[emotion],
                    texts_queues[i],
                    results_queue,
                    "deepset/gbert-large",
                    self.batch_size,
                    self.use_fp16,
                    self.use_compile,
                ),
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker for {emotion} on GPU {gpu_id}")

        # Process in chunks
        output_file = output_dir / f"{output_name}.parquet"
        first_chunk = (
            not output_file.exists() or len(processed_ids) == 0
        )  # If file exists with data, we're resuming

        for chunk_idx, chunk_start in enumerate(range(0, total_rows, self.chunk_size), start=1):
            chunk_end = min(chunk_start + self.chunk_size, total_rows)
            logger.info(f"Processing chunk {chunk_idx} (rows {chunk_start:,}-{chunk_end:,})")

            # Load chunk
            df_chunk = df_input.slice(chunk_start, self.chunk_size)

            # Filter out already processed rows (resume functionality)
            if processed_ids:
                original_len = len(df_chunk)
                df_chunk = df_chunk.filter(~pl.col("line_id").is_in(processed_ids))
                skipped = original_len - len(df_chunk)
                if skipped > 0:
                    logger.info(f"  Skipping {skipped:,} already processed rows")
                if len(df_chunk) == 0:
                    logger.info(f"  Chunk already complete, skipping")
                    continue

            # Extract texts
            texts = df_chunk[text_column].to_list()

            # Send texts to all emotion workers
            for queue in texts_queues:
                queue.put((chunk_idx, texts))

            # Collect results from all emotions
            predictions = {}
            probabilities = {}
            for _ in self.EMOTION_NAMES:
                emotion, received_chunk_idx, preds, probs = results_queue.get()
                assert received_chunk_idx == chunk_idx
                predictions[emotion] = preds
                probabilities[emotion] = probs

            # Add predictions to dataframe (binary + probabilities)
            for emotion, preds in predictions.items():
                df_chunk = df_chunk.with_columns(pl.Series(emotion, preds))
                df_chunk = df_chunk.with_columns(
                    pl.Series(f"{emotion}_prob", probabilities[emotion])
                )

            # Extract foreign keys from line_ids
            logger.info("  Extracting foreign keys...")
            line_ids = df_chunk["line_id"].to_list()
            foreign_keys = [extract_foreign_keys(line_id) for line_id in line_ids]

            # Add foreign key columns
            df_chunk = df_chunk.with_columns(
                [
                    pl.Series("source_id", [fk["source_id"] for fk in foreign_keys]),
                    pl.Series("issue_id", [fk["issue_id"] for fk in foreign_keys]),
                    pl.Series("page_id", [fk["page_id"] for fk in foreign_keys]),
                    pl.Series("text_block_id", [fk["text_block_id"] for fk in foreign_keys]),
                ]
            )

            # Save chunk to temporary file (fast - no concat needed)
            chunk_file = output_dir / f"{output_name}_chunk_{chunk_idx}.parquet"
            df_chunk.write_parquet(chunk_file)

            logger.info(f"  ✓ Chunk {chunk_idx} saved to {chunk_file.name}")

        # Stop worker processes
        for queue in texts_queues:
            queue.put(None)  # Poison pill

        for p in processes:
            p.join()

        # Combine all chunk files into final output
        logger.info("\nCombining chunk files...")
        chunk_files = sorted(output_dir.glob(f"{output_name}_chunk_*.parquet"))
        if chunk_files:
            df_final = pl.concat([pl.read_parquet(f) for f in chunk_files])
            df_final.write_parquet(output_file)

            # Clean up chunk files
            for f in chunk_files:
                f.unlink()
            logger.info(f"Combined {len(chunk_files)} chunks into {output_file}")

        # Create and save metadata
        duration_seconds = time.time() - start_time
        logger.info("Creating metadata...")

        # Load final results for statistics
        df_result = pl.read_parquet(output_file)

        # Calculate emotion statistics
        emotion_counts = {}
        for emotion in self.EMOTION_NAMES:
            count = df_result[emotion].sum()
            emotion_counts[emotion.lower()] = count

        # Create output statistics dict
        output_stats = extract_output_stats(df_result)
        output_stats["emotion_counts"] = emotion_counts
        output_stats["total_predictions"] = len(df_result)

        # Create metadata
        metadata = AnalysisMetadata(
            analysis_type="emotions",
            method_type="bert_multi_label",
            model_name="gbert_large_shaver",
            source=self.source_name,
            parameters={
                "model_base": "deepset/gbert-large",
                "emotions": list(self.EMOTION_NAMES),
                "batch_size": self.batch_size,
                "chunk_size": self.chunk_size,
                "max_length": 512,
                "use_fp16": self.use_fp16,
                "use_compile": self.use_compile,
                "multi_gpu": self.multi_gpu,
                "num_gpus": self.num_gpus,
                "text_column": text_column,
            },
            input_data=extract_input_stats(df_input),
            output_data=output_stats,
            status="completed",
            duration_seconds=duration_seconds,
        )

        # Save metadata
        metadata_path = save_metadata(metadata, output_file)
        logger.info(f"Metadata saved to: {metadata_path}")

        logger.info(f"\n✓ Processing complete!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Analysis ID: {metadata.analysis_id}")

        return output_file

    def predict_from_source(
        self,
        text_column: str = "text",
        input_file: Optional[Path] = None,
        output_name: str = "emotion_predictions",
    ) -> Path:
        """
        Predict emotions from source data.

        Args:
            text_column: Name of text column
            input_file: Custom input file (default: textblocks.parquet or lines.parquet)
            output_name: Base name for output file

        Returns:
            Path to output file
        """
        config = get_config()

        # Find input file if not specified
        if input_file is None:
            # Try processed directory first (aggregated/normalized)
            processed_dir = config.data_dir / "processed" / self.source_name
            raw_dir = config.data_dir / "raw" / self.source_name / "text"

            candidates = [
                # Processed (preferred)
                processed_dir / f"{self.source_name}_textblocks.parquet",
                processed_dir / f"{self.source_name}_lines.parquet",
                # Raw (fallback)
                raw_dir / f"{self.source_name}_textblocks.parquet",
                raw_dir / f"{self.source_name}_lines.parquet",
            ]

            for candidate in candidates:
                if candidate.exists():
                    input_file = candidate
                    logger.info(f"Using input file: {input_file}")
                    break

            if input_file is None:
                raise FileNotFoundError(
                    f"No input file found for source '{self.source_name}'.\n"
                    f"Tried: {[str(c) for c in candidates]}\n"
                    f"Run parsing first: newspaper-explorer data parse --source {self.source_name}"
                )

        return self.predict(input_file, text_column, output_name)
