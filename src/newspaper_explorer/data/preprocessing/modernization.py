"""
Historical spelling modernization methods.

Provides heavy/external normalization for historical German texts:
- Transnormer: Neural transformer-based spelling modernization (local GPU)
- DTA-CAB: Web API-based spelling modernization (Deutsches Textarchiv)

These operations are expensive (require external models/APIs) and are separated
from the lightweight rule-based functions in normalization.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
from multiprocessing import Barrier, Process, Queue
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import requests
import torch
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from newspaper_explorer.config.base import get_config
import newspaper_explorer.config.external_tools as _external_tools
from newspaper_explorer.data.utils.text import chunk_text

# Ensure external tools import is used (sets HF_HOME environment variable)
del _external_tools

if TYPE_CHECKING:
    from multiprocessing.synchronize import Barrier as BarrierType

    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# DTA-CAB API endpoint
DTA_CAB_API_URL = "http://www.deutschestextarchiv.de/demo/cab/query"

# Minimum parts for DTA-CAB CSV parsing
MIN_CSV_PARTS = 2

# Supported Transnormer models for historical German text normalization
TRANSNORMER_MODELS: dict[str, str] = {
    "19c": "ybracke/transnormer-19c-beta-v02",  # 1780-1899
    "18-19c": "ybracke/transnormer-18-19c-beta-v01",  # 1700-1899
}


# =============================================================================
# Cache Helper Classes
# =============================================================================


class _TextCache:
    """File-based cache for normalized text chunks."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        use_cache: bool = True,
        file_extension: str = "txt",
    ) -> None:
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.file_extension = file_extension

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text."""
        return self.cache_dir / f"{self._get_cache_key(text)}.{self.file_extension}"

    def get(self, text: str) -> str | None:
        """Try to get normalized text from cache."""
        if not self.use_cache:
            return None

        cache_file = self._get_cache_path(text)
        if not cache_file.exists():
            return None

        try:
            return cache_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def save(self, text: str, normalized: str) -> None:
        """Save normalized text to cache."""
        if not self.use_cache:
            return

        cache_file = self._get_cache_path(text)
        try:
            cache_file.write_text(normalized, encoding="utf-8")
        except OSError as e:
            logger.debug(f"Cache write error: {e}")


class _JsonCache(_TextCache):
    """JSON-based cache for DTA-CAB results."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        use_cache: bool = True,
        format_name: str = "csv",
    ) -> None:
        super().__init__(cache_dir, use_cache=use_cache, file_extension="json")
        self.format_name = format_name

    def get(self, text: str) -> str | None:
        """Try to get normalized text from cache."""
        if not self.use_cache:
            return None

        cache_file = self._get_cache_path(text)
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            normalized = data.get("normalized")
            return normalized if isinstance(normalized, str) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def save(self, text: str, normalized: str) -> None:
        """Save normalized text to cache."""
        if not self.use_cache:
            return

        cache_file = self._get_cache_path(text)
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "original": text,
                        "normalized": normalized,
                        "format": self.format_name,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug(f"Cache write error: {e}")


# =============================================================================
# Transnormer Helper Functions
# =============================================================================


def _process_chunks_on_gpu(
    device: str,
    chunks: list[str],
    chunk_indices: list[int],
    model_name: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    max_input_length: int,
    result_queue: Queue[list[tuple[int, str]]] | None = None,
    barrier: BarrierType | None = None,
    gpu_id: int | None = None,
) -> list[tuple[int, str]] | None:
    """
    Helper function to process text chunks on a specific device.

    Can be used in two modes:
    1. Direct call (single-GPU): result_queue=None, barrier=None - returns results
    2. Spawned process (multi-GPU): result_queue and barrier provided - sends to queue

    Args:
        device: Device string (e.g., "cuda:0", "cuda", "cpu")
        chunks: List of text chunks to normalize
        chunk_indices: Original indices of chunks for reassembly
        model_name: HuggingFace model identifier
        batch_size: Batch size for inference
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum tokens to generate
        max_input_length: Maximum input sequence length
        result_queue: Optional multiprocessing queue for results (multi-GPU mode)
        barrier: Optional multiprocessing barrier for synchronization (multi-GPU mode)
        gpu_id: Optional GPU ID for progress bar labeling (multi-GPU mode)

    Returns:
        List of (index, normalized_text) tuples if result_queue is None, else None
    """
    model_obj, tokenizer, gen_config = _load_and_compile_model(device, model_name)

    gen_config.num_beams = num_beams
    gen_config.max_new_tokens = max_new_tokens

    # Wait for all GPUs to finish loading (multi-GPU mode only)
    if barrier is not None:
        barrier.wait()

    results = _run_inference_batches(
        model_obj=model_obj,
        tokenizer=tokenizer,
        gen_config=gen_config,
        device=device,
        chunks=chunks,
        chunk_indices=chunk_indices,
        batch_size=batch_size,
        max_input_length=max_input_length,
        gpu_id=gpu_id,
    )

    # Return results or send to queue
    if result_queue is not None:
        result_queue.put(results)
        return None
    return results


def _load_and_compile_model(
    device: str,
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig]:
    """Load model and tokenizer, optionally compile with torch.compile."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[assignment]
    model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # type: ignore[assignment]

    # Get generation config before compilation (torch.compile loses type info)
    gen_config = model_obj.generation_config  # type: ignore[assignment]

    # Try to use torch.compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            logger.info(f"Compiling model on {device} with torch.compile() for faster inference...")
            compiled_model = torch.compile(model_obj, mode="reduce-overhead")  # type: ignore[assignment]
            model_obj = compiled_model  # type: ignore[assignment]
            logger.info(f"Model compilation complete on {device}")
        except RuntimeError as e:
            logger.warning(
                f"Could not compile model on {device}: {e}. Continuing without compilation."
            )

    return model_obj, tokenizer, gen_config  # type: ignore[return-value]


def _run_inference_batches(
    *,
    model_obj: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    gen_config: GenerationConfig,
    device: str,
    chunks: list[str],
    chunk_indices: list[int],
    batch_size: int,
    max_input_length: int,
    gpu_id: int | None,
) -> list[tuple[int, str]]:
    """Run inference on batches of text chunks."""
    results: list[tuple[int, str]] = []

    # Configure progress bar based on mode
    pbar_desc, pbar_position, pbar_leave = _get_progress_bar_config(gpu_id)

    for i in tqdm(
        range(0, len(chunks), batch_size),
        desc=pbar_desc,
        position=pbar_position,
        leave=pbar_leave,
    ):
        batch_end = min(i + batch_size, len(chunks))
        batch_texts = chunks[i:batch_end]
        batch_indices = chunk_indices[i:batch_end]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model_obj.generate(  # type: ignore[attr-defined]
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config,
            )

        # Decode
        batch_normalized = tokenizer.batch_decode(outputs, skip_special_tokens=True)  # type: ignore[assignment]

        # Store with original indices
        for idx, normalized in zip(batch_indices, batch_normalized):
            results.append((idx, normalized))

    return results


def _get_progress_bar_config(gpu_id: int | None) -> tuple[str, int | None, bool]:
    """Get progress bar configuration based on GPU mode."""
    if gpu_id is not None:
        # Multi-GPU mode: labeled, positioned progress bar
        return f"GPU {gpu_id}", gpu_id, False
    # Single-GPU mode: simple progress bar
    return "Normalizing", None, True


def _resolve_model_name(model: str) -> str:
    """Resolve model shorthand to full HuggingFace model name."""
    if model in TRANSNORMER_MODELS:
        model_name = TRANSNORMER_MODELS[model]
        logger.info(f"Using Transnormer model '{model}': {model_name}")
    else:
        model_name = model
        logger.info(f"Using custom model: {model_name}")
    return model_name


def _setup_transnormer_cache(
    cache_dir: str | Path | None,
    model: str,
    *,
    use_cache: bool,
) -> _TextCache:
    """Setup cache directory for Transnormer."""
    if cache_dir is None:
        config = get_config()
        resolved_cache_dir = (
            Path(config.data_dir) / ".cache" / "transnormer" / model.replace("/", "_")
        )
    else:
        resolved_cache_dir = Path(cache_dir)

    cache = _TextCache(resolved_cache_dir, use_cache=use_cache)
    if use_cache:
        logger.info(f"Using cache directory: {resolved_cache_dir}")
    return cache


def _detect_device_and_gpus(
    device: str | int | None,
    num_gpus: int,
) -> tuple[str, int]:
    """Auto-detect device and validate GPU count."""
    # Auto-detect device if not specified
    resolved_device: str | int = (
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Auto-detect available GPUs
    if resolved_device == "cuda" and num_gpus > 1:
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
        logger.info(f"Using device: {resolved_device}")

    # Convert device to string if it's an int
    device_str = (
        f"cuda:{resolved_device}" if isinstance(resolved_device, int) else str(resolved_device)
    )
    return device_str, num_gpus


def _chunk_texts_with_cache(
    texts: list[str],
    max_input_length: int,
    cache: _TextCache,
) -> tuple[list[str], list[int], dict[int, str]]:
    """
    Chunk texts and check cache for each chunk.

    Returns:
        Tuple of (all_chunks, chunk_to_row_map, cached_results)
    """
    chunk_to_row_map: list[int] = []
    all_chunks: list[str] = []
    cached_results: dict[int, str] = {}

    for row_idx, text in tqdm(enumerate(texts), total=len(texts), desc="Chunking & checking cache"):
        if not text or not text.strip():
            # Empty text - keep as single chunk
            all_chunks.append(" ")
            chunk_to_row_map.append(row_idx)
        else:
            # Split long texts into chunks using utility function
            chunks = chunk_text(text, max_length=max_input_length)
            for chunk in chunks:
                chunk_idx = len(all_chunks)

                # Check cache for this chunk
                cached = cache.get(chunk)
                if cached is not None:
                    cached_results[chunk_idx] = cached

                all_chunks.append(chunk)
                chunk_to_row_map.append(row_idx)

    return all_chunks, chunk_to_row_map, cached_results


def _log_cache_stats(
    total_chunks: int,
    cache_hits: int,
    *,
    use_cache: bool,
) -> None:
    """Log cache hit statistics."""
    chunks_to_process = total_chunks - cache_hits

    if use_cache:
        cache_rate = (cache_hits / total_chunks * 100) if total_chunks > 0 else 0
        logger.info(f"Cache hits: {cache_hits:,} ({cache_rate:.1f}%)")
        logger.info(f"Chunks to process: {chunks_to_process:,} ({100 - cache_rate:.1f}%)")


def _process_with_multi_gpu(
    chunks_to_process_list: list[str],
    chunk_indices_to_process: list[int],
    num_gpus: int,
    model_name: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    max_input_length: int,
) -> list[tuple[int, str]]:
    """Process chunks using multiple GPUs in parallel."""
    logger.info(
        f"Distributing {len(chunks_to_process_list)} uncached chunks across {num_gpus} GPUs"
    )

    # CRITICAL: Set start method to 'spawn' for CUDA compatibility
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

    # Create a barrier to synchronize GPU processes
    barrier = Barrier(num_gpus)

    # Split UNCACHED chunks across GPUs
    num_uncached = len(chunks_to_process_list)
    chunks_per_gpu = (num_uncached + num_gpus - 1) // num_gpus
    processes: list[Process] = []
    result_queue: Queue[list[tuple[int, str]]] = Queue()

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * chunks_per_gpu
        end_idx = min(start_idx + chunks_per_gpu, num_uncached)

        if start_idx >= num_uncached:
            break

        gpu_chunks = chunks_to_process_list[start_idx:end_idx]
        gpu_indices = chunk_indices_to_process[start_idx:end_idx]

        logger.info(f"GPU {gpu_id}: will process {len(gpu_chunks)} chunks")

        p = Process(
            target=_process_chunks_on_gpu,
            kwargs={
                "device": f"cuda:{gpu_id}",
                "chunks": gpu_chunks,
                "chunk_indices": gpu_indices,
                "model_name": model_name,
                "batch_size": batch_size,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "max_input_length": max_input_length,
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
    processing_results: list[tuple[int, str]] = []
    for _ in range(len(processes)):
        processing_results.extend(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    return processing_results


def _process_with_single_device(
    chunks_to_process_list: list[str],
    chunk_indices_to_process: list[int],
    device_str: str,
    model_name: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    max_input_length: int,
) -> list[tuple[int, str]]:
    """Process chunks on a single device (GPU or CPU)."""
    logger.info("Loading Transnormer model (this may take a while on first run)")
    logger.info("Normalizing chunks in batches...")

    # Call worker function directly (no multiprocessing)
    processing_results = _process_chunks_on_gpu(
        device=device_str,
        chunks=chunks_to_process_list,
        chunk_indices=chunk_indices_to_process,
        model_name=model_name,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
        result_queue=None,  # Direct return mode
        barrier=None,  # No synchronization needed
        gpu_id=None,  # Single GPU mode
    )

    return processing_results or []


def _merge_results_with_cache(
    processing_results: list[tuple[int, str]],
    cached_results: dict[int, str],
    all_chunks: list[str],
    cache: _TextCache,
) -> None:
    """Merge processing results with cached results and save to cache."""
    logger.info("Merging results with cache...")
    for chunk_idx, normalized in processing_results:
        cached_results[chunk_idx] = normalized
        # Save new result to cache
        cache.save(all_chunks[chunk_idx], normalized)


def _reassemble_chunks(
    chunk_to_row_map: list[int],
    normalized_chunks: list[str],
    num_texts: int,
) -> list[str]:
    """Reassemble chunks back into original rows."""
    logger.info("Reassembling chunks...")
    normalized_texts = [""] * num_texts

    for chunk_idx, row_idx in enumerate(chunk_to_row_map):
        if normalized_texts[row_idx]:
            # Join with space if there are multiple chunks for this row
            normalized_texts[row_idx] += " " + normalized_chunks[chunk_idx]
        else:
            normalized_texts[row_idx] = normalized_chunks[chunk_idx]

    return normalized_texts


# =============================================================================
# DTA-CAB Helper Functions
# =============================================================================


def _setup_dtacab_cache(
    cache_dir: str | Path | None,
    format_name: str,
    *,
    use_cache: bool,
) -> _JsonCache:
    """Setup cache directory for DTA-CAB."""
    resolved_cache_dir = (
        Path(cache_dir) if cache_dir is not None else Path("data") / ".cache" / "dtacab"
    )

    cache = _JsonCache(resolved_cache_dir, use_cache=use_cache, format_name=format_name)
    if use_cache:
        logger.info(f"Using cache directory: {resolved_cache_dir}")
    return cache


def _normalize_single_text_dtacab(
    text: str,
    cache: _JsonCache,
    format_name: str,
    timeout: int,
) -> str:
    """Normalize a single text via DTA-CAB API (with caching)."""
    if not text or not text.strip():
        return text

    # Try cache first
    cached = cache.get(text)
    if cached is not None:
        return cached

    try:
        result = _call_dtacab_api(text, format_name, timeout)
        cache.save(text, result)
        return result

    except requests.exceptions.Timeout:
        logger.warning(f"DTA-CAB request timeout for text: {text[:50]}...")
        return text
    except requests.exceptions.RequestException as e:
        logger.warning(f"DTA-CAB request failed: {e}")
        return text


def _call_dtacab_api(text: str, format_name: str, timeout: int) -> str:
    """Call DTA-CAB API and parse response."""
    params = {
        "fmt": format_name,
        "clean": "1",
        "q": text,
    }

    response = requests.get(DTA_CAB_API_URL, params=params, timeout=timeout)
    response.raise_for_status()

    return _parse_dtacab_response(response.text, format_name, text)


def _parse_dtacab_response(response_text: str, format_name: str, original_text: str) -> str:
    """Parse DTA-CAB response based on format."""
    if format_name == "csv":
        return _parse_csv_response(response_text)

    if format_name == "txt":
        return response_text.strip()

    logger.warning(f"Unsupported format '{format_name}', returning original text")
    return original_text


def _parse_csv_response(response_text: str) -> str:
    """Parse CSV format response from DTA-CAB."""
    # CSV format: token\tnorm\tlemma\tpos
    # We want the normalized forms (second column)
    lines = response_text.strip().split("\n")
    normalized_tokens: list[str] = []

    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= MIN_CSV_PARTS:
            normalized_tokens.append(parts[1])

    return " ".join(normalized_tokens)


# =============================================================================
# Main Public Functions
# =============================================================================


def transnormer(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: str | None = None,
    model: str = "19c",
    batch_size: int = 32,
    num_beams: int = 4,
    max_new_tokens: int = 512,
    max_input_length: int = 512,
    device: str | int | None = None,
    num_gpus: int = 1,
    cache_dir: str | Path | None = None,
    *,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Normalize historical German text using Transnormer transformer model.

    Uses transformer-based neural normalization specifically trained for
    historical German text (1600-1900). Much faster than DTA-CAB API
    and higher quality than basic character replacement.

    Advantages over other methods:
    - Neural model trained on historical German corpus
    - Handles complex spelling variations (not just character mapping)
    - Local inference (no API calls)
    - Batched processing for efficiency
    - Better context awareness than rule-based methods

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_transnormer)
        model: Model to use. Options:
               - "19c": For text from 1780-1899 (transnormer-19c-beta-v02)
               - "18-19c": For text from 1700-1899 (transnormer-18-19c-beta-v01)
               - Or provide full model name from Hugging Face Hub
        batch_size: Batch size for inference (adjust based on GPU memory)
        num_beams: Number of beams for beam search (default: 4, higher = better quality)
        max_new_tokens: Maximum length of generated normalized text (default: 512)
        max_input_length: Maximum input chunk length in characters (default: 512)
                         Texts longer than this will be split into chunks
        device: Device for inference ('cuda', 'cpu', or None for auto-detect)
        num_gpus: Number of GPUs to use for parallel processing (default: 1)
        cache_dir: Directory for caching normalized chunks (default: .cache/transnormer)
                  Set to None to disable caching
        use_cache: Whether to use caching (default: True)
                  Caches individual text chunks by their hash for fast resume

    Returns:
        DataFrame with Transnormer normalized text column

    Example:
        >>> df = transnormer(
        ...     df,
        ...     model="19c",
        ...     batch_size=64,  # Larger batches for GPU
        ...     device='cuda',
        ...     use_cache=True  # Enable resume capability
        ... )

    Note:
        Caching is done at the chunk level (not row level), so if processing
        is interrupted, already-normalized chunks will be read from cache
        when you re-run, avoiding redundant GPU computation.
    """
    output_column = output_column or f"{input_column}_transnormer"
    logger.info(f"Normalizing with Transnormer: {input_column} → {output_column}")

    # Setup
    model_name = _resolve_model_name(model)
    cache = _setup_transnormer_cache(cache_dir, model, use_cache=use_cache)
    device_str, num_gpus = _detect_device_and_gpus(device, num_gpus)

    # Get texts and chunk them
    texts = df[input_column].to_list()
    logger.info(f"Chunking texts (max_input_length={max_input_length} chars)")

    all_chunks, chunk_to_row_map, cached_results = _chunk_texts_with_cache(
        texts, max_input_length, cache
    )

    total_chunks = len(all_chunks)
    cache_hits = len(cached_results)

    logger.info(f"Split {len(texts):,} texts into {total_chunks:,} chunks")
    _log_cache_stats(total_chunks, cache_hits, use_cache=use_cache)
    logger.info(f"Processing with batch_size={batch_size}, num_beams={num_beams}")

    # Filter out chunks that are already cached
    chunks_to_process_list: list[str] = []
    chunk_indices_to_process: list[int] = []

    for chunk_idx, chunk in enumerate(all_chunks):
        if chunk_idx not in cached_results:
            chunks_to_process_list.append(chunk)
            chunk_indices_to_process.append(chunk_idx)

    # Process chunks
    if len(chunks_to_process_list) == 0:
        logger.info("All chunks found in cache, skipping model inference")
    elif num_gpus > 1:
        processing_results = _process_with_multi_gpu(
            chunks_to_process_list,
            chunk_indices_to_process,
            num_gpus,
            model_name,
            batch_size,
            num_beams,
            max_new_tokens,
            max_input_length,
        )
        _merge_results_with_cache(processing_results, cached_results, all_chunks, cache)
        logger.info("Multi-GPU processing complete")
    else:
        processing_results = _process_with_single_device(
            chunks_to_process_list,
            chunk_indices_to_process,
            device_str,
            model_name,
            batch_size,
            num_beams,
            max_new_tokens,
            max_input_length,
        )
        _merge_results_with_cache(processing_results, cached_results, all_chunks, cache)

    # Build final list in order and reassemble
    normalized_chunks = [cached_results.get(i, "") for i in range(total_chunks)]
    normalized_texts = _reassemble_chunks(chunk_to_row_map, normalized_chunks, len(texts))

    # Add normalized column
    result_df = df.with_columns([pl.Series(name=output_column, values=normalized_texts)])
    logger.info(f"Transnormer normalized {len(result_df):,} rows from {total_chunks:,} chunks")
    return result_df


def dta_cab(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: str | None = None,
    batch_size: int = 100,
    timeout: int = 30,
    format: str = "csv",
    cache_dir: str | Path | None = None,
    *,
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Normalize historical German text using DTA-CAB web service.

    Uses the Deutsches Textarchiv Cascaded Analysis Broker for
    sophisticated normalization of historical German texts (16th-20th century).
    Includes spelling normalization, tokenization, and more.

    NOTE: This is MUCH slower than basic normalization due to API calls.
    Recommended for smaller datasets or when high-quality normalization is critical.

    Results are cached by default to avoid re-processing identical texts.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_dtacab)
        batch_size: Number of texts to process in each batch
        timeout: Request timeout in seconds
        format: Output format ('csv', 'txt', or 'tcf')
        cache_dir: Directory for cache files (default: data/.cache/dtacab)
        use_cache: Whether to use caching (default: True)

    Returns:
        DataFrame with DTA-CAB normalized text column
    """
    output_column = output_column or f"{input_column}_dtacab"

    logger.info(f"Normalizing with DTA-CAB: {input_column} → {output_column}")
    logger.warning("DTA-CAB normalization is SLOW - expect 1-10 seconds per text")

    cache = _setup_dtacab_cache(cache_dir, format, use_cache=use_cache)

    # Process texts in batches with progress bar
    texts = df[input_column].to_list()
    normalized_texts: list[str] = []

    cache_hits = 0
    api_calls = 0

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    for i in tqdm(range(0, len(texts), batch_size), desc="DTA-CAB batches"):
        batch = texts[i : i + batch_size]

        for text in batch:
            # Check if we got from cache
            if use_cache and cache.get(text) is not None:
                cache_hits += 1
            else:
                api_calls += 1

            normalized_texts.append(_normalize_single_text_dtacab(text, cache, format, timeout))

    # Add normalized column
    result_df = df.with_columns([pl.Series(name=output_column, values=normalized_texts)])

    # Show cache statistics
    if use_cache:
        cache_rate = (cache_hits / len(texts) * 100) if len(texts) > 0 else 0
        logger.info(f"Cache hits: {cache_hits:,} ({cache_rate:.1f}%)")
        logger.info(f"API calls: {api_calls:,} ({100 - cache_rate:.1f}%)")

    logger.info(f"DTA-CAB normalized {len(result_df):,} rows")
    return result_df
