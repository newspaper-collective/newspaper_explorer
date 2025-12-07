"""
Historical spelling modernization methods.

Provides heavy/external normalization for historical German texts:
- Transnormer: Neural transformer-based spelling modernization (local GPU)
- DTA-CAB: Web API-based spelling modernization (Deutsches Textarchiv)

These operations are expensive (require external models/APIs) and are separated
from the lightweight rule-based functions in normalization.py.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import polars as pl
import torch
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from newspaper_explorer.config import external_tools  # noqa: F401 (sets HF_HOME)
from newspaper_explorer.config.base import get_config
from newspaper_explorer.data.utils.text import chunk_text

logger = logging.getLogger(__name__)


def _process_chunks_on_gpu(
    device: str,
    chunks: list[str],
    chunk_indices: list[int],
    model_name: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    max_input_length: int,
    result_queue=None,
    barrier=None,
    gpu_id: Optional[int] = None,
):
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

    # Load model on this device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Get generation config before compilation (torch.compile loses type info)
    gen_config = model_obj.generation_config

    # Try to use torch.compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            logger.info(f"Compiling model on {device} with torch.compile() for faster inference...")
            # Store original type before compilation for type checking
            compiled_model = torch.compile(model_obj, mode="reduce-overhead")
            # Keep reference to compiled model but preserve original type hint
            model_obj = compiled_model  # type: ignore[assignment]
            logger.info(f"Model compilation complete on {device}")
        except Exception as e:
            logger.warning(
                f"Could not compile model on {device}: {e}. Continuing without compilation."
            )
    gen_config.num_beams = num_beams
    gen_config.max_new_tokens = max_new_tokens

    # Wait for all GPUs to finish loading (multi-GPU mode only)
    if barrier is not None:
        barrier.wait()

    # Process chunks in batches
    results = []

    # Configure progress bar based on mode
    if gpu_id is not None:
        # Multi-GPU mode: labeled, positioned progress bar
        pbar_desc = f"GPU {gpu_id}"
        pbar_position = gpu_id
        pbar_leave = False
    else:
        # Single-GPU mode: simple progress bar
        pbar_desc = "Normalizing"
        pbar_position = None
        pbar_leave = True

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
        batch_normalized = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Store with original indices
        for idx, normalized in zip(batch_indices, batch_normalized):
            results.append((idx, normalized))

    # Return results or send to queue
    if result_queue is not None:
        result_queue.put(results)
    else:
        return results


def transnormer(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    model: str = "19c",
    batch_size: int = 32,
    num_beams: int = 4,
    max_new_tokens: int = 512,
    max_input_length: int = 512,
    device: Optional[Union[str, int]] = None,
    num_gpus: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
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
        >>> df = normalize_transnormer(
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
    if output_column is None:
        output_column = f"{input_column}_transnormer"

    logger.info(f"Normalizing with Transnormer: {input_column} → {output_column}")

    # Supported Transnormer models for historical German text normalization
    transnormer_models: dict[str, str] = {
        "19c": "ybracke/transnormer-19c-beta-v02",  # 1780-1899
        "18-19c": "ybracke/transnormer-18-19c-beta-v01",  # 1700-1899
    }

    # Resolve model name
    if model in transnormer_models:
        model_name = transnormer_models[model]
        logger.info(f"Using Transnormer model '{model}': {model_name}")
    else:
        model_name = model
        logger.info(f"Using custom model: {model_name}")

    # Setup cache directory
    if cache_dir is None:
        config = get_config()
        cache_dir = Path(config.data_dir) / ".cache" / "transnormer" / model.replace("/", "_")
    else:
        cache_dir = Path(cache_dir)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")

    def get_cache_key(text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_from_cache(text: str) -> Optional[str]:
        """Try to get normalized text from cache."""
        if not use_cache:
            return None

        cache_key = get_cache_key(text)
        cache_file = cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
                return None
        return None

    def save_to_cache(text: str, normalized: str) -> None:
        """Save normalized text to cache."""
        if not use_cache:
            return

        cache_key = get_cache_key(text)
        cache_file = cache_dir / f"{cache_key}.txt"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(normalized)
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    # Auto-detect device and GPU count if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect available GPUs
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

    # Get texts to normalize
    texts = df[input_column].to_list()

    logger.info(f"Chunking texts (max_input_length={max_input_length} chars)")

    # Track which chunks belong to which original row
    chunk_to_row_map = []
    all_chunks = []
    cached_results = {}  # chunk_idx -> normalized_text

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
                cached = get_from_cache(chunk)
                if cached is not None:
                    cached_results[chunk_idx] = cached

                all_chunks.append(chunk)
                chunk_to_row_map.append(row_idx)

    total_chunks = len(all_chunks)
    cache_hits = len(cached_results)
    chunks_to_process = total_chunks - cache_hits

    logger.info(f"Split {len(texts):,} texts into {total_chunks:,} chunks")
    if use_cache:
        cache_rate = (cache_hits / total_chunks * 100) if total_chunks > 0 else 0
        logger.info(f"Cache hits: {cache_hits:,} ({cache_rate:.1f}%)")
        logger.info(f"Chunks to process: {chunks_to_process:,} ({100 - cache_rate:.1f}%)")
    logger.info(f"Processing with batch_size={batch_size}, num_beams={num_beams}")

    # Filter out chunks that are already cached
    chunks_to_process_list = []
    chunk_indices_to_process = []

    for chunk_idx, chunk in enumerate(all_chunks):
        if chunk_idx not in cached_results:
            chunks_to_process_list.append(chunk)
            chunk_indices_to_process.append(chunk_idx)

    # Skip processing if all chunks are cached
    if len(chunks_to_process_list) == 0:
        logger.info("All chunks found in cache, skipping model inference")
        normalized_chunks = [cached_results.get(i, "") for i in range(total_chunks)]

    # Use multi-GPU processing if requested
    elif num_gpus > 1:
        logger.info(
            f"Distributing {len(chunks_to_process_list)} uncached chunks across {num_gpus} GPUs"
        )

        import multiprocessing as mp

        # CRITICAL: Set start method to 'spawn' for CUDA compatibility
        # This must be done before creating any processes
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, ignore
            pass

        from multiprocessing import Barrier, Process, Queue

        # Create a barrier to synchronize GPU processes
        # All GPUs will wait until all have loaded their models
        barrier = Barrier(num_gpus)

        # Split UNCACHED chunks across GPUs
        num_uncached = len(chunks_to_process_list)
        chunks_per_gpu = (num_uncached + num_gpus - 1) // num_gpus
        processes = []
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
        processing_results = []
        for _ in range(len(processes)):
            processing_results.extend(result_queue.get())

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Merge processing results with cached results
        logger.info("Merging results with cache...")
        for chunk_idx, normalized in processing_results:
            cached_results[chunk_idx] = normalized
            # Save new result to cache
            save_to_cache(all_chunks[chunk_idx], normalized)

        # Build final list in order
        normalized_chunks = [cached_results.get(i, "") for i in range(total_chunks)]

        logger.info(f"Multi-GPU processing complete")

    else:
        # Single GPU/CPU processing - use the same worker function
        logger.info(f"Loading Transnormer model (this may take a while on first run)")
        logger.info("Normalizing chunks in batches...")

        # Convert device to string if it's an int (e.g., 0 -> "cuda:0")
        device_str = f"cuda:{device}" if isinstance(device, int) else device

        # Call worker function directly (no multiprocessing) - only for uncached chunks
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

        # Merge processing results with cached results
        logger.info("Merging results with cache...")
        if processing_results is None:
            processing_results = []
        for chunk_idx, normalized in processing_results:
            cached_results[chunk_idx] = normalized
            # Save new result to cache
            save_to_cache(all_chunks[chunk_idx], normalized)

        # Build final list in order
        normalized_chunks = [cached_results.get(i, "") for i in range(total_chunks)]

    # Reassemble chunks back into original rows
    logger.info("Reassembling chunks...")
    normalized_texts = [""] * len(texts)

    for chunk_idx, row_idx in enumerate(chunk_to_row_map):
        if normalized_texts[row_idx]:
            # Join with space if there are multiple chunks for this row
            normalized_texts[row_idx] += " " + normalized_chunks[chunk_idx]
        else:
            normalized_texts[row_idx] = normalized_chunks[chunk_idx]

    # Add normalized column
    df = df.with_columns([pl.Series(name=output_column, values=normalized_texts)])

    logger.info(f"Transnormer normalized {len(df):,} rows from {total_chunks:,} chunks")
    return df


def dta_cab(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    batch_size: int = 100,
    timeout: int = 30,
    format: str = "csv",
    cache_dir: Optional[Union[str, Path]] = None,
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
    if output_column is None:
        output_column = f"{input_column}_dtacab"

    logger.info(f"Normalizing with DTA-CAB: {input_column} → {output_column}")
    logger.warning("DTA-CAB normalization is SLOW - expect 1-10 seconds per text")

    try:
        import requests
    except ImportError:
        raise ImportError(
            "DTA-CAB normalization requires 'requests' package.\nInstall with: pip install requests"
        )

    import hashlib
    import json

    from tqdm import tqdm

    # DTA-CAB API endpoint
    API_URL = "http://www.deutschestextarchiv.de/demo/cab/query"

    # Setup cache directory
    if cache_dir is None:
        cache_dir = Path("data") / ".cache" / "dtacab"
    else:
        cache_dir = Path(cache_dir)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")

    def get_cache_key(text: str) -> str:
        """Generate cache key from text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_from_cache(text: str) -> Optional[str]:
        """Try to get normalized text from cache."""
        if not use_cache:
            return None

        cache_key = get_cache_key(text)
        cache_file = cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    normalized = data.get("normalized")
                    return normalized if isinstance(normalized, str) else None
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
                return None
        return None

    def save_to_cache(text: str, normalized: str) -> None:
        """Save normalized text to cache."""
        if not use_cache:
            return

        cache_key = get_cache_key(text)
        cache_file = cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "original": text,
                        "normalized": normalized,
                        "format": format,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def normalize_text(text: str) -> str:
        """Normalize a single text via DTA-CAB API (with caching)."""
        if not text or not text.strip():
            return text

        # Try cache first
        cached = get_from_cache(text)
        if cached is not None:
            return cached

        try:
            # Prepare request
            params = {
                "fmt": format,
                "clean": "1",  # Clean output
                "q": text,
            }

            # Make request
            response = requests.get(
                API_URL,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()

            # Parse response based on format
            if format == "csv":
                # CSV format: token\tnorm\tlemma\tpos
                # We want the normalized forms (second column)
                lines = response.text.strip().split("\n")
                normalized_tokens = []

                for line in lines:
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        normalized_tokens.append(parts[1])

                result = " ".join(normalized_tokens)

            elif format == "txt":
                # Plain text format - already normalized
                result = response.text.strip()

            else:
                logger.warning(f"Unsupported format '{format}', returning original text")
                result = text

            # Save to cache
            save_to_cache(text, result)
            return result

        except requests.exceptions.Timeout:
            logger.warning(f"DTA-CAB request timeout for text: {text[:50]}...")
            return text
        except requests.exceptions.RequestException as e:
            logger.warning(f"DTA-CAB request failed: {e}")
            return text
        except Exception as e:
            logger.warning(f"DTA-CAB processing error: {e}")
            return text

    # Process texts in batches with progress bar
    texts = df[input_column].to_list()
    normalized_texts = []

    cache_hits = 0
    api_calls = 0

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    for i in tqdm(range(0, len(texts), batch_size), desc="DTA-CAB batches"):
        batch = texts[i : i + batch_size]

        for text in batch:
            # Check if we got from cache
            if use_cache and get_from_cache(text) is not None:
                cache_hits += 1
            else:
                api_calls += 1

            normalized_texts.append(normalize_text(text))

    # Add normalized column
    df = df.with_columns([pl.Series(name=output_column, values=normalized_texts)])

    # Show cache statistics
    if use_cache:
        cache_rate = (cache_hits / len(texts) * 100) if len(texts) > 0 else 0
        logger.info(f"Cache hits: {cache_hits:,} ({cache_rate:.1f}%)")
        logger.info(f"API calls: {api_calls:,} ({100 - cache_rate:.1f}%)")

    logger.info(f"DTA-CAB normalized {len(df):,} rows")
    return df
