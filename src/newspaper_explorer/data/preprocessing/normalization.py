"""
Historical German text normalization methods.

Provides specialized normalization for historical German newspaper texts:
- Basic character mapping (ſ→s, ß→ss)
- Transnormer neural normalization (transformer-based)
- DTA-CAB API normalization (web service)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


def simple(
    df: pl.DataFrame,
    text_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Normalize historical German text characters.

    Replaces archaic German characters:
    - ſ (long s) → s
    - ẞ (capital sharp s) → SS
    - ß (sharp s) → ss

    Args:
        df: Input DataFrame
        text_column: Column containing text to process
        output_column: Name for output column (default: {text_column}_normalized)

    Returns:
        DataFrame with normalized text column
    """
    if output_column is None:
        output_column = f"{text_column}_normalized"

    logger.info(f"Normalizing historical characters: {text_column} → {output_column}")

    df = df.with_columns(
        [
            pl.col(text_column)
            .str.replace_all("ẞ", "SS")
            .str.replace_all("ß", "ss")
            .str.replace_all("ſs", "ss")
            .str.replace_all("ſ", "s")
            .alias(output_column)
        ]
    )

    logger.info(f"Normalized {len(df):,} rows")
    return df


def transnormer(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    model: str = "19c",
    batch_size: int = 32,
    num_beams: int = 4,
    max_length: int = 128,
    device: Optional[Union[str, int]] = None,
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
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_transnormer)
        model: Model to use. Options:
               - "19c": For text from 1780-1899 (transnormer-19c-beta-v02)
               - "18-19c": For text from 1700-1899 (transnormer-18-19c-beta-v01)
               - Or provide full model name from Hugging Face Hub
        batch_size: Batch size for inference (adjust based on GPU memory)
        num_beams: Number of beams for beam search (default: 4, higher = better quality)
        max_length: Maximum length of normalized text (default: 128)
        device: Device for inference ('cuda', 'cpu', 0, -1, or None for auto-detect)

    Returns:
        DataFrame with Transnormer normalized text column

    Example:
        >>> df = normalize_transnormer(
        ...     df,
        ...     model="19c",
        ...     batch_size=64,  # Larger batches for GPU
        ...     device='cuda'
        ... )
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_transnormer"

    logger.info(f"Normalizing with Transnormer: {input_column} → {output_column}")

    try:
        import torch
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Transnormer normalization requires 'transformers' and 'torch'.\n"
            "Install with: pip install -e '.[normalize]'"
        )

    from tqdm import tqdm

    # Supported Transnormer models for historical German text normalization
    TRANSNORMER_MODELS = {
        "19c": "ybracke/transnormer-19c-beta-v02",  # 1780-1899
        "18-19c": "ybracke/transnormer-18-19c-beta-v01",  # 1700-1899
    }

    # Resolve model name
    if model in TRANSNORMER_MODELS:
        model_name = TRANSNORMER_MODELS[model]
        logger.info(f"Using Transnormer model '{model}': {model_name}")
    else:
        model_name = model
        logger.info(f"Using custom model: {model_name}")

    # Auto-detect device if not specified
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        logger.info(f"Using device: {device_name}")

    # Load transnormer pipeline
    logger.info(f"Loading Transnormer model (this may take a while on first run)")
    try:
        normalizer = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    # Get texts to normalize
    texts = df[input_column].to_list()
    normalized_texts = []

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Transnormer batches"):
        batch = texts[i : i + batch_size]

        # Filter out empty texts
        batch_filtered = [t if t and t.strip() else " " for t in batch]

        # Run inference
        results = normalizer(
            batch_filtered,
            num_beams=num_beams,
            max_length=max_length,
            truncation=True,
            batch_size=batch_size,
        )

        # Extract normalized text
        batch_normalized = [r["generated_text"] for r in results]
        normalized_texts.extend(batch_normalized)

    # Add normalized column
    df = df.with_columns([pl.Series(name=output_column, values=normalized_texts)])

    logger.info(f"Transnormer normalized {len(df):,} rows")
    return df


def dta_cab(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
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
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_dtacab)
        batch_size: Number of texts to process in each batch
        timeout: Request timeout in seconds
        format: Output format ('csv', 'txt', or 'tcf')
        cache_dir: Directory for cache files (default: data/.cache/dtacab)
        use_cache: Whether to use caching (default: True)

    Returns:
        DataFrame with DTA-CAB normalized text column
    """
    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_dtacab"

    logger.info(f"Normalizing with DTA-CAB: {input_column} → {output_column}")
    logger.warning("DTA-CAB normalization is SLOW - expect 1-10 seconds per text")

    try:
        import requests
    except ImportError:
        raise ImportError(
            "DTA-CAB normalization requires 'requests' package.\n"
            "Install with: pip install requests"
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
        logger.info(f"API calls: {api_calls:,} ({100-cache_rate:.1f}%)")

    logger.info(f"DTA-CAB normalized {len(df):,} rows")
    return df
