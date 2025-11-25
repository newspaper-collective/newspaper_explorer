"""
Text normalization methods.

Provides comprehensive normalization for historical German newspaper texts:
- Unicode normalization (NFC with ftfy encoding repair, character translation, control chars)
- Diacritic removal (ä→a, ö→o, ü→u)
- Historical German character mapping (ſ→s, ß→ss)
- Transnormer neural normalization (transformer-based)
- DTA-CAB API normalization (web service)
"""

import hashlib
import logging
import os
import unicodedata
from pathlib import Path
from typing import Optional, Union

import ftfy
import polars as pl
from tqdm import tqdm

# Import text chunking utility
from newspaper_explorer.data.utils.text import chunk_text

# Set Hugging Face cache directory to avoid filling up home directory
# Use project's .cache directory for model downloads
_project_root = Path(__file__).parent.parent.parent.parent
os.environ["HF_HOME"] = str(_project_root / ".cache" / "huggingface")

logger = logging.getLogger(__name__)


# Character translation maps for Unicode normalization
# Two modes: conservative (default) and aggressive

# CONSERVATIVE: Only fix clear OCR errors, preserve semantic distinctions
UNICODE_TRANSLATION_CONSERVATIVE = {
    # Remove problematic hyphens/spaces
    0x00AD: None,  # soft hyphen (invisible, remove)
    0x2011: "-",  # non-breaking hyphen → regular hyphen
    # NOTE: KEEP - (hyphen), – (en dash), — (em dash) - these have semantic meaning!
    # Normalize various space types to regular space (OCR-specific, not encoding)
    0x00A0: 32,  # non-breaking space
    0x202F: 32,  # narrow no-break space
    0x2000: 32,  # en quad
    0x2001: 32,  # em quad
    0x2002: 32,  # en space
    0x2003: 32,  # em space
    0x2004: 32,  # three-per-em space
    0x2005: 32,  # four-per-em space
    0x2006: 32,  # six-per-em space
    0x2007: 32,  # figure space
    0x2008: 32,  # punctuation space
    0x2009: 32,  # thin space
    0x200A: 32,  # hair space
    # Remove OCR artifacts (bullets, boxes, etc.) - OCR garbage, not encoding
    0x2022: None,  # • bullet
    0x25AA: None,  # ▪ black small square
    0x2023: None,  # ‣ triangular bullet
    0x25E6: None,  # ◦ white bullet
    0x25A0: None,  # ■ black square
    0x25FC: None,  # ◼ black medium square
    0x25AE: None,  # ▮ black vertical rectangle
    0x261E: None,  # ☞ white right pointing index
    # Cyrillic confusables (OCR errors - looks like Latin but isn't)
    0x0410: "A",  # А → A (Cyrillic A)
    0x0412: "B",  # В → B (Cyrillic Ve)
    0x0415: "E",  # Е → E (Cyrillic Ie)
    0x041A: "K",  # К → K (Cyrillic Ka)
    0x041C: "M",  # М → M (Cyrillic Em)
    0x041D: "H",  # Н → H (Cyrillic En)
    0x041E: "O",  # О → O (Cyrillic O)
    0x0420: "P",  # Р → P (Cyrillic Er)
    0x0421: "C",  # С → C (Cyrillic Es)
    0x0422: "T",  # Т → T (Cyrillic Te)
    0x0425: "X",  # Х → X (Cyrillic Kha)
    0x0430: "a",  # а → a (Cyrillic a)
    0x0435: "e",  # е → e (Cyrillic ie)
    0x043E: "o",  # о → o (Cyrillic o)
    0x0440: "p",  # р → p (Cyrillic er)
    0x0441: "c",  # с → c (Cyrillic es)
    0x0443: "y",  # у → y (Cyrillic u)
    0x0445: "x",  # х → x (Cyrillic kha)
    0x0455: "s",  # ѕ → s (Cyrillic dze)
    0x0456: "i",  # і → i (Cyrillic byelorussian-ukrainian i)
    0x0458: "j",  # ј → j (Cyrillic je)
    0x04CF: "l",  # ӏ → l (Cyrillic palochka)
    0x0405: "S",  # Ѕ → S (Cyrillic Dze)
    0x0406: "I",  # І → I (Cyrillic Byelorussian-Ukrainian I)
    # Greek confusables (OCR errors - looks like Latin but isn't)
    0x0391: "A",  # Α → A (Greek Alpha)
    0x0392: "B",  # Β → B (Greek Beta)
    0x0395: "E",  # Ε → E (Greek Epsilon)
    0x0396: "Z",  # Ζ → Z (Greek Zeta)
    0x0397: "H",  # Η → H (Greek Eta)
    0x0399: "I",  # Ι → I (Greek Iota)
    0x039A: "K",  # Κ → K (Greek Kappa)
    0x039C: "M",  # Μ → M (Greek Mu)
    0x039D: "N",  # Ν → N (Greek Nu)
    0x039F: "O",  # Ο → O (Greek Omicron)
    0x03A1: "P",  # Ρ → P (Greek Rho)
    0x03A4: "T",  # Τ → T (Greek Tau)
    0x03A7: "X",  # Χ → X (Greek Chi)
    # Accented i (German doesn't use these - OCR errors)
    0x00ED: "i",  # í → i (acute)
    0x00EC: "i",  # ì → i (grave)
    0x00EE: "i",  # î → i (circumflex)
    0x00EF: "i",  # ï → i (diaeresis)
    0x0129: "i",  # ĩ → i (tilde)
    0x012B: "i",  # ī → i (macron)
    0x012D: "i",  # ĭ → i (breve)
    0x012F: "i",  # į → i (ogonek)
    0x0131: "i",  # ı → i (dotless)
    0x00CD: "I",  # Í → I (uppercase acute)
    0x00CC: "I",  # Ì → I (uppercase grave)
    0x00CE: "I",  # Î → I (uppercase circumflex)
    0x00CF: "I",  # Ï → I (uppercase diaeresis)
    0x0128: "I",  # Ĩ → I (uppercase tilde)
    0x012A: "I",  # Ī → I (uppercase macron)
    0x012C: "I",  # Ĭ → I (uppercase breve)
    0x012E: "I",  # Į → I (uppercase ogonek)
}

# AGGRESSIVE: Unify all dashes (good for NLP/topic modeling, loses nuance)
UNICODE_TRANSLATION_AGGRESSIVE = {
    **UNICODE_TRANSLATION_CONSERVATIVE,
    # Add aggressive dash normalization
    0x2013: "-",  # – en dash → hyphen
    0x2014: "-",  # — em dash → hyphen
    0x2212: "-",  # − minus sign → hyphen
}


def normalize_unicode(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    aggressive: bool = False,
) -> pl.DataFrame:
    """
    Normalize Unicode characters for OCR text.

    Recommended as the FIRST STEP in preprocessing pipelines.
    Handles common OCR issues while preserving semantic distinctions.

    **CRITICAL: Uses NFC normalization (NOT NFKC) for historical text preservation.**

    NFC vs NFKC:
    - NFC (Canonical Composition): Preserves character distinctions needed for
      historical analysis. Represents ä, ö, ü, ß as single codepoints.
    - NFKC (Compatibility): Performs aggressive normalization that LOSES information.
      NOT suitable for historical text preservation!

    **Processing stages:**
    1. ftfy encoding repair: Fixes mojibake, HTML entities, ligatures, control chars, NFC
    2. Character translation: Cyrillic/Greek confusables, accented i, spaces, OCR artifacts

    **What ftfy.fix_text() handles:**
    - Mojibake repair: "schÃ¶n" → "schön"
    - HTML entities: "&auml;" → "ä"
    - Ligatures: "ﬁ ﬂ ﬀ" → "fi fl ff" (Fraktur!)
    - Curly quotes: "„text"" → "\"text\""
    - Control characters removal (keeps newlines/tabs)
    - NFC normalization at the end

    **What our character translation handles:**
    - Cyrillic confusables: "Теxt" → "Text" (А→A, о→o)
    - Greek confusables: "Grееk" → "Greek" (Α→A, Ε→E)
    - Accented i: "ínsíde" → "inside" (German doesn't use these)
    - Space normalization: Various Unicode spaces → regular space
    - OCR artifacts: Bullets, boxes → removed
    - Dash handling: Conservative vs aggressive modes

    **Two dash handling modes:**

    **Conservative mode (default, aggressive=False):**
    - Preserves semantic punctuation (en dash, em dash)
    - "1914–1918" stays as "1914–1918" (en dash for ranges)
    - "Der Kaiser — so berichtet" stays with em dash (emphasis)
    - Good for: historical analysis, quotations, entity extraction

    **Aggressive mode (aggressive=True):**
    - Unifies all dashes to hyphen: – → -, — → -, − → -
    - "1914–1918" becomes "1914-1918"
    - Good for: topic modeling, embeddings, NLP where nuance doesn't matter

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_unicode)
        aggressive: If True, unifies all dashes to hyphen (default: False)

    Returns:
        DataFrame with normalized Unicode text

    Example:
        >>> # Conservative (default): preserves semantic dashes
        >>> df = normalize_unicode(df)
        >>> # "1914–1918" → "1914–1918" (en dash kept)
        >>> # „Quoted text" → "\"Quoted text\""
        >>> # "schÃ¶n" → "schön" (mojibake fixed by ftfy)
        >>> # "Теxt with Суrilliс" → "Text with Cyrillic"
        >>> # "ﬁ ﬂ ﬀ" → "fi fl ff" (ligatures)
        >>>
        >>> # Aggressive: unifies all dashes
        >>> df = normalize_unicode(df, aggressive=True)
        >>> # "1914–1918" → "1914-1918" (en dash → hyphen)
    """
    if output_column is None:
        output_column = f"{input_column}_unicode"

    # Choose translation map based on mode
    translation_map = (
        UNICODE_TRANSLATION_AGGRESSIVE if aggressive else UNICODE_TRANSLATION_CONSERVATIVE
    )
    mode_name = "aggressive" if aggressive else "conservative"

    logger.info(f"Normalizing Unicode ({mode_name}): {input_column} → {output_column}")

    def normalize_text(text: str) -> str:
        """Apply Unicode normalization to a single text."""
        if not text:
            return text

        # 1. Fix encoding corruption with ftfy
        # This must happen FIRST before any other processing
        # ftfy.fix_text() automatically does:
        # - Mojibake repair (schÃ¶n → schön)
        # - HTML entity decoding (&auml; → ä)
        # - Ligature fixing (ﬁ, ﬂ → fi, fl) - perfect for Fraktur OCR!
        # - Quote normalization
        # - Control character removal
        # - NFC normalization (at the end)
        text = ftfy.fix_text(text, normalization="NFC")

        # 2. Apply character translation map (our custom confusables handling)
        # This adds OCR-specific fixes that ftfy doesn't handle:
        # - Cyrillic/Greek confusables (А→A, о→o, Α→A)
        # - Accented i mappings (í→i, German doesn't use these)
        # - Space normalization (various Unicode spaces → regular space)
        # - OCR artifacts removal
        text = text.translate(translation_map)

        return text

    # Apply normalization to all texts with progress bar
    texts = df[input_column].to_list()
    normalized_texts = [
        normalize_text(text) for text in tqdm(texts, desc="Normalizing Unicode", leave=False)
    ]

    df = df.with_columns([pl.Series(output_column, normalized_texts)])

    logger.info(f"Unicode normalized for {len(df):,} rows")
    return df


def remove_diacritics(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Remove diacritics from text using unidecode.

    Converts accented characters to their ASCII equivalents:
    - ä → a, ö → o, ü → u
    - é → e, à → a, etc.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_no_diacritics)

    Returns:
        DataFrame with diacritics removed

    Example:
        >>> df = remove_diacritics(df)
        >>> # "Münchner Straße" → "Munchner Strasse"

    Note:
        Requires unidecode package: pip install unidecode
    """
    try:
        from unidecode import unidecode
    except ImportError:
        raise ImportError(
            "unidecode is required for diacritic removal. Install with: pip install unidecode"
        )

    if output_column is None:
        output_column = f"{input_column}_no_diacritics"

    logger.info(f"Removing diacritics: {input_column} → {output_column}")

    # Apply unidecode to each text
    texts = df[input_column].to_list()
    processed = [unidecode(str(text)) if text else "" for text in texts]

    df = df.with_columns([pl.Series(output_column, processed)])

    logger.info(f"Removed diacritics from {len(df):,} rows")
    return df


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
    import torch
    from tqdm import tqdm
    from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

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


def simple(
    df: pl.DataFrame,
    input_column: str = "text",
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
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_simple)

    Returns:
        DataFrame with normalized text column
    """
    if output_column is None:
        output_column = f"{input_column}_simple"

    logger.info(f"Normalizing historical characters: {input_column} → {output_column}")

    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all("ẞ", "SS")
            .str.replace_all("ß", "ss")
            .str.replace_all("ſs", "ss")
            .str.replace_all("ſ", "s")
            .alias(output_column)
        ]
    )

    logger.info(f"Normalized {len(df):,} rows")
    return df


def normalize_long_s(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    mode: str = "simple",
) -> pl.DataFrame:
    """
    Normalize historical German long s (ſ) character.

    The long s (ſ, U+017F) was mandatory in 1910-1920 German Fraktur typography.
    This function provides three normalization strategies depending on your use case.

    **Three normalization modes:**

    **1. "simple" (default, recommended for NLP):**
    - Replaces all ſ → s unconditionally
    - Fast, deterministic, perfect for search/NLP/topic modeling
    - Example: "Hauſe" → "Hause", "faſten" → "fasten"
    - Use when: Text analysis, embeddings, search indices

    **2. "context-aware" (linguistic rules):**
    - Applies simplified linguistic rules for ſ/s distinction
    - Considers position (word-final), following characters (t, p, ch), vowels
    - More historically accurate but slower
    - Example: "faſten" → "fasten" (context rules)
    - Use when: Digital scholarly editions, reading texts

    **3. "preserve" (archival):**
    - No normalization, keeps original ſ character
    - For philological analysis, digital archives
    - Example: "Hauſe" → "Hauſe" (unchanged)
    - Use when: Historical text preservation, manuscript studies

    **Common OCR issue:** ſ frequently misread as 'f' in Fraktur OCR
    If you see "Haufe" instead of "Hauſe", the OCR misrecognized ſ as f.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_long_s)
        mode: Normalization strategy - "simple", "context-aware", or "preserve"

    Returns:
        DataFrame with long s normalized

    Example:
        >>> # Simple mode (recommended for NLP)
        >>> df = normalize_long_s(df, mode="simple")
        >>> # "Hauſe" → "Hause"
        >>>
        >>> # Context-aware mode (scholarly editions)
        >>> df = normalize_long_s(df, mode="context-aware")
        >>> # Applies linguistic rules for ſ/s
        >>>
        >>> # Preserve mode (archival)
        >>> df = normalize_long_s(df, mode="preserve")
        >>> # "Hauſe" → "Hauſe" (unchanged)

    Note:
        Based on historical typography rules for 1910-1920 German Fraktur.
        For most NLP applications, "simple" mode is sufficient and recommended.
    """
    import re

    if output_column is None:
        output_column = f"{input_column}_long_s"

    logger.info(f"Normalizing long s ({mode}): {input_column} → {output_column}")

    if mode == "preserve":
        # No transformation, just copy
        df = df.with_columns([pl.col(input_column).alias(output_column)])
        logger.info(f"Preserved long s for {len(df):,} rows (archival mode)")
        return df

    elif mode == "simple":
        # Simple replacement: all ſ → s
        df = df.with_columns(
            [pl.col(input_column).str.replace_all("ſ", "s").alias(output_column)]  # U+017F
        )
        logger.info(f"Normalized long s (simple) for {len(df):,} rows")
        return df

    elif mode == "context-aware":
        # Context-aware normalization using linguistic rules
        def normalize_contextual(text: str) -> str:
            if not text or "ſ" not in text:
                return text

            # Simplified linguistic rules for ſ/s distinction
            # Full implementation would require:
            # 1. Syllable boundary detection
            # 2. Compound word analysis
            # 3. Historical German lexicon lookup

            # Rule 1: ſ before t, p, ch → s (common patterns: faſten, ſprechen)
            text = re.sub(r"ſ([tpc])", r"s\1", text)

            # Rule 2: ſ at word end → s (should be round s in proper typography)
            text = re.sub(r"ſ\b", "s", text)

            # Rule 3: ſ before vowels → s
            text = re.sub(r"ſ([aeiouäöüAEIOUÄÖÜ])", r"s\1", text)

            # Rule 4: ſſ → ss (geminate long s: Waſſer → Wasser)
            text = text.replace("ſſ", "ss")

            # Rule 5: ſs → ss (long s + round s: also becomes ss)
            text = text.replace("ſs", "ss")

            # NOTE: No catch-all rule here!
            # Any remaining ſ characters might actually be OCR misreadings of 'f'
            # These should be manually reviewed or handled separately

            return text

        # Apply context-aware normalization
        texts = df[input_column].to_list()
        normalized_texts = [
            normalize_contextual(text)
            for text in tqdm(texts, desc="Normalizing long s (context-aware)", leave=False)
        ]

        df = df.with_columns([pl.Series(output_column, normalized_texts)])
        logger.info(f"Normalized long s (context-aware) for {len(df):,} rows")
        logger.info("Note: Remaining ſ characters may be OCR errors (misread 'f')")
        return df

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simple', 'context-aware', or 'preserve'")


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

    try:
        import torch
        from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except ImportError:
        raise ImportError(
            "Transnormer normalization requires 'transformers' and 'torch'.\n"
            "Install with: pip install transformers torch"
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

    # Setup cache directory
    if cache_dir is None:
        from newspaper_explorer.config.base import get_config

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
