"""
Text normalization methods.

Provides comprehensive normalization for historical German newspaper texts:
- Unicode normalization (NFC with ftfy encoding repair, character translation, control chars)
- Historical German character mapping (ſ→s, ß→ss)
- Transnormer neural normalization (transformer-based)
- DTA-CAB API normalization (web service)
"""  # noqa: RUF002

import logging
from typing import Optional, Union

import ftfy
import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Character translation maps for Unicode normalization
# Two modes: conservative (default) and aggressive

# CONSERVATIVE: Only fix clear OCR errors, preserve semantic distinctions
UNICODE_TRANSLATION_CONSERVATIVE: dict[int, Optional[Union[str, int]]] = {
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
UNICODE_TRANSLATION_AGGRESSIVE: dict[int, Optional[Union[str, int]]] = {
    **UNICODE_TRANSLATION_CONSERVATIVE,
    # Add aggressive dash normalization
    0x2013: "-",  # en dash → hyphen
    0x2014: "-",  # em dash → hyphen
    0x2212: "-",  # minus sign → hyphen
}


def normalize_unicode(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    *,
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


def normalize_whitespace(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    *,
    keep_newlines: bool = False,
) -> pl.DataFrame:
    """
    Normalize whitespace characters in text.

    Two modes available:

    **Default mode (keep_newlines=False):**
    - Collapses ALL whitespace (spaces, tabs, newlines) to single space
    - Good for: aggregated text blocks, NLP tasks, topic modeling
    - Example: "Hello    world\\n\\tNext" → "Hello world Next"

    **Newline-preserving mode (keep_newlines=True):**
    - Collapses multiple spaces/tabs to single space
    - KEEPS newlines intact, removes spaces around them
    - Good for: line-by-line processing, preserving text structure
    - Example: "Hello    world\\n\\tNext" → "Hello world\\nNext"

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_whitespace)
        keep_newlines: If True, preserves newlines (default: False)

    Returns:
        DataFrame with normalized whitespace

    Example:
        >>> # Default: collapse all whitespace
        >>> df = normalize_whitespace(df)
        >>> # "Hello    world\\n\\ttab  " → "Hello world tab"
        >>>
        >>> # Preserve newlines
        >>> df = normalize_whitespace(df, keep_newlines=True)
        >>> # "Hello    world\\n\\ttab  " → "Hello world\\ntab"
    """
    if output_column is None:
        output_column = f"{input_column}_whitespace"

    mode = "preserve newlines" if keep_newlines else "collapse all"
    logger.info(f"Normalizing whitespace ({mode}): {input_column} → {output_column}")

    if keep_newlines:
        # Preserve newlines, collapse only spaces/tabs
        def normalize_with_newlines(text: str) -> str:
            if not text:
                return text
            # Normalize line breaks to \n
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Collapse multiple spaces/tabs to single space
            text = re.sub(r"[ \t]+", " ", text)
            # Remove spaces/tabs around newlines
            text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
            return text.strip()

        df = df.with_columns(
            [
                pl.col(input_column)
                .map_elements(normalize_with_newlines, return_dtype=pl.Utf8)
                .alias(output_column)
            ]
        )
    else:
        # Default: collapse all whitespace to single space
        df = df.with_columns(
            [
                pl.col(input_column)
                .str.replace_all(r"\s+", " ")  # All whitespace → single space
                .str.strip_chars()  # Remove leading/trailing
                .alias(output_column)
            ]
        )

    logger.info(f"Whitespace normalized for {len(df):,} rows")
    return df


def normalize_umlauts(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Normalize German umlauts and ß to two-letter representations.

    Converts:
    - ä → ae
    - ö → oe
    - ü → ue
    - Ä → Ae
    - Ö → Oe
    - Ü → Ue
    - ß → ss

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_umlaut_norm)

    Returns:
        DataFrame with normalized umlauts
    """
    if output_column is None:
        output_column = f"{input_column}_umlaut_norm"

    logger.info(f"Normalizing umlauts: {input_column} → {output_column}")

    df = df.with_columns(
        [
            pl.col(input_column)
            .str.replace_all(r"ä", "ae")
            .str.replace_all(r"ö", "oe")
            .str.replace_all(r"ü", "ue")
            .str.replace_all(r"Ä", "Ae")
            .str.replace_all(r"Ö", "Oe")
            .str.replace_all(r"Ü", "Ue")
            .str.replace_all(r"ß", "ss")
            .alias(output_column)
        ]
    )

    logger.info(f"Umlauts normalized for {len(df):,} rows")
    return df


def normalize_casing(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    *,
    mode: str = "lower",
) -> pl.DataFrame:
    """
    Normalize text casing.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_casing)
        mode: Casing mode - "lower", "upper", or "title" (default: "lower")

    Returns:
        DataFrame with normalized casing

    Example:
        >>> # Lowercase (default)
        >>> df = normalize_casing(df)
        >>> # "Hello World" → "hello world"
        >>>
        >>> # Uppercase
        >>> df = normalize_casing(df, mode="upper")
        >>> # "Hello World" → "HELLO WORLD"
        >>>
        >>> # Title case
        >>> df = normalize_casing(df, mode="title")
        >>> # "hello world" → "Hello World"
    """
    if output_column is None:
        output_column = f"{input_column}_casing"

    logger.info(f"Normalizing casing ({mode}): {input_column} → {output_column}")

    if mode == "lower":
        df = df.with_columns([pl.col(input_column).str.to_lowercase().alias(output_column)])
    elif mode == "upper":
        df = df.with_columns([pl.col(input_column).str.to_uppercase().alias(output_column)])
    elif mode == "title":
        df = df.with_columns([pl.col(input_column).str.to_titlecase().alias(output_column)])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'lower', 'upper', or 'title'")

    logger.info(f"Normalized casing for {len(df):,} rows")
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


def dehyphenate(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    language: str = "de_DE",
    validate: bool = True,
) -> pl.DataFrame:
    """
    Remove line-break hyphens from newspaper OCR text.

    Newspapers split words across line breaks with hyphens. After line
    aggregation, these appear as "word- word". This function joins them
    back together based on whitespace patterns.

    The key insight: Line-break hyphens are ALWAYS followed by whitespace,
    while compound words (Nord-Süd) have no space after the hyphen.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_dehyphen)
        language: Language code for pyphen (default: de_DE)
        validate: Use pyphen to validate syllable breaks (default: True).
                 Checks if the rejoined word would be hyphenated at that position.
                 If False, removes ALL hyphen+whitespace patterns (faster but less safe).

    Returns:
        DataFrame with dehyphenated text column

    Raises:
        ImportError: If pyphen is not installed

    Example:
        >>> # Line-break hyphen (HAS whitespace) → REMOVE
        >>> # "Zeitungs- papier" → "Zeitungspapier"
        >>>
        >>> # Compound word (NO whitespace) → KEEP
        >>> # "Nord-Süd-Konflikt" → "Nord-Süd-Konflikt"

    Note:
        With validate=True, pyphen checks if the split is a valid syllable
        boundary, reducing false positives from OCR errors. However, pyphen
        validates syllable breaks, not semantic meaning - it will approve
        joining "Nord- Süd" because that's a valid break point for "NordSüd".

        This is okay because real compound words in newspapers never have
        whitespace after hyphens - that pattern only appears from line breaks.
    """
    try:
        import pyphen
    except ImportError:
        raise ImportError("pyphen is required for dehyphenation. Install with: pip install pyphen")

    if output_column is None:
        output_column = f"{input_column}_dehyphen"

    logger.info(f"Dehyphenating text: {input_column} → {output_column}")

    dic = pyphen.Pyphen(lang=language) if validate else None

    if validate:
        logger.info("Using pyphen dictionary validation")
    else:
        logger.info("Using simple regex-based removal (no validation)")

    def dehyphenate_text(text: str) -> str:
        """Remove line-break hyphens from text."""
        if not text:
            return text

        import re

        def check_and_replace(match):
            """Check if hyphen is at a valid syllable break, if so remove it."""
            before = match.group(1)  # Word part before hyphen
            after = match.group(2)  # Word part after hyphen

            # Heuristic: Both parts capitalized → likely proper noun compound
            # Examples: "Nord- Süd", "Ost- West", "New- York"
            # These should stay separated even with whitespace
            if before and after and before[0].isupper() and after[0].isupper():
                return match.group(0)  # Keep hyphen and space

            if not validate or dic is None:
                # Simple mode: always join
                return before + after

            # Validation mode: check if it's a valid syllable break
            full_word = before + after
            positions = dic.positions(full_word)

            # If the hyphen position matches a valid syllable break, remove it
            if len(before) in positions:
                return full_word
            else:
                # Not a valid break point, keep hyphen and space
                return match.group(0)

        # Pattern: word-characters, hyphen, whitespace, word-characters
        # Matches both newlines and multiple spaces
        pattern = r"(\w+)-\s+(\w+)"
        text = re.sub(pattern, check_and_replace, text)

        return text

    df = df.with_columns(
        [
            pl.col(input_column)
            .map_elements(dehyphenate_text, return_dtype=pl.Utf8)
            .alias(output_column)
        ]
    )

    logger.info(f"Dehyphenated {len(df):,} rows")
    return df


def dehyphenate_auto(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    language: str = "de_DE",
    validate: bool = True,
) -> pl.DataFrame:
    """
    Auto-detect and apply the appropriate dehyphenation method.

    Chooses between line-level dehyphenation (dehyphenate_lines) and simple
    dehyphenation (dehyphenate) based on whether the DataFrame has the required
    columns for line-level processing.

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_dehyphen)
        language: Language code for pyphen (default: "de_DE")
        validate: Use pyphen to validate syllable breaks (default: True)

    Returns:
        DataFrame with dehyphenated text column

    Note:
        Uses dehyphenate_lines if columns text_block_id, line_id, x, y, width
        are present; otherwise falls back to simple dehyphenate.
    """
    required_cols = ["text_block_id", "line_id", "x", "y", "width"]

    if all(col in df.columns for col in required_cols):
        logger.info("Using line-level dehyphenation (preserves line structure)")
        return dehyphenate_lines(
            df,
            input_column=input_column,
            output_column=output_column,
            language=language,
            validate=validate,
        )
    else:
        logger.info("Using simple dehyphenation (line structure not available)")
        return dehyphenate(
            df,
            input_column=input_column,
            output_column=output_column,
            language=language,
            validate=validate,
        )


def dehyphenate_lines(
    df: pl.DataFrame,
    input_column: str = "text",
    text_block_id_column: str = "text_block_id",
    line_id_column: str = "line_id",
    x_column: str = "x",
    y_column: str = "y",
    width_column: str = "width",
    output_column: Optional[str] = None,
    language: str = "de_DE",
    validate: bool = True,
    max_y_distance: int = 100,
) -> pl.DataFrame:
    """
    Remove line-break hyphens while preserving line-level structure.

    This function is designed for line-level OCR data where words are split
    across consecutive lines with hyphens. Unlike dehyphenate() which works
    on aggregated text, this function:
    - Preserves the line-level DataFrame structure
    - Uses spatial coordinates to verify consecutive lines
    - Moves wrapped word parts to the correct line
    - Updates both lines in place

    Example:
        Line 1: "Die Zeitungs-" (x=100, y=200)
        Line 2: "papier wird knapp" (x=100, y=250)

        After dehyphenation:
        Line 1: "Die Zeitungspapier" (x=100, y=200)
        Line 2: "wird knapp" (x=100, y=250)

    Args:
        df: Input DataFrame with line-level OCR data
        input_column: Column containing text (default: "text")
        text_block_id_column: Column with text block IDs (default: "text_block_id")
        line_id_column: Column with line IDs (default: "line_id")
        x_column: Column with x coordinate (default: "x")
        y_column: Column with y coordinate (default: "y")
        width_column: Column with width (default: "width")
        output_column: Name for output column (default: {input_column}_dehyphen)
        language: Language code for pyphen (default: "de_DE")
        validate: Use pyphen to validate syllable breaks (default: True)
        max_y_distance: Maximum vertical distance to consider lines consecutive (default: 100)

    Returns:
        DataFrame with dehyphenated text, preserving line structure

    Note:
        Requires columns: text, text_block_id, line_id, x, y, width
        All other columns are preserved unchanged.
    """
    try:
        import pyphen
    except ImportError:
        raise ImportError("pyphen is required for dehyphenation. Install with: pip install pyphen")

    if output_column is None:
        output_column = f"{input_column}_dehyphen"

    logger.info(f"Dehyphenating lines: {input_column} → {output_column}")

    # Validate required columns
    required_cols = [
        input_column,
        text_block_id_column,
        line_id_column,
        x_column,
        y_column,
        width_column,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dic = pyphen.Pyphen(lang=language) if validate else None

    if validate:
        logger.info("Using pyphen dictionary validation")
    else:
        logger.info("Using simple pattern matching (no validation)")

    # Sort by text_block, then y coordinate (top to bottom)
    df = df.sort([text_block_id_column, y_column])

    # Create output column as copy of input
    df = df.with_columns([pl.col(text_column).alias(output_column)])

    # OPTIMIZATION: Pre-filter lines ending with hyphen to reduce search space
    # This dramatically reduces the number of iterations from 61M to ~100k-500k
    logger.info("Finding lines ending with hyphens...")
    hyphen_mask = df.select(
        pl.col(output_column).str.strip_chars_end().str.ends_with("-")
    ).to_series()

    hyphen_indices = [i for i, has_hyphen in enumerate(hyphen_mask) if has_hyphen]
    logger.info(
        f"Found {len(hyphen_indices):,} lines ending with hyphens (out of {len(df):,} total)"
    )

    if not hyphen_indices:
        logger.info("No line-break hyphens found")
        return df

    # OPTIMIZATION: Extract columns as numpy arrays for fast random access
    texts = df[output_column].to_list()
    blocks = df[text_block_id_column].to_list()
    y_coords = df[y_column].to_list()
    x_coords = df[x_column].to_list()
    widths = df[width_column].to_list()

    # Process only hyphenated lines
    modifications = []
    merge_examples = []  # Collect examples for review

    from tqdm import tqdm

    for i in tqdm(hyphen_indices, desc="Processing hyphenated lines", unit="line"):
        # Bounds check
        if i >= len(df) - 1:
            continue

        # Fast array access instead of df[i]
        current_block = blocks[i]
        next_block = blocks[i + 1]
        same_block = current_block == next_block

        # If not same block, check if we're at the last line of current block
        if not same_block:
            # Quick check: is there any line after i+1 with same block?
            has_more_in_block = any(
                blocks[j] == current_block for j in range(i + 2, min(i + 10, len(blocks)))
            )
            if has_more_in_block:
                continue

        # Check if lines are vertically close (consecutive)
        current_y = y_coords[i]
        next_y = y_coords[i + 1]
        y_distance = abs(next_y - current_y)
        if y_distance > max_y_distance:
            continue

        # ADDITIONAL CHECK: Verify next line is actually BELOW current line
        # (not above or at same level - this catches ordering issues)
        if next_y <= current_y:
            continue

        # ADDITIONAL CHECK: Verify horizontal alignment
        # Lines should start at similar x positions (within reason)
        current_x = x_coords[i]
        next_x = x_coords[i + 1]
        x_distance = abs(next_x - current_x)

        # Allow some horizontal variation but not too much
        # (columns, indentation, etc. should be roughly aligned)
        max_x_distance = 200  # pixels - adjust based on typical column width
        if x_distance > max_x_distance:
            continue

        # ADDITIONAL CHECK: Check if next line starts far to the right
        # (might be a different column or continuation mark)
        current_width = widths[i]
        # If next line starts after current line ends, it's likely a new section
        current_end_x = current_x + current_width
        if next_x > current_end_x + 50:  # 50px grace period
            continue

        # Get text (already verified to end with hyphen by pre-filter)
        current_text = texts[i]
        if not current_text:
            continue

        next_text = texts[i + 1]
        if not next_text or not next_text.strip():
            continue

        # Extract the word parts
        current_words = current_text.rstrip().split()
        if not current_words:
            continue

        last_word = current_words[-1]
        if not last_word.endswith("-"):
            continue

        # Get first word from next line
        next_words = next_text.strip().split()
        if not next_words:
            continue

        first_word = next_words[0]

        # Remove hyphen and join
        word_part1 = last_word[:-1]  # Remove trailing hyphen
        joined_word = word_part1 + first_word

        # VALIDATION CHECKS to prevent bad merges:

        # 1. Skip if first word is all caps (likely a heading/title)
        if first_word.isupper() and len(first_word) > 2:
            continue

        # 2. Skip if word parts are identical (repetition, not continuation)
        if word_part1.lower() == first_word.lower().rstrip(",-.:;!?"):
            continue

        # 3. Skip if capitalization suggests separate words
        # (lowercase + Capitalized = probably two words)
        if word_part1 and word_part1[-1].islower() and first_word and first_word[0].isupper():
            # Exception: if word_part1 is very short (1-2 chars), might be valid
            if len(word_part1) > 2:
                continue

        # Validate with pyphen if requested
        if validate and dic:
            # Check if this would be a valid hyphenation point
            hyphenated = dic.inserted(joined_word)
            # pyphen inserts "-" at valid break points
            # Check if our split point (word_part1) matches a valid break
            # Example: "Zeitungs" should appear before a "-" in "Zei-tungs-pa-pier"
            parts = hyphenated.split("-")
            # Build cumulative parts to find if word_part1 matches
            cumulative = ""
            valid_break = False
            for part in parts[:-1]:  # Don't check last part
                cumulative += part
                if cumulative == word_part1:
                    valid_break = True
                    break

            if not valid_break:
                # Not a valid syllable break, skip
                continue

        # Valid dehyphenation - prepare modifications
        # Modify current line: replace last word with joined word
        current_words[-1] = joined_word
        new_current_text = " ".join(current_words)

        # Modify next line: remove first word
        new_next_text = " ".join(next_words[1:]) if len(next_words) > 1 else ""

        modifications.append({"index": i, "text": new_current_text})
        modifications.append({"index": i + 1, "text": new_next_text})

        # Save example for review (first 100)
        if len(merge_examples) < 100:
            merge_examples.append(
                {
                    "line_index": i,
                    "word_part1": word_part1,
                    "word_part2": first_word,
                    "joined_word": joined_word,
                    "line1_before": current_text.strip(),
                    "line2_before": next_text.strip(),
                    "line1_after": new_current_text.strip(),
                    "line2_after": new_next_text.strip(),
                }
            )

    # Apply modifications
    if modifications:
        logger.info(f"Found {len(modifications) // 2:,} line-break hyphens to join")

        # Create a mapping of index to new text
        mod_map = {mod["index"]: mod["text"] for mod in modifications}

        # Apply modifications to the text list
        for idx, new_text in mod_map.items():
            texts[idx] = new_text

        # Update dataframe with modified texts
        df = df.with_columns([pl.Series(output_column, texts)])

        # Log sample merges for review
        if merge_examples:
            logger.info(f"\nSample merges (showing first {len(merge_examples)}):")
            for i, ex in enumerate(merge_examples[:10], 1):  # Show first 10 in log
                logger.info(f"\n  {i}. Line {ex['line_index']}:")
                logger.info(
                    f"     Merged: '{ex['word_part1']}-' + '{ex['word_part2']}' → '{ex['joined_word']}'"
                )
                logger.info(f"     Before: '{ex['line1_before']}'")
                logger.info(f"             '{ex['line2_before']}'")
                logger.info(f"     After:  '{ex['line1_after']}'")
                logger.info(f"             '{ex['line2_after']}'")
    else:
        logger.info("No line-break hyphens found to join")

    logger.info(f"Dehyphenated {len(df):,} lines")
    return df
