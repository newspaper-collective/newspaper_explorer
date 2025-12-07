"""
Text normalization methods.

Provides comprehensive normalization for historical German newspaper texts:
- Unicode normalization (NFC with ftfy encoding repair, character translation, control chars)
- Historical German character mapping (ſ→s, ß→ss)
- Transnormer neural normalization (transformer-based)
- DTA-CAB API normalization (web service)
"""  # noqa: RUF002

import logging
import re
from typing import Optional, Union

import ftfy
import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Character translation map for Unicode normalization
# Handles OCR errors, confusables, spaces, and artifacts
# NOTE: Hyphen/dash normalization is handled separately by normalize_hyphens()

UNICODE_TRANSLATION: dict[int, Optional[Union[str, int]]] = {
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
    0x0410: "A",  # А → A (Cyrillic A)  # noqa: RUF003
    0x0412: "B",  # В → B (Cyrillic Ve)  # noqa: RUF003
    0x0415: "E",  # Е → E (Cyrillic Ie)  # noqa: RUF003
    0x041A: "K",  # К → K (Cyrillic Ka)  # noqa: RUF003
    0x041C: "M",  # М → M (Cyrillic Em)  # noqa: RUF003
    0x041D: "H",  # Н → H (Cyrillic En)  # noqa: RUF003
    0x041E: "O",  # О → O (Cyrillic O)  # noqa: RUF003
    0x0420: "P",  # Р → P (Cyrillic Er)  # noqa: RUF003
    0x0421: "C",  # С → C (Cyrillic Es)  # noqa: RUF003
    0x0422: "T",  # Т → T (Cyrillic Te)  # noqa: RUF003
    0x0425: "X",  # Х → X (Cyrillic Kha)  # noqa: RUF003
    0x0430: "a",  # а → a (Cyrillic a)  # noqa: RUF003
    0x0435: "e",  # е → e (Cyrillic ie)  # noqa: RUF003
    0x043E: "o",  # о → o (Cyrillic o)  # noqa: RUF003
    0x0440: "p",  # р → p (Cyrillic er)  # noqa: RUF003
    0x0441: "c",  # с → c (Cyrillic es)  # noqa: RUF003
    0x0443: "y",  # у → y (Cyrillic u)  # noqa: RUF003
    0x0445: "x",  # х → x (Cyrillic kha)  # noqa: RUF003
    0x0455: "s",  # ѕ → s (Cyrillic dze)  # noqa: RUF003
    0x0456: "i",  # і → i (Cyrillic byelorussian-ukrainian i)  # noqa: RUF003
    0x0458: "j",  # ј → j (Cyrillic je)  # noqa: RUF003
    0x04CF: "l",  # ӏ → l (Cyrillic palochka)  # noqa: RUF003
    0x0405: "S",  # Ѕ → S (Cyrillic Dze)  # noqa: RUF003
    0x0406: "I",  # І → I (Cyrillic Byelorussian-Ukrainian I)  # noqa: RUF003
    # Greek confusables (OCR errors - looks like Latin but isn't)
    0x0391: "A",  # Α → A (Greek Alpha)  # noqa: RUF003
    0x0392: "B",  # Β → B (Greek Beta)  # noqa: RUF003
    0x0395: "E",  # Ε → E (Greek Epsilon)  # noqa: RUF003
    0x0396: "Z",  # Ζ → Z (Greek Zeta)  # noqa: RUF003
    0x0397: "H",  # Η → H (Greek Eta)  # noqa: RUF003
    0x0399: "I",  # Ι → I (Greek Iota)  # noqa: RUF003
    0x039A: "K",  # Κ → K (Greek Kappa)  # noqa: RUF003
    0x039C: "M",  # Μ → M (Greek Mu)  # noqa: RUF003
    0x039D: "N",  # Ν → N (Greek Nu)  # noqa: RUF003
    0x039F: "O",  # Ο → O (Greek Omicron)  # noqa: RUF003
    0x03A1: "P",  # Ρ → P (Greek Rho)  # noqa: RUF003
    0x03A4: "T",  # Τ → T (Greek Tau)  # noqa: RUF003
    0x03A7: "X",  # Χ → X (Greek Chi)  # noqa: RUF003
    # Accented i (German doesn't use these - OCR errors)
    0x00ED: "i",  # í → i (acute)
    0x00EC: "i",  # ì → i (grave)
    0x00EE: "i",  # î → i (circumflex)
    0x00EF: "i",  # ï → i (diaeresis)
    0x0129: "i",  # ĩ → i (tilde)
    0x012B: "i",  # ī → i (macron)
    0x012D: "i",  # ĭ → i (breve)
    0x012F: "i",  # į → i (ogonek)
    0x0131: "i",  # ı → i (dotless)  # noqa: RUF003
    0x00CD: "I",  # Í → I (uppercase acute)
    0x00CC: "I",  # Ì → I (uppercase grave)
    0x00CE: "I",  # Î → I (uppercase circumflex)
    0x00CF: "I",  # Ï → I (uppercase diaeresis)
    0x0128: "I",  # Ĩ → I (uppercase tilde)
    0x012A: "I",  # Ī → I (uppercase macron)
    0x012C: "I",  # Ĭ → I (uppercase breve)
    0x012E: "I",  # Į → I (uppercase ogonek)
}


# =============================================================================
# HYPHEN/DASH NORMALIZATION
# =============================================================================


def normalize_hyphens(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    *,
    mode: str = "unify",
) -> pl.DataFrame:
    """
    Normalize various hyphen and dash characters.

    Historical German newspapers (especially Fraktur OCR) use various hyphen-like
    characters that should be normalized for consistent processing.

    **Three modes available:**

    **1. "unify" (default, recommended for NLP):**
    - Converts ALL hyphen/dash variants to regular hyphen (-)
    - Good for: topic modeling, embeddings, search, dehyphenation
    - Example: "1914–1918" → "1914-1918", "Nachrichten⸗Teil" → "Nachrichten-Teil"

    **2. "conservative":**
    - Only normalizes OCR artifacts (double hyphen ⸗, non-breaking hyphen)
    - Preserves semantic dashes (en dash for ranges, em dash for emphasis)
    - Good for: historical analysis, quotations, scholarly editions
    - Example: "1914–1918" stays, "Nachrichten⸗Teil" → "Nachrichten-Teil"

    **3. "soft_only":**
    - Only removes soft hyphens (invisible line-break hints)
    - Minimal intervention, preserves all visible characters
    - Good for: when you want to preserve original typography

    **Characters handled:**

    | Character | Unicode | Name | unify | conservative | soft_only |
    |-----------|---------|------|-------|--------------|----------|
    | ⸗ | U+2E17 | Double hyphen | → - | → - | kept |
    | ‐ | U+2010 | Hyphen | → - | → - | kept |
    | ‑ | U+2011 | Non-breaking hyphen | → - | → - | kept |
    | – | U+2013 | En dash | → - | kept | kept |
    | — | U+2014 | Em dash | → - | kept | kept |
    | − | U+2212 | Minus sign | → - | kept | kept |
    | ­ | U+00AD | Soft hyphen | removed | removed | removed |

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_hyphens)
        mode: Normalization mode - "unify", "conservative", or "soft_only"

    Returns:
        DataFrame with normalized hyphens

    Example:
        >>> # For NLP/dehyphenation (recommended)
        >>> df = normalize_hyphens(df, mode="unify")
        >>> # "Nachrichten⸗Teil" → "Nachrichten-Teil"
        >>> # "1914–1918" → "1914-1918"
        >>>
        >>> # For historical preservation
        >>> df = normalize_hyphens(df, mode="conservative")
        >>> # "Nachrichten⸗Teil" → "Nachrichten-Teil"
        >>> # "1914–1918" → "1914–1918" (en dash preserved)

    Note:
        Run this BEFORE dehyphenation so that all hyphen variants are
        recognized by the dehyphenation regex patterns.
    """  # noqa: RUF002
    if output_column is None:
        output_column = f"{input_column}_hyphens"

    logger.info(f"Normalizing hyphens ({mode}): {input_column} → {output_column}")

    # Build translation map based on mode
    if mode == "unify":
        # All hyphens/dashes → regular hyphen
        translation_map: dict[int, Optional[str]] = {
            0x00AD: None,  # soft hyphen → removed
            0x2E17: "-",  # U+2E17 double hyphen → hyphen (Fraktur OCR)
            0x2010: "-",  # U+2010 hyphen → hyphen
            0x2011: "-",  # U+2011 non-breaking hyphen → hyphen
            0x2013: "-",  # U+2013 en dash → hyphen
            0x2014: "-",  # U+2014 em dash → hyphen
            0x2212: "-",  # U+2212 minus sign → hyphen
        }
    elif mode == "conservative":
        # Only OCR artifacts, preserve semantic dashes
        translation_map = {
            0x00AD: None,  # soft hyphen → removed
            0x2E17: "-",  # U+2E17 double hyphen → hyphen (Fraktur OCR artifact)
            0x2010: "-",  # U+2010 hyphen → hyphen
            0x2011: "-",  # U+2011 non-breaking hyphen → hyphen
            # Keep: en dash (U+2013), em dash (U+2014), minus (U+2212)
        }
    elif mode == "soft_only":
        # Minimal: only remove soft hyphens
        translation_map = {
            0x00AD: None,  # soft hyphen → removed
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'unify', 'conservative', or 'soft_only'")

    # Apply translation using Polars (fast, vectorized)
    # We need to use map_elements for the translation
    def translate_hyphens(text: str) -> str:
        if not text:
            return text
        return text.translate(translation_map)

    df = df.with_columns(
        [
            pl.col(input_column)
            .map_elements(translate_hyphens, return_dtype=pl.Utf8)
            .alias(output_column)
        ]
    )

    logger.info(f"Hyphen normalization ({mode}) applied to {len(df):,} rows")
    return df


# =============================================================================
# UNICODE NORMALIZATION
# =============================================================================


def normalize_unicode(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Normalize Unicode characters for OCR text.

    Recommended as the FIRST STEP in preprocessing pipelines.
    Handles common OCR issues: encoding errors, confusables, spaces, artifacts.

    **NOTE: Hyphen/dash normalization is handled separately by normalize_hyphens().**
    Use normalize_hyphens() before dehyphenation for best results.

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

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_unicode)

    Returns:
        DataFrame with normalized Unicode text

    Example:
        >>> df = normalize_unicode(df)
        >>> # „Quoted text" → "\"Quoted text\""
        >>> # "schÃ¶n" → "schön" (mojibake fixed by ftfy)
        >>> # "Теxt with Суrilliс" → "Text with Cyrillic"
        >>> # "ﬁ ﬂ ﬀ" → "fi fl ff" (ligatures)
        >>>
        >>> # For hyphen normalization, use normalize_hyphens() separately:
        >>> df = normalize_hyphens(df, mode="unify")
        >>> # "1914–1918" → "1914-1918" (en dash → hyphen)
    """  # noqa: RUF002
    if output_column is None:
        output_column = f"{input_column}_unicode"

    logger.info(f"Normalizing Unicode: {input_column} → {output_column}")

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
        # - Cyrillic/Greek confusables (А→A, о→o, Α→A) # noqa: RUF003
        # - Accented i mappings (í→i, German doesn't use these)
        # - Space normalization (various Unicode spaces → regular space)
        # - OCR artifacts removal
        # NOTE: Hyphen/dash normalization is handled by normalize_hyphens()

        return text.translate(UNICODE_TRANSLATION)

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
            return re.sub(r"[ \t]*\n[ \t]*", "\n", text).strip()

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

    The long s (ſ, U+017F, "langes s", "Schaft-s") is a positional allograph of the
    letter s used in Fraktur and other broken scripts, as well as in Kurrent handwriting.
    It was standard in German typography until the 1941 Normalschrifterlass.

    **Historical Background:**

    The distinction between long s (ſ) and round s (Schluss-s, Auslaut-s) follows rules
    codified at the 1901 Orthographic Conference. The long s appears at syllable onsets
    (Anlaut) and within syllables (Inlaut), while the round s appears at syllable endings
    (Auslaut). This distinction aids readability of compound words - e.g., distinguishing
    "Wachſtube" (guard room) from "Wachstube" (wax tube).

    The ligatures ſʒ (ſz) and ſs are considered the origin of the German Eszett (ß).
    In modern Antiqua, ſs was typically rendered as ß until 1901, after which ß became
    standard in both Fraktur and Antiqua typography.

    **Two normalization modes:**

    **1. "simple" (default, recommended for NLP):**
    - Replaces all ſ → s unconditionally
    - Fast, deterministic, perfect for search/NLP/topic modeling
    - Example: "Hauſe" → "Hause", "faſten" → "fasten"
    - Use when: Text analysis, embeddings, search indices

    **2. "context-aware" (1901 orthographic rules):**
    - Implements the historical rules from the 1901 Orthographic Conference
    - Handles syllable positions, compound words, and special combinations
    - More historically accurate but slower
    - Example: "Wachſtube" → "Wachstube" (preserves morpheme boundaries)
    - Use when: Digital scholarly editions, reading texts

    **1901 Orthographic Rules (context-aware mode):**

    Round s (Schluss-s) is used:
    - At word end: das Haus, der Kosmos, des Bundes
    - At end of prefixes/first parts of compounds (Fugen-s): Liebesbrief, Arbeitsamt,
      Haustür, Wirtsstube, Aussicht
    - Before consonant-initial suffixes (-lein, -chen, -bar, -heit, -tum):
      Mäuschen, Weisheit, Wachstum
    - At syllable end before k, m, n, w, d: Dresden, Oswald, Schleſwig → Schleswig

    Long s (ſ) is used:
    - At syllable onset (Anlaut): ſauſen, einſpielen
    - Within syllables before vowels (Inlaut): Maſuren, Hauſe
    - In the combinations ſch, ſt, ſp (not from compounds): Weſpe, faſten, Buſch
    - In digraphs ſſ: Waſſer, Biſſen
    - Before l, n, r when e is elided: unſre, Pilſner, Wechſler
    - Before apostrophe: ich laſſ'

    **Common OCR issues:**
    - ſ frequently misread as 'f' in Fraktur OCR (Haufe instead of Hauſe)
    - Round s sometimes misread as 'ſ' or vice versa
    - The context-aware mode helps identify potential OCR errors

    Args:
        df: Input DataFrame
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: {input_column}_long_s)
        mode: Normalization strategy - "simple" or "context-aware"

    Returns:
        DataFrame with long s normalized

    Example:
        >>> # Simple mode (recommended for NLP)
        >>> df = normalize_long_s(df, mode="simple")
        >>> # "Hauſe" → "Hause"
        >>>
        >>> # Context-aware mode (1901 rules)
        >>> df = normalize_long_s(df, mode="context-aware")
        >>> # "Wirtſſtube" → "Wirtsstube" (compound word)
        >>> # "Waſſer" → "Wasser" (digraph)

    Note:
        Based on historical typography rules from the 1901 Orthographic Conference.
        For most NLP applications, "simple" mode is sufficient and recommended.
        The context-aware mode implements a best-effort approximation of the historical
        rules without full syllable/morpheme analysis, which would require a lexicon.

    References:
        - German Wikipedia: "Langes s", "Schluss-s"
        - 1901 Orthographic Conference rules
        - Duden historical editions (1880, 1915, 1941)
    """  # noqa: RUF002

    if output_column is None:
        output_column = f"{input_column}_long_s"

    logger.info(f"Normalizing long s ({mode}): {input_column} → {output_column}")

    if mode == "simple":
        # Simple replacement: all long s (U+017F) to round s
        df = df.with_columns(
            [
                pl.col(input_column).str.replace_all("ſ", "s").alias(output_column)  # noqa: RUF001
            ]
        )
        logger.info(f"Normalized long s (simple) for {len(df):,} rows")
        return df

    if mode == "context-aware":
        # Context-aware normalization implementing 1901 Orthographic Conference rules
        # Reference: German Wikipedia "Langes s", "Schluss-s"

        def normalize_contextual(text: str) -> str:
            """Apply 1901 orthographic rules for long s normalization."""
            if not text or "ſ" not in text:  # noqa: RUF001
                return text

            # ============================================================
            # 1901 ORTHOGRAPHIC RULES FOR LONG S (langes s) NORMALIZATION
            # ============================================================
            #
            # The rules distinguish between:
            # - Long s (langes s, Schaft-s): used at syllable onset (Anlaut)
            #   and within syllables (Inlaut)
            # - Round s (Schluss-s, Auslaut-s): used at syllable end (Auslaut)
            #
            # Since we're NORMALIZING (converting historical to modern), we
            # convert ALL long s to round s. The rules below help identify
            # edge cases and potential OCR errors.
            # ============================================================

            # ----- DIGRAPHS AND LIGATURES (process first) -----

            # Rule 1: Double long s (geminate): Wasser, Bissen
            # Historical: Waſſer → Modern: Wasser  # noqa: RUF003
            text = text.replace("ſſ", "ss")  # noqa: RUF001

            # Rule 2: Long s + round s ligature (origin of Eszett)
            # Historical: This combination (long s + round s) is rare in OCR
            # as it usually appears as the Eszett ligature in proper Fraktur.
            # When it does appear, normalize to ss (modern) or keep as-is for
            # texts following pre-1996 spelling (where some words used ss).
            text = text.replace("ſs", "ss")  # noqa: RUF001

            # ----- SPECIAL COMBINATIONS (long s is correct here) -----

            # Rule 3: ſch digraph (one sound): Busch, Esche, wunschen  # noqa: RUF003
            # The ſ is correct in historical text, normalize to sch # noqa: RUF003
            text = text.replace("ſch", "sch")  # noqa: RUF001

            # Rule 4: ſt combination (not from compounds): fasten, Ast  # noqa: RUF003
            # The ſ is correct when st is within a morpheme # noqa: RUF003
            # Note: In compounds like "Haus-tur", the s is round (Fugen-s)
            text = re.sub(r"ſt", "st", text)  # noqa: RUF001

            # Rule 5: ſp combination: Wespe, Knospe  # noqa: RUF003
            text = re.sub(r"ſp", "sp", text)  # noqa: RUF001

            # Rule 6: ſz combination (rare, usually appears as Eszett ligature) # noqa: RUF003
            # In words like "faszinierend", "Oszillograph"
            text = re.sub(r"ſz", "sz", text)  # noqa: RUF001

            # ----- POSITIONAL RULES -----

            # Rule 7: Long s at word end → round s
            # This is technically an OCR error (should be round s in original)
            # Words always end with round s: Haus, Kosmos, des
            text = re.sub(r"ſ\b", "s", text)  # noqa: RUF001

            # Rule 8: Long s before apostrophe → round s
            # Exception: "ich laſſ'" keeps long s before apostrophe in historical # noqa: RUF003
            # But for normalization, we convert to modern: "ich lass'"
            text = re.sub(r"ſ'", "s'", text)  # noqa: RUF001

            # Rule 9: Long s before consonants l, n, r (elided e)
            # Examples: unsre (from unsere), Pilsner, Wechsler
            # The long s is correct here historically, normalize to s
            text = re.sub(r"ſ([lnr])", r"s\1", text)  # noqa: RUF001

            # Rule 10: Long s before other consonants
            # In syllable-final position before k, m, n, w, d: Dresden, Oswald
            # Historically should be round s, but OCR may produce long s
            text = re.sub(r"ſ([bdfgkmnwBDFGKMNW])", r"s\1", text)  # noqa: RUF001

            # ----- VOWEL CONTEXTS -----

            # Rule 11: Long s before vowels (syllable onset)
            # This is CORRECT in historical text (Anlaut/Inlaut position)
            # Examples: sausen, Masuren, Hause (within syllable)
            # Normalize to modern s
            return re.sub(r"ſ([aeiouäöüAEIOUÄÖÜyY])", r"s\1", text)  # noqa: RUF001

            # NOTE: Any remaining long s characters are left as-is.
            # These may be:
            # - OCR errors (f misread as long s, or vice versa)
            # - Unusual contexts not covered by the rules above
            # Use simple mode if you want all long s converted unconditionally.

        # Apply context-aware normalization
        texts = df[input_column].to_list()
        normalized_texts = [
            normalize_contextual(text)
            for text in tqdm(texts, desc="Normalizing long s (context-aware)", leave=False)
        ]

        df = df.with_columns([pl.Series(output_column, normalized_texts)])
        logger.info(f"Normalized long s (context-aware) for {len(df):,} rows")
        return df

    raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'context-aware'")


# =============================================================================
# DEHYPHENATION
# =============================================================================

# Default German conjunctions to skip during dehyphenation
# These often appear after hyphens in contexts like "Ost- und West-"
DEFAULT_CONJUNCTIONS: set[str] = {"und", "oder", "bzw", "sowie", "als", "wie"}


def _dehyphenate_text(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    conjunctions: Optional[set[str]] = None,
) -> pl.DataFrame:
    """
    Remove line-break hyphens from aggregated text strings.

    Internal implementation for aggregated/block-level text.
    Use dehyphenate() as the public entry point.
    """
    if output_column is None:
        output_column = input_column

    if conjunctions is None:
        conjunctions = DEFAULT_CONJUNCTIONS

    # Lowercase for comparison
    conj_lower = {c.lower() for c in conjunctions}

    logger.info(f"Dehyphenating text: {input_column} → {output_column}")

    def process(text: str) -> str:
        if not text or "- " not in text:
            return text

        def replace(m: re.Match[str]) -> str:
            before, after = m.group(1), m.group(2)
            after_stripped = after.rstrip(",;.:")

            # Pure digits → likely range, skip
            if after_stripped.isdigit():
                return m.group(0)
            # Conjunction → skip
            if after_stripped.lower() in conj_lower:
                return m.group(0)
            # Capitalized → keep hyphen (Nord-Süd)
            if after[0].isupper() or after[0] in "ÄÖÜ":
                return f"{before}-{after}"
            return before + after

        # Unicode-aware pattern (includes German umlauts)
        return re.sub(r"([\w\u00C0-\u00FF]+)-\s+([\w\u00C0-\u00FF]+)", replace, text)

    result = df.with_columns(
        pl.col(input_column).map_elements(process, return_dtype=pl.Utf8).alias(output_column)
    )

    logger.info(f"Dehyphenated {len(df):,} rows")
    return result


def _dehyphenate_lines(
    df: pl.DataFrame,
    text_col: str = "text",
    block_col: str = "text_block_id",
    y_col: str = "y",
    output_col: Optional[str] = None,
    conjunctions: Optional[set[str]] = None,
) -> pl.DataFrame:
    """
    Remove line-break hyphens while preserving line-level structure.

    Internal implementation for line-level OCR data.
    Use dehyphenate() as the public entry point.
    """
    if output_col is None:
        output_col = text_col

    if conjunctions is None:
        conjunctions = DEFAULT_CONJUNCTIONS

    conj_list = list(conjunctions)

    logger.info(f"Dehyphenating lines: {text_col} → {output_col}")

    # Validate required columns
    required_cols = [text_col, block_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Unicode-aware word pattern (includes German umlauts)
    word_char = r"[\w\u00C0-\u00FF]"

    # Punctuation that can follow a word at end of line
    trailing_punct = r"[,;.!?:»«\"\')]?"

    result = (
        df.with_columns(
            pl.col(text_col).shift(-1).over(block_col, order_by=y_col).alias("_next"),
        )
        .with_columns(
            # Ends with single hyphen (not -- or —), preceded by word char
            pl.col(text_col)
            .str.strip_chars_end()
            .str.contains(rf"{word_char}-$")
            .fill_null(value=False)
            .alias("_ends_hyphen"),
            # Extract first word or hyphenated compound WITH trailing punctuation
            # e.g., "Süd-Bahn." from "Süd-Bahn. Der" or "papier," from "papier, sagte"
            pl.col("_next")
            .str.extract(rf"^\s*((?:{word_char}+-)*{word_char}+{trailing_punct})")
            .alias("_next_word_full"),
            # Also extract just the word part (without punctuation) for conjunction check
            pl.col("_next")
            .str.extract(rf"^\s*((?:{word_char}+-)*{word_char}+)")
            .alias("_next_word"),
        )
        .with_columns(
            pl.when(~pl.col("_ends_hyphen"))
            .then(pl.lit("none"))
            .when(pl.col("_next_word").is_null() | (pl.col("_next_word").str.len_chars() == 0))
            .then(pl.lit("none"))
            # Skip conjunctions (case-insensitive, strip trailing punct)
            .when(
                pl.col("_next_word").str.replace(r"[,;.:]$", "").str.to_lowercase().is_in(conj_list)
            )
            .then(pl.lit("skip"))
            # Next word starts with digit → keep hyphen (ranges: 20-30, compounds: Artikel-123)
            .when(pl.col("_next_word").str.contains(r"^\d"))
            .then(pl.lit("keep_hyphen"))
            # Both parts capitalized → keep hyphen (Nord-Süd)
            .when(pl.col("_next_word").str.contains(r"^[A-ZÄÖÜ]"))
            .then(pl.lit("keep_hyphen"))
            .otherwise(pl.lit("join"))
            .alias("_join_type"),
        )
        .with_columns(
            pl.col("_join_type")
            .shift(1)
            .over(block_col, order_by=y_col)
            .fill_null("none")
            .alias("_prev_join_type"),
        )
        .with_columns(
            pl.when(pl.col("_join_type") == "join")
            .then(pl.col(text_col).str.replace(r"-\s*$", "") + pl.col("_next_word_full"))
            .when(pl.col("_join_type") == "keep_hyphen")
            .then(pl.col(text_col).str.replace(r"-\s*$", "-") + pl.col("_next_word_full"))
            .when(pl.col("_prev_join_type").is_in(["join", "keep_hyphen"]))
            # Remove first word/compound WITH trailing punctuation and following whitespace
            .then(
                pl.col(text_col).str.replace(
                    rf"^\s*(?:{word_char}+-)*{word_char}+{trailing_punct}\s*", ""
                )
            )
            .otherwise(pl.col(text_col))
            .alias(output_col)
        )
        .drop(
            "_next",
            "_ends_hyphen",
            "_next_word",
            "_next_word_full",
            "_join_type",
            "_prev_join_type",
        )
    )

    logger.info(f"Dehyphenated {len(df):,} lines")
    return result


def dehyphenate(
    df: pl.DataFrame,
    input_column: str = "text",
    output_column: Optional[str] = None,
    conjunctions: Optional[set[str]] = None,
) -> pl.DataFrame:
    """
    Remove line-break hyphens from text.

    Auto-detects data structure and applies the appropriate method:
    - **Line-level data** (has text_block_id, y columns): Preserves line structure,
      moves word continuations to the previous line
    - **Aggregated text**: Uses regex-based pattern matching on "word- word" patterns

    Newspapers split words across line breaks with hyphens. This function joins them
    back together intelligently.

    **Smart heuristics to avoid incorrect joins:**
    - Skips conjunctions (und, oder, etc.) - "Ost- und West-" stays intact
    - Skips pure digits - "20- 30" is likely a range, not a word
    - Keeps hyphen for capitalized compounds - "Nord- Süd" → "Nord-Süd"
    - Joins lowercase continuations - "Zeitungs- papier" → "Zeitungspapier"

    Args:
        df: Input DataFrame (line-level or aggregated text)
        input_column: Column to process (default: "text")
        output_column: Name for output column (default: same as input_column)
        conjunctions: Set of words to skip (default: German conjunctions)

    Returns:
        DataFrame with dehyphenated text column

    Example:
        >>> # Line-break hyphen (HAS whitespace) → REMOVE
        >>> # "Zeitungs- papier" → "Zeitungspapier"
        >>>
        >>> # Compound word (NO whitespace) → KEEP
        >>> # "Nord-Süd-Konflikt" → "Nord-Süd-Konflikt"
        >>>
        >>> # Capitalized compound → KEEP HYPHEN
        >>> # "Nord- Süd" → "Nord-Süd"
        >>>
        >>> # Conjunction → SKIP
        >>> # "Ost- und West-Grenze" → "Ost- und West-Grenze"
        >>>
        >>> # Line-level example:
        >>> # Line 1: "Die Zeitungs-" → "Die Zeitungspapier"
        >>> # Line 2: "papier wird knapp" → "wird knapp"
    """
    # Auto-detect: use line-level if structure columns are present
    required_line_cols = ["text_block_id", "y"]

    if all(col in df.columns for col in required_line_cols):
        logger.info("Using line-level dehyphenation (preserves line structure)")
        return _dehyphenate_lines(
            df,
            text_col=input_column,
            output_col=output_column,
            conjunctions=conjunctions,
        )

    logger.info("Using text-based dehyphenation (aggregated text)")
    return _dehyphenate_text(
        df,
        input_column=input_column,
        output_column=output_column,
        conjunctions=conjunctions,
    )
