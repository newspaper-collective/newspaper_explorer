"""
Linguistic processing operations.

Provides advanced linguistic processing methods:
- Dehyphenation (line-break hyphen removal)
- Line-level dehyphenation (preserves line structure)
- Lemmatization (spaCy and GermaLemma)
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import polars as pl

# Set spaCy data directory to avoid filling up home directory
# Use project's .cache directory for model downloads
_project_root = Path(__file__).parent.parent.parent.parent
os.environ["SPACY_DATA"] = str(_project_root / ".cache" / "spacy")

logger = logging.getLogger(__name__)


def dehyphenate(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
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
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
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
        raise ImportError(
            "pyphen is required for dehyphenation. " "Install with: pip install pyphen"
        )

    if input_column is None:
        input_column = text_column
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


def dehyphenate_lines(
    df: pl.DataFrame,
    text_column: str = "text",
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
        text_column: Column containing text (default: "text")
        text_block_id_column: Column with text block IDs (default: "text_block_id")
        line_id_column: Column with line IDs (default: "line_id")
        x_column: Column with x coordinate (default: "x")
        y_column: Column with y coordinate (default: "y")
        width_column: Column with width (default: "width")
        output_column: Name for output column (default: {text_column}_dehyphen)
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
        raise ImportError(
            "pyphen is required for dehyphenation. " "Install with: pip install pyphen"
        )

    if output_column is None:
        output_column = f"{text_column}_dehyphen"

    logger.info(f"Dehyphenating lines: {text_column} → {output_column}")

    # Validate required columns
    required_cols = [
        text_column,
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


def lemmatize_spacy(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    model: str = "de_core_news_sm",
    batch_size: int = 1000,
) -> pl.DataFrame:
    """
    Lemmatize German text using spaCy (FAST).

    Much faster than GermaLemma (100x) and context-aware.
    Uses part-of-speech information for better lemmatization.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_lemma)
        model: spaCy model to use (default: de_core_news_sm)
        batch_size: Batch size for processing (default: 1000)

    Returns:
        DataFrame with lemmatized text column

    Raises:
        ImportError: If spaCy is not installed
        OSError: If spaCy model is not downloaded

    Example:
        >>> # First download model: python -m spacy download de_core_news_sm
        >>> df = lemmatize_spacy(df, batch_size=5000)
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy is required for lemmatization. " "Install with: pip install -e '.[nlp]'"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing with spaCy: {input_column} → {output_column}")

    try:
        nlp = spacy.load(model, disable=["ner", "parser"])  # Faster: only need lemmatizer
    except OSError:
        logger.error(f"spaCy model '{model}' not found!")
        logger.error(f"Download it with: python -m spacy download {model}")
        raise

    from tqdm import tqdm

    texts = df[input_column].to_list()
    lemmatized_texts = []

    logger.info(f"Processing {len(texts):,} texts in batches of {batch_size}")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="spaCy lemmatization"):
        batch = texts[i : i + batch_size]

        # Process batch
        for doc in nlp.pipe(batch, batch_size=batch_size):
            lemmas = [token.lemma_ for token in doc]
            lemmatized_texts.append(" ".join(lemmas))

    df = df.with_columns([pl.Series(name=output_column, values=lemmatized_texts)])

    logger.info(f"Lemmatized {len(df):,} rows with spaCy")
    return df


def lemmatize_germalemma(
    df: pl.DataFrame,
    text_column: str = "text",
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
) -> pl.DataFrame:
    """
    Lemmatize German text using GermaLemma.

    Args:
        df: Input DataFrame
        text_column: Default column containing text (for backward compatibility)
        input_column: Column to process (default: text_column)
        output_column: Name for output column (default: {input_column}_lemma)

    Returns:
        DataFrame with lemmatized text column

    Raises:
        ImportError: If germalemma is not installed
    """
    try:
        from germalemma import GermaLemma
    except ImportError:
        raise ImportError(
            "germalemma is required for lemmatization. " "Install with: pip install -e '.[nlp]'"
        )

    if input_column is None:
        input_column = text_column
    if output_column is None:
        output_column = f"{input_column}_lemma"

    logger.info(f"Lemmatizing text: {input_column} → {output_column}")
    logger.warning("Lemmatization is slow and may take considerable time!")

    lemmatizer = GermaLemma()

    texts = df[input_column].to_list()
    lemmatized_texts = []

    for i, text in enumerate(texts):
        if text:
            tokens = str(text).split()
            # Assume NOUN for all tokens (simplification)
            lemmas = [lemmatizer.find_lemma(token, "NOUN") for token in tokens]
            lemmatized_texts.append(" ".join(lemmas))
        else:
            lemmatized_texts.append("")

        if (i + 1) % 10000 == 0:
            logger.info(f"Lemmatized {i + 1:,} / {len(texts):,} texts")

    df = df.with_columns([pl.Series(output_column, lemmatized_texts)])

    logger.info(f"Lemmatized {len(df):,} rows")
    return df
