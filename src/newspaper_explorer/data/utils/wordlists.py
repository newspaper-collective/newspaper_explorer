"""
Utilities for generating and managing German wordlists.

Provides functions to extract wordlists from various sources:
- spaCy German models
- Hunspell dictionaries
- Custom German frequency lists
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Constants
MIN_WORD_LENGTH = 2


def extract_spacy_vocab(
    model_name: str = "de_core_news_sm",
    output_path: Optional[str] = None,
    min_length: int = 2,
    *,
    lowercase: bool = True,
) -> set[str]:
    """
    Extract vocabulary from spaCy German model.

    SpaCy models contain vocabulary learned from training data.
    The 'de_core_news_sm' model has ~460,000 German word entries.

    Args:
        model_name: spaCy model to load (default: "de_core_news_sm")
        output_path: Optional path to save wordlist file
        min_length: Minimum word length (default: 2)
        lowercase: Convert to lowercase (default: True)

    Returns:
        Set of German words

    Example:
        >>> # Extract and save wordlist
        >>> vocab = extract_spacy_vocab(output_path="data/wordlist_spacy.txt")
        >>> print(f"Extracted {len(vocab):,} words")

        >>> # Just get the set without saving
        >>> vocab = extract_spacy_vocab()
    """
    try:
        import spacy
    except ImportError as e:
        raise ImportError(
            "spaCy not installed. Install with: pip install spacy\n"
            "Then download model: python -m spacy download de_core_news_sm"
        ) from e

    logger.info(f"Loading spaCy model: {model_name}")
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise OSError(
            f"spaCy model '{model_name}' not found.\n"
            f"Download with: python -m spacy download {model_name}"
        ) from e

    # Extract all vocabulary strings
    vocab_words = set()
    for word in nlp.vocab.strings:
        if not word:  # Skip empty
            continue
        if word.startswith("_"):  # Skip internal tokens
            continue
        if word.strip() != word or not word.strip():  # Skip whitespace-only
            continue

        # Must contain at least one letter
        if not any(c.isalpha() for c in word):
            continue

        # Skip if contains special characters (except hyphens, apostrophes, umlauts)
        # Allow: letters, hyphens, apostrophes, German umlauts
        allowed_special = {"-", "'", "ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"}
        if any(not (c.isalpha() or c in allowed_special) for c in word):
            continue

        # Skip if starts with punctuation
        if not word[0].isalpha():
            continue

        if len(word) < min_length:  # Skip very short
            continue

        # Add word (optionally lowercased)
        clean_word = word.lower() if lowercase else word
        vocab_words.add(clean_word)

    logger.info(f"Extracted {len(vocab_words):,} words from {model_name}")

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for word in sorted(vocab_words):
                f.write(f"{word}\n")

        logger.info(f"Saved wordlist to: {output_path}")

    return vocab_words


def load_wordlist(path: str, *, lowercase: bool = True) -> set[str]:
    """
    Load wordlist from file.

    Args:
        path: Path to wordlist file (one word per line)
        lowercase: Convert to lowercase (default: True)

    Returns:
        Set of words

    Example:
        >>> vocab = load_wordlist("data/wordlist_de.txt")
        >>> print(f"Loaded {len(vocab):,} words")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Wordlist not found: {path}")

    logger.info(f"Loading wordlist from: {path}")

    words = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word.lower() if lowercase else word)

    logger.info(f"Loaded {len(words):,} words from {path}")
    return words


def extract_hunspell_wordlist(
    output_path: Optional[str] = None,
    *,
    include_compounds: bool = True,
) -> set[str]:
    """
    Extract German wordlist from Hunspell dictionary (igerman98).

    Hunspell dictionaries contain word stems and affixes that can generate
    millions of word forms. The igerman98 dictionary is the most comprehensive
    German dictionary for spell checking.

    Installation:
        sudo apt-get install hunspell hunspell-de-de  # Debian/Ubuntu
        brew install hunspell                         # macOS

    Dictionary files are typically in:
        /usr/share/hunspell/de_DE.dic
        /usr/share/hunspell/de_DE.aff

    Args:
        output_path: Optional path to save wordlist file
        include_compounds: Generate compound words (default: True)
                          WARNING: Can generate 2+ million words, slower

    Returns:
        Set of German words

    Example:
        >>> # Extract all word forms (2+ million words, takes a few minutes)
        >>> vocab = extract_hunspell_wordlist(
        ...     output_path="data/wordlist_hunspell.txt",
        ...     include_compounds=True
        ... )

        >>> # Extract just dictionary words (faster, ~120k words)
        >>> vocab = extract_hunspell_wordlist(include_compounds=False)

    Note:
        Requires 'hunspell' package to be installed.
        If you get an error, install with your package manager.
    """
    try:
        import hunspell
    except ImportError as e:
        raise ImportError(
            "hunspell not installed. Install with:\n"
            "  pip install pyhunspell\n"
            "  sudo apt-get install hunspell hunspell-de-de  # Linux\n"
            "  brew install hunspell                         # macOS"
        ) from e

    logger.info("Loading Hunspell German dictionary (de_DE)")

    try:
        hobj = hunspell.HunSpell("/usr/share/hunspell/de_DE.dic", "/usr/share/hunspell/de_DE.aff")
    except (OSError, RuntimeError) as e:
        raise FileNotFoundError(
            f"Hunspell dictionary not found: {e}\nInstall with: sudo apt-get install hunspell-de-de"
        ) from e

    words = set()

    # Get all stems from dictionary
    logger.info("Extracting word stems from dictionary...")
    # Note: hunspell doesn't provide direct access to all stems
    # We need to read the .dic file directly
    dic_path = Path("/usr/share/hunspell/de_DE.dic")

    if not dic_path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {dic_path}")

    with dic_path.open(encoding="latin-1") as f:
        # First line is word count
        next(f)

        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Parse word/flags format
            word = line.split("/")[0] if "/" in line else line

            if word and len(word) >= MIN_WORD_LENGTH:
                words.add(word.lower())

                # If include_compounds, also generate inflected forms
                # This is slow but comprehensive
                if include_compounds:
                    # Try to generate suggestions (gives inflected forms)
                    suggestions = hobj.suggest(word)
                    for sugg in suggestions:
                        if sugg and len(sugg) >= MIN_WORD_LENGTH:
                            words.add(sugg.lower())

    logger.info(f"Extracted {len(words):,} words from Hunspell")

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for word in sorted(words):
                f.write(f"{word}\n")

        logger.info(f"Saved wordlist to: {output_path}")

    return words


def download_leipzig_wordlist(
    output_path: Optional[str] = None,
    *,
    year: int = 2021,
) -> set[str]:
    """
    Download German wordlist from Leipzig Corpora Collection.

    The Leipzig Corpora Collection provides frequency-based word lists
    extracted from large German text corpora (news, web, Wikipedia).

    Available at: https://wortschatz.uni-leipzig.de/en/download/German

    Downloads the frequency wordlist (most common ~100k words).

    Args:
        output_path: Optional path to save wordlist file
        year: Corpus year (default: 2021, latest available)

    Returns:
        Set of German words

    Example:
        >>> # Download from Leipzig Corpora
        >>> vocab = download_leipzig_wordlist(
        ...     output_path="data/wordlist_leipzig.txt"
        ... )

    Note:
        Downloads ~10MB tar file, extracts word frequencies.
        Requires internet connection.
    """
    import tarfile
    import tempfile
    from urllib.request import urlretrieve

    logger.info(f"Downloading Leipzig Corpora Collection (deu_news_{year})")

    # Leipzig Corpora URL pattern
    base_url = "https://downloads.wortschatz-leipzig.de/corpora"
    corpus_name = f"deu_news_{year}_100K"
    filename = f"{corpus_name}.tar.gz"
    url = f"{base_url}/{filename}"

    words = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpdir = Path(tmp_dir)
        tar_path = tmpdir / filename

        # Download
        logger.info(f"Downloading from: {url}")
        try:
            urlretrieve(url, tar_path)  # noqa: S310
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Failed to download Leipzig Corpora: {e}\n"
                f"URL: {url}\n"
                f"Check https://wortschatz.uni-leipzig.de/en/download/German for available corpora"
            ) from e

        # Extract
        logger.info("Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)  # noqa: S202

        # Read word list from extracted archive
        words_file = tmpdir / corpus_name / f"{corpus_name}-words.txt"

        if not words_file.exists():
            raise FileNotFoundError(f"Words file not found in archive: {words_file}")

        logger.info("Reading word frequencies...")
        with words_file.open(encoding="utf-8") as f:
            for line in f:
                # Format: rank\tword\tfrequency
                parts = line.strip().split("\t")
                if len(parts) >= MIN_WORD_LENGTH:
                    word = parts[1]
                    if word and len(word) >= MIN_WORD_LENGTH and word[0].isalpha():
                        words.add(word.lower())

    logger.info(f"Extracted {len(words):,} words from Leipzig Corpora")

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for word in sorted(words):
                f.write(f"{word}\n")

        logger.info(f"Saved wordlist to: {output_path}")

    return words


def download_dta_wordlist() -> set[str]:
    """
    Download German wordlist from Deutsches Textarchiv (DTA).

    The DTA is a corpus of historical German texts (1600-1900) providing
    authentic historical vocabulary and spellings essential for 1910-1920
    newspaper OCR quality assessment.

    Downloads the DTA token list which includes historical German variants.

    Available at: https://www.deutschestextarchiv.de/

    Args:
        output_path: Optional path to save wordlist file

    Returns:
        Set of German words (historical variants included)

    Example:
        >>> # Download DTA historical wordlist
        >>> vocab = download_dta_wordlist(
        ...     output_path="data/wordlist_dta.txt"
        ... )

    Note:
        This includes historical spellings like "Thal" (→ "Tal"),
        "eigenthümlich" (→ "eigentümlich"), essential for 1910-1920 text.
        Requires internet connection.
    """
    logger.info("Downloading DTA (Deutsches Textarchiv) wordlist")

    # DTA provides various resources
    # For a complete wordlist, we'd need to process their corpus
    # Alternative: Use DTA CAB web service token list

    # Note: This is a placeholder - actual DTA download would require
    # parsing their corpus or using their API
    logger.warning(
        "DTA wordlist download not yet implemented.\n"
        "To get historical German vocabulary:\n"
        "1. Visit: https://www.deutschestextarchiv.de/\n"
        "2. Download DTA corpus or use DTA CAB service\n"
        "3. Extract token frequencies\n"
        "\n"
        "Alternative: Use Leipzig Historical Corpora:\n"
        "  https://wortschatz.uni-leipzig.de/de/download/German\n"
        "  Look for 'deu-historical' corpora"
    )

    # For now, return empty set with instructions
    raise NotImplementedError(
        "DTA wordlist download requires manual corpus processing.\n"
        "Use Leipzig historical corpora as alternative."
    )


def download_german_wordlist(
    source: str = "spacy",
    output_path: str = "data/wordlist_de.txt",
    **kwargs: Any,
) -> Path:
    """
    Download/generate German wordlist from specified source.

    Args:
        source: Wordlist source
            - "spacy": Extract from spaCy de_core_news_sm (~244k words)
            - "spacy_lg": Extract from spaCy de_core_news_lg (~500k words)
            - "hunspell": Extract from Hunspell de_DE (~120k stems, or 2M+ with compounds)
            - "leipzig": Download Leipzig Corpora (~100k most common words)
            - "dta": Download DTA historical German (not yet implemented)
        output_path: Where to save wordlist
        **kwargs: Additional arguments passed to extraction function
            - For hunspell: include_compounds=True (default)
            - For leipzig: year=2021 (default)

    Returns:
        Path to saved wordlist

    Example:
        >>> # Fast: spaCy wordlist (244k words)
        >>> path = download_german_wordlist(source="spacy")

        >>> # Comprehensive: Hunspell with compounds (2M+ words, slow)
        >>> path = download_german_wordlist(
        ...     source="hunspell",
        ...     output_path="data/wordlist_hunspell.txt",
        ...     include_compounds=True
        ... )

        >>> # Frequency-based: Leipzig most common words
        >>> path = download_german_wordlist(
        ...     source="leipzig",
        ...     output_path="data/wordlist_leipzig.txt"
        ... )
    """
    output_path = Path(output_path)

    if source == "spacy":
        extract_spacy_vocab(
            model_name="de_core_news_sm",
            output_path=str(output_path),
        )
    elif source == "spacy_lg":
        extract_spacy_vocab(
            model_name="de_core_news_lg",
            output_path=str(output_path),
        )
    elif source == "hunspell":
        extract_hunspell_wordlist(output_path=str(output_path), **kwargs)
    elif source == "leipzig":
        download_leipzig_wordlist(output_path=str(output_path), **kwargs)
    elif source == "dta":
        download_dta_wordlist()
    else:
        raise ValueError(
            f"Unknown source: {source}\nAvailable: spacy, spacy_lg, hunspell, leipzig, dta"
        )

    return output_path


def get_wordlist_info(path: str) -> dict:
    """
    Get information about a wordlist file.

    Args:
        path: Path to wordlist file

    Returns:
        Dictionary with wordlist statistics

    Example:
        >>> info = get_wordlist_info("data/wordlist_de.txt")
        >>> print(f"Words: {info['word_count']:,}")
        >>> print(f"Avg length: {info['avg_length']:.1f}")
    """
    words = load_wordlist(path, lowercase=False)

    lengths = [len(w) for w in words]

    return {
        "word_count": len(words),
        "unique_count": len({w.lower() for w in words}),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
    }
