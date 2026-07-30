"""
Utilities for generating and managing German wordlists.

All wordlists are stored centrally in the configured wordlists directory
(default: data/wordlists/).

Provides functions to generate wordlists from various sources:
- spaCy German models (modern German, ~244k words)
- Leipzig Corpora Collection (frequency-based, ~155k words)
- Hunspell/igerman98 dictionary (modern + old orthography, ~170k words)
- DTA Deutsches Textarchiv (historical German)
"""

import logging
from pathlib import Path
import tarfile
import tempfile
from typing import Optional
from urllib.request import urlretrieve
import zipfile

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)

# Constants
MIN_WORD_LENGTH = 2
MIN_LEIPZIG_PARTS = 2

# Standard wordlist filenames
WORDLIST_FILENAMES: dict[str, str] = {
    "spacy": "wordlist_spacy_de.txt",
    "spacy_lg": "wordlist_spacy_de.txt",
    "leipzig": "wordlist_leipzig_de.txt",
    "hunspell": "wordlist_hunspell_de.txt",
    "dta": "wordlist_dta_de.txt",
}

# System paths where Hunspell dictionaries may be installed
HUNSPELL_SYSTEM_PATHS = [
    Path("/usr/share/hunspell/de_DE.dic"),
    Path("/usr/share/myspell/de_DE.dic"),
    Path("/usr/local/share/hunspell/de_DE.dic"),
]
HUNSPELL_1901_SYSTEM_PATHS = [
    Path("/usr/share/hunspell/de_DE-1901.dic"),
    Path("/usr/share/myspell/de_DE-1901.dic"),
]

# Download URLs
HUNSPELL_DIC_URL = (
    "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/de/index.dic"
)
HUNSPELL_1901_DIC_URL = (
    "https://raw.githubusercontent.com/elastic/hunspell/master/dicts/de_DE-1901/de-DE-1901.dic"
)
LEIPZIG_BASE_URL = "https://downloads.wortschatz-leipzig.de/corpora"
DTA_LEMMATIZED_BASE_URL = (
    "https://www.deutschestextarchiv.de/media/download/dtak/2020-10-23/lemmatized"
)
DTA_PERIOD_FILES = ["1800-1899.zip", "1900-1999.zip"]


def get_wordlists_dir() -> Path:
    """Get the central wordlists directory path."""
    return get_config().wordlists_dir


def get_wordlist_path(source: str) -> Path:
    """
    Get the canonical path for a wordlist by source name.

    Args:
        source: Wordlist source name (spacy, spacy_lg, leipzig, hunspell, dta)

    Returns:
        Absolute path to the wordlist file in the central wordlists directory

    Example:
        >>> path = get_wordlist_path("spacy")
        >>> print(path)  # .../data/wordlists/wordlist_spacy_de.txt
    """
    if source not in WORDLIST_FILENAMES:
        available = sorted({k for k in WORDLIST_FILENAMES if k != "spacy_lg"})
        raise ValueError(f"Unknown source: {source}. Available: {available}")
    return get_wordlists_dir() / WORDLIST_FILENAMES[source]


def _save_wordlist(words: set[str], output_path: Path) -> None:
    """Save a set of words to a file, one word per line, sorted."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(f"{word}\n")
    logger.info(f"Saved {len(words):,} words to: {output_path}")


def load_wordlist(path: str | Path, *, lowercase: bool = True) -> set[str]:
    """
    Load wordlist from file.

    Args:
        path: Path to wordlist file (one word per line)
        lowercase: Convert to lowercase (default: True)

    Returns:
        Set of words

    Example:
        >>> vocab = load_wordlist(get_wordlist_path("spacy"))
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


def extract_spacy_vocab(
    model_name: str = "de_core_news_sm",
    output_path: Optional[str | Path] = None,
    min_length: int = MIN_WORD_LENGTH,
    *,
    lowercase: bool = True,
) -> set[str]:
    """
    Extract vocabulary from spaCy German model.

    SpaCy models contain vocabulary learned from training data.
    The 'de_core_news_sm' model has ~244k German word entries.

    Args:
        model_name: spaCy model to load (default: "de_core_news_sm")
        output_path: Path to save wordlist file (default: None, no save)
        min_length: Minimum word length (default: 2)
        lowercase: Convert to lowercase (default: True)

    Returns:
        Set of German words

    Example:
        >>> vocab = extract_spacy_vocab(output_path=get_wordlist_path("spacy"))
        >>> print(f"Extracted {len(vocab):,} words")
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

    vocab_words = set()
    allowed_special = {
        "-",
        "'",
        "\u00e4",
        "\u00f6",
        "\u00fc",
        "\u00c4",
        "\u00d6",
        "\u00dc",
        "\u00df",
    }

    for word in nlp.vocab.strings:
        if not word or word.startswith("_"):
            continue
        if word.strip() != word or not word.strip():
            continue
        if not any(c.isalpha() for c in word):
            continue
        if any(not (c.isalpha() or c in allowed_special) for c in word):
            continue
        if not word[0].isalpha():
            continue
        if len(word) < min_length:
            continue

        vocab_words.add(word.lower() if lowercase else word)

    logger.info(f"Extracted {len(vocab_words):,} words from {model_name}")

    if output_path:
        _save_wordlist(vocab_words, Path(output_path))

    return vocab_words


def download_leipzig_wordlist(
    output_path: Optional[str | Path] = None,
    *,
    year: int = 2021,
) -> set[str]:
    """
    Download German wordlist from Leipzig Corpora Collection.

    The Leipzig Corpora Collection provides frequency-based word lists
    extracted from large German text corpora (news, web, Wikipedia).

    Available at: https://wortschatz.uni-leipzig.de/en/download/German

    Args:
        output_path: Path to save wordlist file (default: None, no save)
        year: Corpus year (default: 2021)

    Returns:
        Set of German words

    Example:
        >>> vocab = download_leipzig_wordlist(
        ...     output_path=get_wordlist_path("leipzig")
        ... )
    """
    logger.info(f"Downloading Leipzig Corpora Collection (deu_news_{year})")

    corpus_name = f"deu_news_{year}_100K"
    filename = f"{corpus_name}.tar.gz"
    url = f"{LEIPZIG_BASE_URL}/{filename}"

    words = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpdir = Path(tmp_dir)
        tar_path = tmpdir / filename

        logger.info(f"Downloading from: {url}")
        try:
            urlretrieve(url, tar_path)  # noqa: S310
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Failed to download Leipzig Corpora: {e}\n"
                f"URL: {url}\n"
                f"Check https://wortschatz.uni-leipzig.de/en/download/German for available corpora"
            ) from e

        logger.info("Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)  # noqa: S202

        words_file = tmpdir / corpus_name / f"{corpus_name}-words.txt"

        if not words_file.exists():
            raise FileNotFoundError(f"Words file not found in archive: {words_file}")

        logger.info("Reading word frequencies...")
        with words_file.open(encoding="utf-8") as f:
            for line in f:
                # Format: rank\tword\tfrequency
                parts = line.strip().split("\t")
                if len(parts) >= MIN_LEIPZIG_PARTS:
                    word = parts[1]
                    if word and len(word) >= MIN_WORD_LENGTH and word[0].isalpha():
                        words.add(word.lower())

    logger.info(f"Extracted {len(words):,} words from Leipzig Corpora")

    if output_path:
        _save_wordlist(words, Path(output_path))

    return words


def _find_hunspell_dic() -> Optional[Path]:
    """Find Hunspell German dictionary on the system."""
    for path in HUNSPELL_SYSTEM_PATHS:
        if path.exists():
            return path
    return None


def _parse_hunspell_dic(dic_path: Path, *, encoding: str = "utf-8") -> set[str]:
    """
    Parse word stems from a Hunspell .dic file.

    The .dic format has an optional word count on the first line,
    then one entry per line in the format: word/flags
    We extract just the word stems (part before the /).
    Comment lines starting with '#' are skipped (used in some .dic files
    like the de_DE-1901 old orthography dictionary).
    """
    words = set()
    try:
        with dic_path.open(encoding=encoding) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # Skip word count line (first non-comment, non-empty line may be a number)
                if line.isdigit():
                    continue
                word = line.split("/")[0] if "/" in line else line
                if word and len(word) >= MIN_WORD_LENGTH:
                    words.add(word.lower())
    except UnicodeDecodeError:
        if encoding != "latin-1":
            return _parse_hunspell_dic(dic_path, encoding="latin-1")
        raise

    return words


def extract_hunspell_wordlist(
    output_path: Optional[str | Path] = None,
) -> set[str]:
    """
    Extract German wordlist from Hunspell dictionaries (igerman98).

    Merges two dictionaries for comprehensive coverage:
    - de_DE: Modern German orthography (post-1996 reform)
    - de_DE-1901: Traditional/old orthography (pre-1996, classical spelling)

    The old orthography is essential for 1900-1920 newspaper text, which uses
    historical spellings like "dass" -> "dass", "Schloss" -> "Schloss",
    "Fotografie" -> "Photographie", "Telefon" -> "Telephon".

    Checks for system-installed dictionaries first, then downloads from GitHub.
    No additional Python packages required -- parses .dic files directly.

    Args:
        output_path: Path to save wordlist file (default: None, no save)

    Returns:
        Set of German word stems (~170k words, modern + old orthography)

    Example:
        >>> vocab = extract_hunspell_wordlist(
        ...     output_path=get_wordlist_path("hunspell")
        ... )
    """
    # Modern dictionary (de_DE)
    system_dic = _find_hunspell_dic()
    if system_dic:
        logger.info(f"Using system Hunspell dictionary: {system_dic}")
        words = _parse_hunspell_dic(system_dic, encoding="latin-1")
    else:
        logger.info("Downloading modern Hunspell dictionary (de_DE)...")
        words = _download_and_parse_hunspell_dic(HUNSPELL_DIC_URL, "de.dic")
    logger.info(f"Modern dictionary: {len(words):,} stems")

    # Old orthography dictionary (de_DE-1901)
    system_1901 = _find_hunspell_1901_dic()
    if system_1901:
        logger.info(f"Using system old orthography dictionary: {system_1901}")
        old_words = _parse_hunspell_dic(system_1901, encoding="latin-1")
    else:
        logger.info("Downloading old orthography dictionary (de_DE-1901)...")
        old_words = _download_and_parse_hunspell_dic(HUNSPELL_1901_DIC_URL, "de-DE-1901.dic")
    logger.info(f"Old orthography dictionary: {len(old_words):,} stems")

    words |= old_words
    logger.info(f"Merged total: {len(words):,} unique word stems")

    if output_path:
        _save_wordlist(words, Path(output_path))

    return words


def _find_hunspell_1901_dic() -> Optional[Path]:
    """Find Hunspell old orthography (de_DE-1901) dictionary on the system."""
    for path in HUNSPELL_1901_SYSTEM_PATHS:
        if path.exists():
            return path
    return None


def _download_and_parse_hunspell_dic(url: str, filename: str) -> set[str]:
    """Download a Hunspell .dic file from a URL and parse stems."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dic_path = Path(tmp_dir) / filename

        logger.info(f"Downloading Hunspell dictionary from: {url}")
        try:
            urlretrieve(url, dic_path)  # noqa: S310
        except (OSError, ValueError) as e:
            raise RuntimeError(f"Failed to download Hunspell dictionary: {e}\nURL: {url}") from e

        return _parse_hunspell_dic(dic_path)


def download_dta_wordlist(
    output_path: Optional[str | Path] = None,
    *,
    periods: Optional[list[str]] = None,
) -> set[str]:
    """
    Download German wordlist from Deutsches Textarchiv (DTA).

    Downloads lemmatized text archives from the DTA Kernkorpus and extracts
    unique word tokens. The DTA is a corpus of historical German texts
    (1600-1900+), providing authentic historical vocabulary essential for
    1910-1920 newspaper OCR quality assessment.

    By default downloads the 1800-1899 and 1900-1999 periods, which are
    most relevant for early 20th century newspaper text.

    Args:
        output_path: Path to save wordlist file (default: None, no save)
        periods: List of period archive filenames to download
            (default: ["1800-1899.zip", "1900-1999.zip"])

    Returns:
        Set of German words (historical variants included)

    Example:
        >>> vocab = download_dta_wordlist(
        ...     output_path=get_wordlist_path("dta")
        ... )

    Note:
        This includes historical spellings like "Thal" (-> "Tal"),
        "eigenthumlich" (-> "eigentumlich"), essential for 1910-1920 text.
        Downloads ~140MB total (1800-1899: ~133MB, 1900-1999: ~8MB).
    """
    if periods is None:
        periods = DTA_PERIOD_FILES

    logger.info(f"Downloading DTA Kernkorpus lemmatized text ({len(periods)} periods)")

    words = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpdir = Path(tmp_dir)

        for period_file in periods:
            url = f"{DTA_LEMMATIZED_BASE_URL}/{period_file}"
            zip_path = tmpdir / period_file

            logger.info(f"Downloading {period_file} from: {url}")
            try:
                urlretrieve(url, zip_path)  # noqa: S310
            except (OSError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to download DTA period {period_file}: {e}\n"
                    f"URL: {url}\n"
                    "Visit https://www.deutschestextarchiv.de/download for available data."
                ) from e

            logger.info(f"Extracting tokens from {period_file}...")
            period_words = _extract_tokens_from_dta_zip(zip_path)
            logger.info(f"  {period_file}: {len(period_words):,} unique tokens")
            words |= period_words

    logger.info(f"Extracted {len(words):,} unique words from DTA Kernkorpus")

    if output_path:
        _save_wordlist(words, Path(output_path))

    return words


def _extract_tokens_from_dta_zip(zip_path: Path) -> set[str]:
    """Extract unique word tokens from a DTA lemmatized ZIP archive."""
    words = set()

    with zipfile.ZipFile(zip_path, "r") as z:
        text_files = [n for n in z.namelist() if n.endswith(".txt")]

        for text_file in text_files:
            with z.open(text_file) as f:
                for raw_line in f:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    for token in line.split():
                        token_clean = token.strip()
                        if (
                            token_clean
                            and len(token_clean) >= MIN_WORD_LENGTH
                            and token_clean[0].isalpha()
                            and any(c.isalpha() for c in token_clean)
                        ):
                            words.add(token_clean.lower())

    return words


def generate_wordlist(
    source: str = "spacy",
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Generate a German wordlist from the specified source.

    Saves the wordlist to the central wordlists directory using
    the standard filename for each source.

    Args:
        source: Wordlist source
            - "spacy": Extract from spaCy de_core_news_sm (~244k words)
            - "spacy_lg": Extract from spaCy de_core_news_lg (~500k words)
            - "hunspell": Extract from Hunspell de_DE + de_DE-1901 (~170k stems)
            - "leipzig": Download Leipzig Corpora (~155k most common words)
            - "dta": Download DTA historical German
        output_path: Override default save path (default uses central wordlists dir)

    Returns:
        Path to the saved wordlist file

    Example:
        >>> path = generate_wordlist(source="spacy")
        >>> path = generate_wordlist(source="leipzig")
        >>> path = generate_wordlist(source="hunspell")
        >>> path = generate_wordlist(source="dta")
    """
    if output_path is None:
        output_path = get_wordlist_path(source)
    output_path = Path(output_path)

    if source == "spacy":
        extract_spacy_vocab(model_name="de_core_news_sm", output_path=output_path)
    elif source == "spacy_lg":
        extract_spacy_vocab(model_name="de_core_news_lg", output_path=output_path)
    elif source == "hunspell":
        extract_hunspell_wordlist(output_path=output_path)
    elif source == "leipzig":
        download_leipzig_wordlist(output_path=output_path)
    elif source == "dta":
        download_dta_wordlist(output_path=output_path)
    else:
        available = sorted({k for k in WORDLIST_FILENAMES if k != "spacy_lg"})
        raise ValueError(f"Unknown source: {source}. Available: {available}")

    return output_path


def get_wordlist_info(path: str | Path) -> dict:
    """
    Get information about a wordlist file.

    Args:
        path: Path to wordlist file

    Returns:
        Dictionary with wordlist statistics

    Example:
        >>> info = get_wordlist_info(get_wordlist_path("spacy"))
        >>> print(f"Words: {info['word_count']:,}")
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
