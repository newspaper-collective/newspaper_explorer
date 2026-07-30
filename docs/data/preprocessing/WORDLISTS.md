# German Wordlists for OCR Quality Validation

**Location:** `data/wordlists/` (centralized, configured via `Config.wordlists_dir`)
**Status:** Ready
**Purpose:** Provide vocabulary for out-of-vocabulary (OOV) rate calculation

> For comprehensive preprocessing documentation, see [PREPROCESSING.md](PREPROCESSING.md)
> For quality validation details, see [QUALITY_VALIDATION.md](QUALITY_VALIDATION.md)

---

## Summary

Four German wordlist sources for OCR quality validation, all stored centrally:

| Source | File | Words | Best For |
|--------|------|-------|----------|
| **spaCy** | `wordlist_spacy_de.txt` | ~244k | General modern German, fast |
| **Leipzig** | `wordlist_leipzig_de.txt` | ~155k | Most common words, frequency-based |
| **Hunspell** | `wordlist_hunspell_de.txt` | ~142k | Modern + old orthography stems |
| **DTA** | `wordlist_dta_de.txt` | varies | Historical German (1600-1900+) |

## Available Wordlists

### 1. spaCy
- **File**: `data/wordlists/wordlist_spacy_de.txt`
- **Words**: ~244,000
- **Source**: spaCy `de_core_news_sm` model vocabulary
- **Use for**: General German text, modern vocabulary

### 2. Leipzig Corpora
- **File**: `data/wordlists/wordlist_leipzig_de.txt`
- **Words**: ~155,000
- **Source**: German news corpus 2021 (100k sentences), frequency-based
- **Use for**: Most common modern German words

### 3. Hunspell (igerman98)
- **File**: `data/wordlists/wordlist_hunspell_de.txt`
- **Words**: ~142,000 stems
- **Source**: Merged from two igerman98 Hunspell dictionaries:
  - `de_DE` (modern, post-1996 orthography) via [wooorm/dictionaries](https://github.com/wooorm/dictionaries)
  - `de_DE-1901` (old/classical orthography, pre-1996) via [elastic/hunspell](https://github.com/elastic/hunspell/tree/master/dicts/de_DE-1901)
- **Use for**: Comprehensive coverage of both modern and historical German spellings
- **Note**: The old orthography variant includes forms like "dass" (old: "dass"), "Schloss" (old: "Schloss"), "Fotografie" (old: "Photographie"), "Telefon" (old: "Telephon") -- essential for 1900-1920 newspapers

### 4. DTA - Deutsches Textarchiv
- **File**: `data/wordlists/wordlist_dta_de.txt`
- **Words**: varies by period selection
- **Source**: DTA Kernkorpus lemmatized text (1800-1899 + 1900-1999 periods)
- **Use for**: Historical German vocabulary, essential for 1910-1920 newspaper OCR

## Usage

### Generating Wordlists

```python
from newspaper_explorer.data.utils.wordlists import generate_wordlist

# Generate any wordlist (saves to central data/wordlists/ directory)
generate_wordlist(source="spacy")
generate_wordlist(source="leipzig")
generate_wordlist(source="hunspell")
generate_wordlist(source="dta")
```

### Loading Wordlists

```python
from newspaper_explorer.data.utils.wordlists import load_wordlist, get_wordlist_path

# Load by source name
vocab = load_wordlist(get_wordlist_path("spacy"))
print(f"Loaded {len(vocab):,} words")

# Load by explicit path
vocab = load_wordlist("data/wordlists/wordlist_spacy_de.txt")
```

### Quality Validation

```python
from newspaper_explorer.data.preprocessing.validation import calculate_quality_metrics
from newspaper_explorer.data.utils.wordlists import get_wordlist_path

# Use spaCy wordlist
df = calculate_quality_metrics(
    df,
    input_column="text",
    german_wordlist_path=str(get_wordlist_path("spacy")),
)

# Or use Leipzig wordlist
df = calculate_quality_metrics(
    df,
    input_column="text",
    german_wordlist_path=str(get_wordlist_path("leipzig")),
)
```

### Combining Multiple Wordlists

For best coverage, combine multiple wordlists:

```python
from newspaper_explorer.data.utils.wordlists import load_wordlist, get_wordlist_path

spacy_words = load_wordlist(get_wordlist_path("spacy"))
leipzig_words = load_wordlist(get_wordlist_path("leipzig"))
hunspell_words = load_wordlist(get_wordlist_path("hunspell"))

combined = spacy_words | leipzig_words | hunspell_words
print(f"Combined: {len(combined):,} unique words")
```

### Path Helpers

```python
from newspaper_explorer.data.utils.wordlists import get_wordlists_dir, get_wordlist_path

# Get the central wordlists directory
print(get_wordlists_dir())  # .../data/wordlists/

# Get canonical path for any source
print(get_wordlist_path("spacy"))    # .../data/wordlists/wordlist_spacy_de.txt
print(get_wordlist_path("leipzig"))  # .../data/wordlists/wordlist_leipzig_de.txt
print(get_wordlist_path("hunspell")) # .../data/wordlists/wordlist_hunspell_de.txt
print(get_wordlist_path("dta"))      # .../data/wordlists/wordlist_dta_de.txt
```

## Recommendations for 1910-1920 German Newspapers

**Best combination**:
1. **Start with**: spaCy wordlist -- 244k modern German words (fast, already generated)
2. **Add**: DTA corpus -- historical spellings from your newspaper period
3. **Supplement**: Hunspell -- comprehensive word stems
4. **Optional**: Leipzig -- frequency-weighted common words

**Why this combination**:
- spaCy: Modern German baseline vocabulary
- DTA: Historical spellings and vocabulary ("Thal" -> "Tal", "eigenthumlich" -> "eigentumlich")
- Hunspell: Comprehensive morphological coverage
- Leipzig: Most common words for better OOV rate calculation

## File Layout

```
data/wordlists/
    wordlist_spacy_de.txt       # spaCy (244k words)
    wordlist_leipzig_de.txt     # Leipzig (155k words)
    wordlist_hunspell_de.txt    # Hunspell stems (~142k words, modern + old orthography)
    wordlist_dta_de.txt         # DTA historical German
```

## See Also

- **[PREPROCESSING.md](PREPROCESSING.md)** - Complete preprocessing guide
- **[QUALITY_VALIDATION.md](QUALITY_VALIDATION.md)** - OCR quality validation
