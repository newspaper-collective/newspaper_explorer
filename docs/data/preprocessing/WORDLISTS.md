# German Wordlists for OCR Quality Validation

**Location:** `data/wordlist_de.txt` and `data/wordlists/`
**Status:** ✅ Ready
**Purpose:** Provide vocabulary for out-of-vocabulary (OOV) rate calculation

> **📚 For comprehensive preprocessing documentation, see [PREPROCESSING.md](PREPROCESSING.md)**
> For quality validation details, see [QUALITY_VALIDATION.md](QUALITY_VALIDATION.md)

---

## Summary

You now have **three German wordlists** for OCR quality validation:

| Source | Words | Size | Best For |
|--------|-------|------|----------|
| **spaCy** | 244,057 | 2.7 MB | General modern German, fast |
| **Leipzig** | 155,006 | 1.9 MB | Most common words, frequency-based |
| **DTA** | Manual | - | Historical German (1500-1900+) |

## Available Wordlists

### 1. spaCy (✅ Ready)
- **Location**: `data/wordlist_de.txt`
- **Words**: 244,057
- **Source**: spaCy de_core_news_sm model vocabulary
- **Generated**: Already done

**Use for**: General German text, modern vocabulary

### 2. Leipzig Corpora (✅ Ready)
- **Location**: `data/wordlists/wordlist_leipzig.txt`
- **Words**: 155,006
- **Source**: German news corpus 2021 (100k sentences)
- **Downloaded**: Automatically

**Use for**: Most common modern German words, frequency-based

### 3. DTA - Deutsches Textarchiv (⚠️ Manual Download)
- **Location**: TBD (after download)
- **Words**: Est. 200k-300k historical words
- **Source**: Historical German texts (1500-1900+)
- **Download**: https://www.deutschestextarchiv.de/download

**Recommended**: DTA-Kernkorpus, lemmatisiert (270MB, 1467 texts)
- Direct link: http://media.dwds.de/dta/download/dta_komplett_2020-10-23.lemma.zip
- Extract to: `data/dta/kern_lemmatized/`

**Use for**: Historical German from 1910-1920 newspaper period

### 4. Hunspell (❌ Download Failed)
- **Alternative**: System package installation
- **Install**: `sudo apt-get install hunspell-de-de`
- **Location**: `/usr/share/hunspell/de_DE.dic`
- **Words**: ~2 million word forms (with compounds)

## Usage

### Basic Usage

```python
from newspaper_explorer.data.preprocessing.validation import calculate_quality_metrics

# Use spaCy wordlist (general modern German)
df = calculate_quality_metrics(
    df,
    input_column="text",
    german_wordlist_path="data/wordlist_de.txt"
)

# Or use Leipzig wordlist (frequency-based)
df = calculate_quality_metrics(
    df,
    input_column="text",
    german_wordlist_path="data/wordlists/wordlist_leipzig.txt"
)
```

### Combine Multiple Wordlists

For best coverage, combine multiple wordlists:

```python
# Load all wordlists
spacy_words = set(line.strip() for line in open("data/wordlist_de.txt"))
leipzig_words = set(line.strip() for line in open("data/wordlists/wordlist_leipzig.txt"))

# Combine
combined = spacy_words | leipzig_words  # Union

# Save combined
with open("data/wordlist_combined.txt", "w") as f:
    for word in sorted(combined):
        f.write(f"{word}\n")

# Use combined
df = calculate_quality_metrics(
    df,
    german_wordlist_path="data/wordlist_combined.txt"
)
```

### Test Quality Validation

```python
# Test with existing script
python3 test_quality_with_wordlist.py
```

## Recommendations

### For 1910-1920 German Newspapers

**Best combination**:
1. **Start with**: spaCy wordlist (`data/wordlist_de.txt`) - 244k words
2. **Add**: DTA corpus (after manual download) - historical variants
3. **Optional**: Leipzig for frequency weighting

**Why this combination**:
- spaCy: Modern German baseline
- DTA: Historical spellings and vocabulary from your newspaper period
- Leipzig: Most common words for better OOV rate calculation

### Quick Start (Already Working)

Just use the spaCy wordlist - it's already generated and works well:

```python
df = calculate_quality_metrics(
    df,
    german_wordlist_path="data/wordlist_de.txt"
)
```

## Next Steps

### 1. Download DTA Corpus (Recommended)

```bash
# Download DTA-Kernkorpus, lemmatisiert
cd data
wget http://media.dwds.de/dta/download/dta_komplett_2020-10-23.lemma.zip

# Extract
unzip dta_komplett_2020-10-23.lemma.zip -d dta/kern_lemmatized

# Generate wordlist
python3 -c "
from newspaper_explorer.data.preprocessing.wordlist_utils import extract_dta_vocab

vocab = extract_dta_vocab(
    'data/dta/kern_lemmatized',
    output_path='data/wordlists/wordlist_dta.txt'
)
print(f'DTA wordlist: {len(vocab):,} words')
"
```

### 2. Test Quality Validation

```bash
# Test with wordlist
python3 test_quality_with_wordlist.py
```

### 3. Integrate with Preprocessing

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor()

# Add quality validation to your pipeline
df = preprocessor.pipeline(
    df,
    steps=[
        "normalize_unicode",
        "normalize_long_s",
        "calculate_quality_metrics",  # Calculate quality metrics
        "filter_by_quality_score",    # Remove poor quality
    ],
)
```

## Files Created

```
data/
├── wordlist_de.txt                           # spaCy (244k words) ✅
└── wordlists/
    ├── wordlist_leipzig.txt                  # Leipzig (155k words) ✅
    ├── wordlist_hunspell.txt                 # Hunspell (pending) ❌
    └── wordlist_dta.txt                      # DTA (after manual download) ⚠️
```

## See Also

- **[PREPROCESSING.md](PREPROCESSING.md)** - Complete preprocessing guide
- **[QUALITY_VALIDATION.md](QUALITY_VALIDATION.md)** - OCR quality validation
- **[NORMALIZATION.md](NORMALIZATION.md)** - Text normalization methods
- **[METADATA.md](METADATA.md)** - Preprocessing metadata system
