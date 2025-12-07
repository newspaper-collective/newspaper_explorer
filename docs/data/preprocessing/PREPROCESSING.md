# Text Preprocessing Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Preprocessing Steps](#preprocessing-steps)
7. [Normalization Methods](#normalization-methods)
8. [Performance & Configuration](#performance--configuration)
9. [Data Schemas](#data-schemas)
10. [Complete Workflows](#complete-workflows)
11. [Analyzing Results](#analyzing-results)
12. [Integration with Other Analyses](#integration-with-other-analyses)
13. [Troubleshooting](#troubleshooting)
14. [Testing](#testing)
15. [Dependencies](#dependencies)

---

## Overview

The preprocessing module provides a comprehensive pipeline for cleaning and normalizing historical German newspaper texts. It addresses challenges specific to OCR'd historical documents including outdated spelling conventions, OCR artifacts, and archaic characters.

**Key Features**:
- ✅ **Modular pipeline**: Chain multiple preprocessing steps in any order
- ✅ **Historical text support**: Three normalization methods (simple, Transnormer, DTA-CAB)
- ✅ **Multi-GPU support**: Parallel Transnormer processing across multiple GPUs
- ✅ **Resume functionality**: Cache results to avoid reprocessing (Transnormer & DTA-CAB)
- ✅ **Configurable**: Fine-tune batch sizes, beam search, and GPU allocation
- ✅ **Preserve traceability**: All foreign keys (IDs) maintained throughout pipeline

**Preprocessing Categories**:
1. **Normalization** - Transform text to standard form (Unicode, diacritics, historical characters, spelling)
2. **Cleaning** - Remove noise (whitespace, case)
3. **Filtering** - Remove content (punctuation, numbers, stopwords, OCR artifacts)
4. **Linguistic** - Language-aware processing (dehyphenation, lemmatization)

---

## Architecture

### File Structure

```
src/newspaper_explorer/data/preprocessing/
├── pipeline.py              # Main TextPreprocessor class
├── normalization.py         # Text normalization (Unicode, diacritics, historical chars, spelling)
├── cleaning.py              # Text cleaning (whitespace, case)
├── filtering.py             # Content filtering and removal
└── linguistic.py            # Linguistic processing (dehyphenate, lemmatize)

cli/data/
└── preprocessing.py         # CLI command registration

docs/data/preprocessing/
├── PREPROCESSING.md         # This comprehensive guide
├── NORMALIZATION.md         # Detailed normalization documentation
└── METADATA.md              # Metadata system documentation
```

### Integration with Data Pipeline

```
Download → Parse ALTO/METS → Polars DataFrame
                                    ↓
                              PREPROCESSING
                  (Sequential pipeline - steps chain together)
                                    ↓
                            ┌───────────────┐  Unicode & Characters:
                            │ Normalization │  - normalize-unicode (NFKC, quotes, spaces)
                            │   (optional)  │  - remove-diacritics (ä→a, ö→o)
                            └───────┬───────┘
                                    ↓          Historical German Spelling:
                                    ↓          - normalize (simple: ſ→s, ß→ss)
                                    ↓          - normalize-dtacab (API, slow)
                                    ↓          - normalize-transnormer (neural, fast)
                                    ↓
                            ┌───────────────┐  Use any/all:
                            │   Cleaning    │  - normalize-whitespace
                            │   (optional)  │  - lowercase
                            └───────┬───────┘
                                    ↓
                            ┌───────────────┐  Use any/all:
                            │   Filtering   │  - filter-length
                            │   (optional)  │  - filter-word-count
                            └───────┬───────┘  - remove-punctuation
                                    ↓          - remove-numbers
                                    ↓          - remove-stopwords
                                    ↓          - clean-ocr
                                    ↓
                            ┌───────────────┐  Linguistic processing:
                            │  Linguistic   │  - dehyphenate (remove line-breaks)
                            │   (optional)  │  - lemmatize-spacy (fast)
                            └───────┬───────┘  - lemmatize (thorough)
                                    ↓
                                    ↓
                              Original text
                                    ↓
                    Step 1 output → Step 2 input → Step 3 output → ... → Final output
                                    ↓
                      data/processed/{source}/text/{source}_preprocessed.parquet
                                    ↓
                         Analysis (Entities, Topics, Emotions, etc.)
```

**Key Points**:
- **Sequential processing**: Each step's output becomes the next step's input
- **All steps are compatible**: You can chain any combination in any order
- **Categories are conceptual**: The actual pipeline doesn't enforce these groupings
- **Original data preserved**: All foreign keys and metadata columns remain intact

**Integration Points**:
- Input: Parquet files from `DataLoader` (line-level or text-block-level)
- Output: Parquet files with additional processed text column(s)
- All original columns preserved (IDs, metadata, coordinates)
- Compatible with all analysis modules (entities, topics, emotions, layout)

---

## Core Components

### 1. TextPreprocessor

**Purpose**: Main pipeline class that chains preprocessing steps in sequence.

**Features**:
- Sequential step application
- Column chaining (each step outputs to next input)
- Temporary column cleanup
- Progress logging
- Error handling

**Usage**:
```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor(text_column="text")

# Apply pipeline
df = preprocessor.pipeline(
    df,
    steps=["normalize", "lowercase", "remove-stopwords"],
    output_column="text_processed"
)
```

**How the pipeline works**:
- Steps are applied **sequentially** - each step's output becomes the next step's input
- All steps are **compatible** and can be chained in any order
- The pipeline creates temporary columns internally and cleans them up at the end
- **All original columns** (IDs, metadata) are preserved throughout

**Example flow**:
```
Original text → normalize → temp1 → lowercase → temp2 → remove-stopwords → text_processed
```

### Step Categories

The preprocessing steps fall into four categories that can all work together:

**1. Normalization** (use any/all, order matters):
- `normalize-unicode` - Unicode normalization (NFKC, quotes, spaces, control chars) - **RECOMMENDED FIRST STEP**
- `remove-diacritics` - Remove accents (ä→a, ö→o, ü→u)
- `normalize` - Simple historical German (ſ→s, ß→ss)
- `normalize-transnormer` - Neural historical German normalization (spelling modernization)
- `normalize-dtacab` - API historical German normalization (slower, academic-grade)

**2. Cleaning** (use any/all):
- `normalize-whitespace` - Collapse/normalize whitespace
- `lowercase` - Convert to lowercase

**3. Filtering** (use any/all):
- `remove-punctuation` - Remove punctuation
- `remove-numbers` - Remove digits
- `remove-stopwords` - Remove common words
- `filter-length` - Filter by character length
- `filter-word-count` - Filter by word count
- `clean-ocr` - Remove OCR artifacts

**4. Linguistic** (use any/all):
- `dehyphenate` - Remove line-break hyphens (language-aware with pyphen)
- `lemmatize-spacy` - Fast lemmatization (spaCy)
- `lemmatize` - Thorough lemmatization (GermaLemma)

**Key Insights**:
- **normalize-unicode** should typically be **FIRST** - it handles OCR artifacts and standardizes characters
- **Normalization methods are complementary**: `normalize-unicode` handles Unicode/OCR issues, while `normalize`/`normalize-transnormer`/`normalize-dtacab` handle historical spelling
- **dehyphenate** is in Linguistic (not Cleaning) because it uses language-specific dictionaries to distinguish line-break hyphens from compound-word hyphens
- All steps can be chained in any order, but recommended order: normalize-unicode → historical normalization → cleaning → filtering → linguistic

**Available Steps (Complete List)**:
- `normalize-unicode` - Unicode normalization (NFKC + character translation)
- `normalize` - Simple character normalization (ſ→s, ẞ→SS, ß→ss)
- `normalize-transnormer` - Neural normalization with Transnormer
- `normalize-dtacab` - API normalization with DTA-CAB
- `remove-diacritics` - Remove accents (ä→a, ö→o)
- `normalize-whitespace` - Normalize whitespace
- `lowercase` - Convert to lowercase
- `remove-punctuation` - Remove punctuation marks
- `remove-numbers` - Remove numeric digits
- `remove-stopwords` - Remove German stopwords
- `dehyphenate` - Remove line-break hyphens
- `lemmatize-spacy` - Fast spaCy lemmatization
- `lemmatize` - Thorough GermaLemma lemmatization
- `filter-length` - Filter by character length
- `filter-word-count` - Filter by word count
- `clean-ocr` - Remove OCR artifacts

### 2. Normalization Module

**Purpose**: Transform text to standard form - handles Unicode issues, diacritics, and historical German spelling.

**Location**: `src/newspaper_explorer/data/preprocessing/normalization.py`

**Methods**:

#### 2.1 Unicode Normalization
- **Function**: `normalize_unicode()`
- **Speed**: ⚡⚡ Fast (Python built-ins)
- **Quality**: ★★★★★ Essential
- **Use case**: **ALWAYS FIRST** - handles OCR artifacts, unifies quotes/spaces

```python
from newspaper_explorer.data.preprocessing.normalization import normalize_unicode

# Conservative mode (default): preserves semantic punctuation
df = normalize_unicode(df, input_column="text", output_column="text_unicode")

# Aggressive mode: unifies all dashes (good for NLP)
df = normalize_unicode(df, input_column="text", aggressive=True)
```

**Transformations**:
- NFKC normalization
- Unifies quotation marks („"" → ")
- Normalizes various space types → regular space
- Removes soft hyphens, zero-width chars
- Removes OCR artifacts (bullets, boxes)
- Optional: strips control characters

**Two modes**:
- **Conservative** (default): Preserves en dash, em dash (semantic meaning)
- **Aggressive**: Unifies all dashes → hyphen (better for NLP)

#### 2.2 Diacritic Removal
- **Function**: `remove_diacritics()`
- **Speed**: ⚡⚡ Fast (unidecode)
- **Quality**: ★★★☆☆ Lossy but useful
- **Use case**: ASCII normalization, searchability

```python
from newspaper_explorer.data.preprocessing.cleaning import remove_diacritics

df = remove_diacritics(df, input_column="text", output_column="text_no_diacritics")
```

**Transformations**:
- ä → a, ö → o, ü → u
- é → e, à → a, etc.

#### 2.3 Simple Historical German
- **Function**: `simple()`
- **Speed**: ⚡ Instant (regex replacement)
- **Quality**: ★★☆☆☆ Basic
- **Use case**: Quick character mapping

```python
from newspaper_explorer.data.preprocessing.normalization import simple

df = simple(df, input_column="text", output_column="text_simple")
```

**Transformations**:
- `ſ` → `s` (long s)
- `ẞ` → `SS` (capital sharp s)
- `ß` → `ss` (sharp s)

#### 2.4 Transnormer (Neural Historical German)
- **Function**: `transnormer()`
- **Speed**: ⚡⚡⚡ Medium-Fast (GPU: 100-500 sent/sec, CPU: 10-100 sent/sec)
- **Quality**: ★★★★★ Excellent (98.88% accuracy)
- **Use case**: High-quality spelling modernization at scale

```python
from newspaper_explorer.data.preprocessing.normalization import transnormer

df = transnormer(
    df,
    input_column="text",
    output_column="text_transnormer",
    model="19c",           # or "18-19c" for 1700-1899
    batch_size=32,         # Increase for faster processing
    num_beams=4,           # Decrease for faster processing
    num_gpus=1,            # Multi-GPU parallel processing
    use_cache=True,        # Enable resume functionality
)
```

**Available Models**:
- `"19c"`: `ybracke/transnormer-19c-beta-v02` (1780-1899, 98.88% accuracy)
- `"18-19c"`: `ybracke/transnormer-18-19c-beta-v01` (1700-1899, 99.53% accuracy)
- Custom: Any HuggingFace seq2seq model

**Features**:
- Multi-GPU support (parallel processing across GPUs)
- Caching for resume capability
- Text chunking (handles long documents)
- FP16 support (faster inference on Ampere+ GPUs)
- torch.compile optimization (PyTorch 2.0+)

#### 2.3 DTA-CAB (API)
- **Function**: `dta_cab()`
- **Speed**: ⚡ Slow (API calls)
- **Quality**: ★★★★★ Excellent
- **Use case**: Small datasets, research validation

```python
from newspaper_explorer.data.preprocessing.normalization import dta_cab

df = dta_cab(
    df,
    text_column="text",
    output_column="text_normalized",
    batch_size=100,
    timeout=30,
    use_cache=True,
)
```

**Note**: Requires internet connection to BBAW DTA-CAB service.

### 3. Cleaning Module

**Purpose**: Remove noise from text (whitespace, case).

**Location**: `src/newspaper_explorer/data/preprocessing/cleaning.py`

**Functions**:

#### 3.1 Whitespace Normalization
```python
from newspaper_explorer.data.preprocessing.cleaning import normalize_whitespace

# Default: Collapse ALL whitespace (spaces, tabs, newlines) to single space
df = normalize_whitespace(df, input_column="text", output_column="text_clean")
# "Hello    world\n\ttab  " → "Hello world tab"

# Preserve newlines: Collapse spaces/tabs but keep line breaks
df = normalize_whitespace(
    df,
    input_column="text",
    output_column="text_clean",
    keep_newlines=True
)
# "Hello    world\n\ttab  " → "Hello world\ntab"
```

**Two modes:**

1. **Default (keep_newlines=False):**
   - Collapses ALL whitespace (spaces, tabs, newlines) to single space
   - Removes leading/trailing whitespace
   - **Use for:** Aggregated text blocks, NLP tasks, topic modeling
   - Example: Paragraph-level or document-level analysis

2. **Newline-preserving (keep_newlines=True):**
   - Collapses multiple spaces/tabs to single space
   - KEEPS newlines intact, removes spaces around them
   - Normalizes `\r\n` and `\r` to `\n`
   - **Use for:** Line-by-line processing, preserving text structure
   - Example: Poetry, structured documents, visual layout preservation

**When to use which:**
- **Aggregated text** (text blocks, articles): Use default (removes newlines)
- **Line-level analysis** (OCR output, structured text): Use `keep_newlines=True`
- **Topic modeling, embeddings**: Use default (newlines are noise)
- **Preserving layout** (poetry, tables): Use `keep_newlines=True`

**Practical example:**
```python
# Newspaper text with inconsistent spacing and line breaks
text = """Der Kaiser    reiste gestern
nach   Berlin.

    Die Verhandlungen   über
den Friedensvertrag    dauern an."""

# Default mode → single line for analysis
# "Der Kaiser reiste gestern nach Berlin. Die Verhandlungen über den Friedensvertrag dauern an."

# Newline-preserving → structured text
# "Der Kaiser reiste gestern\nnach Berlin.\n\nDie Verhandlungen über\nden Friedensvertrag dauern an."
```

#### 3.2 Lowercase Conversion
```python
from newspaper_explorer.data.preprocessing.cleaning import lowercase

df = lowercase(df, input_column="text", output_column="text_lower")
```
- Converts all characters to lowercase
- Useful for case-insensitive analysis

### 4. Cleaning Module (continued)

**Purpose**: Transform text content while keeping all rows.

**Functions**:

#### 4.1 Punctuation Removal
```python
from newspaper_explorer.data.preprocessing.cleaning import remove_punctuation

df = remove_punctuation(
    df,
    input_column="text",
    output_column="text_nopunct",
    keep_chars="-'"  # Optional: keep specific characters
)
```

#### 4.2 Number Removal
```python
from newspaper_explorer.data.preprocessing.cleaning import remove_numbers

df = remove_numbers(df, input_column="text", output_column="text_nonum")
```

#### 4.3 Stopword Removal
```python
from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords

df = remove_stopwords(
    df,
    input_column="text",
    output_column="text_nostop",
    language="de"  # or "en"
)
```
- Requires spaCy
- Uses spaCy's built-in stopword lists

### 5. Filtering Module

**Purpose**: Remove rows from DataFrame based on criteria.

**Functions**:

#### 5.1 Length Filtering
```python
from newspaper_explorer.data.preprocessing.filtering import filter_by_total_character_length

df = filter_by_total_character_length(
    df,
    input_column="text",
    min_length=10,
    max_length=10000
)
```
- Removes too short or too long texts
- Useful for removing headers, footers, OCR artifacts

#### 4.4 Character Allowlist Filtering
```python
from newspaper_explorer.data.preprocessing.cleaning import only_keep_allowed_chars

df = only_keep_allowed_chars(
    df,
    input_column="text",
    output_column="text_filtered",
    allowed_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0-9äöüßÄÖÜ .,!?-",
    replace_with=""
)
```
- Keeps only characters in the allowed set
- Removes all other characters

#### 4.6 Word Count Filtering
```python
from newspaper_explorer.data.preprocessing.filtering import filter_by_word_count

df = filter_by_word_count(
    df,
    input_column="text",
    min_words=2,
    max_words=100
)
```
- Filters rows by number of words (not characters)
- Removes single-word artifacts
- More meaningful than character length for content filtering

### 5. Linguistic Module

**Purpose**: Advanced linguistic processing.

**Functions**:

#### 5.1 Dehyphenation
```python
from newspaper_explorer.data.preprocessing.linguistic import dehyphenate

df = dehyphenate(
    df,
    input_column="text",
    output_column="text_dehyphen",
    language="de_DE"
)
```
- Removes line-break hyphens (`"Zeitung-\nspapier"` → `"Zeitungspapier"`)
- Preserves legitimate compound word hyphens (`"Nord-Süd"` stays `"Nord-Süd"`)
- **Language-aware**: Uses pyphen library with language-specific dictionaries
- **Why "linguistic"**: While it removes hyphens (like cleaning), it requires understanding of word formation rules and language-specific hyphenation patterns

#### 5.2 Lemmatization (spaCy - Fast)
```python
from newspaper_explorer.data.preprocessing.linguistic import lemmatize_spacy

df = lemmatize_spacy(
    df,
    input_column="text",
    output_column="text_lemma",
    model="de_core_news_sm",  # or de_core_news_md, de_core_news_lg
    batch_size=1000
)
```
- Context-aware lemmatization
- 100x faster than GermaLemma
- Requires spaCy model: `python -m spacy download de_core_news_sm`

#### 5.3 Lemmatization (GermaLemma - Thorough)
```python
from newspaper_explorer.data.preprocessing.linguistic import lemmatize_germalemma

df = lemmatize_germalemma(
    df,
    input_column="text",
    output_column="text_lemma"
)
```
- Very thorough but slow
- ~100x slower than spaCy
- Use only for small datasets

---

## CLI Commands

### Main Command

```bash
newspaper-explorer data preprocess --source <name> --steps <steps> [options]
```

### Basic Examples

#### Quick Normalization
```bash
# Simple character normalization
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase
```

#### Full Cleaning Pipeline
```bash
# Complete preprocessing
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase,remove-punctuation,remove-stopwords
```

#### Test on Sample
```bash
# Process only first 1000 rows
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase \
    --sample 1000
```

### Transnormer Examples

#### High-Quality Normalization
```bash
# Default settings (balanced quality/speed)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer
```

#### Fast Processing
```bash
# Optimize for speed
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --batch-size 128 \
    --num-beams 2
```

#### Multi-GPU Processing
```bash
# Use all 4 GPUs for maximum speed
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --num-gpus 4 \
    --batch-size 128
```

#### Resume After Interruption
```bash
# Resume processing (uses cached results)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer
    # Cache is enabled by default

# Disable cache (reprocess everything)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --no-cache
```

### Custom I/O

```bash
# Custom input/output paths
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/custom/input.parquet \
    --output data/custom/output.parquet \
    --steps normalize,lowercase
```

### Combined Workflows

```bash
# Full preprocessing pipeline
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer,lowercase,remove-punctuation,remove-stopwords,lemmatize-spacy \
    --batch-size 64 \
    --num-gpus 2
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source`, `-s` | Source name (required) | - |
| `--input`, `-i` | Input parquet file | `data/processed/{source}/text/textblocks.parquet` |
| `--output`, `-o` | Output parquet file | Auto-generated based on steps |
| `--steps` | Comma-separated preprocessing steps (required) | - |
| `--text-column` | Column name containing text | `text` |
| `--output-column` | Output column name | `text_processed` |
| `--sample` | Process only first N rows | - |
| `--batch-size` | Batch size for Transnormer | 32 |
| `--num-beams` | Beam search parameter for Transnormer | 4 |
| `--num-gpus` | Number of GPUs for Transnormer | 1 |
| `--no-cache` | Disable caching for Transnormer | False |

---

## Python API

### Pipeline API

#### Basic Pipeline
```python
import polars as pl
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

# Load data
df = pl.read_parquet("data/processed/der_tag/text/textblocks.parquet")

# Create preprocessor
preprocessor = TextPreprocessor(text_column="text")

# Apply pipeline
df = preprocessor.pipeline(
    df,
    steps=["normalize", "lowercase", "remove-stopwords"],
    output_column="text_processed"
)

# Save results
df.write_parquet("data/processed/der_tag/text/preprocessed.parquet")
```

#### Advanced Pipeline with Transnormer
```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor(text_column="text")

# High-quality normalization with multi-GPU
df = preprocessor.pipeline(
    df,
    steps=[
        "normalize-transnormer",
        "lowercase",
        "remove-punctuation",
        "remove-stopwords",
        "lemmatize-spacy"
    ],
    output_column="text_processed",
    batch_size=128,      # Larger batches for speed
    num_beams=4,         # Quality parameter
    num_gpus=4,          # Use 4 GPUs in parallel
    use_cache=True,      # Enable resume
)
```

### Direct Function API

#### Import Individual Functions
```python
from newspaper_explorer.data.preprocessing.normalization import simple, transnormer
from newspaper_explorer.data.preprocessing.cleaning import lowercase, normalize_whitespace
from newspaper_explorer.data.preprocessing.cleaning import remove_stopwords, remove_punctuation
from newspaper_explorer.data.preprocessing.linguistic import dehyphenate, lemmatize_spacy
```

#### Manual Step-by-Step Processing
```python
import polars as pl
from newspaper_explorer.data.preprocessing import (
    normalization,
    cleaning,
    filtering,
    linguistic
)

# Load data
df = pl.read_parquet("input.parquet")

# Step 1: Normalize historical characters
df = normalization.simple(df, text_column="text", output_column="text_norm")

# Step 2: Normalize whitespace
df = cleaning.normalize_whitespace(df, input_column="text_norm", output_column="text_clean")

# Step 3: Lowercase
df = cleaning.lowercase(df, input_column="text_clean", output_column="text_lower")

# Step 4: Remove stopwords
df = cleaning.remove_stopwords(df, input_column="text_lower", output_column="text_nostop")

# Step 5: Lemmatize
df = linguistic.lemmatize_spacy(df, input_column="text_nostop", output_column="text_lemma")

# Save
df.write_parquet("output.parquet")
```

#### Custom Normalization Parameters
```python
from newspaper_explorer.data.preprocessing.normalization import transnormer

# Fine-tune Transnormer
df = transnormer(
    df,
    text_column="text",
    output_column="text_normalized",
    model="18-19c",              # For earlier texts (1700-1899)
    batch_size=64,               # Larger batches
    num_beams=2,                 # Faster, slightly lower quality
    max_new_tokens=512,          # Max output length
    max_input_length=512,        # Max input length
    device="cuda:0",             # Specific GPU
    num_gpus=2,                  # Multi-GPU
    use_cache=True,              # Enable caching
    cache_dir=".cache/transnormer"  # Custom cache location
)
```

#### Process in Chunks (Memory Management)
```python
import polars as pl
from newspaper_explorer.data.preprocessing.normalization import transnormer

# Load data
df = pl.read_parquet("large_file.parquet")

# Process in chunks
chunk_size = 10000
results = []

for i in range(0, len(df), chunk_size):
    print(f"Processing chunk {i//chunk_size + 1}/{len(df)//chunk_size + 1}")

    chunk = df[i:i + chunk_size]
    normalized_chunk = transnormer(chunk, batch_size=64, num_gpus=4)
    results.append(normalized_chunk)

    # Optional: save intermediate results
    normalized_chunk.write_parquet(f"chunks/chunk_{i}.parquet")

# Combine results
final_df = pl.concat(results)
final_df.write_parquet("output.parquet")
```

---

## Preprocessing Steps

### Quick Reference Table

| Step | Speed | Quality | Use Case | Dependencies |
|------|-------|---------|----------|-------------|
| `normalize` | ⚡⚡⚡⚡⚡ Instant | ★★☆☆☆ | Quick char mapping | None |
| `normalize-transnormer` | ⚡⚡⚡⚡ Fast | ★★★★★ | High-quality normalization | transformers, torch |
| `normalize-dtacab` | ⚡ Very slow | ★★★★★ | Small datasets, validation | requests |
| `remove-diacritics` | ⚡⚡⚡⚡⚡ Instant | ★★★☆☆ | ASCII conversion | unidecode |
| `normalize-whitespace` | ⚡⚡⚡⚡⚡ Instant | ★★★★★ | Clean whitespace | None |
| `lowercase` | ⚡⚡⚡⚡⚡ Instant | ★★★★★ | Case normalization | None |
| `remove-punctuation` | ⚡⚡⚡⚡⚡ Instant | ★★★★☆ | Clean text | None |
| `remove-numbers` | ⚡⚡⚡⚡⚡ Instant | ★★★★☆ | Remove digits | None |
| `remove-stopwords` | ⚡⚡⚡⚡ Fast | ★★★★☆ | Content words only | spacy |
| `dehyphenate` | ⚡⚡⚡ Medium | ★★★★☆ | Remove line breaks | pyphen |
| `lemmatize-spacy` | ⚡⚡⚡⚡ Fast | ★★★★☆ | Reduce inflection | spacy |
| `lemmatize` | ⚡ Very slow | ★★★★★ | Thorough lemmatization | germalemma |
| `filter-length` | ⚡⚡⚡⚡⚡ Instant | ★★★★★ | Remove by char count | None |
| `filter-word-count` | ⚡⚡⚡⚡⚡ Instant | ★★★★★ | Remove by word count | None |
| `clean-ocr` | ⚡⚡⚡⚡ Fast | ★★★★☆ | OCR cleanup | None |

### Step Compatibility

**All steps can be chained together** - the pipeline applies them sequentially, with each step's output becoming the next step's input. However, some logical considerations:

**Choose ONE normalization method:**
- `normalize` - Basic character mapping (ſ→s, ß→ss)
- `normalize-transnormer` - Neural normalization (recommended for quality)
- `normalize-dtacab` - API-based normalization (best quality, slowest)

**Choose ONE lemmatization method:**
- `lemmatize-spacy` - Fast, context-aware (recommended)
- `lemmatize` - Thorough but very slow

**Steps that work together:**
- `normalize-transnormer` + `lowercase` + `remove-stopwords` ✅
- `normalize` + `normalize-whitespace` + `lowercase` ✅
- `dehyphenate` + `normalize-transnormer` + `lemmatize-spacy` ✅

**Steps that don't make sense together:**
- Multiple normalization methods (choose one)
- Multiple lemmatization methods (choose one)
- `remove-diacritics` after `normalize-transnormer` (Transnormer already handles this)

### Recommended Step Order

For best results, apply steps in this general order:

1. **Dehyphenation** (if needed) - Before normalization to handle line breaks correctly
2. **Normalization** - Choose ONE: `normalize`, `normalize-transnormer`, or `normalize-dtacab`
3. **Whitespace normalization** - Clean up spacing
4. **Lowercase** - Case normalization
5. **Content removal** - `remove-punctuation`, `remove-numbers`, `clean-ocr`
6. **Stopword removal** - Remove common words
7. **Lemmatization** - Choose ONE: `lemmatize-spacy` or `lemmatize`
8. **Filtering** - `filter-length` to remove too short/long texts

**Example of good ordering:**
```bash
--steps dehyphenate,normalize-transnormer,normalize-whitespace,lowercase,remove-punctuation,remove-stopwords,lemmatize-spacy
```

**Why this order?**
- Dehyphenation first prevents hyphens from interfering with normalization
- Normalization before lowercase preserves historical case patterns
- Cleaning (punctuation, numbers) before stopword removal reduces vocabulary
- Lemmatization last because it needs clean, normalized input

### Recommended Pipelines

#### Fast & Basic
```bash
--steps normalize,lowercase,remove-punctuation
```
- Good for: Quick exploration, testing
- Time: <1 minute for 100k texts

#### Balanced Quality/Speed
```bash
--steps normalize-transnormer,lowercase,remove-stopwords
```
- Good for: Most use cases
- Time: 5-30 minutes for 100k texts (GPU)

#### High Quality (Research)
```bash
--steps normalize-transnormer,lowercase,remove-punctuation,remove-stopwords,lemmatize-spacy
```
- Good for: Academic research, publications
- Time: 10-60 minutes for 100k texts (GPU)

#### Topic Modeling Optimized
```bash
--steps normalize-transnormer,lowercase,remove-punctuation,remove-numbers,remove-stopwords,lemmatize-spacy
```
- Good for: Topic modeling, keyword analysis
- Time: 10-60 minutes for 100k texts (GPU)

#### Entity Extraction Optimized
```bash
--steps normalize-transnormer,normalize-whitespace
```
- Good for: Named entity recognition (preserve case, punctuation)
- Time: 5-30 minutes for 100k texts (GPU)

---

## Normalization Methods

### Comparison Matrix

| Feature | Simple | Transnormer | DTA-CAB |
|---------|--------|-------------|---------|
| **Speed** | ⚡⚡⚡⚡⚡ Instant | ⚡⚡⚡⚡ Fast (GPU) | ⚡ Very slow (API) |
| **Quality** | ★★☆☆☆ Basic | ★★★★★ Excellent | ★★★★★ Excellent |
| **Accuracy** | ~90% | 98.88% | ~99% |
| **Offline** | ✅ Yes | ✅ Yes | ❌ No (API) |
| **Multi-GPU** | N/A | ✅ Yes | N/A |
| **Resume** | N/A | ✅ Yes | ✅ Yes |
| **Memory** | ~0 MB | ~1-2 GB | ~0 MB |
| **Model Size** | None | ~500 MB | None |
| **Dependencies** | None | transformers, torch | requests |
| **Use Case** | Quick tests | Production | Validation |

### Example Transformations

**Input (Historical German)**:
```
Die Königinn ſaß auf des Pallaſtes mittlerer Tribune und beobachtete
das Schauſpiel mit lebhaftem Intereſſe.
```

**Simple Normalization**:
```
Die Königinn sass auf des Pallastes mittlerer Tribune und beobachtete
das Schauspiel mit lebhaftem Interesse.
```
- Only character mapping
- Doesn't fix spelling errors

**Transnormer Normalization**:
```
Die Königin saß auf des Palastes mittlerer Tribüne und beobachtete
das Schauspiel mit lebhaftem Interesse.
```
- Character mapping + spelling correction
- Context-aware normalization

**DTA-CAB Normalization**:
```
Die Königin saß auf der mittleren Tribüne des Palastes und beobachtete
das Schauspiel mit lebhaftem Interesse.
```
- Character + spelling + grammar
- Most thorough but slowest

### Performance Benchmarks

**Dataset**: 100,000 newspaper text blocks (~50 words each)

| Method | CPU Time | GPU Time (1x) | GPU Time (4x) | Memory |
|--------|----------|---------------|---------------|--------|
| Simple | 2 sec | 2 sec | 2 sec | <100 MB |
| Transnormer (batch=32) | 6-12 hours | 15-30 min | 5-10 min | ~2 GB |
| Transnormer (batch=128) | 4-8 hours | 10-20 min | 3-7 min | ~4 GB |
| DTA-CAB | 20-40 hours | 20-40 hours | 20-40 hours | <500 MB |

**GPU Specifications**: NVIDIA A100 40GB

---

## Performance & Configuration

### Transnormer Optimization

#### Batch Size
- **Small (16-32)**: Lower memory, slower
- **Medium (64-128)**: Balanced (recommended)
- **Large (256+)**: Faster but needs more memory

```python
# Memory-constrained
df = transnormer(df, batch_size=16, num_beams=2)

# Balanced
df = transnormer(df, batch_size=64, num_beams=4)

# Speed-optimized (high memory)
df = transnormer(df, batch_size=256, num_beams=2)
```

#### Beam Search
- **num_beams=1**: Greedy decoding (fastest, lower quality)
- **num_beams=2-4**: Balanced (recommended)
- **num_beams=5-8**: Higher quality (slower)

```python
# Fast
df = transnormer(df, num_beams=1)

# Balanced (recommended)
df = transnormer(df, num_beams=4)

# High quality
df = transnormer(df, num_beams=8)
```

#### Multi-GPU

Transnormer supports parallel processing across multiple GPUs:

```python
# Use 4 GPUs in parallel (4x speedup)
df = transnormer(df, num_gpus=4, batch_size=128)
```

**How it works**:
- Text is split into chunks
- Each GPU processes different chunks simultaneously
- Results are combined in original order
- Nearly linear speedup (4 GPUs ≈ 4x faster)

**Requirements**:
- PyTorch with CUDA support
- Multiple CUDA-capable GPUs
- Sufficient GPU memory (2-4 GB per GPU)

#### Caching & Resume

Transnormer and DTA-CAB support caching for resume capability:

```python
# Enable caching (default)
df = transnormer(df, use_cache=True)

# Custom cache location
df = transnormer(df, use_cache=True, cache_dir=".cache/custom")

# Disable caching (reprocess everything)
df = transnormer(df, use_cache=False)
```

**Cache behavior**:
- Saves normalized texts with SHA-256 hash keys
- Automatically resumes from last processed text
- Survives process interruption
- Cache stored in `.cache/transnormer/` by default

### Memory Management

#### Chunk Processing

For very large datasets (>1M texts), process in chunks:

```python
from newspaper_explorer.data.preprocessing.normalization import transnormer
import polars as pl

df = pl.read_parquet("large_file.parquet")

chunk_size = 50000
results = []

for i in range(0, len(df), chunk_size):
    chunk = df[i:i + chunk_size]
    normalized = transnormer(chunk, batch_size=128, num_gpus=4)
    results.append(normalized)

    # Save intermediate results
    normalized.write_parquet(f"chunks/chunk_{i}.parquet")

# Combine
final_df = pl.concat(results)
```

#### GPU Memory

Estimate required GPU memory:

```
GPU Memory = model_size + (batch_size × avg_text_length × 4)
           ≈ 1.5 GB + (batch_size × 512 × 4 bytes)
```

**Examples**:
- batch_size=32: ~2.5 GB
- batch_size=64: ~3.5 GB
- batch_size=128: ~5.5 GB

### CPU vs GPU

**CPU Processing**:
- 10-100 sentences/second
- Good for small datasets (<10k texts)
- No special hardware required

**GPU Processing (1x)**:
- 100-500 sentences/second
- 10-50x faster than CPU
- Requires CUDA-capable GPU

**Multi-GPU (4x)**:
- 400-2000 sentences/second
- Near-linear speedup
- Best for large datasets (>100k texts)

---

## Data Schemas

### Input Schema

Preprocessing accepts any Polars DataFrame with a text column:

```python
{
    "text": str,                    # Required: text to process
    "source_id": str,               # Optional: preserved
    "issue_id": str,                # Optional: preserved
    "page_id": str,                 # Optional: preserved
    "text_block_id": str,           # Optional: preserved
    "line_id": str,                 # Optional: preserved
    # ... any other columns preserved ...
}
```

**Key principle**: All columns are preserved throughout preprocessing pipeline.

### Output Schema

Preprocessing adds new columns while preserving all original columns:

```python
{
    # Original columns (preserved)
    "text": str,
    "source_id": str,
    "issue_id": str,
    # ... all other original columns ...

    # New columns (added by preprocessing)
    "text_processed": str,          # Final processed text
    # Optional intermediate columns (if using direct API)
    "text_normalized": str,
    "text_lower": str,
    "text_nostop": str,
    "text_lemma": str,
}
```

**Note**: Pipeline automatically cleans up temporary columns (`_tmp_*`).

### Normalization Cache Schema

Transnormer and DTA-CAB use Parquet-based caching:

```python
{
    "text_hash": str,               # SHA-256 hash of input text
    "original_text": str,           # Input text
    "normalized_text": str,         # Output text
    "timestamp": datetime,          # When cached
}
```

**Cache location**: `.cache/transnormer/{source}/` or `.cache/dtacab/{source}/`

---

## Complete Workflows

### Workflow 1: Basic Preprocessing

**Goal**: Clean and normalize text for exploration.

```bash
# Download and parse data
newspaper-explorer data download --source der_tag
newspaper-explorer data parse --source der_tag
newspaper-explorer data aggregate --source der_tag

# Preprocess
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase,remove-punctuation

# Output: data/processed/der_tag/text/der_tag_preprocessed_normalize_lowercase_remove-punctuation.parquet
```

### Workflow 2: High-Quality Normalization for NLP

**Goal**: Prepare text for entity extraction, topic modeling.

```bash
# Full preprocessing pipeline
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer,lowercase,remove-stopwords,lemmatize-spacy \
    --batch-size 128 \
    --num-gpus 4

# Now run analysis
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_*.parquet

newspaper-explorer analyze topics extract \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_*.parquet
```

### Workflow 3: Iterative Development

**Goal**: Test different preprocessing configurations.

```bash
# Test on sample
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase \
    --sample 1000 \
    --output test_output.parquet

# Compare normalization methods
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize \
    --sample 5000 \
    --output simple_norm.parquet

newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --sample 5000 \
    --output transnormer_norm.parquet

# Analyze differences in Python
# (see Python API examples below)
```

### Workflow 4: Multi-Stage Processing

**Goal**: Process in stages for flexibility.

```bash
# Stage 1: Normalization only
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --output-column text_normalized \
    --batch-size 128 \
    --num-gpus 4

# Stage 2: Cleaning (use normalized text as input)
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_normalize-transnormer.parquet \
    --text-column text_normalized \
    --steps lowercase,remove-punctuation,remove-stopwords \
    --output-column text_clean

# Stage 3: Lemmatization (use cleaned text as input)
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_lowercase_remove-punctuation_remove-stopwords.parquet \
    --text-column text_clean \
    --steps lemmatize-spacy \
    --output-column text_lemma
```

### Workflow 5: Python API Integration

**Goal**: Custom preprocessing with full control.

```python
import polars as pl
from newspaper_explorer.data.loading.loader import DataLoader
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

# Load data
loader = DataLoader(source_name="der_tag")
df = loader.load_source()  # or load_parquet()

# Sample for testing
sample_df = df.head(1000)

# Test different pipelines
preprocessor = TextPreprocessor(text_column="text")

# Pipeline 1: Basic
df1 = preprocessor.pipeline(
    sample_df,
    steps=["normalize", "lowercase"],
    output_column="text_basic"
)

# Pipeline 2: Advanced
df2 = preprocessor.pipeline(
    sample_df,
    steps=["normalize-transnormer", "lowercase", "remove-stopwords", "lemmatize-spacy"],
    output_column="text_advanced",
    batch_size=64,
    num_gpus=2
)

# Compare results
print("\n=== COMPARISON ===\n")
for i in range(3):
    row = df2[i]
    print(f"Original:  {row['text']}")
    print(f"Basic:     {row['text_basic']}")
    print(f"Advanced:  {row['text_advanced']}")
    print()

# Choose best and process full dataset
best_df = preprocessor.pipeline(
    df,
    steps=["normalize-transnormer", "lowercase", "remove-stopwords", "lemmatize-spacy"],
    output_column="text_processed",
    batch_size=128,
    num_gpus=4
)

# Save
best_df.write_parquet("data/processed/der_tag/text/final_preprocessed.parquet")
```

---

## Analyzing Results

### Compare Normalization Methods

```python
import polars as pl

# Load results from different methods
simple_df = pl.read_parquet("simple_norm.parquet")
transnormer_df = pl.read_parquet("transnormer_norm.parquet")

# Compare outputs
comparison = pl.DataFrame({
    "original": simple_df["text"],
    "simple": simple_df["text_normalized"],
    "transnormer": transnormer_df["text_normalized"]
})

print(comparison.head(10))

# Find differences
different = comparison.filter(
    pl.col("simple") != pl.col("transnormer")
)
print(f"\nDifferent normalizations: {len(different)}/{len(comparison)}")
```

### Analyze Text Statistics

```python
import polars as pl

df = pl.read_parquet("preprocessed.parquet")

# Before preprocessing
print("=== BEFORE ===")
print(f"Avg length: {df['text'].str.len_chars().mean():.0f}")
print(f"Avg words: {df['text'].str.count_matches(r'\w+').mean():.0f}")

# After preprocessing
print("\n=== AFTER ===")
print(f"Avg length: {df['text_processed'].str.len_chars().mean():.0f}")
print(f"Avg words: {df['text_processed'].str.count_matches(r'\w+').mean():.0f}")

# Length distribution
print("\n=== LENGTH DISTRIBUTION ===")
print(df['text_processed'].str.len_chars().describe())
```

### Visualize Changes

```python
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("preprocessed.parquet")

# Character length comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before
axes[0].hist(df['text'].str.len_chars(), bins=50)
axes[0].set_title("Before Preprocessing")
axes[0].set_xlabel("Character Length")
axes[0].set_ylabel("Frequency")

# After
axes[1].hist(df['text_processed'].str.len_chars(), bins=50)
axes[1].set_title("After Preprocessing")
axes[1].set_xlabel("Character Length")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=300)
```

### Word Frequency Analysis

```python
import polars as pl
from collections import Counter

df = pl.read_parquet("preprocessed.parquet")

# Extract all words
all_words = []
for text in df['text_processed']:
    if text:
        all_words.extend(text.split())

# Count frequencies
word_freq = Counter(all_words)

# Top 20 most common
print("=== TOP 20 WORDS ===")
for word, count in word_freq.most_common(20):
    print(f"{word:20s} {count:8d}")
```

---

## Integration with Other Analyses

### Entity Extraction

```bash
# Preprocess (preserve case and punctuation for NER)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer,normalize-whitespace \
    --output-column text_normalized

# Extract entities from normalized text
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_*.parquet \
    --text-column text_normalized
```

### Topic Modeling

```bash
# Preprocess (aggressive cleaning for topic modeling)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer,lowercase,remove-punctuation,remove-numbers,remove-stopwords,lemmatize-spacy \
    --output-column text_clean

# Extract topics from cleaned text
newspaper-explorer analyze topics extract \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_*.parquet \
    --text-column text_clean
```

### Emotion Analysis

```bash
# Preprocess (moderate cleaning for emotion analysis)
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer,lowercase \
    --output-column text_normalized

# Analyze emotions from normalized text
newspaper-explorer analyze emotions predict \
    --source der_tag \
    --input data/processed/der_tag/text/der_tag_preprocessed_*.parquet \
    --text-column text_normalized
```

### Query Engine

```python
from newspaper_explorer.analysis.query.engine import QueryEngine

# Query preprocessed data
engine = QueryEngine(source="der_tag")

# Query using preprocessed text
results = engine.query("""
    SELECT
        text,
        text_processed,
        LENGTH(text) - LENGTH(text_processed) as char_reduction
    FROM preprocessed
    WHERE char_reduction > 100
    ORDER BY char_reduction DESC
    LIMIT 10
""")

print(results)
```

---

## Troubleshooting

### Common Issues

#### 1. Transnormer Model Download Fails

**Problem**: Model download times out or fails.

**Solution**:
```python
# Manually download model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "ybracke/transnormer-19c-beta-v02"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Now preprocessing will use cached model
```

#### 2. Out of Memory (GPU)

**Problem**: CUDA out of memory error.

**Solution**:
```bash
# Reduce batch size
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer \
    --batch-size 16  # Smaller batches

# Or process in chunks (Python)
```

#### 3. Out of Memory (CPU)

**Problem**: System memory exhausted.

**Solution**:
```python
# Process in smaller chunks
import polars as pl
from newspaper_explorer.data.preprocessing.normalization import transnormer

df = pl.read_parquet("large_file.parquet")

chunk_size = 10000
results = []

for i in range(0, len(df), chunk_size):
    chunk = df[i:i + chunk_size]
    normalized = transnormer(chunk, batch_size=32)
    results.append(normalized)

final_df = pl.concat(results)
```

#### 4. Slow Processing

**Problem**: Preprocessing takes too long.

**Solutions**:
```bash
# Use GPU instead of CPU
# Check: nvidia-smi

# Increase batch size (if memory allows)
--batch-size 128

# Reduce beam search
--num-beams 2

# Use multi-GPU
--num-gpus 4

# Use simpler normalization
--steps normalize  # instead of normalize-transnormer
```

#### 5. spaCy Model Not Found

**Problem**: `OSError: [E050] Can't find model 'de_core_news_sm'`

**Solution**:
```bash
# Download spaCy model
python -m spacy download de_core_news_sm

# Or use larger model
python -m spacy download de_core_news_md
```

#### 6. Cache Not Working

**Problem**: Transnormer reprocesses everything despite caching.

**Solution**:
```bash
# Check cache directory exists
ls -la .cache/transnormer/

# Explicitly enable cache
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-transnormer
    # (cache is enabled by default, --no-cache disables it)

# Clear cache if corrupted
rm -rf .cache/transnormer/
```

### Performance Diagnostics

#### Check GPU Availability
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

#### Monitor GPU Usage
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use Python
```python
import torch
torch.cuda.memory_summary()
```

#### Benchmark Processing Speed
```python
import time
import polars as pl
from newspaper_explorer.data.preprocessing.normalization import transnormer

# Load sample
df = pl.read_parquet("data.parquet").head(1000)

# Benchmark
start = time.time()
df_norm = transnormer(df, batch_size=32, num_gpus=1)
elapsed = time.time() - start

print(f"Processed {len(df)} texts in {elapsed:.1f}s")
print(f"Speed: {len(df)/elapsed:.1f} texts/sec")
```

---

## Testing

### Unit Tests

```bash
# Run all preprocessing tests
pytest tests/data/preprocessing/ -v

# Run specific test
pytest tests/data/preprocessing/test_pipeline.py::test_basic_pipeline -v

# Run with coverage
pytest tests/data/preprocessing/ --cov=newspaper_explorer.data.preprocessing
```

### Manual Testing

#### Test Pipeline
```python
import polars as pl
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

# Create test data
df = pl.DataFrame({
    "text": [
        "Die Königinn ſaß auf des Pallaſtes Tribune.",
        "Das Schauſpiel war ſehr intereſſant.",
        "Viele Menſchen waren anweſend."
    ]
})

# Test pipeline
preprocessor = TextPreprocessor()
result = preprocessor.pipeline(
    df,
    steps=["normalize", "lowercase"],
    output_column="text_processed"
)

print(result.select(["text", "text_processed"]))
```

#### Test Normalization
```python
from newspaper_explorer.data.preprocessing.normalization import simple, transnormer

df = pl.DataFrame({"text": ["Das iſt ein Beiſpiel"]})

# Test simple
simple_result = simple(df)
print(simple_result)

# Test transnormer (requires GPU)
transnormer_result = transnormer(df, batch_size=1)
print(transnormer_result)
```

### Integration Tests

```bash
# Test full workflow
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase \
    --sample 100 \
    --output test_output.parquet

# Verify output
python -c "
import polars as pl
df = pl.read_parquet('test_output.parquet')
print(df.columns)
print(df.head())
"
```

---

## Dependencies

### Core Dependencies
- **polars** - DataFrame operations
- **numpy** - Numerical operations

### Normalization
- **transformers** - Transnormer models
- **torch** - PyTorch backend for Transnormer
- **requests** - DTA-CAB API calls

### Cleaning
- **unidecode** - Diacritic removal

### Filtering
- **spacy** - Stopword removal

### Linguistic
- **pyphen** - Dehyphenation
- **spacy** - Fast lemmatization
- **germalemma** - Thorough lemmatization

### Installation

```bash
# Core + normalization (Transnormer)
pip install -e ".[normalize]"

# Core + all preprocessing
pip install -e ".[nlp]"

# Development (includes all)
pip install -e ".[dev]"
```

### Optional Models

```bash
# spaCy German models
python -m spacy download de_core_news_sm  # Small (40 MB)
python -m spacy download de_core_news_md  # Medium (500 MB)
python -m spacy download de_core_news_lg  # Large (800 MB)

# Transnormer models (auto-downloaded on first use)
# - ybracke/transnormer-19c-beta-v02 (~500 MB)
# - ybracke/transnormer-18-19c-beta-v01 (~500 MB)
```

---

## See Also

- **[NORMALIZATION.md](NORMALIZATION.md)** - Detailed normalization guide
- **[DATA_LOADER.md](../DATA_LOADER.md)** - Data loading documentation
- **[ENTITIES.md](../../analysis/ENTITIES.md)** - Entity extraction
- **[TOPICS.md](../../analysis/TOPICS.md)** - Topic modeling
- **[EMOTIONS.md](../../analysis/EMOTIONS.md)** - Emotion analysis
- **[CLI.md](../../cli/CLI.md)** - Complete CLI reference

---

## Citation

If you use the Transnormer models in research, please cite:

```bibtex
@misc{transnormer2024,
  author = {Bawden, Rachel and Bracke, Yannik and Khetan, Vivek and Schopper, Daniel and Herrmann, J. Berenike and Pielström, Steffen},
  title = {Transnormer: A Tool for Automatic Normalization of Historical German Texts},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/ybracke/transnormer-19c-beta-v02}
}
```

For DTA-CAB, please cite the CLARIN-D service:

```bibtex
@misc{dtacab,
  author = {Berlin-Brandenburg Academy of Sciences and Humanities},
  title = {DTA::CAB - Deutsches Textarchiv: Cascaded Analysis Broker},
  url = {https://kaskade.dwds.de/dstar/cab/}
}
```
