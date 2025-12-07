# Text Normalization

> **📚 For comprehensive preprocessing documentation, see [PREPROCESSING.md](PREPROCESSING.md)**
> This document focuses specifically on normalization methods.

Historical German newspaper text often uses outdated spelling conventions and characters that can pose challenges for modern NLP analysis. The Newspaper Explorer supports multiple normalization approaches, from simple character replacement to state-of-the-art neural models.

## Overview

Text normalization converts historical German text to modern orthography, handling:

- **Outdated characters**: `ſ` → `s`, `aͤ` → `ä`, etc.
- **Historical spelling**: `Königinn` → `Königin`, `Pallaſtes` → `Palastes`
- **Grammatical modernization**: Historical verb forms and grammar

## Historical Long S (ſ) Normalization

The **long s** (ſ, U+017F, "langes s", "Schaft-s") is a positional variant of the letter "s" used in Fraktur typography and German Kurrent handwriting until 1941.

### Historical Background

The distinction between **long s (ſ)** and **round s** (Schluss-s, Auslaut-s) follows rules codified at the **1901 Orthographic Conference**:

- **Long s (ſ)**: Used at syllable onset (Anlaut) and within syllables (Inlaut)
- **Round s**: Used at syllable endings (Auslaut)

This distinction aids readability of compound words. For example:
- **Wachſtube** (Wach·stube) = guard room
- **Wachstube** (Wachs·tube) = wax tube

The ligatures **ſʒ** (ſz) and **ſs** are considered the origin of the German **Eszett (ß)**.

### 1901 Orthographic Rules

**Round s (Schluss-s) is used:**

| Position | Examples |
|----------|----------|
| Word end | das Haus, der Kosmos, des Bundes |
| End of prefixes/compounds (Fugen-s) | Liebesbrief, Arbeitsamt, Haustür, Wirtsstube |
| Before consonant-initial suffixes | Mäuschen, Weisheit, Wachstum (-chen, -heit, -tum) |
| Syllable-final before k, m, n, w, d | Dresden, Oswald, Schleswig |

**Long s (ſ) is used:**

| Position | Examples |
|----------|----------|
| Syllable onset (Anlaut) | ſauſen, einſpielen, ausſpielen |
| Within syllables (Inlaut) | Maſuren, Hauſe |
| In ſch, ſt, ſp combinations | Weſpe, faſten, Buſch, löſchen |
| Digraph ſſ | Waſſer, Biſſen, Gaſſe |
| Before l, n, r (elided e) | unſre, Pilſner, Wechſler |
| Before apostrophe | ich laſſ' |

### Normalization Modes

The `normalize_long_s()` function provides two modes:

```python
from newspaper_explorer.data.preprocessing.normalization import normalize_long_s

# Mode 1: Simple (recommended for NLP)
# Fast, replaces all ſ → s unconditionally
df = normalize_long_s(df, mode="simple")
# "Hauſe" → "Hause", "Waſſer" → "Wasser"

# Mode 2: Context-aware (1901 rules)
# Implements historical rules, useful for scholarly work
df = normalize_long_s(df, mode="context-aware")
# Processes ſch, ſt, ſp, ſſ combinations correctly
```

If you need to preserve the original long s characters for archival purposes, simply don't apply this normalization step.

### Common OCR Issues

- **ſ misread as f**: "Haufe" instead of "Hauſe" (very common in Fraktur OCR)
- **Round s as ſ**: OCR may produce ſ where round s should appear
- **f misread as ſ**: Less common but possible

The context-aware mode helps identify potential OCR errors by applying proper positional rules.

---

## Neural Normalization (Transnormer)

## Performance

The Transnormer models achieve excellent accuracy on historical German text:

| Model                       | Time Period | Word Accuracy | Word Accuracy (case-insensitive) |
| --------------------------- | ----------- | ------------- | -------------------------------- |
| transnormer-19c-beta-v02    | 1780-1899   | 98.88%        | 99.34%                           |
| transnormer-18-19c-beta-v01 | 1700-1899   | 99.46%        | 99.53%                           |

Compared to a simple character replacement baseline (88-95% accuracy), these models provide significantly better results.

## Installation

Install the normalization dependencies:

```bash
# Using pip
pip install newspaper-explorer[normalize]

# Or install transformers and torch manually
pip install transformers torch
```

**Note**: The models are ~500MB and will be downloaded on first use. Subsequent uses will use the cached model.

## Available Models

### 19th Century Model (`19c`)

- **Full name**: `ybracke/transnormer-19c-beta-v02`
- **Time period**: 1780-1899
- **Best for**: Most 19th century German newspapers

### 18th-19th Century Model (`18-19c`)

- **Full name**: `ybracke/transnormer-18-19c-beta-v01`
- **Time period**: 1700-1899
- **Best for**: Earlier newspapers or mixed period corpora

## Usage

### Method 1: Integrated Pipeline

The easiest way is to use the all-in-one function:

```python
from newspaper_explorer.data.utils.text import load_aggregate_and_split

# Process and normalize in one call
df = load_aggregate_and_split(
    "data/raw/der_tag/text/lines.parquet",
    normalize=True,              # Enable normalization
    normalize_model="19c",       # Use 19th century model
    save_path="data/processed/der_tag_normalized.parquet"
)

# Result has both original and normalized text
print(df.select(["sentence", "normalized_text"]).head())
```

### Method 2: Standalone Normalization

Normalize any DataFrame with text:

```python
import polars as pl
from newspaper_explorer.data.utils.text import normalize_text

# Your data with text
df = pl.DataFrame({
    "text": [
        "Die Königinn ſaß auf des Pallaſtes mittlerer Tribune.",
        "Die Preußiſche Armee marſchirte durch die Stadt."
    ]
})

# Normalize
normalized_df = normalize_text(
    df,
    text_column="text",
    model="19c",                 # or "18-19c"
    batch_size=32,               # Process 32 texts at once
    num_beams=4,                 # Quality vs speed tradeoff
)

print(normalized_df.select(["text", "normalized_text"]))
```

### Method 3: Custom Hugging Face Model

You can also use any compatible model from Hugging Face:

```python
normalized_df = normalize_text(
    df,
    text_column="text",
    model="your-username/your-model-name"
)
```

## Parameters

### `normalize_text()`

- **`df`**: Polars DataFrame with text to normalize
- **`text_column`**: Column name containing the text (default: `"sentence"`)
- **`model`**: Model to use - `"19c"`, `"18-19c"`, or full Hugging Face model name
- **`batch_size`**: Number of texts to process at once (default: 32)
  - Larger = faster but more memory
  - Smaller = slower but less memory
- **`num_beams`**: Beam search parameter (default: 4)
  - Higher = better quality but slower (1-8 recommended)
  - Lower = faster but potentially lower quality
- **`max_length`**: Maximum length of output (default: 128 tokens)
- **`output_column`**: Name for normalized text column (default: `"normalized_text"`)

### `load_aggregate_and_split()`

All the above parameters, plus:

- **`normalize`**: Set to `True` to enable normalization (default: `False`)
- **`normalize_model`**: Which model to use (default: `"19c"`)
- **`normalize_batch_size`**: Batch size for normalization (default: 32)

## Performance Tips

### CPU vs GPU

By default, the normalization runs on CPU. For GPU acceleration:

1. Install CUDA-compatible PyTorch
2. The code will need to be modified to use `device=0` instead of `device=-1`

### Batch Size

- **Small datasets (<10k sentences)**: Use `batch_size=32`
- **Medium datasets (10k-100k)**: Use `batch_size=64` with GPU or `batch_size=16` with CPU
- **Large datasets (>100k)**: Process in chunks and save intermediate results

### Memory Management

For very large datasets, process in chunks:

```python
import polars as pl
from newspaper_explorer.data.utils.text import normalize_text

# Load data
df = pl.read_parquet("large_file.parquet")

# Process in chunks
chunk_size = 10000
results = []

for i in range(0, len(df), chunk_size):
    chunk = df[i:i + chunk_size]
    normalized_chunk = normalize_text(chunk, model="19c", batch_size=32)
    results.append(normalized_chunk)

# Combine results
final_df = pl.concat(results)
```

## Examples

See `examples/normalize_text.py` for complete examples:

```bash
python examples/normalize_text.py
```

## Example Output

**Original historical text:**

```
Die Königinn ſaß auf des Pallaſtes mittlerer Tribune.
```

**Normalized modern text:**

```
Die Königin saß auf des Palastes mittlerer Tribüne.
```

## Use Cases

Normalization is particularly valuable for:

1. **Named Entity Recognition**: Modern NER models work better with modern spelling
2. **Topic Modeling**: Reduces vocabulary fragmentation
3. **Search**: Find modern queries in historical text
4. **Cross-temporal Analysis**: Compare historical and modern texts
5. **Machine Translation**: Improves translation quality
6. **Text Mining**: More accurate pattern matching

## Limitations

- **Processing time**: Normalization is computationally intensive
  - ~10-100 sentences/second on CPU
  - ~100-500 sentences/second on GPU
- **Model coverage**: Optimized for 1700-1899 German text
- **Accuracy**: While very high (>98%), some errors may occur
- **Memory**: Models require ~1-2GB RAM when loaded

## Citation

If you use the Transnormer models in research, please cite:

```bibtex
@misc{transnormer2024,
  author = {Bawden, Rachel and Bracke, Yannik and others},
  title = {Transnormer: Historical Text Normalization for German},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/ybracke/transnormer-19c-beta-v02}
}
```

## Troubleshooting

### Model Download Issues

If the model fails to download:

```python
# Manually download first
from transformers import pipeline
pipeline(model='ybracke/transnormer-19c-beta-v02')
```

### Memory Errors

Reduce batch size:

```python
normalized_df = normalize_text(df, model="19c", batch_size=8)
```

### Slow Performance

- Use GPU if available
- Reduce `num_beams` (e.g., to 2)
- Increase `batch_size` (if memory allows)

## Related Documentation

- **[PREPROCESSING.md](PREPROCESSING.md)** - Complete preprocessing guide with all methods
- [Text Processing](TEXT_PROCESSING.md) - Text cleaning and processing
- [Data Loading](LOADING.md) - Loading newspaper data
- [Configuration](CONFIGURATION_PHILOSOPHY.md) - Environment setup
