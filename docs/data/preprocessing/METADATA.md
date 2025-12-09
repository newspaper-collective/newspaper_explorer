# Preprocessing Metadata System

**Module:** `newspaper_explorer.data.utils.metadata` and `newspaper_explorer.models.data.metadata`
**Status:** Production-ready
**Purpose:** Track preprocessing steps for full reproducibility and provenance

> **For comprehensive preprocessing documentation, see [PREPROCESSING.md](PREPROCESSING.md)**
> This document focuses specifically on the metadata tracking system.

---

## Overview

The preprocessing metadata system provides:

1. **Full Reproducibility** - Every preprocessing run saves complete metadata
2. **Chaining Support** - Can use preprocessed data as input and track the full history
3. **Provenance Tracking** - Analysis results can trace back to preprocessing steps
4. **Standardized Storage** - Same pattern as analysis results (subdirectories with parquet + JSON)

### File Structure

```
src/newspaper_explorer/
├── data/
│   ├── preprocessing/
│   │   └── pipeline.py          # TextPreprocessor.run() saves metadata
│   └── utils/
│       ├── metadata.py          # find_metadata_for_parquet(), load_metadata()
│       └── results.py           # save_preprocessing_results()
└── models/
    └── data/
        └── metadata.py          # PreprocessingMetadata, PreprocessingResult models
```

---

## Quick Start

### Basic Preprocessing with Metadata

```bash
# Run preprocessing - automatically saves metadata
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize_unicode,normalize_casing,remove_stopwords \
    --sample 1000

# Output structure:
# data/processed/der_tag/text/normalize_unicode_normalize_casing_remove_stopwords_20251110_120000/
#   ├── textblocks.parquet
#   └── textblocks.json  # Preprocessing metadata
```

### Chaining Preprocessing Steps

```bash
# First preprocessing
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize_unicode,normalize_casing \
    --output data/processed/der_tag/text/step1/textblocks.parquet

# Second preprocessing (uses first as input)
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/step1/textblocks.parquet \
    --steps remove_stopwords,lemmatize_spacy

# The metadata automatically chains steps:
# All steps: ["normalize_unicode", "normalize_casing", "remove_stopwords", "lemmatize_spacy"]
```

---

## Metadata Schema

### PreprocessingMetadata

```python
{
  "preprocessing_id": "normalize_unicode_normalize_casing_remove_stopwords_20251110_120000",
  "source": "der_tag",
  "steps": ["normalize_unicode", "normalize_casing", "remove_stopwords"],
  "parameters": {
    "text_column": "text",
    "output_column": "text_processed",
    "batch_size": 32,
    "num_beams": 4
  },
  "created_at": "2025-11-10T12:00:00.000000",
  "completed_at": "2025-11-10T12:00:45.123456",
  "duration_seconds": 45.12,
  "input_data": {
    "row_count": 100000,
    "columns": ["line_id", "text", "date", ...],
    "date_range": ["1901-01-08", "1920-12-31"]
  },
  "output_data": {
    "row_count": 98500,
    "columns": ["line_id", "text", "text_processed", "date", ...]
  },
  "previous_preprocessing": {
    "preprocessing_id": "normalize_unicode_20251110_110000",
    "steps": ["normalize_unicode"],
    "parameters": {...}
  },
  "status": "completed"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source dataset name (e.g., "der_tag") |
| `steps` | array | List of preprocessing steps applied in order |
| `parameters` | object | Processing parameters (columns, batch sizes, etc.) |

### Auto-Generated Fields

| Field | Type | Description |
|-------|------|-------------|
| `preprocessing_id` | string | Unique identifier (auto-generated from steps + timestamp) |
| `created_at` | string | ISO timestamp of creation |
| `completed_at` | string | ISO timestamp of completion |
| `duration_seconds` | float | Processing time in seconds |
| `input_data` | object | Input dataset statistics |
| `output_data` | object | Output dataset statistics |

### Chaining Fields

| Field | Type | Description |
|-------|------|-------------|
| `previous_preprocessing` | object | Previous preprocessing metadata (if input was preprocessed) |

---

## File Organization

### Directory Structure

```
data/processed/{source}/text/{preprocessing_id}/
├── textblocks.parquet      # Preprocessed data
└── textblocks.json         # Preprocessing metadata
```

### Example

```
data/processed/der_tag/text/
├── normalize_lowercase_20251110_120000/
│   ├── textblocks.parquet
│   └── textblocks.json
└── normalize_lowercase_remove_stopwords_20251110_130000/
    ├── textblocks.parquet
    └── textblocks.json
```

---

## Usage Examples

### 1. Programmatic Usage

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

# Simple usage - handles everything automatically
result = TextPreprocessor(source="der_tag").run(
    steps=["normalize_unicode", "normalize_casing", "remove_stopwords"],
)

print(f"Processed {result.input_rows} → {result.output_rows} rows")
print(f"Output: {result.results_path}")
print(f"Preprocessing ID: {result.metadata.preprocessing_id}")
```

### 2. Chaining Preprocessing

When you run preprocessing on already-preprocessed data, the metadata is automatically
chained to track the full pipeline history:

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

# First preprocessing run
preprocessor = TextPreprocessor(source="der_tag")
result1 = preprocessor.run(
    steps=["normalize_unicode", "normalize_casing"],
)

# Second preprocessing run (on the output of the first)
result2 = TextPreprocessor(source="der_tag").run(
    steps=["lemmatize_spacy"],
    input_path=result1.results_path,  # Use output from first run
)

# Get all steps (including previous)
all_steps = result2.metadata.get_all_steps()
print(f"Complete pipeline: {all_steps}")
# Output: ['normalize_unicode', 'normalize_casing', 'lemmatize_spacy']
```

### 3. Using Preprocessed Data in Analysis

```python
from newspaper_explorer.data.utils.metadata import (
    AnalysisMetadata,
    extract_input_stats,
)
import polars as pl

# Load preprocessed data
input_path = "data/processed/der_tag/text/normalize_lowercase_20251110/textblocks.parquet"
df = pl.read_parquet(input_path)

# Extract input stats (automatically includes preprocessing metadata)
input_stats = extract_input_stats(df, input_path=input_path)

# Check preprocessing info
if "preprocessing" in input_stats:
    preprocessing = input_stats["preprocessing"]
    print(f"Input was preprocessed with ID: {preprocessing['preprocessing_id']}")
    print(f"Steps applied: {preprocessing['steps']}")

# Create analysis metadata with preprocessing provenance
metadata = AnalysisMetadata(
    analysis_type="entities",
    method_type="gliner",
    model_name="multi-v2.1",
    source="der_tag",
    parameters={"threshold": 0.2},
    input_data=input_stats,  # Includes preprocessing info!
)

# When saved, metadata.json will contain full preprocessing history
```

### 4. CLI Workflow

```bash
# Step 1: Basic normalization
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize_unicode,normalize_casing \
    --sample 1000

# Note the preprocessing ID in output:
# Preprocessing ID: normalize_unicode_normalize_casing_20251110_120000

# Step 2: Use preprocessed data as input for more processing
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/normalize_unicode_normalize_casing_20251110_120000/textblocks.parquet \
    --steps remove_stopwords,lemmatize_spacy

# Output will show chained steps:
# Previous preprocessing: normalize_unicode_normalize_casing_20251110_120000 (2 steps)
# New steps: 2
# Total pipeline: 4 steps
```

---

## Integration with Analysis Results

When you run analysis on preprocessed data, the analysis metadata automatically includes the preprocessing history:

```python
# Analysis metadata includes preprocessing provenance
{
  "analysis_id": "gliner_multi_v2_1_20251110_140000",
  "analysis_type": "entities",
  "source": "der_tag",
  "input_data": {
    "row_count": 98500,
    "preprocessing": {
      "preprocessing_id": "normalize_unicode_normalize_casing_remove_stopwords_20251110_120000",
      "steps": ["normalize_unicode", "normalize_casing", "remove_stopwords"],
      "parameters": {...},
      "metadata_path": "data/processed/der_tag/text/.../textblocks.json"
    }
  }
}
```

This enables:
- **Full reproducibility** - Know exactly what preprocessing was applied
- **UI display** - Show preprocessing info in the UI alongside analysis results
- **Provenance tracking** - Trace results back through preprocessing to raw data
- **Research transparency** - Document complete analysis pipeline

---

## Benefits

### For Reproducibility

Every preprocessing run is fully documented:
- Exact steps applied in order
- All parameters used
- Input/output statistics
- Processing time
- Chained preprocessing history

### For Analysis

Analysis metadata automatically includes preprocessing provenance:
- Know if input was normalized, lemmatized, etc.
- Track which preprocessing version was used
- Compare analysis results across different preprocessing pipelines

### For UI

UI can display:
- Which preprocessing was applied to input data
- Complete processing pipeline (raw → preprocessed → analysis)
- Links to preprocessing metadata for details

---

## Best Practices

### 1. Always Use Output Path from CLI

The CLI returns the preprocessing ID - use it for subsequent processing:

```bash
# First preprocessing
newspaper-explorer data preprocess --source der_tag --steps normalize,lowercase

# Output shows:
# Preprocessing ID: normalize_lowercase_20251110_120000
# File saved: data/processed/der_tag/text/normalize_lowercase_20251110_120000/textblocks.parquet

# Use this path for analysis:
newspaper-explorer analyze entities extract \
    --input data/processed/der_tag/text/normalize_unicode_normalize_casing_20251110_120000/textblocks.parquet
```

### 2. Keep Preprocessing Steps Modular

Instead of one giant preprocessing run, break into logical stages:

```bash
# Stage 1: Normalization
newspaper-explorer data preprocess --source der_tag --steps normalize_unicode,normalize_casing

# Stage 2: Linguistic processing (can be slow, keep separate)
newspaper-explorer data preprocess --source der_tag \
    --input <stage1_output> \
    --steps lemmatize_spacy

# Stage 3: Analysis-specific preprocessing
newspaper-explorer data preprocess --source der_tag \
    --input <stage2_output> \
    --steps remove_stopwords
```

### 3. Use Metadata for Analysis Configuration

When running analysis, extract preprocessing info to adapt parameters:

```python
input_stats = extract_input_stats(df, input_path=input_path)
if "preprocessing" in input_stats:
    steps = input_stats["preprocessing"]["steps"]
    if "lemmatize_spacy" in steps:
        # Use lemmatized text column
        text_column = "text_lemma"
    else:
        text_column = "text"
```

---

## See Also

- **[PREPROCESSING.md](PREPROCESSING.md)** - Complete preprocessing guide
- **[NORMALIZATION.md](NORMALIZATION.md)** - Text normalization methods
- **[QUALITY_VALIDATION.md](QUALITY_VALIDATION.md)** - OCR quality validation
- **[WORDLISTS.md](WORDLISTS.md)** - German wordlists for OOV calculation
