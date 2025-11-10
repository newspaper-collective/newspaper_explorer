# Preprocessing Metadata System

> **Status**: Production-ready (November 2025)  
> **Implementation**: `src/newspaper_explorer/data/utils/metadata.py`  
> **Pipeline**: `src/newspaper_explorer/data/preprocessing/pipeline.py`

This document describes the preprocessing metadata system that tracks all preprocessing steps applied to newspaper text data, enabling full reproducibility and provenance tracking.

---

## Overview

The preprocessing metadata system provides:

1. **Full Reproducibility** - Every preprocessing run saves complete metadata
2. **Chaining Support** - Can use preprocessed data as input and track the full history
3. **Provenance Tracking** - Analysis results can trace back to preprocessing steps
4. **Standardized Storage** - Same pattern as analysis results (subdirectories with parquet + JSON)

---

## Quick Start

### Basic Preprocessing with Metadata

```bash
# Run preprocessing - automatically saves metadata
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase,remove-stopwords \
    --sample 1000

# Output structure:
# data/processed/der_tag/text/normalize_lowercase_remove_stopwords_20251110_120000/
#   ├── textblocks.parquet
#   └── textblocks.json  # Preprocessing metadata
```

### Chaining Preprocessing Steps

```bash
# First preprocessing
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase \
    --output data/processed/der_tag/text/step1/textblocks.parquet

# Second preprocessing (uses first as input)
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/step1/textblocks.parquet \
    --steps remove-stopwords,lemmatize-spacy

# The metadata automatically chains steps:
# All steps: ["normalize", "lowercase", "remove-stopwords", "lemmatize-spacy"]
```

---

## Metadata Schema

### PreprocessingMetadata

```python
{
  "preprocessing_id": "normalize_lowercase_remove_stopwords_20251110_120000",
  "source": "der_tag",
  "steps": ["normalize", "lowercase", "remove-stopwords"],
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
    "preprocessing_id": "normalize_20251110_110000",
    "steps": ["normalize"],
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
from pathlib import Path
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
from newspaper_explorer.data.utils.metadata import save_preprocessing_results
from newspaper_explorer.config.base import get_config
import polars as pl
import time

# Setup
config = get_config()
source_name = "der_tag"

# Load raw data
input_path = config.data_dir / "raw" / source_name / "text" / "textblocks.parquet"
df = pl.read_parquet(input_path)

# Apply preprocessing
preprocessor = TextPreprocessor(text_column="text")
start_time = time.time()

processed_df = preprocessor.pipeline(
    df,
    steps=["normalize", "lowercase", "remove-stopwords"],
    output_column="text_processed",
)

duration = time.time() - start_time

# Create metadata
metadata = preprocessor.create_metadata(
    source=source_name,
    steps=["normalize", "lowercase", "remove-stopwords"],
    parameters={
        "text_column": "text",
        "output_column": "text_processed",
    },
    input_df=df,
    output_df=processed_df,
    duration_seconds=duration,
)

# Save with metadata
processed_base = config.data_dir / "processed"
paths = save_preprocessing_results(
    results_df=processed_df,
    metadata=metadata,
    processed_base_dir=processed_base,
)

print(f"Saved to: {paths['output_dir']}")
print(f"Preprocessing ID: {metadata.preprocessing_id}")
```

### 2. Chaining Preprocessing

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
import polars as pl

preprocessor = TextPreprocessor()

# Load previously preprocessed data
input_path = "data/processed/der_tag/text/normalize_lowercase_20251110/textblocks.parquet"
df = pl.read_parquet(input_path)

# Load previous metadata
previous_metadata = preprocessor.load_previous_metadata(input_path)

# Apply additional preprocessing
processed_df = preprocessor.pipeline(
    df,
    steps=["lemmatize-spacy"],
    output_column="text_lemma",
)

# Create chained metadata
metadata = preprocessor.create_metadata(
    source="der_tag",
    steps=["lemmatize-spacy"],
    parameters={"text_column": "text_processed"},
    input_df=df,
    output_df=processed_df,
    duration_seconds=42.5,
    previous_preprocessing=previous_metadata,  # Chain the metadata
)

# Get all steps (including previous)
all_steps = metadata.get_all_steps()
print(f"Complete pipeline: {all_steps}")
# Output: ['normalize', 'lowercase', 'lemmatize-spacy']
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
    --steps normalize,lowercase \
    --sample 1000

# Note the preprocessing ID in output:
# Preprocessing ID: normalize_lowercase_20251110_120000

# Step 2: Use preprocessed data as input for more processing
newspaper-explorer data preprocess \
    --source der_tag \
    --input data/processed/der_tag/text/normalize_lowercase_20251110_120000/textblocks.parquet \
    --steps remove-stopwords,lemmatize-spacy

# Output will show chained steps:
# Previous preprocessing: normalize_lowercase_20251110_120000 (2 steps)
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
      "preprocessing_id": "normalize_lowercase_remove_stopwords_20251110_120000",
      "steps": ["normalize", "lowercase", "remove-stopwords"],
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
    --input data/processed/der_tag/text/normalize_lowercase_20251110_120000/textblocks.parquet
```

### 2. Keep Preprocessing Steps Modular

Instead of one giant preprocessing run, break into logical stages:

```bash
# Stage 1: Normalization
newspaper-explorer data preprocess --source der_tag --steps normalize,lowercase

# Stage 2: Linguistic processing (can be slow, keep separate)
newspaper-explorer data preprocess --source der_tag \
    --input <stage1_output> \
    --steps lemmatize-spacy

# Stage 3: Analysis-specific preprocessing
newspaper-explorer data preprocess --source der_tag \
    --input <stage2_output> \
    --steps remove-stopwords
```

### 3. Use Metadata for Analysis Configuration

When running analysis, extract preprocessing info to adapt parameters:

```python
input_stats = extract_input_stats(df, input_path=input_path)
if "preprocessing" in input_stats:
    steps = input_stats["preprocessing"]["steps"]
    if "lemmatize-spacy" in steps:
        # Use lemmatized text column
        text_column = "text_lemma"
    else:
        text_column = "text"
```

---

## See Also

- [Preprocessing Pipeline Documentation](PREPROCESSING.md) - Complete guide to preprocessing steps
- [Analysis Metadata System](../../development/ANALYSIS_METADATA.md) - Analysis metadata pattern
- [Configuration Management](../../development/CONFIGURATION.md) - Configuration system
