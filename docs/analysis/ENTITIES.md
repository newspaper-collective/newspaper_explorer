# Entity Extraction Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Entity Types](#entity-types)
7. [Method Comparison](#method-comparison)
8. [Performance & Configuration](#performance--configuration)
9. [Data Schemas](#data-schemas)
10. [Complete Workflows](#complete-workflows)
11. [Analyzing Results](#analyzing-results)
12. [Integration with Other Analyses](#integration-with-other-analyses)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The entity extraction module provides three methods for extracting named entities (persons, organizations, locations, events) from newspaper texts:

**Methods**:
1. **GLiNER** - Fast local model with multi-GPU support (250MB-1.2GB)
2. **GLiNER2** - Optimized CPU inference with label descriptions (205M params) [WARNING] *Not publicly available yet*
3. **LLM** - High-quality extraction with structured validation (requires API)

**Supported Entity Types**:
- **Person** - Names of individuals (Menschen, Personen)
- **Organisation** - Organizations, companies, institutions (Organisationen, Unternehmen)
- **Ort** - Locations, places, cities (Orte, Städte, Länder)
- **Ereignis** - Events, occurrences (Ereignisse, Vorfälle)

**Key Features**:
- **Three extraction methods**: GLiNER (fast GPU), GLiNER2 (optimized CPU), LLM (highest quality)
- **Multi-GPU parallelism**: GLiNER supports parallel processing across GPUs
- **Label descriptions**: GLiNER2 uses descriptions for better accuracy
- **Method comparison**: Compare results from different extraction methods
- **Resume functionality**: Skip already-processed texts
- **Confidence scores**: GLiNER/GLiNER2 provide confidence values, LLM provides structured validation

---

## Architecture

### File Structure

```
src/newspaper_explorer/
├── analyze/
│   └── entities/
│       ├── gliner.py                # GLiNER v1 extractor (multi-GPU)
│       ├── gliner2.py               # GLiNER2 extractor (CPU-optimized)
│       ├── llm.py                   # LLM-based extractor
│       └── utils.py                 # Comparison and utility functions
├── cli/
│   └── analyze/
│       ├── commands.py              # Register entities group
│       └── entities/
│           └── commands.py          # CLI commands (gliner, gliner2, llm, compare)
├── llm/
│   ├── client.py                    # LLM client
│   ├── prompts/
│   │   └── entity_extraction.py    # Entity extraction prompt
│   └── schemas/
│       └── entity_extraction.py    # EntityResponse schema

results/{source}/entities/           # Output directory
└── {method_id}/                     # One directory per extraction run
    ├── entities.parquet             # Extracted entities
    └── metadata.json                # Extraction metadata
```

### Integration with Data Pipeline

```
Download → Parse ALTO/METS → Polars DataFrame → Preprocessing (optional)
                                                        ↓
                                              Entity Extraction
                                      (GLiNER / GLiNER2 / LLM)
                                                        ↓
                          results/{source}/entities/{method_id}/
                                                        ↓
                                        Analysis & Visualization
```

**Integration Points**:
- Uses existing Polars DataFrames from `DataLoader`
- Works with line-level or text-block-level data
- Supports normalized/preprocessed text
- Outputs to standard `results/{source}/entities/` directory
- Queryable via `QueryEngine` with foreign key relationships

---

## Core Components

### 1. GLiNEREntityExtractor

**Purpose**: Fast entity extraction using GLiNER models with multi-GPU support.

**Features**:
- Multiple model sizes (250MB to 1.2GB)
- Multi-GPU parallel processing
- Automatic text chunking
- Confidence scores
- Resume functionality

**Available Models**:
```python
from newspaper_explorer.analyze.entities.gliner import GLINER_MODELS

# Models dictionary:
{
    "multi-v2.1": {
        "model_id": "urchade/gliner_multi-v2.1",
        "size": "~250MB",
        "max_tokens": 384,
        "description": "Multilingual, balanced speed/quality",
        "default": True
    },
    "xxl-v2.5": {
        "model_id": "gliner-community/gliner_xxl-v2.5",
        "size": "~1.2GB",
        "max_tokens": 512,
        "description": "Extra large, highest quality"
    },
    "small-v2.1": {
        "model_id": "urchade/gliner_small-v2.1",
        "size": "~150MB",
        "max_tokens": 384,
        "description": "Smaller, faster inference"
    },
    "large-v2.1": {
        "model_id": "urchade/gliner_large-v2.1",
        "size": "~650MB",
        "max_tokens": 384,
        "description": "Larger, better accuracy"
    }
}
```

**Usage**:
```python
from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor

# Basic usage
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    model_name="multi-v2.1",  # or full ID: "urchade/gliner_multi-v2.1"
    labels=["Person", "Organisation", "Ort", "Ereignis"],
    threshold=0.2,
    batch_size=32,
)

# Extract and save
results = extractor.extract_and_save(limit=100)

# Multi-GPU processing
results = extractor.extract_and_save(num_gpus=4)
```

### 2. GLiNER2EntityExtractor

**Purpose**: Optimized CPU-based entity extraction with label descriptions for improved accuracy.

**Model Availability**: [WARNING] **Note**: GLiNER2 models are currently not publicly available. This implementation is ready but requires access to the model weights.

**Features**:
- Fast CPU inference (no GPU required)
- Label descriptions for better accuracy
- Single unified model (205M parameters)
- Efficient batch processing
- Automatic text chunking

**Usage**:
```python
from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor

# With label descriptions (recommended)
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    model_name="fastino/gliner2-base-0207",
    labels={
        "Person": "Namen von Personen oder historischen Figuren",
        "Organisation": "Organisationen, Unternehmen oder Institutionen",
        "Ort": "Städte, Länder oder geografische Orte",
        "Ereignis": "Historische Ereignisse oder Vorfälle"
    },
    threshold=0.5,
    batch_size=32,
)

# Simple list of labels
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    labels=["Person", "Organisation", "Ort", "Ereignis"],
    threshold=0.5,
)

# Extract and save
results = extractor.extract_and_save(limit=100)
```

### 3. LLMEntityExtractor

**Purpose**: High-quality entity extraction using LLMs with structured validation.

**Features**:
- Context-aware extraction
- Structured response validation (Pydantic)
- Automatic retry on failures
- Metadata support (dates, sources)
- Text chunking for long texts

**Usage**:
```python
from newspaper_explorer.analyze.entities.llm import LLMEntityExtractor

extractor = LLMEntityExtractor(
    source_name="der_tag",
    model_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=2000,
    batch_size=10,
)

# Extract and save
results = extractor.extract_and_save(limit=100)
```

**Configuration** (`.env` file):
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000
```

### 4. Method Comparison

**Purpose**: Compare entity extraction results from different methods.

**Features**:
- List available results
- Compare two methods side-by-side
- Entity overlap analysis
- Coverage statistics
- Type distribution

**Functions**:
```python
from newspaper_explorer.analyze.entities.utils import (
    list_entity_results,
    compare_existing_results,
    print_comparison_report,
)

# List all available results
results = list_entity_results("der_tag")
for r in results:
    print(f"{r['method_id']}: {r['entity_count']} entities")

# Compare two methods
comparison = compare_existing_results(
    source_name="der_tag",
    method_id_1="gliner_multi_v2_1_20241106",
    method_id_2="llm_gpt_4o_mini_20241106",
)

# Print formatted report
print_comparison_report(comparison)
```

---

## CLI Commands

All commands follow: `newspaper-explorer analyze entities <command> --source <name>`

### 1. List Available Models

```bash
# Show GLiNER models with sizes and descriptions
newspaper-explorer analyze entities list-models
```

**Output**:
- Model short names and full Hugging Face IDs
- Model sizes and token limits
- Descriptions and recommendations
- Usage examples

### 2. GLiNER Extraction

**Basic Usage**:
```bash
# Extract with default model (multi-v2.1)
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation,Ort,Ereignis"

# Use larger model for better accuracy
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation,Ort" \
    --model xxl-v2.5 \
    --threshold 0.3
```

**Multi-GPU Processing**:
```bash
# Use all available GPUs
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation,Ort" \
    --num-gpus 4 \
    --batch-size 64
```

**Custom Input**:
```bash
# Use textblocks instead of lines
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation" \
    --input-type textblocks \
    --text-column text

# Use custom parquet file
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation" \
    --input-file data/processed/der_tag/textblocks_normalized.parquet \
    --text-column text_normalized
```

**Performance Tuning**:
```bash
# High-performance mode (large GPU)
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation,Ort" \
    --batch-size 128 \
    --min-length 50 \
    --max-length 2000

# Conservative mode (small GPU/CPU)
newspaper-explorer analyze entities gliner \
    --source der_tag \
    --labels "Person,Organisation" \
    --batch-size 16 \
    --threshold 0.4
```

### 3. GLiNER2 Extraction

**Basic Usage**:
```bash
# Extract with default settings (CPU-optimized)
newspaper-explorer analyze entities gliner2 \
    --source der_tag \
    --labels "Person,Organisation,Ort"

# With label descriptions (recommended)
newspaper-explorer analyze entities gliner2 \
    --source der_tag \
    --labels "Person,Organisation,Ort" \
    --label-descriptions '{
        "Person": "Namen von Personen",
        "Organisation": "Organisationen oder Unternehmen",
        "Ort": "Städte oder Länder"
    }'
```

**Custom Configuration**:
```bash
# Use textblocks with normalized text
newspaper-explorer analyze entities gliner2 \
    --source der_tag \
    --labels "Person,Organisation" \
    --input-type textblocks \
    --text-column text_normalized \
    --threshold 0.6 \
    --batch-size 64
```

### 4. LLM Extraction

**Basic Usage**:
```bash
# Extract with default model (gpt-4o-mini)
newspaper-explorer analyze entities llm --source der_tag

# Use different model
newspaper-explorer analyze entities llm \
    --source der_tag \
    --model gpt-4o \
    --temperature 0.2
```

**Custom Configuration**:
```bash
# Process textblocks with custom settings
newspaper-explorer analyze entities llm \
    --source der_tag \
    --input-type textblocks \
    --text-column text \
    --max-length 10000 \
    --batch-size 20
```

### 5. List Results

```bash
# Show all entity extraction results for a source
newspaper-explorer analyze entities list-results --source der_tag
```

**Output**:
- Method IDs with timestamps
- Method types (gliner, gliner2, llm)
- Models used
- Entity counts
- Processing durations

### 6. Compare Methods

```bash
# Compare two extraction results
newspaper-explorer analyze entities compare \
    --source der_tag \
    --method-1 gliner_multi_v2_1_20241106_120000 \
    --method-2 llm_gpt_4o_mini_20241106_140000

# Detailed comparison (shows entity-level differences)
newspaper-explorer analyze entities compare \
    --source der_tag \
    --method-1 gliner_multi_v2_1_20241106_120000 \
    --method-2 gliner2_base_20241106_150000 \
    --detailed
```

**Output**:
- Execution metadata (models, durations, parameters)
- Overall statistics (total/unique entities)
- Entity type distribution
- Top entities found by both methods
- Method-specific entities (unique to each)
- Coverage comparison

---

## Python API

### Quick Start Examples

**GLiNER (Fast, Multi-GPU)**:
```python
from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor

# Basic extraction
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    labels=["Person", "Organisation", "Ort"],
    threshold=0.2,
)
results = extractor.extract_and_save(limit=100)

# Multi-GPU extraction
extractor = GLiNEREntityExtractor(source_name="der_tag")
results = extractor.extract_and_save(num_gpus=4)
```

**GLiNER2 (CPU-Optimized, Label Descriptions)**:
```python
from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor

# With descriptions for better accuracy
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    labels={
        "Person": "Namen von Personen",
        "Organisation": "Organisationen oder Unternehmen",
        "Ort": "Städte, Länder oder Orte"
    },
    threshold=0.5,
)
results = extractor.extract_and_save(limit=100)
```

**LLM (Highest Quality)**:
```python
from newspaper_explorer.analyze.entities.llm import LLMEntityExtractor

extractor = LLMEntityExtractor(
    source_name="der_tag",
    model_name="gpt-4o-mini",
    temperature=0.3,
)
results = extractor.extract_and_save(limit=100)
```

### Working with DataFrames

All extractors support custom DataFrames:

```python
from newspaper_explorer.data.loading.loader import DataLoader
import polars as pl

# Load data
loader = DataLoader(source_name="der_tag")
df = loader.load_source()

# Filter to specific year
df_1902 = df.filter(pl.col("date").dt.year() == 1902)

# Extract from filtered data
extractor = GLiNEREntityExtractor(source_name="der_tag")
entities = extractor.extract_from_dataframe(
    df_1902,
    text_column="text",
    id_column="line_id",
)

print(f"Extracted {len(entities)} entities from 1902")
```

### Custom Input Files

Use preprocessed or normalized text:

```python
from pathlib import Path

# Use normalized textblocks
extractor = GLiNER2EntityExtractor(source_name="der_tag")
results = extractor.extract_and_save(
    source_parquet=Path("data/processed/der_tag/textblocks_normalized.parquet"),
    text_column="text_normalized",
    id_column="text_block_id",
)
```

### Batch Processing and Performance

**GLiNER Multi-GPU**:
```python
# High-performance configuration
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    model_name="xxl-v2.5",  # Larger model
    batch_size=128,         # Large batches
    min_text_length=50,     # Skip very short texts
    max_text_length=2000,   # Chunk longer texts
)

# Process with all available GPUs
results = extractor.extract_and_save(num_gpus=4)
print(f"Duration: {results['duration_seconds']:.1f}s")
print(f"Entities: {len(results['results_df'])}")
```

**GLiNER2 CPU Optimization**:
```python
# Optimized for CPU
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    batch_size=64,          # Larger batches for CPU
    min_text_length=100,    # Skip short texts
    max_text_length=2500,   # Optimal chunk size
)

results = extractor.extract_and_save()
```

---

## Entity Types

### Standard Types

The extractors recognize four main entity types:

1. **Person** (Menschen, Personen)
   - Individual names
   - Historical figures
   - Authors, politicians, public figures
   - Example: "Kaiser Wilhelm II", "Otto von Bismarck"

2. **Organisation** (Organisationen)
   - Companies and businesses
   - Government institutions
   - Political parties
   - Newspapers and publishers
   - Example: "Reichstag", "Berliner Tageblatt"

3. **Ort** (Orte, Locations)
   - Cities and towns
   - Countries and regions
   - Geographic features
   - Buildings and landmarks
   - Example: "Berlin", "Deutschland", "Reichstagsgebäude"

4. **Ereignis** (Ereignisse, Events)
   - Historical events
   - Wars and battles
   - Conferences and treaties
   - Disasters and incidents
   - Example: "Erster Weltkrieg", "Versailler Vertrag"

### Custom Labels

You can define custom labels for specific use cases:

```python
# Domain-specific extraction
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    labels={
        "Monarch": "Namen von Königen, Kaisern oder Fürsten",
        "Militäreinheit": "Regimenter, Bataillone oder Armeen",
        "Zeitung": "Namen von Zeitungen oder Zeitschriften",
    },
)
```

---

## Method Comparison

### Comparison Workflow

**1. List Available Results**:
```python
from newspaper_explorer.analyze.entities.utils import list_entity_results

results = list_entity_results("der_tag")
for r in results:
    print(f"{r['method_id']}: {r['method_type']}, {r['entity_count']} entities")
```

**2. Compare Two Methods**:
```python
from newspaper_explorer.analyze.entities.utils import (
    compare_existing_results,
    print_comparison_report,
)

comparison = compare_existing_results(
    source_name="der_tag",
    method_id_1="gliner_multi_v2_1_20241106_120000",
    method_id_2="llm_gpt_4o_mini_20241106_140000",
)

print_comparison_report(comparison)
```

### Comparison Metrics

The comparison provides:

**Overall Statistics**:
- Total entities per method
- Unique entities per method
- Entity overlap (found by both)
- Method-specific entities (unique to each)

**Type Distribution**:
- Entity counts by type
- Percentage distribution
- Type-specific comparisons

**Entity-Level Analysis**:
- Top entities found by both
- Entities unique to method 1
- Entities unique to method 2
- Frequency distributions

**Coverage Statistics**:
- Lines with entities (per method)
- Coverage overlap
- Method-specific coverage

### Interpreting Results

**High Agreement** (>70% overlap):
- Methods produce consistent results
- Either method likely sufficient
- Consider using faster method (GLiNER/GLiNER2)

**Medium Agreement** (30-70% overlap):
- Methods complement each other
- Consider combining results
- Use union for maximum coverage
- Use intersection for high precision

**Low Agreement** (<30% overlap):
- Methods have different strengths
- Investigate discrepancies
- May indicate data quality issues
- Consider parameter tuning

---

## Performance & Configuration

### Performance Comparison Table

| Method   | Speed (ent/sec) | Memory    | Cost      | Quality | Setup    |
|----------|-----------------|-----------|-----------|---------|----------|
| GLiNER   | 500-2000       | 2-20 GB   | Free      | Good    | GPU req  |
| GLiNER2  | 300-500        | 4-8 GB    | Free      | Good    | CPU only |
| LLM      | 10-50          | <1 GB     | $0.0001/t | Best    | API req  |

### GLiNER Performance

**Speed** (entities per second):
| Model        | GPU (A100) | GPU (RTX 3090) | CPU      |
|--------------|------------|----------------|----------|
| small-v2.1   | ~2000      | ~1200          | ~100     |
| multi-v2.1   | ~1500      | ~900           | ~80      |
| large-v2.1   | ~1000      | ~600           | ~50      |
| xxl-v2.5     | ~800       | ~400           | ~30      |

**Memory Requirements**:
| Model        | Model Size | GPU Memory | CPU Memory |
|--------------|------------|------------|------------|
| small-v2.1   | ~150MB     | 2-4 GB     | 4-6 GB     |
| multi-v2.1   | ~250MB     | 3-5 GB     | 5-8 GB     |
| large-v2.1   | ~650MB     | 6-10 GB    | 10-16 GB   |
| xxl-v2.5     | ~1.2GB     | 12-20 GB   | 20-32 GB   |

**Multi-GPU Scaling**:
- 1 GPU: baseline
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.2x speedup
- 8 GPUs: ~5.5x speedup

**Recommended Configurations**:

```python
# Small GPU (4-8GB) - Conservative
extractor = GLiNEREntityExtractor(
    model_name="small-v2.1",
    batch_size=16,
    threshold=0.3,
)

# Medium GPU (8-16GB) - Balanced
extractor = GLiNEREntityExtractor(
    model_name="multi-v2.1",
    batch_size=32,
    threshold=0.2,
)

# Large GPU (16GB+) - High Quality
extractor = GLiNEREntityExtractor(
    model_name="xxl-v2.5",
    batch_size=64,
    threshold=0.15,
)

# Multi-GPU Server (4x A100)
extractor = GLiNEREntityExtractor(
    model_name="xxl-v2.5",
    batch_size=128,
    num_gpus=4,
)
```

### GLiNER2 Performance

**Speed** (entities per second):
| Batch Size | CPU (32 cores) | CPU (16 cores) | CPU (8 cores) |
|------------|----------------|----------------|---------------|
| 16         | ~300           | ~180           | ~100          |
| 32         | ~400           | ~240           | ~130          |
| 64         | ~500           | ~300           | ~150          |

**Memory Requirements**:
- Model: ~800MB
- Runtime: 4-8 GB (scales with batch size)

**Recommended Configuration**:
```python
# High-core CPU server
extractor = GLiNER2EntityExtractor(
    batch_size=64,
    threshold=0.5,
)

# Standard workstation
extractor = GLiNER2EntityExtractor(
    batch_size=32,
    threshold=0.5,
)

# Limited resources
extractor = GLiNER2EntityExtractor(
    batch_size=16,
    threshold=0.6,  # Higher threshold reduces processing
)
```

### LLM Performance

**Speed** (requests per minute):
- GPT-4o-mini: ~60-100 RPM (API tier dependent)
- GPT-4o: ~30-50 RPM
- Claude-3.5-Sonnet: ~40-80 RPM

**Cost** (per 1000 texts, ~200 tokens each):
- GPT-4o-mini: ~$0.03
- GPT-4o: ~$0.30
- Claude-3.5-Sonnet: ~$0.15

**Recommended Configuration**:
```python
# Cost-effective
extractor = LLMEntityExtractor(
    model_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1500,
    batch_size=20,
)

# High quality
extractor = LLMEntityExtractor(
    model_name="gpt-4o",
    temperature=0.2,
    max_tokens=2000,
    batch_size=10,
)
```

### Performance Tips

**1. Text Length Filtering**:
```python
# Skip very short texts (often headers/footers)
extractor = GLiNEREntityExtractor(
    min_text_length=100,  # Skip < 100 chars
)

# Avoid processing very long texts
extractor = GLiNEREntityExtractor(
    max_text_length=2000,  # Chunk > 2000 chars
)
```

**2. Preprocessing**:
```python
# Use normalized text for better accuracy
from newspaper_explorer.data.preprocessing.pipeline import preprocess_text_data

df_normalized = preprocess_text_data(
    source_name="der_tag",
    normalize=True,
)

extractor = GLiNER2EntityExtractor(source_name="der_tag")
results = extractor.extract_and_save(
    source_parquet=df_normalized,
    text_column="text_normalized",
)
```

**3. Threshold Tuning**:
```python
# Lower threshold = more entities (higher recall, lower precision)
extractor = GLiNEREntityExtractor(threshold=0.1)

# Higher threshold = fewer entities (lower recall, higher precision)
extractor = GLiNEREntityExtractor(threshold=0.5)

# Recommended starting points:
# - GLiNER: 0.2 (balanced)
# - GLiNER2: 0.5 (conservative)
```

---

## Data Schemas

### Entity Output Schema

**Parquet File** (`entities.parquet`):

```python
{
    "source_id": str,        # Foreign key (line_id or text_block_id)
    "entity_text": str,      # The extracted entity text
    "entity_type": str,      # Entity category (person, organisation, ort, ereignis)
    "confidence": float,     # Confidence score (GLiNER/GLiNER2 only, 0.0-1.0)
    "detection_count": int,  # How many times detected (GLiNER2 only, for chunked texts)
}
```

**Notes**:
- `source_id` column name matches input `id_column` parameter
- LLM method doesn't provide confidence scores (always produces validated output)
- GLiNER2 includes `detection_count` for entities found multiple times in chunked text

### Metadata Schema

**JSON File** (`metadata.json`):

```json
{
    "analysis_id": "gliner_multi_v2_1_20241106_120000",
    "method_type": "gliner",
    "source_name": "der_tag",
    "timestamp": "2024-11-06T12:00:00",
    "parameters": {
        "model": "urchade/gliner_multi-v2.1",
        "labels": ["Person", "Organisation", "Ort", "Ereignis"],
        "threshold": 0.2,
        "batch_size": 32,
        "min_text_length": 10,
        "max_text_length": 1000
    },
    "input_file": "data/processed/der_tag/der_tag_lines.parquet",
    "text_column": "text",
    "id_column": "line_id",
    "line_count": 150000,
    "entity_count": 45000,
    "unique_entities": 8500,
    "duration_seconds": 3600.5,
    "entities_per_second": 41.7
}
```

### Query Integration

Entities link to source data via foreign keys:

```python
from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Join entities with source text
    results = qe.query("""
        SELECT 
            e.entity_text,
            e.entity_type,
            e.confidence,
            l.text,
            l.date,
            l.newspaper_title
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN lines l ON e.source_id = l.line_id
        WHERE e.entity_type = 'person'
        ORDER BY e.confidence DESC
        LIMIT 100
    """)
```

---

## Complete Workflows

### Workflow 1: Quick Exploration

```python
"""Extract entities from a small sample to evaluate methods."""

from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor
from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor
from newspaper_explorer.analyze.entities.llm import LLMEntityExtractor
from newspaper_explorer.analyze.entities.utils import (
    list_entity_results,
    compare_existing_results,
    print_comparison_report,
)

# Test each method on 100 texts
limit = 100

# GLiNER (fast)
gliner = GLiNEREntityExtractor(source_name="der_tag", threshold=0.2)
gliner_results = gliner.extract_and_save(limit=limit)

# GLiNER2 (CPU, with descriptions)
gliner2 = GLiNER2EntityExtractor(
    source_name="der_tag",
    labels={
        "Person": "Namen von Personen",
        "Organisation": "Organisationen",
        "Ort": "Städte oder Länder"
    },
    threshold=0.5,
)
gliner2_results = gliner2.extract_and_save(limit=limit)

# LLM (high quality)
llm = LLMEntityExtractor(source_name="der_tag", model_name="gpt-4o-mini")
llm_results = llm.extract_and_save(limit=limit)

# Compare results
method_ids = [
    gliner_results["metadata"]["analysis_id"],
    gliner2_results["metadata"]["analysis_id"],
    llm_results["metadata"]["analysis_id"],
]

for i in range(len(method_ids)):
    for j in range(i + 1, len(method_ids)):
        comparison = compare_existing_results(
            source_name="der_tag",
            method_id_1=method_ids[i],
            method_id_2=method_ids[j],
        )
        print_comparison_report(comparison)
```

### Workflow 2: Production Pipeline

```python
"""Full-scale entity extraction with optimized settings."""

from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor
from newspaper_explorer.data.preprocessing.pipeline import preprocess_text_data
from pathlib import Path

# 1. Preprocess text data (normalize historical German)
print("Preprocessing text...")
preprocessed_path = preprocess_text_data(
    source_name="der_tag",
    normalize=True,
    output_name="textblocks_normalized",
)

# 2. Extract entities with optimized settings
print("Extracting entities...")
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    model_name="xxl-v2.5",  # Best quality
    labels=["Person", "Organisation", "Ort", "Ereignis"],
    threshold=0.15,  # Lower threshold for recall
    batch_size=128,
    min_text_length=100,  # Skip short texts
)

# 3. Run extraction with multi-GPU
results = extractor.extract_and_save(
    source_parquet=preprocessed_path,
    text_column="text_normalized",
    id_column="text_block_id",
    num_gpus=4,
)

# 4. Print results
print(f"\nExtraction Complete!")
print(f"Method ID: {results['metadata']['analysis_id']}")
print(f"Duration: {results['duration_seconds'] / 60:.1f} minutes")
print(f"Total entities: {results['entity_count']:,}")
print(f"Unique entities: {results['unique_entities']:,}")
print(f"Speed: {results['entities_per_second']:.1f} entities/second")
```

### Workflow 3: Year-by-Year Analysis

```python
"""Extract entities year by year for temporal analysis."""

from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor
from newspaper_explorer.data.loading.loader import DataLoader
import polars as pl

# Load data
loader = DataLoader(source_name="der_tag")
df = loader.load_source()

# Get available years
years = sorted(df.select(pl.col("date").dt.year()).unique().to_series().to_list())

# Extract per year
extractor = GLiNER2EntityExtractor(
    source_name="der_tag",
    labels=["Person", "Organisation", "Ort"],
    threshold=0.5,
)

for year in years:
    print(f"\nProcessing year {year}...")
    
    # Filter to year
    df_year = df.filter(pl.col("date").dt.year() == year)
    
    # Extract entities
    entities = extractor.extract_from_dataframe(
        df_year,
        text_column="text",
        id_column="line_id",
    )
    
    # Save with year in filename
    output_path = Path(f"results/der_tag/entities/per_year/entities_{year}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entities.write_parquet(output_path)
    
    print(f"  Extracted {len(entities)} entities from {len(df_year)} lines")
```

### Workflow 4: Custom Analysis with Filtering

```python
"""Extract entities with post-processing and filtering."""

from newspaper_explorer.analyze.entities.gliner import GLiNEREntityExtractor
from newspaper_explorer.data.loading.loader import DataLoader
import polars as pl

# Load data
loader = DataLoader(source_name="der_tag")
df = loader.load_source()

# Filter to specific newspaper and date range
df_filtered = df.filter(
    (pl.col("newspaper_title") == "Der Tag") &
    (pl.col("date") >= pl.datetime(1902, 1, 1)) &
    (pl.col("date") <= pl.datetime(1905, 12, 31))
)

# Extract entities
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    labels=["Person", "Organisation", "Ort"],
    threshold=0.3,
)

entities = extractor.extract_from_dataframe(
    df_filtered,
    text_column="text",
    id_column="line_id",
)

# Post-processing filters
entities_filtered = entities.filter(
    # Remove single-character entities
    (pl.col("entity_text").str.len_chars() > 1) &
    # Keep only high-confidence entities
    (pl.col("confidence") > 0.4) &
    # Remove entities that are all uppercase (often OCR errors)
    (~pl.col("entity_text").str.to_uppercase().eq(pl.col("entity_text")))
)

# Get top entities
top_persons = (
    entities_filtered
    .filter(pl.col("entity_type") == "person")
    .group_by("entity_text")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(20)
)

print("Top 20 persons mentioned (1902-1905):")
print(top_persons)
```

---

## Analyzing Results

### Basic Queries

**Load entities**:
```python
import polars as pl

# Load specific method's results
entities = pl.read_parquet(
    "results/der_tag/entities/gliner_multi_v2_1_20241106/entities.parquet"
)

# Show entity type distribution
type_dist = (
    entities
    .group_by("entity_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print(type_dist)

# Top entities by type
top_persons = (
    entities
    .filter(pl.col("entity_type") == "person")
    .group_by("entity_text")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(20)
)
print(top_persons)
```

### QueryEngine Integration

**Entity frequency over time**:
```python
from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Entities per year
    yearly = qe.query("""
        SELECT 
            YEAR(l.date) as year,
            e.entity_type,
            COUNT(DISTINCT e.entity_text) as unique_entities,
            COUNT(*) as total_mentions
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN lines l ON e.source_id = l.line_id
        GROUP BY year, e.entity_type
        ORDER BY year, e.entity_type
    """)
    print(yearly)
```

**Find entity co-occurrences**:
```python
with QueryEngine(source="der_tag") as qe:
    # Persons mentioned together (same line)
    cooccur = qe.query("""
        SELECT 
            e1.entity_text as person1,
            e2.entity_text as person2,
            COUNT(*) as cooccurrence_count
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e1
        JOIN 'results/der_tag/entities/{method_id}/entities.parquet' e2
            ON e1.source_id = e2.source_id
            AND e1.entity_text < e2.entity_text
        WHERE 
            e1.entity_type = 'person' 
            AND e2.entity_type = 'person'
        GROUP BY person1, person2
        HAVING cooccurrence_count > 5
        ORDER BY cooccurrence_count DESC
        LIMIT 20
    """)
    print(cooccur)
```

**Entity context retrieval**:
```python
with QueryEngine(source="der_tag") as qe:
    # Get context for specific entity mentions
    context = qe.query("""
        SELECT 
            e.entity_text,
            e.entity_type,
            l.text,
            l.date,
            l.newspaper_title
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN lines l ON e.source_id = l.line_id
        WHERE e.entity_text = 'Kaiser Wilhelm II'
        ORDER BY l.date
        LIMIT 10
    """)
    print(context)
```

### Visualization Examples

**Entity frequency over time**:
```python
import polars as pl
import matplotlib.pyplot as plt
from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Get entity mentions over time
    df = qe.query("""
        SELECT 
            YEAR(l.date) as year,
            COUNT(*) as mentions
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN lines l ON e.source_id = l.line_id
        WHERE e.entity_text = 'Berlin'
        GROUP BY year
        ORDER BY year
    """)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["year"], df["mentions"], marker='o')
plt.title("Mentions of 'Berlin' over time")
plt.xlabel("Year")
plt.ylabel("Number of mentions")
plt.grid(True, alpha=0.3)
plt.show()
```

**Entity network graph**:
```python
import networkx as nx
import matplotlib.pyplot as plt
from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Get co-occurrences
    cooccur = qe.query("""
        SELECT 
            e1.entity_text as entity1,
            e2.entity_text as entity2,
            COUNT(*) as weight
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e1
        JOIN 'results/der_tag/entities/{method_id}/entities.parquet' e2
            ON e1.source_id = e2.source_id
            AND e1.entity_text < e2.entity_text
        WHERE 
            e1.entity_type = 'person' 
            AND e2.entity_type = 'person'
        GROUP BY entity1, entity2
        HAVING weight > 10
        ORDER BY weight DESC
        LIMIT 50
    """)

# Create network
G = nx.Graph()
for row in cooccur.iter_rows(named=True):
    G.add_edge(row["entity1"], row["entity2"], weight=row["weight"])

# Plot
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', 
                 node_size=1000, font_size=8, font_weight='bold')
plt.title("Person Co-occurrence Network")
plt.axis('off')
plt.tight_layout()
plt.show()
```

---

## Integration with Other Analyses

### Combine with Emotion Analysis

```python
"""Find emotional contexts for specific entities."""

from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Entities with emotional context
    emotional_entities = qe.query("""
        SELECT 
            e.entity_text,
            e.entity_type,
            em.emotion,
            em.probability,
            l.text,
            l.date
        FROM 'results/der_tag/entities/{entity_method_id}/entities.parquet' e
        JOIN 'results/der_tag/emotions/emotion_predictions.parquet' em
            ON e.source_id = em.source_id
        JOIN lines l ON e.source_id = l.line_id
        WHERE 
            e.entity_type = 'person'
            AND em.prediction = 1
            AND em.emotion IN ('anger', 'fear')
        ORDER BY em.probability DESC
        LIMIT 50
    """)
    
    print("Entities mentioned in angry/fearful contexts:")
    print(emotional_entities)
```

### Combine with Layout Analysis

```python
"""Extract entities from specific layout regions (e.g., headlines)."""

from newspaper_explorer.analyze.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Entities from headlines only
    headline_entities = qe.query("""
        SELECT 
            e.entity_text,
            e.entity_type,
            COUNT(*) as headline_mentions
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN 'results/der_tag/layout/headlines/headlines.parquet' h
            ON e.source_id = h.source_id
        GROUP BY e.entity_text, e.entity_type
        ORDER BY headline_mentions DESC
        LIMIT 50
    """)
    
    print("Top entities in headlines:")
    print(headline_entities)
```

### Combined Analysis Pipeline

```python
"""Complete analysis combining entities, emotions, and layout."""

from newspaper_explorer.analyze.query.engine import QueryEngine
import polars as pl

with QueryEngine(source="der_tag") as qe:
    # Complex multi-analysis query
    combined = qe.query("""
        SELECT 
            e.entity_text,
            e.entity_type,
            em.emotion,
            h.detection_type as layout_element,
            l.text,
            l.date,
            l.newspaper_title
        FROM 'results/der_tag/entities/{entity_method_id}/entities.parquet' e
        LEFT JOIN 'results/der_tag/emotions/emotion_predictions.parquet' em
            ON e.source_id = em.source_id AND em.prediction = 1
        LEFT JOIN 'results/der_tag/layout/detections/*.parquet' h
            ON e.source_id = h.source_id
        JOIN lines l ON e.source_id = l.line_id
        WHERE e.entity_type = 'person'
        ORDER BY l.date
        LIMIT 1000
    """)
    
    # Analyze emotional profiles of entities
    entity_emotions = (
        combined
        .group_by(["entity_text", "emotion"])
        .agg(pl.len().alias("count"))
        .pivot(values="count", index="entity_text", columns="emotion")
        .fill_null(0)
    )
    
    print("Emotional profiles of top entities:")
    print(entity_emotions.head(20))
```

---

## Troubleshooting

### GLiNER Issues

**Problem**: Out of memory errors
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# Reduce batch size
extractor = GLiNEREntityExtractor(batch_size=16)

# Use smaller model
extractor = GLiNEREntityExtractor(model_name="small-v2.1")

# Fall back to CPU
extractor = GLiNEREntityExtractor(device="cpu")
```

**Problem**: Slow processing on CPU
```
Processing is taking too long on CPU
```

**Solutions**:
```python
# Use smaller model
extractor = GLiNEREntityExtractor(model_name="small-v2.1")

# Increase text length threshold
extractor = GLiNEREntityExtractor(min_text_length=200)

# Consider using GLiNER2 instead (optimized for CPU)
from newspaper_explorer.analyze.entities.gliner2 import GLiNER2EntityExtractor
extractor = GLiNER2EntityExtractor(source_name="der_tag")
```

**Problem**: Too many false positives
```
Extracting irrelevant entities or OCR artifacts
```

**Solutions**:
```python
# Increase threshold
extractor = GLiNEREntityExtractor(threshold=0.4)

# Filter by text length
extractor = GLiNEREntityExtractor(min_text_length=100)

# Post-process results to remove artifacts
entities = entities.filter(
    (pl.col("entity_text").str.len_chars() > 2) &
    (pl.col("confidence") > 0.3)
)
```

### GLiNER2 Issues

**Problem**: Label descriptions not improving accuracy
```
Using descriptions but results similar to plain labels
```

**Solutions**:
```python
# Use more detailed, distinctive descriptions
extractor = GLiNER2EntityExtractor(
    labels={
        "Person": "Vollständige Namen von Personen, einschließlich Titel und Familiennamen. "
                  "Beispiele: 'Kaiser Wilhelm II', 'Dr. Otto von Bismarck'",
        "Organisation": "Namen von Organisationen, Institutionen, Unternehmen oder Regierungsbehörden. "
                       "Beispiele: 'Reichstag', 'Berliner Tageblatt', 'Auswärtiges Amt'"
    }
)

# Increase threshold to be more selective
extractor = GLiNER2EntityExtractor(threshold=0.6)
```

**Problem**: Slow batch processing
```
CPU processing slower than expected
```

**Solutions**:
```python
# Increase batch size (CPU works well with larger batches)
extractor = GLiNER2EntityExtractor(batch_size=128)

# Skip short texts
extractor = GLiNER2EntityExtractor(min_text_length=200)

# Reduce max text length (less chunking overhead)
extractor = GLiNER2EntityExtractor(max_text_length=2000)
```

### LLM Issues

**Problem**: API rate limit errors
```
openai.RateLimitError: Rate limit reached
```

**Solutions**:
```python
# Reduce batch size to slow down requests
extractor = LLMEntityExtractor(batch_size=5)

# Add delay between batches (modify in llm/client.py)
# Or upgrade API tier for higher rate limits
```

**Problem**: High API costs
```
Processing large datasets is expensive
```

**Solutions**:
```python
# Use cheaper model
extractor = LLMEntityExtractor(model_name="gpt-4o-mini")

# Filter input to high-value texts only
df_filtered = df.filter(pl.col("text").str.len_chars() > 200)

# Consider GLiNER/GLiNER2 for bulk processing
# Use LLM only for validation/quality checks
```

**Problem**: Inconsistent entity extraction
```
Same entity extracted with different spellings
```

**Solutions**:
```python
# Lower temperature for more consistent output
extractor = LLMEntityExtractor(temperature=0.1)

# Post-process for entity normalization
entities = normalize_entity_names(entities)

# Provide context in metadata for better consistency
metadata = {
    "source": "der_tag",
    "date": "1902-01-15",
    "newspaper_title": "Der Tag"
}
```

### General Issues

**Problem**: Method comparison shows low agreement
```
Different methods extracting very different entities
```

**Investigation**:
```python
from newspaper_explorer.analyze.entities.utils import (
    compare_existing_results,
    print_comparison_report,
)

# Detailed comparison
comparison = compare_existing_results(
    source_name="der_tag",
    method_id_1="method1_id",
    method_id_2="method2_id",
)

# Check unique entities from each method
print(comparison["comparisons"]["method1_only"])
print(comparison["comparisons"]["method2_only"])

# Analyze confidence distributions (for GLiNER/GLiNER2)
entities1 = pl.read_parquet(f"results/der_tag/entities/method1_id/entities.parquet")
print(entities1["confidence"].describe())
```

**Problem**: Cannot find extraction results
```
FileNotFoundError: Results directory doesn't exist
```

**Solutions**:
```python
from newspaper_explorer.analyze.entities.utils import list_entity_results

# List all available results
results = list_entity_results("der_tag")
for r in results:
    print(f"Method ID: {r['method_id']}")
    print(f"  Type: {r['method_type']}")
    print(f"  Model: {r['model']}")
    print(f"  Entities: {r['entity_count']}")
    print()

# Check correct paths
from newspaper_explorer.config.base import get_config
config = get_config()
entities_dir = config.results_dir / "der_tag" / "entities"
print(f"Entities directory: {entities_dir}")
print(f"Exists: {entities_dir.exists()}")
```

---

## Related Documentation

- [LLM Utilities](LLM.md) - LLM client, prompts, and schemas
- [Query Architecture](QUERY_ARCHITECTURE.md) - QueryEngine and DuckDB integration
- [Data Loading](DATA_LOADER.md) - Loading and preprocessing newspaper data
- [Emotions](EMOTIONS.md) - Emotion analysis documentation
- [Layout](LAYOUT.md) - Layout analysis documentation
- [CLI Reference](../cli/CLI.md) - Complete command reference
