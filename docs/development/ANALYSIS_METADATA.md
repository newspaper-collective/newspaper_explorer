# Analysis Metadata System

> **Status**: Production-ready (November 2025)  
> **Implementation**: `src/newspaper_explorer/data/utils/metadata.py`  
> **ID Utilities**: `src/newspaper_explorer/data/utils/ids.py`

This document describes the unified metadata system for analysis results in the newspaper explorer codebase. All analysis outputs (entities, topics, emotions, keywords, layout, etc.) follow a consistent pattern for metadata storage and ID tracking.

---

## Quick Start - Unified Save Function

**Use `save_analysis_results()` for all analysis types** - it handles subdirectory creation, parquet saving, and metadata generation:

```python
from newspaper_explorer.data.utils.metadata import (
    AnalysisMetadata,
    save_analysis_results,
    extract_input_stats,
    extract_output_stats,
)
from newspaper_explorer.config.base import get_config

# Create metadata
metadata = AnalysisMetadata(
    analysis_type="keywords",  # or "entities", "topics", "emotions", "layout"
    method_type="keybert",
    model_name="multi-v1",
    source="der_tag",
    parameters={"top_k": 10, "diversity": 0.5},
    input_data=extract_input_stats(input_df),
    output_data=extract_output_stats(results_df),
    duration_seconds=42.5,
    status="completed",
)

# Save everything in one call
config = get_config()
results_base = config.results_dir / "der_tag"
paths = save_analysis_results(
    results_df=results_df,
    metadata=metadata,
    results_base_dir=results_base,
)

# Returns: {"output_dir": Path, "results_path": Path, "metadata_path": Path}
```

This automatically creates the structure:
```
results/der_tag/keywords/keybert_multi_v1_20251109_120000/
├── keywords.parquet
└── metadata.json
```

---

## Design Principles

1. **Reproducibility** - Every analysis result includes complete metadata to reproduce the analysis
2. **Traceability** - Foreign keys link analysis results back to source data
3. **Uniformity** - All analysis types use the same metadata schema
4. **Co-location** - Metadata JSON stored alongside parquet results
5. **Machine & Human Readable** - JSON format with clear structure
6. **Subdirectory Pattern** - Each analysis run gets its own timestamped directory

---

## File Organization Pattern

Each analysis result consists of a **subdirectory** with results and metadata:

```
results/
  {source}/
    {analysis_type}/
      {analysis_id}/
        {type}.parquet     ← Analysis results (entities.parquet, keywords.parquet, etc.)
        metadata.json      ← Analysis metadata
```

Example:
```
results/
  der_tag/
    entities/
      gliner_small_v2_1_20251109_190002/
        ├── entities.parquet
        └── metadata.json
    keywords/
      keybert_multi_v1_20251109_120000/
        ├── keywords.parquet
        └── metadata.json
```
        {analysis_id}.json     ← Metadata
```

Or alternatively (flat structure):

```
results/
  {source}/
    {analysis_type}/
      analysis_name.parquet  ← Analysis results
      analysis_name.json     ← Metadata
```

**Examples**:

```
results/der_tag/entities/gliner_multi_v2_1_20251109_004150/
  ├── entities.parquet          # Entity extraction results
  └── metadata.json             # Analysis metadata

results/der_tag/keywords/
  ├── keybert_optimized.parquet # Keyword extraction results
  └── keybert_optimized.json    # Analysis metadata

results/der_tag/emotions/
  ├── emotion_predictions.parquet
  └── emotion_predictions.json
```

---

## Metadata Schema

### Complete Metadata Structure

```json
{
  "analysis_id": "gliner_multi_v2_1_20251109_004150",
  "analysis_type": "entities",
  "method_type": "gliner",
  "model_name": "multi-v2.1",
  "model_version": "2.1.0",
  "source": "der_tag",
  "parameters": {
    "threshold": 0.2,
    "batch_size": 16,
    "labels": ["Person", "Organisation", "Ort", "Ereignis"],
    "min_text_length": 10,
    "max_text_length": 1000
  },
  "input_data": {
    "parquet_path": "data/raw/der_tag/text/lines.parquet",
    "row_count": 100000,
    "date_range": ["1901-01-08", "1920-12-31"],
    "columns": ["line_id", "text", "date", "..."],
    "id_column": "line_id",
    "id_type": "line_id"
  },
  "output_data": {
    "row_count": 5420,
    "columns": ["entity_id", "entity_text", "entity_type", "..."],
    "schema": {
      "entity_id": "String",
      "entity_text": "String",
      "confidence": "Float64"
    }
  },
  "created_at": "2025-11-09T00:41:50.008260",
  "completed_at": "2025-11-09T00:42:12.335678",
  "duration_seconds": 22.33,
  "status": "completed",
  "error_message": null,
  "software_version": "0.3.0",
  "environment": {
    "python_version": "3.11.5",
    "gpu": "NVIDIA RTX 4090",
    "cuda_version": "12.1"
  },
  "notes": "Initial test run on subset of data"
}
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `analysis_id` | string | Unique identifier (auto-generated) | `gliner_multi_v2_1_20251109_004150` |
| `analysis_type` | string | Type of analysis | `entities`, `topics`, `emotions`, `keywords`, `layout` |
| `method_type` | string | Method category | `gliner`, `llm`, `transformer`, `yolo`, `keybert` |
| `model_name` | string | Specific model identifier | `multi-v2.1`, `gpt-4o-mini`, `yolo11n` |
| `source` | string | Source dataset name | `der_tag` |
| `parameters` | object | All analysis parameters | `{"threshold": 0.2, "batch_size": 16}` |

### Optional/Auto-Generated Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_version` | string | Model version (if applicable) |
| `created_at` | string | ISO timestamp of creation (auto) |
| `completed_at` | string | ISO timestamp of completion |
| `duration_seconds` | float | Processing time in seconds |
| `input_data` | object | Input dataset information |
| `output_data` | object | Output statistics |
| `status` | string | Status: `completed`, `failed`, `in_progress` |
| `error_message` | string | Error details if failed |
| `software_version` | string | Version of newspaper_explorer |
| `environment` | object | Environment details (GPU, Python, etc.) |
| `notes` | string | Optional human-readable notes |

---

## Usage Examples

### Creating and Saving Metadata

```python
from pathlib import Path
from newspaper_explorer.data.utils.metadata import (
    AnalysisMetadata,
    save_metadata,
    extract_input_stats,
    extract_output_stats,
)
import polars as pl

# Load input data
input_df = pl.read_parquet("data/raw/der_tag/text/lines.parquet")

# Create metadata
metadata = AnalysisMetadata(
    analysis_type="entities",
    method_type="gliner",
    model_name="multi-v2.1",
    model_version="2.1.0",
    source="der_tag",
    parameters={
        "threshold": 0.2,
        "batch_size": 16,
        "labels": ["Person", "Organisation", "Ort", "Ereignis"],
    },
    input_data=extract_input_stats(input_df, id_column="line_id"),
    notes="Initial test run on subset",
)

# Run analysis
start_time = time.time()
results_df = extractor.extract(input_df)
duration = time.time() - start_time

# Update metadata with results
metadata.completed_at = datetime.now().isoformat()
metadata.duration_seconds = duration
metadata.output_data = extract_output_stats(results_df)

# Save both parquet and metadata
output_dir = Path(f"results/der_tag/entities/{metadata.analysis_id}")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.write_parquet(output_dir / "entities.parquet")
save_metadata(metadata, output_dir / "entities.parquet")
```

### Loading Metadata

```python
from newspaper_explorer.data.utils.metadata import load_metadata, find_metadata_for_parquet

# Load directly
metadata = load_metadata("results/der_tag/entities/gliner_multi_v2_1/metadata.json")

# Or find metadata for a parquet file
parquet_path = "results/der_tag/entities/gliner_multi_v2_1/entities.parquet"
metadata_path = find_metadata_for_parquet(parquet_path)
if metadata_path:
    metadata = load_metadata(metadata_path)
    print(f"Model: {metadata.model_name}")
    print(f"Parameters: {metadata.parameters}")
```

### Updating Metadata (Long-Running Processes)

```python
from newspaper_explorer.data.utils.metadata import update_metadata_status

# Initial save with in_progress status
metadata.status = "in_progress"
save_metadata(metadata, output_dir / "analysis.json")

# ... processing happens ...

# Update on completion
update_metadata_status(
    output_dir / "analysis.json",
    status="completed",
    completed_at=datetime.now().isoformat(),
    duration_seconds=123.45,
    output_data={"row_count": 5000},
)

# Or update on failure
update_metadata_status(
    output_dir / "analysis.json",
    status="failed",
    error_message="CUDA out of memory",
)
```

### Listing All Analyses

```python
from newspaper_explorer.data.utils.metadata import list_analyses
from newspaper_explorer.config.base import get_config

config = get_config()

# List all entity analyses for der_tag
analyses = list_analyses(
    config.results_dir,
    analysis_type="entities",
    source="der_tag",
)

for analysis in analyses:
    print(f"{analysis['analysis_id']}")
    print(f"  Model: {analysis['model_name']}")
    print(f"  Created: {analysis['created_at']}")
    print(f"  Duration: {analysis['duration_seconds']}s")
    print(f"  Parquet: {analysis.get('parquet_path', 'N/A')}")
```

---

## Foreign Key System

### ID Hierarchy

Analysis results reference source data through foreign key IDs:

```
source_id (e.g., "der_tag")
  └── issue_id (e.g., "der_tag_1902-09-05_415_2")
      └── page_id (e.g., "der_tag_1902-09-05_415_2_005")
          ├── text_block_id (e.g., "der_tag_1902-09-05_415_2_005_TB_1")
          │   └── line_id (e.g., "der_tag_1902-09-05_415_2_005_TB_1_TL_1")
          ├── detection_id (e.g., "der_tag_1902-09-05_415_2_005_headline_a3f9c2")
          └── article_id (e.g., "der_tag_1902-09-05_415_2_005_art_b7e4d1")
```

### Identifying ID Types

```python
from newspaper_explorer.data.utils.ids import identify_id_type

# Determine what type of ID you have
id_type = identify_id_type("3074409-X_1902-09-05_415_2_005_TB_1_TL_1")
print(id_type)  # "line_id"

id_type = identify_id_type("3074409-X_1902-09-05_415_2_005")
print(id_type)  # "page_id"

id_type = identify_id_type("3074409-X_1902-09-05_415_2_005_headline_a3f9c2")
print(id_type)  # "detection_id"
```

### Extracting Foreign Keys

```python
from newspaper_explorer.data.utils.ids import extract_foreign_keys

# Extract all parent IDs from any ID
fks = extract_foreign_keys("3074409-X_1902-09-05_415_2_005_TB_1_TL_1")

print(fks["source_id"])       # "3074409-X"
print(fks["issue_id"])        # "3074409-X_1902-09-05_415_2"
print(fks["page_id"])         # "3074409-X_1902-09-05_415_2_005"
print(fks["text_block_id"])   # "3074409-X_1902-09-05_415_2_005_TB_1"
print(fks["line_id"])         # "3074409-X_1902-09-05_415_2_005_TB_1_TL_1"
```

### Using Foreign Keys in Analysis

```python
import polars as pl
from newspaper_explorer.data.utils.ids import extract_foreign_keys

# Load analysis results
entities_df = pl.read_parquet("results/der_tag/entities/analysis/entities.parquet")

# Extract foreign keys if not already present
if "page_id" not in entities_df.columns:
    # Add foreign keys from line_id
    entities_df = entities_df.with_columns([
        pl.col("line_id").map_elements(
            lambda x: extract_foreign_keys(x)["page_id"],
            return_dtype=pl.Utf8
        ).alias("page_id"),
        pl.col("line_id").map_elements(
            lambda x: extract_foreign_keys(x)["issue_id"],
            return_dtype=pl.Utf8
        ).alias("issue_id"),
    ])
```

---

## Analysis-Specific Schemas

### Entity Extraction Results

```python
# entities.parquet schema
{
    "entity_id": str,           # Unique entity ID (if per-instance)
    "line_id": str,             # Foreign key to source line
    "entity_text": str,         # Extracted entity text
    "entity_type": str,         # person, organization, location, event
    "confidence": float,        # Model confidence (0-1)
    
    # Foreign keys (recommended to include)
    "source_id": str,
    "issue_id": str,
    "page_id": str,
    "text_block_id": str,
}
```

### Keywords Extraction Results

```python
# keywords.parquet schema
{
    "doc_id": str,              # Document ID (line_id or text_block_id)
    "keywords": list[str],      # Extracted keywords
    "scores": list[float],      # Keyword relevance scores
    
    # Foreign keys
    "source_id": str,
    "issue_id": str,
    "page_id": str,
}
```

### Emotion Classification Results

```python
# emotions.parquet schema
{
    "line_id": str,             # Foreign key to source line
    
    # Emotion predictions (binary + probability)
    "Joy": int,                 # 0 or 1
    "Joy_prob": float,          # Probability score
    "Fear": int,
    "Fear_prob": float,
    # ... (other emotions)
    
    # Foreign keys
    "source_id": str,
    "issue_id": str,
    "page_id": str,
    "text_block_id": str,
}
```

### Layout Detection Results

```python
# detections.parquet schema
{
    "detection_id": str,        # Unique detection ID
    "page_id": str,             # Foreign key to page
    "element_type": str,        # headline, image, caption, advertisement
    "confidence": float,        # Detection confidence
    
    # Bounding box
    "x": int, "y": int, 
    "width": int, "height": int,
    
    # Optional linked text
    "text_block_ids": list[str],  # Linked text blocks
    "text": str,                   # Extracted text
    
    # Foreign keys
    "source_id": str,
    "issue_id": str,
}
```

---

## Best Practices

### 1. Always Include Foreign Keys

Analysis results should preserve or extract foreign keys for traceability:

```python
# ✅ GOOD: Preserve foreign keys from input
results_df = input_df.select([
    "line_id",         # Primary/foreign key
    "source_id",       # Foreign key
    "issue_id",        # Foreign key
    "page_id",         # Foreign key
    "text_block_id",   # Foreign key
]).with_columns([
    # ... analysis results ...
])

# ❌ BAD: Drop all ID columns
results_df = pl.DataFrame({
    "entity_text": [...],
    "entity_type": [...],
    # No way to link back to source!
})
```

### 2. Extract Input/Output Statistics

Use utility functions to capture dataset info:

```python
from newspaper_explorer.data.utils.metadata import extract_input_stats, extract_output_stats

metadata.input_data = extract_input_stats(input_df, id_column="line_id")
metadata.output_data = extract_output_stats(results_df)
```

### 3. Use Consistent analysis_id Format

The `AnalysisMetadata` class auto-generates IDs in the format:

```
{method_type}_{model_name}_{timestamp}
```

Examples:
- `gliner_multi_v2_1_20251109_004150`
- `llm_gpt4o_mini_20251109_120530`
- `keybert_optimized_20251109_143022`

### 4. Store Analysis Parameters Completely

Include **all** parameters used in the analysis:

```python
# ✅ GOOD: Complete parameters
parameters = {
    "threshold": 0.2,
    "batch_size": 16,
    "labels": ["Person", "Organisation", "Ort"],
    "min_text_length": 10,
    "max_text_length": 1000,
    "chunking": "enabled",
    "device": "cuda:0",
}

# ❌ BAD: Incomplete parameters
parameters = {
    "threshold": 0.2,
}
```

### 5. Handle Failures Gracefully

Update metadata on errors:

```python
try:
    results_df = extractor.extract(input_df)
    metadata.status = "completed"
except Exception as e:
    metadata.status = "failed"
    metadata.error_message = str(e)
    raise
finally:
    save_metadata(metadata, output_path)
```

---

## Migration Guide

### Updating Existing Analysis Code

If you have existing analysis code without metadata support:

1. **Add metadata creation** before analysis:

```python
from newspaper_explorer.data.utils.metadata import AnalysisMetadata, save_metadata

metadata = AnalysisMetadata(
    analysis_type="your_type",
    method_type="your_method",
    model_name="your_model",
    source=source_name,
    parameters={...},  # All your parameters
)
```

2. **Update on completion**:

```python
metadata.completed_at = datetime.now().isoformat()
metadata.duration_seconds = time.time() - start_time
metadata.output_data = extract_output_stats(results_df)
```

3. **Save alongside parquet**:

```python
# Save results
results_df.write_parquet(output_path)

# Save metadata with same base name
save_metadata(metadata, output_path)
```

### Adding Foreign Keys to Existing Results

If your existing parquet files lack foreign keys:

```python
import polars as pl
from newspaper_explorer.data.utils.ids import extract_foreign_keys

# Load existing results
df = pl.read_parquet("results/entities/old_analysis.parquet")

# Extract foreign keys from existing ID column
df = df.with_columns([
    pl.col("line_id").map_elements(
        lambda x: extract_foreign_keys(x)["source_id"],
        return_dtype=pl.Utf8
    ).alias("source_id"),
    pl.col("line_id").map_elements(
        lambda x: extract_foreign_keys(x)["issue_id"],
        return_dtype=pl.Utf8
    ).alias("issue_id"),
    pl.col("line_id").map_elements(
        lambda x: extract_foreign_keys(x)["page_id"],
        return_dtype=pl.Utf8
    ).alias("page_id"),
])

# Overwrite with updated schema
df.write_parquet("results/entities/old_analysis.parquet")
```

---

## See Also

- **ID Generation**: `docs/development/ID_GENERATION.md` - Complete ID system documentation
- **Query Engine**: `docs/analysis/QUERY.md` - Querying analysis results with foreign keys
- **Configuration**: `docs/development/CONFIGURATION.md` - Managing paths and settings
