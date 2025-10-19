# Data Architecture Quick Reference

## Storage Pattern

```
data/raw/{source}/text/
└── {source}_lines.parquet          # Source data

results/{source}/{analysis_type}/{method_id}/
├── results.parquet                  # Analysis results with line_id FK
└── metadata.json                    # Method/model info
```

## Result Schema (All Types)

```python
{
    "line_id": str,          # REQUIRED: FK to source
    ...                      # Analysis-specific fields
}
```

## Query Pattern

```python
from newspaper_explorer.utils.queries import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Find entity mentions
    mentions = qe.find_entity_mentions("Kaiser Wilhelm II")

    # Get full text for line
    line = qe.get_line(line_id="...")

    # Custom SQL
    result = qe.query("SELECT ... FROM source_lines WHERE ...")
```

## Method ID Format

```
{method_type}_{model_name}_{timestamp}
```

Examples:

- `llm_gpt4o_mini_20241019_143022`
- `spacy_de_core_news_lg_20241019_150000`
- `bertopic_20241019_151500`

## Metadata Template

```python
from newspaper_explorer.utils.queries import create_result_metadata

metadata = create_result_metadata(
    analysis_type="entities",
    method_type="llm",
    model_name="gpt-4o-mini",
    source="der_tag",
    parameters={"temperature": 0.3},
    line_count=1000,
    duration_seconds=120.5
)
```

## Save Results Pattern

```python
import polars as pl
import json
from pathlib import Path

# Create results DataFrame with line_id
results_df = pl.DataFrame({
    "line_id": source_df["line_id"],
    "entity_text": entities,
    "entity_type": types,
})

# Save to method-specific directory
output_dir = Path(f"results/der_tag/entities/{metadata['analysis_id']}")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.write_parquet(output_dir / "entities.parquet")

# Save metadata
with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

## UI Pattern

### FastAPI

```python
from fastapi import FastAPI
from newspaper_explorer.utils.queries import QueryEngine

app = FastAPI()
qe = QueryEngine(source="der_tag")

@app.get("/api/entities/{name}")
def get_entity(name: str, method: str = "llm_gpt4o_mini"):
    return qe.find_entity_mentions(name, method).to_dicts()
```

### Streamlit

```python
import streamlit as st
from newspaper_explorer.utils.queries import QueryEngine

qe = QueryEngine(source="der_tag")
entity = st.text_input("Entity:")
if entity:
    st.dataframe(qe.find_entity_mentions(entity))
```

## Common Queries

### All mentions of entity

```python
qe.find_entity_mentions("Berlin", method="llm_gpt4o_mini")
```

### Entity frequency by year

```python
qe.entity_frequency(method="llm_gpt4o_mini", group_by="year")
```

### Compare methods

```python
qe.compare_entity_methods("llm_gpt4o_mini", "spacy_de")
```

### Search text

```python
qe.search_text("Kaiser", start_date="1901-01-01", limit=100)
```

### Get line details

```python
qe.get_line(line_id="der_tag_1901_01_15_001")
```

## Installation

```bash
pip install duckdb
```

## Files

- **Architecture**: `docs/DATA_ARCHITECTURE.md`
- **Implementation**: `utils/queries.py`
- **Examples**: `utils/query_examples.py`
