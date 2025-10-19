# Entity Extraction Adaptation Summary

## What We Did

Adapted entity extraction to use the **new LLM utilities and data architecture** pattern:

### ✅ New LLM-based Entity Extractor

**File:** `analysis/entities/llm_extraction.py`

Features:

- Uses `LLMClient` with automatic retry logic
- Validates responses with `EntityResponse` Pydantic schema
- Uses `entity_extraction` prompt template
- Saves results following new architecture:
  - `results/{source}/entities/{method_id}/entities.parquet`
  - `results/{source}/entities/{method_id}/metadata.json`
- Maintains `line_id` foreign keys to source data
- Creates metadata for full reproducibility

### ✅ Result Schema

**Parquet output:**

```python
{
    "line_id": str,          # FK to source lines
    "entity_text": str,      # "Kaiser Wilhelm II"
    "entity_type": str,      # "person", "location", "organization"
}
```

**Metadata JSON:**

```json
{
    "analysis_id": "llm_gpt4o_mini_20241019_143022",
    "analysis_type": "entities",
    "method_type": "llm",
    "model_name": "gpt-4o-mini",
    "parameters": {...},
    "created_at": "2024-10-19T14:30:22Z",
    "line_count": 15420,
    "duration_seconds": 3600.5
}
```

### ✅ Queryable Results

Can now use `QueryEngine` to query entities:

```python
from newspaper_explorer.utils.queries import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Find mentions
    mentions = qe.find_entity_mentions("Kaiser Wilhelm II", method="llm_gpt4o_mini_...")

    # Get frequency
    freq = qe.entity_frequency(method="...", group_by="year")

    # Compare methods
    diff = qe.compare_entity_methods("llm_...", "gliner_...")
```

## Usage

### Simple Extraction

```python
from newspaper_explorer.analysis.entities.llm_extraction import extract_entities_llm

# Test with 10 lines
results = extract_entities_llm(source_name="der_tag", limit=10)

# Full extraction
results = extract_entities_llm(source_name="der_tag")
```

### Custom Configuration

```python
from newspaper_explorer.analysis.entities.llm_extraction import LLMEntityExtractor

extractor = LLMEntityExtractor(
    source_name="der_tag",
    model_name="gpt-4o-mini",
    temperature=0.2,      # More deterministic
    max_retries=5,
)

results = extractor.extract_and_save(limit=50)
```

### Query Results

```python
from newspaper_explorer.utils.queries import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Custom SQL
    result = qe.query("""
        SELECT entity_text, COUNT(*) as mentions
        FROM 'results/der_tag/entities/llm_gpt4o_mini_.../entities.parquet'
        WHERE entity_type = 'person'
        GROUP BY entity_text
        ORDER BY mentions DESC
        LIMIT 10
    """)
```

## Files Created

1. **`analysis/entities/llm_extraction.py`** - LLM-based extractor (300+ lines)
2. **`analysis/entities/llm_examples.py`** - Usage examples
3. **`docs/ENTITIES.md`** - Updated documentation

## Key Differences from Old Method

### Old (GLiNER)

```python
extractor = EntityExtractor(source_name="der_tag")
results = extractor.extract_and_save(input_path="...")

# Results saved to: results/der_tag/entities/entities_raw.parquet
# No method tracking, no metadata, no line_id FK
```

### New (LLM)

```python
results = extract_entities_llm(source_name="der_tag", limit=100)

# Results saved to: results/der_tag/entities/{method_id}/entities.parquet
# With metadata.json, line_id FK, queryable via DuckDB
```

## Benefits

1. **Structured validation** - Pydantic ensures correct format
2. **Automatic retries** - Handles API failures gracefully
3. **Method tracking** - Know exactly which model/parameters were used
4. **Queryable** - Use SQL to analyze results
5. **Source attribution** - Every entity linked back to original text via `line_id`
6. **Reproducible** - Full metadata for reproducibility

## Old GLiNER Extractor

Still available in `analysis/entities/extraction.py` for:

- Faster extraction
- Offline work
- No API costs

Both extractors work side-by-side!

## Next Steps

1. **Test**: Run `python -m newspaper_explorer.analysis.entities.llm_examples`
2. **Configure**: Set `LLM_BASE_URL` and `LLM_API_KEY` in `.env`
3. **Extract**: Start with `limit=10` for testing
4. **Query**: Use `QueryEngine` to explore results
5. **Compare**: Run both LLM and GLiNER, compare results
6. **Scale**: Remove `limit` for full extraction

## Example Workflow

```python
# 1. Extract entities
from newspaper_explorer.analysis.entities.llm_extraction import extract_entities_llm

results = extract_entities_llm(source_name="der_tag", limit=100)
method_id = results['metadata']['analysis_id']

# 2. Query results
from newspaper_explorer.utils.queries import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Find top persons
    freq = qe.entity_frequency(
        method=method_id,
        entity_type="person",
        min_mentions=3
    )
    print(freq.head(10))

    # Get mentions with context
    mentions = qe.find_entity_mentions(
        entity_name="Kaiser Wilhelm II",
        method=method_id
    )

    for row in mentions.iter_rows(named=True):
        print(f"[{row['date']}] {row['text'][:100]}...")
```

Perfect for your newspaper analysis pipeline! 🎉
