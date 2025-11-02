# Entity Extraction

Extract named entities (persons, organizations, locations) from historical newspaper text using **LLM** or **GLiNER**.

## Overview

**Purpose**: Identify and extract named entities from newspaper text for analysis and research.

**Two Methods Available**:

1. **LLM-based** - High quality, structured validation, context-aware (requires API)
2. **GLiNER-based** - Fast, offline, traditional ML (runs locally)

**Supported Entities**:

- **person**: Names of individuals
- **location**: Places, cities, countries
- **organization**: Companies, institutions, groups

**Output Structure**: `results/{source}/entities/{method_id}/`

- `entities.parquet`: Extracted entities with line_id foreign key
- `metadata.json`: Method details, parameters, timestamps

**Architecture**: Both methods follow the same data architecture pattern:

- Save to `results/{source}/entities/{method_id}/`
- Include `line_id` foreign key for source attribution
- Create `metadata.json` for reproducibility
- Queryable via `QueryEngine` with DuckDB

## Quick Start

### LLM-based Extraction (High Quality)

```python
from newspaper_explorer.analysis.entities.llm_extraction import extract_entities_llm

# Test with first 100 lines
results = extract_entities_llm(
    source_name="der_tag",
    model_name="gpt-4o-mini",
    temperature=0.3,
    limit=100
)

# Full extraction
results = extract_entities_llm(source_name="der_tag")

print(f"Extracted {len(results['results_df'])} entities")
print(f"Saved to: {results['output_dir']}")
print(f"Method ID: {results['metadata']['analysis_id']}")
```

### GLiNER-based Extraction (Fast & Local)

```python
from newspaper_explorer.analysis.entities.gliner_extraction import extract_entities_gliner

# Test with first 100 lines
results = extract_entities_gliner(
    source_name="der_tag",
    threshold=0.5,
    batch_size=32,
    limit=100
)

# Full extraction
results = extract_entities_gliner(source_name="der_tag")

print(f"Extracted {len(results['results_df'])} entities")
print(f"Saved to: {results['output_dir']}")
print(f"Method ID: {results['metadata']['analysis_id']}")
```

### Compare Both Methods

```python
from newspaper_explorer.analysis.entities.method_comparison import run_both_methods, compare_entities

# Run both on same data
results = run_both_methods(source_name="der_tag", limit=100)

# Compare results
comparisons = compare_entities(
    source_name="der_tag",
    llm_method_id=results["llm"]["metadata"]["analysis_id"],
    gliner_method_id=results["gliner"]["metadata"]["analysis_id"]
)

print(comparisons["statistics"])  # Overall stats
print(comparisons["agreement"])   # Entities found by both
```

## LLM-based Extraction

### Configuration

Requires environment variables in `.env`:

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
```

### Python API

```python
from newspaper_explorer.analysis.entities.llm_extraction import LLMEntityExtractor

# Initialize with custom settings
extractor = LLMEntityExtractor(
    source_name="der_tag",
    model_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
    batch_size=10
)

# Extract from DataFrame (limit for testing)
from newspaper_explorer.data.loading import DataLoader
loader = DataLoader(source_name="der_tag")
df = loader.load_source().head(50)

results = extractor.extract_from_dataframe(df)
print(f"Extracted {len(results)} entities")

# Save with metadata
saved = extractor.extract_and_save(limit=100)
print(f"Saved to: {saved['output_dir']}")
```

### Output Schema

Parquet file with schema:

- `line_id` (str): Foreign key to source lines
- `entity_text` (str): The extracted entity text
- `entity_type` (str): One of: person, location, organization
- `start_char` (int): Character offset in text (optional)
- `end_char` (int): Character offset in text (optional)

### Advantages

- High accuracy with context understanding
- Consistent entity type classification
- Handles ambiguous cases well
- Structured validation via Pydantic

### Limitations

- Requires API access and costs
- Slower than GLiNER (API calls)
- Needs internet connection

## GLiNER-based Extraction

### Configuration

No API required - runs completely offline.

### Python API

```python
from newspaper_explorer.analysis.entities.gliner_extraction import GLiNEREntityExtractor

# Initialize extractor
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    model_name="urchade/gliner_multi-v2.1",
    threshold=0.5,
    batch_size=32,
    normalize=True
)

# Extract from DataFrame
from newspaper_explorer.data.loading import DataLoader
loader = DataLoader(source_name="der_tag")
df = loader.load_source().head(100)

results = extractor.extract_from_dataframe(df)
print(f"Extracted {len(results)} entities")

# Save with metadata
saved = extractor.extract_and_save(limit=500)
print(f"Saved to: {saved['output_dir']}")
```

### Output Schema

Same schema as LLM method:

- `line_id` (str): Foreign key to source lines
- `entity_text` (str): The extracted entity text
- `entity_type` (str): One of: person, location, organization
- `confidence` (float): Model confidence score (0-1)
- `start_char` (int): Character offset in text
- `end_char` (int): Character offset in text

### Advantages

- Fast processing (especially with GPU)
- No API costs or internet required
- Good performance on German text
- Confidence scores for filtering

### Limitations

- May miss subtle context
- Requires GPU for best performance
- ~500MB model download

### Model Information

**Default Model**: `urchade/gliner_multi-v2.1`

- Type: GLiNER (Generalist and Lightweight Named Entity Recognition)
- Languages: Multilingual (including German)
- Zero-shot entity recognition
- Good on historical text

**Alternative Models**:

```python
# Larger model for better accuracy
extractor = GLiNEREntityExtractor(
    source_name="der_tag",
    model_name="urchade/gliner_large-v2.1"
)
```

## Querying Results

Both methods output compatible formats queryable with `QueryEngine`:

```python
from newspaper_explorer.analysis.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Find all mentions of "Berlin"
    mentions = qe.find_entity_mentions(
        entity_text="Berlin",
        method_id="gliner_multi_v2_1_20241019"
    )

    # Entity frequency
    freq = qe.entity_frequency(
        method_id="llm_gpt4o_mini_20241019",
        entity_type="person",
        limit=20
    )

    # Compare methods
    diff = qe.compare_entity_methods(
        method1="llm_gpt4o_mini_20241019",
        method2="gliner_multi_v2_1_20241019"
    )
```

## Method Comparison

Use the comparison utility to evaluate both methods:

```python
from newspaper_explorer.analysis.entities.method_comparison import (
    run_both_methods,
    compare_entities,
    print_comparison_report
)

# Run both on same sample
results = run_both_methods(source_name="der_tag", limit=100)

# Get detailed comparison
comparisons = compare_entities(
    source_name="der_tag",
    llm_method_id=results["llm"]["metadata"]["analysis_id"],
    gliner_method_id=results["gliner"]["metadata"]["analysis_id"]
)

# Print formatted report
print_comparison_report(results, comparisons)
```

Output includes:

- Execution metadata (duration, model, parameters)
- Overall statistics (total/unique entities)
- Entity type distribution
- Entities found by both methods
- Method-specific entities (LLM-only, GLiNER-only)
- Line coverage comparison

## Usage Recommendations

**Use LLM when**:

- Quality is critical over speed
- Working with ambiguous historical text
- Budget allows API costs
- Need context-aware extraction

**Use GLiNER when**:

- Processing large volumes quickly
- No internet/API access available
- Running on local hardware
- Cost is a concern

**Use Both when**:

- Evaluating extraction quality
- Building ground truth datasets
- Comparing different approaches
- Maximum coverage desired (union of results)

## Text Normalization

Entity extraction can use text normalization for historical German:

**Normalization steps** (when `normalize=True`):

1. Replace archaic characters (ſ → s, ẞ → SS, ß → ss)
2. Remove diacritics (ä → a, ö → o, ü → u)
3. Lowercase text
4. Normalize whitespace
5. Clean punctuation

**Example**:

```
Input:  "Dieſes Beiſpiel zeigt die Schönheit alter Buchſtaben"
Output: "dieses beispiel zeigt die schonheit alter buchstaben"
```

**Usage**:

```python
# LLM extraction with normalization
results = extract_entities_llm(source_name="der_tag", normalize=True)

# GLiNER extraction without normalization
extractor = GLiNEREntityExtractor(source_name="der_tag", normalize=False)
```

**Recommendation**: Enable normalization for historical text, disable for modern German.

## Performance

### LLM-based

- **Speed**: ~10-50 lines/minute (depends on API rate limits)
- **Memory**: Minimal (<1GB)
- **Cost**: Per API call (~$0.0001-0.001 per line depending on model)

### GLiNER-based

- **Speed**:
  - CPU: ~50-100 lines/minute
  - GPU: ~500-1000 lines/minute
- **Memory**:
  - Model: ~500MB (one-time download)
  - Runtime (CPU): ~2-4GB
  - Runtime (GPU): ~4-8GB (depends on batch size)
- **Cost**: Free (runs locally)

### Batch Size Guidelines

GLiNER batch size based on hardware:

- Small GPU (4-8GB): `batch_size=16`
- Medium GPU (8-16GB): `batch_size=32` (default)
- Large GPU (16GB+): `batch_size=64`
- CPU: `batch_size=8`

## Examples

See dedicated example files:

- `src/newspaper_explorer/analysis/entities/llm_examples.py` - LLM extraction examples
- `src/newspaper_explorer/analysis/entities/gliner_examples.py` - GLiNER extraction examples
- `src/newspaper_explorer/analysis/entities/method_comparison.py` - Comparison examples

## Troubleshooting

### LLM Method

**API errors**:

- Check `.env` has correct `LLM_BASE_URL` and `LLM_API_KEY`
- Verify API key has sufficient credits
- Check network connection

**No entities found**:

- Try lower `temperature` (e.g., `0.1`)
- Increase `max_tokens` if responses are truncated
- Check prompt in `utils/prompts.py` is appropriate for your text

**Rate limiting**:

- Reduce `batch_size` to slow request rate
- Add delays between requests (modify LLMClient)

### GLiNER Method

**No entities found**:

- Lower `threshold` (e.g., `0.3`)
- Check text has sufficient length (`min_text_length`)
- Try without normalization
- Verify text column contains actual text

**Out of memory errors**:

- Reduce `batch_size` (e.g., `16` or `8`)
- Will automatically fall back to CPU
- Use CPU explicitly: `device="cpu"`

**Slow processing**:

- Increase `batch_size` if memory allows
- Use GPU if available (automatic detection)
- Check GPU is being used: `extractor.device`

**Poor entity quality**:

- Increase `threshold` (e.g., `0.6` or `0.7`)
- Try different GLiNER model
- Enable normalization for historical text

### Both Methods

**Missing source attribution**:

- Ensure input DataFrame has `line_id` column
- Check results include `line_id` in output

**Can't query results**:

- Verify parquet file exists in `results/{source}/entities/{method_id}/`
- Check `metadata.json` is present
- Use correct method_id in QueryEngine

## Analysis Examples

### Find Entity Mentions

```python
from newspaper_explorer.analysis.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Find all mentions of "Berlin"
    mentions = qe.find_entity_mentions(
        entity_text="Berlin",
        method_id="gliner_multi_v2_1_20241019"
    )

    # Get original text for context
    for mention in mentions[:5].iter_rows(named=True):
        line_id = mention["line_id"]
        text = qe.query(f"SELECT text FROM lines WHERE line_id = '{line_id}'")
        print(f"Found in: {text}")
```

### Entity Frequency Analysis

```python
with QueryEngine(source="der_tag") as qe:
    # Most common persons
    persons = qe.entity_frequency(
        method_id="llm_gpt4o_mini_20241019",
        entity_type="person",
        limit=20
    )
    print(persons)

    # All organizations
    orgs = qe.entity_frequency(
        method_id="gliner_multi_v2_1_20241019",
        entity_type="organization"
    )
    print(orgs)
```

### Custom Queries

```python
import polars as pl
from newspaper_explorer.analysis.query.engine import QueryEngine

with QueryEngine(source="der_tag") as qe:
    # Entities by year
    by_year = qe.query("""
        SELECT
            YEAR(l.date) as year,
            e.entity_type,
            COUNT(DISTINCT e.entity_text) as unique_entities
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e
        JOIN lines l ON e.line_id = l.line_id
        GROUP BY year, e.entity_type
        ORDER BY year
    """)

    # Co-occurring entities (same line)
    cooccurrence = qe.query("""
        SELECT
            e1.entity_text as entity1,
            e2.entity_text as entity2,
            COUNT(*) as cooccurrence_count
        FROM 'results/der_tag/entities/{method_id}/entities.parquet' e1
        JOIN 'results/der_tag/entities/{method_id}/entities.parquet' e2
            ON e1.line_id = e2.line_id
            AND e1.entity_text < e2.entity_text
        WHERE e1.entity_type = 'person' AND e2.entity_type = 'person'
        GROUP BY entity1, entity2
        ORDER BY cooccurrence_count DESC
        LIMIT 20
    """)
```

## Related Documentation

- [LLM Utilities](LLM.md) - LLM client and prompts
- [Data Architecture](DATA_ARCHITECTURE.md) - Storage and querying patterns
- [Query Architecture](QUERY_ARCHITECTURE.md) - QueryEngine details
- [Data Loading](DATA_LOADER.md) - Loading newspaper data
- [CLI Reference](CLI.md) - Complete command reference
