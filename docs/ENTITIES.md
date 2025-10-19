# Entity Extraction

Extract named entities (persons, organizations, events, locations) from historical newspaper text using GLiNER.

## Overview

**Purpose**: Identify and extract named entities from newspaper text for analysis and research.

**Supported Entities**:

- **Person**: Names of individuals
- **Organisation**: Companies, institutions, groups
- **Ereignis**: Events, occurrences
- **Ort**: Locations, places

**Output Structure**: `results/{source}/entities/`

## Quick Start

```bash
# Extract entities from text blocks
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/textblocks.parquet
```

## Usage

### CLI Command

```bash
newspaper-explorer analyze extract-entities \
    --source SOURCE \
    --input INPUT_FILE \
    [OPTIONS]
```

**Required Arguments**:

- `--source`: Source name (e.g., `der_tag`)
- `--input`: Path to input parquet file (textblocks or sentences)

**Options**:

- `--text-column`: Column containing text (default: `text`)
- `--id-column`: Column to use as identifier (default: `text_block_id`)
- `--normalize / --no-normalize`: Normalize text before extraction (default: `True`)
- `--model`: GLiNER model from Hugging Face (default: `urchade/gliner_multi-v2.1`)
- `--labels`: Comma-separated entity labels (default: `Person,Organisation,Ereignis,Ort`)
- `--threshold`: Confidence threshold 0-1 (default: `0.5`)
- `--batch-size`: Batch size for processing (default: `32`)
- `--min-length`: Minimum text length to process (default: `100`)
- `--max-length`: Maximum text length, truncate longer (default: `500`)
- `--format`: Output format (`parquet`, `json`, or `both`, default: `both`)

### Examples

```bash
# Basic extraction with defaults
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/textblocks.parquet

# Extract from sentences with custom settings
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/sentences.parquet \
    --text-column sentence \
    --batch-size 64 \
    --threshold 0.6

# Custom labels without normalization
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/textblocks.parquet \
    --labels "Person,Ort,Firma,Produkt" \
    --no-normalize \
    --format json
```

### Python API

```python
from newspaper_explorer.analysis.entities.extraction import EntityExtractor

# Initialize extractor
extractor = EntityExtractor(
    source_name="der_tag",
    model_name="urchade/gliner_multi-v2.1",
    labels=["Person", "Organisation", "Ereignis", "Ort"],
    threshold=0.5,
    batch_size=32,
    min_text_length=100,
    max_text_length=500
)

# Run complete pipeline
results = extractor.extract_and_save(
    input_path="data/processed/der_tag/text/textblocks.parquet",
    text_column="text",
    id_column="text_block_id",
    normalize=True,
    output_format="both"
)

# Access results
entities_df = results["entities"]  # Polars DataFrame
serialized = results["serialized"]  # Dict[str, Dict[str, List[str]]]
```

### Advanced Usage

```python
import polars as pl
from newspaper_explorer.analysis.entities.extraction import EntityExtractor

# Initialize
extractor = EntityExtractor(source_name="der_tag")

# Load and prepare data manually
df = pl.read_parquet("data/processed/der_tag/text/textblocks.parquet")
df_prepared = extractor.prepare_texts(df, normalize=True)

# Extract entities
entities_df = extractor.extract_entities(df_prepared)

# Serialize
serialized = extractor.serialize_entities(entities_df)

# Save with custom logic
extractor.save_results(entities_df, serialized, format="json")
```

## Output Format

### Raw Entities (`entities_raw.parquet`)

Polars DataFrame with all extracted entities:

| id               | entity                                                         |
| ---------------- | -------------------------------------------------------------- |
| `text_block_123` | `{"text": "Berlin", "label": "Ort", "score": 0.95}`            |
| `text_block_123` | `{"text": "Kaiser Wilhelm", "label": "Person", "score": 0.88}` |

### Grouped Entities (`entities_grouped.json`)

JSON structure grouped by ID and label:

```json
{
  "text_block_123": {
    "Person": ["kaiser wilhelm", "bismarck"],
    "Organisation": ["reichstag", "berliner morgenpost"],
    "Ereignis": ["weltkrieg"],
    "Ort": ["berlin", "paris"]
  },
  "text_block_124": {
    ...
  }
}
```

## Pipeline Integration

Complete workflow from raw XML to entity extraction:

```bash
# 1. Download and extract XML data
newspaper-explorer data download --source der_tag --all

# 2. Load XML to Parquet (line-level)
newspaper-explorer data load --source der_tag

# 3. Aggregate into text blocks
python -c "
from newspaper_explorer.data.utils.text import load_and_aggregate_textblocks
df = load_and_aggregate_textblocks('data/raw/der_tag/text/der_tag_lines.parquet')
"

# 4. Extract entities
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data/processed/der_tag/text/textblocks.parquet
```

## Text Normalization

Entity extraction benefits from text normalization for historical German:

**Normalization steps**:

1. Replace archaic characters (ſ → s, ẞ → SS, ß → ss)
2. Remove diacritics (ä → a, ö → o, ü → u)
3. Lowercase text
4. Normalize whitespace
5. Remove non-alphabetic characters
6. Lemmatize tokens

**Example**:

```
Input:  "Dieſes Beiſpiel zeigt die Schönheit alter Buchſtaben"
Output: "dieser beispiel zeigen schonheit alter buchstabe"
```

Normalization can be disabled with `--no-normalize` if working with already normalized text or modern German.

## Performance

### Processing Speed

Typical performance with default settings:

- **CPU**: ~50-100 texts/minute
- **GPU**: ~500-1000 texts/minute

Adjust `--batch-size` based on available memory:

- Small GPU (4-8GB): `--batch-size 16`
- Medium GPU (8-16GB): `--batch-size 32` (default)
- Large GPU (16GB+): `--batch-size 64` or higher

### Memory Requirements

- **Model size**: ~500MB (downloaded once, cached)
- **Runtime memory**:
  - CPU: ~2-4GB
  - GPU: ~4-8GB (depends on batch size)

### Disk Space

Output size depends on text volume and entity density:

- **Raw parquet**: ~10-50MB per 10K texts
- **Grouped JSON**: ~5-20MB per 10K texts

## Model Information

### Default Model: `urchade/gliner_multi-v2.1`

- **Type**: GLiNER (Generalist and Lightweight Named Entity Recognition)
- **Languages**: Multilingual (including German)
- **Advantages**:
  - Zero-shot entity recognition
  - Custom labels without retraining
  - Good performance on historical text
  - Efficient inference

### Alternative Models

You can use any GLiNER model from Hugging Face:

```bash
# Larger model for better accuracy
newspaper-explorer analyze extract-entities \
    --source der_tag \
    --input data.parquet \
    --model urchade/gliner_large-v2.1
```

## Troubleshooting

### "No entities found"

- Lower `--threshold` (e.g., `0.3`)
- Check `--min-length` isn't too restrictive
- Verify text column contains actual text
- Try without normalization (`--no-normalize`)

### Out of memory errors

- Reduce `--batch-size` (e.g., `16` or `8`)
- Use CPU instead of GPU (automatic fallback)
- Reduce `--max-length` to process shorter texts

### Slow processing

- Increase `--batch-size` if memory allows
- Use GPU if available (automatic detection)
- Process smaller batches of data

### Poor entity quality

- Increase `--threshold` (e.g., `0.6` or `0.7`)
- Customize `--labels` for your domain
- Use normalization (`--normalize`)

## Analysis Examples

### Count Entity Frequencies

```python
import polars as pl

# Load raw entities
df = pl.read_parquet("results/der_tag/entities/entities_raw.parquet")

# Count by label
by_label = df.group_by(pl.col("entity").struct.field("label")).count()
print(by_label)

# Most common persons
persons = df.filter(pl.col("entity").struct.field("label") == "Person")
top_persons = persons.group_by(pl.col("entity").struct.field("text")).count().sort("count", descending=True)
print(top_persons.head(20))
```

### Find Co-occurring Entities

```python
import json

# Load grouped entities
with open("results/der_tag/entities/entities_grouped.json") as f:
    data = json.load(f)

# Find texts mentioning specific person and place
for text_id, entities in data.items():
    if "berlin" in entities["Ort"] and any("wilhelm" in p for p in entities["Person"]):
        print(f"Found co-occurrence in {text_id}")
```

## Related Documentation

- [Text Utilities](../data/utils/text.py) - Text processing functions
- [Data Loading](DATA_LOADER.md) - Loading newspaper data
- [CLI Reference](CLI.md) - Complete command reference
