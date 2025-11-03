# TF-IDF Keyword Extraction

Extract meaningful keywords from newspaper texts using **TF-IDF** (Term Frequency-Inverse Document Frequency).

## What is TF-IDF?

TF-IDF is a statistical measure that evaluates how relevant a word is to a document within a collection of documents. It works by multiplying two metrics:

1. **Term Frequency (TF)**: How often a word appears in a document
2. **Inverse Document Frequency (IDF)**: How rare a word is across all documents

Words with high TF-IDF scores are **frequent in a specific document but rare across the corpus**, making them excellent keywords that capture the unique content of that document.

### Why TF-IDF for Newspapers?

- **Filters common words automatically**: Words like "der", "die", "und" that appear everywhere get low scores
- **Highlights unique content**: Words specific to particular articles, dates, or topics rise to the top
- **No training required**: Works immediately on any text corpus
- **Interpretable results**: Direct word scores, not black-box embeddings

## Quick Start

### Basic Usage

Extract keywords per newspaper page (recommended):

```bash
newspaper-explorer analyze keywords tfidf --source der_tag
```

This creates one "document" per newspaper page and extracts the top 10 keywords for each.

### Extract Keywords by Date

Get daily keyword trends:

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --document-level date \
    --top-k 20
```

### Extract Keywords by Year

See how topics evolved over time:

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --group-by year \
    --top-k 15 \
    --ngram-range 1,2
```

## Document Levels

The `--document-level` option defines what constitutes a "document" for TF-IDF analysis:

| Level | Description | Use Case |
|-------|-------------|----------|
| **page** | One document per newspaper page | Default, recommended for most analyses |
| **textblock** | One document per text block | Fine-grained analysis of article segments |
| **file** | One document per XML file (entire issue) | Analyzing full newspaper issues |
| **date** | One document per publication date | Daily keyword trends |

### Custom Grouping

For more control, use `--group-by` with any column(s):

```bash
# Group by year and month
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --group-by year,month

# Group by newspaper title (if multiple in corpus)
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --group-by newspaper_title,year
```

## Preprocessing Integration

The TF-IDF extractor integrates with the existing preprocessing pipeline for better results.

### Default Preprocessing

By default, applies basic cleaning:
- Normalize whitespace
- Convert to lowercase
- Remove punctuation

```bash
newspaper-explorer analyze keywords tfidf --source der_tag
# Automatically applies: normalize-whitespace, lowercase, remove-punctuation
```

### Custom Preprocessing

Specify your own preprocessing steps:

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --preprocessing normalize,lowercase,remove-stopwords
```

Available steps (from `data.preprocessing` module):
- `normalize` - Normalize historical German characters
- `normalize-whitespace` - Normalize whitespace
- `lowercase` - Convert to lowercase
- `remove-punctuation` - Remove punctuation
- `remove-stopwords` - Remove German stopwords
- `remove-diacritics` - Remove diacritics/accents
- `normalize-transnormer` - Neural normalization (requires GPU)
- `normalize-dtacab` - DTA-CAB normalization (requires API)

### Use Already Preprocessed Text

If you've already run preprocessing:

```bash
# First, preprocess the data
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize,lowercase,remove-stopwords

# Then use the preprocessed file
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --input-file data/processed/der_tag/textblocks_normalized.parquet \
    --text-column text_normalized \
    --no-preprocessing
```

### Skip Preprocessing (Raw Text)

To analyze raw OCR text:

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --no-preprocessing
```

## Parameters

### Document Configuration

- `--document-level`: Defines document boundaries (page/textblock/file/date)
- `--group-by`: Custom grouping columns (overrides document-level)
- `--input-file`: Custom parquet file (default: auto-detect lines or textblocks)
- `--text-column`: Column containing text (default: "text")

### TF-IDF Parameters

- `--top-k`: Number of keywords per document (default: 10)
- `--min-df`: Minimum document frequency (default: 2)
  - Words must appear in at least this many documents
  - Higher = filters rare/misspelled words
- `--max-df`: Maximum document frequency as fraction (default: 0.8)
  - Ignore words appearing in more than 80% of documents
  - Filters very common words
- `--ngram-range`: N-gram range as "min,max" (default: "1,1")
  - "1,1" = unigrams only ("krieg", "wirtschaft")
  - "1,2" = unigrams + bigrams ("erste weltkrieg", "neue gesetze")

### Stopwords

- `--no-stopwords`: Disable built-in German stopwords
- `--stopwords`: Add custom stopwords (comma-separated)

### Preprocessing

- `--preprocessing`: Comma-separated steps (e.g., "normalize,lowercase")
- `--no-preprocessing`: Skip all preprocessing

### Other

- `--output-name`: Output filename (default: "tfidf_keywords")
- `--limit`: Limit rows for testing

## Examples

### Example 1: Page-Level Keywords with Bigrams

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --document-level page \
    --top-k 15 \
    --ngram-range 1,2
```

Extracts top 15 keywords (including 2-word phrases) from each newspaper page.

### Example 2: Yearly Trends with Strict Filtering

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --group-by year \
    --min-df 5 \
    --max-df 0.5 \
    --top-k 20
```

Yearly keywords, only keeping words that appear in:
- At least 5 documents (min-df)
- No more than 50% of documents (max-df)

### Example 3: Custom Stopwords for Newspaper Domain

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --document-level date \
    --stopwords "zeitung,seite,ausgabe,nummer,tag"
```

Filters out newspaper-specific common words.

### Example 4: Test with Small Sample

```bash
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --limit 1000 \
    --document-level page
```

Process only 1000 rows to test parameters quickly.

### Example 5: Advanced Preprocessing Pipeline

```bash
# Use neural normalization for historical text
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --preprocessing normalize-transnormer,lowercase,remove-stopwords \
    --group-by year \
    --top-k 25
```

## Python API

```python
from newspaper_explorer.analyze.keywords.tf_idf import TFIDFExtractor

# Initialize extractor
extractor = TFIDFExtractor(
    source_name="der_tag",
    use_stopwords=True,
    custom_stopwords=["zeitung", "ausgabe"],
)

# Extract keywords per page
results = extractor.extract_keywords(
    document_level="page",
    top_k=10,
    min_df=2,
    max_df=0.8,
)

# Extract keywords per year with bigrams
yearly_keywords = extractor.extract_keywords(
    group_by=["year"],
    top_k=20,
    ngram_range=(1, 2),
    preprocessing_steps=["normalize", "lowercase", "remove-punctuation"],
)

# Save results
output_file = extractor.save_results(yearly_keywords, output_name="yearly_keywords")
print(f"Saved to: {output_file}")
```

### Single Document Analysis

```python
from newspaper_explorer.analyze.keywords.tf_idf import extract_keywords_from_document

# Your corpus
corpus = [
    "der krieg hat viele opfer gefordert",
    "die wirtschaft wächst stetig",
    # ... more documents
]

# Extract keywords from one document
document = "der erste weltkrieg endete mit vielen verlusten"
result = extract_keywords_from_document(
    document=document,
    corpus=corpus,
    top_k=5,
)

print(result['keywords'])  # ['weltkrieg', 'erste', 'verlusten', ...]
print(result['scores'])    # [0.8234, 0.7123, ...]
```

## Output Format

Results are saved to: `results/{source}/keywords/{output_name}.parquet`

### Schema

```python
{
    # Grouping columns (depends on document_level or group_by)
    "filename": str,        # If document_level="page"
    "page_number": int,     # If document_level="page"
    "year": int,            # If document_level="date" or group_by="year"
    "month": int,           # If document_level="date"
    "day": int,             # If document_level="date"
    
    # Keywords
    "keywords": list[str],  # Top keywords (sorted by score)
    "scores": list[float],  # TF-IDF scores
}
```

### Loading Results

```python
import polars as pl

# Load results
df = pl.read_parquet("results/der_tag/keywords/tfidf_keywords.parquet")

# Get keywords for a specific date
date_keywords = df.filter(
    (pl.col("year") == 1920) & 
    (pl.col("month") == 1) & 
    (pl.col("day") == 15)
)

# Show top keywords
for row in date_keywords.iter_rows(named=True):
    print(f"Keywords: {', '.join(row['keywords'][:10])}")
```

## Understanding the Scores

TF-IDF scores typically range from 0 to ~1 (or higher for very unique terms):

- **0.0 - 0.2**: Low relevance (common words that passed filtering)
- **0.2 - 0.5**: Moderate relevance (fairly distinctive words)
- **0.5 - 0.8**: High relevance (distinctive topic words)
- **0.8+**: Very high relevance (rare, document-specific terms)

### What Makes a Good Keyword?

- **High frequency in document**: Appears many times
- **Low frequency in corpus**: Rare across other documents
- **Distinctive**: Captures unique content

Example for a WWI article:
- ✅ "weltkrieg" (high score) - specific to war articles
- ✅ "offensive" (high score) - military-specific term
- ❌ "krieg" (lower score) - appears in many articles
- ❌ "der" (filtered) - appears everywhere

## Tips & Best Practices

### 1. Choose the Right Document Level

- **Page-level** (default): Best for most analyses, captures article-level topics
- **Date-level**: Good for daily trend analysis
- **Year-level**: Good for long-term topic evolution

### 2. Tune min_df and max_df

- **min_df too low**: Includes OCR errors and rare misspellings
- **min_df too high**: Misses important but infrequent terms
- **max_df too low**: Filters too many words
- **max_df too high**: Includes very common terms

Start with defaults (min_df=2, max_df=0.8) and adjust based on results.

### 3. Use Bigrams for Better Context

```bash
--ngram-range 1,2
```

Captures phrases like "erste weltkrieg" that are more meaningful than individual words.

### 4. Preprocess Historical Text

For historical German newspapers:

```bash
--preprocessing normalize,lowercase,remove-punctuation
```

The `normalize` step handles historical spelling variations.

### 5. Test with --limit First

```bash
--limit 1000
```

Quickly test parameters on a sample before processing the full dataset.

### 6. Add Domain Stopwords

```bash
--stopwords "zeitung,seite,nummer,ausgabe,herausgeber"
```

Filter newspaper-specific common words that aren't meaningful keywords.

## Performance

- **Lines.parquet**: Fast, supports page-level analysis
- **Textblocks.parquet**: Faster (fewer rows), but no page-level
- **Preprocessing**: Adds overhead but improves quality
- **Large corpora**: Use --limit for testing, then run full extraction

Typical speeds (2023 CPU):
- ~1000 documents/second (page-level, basic preprocessing)
- ~5000 documents/second (textblock-level, no preprocessing)

## Troubleshooting

### "Input file not found"

Run parsing first:
```bash
newspaper-explorer data parse --source der_tag
```

### "Grouping columns not found: page_number"

You're using `--document-level page` but input doesn't have page numbers.

Solution: Use lines.parquet (has page_number) or switch to different document level:
```bash
--document-level textblock  # or
--document-level date
```

### Very common words in results

Increase max_df or add custom stopwords:
```bash
--max-df 0.5 --stopwords "common,words,here"
```

### Too few keywords per document

Some documents might be short or have few unique words. Check:
- Minimum text length
- min_df setting (try min_df=1)
- Preprocessing (might be too aggressive)

## See Also

- [Preprocessing Documentation](../preprocessing/NORMALIZATION.md) - Text preprocessing options
- [Entity Extraction](ENTITIES.md) - Extract named entities (people, places, organizations)
- [Layout Analysis](LAYOUT.md) - Analyze page structure
- [Python API](../../PYTHON_API.md) - Using TF-IDF in Python scripts
