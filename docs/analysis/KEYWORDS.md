# Keyword Extraction

## Available Methods

1. **[TF-IDF](#tf-idf-keyword-extraction)** - Statistical distinctive vocabulary (fast, no training)
2. **[RAKE](#rake-keyword-extraction)** - Multi-word phrase extraction (linguistic patterns)
3. **[YAKE](#yake-keyword-extraction)** - Unsupervised statistical features (language-agnostic)
4. **[KeyBERT](#keybert-keyword-extraction)** - Semantic keywords via BERT embeddings

**Default Output Files:**
- TF-IDF: `tfidf_keywords.parquet` + `tfidf_keywords.json`
- RAKE: `rake_keywords.parquet` + `rake_keywords.json`
- YAKE: `yake_keywords.parquet` + `yake_keywords.json`
- KeyBERT: `keybert_keywords.parquet` + `keybert_keywords.json`

Each method uses a distinct default filename to prevent overwriting when running multiple methods on the same source. You can customize the output name with `--output-name`.

## Topic Modeling

For **topic modeling** (discovering latent themes across your corpus), see:
- **[Topic Modeling](TOPICS.md)** - Uses `analyze topics` commands

## Methodology Note

This package distinguishes between **keyword extraction** (this page) and **topic modeling** (separate):

- **Keyword Extractors** (TF-IDF, RAKE, YAKE, KeyBERT): Extract salient words/phrases representing document content
- **Topic Modeling**: Discovers latent themes across the corpus (see [TOPICS.md](TOPICS.md))

Commands reflect this distinction:
- `newspaper-explorer analyze keywords` - Extract keywords
- `newspaper-explorer analyze topics` - Topic modeling

See the [Comparative Analysis](#comparative-analysis) section for detailed methodological discussion.

---

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

## Input Data & Preprocessing

TF-IDF does **not** automatically apply preprocessing. You control the input text via `--input-file` and `--text-column`.

### Using Raw Text (Default)

By default, uses raw OCR text from the parsed data:

```bash
newspaper-explorer analyze keywords tfidf --source der_tag
# Uses: data/raw/der_tag/text/textblocks.parquet, column 'text'
```

The only built-in text processing is **stopword removal** via sklearn's TfidfVectorizer (German stopwords by default, disable with `--no-stopwords`).

### Using Preprocessed Text (Recommended)

For better results with historical newspapers, preprocess your data first:

```bash
# Step 1: Preprocess the data
newspaper-explorer data preprocess \
    --source der_tag \
    --steps lemmatize,normalize-whitespace,lowercase \
    --output-name preprocessed

# Step 2: Extract keywords from preprocessed text
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --input-file data/processed/der_tag/text/preprocessed.parquet \
    --text-column text_processed
```

**Recommended preprocessing steps for TF-IDF:**
- `normalize-whitespace` - Clean up spacing issues
- `lowercase` - Normalize case for consistent matching
- `lemmatize` - Group related word forms (e.g., "war", "waren", "gewesen" → "sein")
- `normalize-historical` - Standardize historical German orthography

For detailed preprocessing documentation, see [Preprocessing Guide](../data/preprocessing/PREPROCESSING.md).

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

### Performance

- `--num-workers`: Number of CPU workers for parallel extraction (default: auto-detect)

### Other

- `--output-name`: Output filename (default: "tfidf_keywords")
- `--limit`: Limit rows for testing

**Output**: Saves to `results/{source}/keywords/{analysis_id}/keywords.parquet` + `metadata.json`

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

### Example 5: Using Preprocessed Text

```bash
# First, preprocess the data
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-historical,lemmatize,lowercase \
    --output-name preprocessed

# Then extract keywords from preprocessed text
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --input-file data/processed/der_tag/text/preprocessed.parquet \
    --text-column text_processed \
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

# Extract keywords per page (default: uses raw text)
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
)

# Save results
output_file = extractor.save_results(yearly_keywords, output_name="yearly_keywords")
print(f"Saved to: {output_file}")
```

**Using Preprocessed Text:**

```python
from pathlib import Path
from newspaper_explorer.analyze.keywords.tf_idf import TFIDFExtractor

# Point to preprocessed data
extractor = TFIDFExtractor(
    source_name="der_tag",
    input_file=Path("data/processed/der_tag/text/preprocessed.parquet"),
    text_column="text_processed",  # Use preprocessed column
    use_stopwords=True,
)

# Extract keywords
results = extractor.extract_keywords(top_k=10)
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
- [GOOD] "weltkrieg" (high score) - specific to war articles
- [GOOD] "offensive" (high score) - military-specific term
- [BAD] "krieg" (lower score) - appears in many articles
- [BAD] "der" (filtered) - appears everywhere

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

For better results with historical German newspapers, preprocess first:

```bash
# Step 1: Preprocess
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalize-historical,lemmatize,lowercase

# Step 2: Use preprocessed text
newspaper-explorer analyze keywords tfidf \
    --source der_tag \
    --input-file data/processed/der_tag/text/preprocessed.parquet \
    --text-column text_processed
```

Historical normalization handles spelling variations (e.g., "thaeler" → "taler").

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

- **Lines.parquet**: Supports page-level analysis (more rows, slower)
- **Textblocks.parquet**: Fewer rows, faster processing
- **Preprocessed data**: Loading overhead but better quality results
- **Large corpora**: Use --limit for testing, then run full extraction
- **Parallel processing**: Use --num-workers to control CPU usage

Typical speeds (modern CPU):
- ~1000-2000 documents/second (page-level)
- ~3000-5000 documents/second (textblock-level)

## Troubleshooting

### "Input file not found"

Run parsing first:
```bash
newspaper-explorer data text parse --source der_tag
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

---

# RAKE Keyword Extraction

Extract **multi-word keyphrases** using **RAKE** (Rapid Automatic Keyword Extraction).

## What is RAKE?

RAKE is a rule-based keyphrase extraction algorithm that identifies multi-word phrases using linguistic patterns and word co-occurrence statistics. It splits text on stopwords and punctuation, then scores candidate phrases based on word frequency and co-occurrence.

**Key features:**
- **Multi-word phrases**: Extracts "erste weltkrieg" not just "weltkrieg"
- **No training required**: Rule-based, works immediately
- **Fast**: Efficient for large corpora
- **Language-agnostic**: Works with any language (stopword list needed)

**Best for**: Extracting domain terminology, technical terms, named entity-like phrases

## Quick Start

### Basic Usage

Extract keyphrases per page:

```bash
newspaper-explorer analyze keywords rake --source der_tag
```

### Extract Longer Phrases

Get keyphrases up to 5 words:

```bash
newspaper-explorer analyze keywords rake \
    --source der_tag \
    --max-length 5 \
    --top-k 15
```

### Extract by Date

Daily keyphrase trends:

```bash
newspaper-explorer analyze keywords rake \
    --source der_tag \
    --group-by date \
    --top-k 20
```

## Options

### `--top-k`
Number of keyphrases per document (default: 10)

### `--min-length` / `--max-length`
Phrase length in words:
- `--min-length 1 --max-length 1`: Single words only
- `--min-length 2 --max-length 4`: 2-4 word phrases (good for German compounds)
- `--min-length 1 --max-length 3`: Mix of single and multi-word (default)

### `--use-stopwords` / `--no-stopwords`
Use German stopwords (recommended, enabled by default)

### `--output-name`
Custom output filename (default: "rake_keywords")

**Output**: Saves to `results/{source}/keywords/rake_keywords.parquet` + `rake_keywords.json` (metadata)

## Output Format

Results saved to: `results/{source}/keywords/{output_name}.parquet`

Columns:
- Grouping columns (e.g., `newspaper_page_id`, `date`)
- `keywords`: List of extracted keyphrases
- `scores`: RAKE scores (higher = more important)

## When to Use RAKE

**Use RAKE when:**
- You need multi-word technical terms or concepts
- You want fast extraction without training
- You're analyzing domain-specific text with specialized vocabulary
- You want to capture German compound concepts ("soziale frage", "neue zeit")

**Don't use RAKE when:**
- You need semantic similarity (use KeyBERT)
- You only want single distinctive words (use TF-IDF)
- You want corpus-wide themes (use LDA)

---

# YAKE Keyword Extraction

Extract keywords using **YAKE** (Yet Another Keyword Extractor).

## What is YAKE?

YAKE is an unsupervised statistical keyword extraction method that uses multiple text features (word frequency, casing, position, context) to identify important keywords without external resources or training.

**Key features:**
- **Unsupervised**: No training, no external corpora needed
- **Statistical**: Uses frequency, casing, position, left/right context
- **Multi-lingual**: Specify language for better results
- **N-grams**: Extracts single words, bigrams, trigrams

**Best for**: General-purpose keyword extraction with balanced statistical approach

## Quick Start

### Basic Usage

Extract keywords per page:

```bash
newspaper-explorer analyze keywords yake --source der_tag
```

### Extract Only Single Words

Focus on unigrams:

```bash
newspaper-explorer analyze keywords yake \
    --source der_tag \
    --max-ngram-size 1 \
    --top-k 15
```

### Strict Deduplication

Reduce redundancy:

```bash
newspaper-explorer analyze keywords yake \
    --source der_tag \
    --deduplication-threshold 0.7
```

## Options

### `--top-k`
Number of keywords per document (default: 10)

### `--language`
Language code for better results:
- `de`: German (default)
- `en`: English
- See YAKE docs for more languages

### `--max-ngram-size`
Maximum n-gram size:
- `1`: Single words only
- `2`: Words + bigrams
- `3`: Words + bigrams + trigrams (default)

### `--deduplication-threshold`
Similarity threshold for removing redundant keywords (0-1):
- `0.9`: Keep similar keywords (default)
- `0.7`: Moderate deduplication
- `0.5`: Strict deduplication

### `--output-name`
Custom output filename (default: "yake_keywords")

**Output**: Saves to `results/{source}/keywords/yake_keywords.parquet` + `yake_keywords.json` (metadata)

## Output Format

Results saved to: `results/{source}/keywords/{output_name}.parquet`

Columns:
- Grouping columns (e.g., `newspaper_page_id`, `date`)
- `keywords`: List of extracted keywords
- `scores`: YAKE scores (**lower = more important**, inverted from others)

[WARNING] **Note**: YAKE scores are inverted - lower scores indicate more important keywords!

## When to Use YAKE

**Use YAKE when:**
- You want balanced statistical extraction
- You need language-aware processing
- You want mix of single and multi-word keywords
- You want unsupervised extraction without dependencies

**Don't use YAKE when:**
- You need semantic understanding (use KeyBERT)
- You only want distinctive vocabulary (use TF-IDF)
- You specifically need long multi-word phrases (use RAKE)

---

# KeyBERT Keyword Extraction

Extract **semantic keywords** using **KeyBERT** (BERT-based extraction).

## What is KeyBERT?

KeyBERT uses BERT embeddings to extract keywords that are semantically similar to the document. It computes document and candidate keyword embeddings, then selects keywords with highest cosine similarity to the document embedding.

**Key features:**
- **Semantic understanding**: Captures meaning, not just statistics
- **Multilingual**: Pre-trained models work across languages
- **Diversity control**: MMR (Maximal Marginal Relevance) prevents redundancy
- **Pre-trained**: No training needed, uses transformer models

**Best for**: Semantic keyword extraction, capturing concepts and meaning

## Quick Start

### Basic Usage

Extract semantic keywords per page:

```bash
newspaper-explorer analyze keywords keybert --source der_tag
```

### Use Larger Model

Better accuracy with larger multilingual model:

```bash
newspaper-explorer analyze keywords keybert \
    --source der_tag \
    --model distiluse-base-multilingual-cased-v2
```

### High Diversity Keywords

Avoid redundant similar keywords:

```bash
newspaper-explorer analyze keywords keybert \
    --source der_tag \
    --diversity 0.9 \
    --top-k 15
```

### Extract Only Single Words

Focus on single-word concepts:

```bash
newspaper-explorer analyze keywords keybert \
    --source der_tag \
    --min-ngram 1 \
    --max-ngram 1
```

## Options

### `--top-k`
Number of keywords per document (default: 10)

### `--model`
Sentence transformer model:
- `paraphrase-multilingual-MiniLM-L12-v2`: Default, good balance for German
- `distiluse-base-multilingual-cased-v2`: Larger, more accurate
- `all-MiniLM-L6-v2`: English only, faster

### `--min-ngram` / `--max-ngram`
N-gram range for candidate phrases:
- `--min-ngram 1 --max-ngram 1`: Single words only
- `--min-ngram 1 --max-ngram 2`: Words + bigrams (default)
- `--min-ngram 1 --max-ngram 3`: Words + bigrams + trigrams

### `--diversity`
Diversity parameter for MMR (0-1):
- `0.0`: No diversity (most similar to document)
- `0.5`: Balanced (default)
- `1.0`: Maximum diversity (varied keywords)

### `--use-mmr` / `--no-mmr`
Enable/disable Maximal Marginal Relevance for diversity (enabled by default)

### `--output-name`
Custom output filename (default: "keybert_keywords")

**Output**: Saves to `results/{source}/keywords/keybert_keywords.parquet` + `keybert_keywords.json` (metadata)

## Output Format

Results saved to: `results/{source}/keywords/{output_name}.parquet`

Columns:
- Grouping columns (e.g., `newspaper_page_id`, `date`)
- `keywords`: List of extracted keywords
- `scores`: Cosine similarity scores (0-1, higher = more semantically relevant)

## When to Use KeyBERT

**Use KeyBERT when:**
- You want semantic understanding of document content
- You need context-aware keyword extraction
- You want to capture conceptual themes, not just frequent words
- You're working with German text (multilingual models)

**Don't use KeyBERT when:**
- You need fast extraction (slower due to BERT inference)
- You want purely statistical distinctive vocabulary (use TF-IDF)
- You need corpus-wide topic discovery (use LDA)
- You have limited computational resources (use RAKE or YAKE)

## Model Notes

**First run**: Downloads the selected model (200-500MB). Subsequent runs reuse cached model.

**GPU acceleration**: Automatically uses GPU if available (much faster).

**Memory**: Requires more memory than statistical methods. For large corpora, consider batch processing or use smaller model.

---

# Topic Modeling with LDA

**This section has moved!** 

LDA is topic modeling, not keyword extraction. For complete LDA documentation, see:

**[→ Topic Modeling Documentation (TOPICS.md)](TOPICS.md)**

Use the separate topic modeling commands:
```bash
newspaper-explorer analyze topics lda --source der_tag --mode train --num-topics 20
newspaper-explorer analyze topics lda --source der_tag --mode topics
newspaper-explorer analyze topics lda --source der_tag --mode documents
```

---

# Comparative Analysis

## Method Comparison Table

| Method | Type | Speed | Training | Semantic | Multi-word | Best For |
|--------|------|-------|----------|----------|-----------|----------|
| **TF-IDF** | Statistical | Very Fast | None | No | No | Distinctive vocabulary |
| **RAKE** | Rule-based | Fast | None | No | Yes | Domain terminology, phrases |
| **YAKE** | Statistical | Fast | None | Limited | Yes | Balanced extraction |
| **KeyBERT** | Embedding | Moderate | Pre-trained | Yes | Yes | Semantic concepts |
| **LDA** | Topic Model | Slow | Required | Themes | No | Topic discovery |

## Methodological Framework

This package follows a **methodologically rigorous** distinction between different approaches:

### Keyword Extraction vs Topic Modeling

**Keyword Extraction** (TF-IDF, RAKE, YAKE, KeyBERT):
- **Goal**: Identify salient words/phrases representing document content
- **Scope**: Document-level or aggregated groups
- **Output**: Lists of keywords with relevance scores
- **Use in DH**: Content analysis, text mining, information retrieval

**Topic Modeling** (LDA):
- **Goal**: Discover latent themes across corpus
- **Scope**: Corpus-level with document distributions
- **Output**: Topics (term distributions) + document-topic mixtures
- **Use in DH**: Thematic analysis, discourse analysis, corpus exploration

### Terminology Precision

We use **precise terminology** to avoid methodological confusion:

- **"Keywords"**: Words/phrases extracted by TF-IDF, RAKE, YAKE, KeyBERT
- **"Topic terms"**: Representative vocabulary from LDA topic distributions
- **"Keyphrases"**: Multi-word expressions (RAKE, YAKE, KeyBERT)

While LDA's topic terms *can* be used like keywords, they are fundamentally different: they represent topic themes, not document salience.

## Selection Guide

### By Research Question

**"What makes this document unique?"**
→ Use **TF-IDF** - finds distinctive vocabulary

**"What technical terms appear?"**
→ Use **RAKE** - extracts domain-specific phrases

**"What are the key concepts here?"**
→ Use **KeyBERT** - captures semantic meaning

**"What are the balanced important keywords?"**
→ Use **YAKE** - statistical feature-based extraction

**"What themes exist across the corpus?"**
→ Use **Topic Modeling** - discovers latent topics (see [TOPICS.md](TOPICS.md))

### By Text Characteristics

**Short documents (< 100 words)**:
- [GOOD] TF-IDF (document-level distinctive terms)
- [GOOD] RAKE (phrase extraction)
- [WARNING] YAKE (may struggle with limited context)
- [BAD] KeyBERT (needs sufficient semantic context)

**Long documents (> 1000 words)**:
- [GOOD] All methods work well
- TF-IDF, RAKE, YAKE for speed
- [GOOD] KeyBERT for semantic depth

**Domain-specific vocabulary**:
- [GREAT] RAKE - extracts technical multi-word terms
- [GOOD] TF-IDF - finds distinctive domain terms
- [GOOD] KeyBERT - understands domain concepts (with domain-tuned model)
- [WARNING] YAKE - good but less specialized

**Multilingual/Historical German**:
- [GOOD] KeyBERT with multilingual models
- [GOOD] TF-IDF (language-agnostic)
- [GOOD] RAKE with German stopwords
- [GOOD] YAKE with language="de"

### By Computational Resources

**Limited resources (CPU only, < 8GB RAM)**:
- [GREAT] TF-IDF - very efficient
- [GREAT] RAKE - fast rule-based
- [GOOD] YAKE - efficient statistical
- [BAD] KeyBERT - requires more memory, slow on CPU

**Good resources (GPU, > 16GB RAM)**:
- All methods work well
- KeyBERT benefits significantly from GPU

### By Analysis Type

**Exploratory analysis**:
1. Start with **TF-IDF** (fast overview)
2. Check **RAKE** (see phrase patterns)
3. Try **Topic Modeling** (discover themes - see [TOPICS.md](TOPICS.md))

**Semantic analysis**:
1. Use **KeyBERT** (semantic understanding)
2. Supplement with **YAKE** (balanced keywords)

**Information retrieval**:
1. Use **TF-IDF** (distinctive terms)
2. Add **RAKE** (multi-word queries)

**Discourse analysis**:
1. Use **Topic Modeling** (thematic structure - see [TOPICS.md](TOPICS.md))
2. Supplement with **TF-IDF** (distinctive vocabulary per theme)

## Combining Methods

Different methods complement each other. Consider using **multiple approaches**:

### Example: Comprehensive Document Analysis

```bash
# 1. Distinctive vocabulary
newspaper-explorer analyze keywords tfidf --source der_tag --top-k 15

# 2. Multi-word concepts
newspaper-explorer analyze keywords rake --source der_tag --top-k 10

# 3. Semantic keywords
newspaper-explorer analyze keywords keybert --source der_tag --top-k 10
```

Then **compare results**:
- **Overlap**: Terms appearing in multiple methods are highly salient
- **TF-IDF only**: Distinctive but perhaps not conceptually deep
- **KeyBERT only**: Semantically relevant but perhaps not statistically distinctive
- **RAKE only**: Domain-specific terminology

For **topic structure** across documents, see [Topic Modeling](TOPICS.md).

### Example: Historical Newspaper Analysis

For 1900-1920 German newspapers:

1. **TF-IDF by year** → Track distinctive vocabulary evolution
2. **RAKE by year** → Identify emerging multi-word concepts
3. **Topic Modeling** → Discover major discourse themes (see [TOPICS.md](TOPICS.md))
4. **KeyBERT per significant event** → Deep semantic analysis of key moments

## Digital Humanities Considerations

### Methodological Transparency

When publishing DH research:
- **Specify method**: Don't just say "keyword extraction"
- **Justify choice**: Explain why TF-IDF vs KeyBERT vs LDA
- **Report parameters**: Document all settings (top_k, ngram ranges, etc.)
- **Discuss limitations**: Each method has blind spots

### Validation Strategies

1. **Manual verification**: Sample and check extracted keywords
2. **Cross-method validation**: Compare results across methods
3. **Historical accuracy**: Check against known historical events/terminology
4. **Expert evaluation**: Domain experts assess relevance

### Reproducibility

All commands in this documentation are **fully reproducible**:
- Documented parameters
- Version-controlled source configurations
- Transparent preprocessing

## See Also

- [Preprocessing Documentation](../preprocessing/NORMALIZATION.md) - Text preprocessing options
- [Entity Extraction](ENTITIES.md) - Extract named entities (people, places, organizations)
- [Layout Analysis](LAYOUT.md) - Analyze page structure
- [Python API](../../PYTHON_API.md) - Using keyword extraction in Python scripts
