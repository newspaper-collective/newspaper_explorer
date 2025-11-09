# Topic Modeling Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [CLI Commands](#cli-commands)
5. [Python API](#python-api)
6. [Method Comparison](#method-comparison)
7. [Performance & Configuration](#performance--configuration)
8. [Data Schemas](#data-schemas)
9. [Complete Workflows](#complete-workflows)
10. [Analyzing Results](#analyzing-results)
11. [Integration with Other Analyses](#integration-with-other-analyses)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The topic modeling module provides two methods for discovering themes and topics in newspaper texts:

**Methods**:
1. **LDA** - Latent Dirichlet Allocation (classic probabilistic topic modeling)
2. **BERTopic** - BERT-based semantic topic modeling with **global topics** approach

**Key Features**:
- ✅ **Two modeling approaches**: LDA (traditional) and BERTopic (semantic with global topics)
- ✅ **Global topics architecture**: Single fit on entire corpus for unified topic space
- ✅ **Temporal analysis**: Track topics over time with consistent topic definitions
- ✅ **4-phase optimized pipeline**: Parallel chunking + Multi-GPU embedding + Global clustering
- ✅ **Flexible grouping**: Analyze topics at page, issue, date, or custom levels
- ✅ **Sentence chunking**: BERTopic intelligently chunks text using spaCy
- ✅ **Stopword removal**: German stopword filtering via spaCy
- ✅ **Model caching**: Automatic model downloads and caching
- ✅ **Resume functionality**: Skip already-processed documents
- ✅ **Metadata tracking**: Full provenance with timing and parameters

---

## Global Topics vs Per-Document Topics

### What Are Global Topics?

**Global Topics** means discovering a **single set of topics for the entire corpus**, then assigning documents to these unified topics. This contrasts with the **per-document** approach where each document gets its own separate topic model.

**Global Topics Approach** (BERTopic - Current):
```
Entire Corpus (134K docs)
    ↓
Chunk all text (3M chunks)
    ↓
Embed all chunks (GPU batch)
    ↓
Single BERTopic.fit() → Discover K global topics
    ↓
Assign each document to global topics
    ↓
Result: All documents use same K topics
```

**Per-Document Approach** (Old, Abandoned):
```
Document 1 → BERTopic.fit() → Topics for doc 1
Document 2 → BERTopic.fit() → Topics for doc 2 (different!)
...
Document 134K → BERTopic.fit() → Topics for doc 134K (different!)
```

### Why Global Topics?

**Advantages**:
1. **Temporal Analysis**: Can track "China" topic across all dates because it's the same topic
2. **Performance**: One fit instead of 134K fits
3. **Coherence**: Topics discovered from more data are more robust
4. **Comparability**: Documents are directly comparable via shared topic space
5. **Statistical Power**: More chunks per topic = better topic quality

**Example - Temporal Analysis**:
```python
# With global topics, you can do:
topics_df.group_by("date").agg([
    pl.col("topic_terms").filter(pl.col("topics") == 5)  # Track topic 5 over time
])

# Topic 5 might be "kaiſer, china, peking" and you can see when it appears
# across all dates because it's the SAME topic everywhere
```

**Trade-offs**:
- ❌ Less document-specific detail (but you can still see which topics dominate each doc)
- ✅ Much better for corpus-wide analysis and temporal trends
- ✅ Essential for multi-document comparisons

---

## Architecture

### File Structure

```
src/newspaper_explorer/
├── analyze/
│   └── topics/
│       ├── lda.py                   # LDA topic modeling
│       ├── bertopic.py              # BERTopic semantic modeling
│       └── schemas.py               # TopicRecord schema
├── cli/
│   └── analyze/
│       ├── commands.py              # Register topics group
│       └── topics/
│           └── commands.py          # CLI commands (lda, bertopic)
├── data/
│   └── utils/
│       └── text.py                  # Text chunking utilities

results/{source}/topics/             # Output directory
├── lda_topics.parquet               # LDA results
├── lda_topics.json                  # LDA metadata
├── bertopic_topics.parquet          # BERTopic results
└── bertopic_topics.json             # BERTopic metadata

models/sentence_transformers/        # Cached embedding models
└── models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
```

### Integration with Data Pipeline

```
Download → Parse ALTO/METS → Polars DataFrame → Preprocessing (optional)
                                                        ↓
                                              Topic Modeling
                                              (LDA / BERTopic)
                                                        ↓
                                     results/{source}/topics/
                                                        ↓
                                        Analysis & Visualization
```

**Integration Points**:
- Uses existing Polars DataFrames from `DataLoader`
- Works with line-level or text-block-level data
- Supports normalized/preprocessed text
- Outputs to standard `results/{source}/topics/` directory
- Queryable via `QueryEngine` with foreign key relationships

---

## Core Components

### 1. LDATopicExtractor

**Location**: `src/newspaper_explorer/analyze/topics/lda.py`

**Purpose**: Traditional probabilistic topic modeling using Latent Dirichlet Allocation.

**Key Features**:
- Bag-of-words approach with TF-IDF weighting
- Configurable number of topics
- Per-document topic distributions
- Term importance scores

**Basic Usage**:
```python
from newspaper_explorer.analyze.topics.lda import LDATopicExtractor

extractor = LDATopicExtractor(
    source_name="der_tag",
    n_topics=10,
    max_features=1000,
)

# Extract topics
results_df = extractor.extract_topics(
    group_by=["date"],
    limit=1000,
)

# Save results
extractor.save_results(results_df, output_name="lda_topics")
```

### 2. BERTopicExtractor

**Location**: `src/newspaper_explorer/analyze/topics/bertopic.py`

**Purpose**: Modern semantic topic modeling using BERT embeddings and clustering with **global topics** approach.

**Key Features**:
- ✅ **Global topics discovery**: Single fit on entire corpus (not per-document)
- ✅ **4-phase workflow**: Optimized pipeline with parallel processing
- ✅ **Multiprocessing**: Parallel chunking across all CPU cores
- ✅ **Multi-GPU support**: Batch embedding with automatic GPU utilization
- ✅ **Semantic embeddings**: Via sentence transformers (multilingual support)
- ✅ **Sentence-level chunking**: Intelligent text splitting with spaCy
- ✅ **German stopword filtering**: Optional stopword removal
- ✅ **Model caching**: Automatic model download and caching
- ✅ **Temporal analysis**: Unified topic space for tracking topics over time

**Architecture - Global Topics Approach**:

Unlike traditional per-document topic modeling, BERTopicExtractor uses a **global topics** architecture:

1. **Single corpus-wide fit**: One BERTopic model is fit on ALL text chunks from the entire corpus
2. **Unified topic space**: All documents share the same topics, enabling temporal analysis
3. **Document assignment**: Each document is assigned probabilities for global topics based on its chunks
4. **Scalability**: Can process 100K+ documents efficiently with 4-phase pipeline

**4-Phase Workflow**:

```
Phase 1: Chunking (CPU, parallel)
├─ Split all documents into sentence chunks
├─ Parallel processing across CPU cores
└─ Output: ~3M text chunks

Phase 2: Embedding (GPU, batch)
├─ Generate embeddings for all chunks
├─ Batch processing (configurable batch_size)
├─ Multi-GPU automatic support
└─ Output: Embeddings matrix

Phase 3: Global Fit (CPU, single)
├─ Single BERTopic.fit_transform() on entire corpus
├─ UMAP dimensionality reduction
├─ HDBSCAN clustering
├─ c-TF-IDF term extraction
└─ Output: Global topic model with K topics

Phase 4: Assignment (CPU, fast)
├─ Map documents to global topics
├─ Count chunk-to-topic frequencies per doc
├─ Calculate topic probabilities
└─ Output: Document-topic assignments
```

**Basic Usage**:
```python
from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

extractor = BERTopicExtractor(
    source_name="der_tag",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    min_text_length=100,
    chunk_sentences=5,
    use_stopwords=True,
)

# Extract global topics from entire corpus
results_df = extractor.extract_topics(
    group_by=["page_id"],
    min_cluster_size=15,    # Minimum chunks per global topic
    n_topics=None,          # Auto-discover optimal number
    batch_size=128,         # GPU batch size (higher = faster)
    top_k=5,
    limit=None,             # Process all documents
)

# Save results
extractor.save_results(results_df, output_name="bertopic_topics")
```

### 3. TopicRecord Schema

**Location**: `src/newspaper_explorer/analyze/topics/schemas.py`

**Purpose**: Pydantic model for topic modeling results with metadata.

**Fields**:
```python
class TopicRecord(BaseModel):
    doc_id: str                    # Document identifier
    source_id: Optional[str]       # Foreign key to source
    issue_id: Optional[str]        # Foreign key to issue
    page_id: Optional[str]         # Foreign key to page
    text_block_id: Optional[str]   # Foreign key to text block
    topic_terms: List[str]         # Top terms for topics
    scores: List[float]            # Term importance scores
    topics: List[int]              # Topic IDs
    topic_probs: List[float]       # Topic probabilities
```

---

## CLI Commands

### Topic Modeling Commands

All commands are under the `newspaper-explorer analyze topics` group:

```bash
newspaper-explorer analyze topics --help
```

---

### 1. LDA Topic Modeling

**Command**: `newspaper-explorer analyze topics lda`

**Description**: Discover topics using Latent Dirichlet Allocation (LDA).

**Basic Usage**:
```bash
# Analyze topics by date
newspaper-explorer analyze topics lda \
  --source der_tag \
  --group-by date \
  --n-topics 10

# Analyze topics by issue with custom output name
newspaper-explorer analyze topics lda \
  --source der_tag \
  --group-by issue_id \
  --n-topics 20 \
  --output-name my_lda_analysis

# Test on limited data
newspaper-explorer analyze topics lda \
  --source der_tag \
  --group-by year,month \
  --n-topics 15 \
  --limit 5000
```

**Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | TEXT | Required | Source name (e.g., "der_tag") |
| `--input-file` | PATH | Auto | Custom input parquet file |
| `--text-column` | TEXT | "text" | Column containing text |
| `--group-by` | TEXT | None | Grouping columns (comma-separated) |
| `--n-topics` | INT | 10 | Number of topics to discover |
| `--max-features` | INT | 1000 | Maximum vocabulary size |
| `--top-k` | INT | 5 | Top K terms per topic |
| `--output-name` | TEXT | "lda_topics" | Output filename (no extension) |
| `--limit` | INT | None | Limit rows for testing |

**Grouping Options**:
- `--group-by page_id` - Topics per page
- `--group-by issue_id` - Topics per issue
- `--group-by date` - Topics per date
- `--group-by year,month` - Topics per month
- No grouping - Topics per text line (not recommended)

**Output**:
```
results/der_tag/topics/
├── lda_topics.parquet      # Topic assignments
└── lda_topics.json         # Metadata (parameters, timing, stats)
```

---

### 2. BERTopic Semantic Modeling

**Command**: `newspaper-explorer analyze topics bertopic`

**Description**: Discover topics using BERTopic with global topics approach (BERT-based semantic clustering on entire corpus).

**Basic Usage**:
```bash
# Analyze topics by page (recommended)
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --min-cluster-size 15 \
  --batch-size 128

# Analyze topics by date for temporal analysis
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by date \
  --min-cluster-size 10 \
  --n-topics 50 \
  --batch-size 128

# Full corpus with all parameters
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --min-cluster-size 15 \
  --n-topics None \
  --batch-size 128 \
  --chunk-sentences 5 \
  --output-name full_corpus_topics

# Disable stopwords for specialized vocabulary
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by issue_id \
  --min-cluster-size 10 \
  --no-stopwords
```

**Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | TEXT | Required | Source name (e.g., "der_tag") |
| `--input-file` | PATH | Auto | Custom input parquet file |
| `--text-column` | TEXT | "text" | Column containing text |
| `--group-by` | TEXT | None | Grouping columns (comma-separated) |
| `--min-cluster-size` | INT | 10 | Minimum chunks per **global topic** |
| `--n-topics` | INT | None | Target number of global topics (None = auto-discover) |
| `--batch-size` | INT | 32 | GPU batch size for embedding (higher = faster) |
| `--num-gpus` | INT | 1 | Number of GPUs (1=single, >1=multi-GPU parallel) |
| `--top-k` | INT | 5 | Top K terms per topic |
| `--embedding-model` | TEXT | See below | Sentence transformer model |
| `--min-text-length` | INT | 100 | Minimum text length (chars) |
| `--chunk-sentences` | INT | 5 | Sentences per chunk |
| `--no-stopwords` | FLAG | False | Disable stopword filtering |
| `--output-name` | TEXT | "bertopic_topics" | Output filename |
| `--limit` | INT | None | Limit rows for testing |

**Default Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`

**Grouping Options**:
- `--group-by page_id` - Topics per page (recommended for BERTopic)
- `--group-by issue_id` - Topics per issue
- `--group-by date` - Topics per date
- `--group-by year,month` - Topics per month

**Important Notes**:
- **Document concept**: When grouping (e.g., by `page_id`), each group becomes one "document"
  - Example: `--group-by page_id --limit 10000` → takes 10,000 lines → groups into ~39 pages → 39 documents
- **Chunking**: Each document is split into sentence chunks → each chunk is embedded → BERTopic clusters chunks
- **Success rate**: ~75-80% of documents typically succeed (others fail due to insufficient text)
- **Min topic size**: `--min-topic-size 3` means need at least 3 chunks to form a topic

**Output**:
```
results/der_tag/topics/
├── bertopic_topics.parquet # Topic assignments
└── bertopic_topics.json    # Metadata (parameters, timing, stats)
```

**Performance Tips**:
- Start with `--limit 10000` for testing
- Use `--group-by page_id` for meaningful document-level analysis
- Increase `--min-topic-size` if getting too many small topics
- Decrease `--chunk-sentences` for more granular topic detection
- First run downloads models (~420MB), subsequent runs use cache

---

## Python API

### LDA Example

```python
from newspaper_explorer.analyze.topics.lda import LDATopicExtractor

# Initialize extractor
extractor = LDATopicExtractor(
    source_name="der_tag",
    n_topics=15,
    max_features=1000,
)

# Extract topics grouped by date
results_df = extractor.extract_topics(
    group_by=["date"],
    top_k=10,
    limit=None,  # Process all data
)

# Save results
output_file = extractor.save_results(
    results_df,
    output_name="lda_by_date",
    top_k=10,
)

print(f"Results saved to: {output_file}")
print(f"Processed {len(results_df)} documents")
```

### BERTopic Example

```python
from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

# Initialize extractor with custom settings
extractor = BERTopicExtractor(
    source_name="der_tag",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    min_text_length=200,      # Skip short texts
    chunk_sentences=5,        # 5 sentences per chunk
    use_stopwords=True,       # Filter German stopwords
)

# Extract topics grouped by page
results_df = extractor.extract_topics(
    group_by=["page_id"],
    min_topic_size=3,         # Min 3 docs per topic
    nr_topics=None,           # Auto-discover
    top_k=5,                  # Top 5 terms
    limit=10000,              # Test on 10k lines
)

# Save results
output_file = extractor.save_results(
    results_df,
    output_name="bertopic_pages",
    top_k=5,
)

print(f"Results saved to: {output_file}")
print(f"Processed {len(results_df)} documents")
print(f"Success rate: {len(results_df) / extractor._last_input_df.shape[0] * 100:.1f}%")
```

### Custom Input Data

```python
import polars as pl
from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

# Load custom preprocessed data
df = pl.read_parquet("my_preprocessed_data.parquet")

# Initialize with custom input
extractor = BERTopicExtractor(
    source_name="der_tag",
    input_file="my_preprocessed_data.parquet",
)

# Extract topics
results_df = extractor.extract_topics(
    group_by=["issue_id"],
    min_topic_size=5,
)

# Save
extractor.save_results(results_df, output_name="custom_analysis")
```

---

## Method Comparison

### LDA vs BERTopic

| Feature | LDA | BERTopic (Global Topics) |
|---------|-----|--------------------------|
| **Approach** | Probabilistic bag-of-words | Semantic embeddings + clustering |
| **Topic Discovery** | Per-document distributions | Global topics, then assignment |
| **Scalability** | Millions of docs | 100K+ docs efficiently |
| **Quality** | Good for traditional analysis | Better semantic coherence |
| **Interpretability** | Topic distributions | Clear global topic clusters |
| **Language Understanding** | Bag-of-words only | Understands semantics |
| **Temporal Analysis** | Limited (different topic spaces) | Excellent (unified topic space) |
| **Hardware** | CPU only | CPU + Multi-GPU |
| **Setup** | No models needed | Downloads embedding models (~420MB) |
| **Best For** | Quick analysis, large datasets | High-quality semantic analysis, temporal tracking |

### When to Use Each Method

**Use LDA when**:
- Speed is absolute priority
- Working with very large datasets (millions of documents)
- Want traditional probabilistic topic modeling
- Don't need semantic understanding
- Running on CPU-only infrastructure

**Use BERTopic when**:
- Quality and semantic coherence matter
- Working with 10K-500K documents
- Want interpretable global topics
- Need temporal analysis (tracking topics over time)
- Have GPU access (recommended but not required)

### Architecture Comparison

**LDA - Per-Document Approach**:
```
Document 1 → Bag-of-words → Topic distribution [0.3, 0.5, 0.2, ...]
Document 2 → Bag-of-words → Topic distribution [0.1, 0.7, 0.2, ...]
...
(Each document treated independently)
```

**BERTopic - Global Topics Approach**:
```
All Documents → Chunk → Embed → Global Clustering
                                        ↓
                                Global Topics [T1, T2, ..., TK]
                                        ↓
Document 1 → Assign to global topics → [T5: 0.6, T12: 0.3, T3: 0.1]
Document 2 → Assign to global topics → [T12: 0.8, T5: 0.2]
...
(Unified topic space enables temporal analysis)
```

### Typical Use Cases

**LDA**:
- Initial exploration of massive corpora (millions of documents)
- Quick topic distribution analysis
- Traditional topic modeling research
- When GPU not available

**BERTopic**:
- Semantic clustering of newspaper pages
- Temporal topic tracking (how topics evolve over time)
- Finding thematic coherence within issues/dates
- Quality-focused analysis with interpretable topics
- Cross-temporal comparisons (all documents share topic space)

---


### Memory Requirements

**LDA**:
- ~100MB base memory
- +1MB per 1,000 documents (sparse matrices)
- Vocabulary-dependent

**BERTopic (Global Topics)**:
- ~500MB base (embedding model)
- ~10MB per 1,000 chunks (embeddings in memory during Phase 2)
- ~200MB for global topic model (Phase 3)
- Peak memory: ~5-10GB for 134K documents (depends on chunking)
- GPU memory: ~4GB recommended (works with less via batch processing)

### Optimization Tips

**For LDA**:
```python
# Reduce vocabulary size for speed
extractor = LDATopicExtractor(
    max_features=500,  # Smaller vocabulary
    n_topics=10,       # Fewer topics
)
```

**For BERTopic (Global Topics)**:
```python
# Optimize for speed (large corpus)
extractor = BERTopicExtractor(
    source_name="der_tag",
    chunk_sentences=5,      # Balance between detail and chunk count
    min_text_length=100,    # Skip very short texts
    use_stopwords=True,     # Filter noise early
)

# Run with optimized parameters
results = extractor.extract_topics(
    group_by=["page_id"],
    min_cluster_size=15,    # Higher = fewer but more robust topics
    batch_size=128,         # Higher = faster GPU processing (if memory allows)
    n_topics=None,          # Let algorithm find optimal number
)
```

**Multi-GPU Configuration**:
```bash
# Enable multi-GPU with --num-gpus flag:
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --num-gpus 4 \
  --batch-size 256  # Higher batch size for better GPU utilization

# Monitor GPU usage during Phase 2:
watch -n 0.5 nvidia-smi

# All available GPUs (cuda:0, cuda:1, cuda:2, cuda:3) will be used in parallel
# for embedding generation, significantly improving speed and GPU utilization
```

**CPU Optimization (Phase 1)**:
```python
# Multiprocessing uses cpu_count() - 1 by default
# Phase 1 will max out all CPU cores during chunking
# No configuration needed - automatic parallel processing
```

### Model Caching

**BERTopic models are cached**:
- Location: `models/sentence_transformers/`
- First run: Downloads model (~420MB)
- Subsequent runs: Loads from cache (fast)
- Environment variable: `SENTENCE_TRANSFORMERS_HOME` is set automatically

**To use different cache location**:
```bash
export SENTENCE_TRANSFORMERS_HOME=/path/to/cache
newspaper-explorer analyze topics bertopic --source der_tag ...
```

---

## Data Schemas

### Output Schema

Both methods produce the same output schema:

```python
{
    "doc_id": str,              # Document identifier (from grouping)
    "topic_terms": List[str],   # Top terms for assigned topics
    "scores": List[float],      # Importance scores for terms
    "topics": List[int],        # Topic IDs assigned
    "topic_probs": List[float], # Probability of each topic
    "source_id": str,           # Foreign key: source
    "issue_id": str,            # Foreign key: issue
    "page_id": str,             # Foreign key: page
    "text_block_id": str,       # Foreign key: text block (if available)
}
```

### Metadata Schema

JSON metadata file contains:

```python
{
    "analysis_type": "topics",
    "method_type": "lda" | "bertopic",
    "model_name": str,
    "source": str,
    "parameters": {
        "group_by": List[str],
        "min_topic_size": int,     # BERTopic only
        "n_topics": int,           # LDA only
        "top_k": int,
        "limit": int | None,
        ...
    },
    "input_data": {
        "num_documents": int,
    },
    "output_data": {
        "unique_topics_assigned": int,
        "total_topic_assignments": int,
    },
    "duration_seconds": float,
    "status": "completed",
    "created_at": str,
}
```

### Foreign Keys

Results include foreign keys for joining with other data:

```python
# Join topics with original text
import polars as pl

topics = pl.read_parquet("results/der_tag/topics/bertopic_topics.parquet")
text = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Join on page_id
joined = topics.join(text, on="page_id", how="left")

# Or join on issue_id
joined = topics.join(text, on="issue_id", how="left")
```

---

## Complete Workflows

### Workflow 1: Quick Topic Exploration

**Goal**: Quickly discover main themes in a newspaper corpus.

```bash
# 1. LDA for fast overview
newspaper-explorer analyze topics lda \
  --source der_tag \
  --group-by date \
  --n-topics 20 \
  --limit 50000

# 2. Examine results
# Load and analyze with Polars or query engine
```

### Workflow 2: Page-Level Semantic Analysis

**Goal**: Understand topics discussed on individual newspaper pages.

```bash
# 1. Extract global topics per page with BERTopic
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --min-cluster-size 15 \
  --batch-size 128

# 2. Results contain page_id foreign key
# Can join with page metadata, layout analysis, etc.
```

### Workflow 3: Temporal Topic Analysis with Global Topics

**Goal**: Track how topics evolve over time using BERTopic's global topics.

**Why BERTopic for temporal analysis?**
- Global topics = same topic definitions across all time periods
- Can track specific topics (e.g., "war", "economy") over months/years
- Enables meaningful temporal comparisons

```bash
# 1. Extract global topics grouped by date
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by date \
  --min-cluster-size 15 \
  --batch-size 128 \
  --output-name temporal_topics

# 2. Analyze temporal patterns with Polars
```

**Analysis Script**:
```python
import polars as pl
import matplotlib.pyplot as plt

# Load results
df = pl.read_parquet("results/der_tag/topics/temporal_topics.parquet")

# Track specific topic over time (e.g., topic 12 about "war")
topic_12 = df.filter(pl.col("topics").list.contains(12))
topic_12_over_time = (
    topic_12
    .group_by("date")
    .agg([
        pl.col("page_id").count().alias("mentions"),
        pl.col("topic_probs").list.get(0).mean().alias("avg_probability"),
    ])
    .sort("date")
)

# Plot temporal trend
plt.figure(figsize=(12, 6))
plt.plot(topic_12_over_time["date"], topic_12_over_time["mentions"])
plt.title("Topic 12 mentions over time")
plt.xlabel("Date")
plt.ylabel("Number of pages")
plt.savefig("topic_12_temporal.png")

# Find all topics per month and track evolution
monthly_topics = (
    df
    .with_columns([
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    ])
    .group_by("month")
    .agg([
        pl.col("topics").explode().value_counts().alias("topic_counts"),
        pl.col("page_id").count().alias("total_pages")
    ])
    .sort("month")
)

print("Topics by month:")
print(monthly_topics)
```

### Workflow 4: Full Corpus Global Topics

**Goal**: Discover global topics from entire corpus for maximum coverage.

```bash
# 1. Run BERTopic on full corpus (no limit)
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --min-cluster-size 15 \
  --batch-size 128 \
  --output-name full_corpus_topics

# Processing time: ~25 minutes for 134K documents
# Result: K global topics discovered from all documents
```

**Why full corpus?**
- More data = better topic quality
- Global topics represent entire newspaper's thematic landscape
- Can then analyze subsets (by date, year, etc.) using these topics
- Enables corpus-wide comparisons and temporal analysis

**Post-processing for temporal analysis**:
```python
import polars as pl

# Load full corpus results (page-level)
df = pl.read_parquet("results/der_tag/topics/full_corpus_topics.parquet")

# Load original data to get dates
lines = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Join to add temporal information
df_temporal = df.join(
    lines.select(["page_id", "date"]).unique(),
    on="page_id",
    how="left"
)

# Now analyze topics by year
yearly_topics = (
    df_temporal
    .with_columns([
        pl.col("date").dt.year().alias("year")
    ])
    .group_by(["year", pl.col("topics").explode()])
    .agg([
        pl.col("page_id").count().alias("frequency"),
        pl.col("topic_probs").explode().mean().alias("avg_prob"),
    ])
    .sort(["year", "frequency"], descending=[False, True])
)

print("Top topics per year:")
print(yearly_topics.head(50))
```

### Workflow 5: Comparing Methods

**Goal**: Compare LDA and BERTopic results on same data.

```bash
# 1. Run LDA
newspaper-explorer analyze topics lda \
  --source der_tag \
  --group-by date \
  --n-topics 10 \
  --output-name lda_comparison

# 2. Run BERTopic with global topics
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by date \
  --min-cluster-size 10 \
  --batch-size 128 \
  --output-name bertopic_comparison

# 3. Compare results programmatically
```

**Comparison Script**:
```python
import polars as pl

lda = pl.read_parquet("results/der_tag/topics/lda_comparison.parquet")
bertopic = pl.read_parquet("results/der_tag/topics/bertopic_comparison.parquet")

# Compare top terms
print("LDA top terms:", lda.select("topic_terms").head(5))
print("BERTopic top terms:", bertopic.select("topic_terms").head(5))

# Compare topic distributions
print(f"LDA unique topics: {len(lda)}")
print(f"BERTopic unique global topics: {len(bertopic)}")

# BERTopic advantage: Topics are consistent across dates
# Can track same topic over time
bertopic_topic_5 = bertopic.filter(pl.col("topics").list.contains(5))
print(f"\nTopic 5 appears on {len(bertopic_topic_5)} dates")
print("Terms:", bertopic_topic_5["topic_terms"].head(1))
```

---

## Analyzing Results

### Loading Results

```python
import polars as pl
import json

# Load topic results
df = pl.read_parquet("results/der_tag/topics/bertopic_topics.parquet")

# Load metadata
with open("results/der_tag/topics/bertopic_topics.json") as f:
    metadata = json.load(f)

print(f"Processed {metadata['input_data']['num_documents']} documents")
print(f"Found {metadata['output_data']['unique_topics_assigned']} unique topics")
print(f"Duration: {metadata['duration_seconds']:.2f}s")
```

### Analyzing Topic Distribution

```python
# Count documents per topic
topic_counts = {}
for topics in df["topics"]:
    for topic_id in topics:
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1

# Sort by frequency
sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

print("Top 10 topics by frequency:")
for topic_id, count in sorted_topics[:10]:
    # Find example terms for this topic
    example = df.filter(pl.col("topics").list.contains(topic_id)).select("topic_terms").head(1)
    print(f"Topic {topic_id}: {count} documents - {example[0, 'topic_terms'][:5]}")
```

### Finding Topic Examples

```python
# Find documents about a specific topic
target_topic = 0

docs_with_topic = df.filter(pl.col("topics").list.contains(target_topic))

print(f"Found {len(docs_with_topic)} documents with Topic {target_topic}")
print("\nTop terms:")
print(docs_with_topic.select("topic_terms").head(3))
```

### Temporal Analysis

```python
# If grouped by date, analyze topic trends
import polars as pl

topics = pl.read_parquet("results/der_tag/topics/lda_topics.parquet")
text = pl.read_parquet("data/raw/der_tag/text/der_tag_lines.parquet")

# Join with dates
joined = topics.join(
    text.select(["page_id", "date"]).unique(),
    on="page_id"
)

# Count topics per month
monthly_topics = (
    joined
    .with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])
    .explode("topics")
    .group_by(["year", "month", "topics"])
    .count()
    .sort(["year", "month", "topics"])
)

print(monthly_topics.head(20))
```

---

## Integration with Other Analyses

### Combining Topics with Entities

```python
import polars as pl

# Load topics and entities
topics = pl.read_parquet("results/der_tag/topics/bertopic_topics.parquet")
entities = pl.read_parquet("results/der_tag/entities/gliner/entities.parquet")

# Join on page_id
combined = topics.join(entities, on="page_id", how="inner")

# Find pages about specific topics with certain entities
pages_with_berlin = combined.filter(
    pl.col("text").str.contains("berlin", literal=True)
)

print(f"Found {len(pages_with_berlin)} pages about Berlin")
print("Topics:", pages_with_berlin.select("topic_terms").head(3))
print("Entities:", pages_with_berlin.select("entity").unique().head(10))
```

### Combining Topics with Emotions

```python
# Load topics and emotions
topics = pl.read_parquet("results/der_tag/topics/lda_topics.parquet")
emotions = pl.read_parquet("results/der_tag/emotions/emotions.parquet")

# Join on line_id (if topics are line-level)
# Or aggregate emotions to page-level first

# Example: Average emotion scores per topic
from collections import defaultdict

topic_emotions = defaultdict(lambda: defaultdict(list))

for row in topics.iter_rows(named=True):
    doc_id = row["doc_id"]
    topics_list = row["topics"]
    
    # Find emotions for this document
    doc_emotions = emotions.filter(pl.col("line_id").str.starts_with(doc_id))
    
    for topic_id in topics_list:
        for emotion in ["joy", "sadness", "anger", "fear"]:
            scores = doc_emotions[emotion].mean()
            if scores:
                topic_emotions[topic_id][emotion].append(scores)

# Average emotions per topic
for topic_id, emotion_scores in topic_emotions.items():
    print(f"\nTopic {topic_id}:")
    for emotion, scores in emotion_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {emotion}: {avg:.3f}")
```

### Using with Query Engine

```python
from newspaper_explorer.analyze.query.engine import QueryEngine

# Initialize query engine
engine = QueryEngine("der_tag")

# Register topics table
engine.register_parquet(
    "topics",
    "results/der_tag/topics/bertopic_topics.parquet"
)

# Query topics with SQL
results = engine.execute("""
    SELECT 
        page_id,
        topic_terms,
        topics,
        topic_probs
    FROM topics
    WHERE list_contains(topics, 0)  -- Documents with topic 0
    LIMIT 10
""")

print(results)
```

---

## Troubleshooting

### Issue: All Documents Skipped

**Symptoms**:
```
No results to process - all documents were skipped or failed
Total documents: 0
```

**Causes**:
1. Too few input lines after grouping
2. Text too short after filtering
3. All texts below `min_text_length`

**Solutions**:
```bash
# Increase --limit to get more groups
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by page_id \
  --limit 50000  # Instead of 1000

# Lower min_text_length
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --min-text-length 50  # Instead of 100
```

### Issue: BERTopic Fails with "max_df < min_df"

**Symptoms**:
```
BERTopic failed for doc X: max_df corresponds to < documents than min_df
```

**Cause**: Document has too few chunks after sentence splitting (less than `min_topic_size`).

**Solutions**:
```bash
# Lower min-topic-size
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --min-topic-size 2  # Instead of 5

# Use more sentences per chunk (fewer total chunks)
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --chunk-sentences 10  # Instead of 5
```

**Note**: This error is expected and normal - it just means some pages don't have enough text. They're skipped gracefully.

### Issue: Model Download Fails

**Symptoms**:
```
Error downloading model: Connection timeout
```

**Solutions**:
```bash
# Manually download model
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Or use different model
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --embedding-model "distiluse-base-multilingual-cased-v2"
```

### Issue: Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Process in smaller batches with --limit
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --limit 5000  # Process in chunks

# Use smaller embedding model
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --embedding-model "paraphrase-MiniLM-L3-v2"

# Or switch to LDA (less memory)
newspaper-explorer analyze topics lda \
  --source der_tag
```

### Issue: Topics Not Meaningful

**Symptoms**: Topic terms don't make sense or are too generic.

**Solutions**:
```bash
# Enable stopword filtering (if disabled)
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  # (stopwords enabled by default)

# Increase min_topic_size for more coherent topics
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --min-topic-size 10  # Larger topics

# Try different grouping level
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --group-by issue_id  # Instead of page_id
```

### Issue: Processing Too Slow

**Symptoms**: BERTopic takes too long to process.

**Solutions**:
```bash
# Use LDA instead
newspaper-explorer analyze topics lda \
  --source der_tag

# Process subset first
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --limit 10000

# Use smaller chunks
newspaper-explorer analyze topics bertopic \
  --source der_tag \
  --chunk-sentences 3  # Fewer, larger chunks
```

### Common Warnings

**"DeprecationWarning: `str.concat` is deprecated"**
- Harmless warning from Polars
- Can be ignored

**"BERTopic failed for doc X: object of type 'numpy.float64' has no len()"**
- Edge case in BERTopic's internal processing
- Document is skipped automatically
- Usually affects <5% of documents

---

## Complete Example

Here's a complete example analyzing topics in newspaper data:

```python
#!/usr/bin/env python3
"""
Complete topic modeling workflow example.
"""
import polars as pl
import json
from pathlib import Path
from newspaper_explorer.analyze.topics.bertopic import BERTopicExtractor

# Configuration
SOURCE = "der_tag"
GROUP_BY = ["page_id"]
LIMIT = 10000

# Initialize extractor
print("Initializing BERTopic extractor...")
extractor = BERTopicExtractor(
    source_name=SOURCE,
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    min_text_length=100,
    chunk_sentences=5,
    use_stopwords=True,
)

# Extract topics
print(f"Extracting topics (limit={LIMIT})...")
results_df = extractor.extract_topics(
    group_by=GROUP_BY,
    min_topic_size=3,
    nr_topics=None,  # Auto-discover
    top_k=5,
    limit=LIMIT,
)

print(f"Extracted topics for {len(results_df)} documents")

# Save results
print("Saving results...")
output_file = extractor.save_results(
    results_df,
    output_name="example_topics",
    top_k=5,
)

print(f"Results saved to: {output_file}")

# Analyze results
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Load metadata
metadata_file = output_file.replace(".parquet", ".json")
with open(metadata_file) as f:
    metadata = json.load(f)

print(f"\nProcessing stats:")
print(f"  Input documents: {metadata['input_data']['num_documents']}")
print(f"  Output documents: {len(results_df)}")
print(f"  Success rate: {len(results_df)/metadata['input_data']['num_documents']*100:.1f}%")
print(f"  Duration: {metadata['duration_seconds']:.2f}s")

# Topic distribution
print(f"\nTopic distribution:")
all_topics = []
for topics in results_df["topics"]:
    all_topics.extend(topics)

unique_topics = set(all_topics)
print(f"  Unique topics: {len(unique_topics)}")
print(f"  Total assignments: {len(all_topics)}")

# Show examples
print(f"\nExample documents:")
for i, row in enumerate(results_df.head(3).iter_rows(named=True)):
    print(f"\n{i+1}. Document: {row['doc_id']}")
    print(f"   Topics: {row['topics']}")
    print(f"   Probabilities: {[f'{p:.2f}' for p in row['topic_probs']]}")
    print(f"   Terms: {row['topic_terms'][:5]}")

print("\n✓ Complete!")
```

**Run it**:
```bash
python topic_modeling_example.py
```

**Expected output**:
```
Initializing BERTopic extractor...
Extracting topics (limit=10000)...
Extracting topics: 100%|████████████████████| 39/39 [00:36<00:00,  1.07it/s]
Extracted topics for 29 documents
Saving results...
Results saved to: /path/to/results/der_tag/topics/example_topics.parquet

================================================================================
ANALYSIS
================================================================================

Processing stats:
  Input documents: 39
  Output documents: 29
  Success rate: 74.4%
  Duration: 37.05s

Topic distribution:
  Unique topics: 3
  Total assignments: 73

Example documents:

1. Document: 3074409-X_1901-01-08_006_1_014
   Topics: [0, 1, 2]
   Probabilities: ['0.40', '0.35', '0.25']
   Terms: ['mk', 'preisen', 'hellmich', '10', 'iſt']

2. Document: 3074409-X_1901-01-10_008_1_002
   Topics: [0, 1]
   Probabilities: ['0.53', '0.47']
   Terms: ['iſt', 'ſich', 'ſein', 'ſo', 'ſeine']

3. Document: 3074409-X_1901-01-09_007_1_003
   Topics: [0, 1]
   Probabilities: ['0.82', '0.18']
   Terms: ['ſich', 'iſt', 'berliner', 'ſind', 'ſie']

✓ Complete!
```

---

**End of Documentation**
