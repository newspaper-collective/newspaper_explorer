# Emotion Analysis Documentation

Complete guide to BERT-based emotion classification for historical newspaper texts.

## Table of Contents

1. [Overview](#overview)
2. [Model Source](#model-source)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [CLI Commands](#cli-commands)
6. [Python API](#python-api)
7. [Emotion Categories](#emotion-categories)
8. [Performance & Configuration](#performance--configuration)
9. [Data Schemas](#data-schemas)
10. [Complete Workflows](#complete-workflows)
11. [Analyzing Results](#analyzing-results)
12. [Integration with Other Analyses](#integration-with-other-analyses)
13. [Troubleshooting](#troubleshooting)
14. [Testing](#testing)
15. [Dependencies](#dependencies)
16. [References](#references)

---

## Overview

The emotion analysis module provides BERT-based emotion classification for newspaper texts using the **Shaver emotion model** with 6 categories:

- **Sadness** (Traurigkeit)
- **Love** (Liebe)  
- **Joy** (Freude)
- **Fear** (Angst)
- **Anger** (Wut/Ärger)
- **Agitation** (Unruhe/Erregung)

**Key Features**:
- ✅ **Binary classification**: Each emotion predicted independently (0 or 1)
- ✅ **Probability scores**: Confidence values (0.0-1.0) for each prediction
- ✅ **Multi-GPU support**: Process-based parallelism (each GPU processes different emotions)
- ✅ **Optimized inference**: FP16 mixed precision, torch.compile, optimized DataLoader
- ✅ **Resume functionality**: Automatically skip already-processed texts
- ✅ **Large-scale processing**: Handles millions of texts efficiently

---

## Model Source

The emotion classifiers are based on research from the University of Würzburg and University of Göttingen:

**Publication:**
- Kröncke, M., Konle, L., Winko, S., & Jannidis, F. (2023). *Gattungen und Emotionen in der Lyrik des Realismus und der frühen Moderne*. DHd 2023 Open Humanities Open Culture. [https://doi.org/10.5281/zenodo.7715402](https://doi.org/10.5281/zenodo.7715402)

**Code Repository:**
- [https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023](https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023)

**Training Data:**
- Manually annotated corpus of ~1,400 German poems (1850-1910)
- Focus on poetry from Realism and Early Modernism periods
- Annotations based on Shaver's hierarchical emotion categorization

**Model Architecture:**
- Base: `deepset/gbert-large` (German BERT)
- Fine-tuned for binary emotion classification on German literary texts
- Separate models for each emotion category
- Input: German text (max 512 BERT tokens)
- Output: Binary prediction (0 or 1) + probability score (0.0-1.0)

**Domain Applicability:**
- Trained on German poetry (1850-1910)
- Generalizes to other German text genres including newspapers
- Based on fundamental emotion categories (Shaver model) that transcend specific text types
- Historical German language compatible with newspaper texts from same period

**Note**: While trained on poetry, BERT's transfer learning capabilities and the universal nature of the Shaver emotion categories enable effective application to newspaper texts. The models detect emotional language patterns rather than genre-specific features.

**Model Download:**
- [OwnCloud GWDG](https://owncloud.gwdg.de/index.php/s/g2PjWWcknSRlMSd) (original source)
- Place `.pt` files in `models/emotions/` directory
- Total size: ~8 GB (6 models × ~1.3 GB each)

---

## Architecture

### File Structure

```
src/newspaper_explorer/
├── analyze/
│   └── emotions/
│       ├── predictor.py          # Main EmotionPredictor class
│       ├── predict_parquet.py    # Standalone script (legacy)
│       └── predict_parquet_multi_gpu.py  # Standalone script (legacy)
├── cli/
│   └── analyze/
│       ├── commands.py           # Register emotions group
│       └── emotions/
│           └── commands.py       # CLI commands (predict, models)

models/emotions/                  # Model files (not in repo)
├── Sadness.pt
├── Love.pt
├── Joy.pt
├── Fear.pt
├── Anger.pt
└── Agitation.pt

results/{source}/emotions/        # Output directory
└── emotion_predictions.parquet
```

### Integration with Data Pipeline

```
Download → Parse ALTO/METS → Polars DataFrame → Preprocessing (optional)
                                                        ↓
                                                Emotion Prediction
                                                        ↓
                                        DataFrame with emotion columns
                                                        ↓
                                        Analysis & Visualization
```

**Integration Points**:
- Uses existing Polars DataFrames from `DataLoader`
- Works with both line-level and text-block-level data
- Outputs to standard `results/{source}/emotions/` directory
- Follows configuration-driven patterns

---

## Core Components

### 1. EmotionPredictor

**Purpose**: Main class for emotion classification using pre-trained BERT models.

**Features**:
- Automatic GPU detection and multi-GPU parallelism
- Resume functionality (skip already-processed texts)
- Chunked processing for large datasets
- Optimized inference with FP16 and torch.compile

**Initialization**:
```python
from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

predictor = EmotionPredictor(
    source_name="der_tag",
    model_dir=Path("models/emotions"),  # Optional, defaults to models/emotions
    batch_size=64,                      # Adjust based on GPU memory
    chunk_size=100000,                  # Number of rows per chunk
    use_fp16=False,                     # Enable FP16 mixed precision
    use_compile=True,                   # Enable torch.compile (PyTorch 2.0+)
    multi_gpu=True,                     # Enable multi-GPU parallelism
)
```

**Key Methods**:
- `predict(input_file, text_column, output_name)`: Predict from custom file
- `predict_from_source(text_column, input_file, output_name)`: Auto-detect input from source
- `load_models()`: Load all emotion models (sequential mode only)
- `_predict_sequential()`: Sequential processing (single GPU or CPU)
- `_predict_parallel()`: Parallel processing (multi-GPU)

### 2. TextDataset

**Purpose**: PyTorch Dataset for efficient batching and tokenization.

**Why needed**:
- DataLoader requires a Dataset for efficient batching
- Handles tokenization and padding to BERT's 512 token limit
- Enables parallel data loading with `num_workers`
- Supports memory pinning for faster GPU transfer

**Implementation**:
```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
```

### 3. Worker Processes (Multi-GPU)

**Purpose**: Enable true parallelism by processing different emotions on different GPUs simultaneously.

**How it works**:
1. Main process distributes emotions across available GPUs
2. Each worker process loads one emotion model
3. Workers receive text chunks via multiprocessing queues
4. Workers predict and return results via result queue
5. Main process combines results into final DataFrame

**Benefits**:
- ~3x speedup for 6 emotions on 4 GPUs
- Better GPU utilization
- Parallel processing without GIL limitations

**Worker function**:
```python
def worker_process_emotion(
    gpu_id: int,
    emotion: str,
    model_path: Path,
    texts_queue: mp.Queue,
    results_queue: mp.Queue,
    tokenizer_name: str,
    batch_size: int,
    use_fp16: bool,
    use_compile: bool,
):
    # Load model on specific GPU
    device = f"cuda:{gpu_id}"
    model = load_checkpoint_cls(...)
    
    # Process chunks from queue
    while True:
        chunk_idx, texts = texts_queue.get()
        predictions, probabilities = predict_batch(...)
        results_queue.put((emotion, chunk_idx, predictions, probabilities))
```

### 4. Model Loading

**Purpose**: Load fine-tuned BERT models with optimizations.

**Features**:
- State dict handling for checkpoint compatibility
- FP16 conversion for faster inference
- torch.compile integration (PyTorch 2.0+)
- TF32 precision for Ampere+ GPUs

**Function**:
```python
def load_checkpoint_cls(model_name, path, device, use_fp16=False, use_compile=False):
    model = BertForSequenceClassification.from_pretrained(model_name)
    state_dict = torch.load(path, map_location="cpu")
    
    # Handle state dict mismatches
    # ...
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    if use_fp16 and device != "cpu":
        model.half()
    
    model.eval()
    
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    
    return model
```

---

## CLI Commands

All commands follow: `newspaper-explorer analyze emotions <command> --source <name>`

### 1. Check Model Availability

```bash
# Check which models are installed and GPU status
newspaper-explorer analyze emotions models

# Check custom model directory
newspaper-explorer analyze emotions models --model-dir /path/to/models
```

**Output**:
- GPU availability and configuration
- Model file status (found/missing)
- Model file sizes
- Instructions for downloading missing models

### 2. Predict Emotions

**Basic Usage**:
```bash
# Auto-detect input file from source
newspaper-explorer analyze emotions predict --source der_tag

# Specify custom input file
newspaper-explorer analyze emotions predict --source der_tag \
    --input-file data/processed/der_tag/textblocks_normalized.parquet

# Use custom text column
newspaper-explorer analyze emotions predict --source der_tag \
    --text-column text_normalized
```

**Performance Optimization**:
```bash
# High-performance mode (4x L40S GPUs, 48GB each)
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 192 \
    --chunk-size 200000 \
    --fp16

# Single GPU with smaller batches
newspaper-explorer analyze emotions predict --source der_tag \
    --single-gpu \
    --batch-size 32

# Disable torch.compile (for PyTorch < 2.0)
newspaper-explorer analyze emotions predict --source der_tag \
    --no-compile
```

**Custom Configuration**:
```bash
# Custom model directory and output name
newspaper-explorer analyze emotions predict --source der_tag \
    --model-dir /path/to/models \
    --output-name custom_predictions

# Process specific text column with custom settings
newspaper-explorer analyze emotions predict --source der_tag \
    --input-file data/processed/der_tag/text_normalized.parquet \
    --text-column text_normalized \
    --batch-size 128 \
    --chunk-size 150000
```

**CLI Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--source` | str | *required* | Source name (e.g., 'der_tag') |
| `--input-file` | path | auto-detect | Custom input parquet file |
| `--text-column` | str | `text` | Name of text column to process |
| `--batch-size` | int | `64` | Batch size for inference |
| `--chunk-size` | int | `100000` | Number of rows per chunk |
| `--model-dir` | path | `models/emotions` | Directory containing model files |
| `--fp16` | flag | `False` | Use FP16 mixed precision |
| `--compile` | flag | `True` | Use torch.compile (default enabled) |
| `--no-compile` | flag | `False` | Disable torch.compile |
| `--single-gpu` | flag | `False` | Force single GPU mode |
| `--output-name` | str | `emotion_predictions` | Base name for output file |

---

## Python API

### Basic Usage

```python
from pathlib import Path
from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    source_name="der_tag",
    batch_size=64,
    use_fp16=False,
    multi_gpu=True,
)

# Predict from source (auto-detect input)
output_file = predictor.predict_from_source(
    text_column="text",
    output_name="emotions"
)

print(f"Results saved to: {output_file}")
```

### Advanced Usage

```python
from pathlib import Path
from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

# High-performance configuration
predictor = EmotionPredictor(
    source_name="der_tag",
    model_dir=Path("models/emotions"),
    batch_size=192,           # Large batches for L40S GPUs
    chunk_size=200000,        # Large chunks if RAM permits
    use_fp16=True,            # FP16 for speed
    use_compile=True,         # torch.compile for additional speedup
    multi_gpu=True,           # Use all GPUs in parallel
)

# Predict from custom file
output_file = predictor.predict(
    input_file=Path("data/processed/der_tag/textblocks_normalized.parquet"),
    text_column="text_normalized",
    output_name="emotions_normalized"
)

# Load and inspect results
import polars as pl
df = pl.read_parquet(output_file)

print("Emotion counts:")
print(df[["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]].sum())

print("\nAverage confidence scores:")
print(df[["Sadness_prob", "Love_prob", "Joy_prob", "Fear_prob", "Anger_prob", "Agitation_prob"]].mean())
```

### Complete Example: Process a Year

```python
from pathlib import Path
import polars as pl
from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

# 1. Setup
source = "der_tag"
year = 1902

# 2. Load preprocessed data (optional)
input_file = Path(f"data/processed/{source}/{source}_textblocks_normalized.parquet")

# 3. Filter to specific year (optional)
df = pl.read_parquet(input_file)
df_year = df.filter(pl.col("year") == year)
year_file = Path(f"data/processed/{source}/{source}_textblocks_{year}.parquet")
df_year.write_parquet(year_file)

# 4. Initialize predictor
predictor = EmotionPredictor(
    source_name=source,
    batch_size=128,
    use_fp16=True,
    multi_gpu=True,
)

# 5. Predict emotions
output_file = predictor.predict(
    input_file=year_file,
    text_column="text_normalized",
    output_name=f"emotions_{year}"
)

# 6. Analyze results
df_results = pl.read_parquet(output_file)

# Emotion statistics
emotion_cols = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
emotion_counts = df_results.select(emotion_cols).sum()
print(f"Emotion counts for {year}:")
print(emotion_counts)

# Temporal trends (by month)
df_monthly = df_results.group_by(pl.col("date").dt.month()).agg([
    pl.col("Sadness").mean().alias("sadness_rate"),
    pl.col("Joy").mean().alias("joy_rate"),
    pl.col("Fear").mean().alias("fear_rate"),
]).sort("date")

print(f"\nMonthly emotion trends for {year}:")
print(df_monthly)
```

---

## Emotion Categories

The Shaver emotion model organizes emotions hierarchically. The 6 categories used are:

| Emotion | German | Description | Common Triggers |
|---------|--------|-------------|-----------------|
| **Sadness** | Traurigkeit | Sorrow, grief, disappointment | Loss, tragedy, failure |
| **Love** | Liebe | Affection, fondness, attraction | Romance, family, friendship |
| **Joy** | Freude | Happiness, delight, contentment | Success, celebration, good news |
| **Fear** | Angst | Anxiety, worry, terror | Threat, danger, uncertainty |
| **Anger** | Wut/Ärger | Rage, irritation, frustration | Injustice, provocation, obstacles |
| **Agitation** | Unruhe/Erregung | Excitement, restlessness, tension | Anticipation, stress, stimulation |

**Note**: These are binary classifications - a text either expresses the emotion (1) or doesn't (0). Multiple emotions can be present simultaneously.

---

## Performance & Configuration

### Optimization Techniques

1. **FP16 Mixed Precision** (`--fp16`):
   - ~30% faster inference
   - Minimal accuracy loss (<1%)
   - Requires modern GPU (Volta or newer)

2. **torch.compile** (enabled by default):
   - Additional 20-30% speedup
   - Requires PyTorch 2.0+
   - Disable with `--no-compile` if needed

3. **Multi-GPU Parallelism**:
   - Each GPU processes different emotions
   - ~3x speedup for 6 emotions on 4 GPUs
   - Automatic when multiple GPUs available
   - Disable with `--single-gpu` if needed

4. **Batch Size**:
   - Larger = faster, but more memory
   - Recommended: 32-64 for 24GB, 128-192 for 48GB
   - Adjust based on GPU memory

5. **Chunk Size**:
   - Larger = less overhead, but more RAM
   - Recommended: 100,000-200,000
   - Reduce if running out of RAM

6. **DataLoader Settings** (in code):
   - `num_workers=4`: Parallel data loading
   - `persistent_workers=True`: Keep workers alive
   - `prefetch_factor=4`: Aggressive prefetching
   - `pin_memory=True`: Faster GPU transfer


### Resume Functionality

The predictor automatically resumes if interrupted:

1. **Chunk-based processing**: Each chunk saved separately
2. **ID tracking**: Remembers processed `line_id` values
3. **Skip processed**: Only processes new data
4. **Final combination**: Merges chunks at end

Example:
```bash
# First run (interrupted after 2M rows)
newspaper-explorer analyze emotions predict --source der_tag

# Resume (skips first 2M rows, continues from 2M+1)
newspaper-explorer analyze emotions predict --source der_tag
# Output: "Already processed: 2,000,000 rows, Remaining: 59,000,000 rows"
```

---

## Data Schemas

### Input Schema

Any Polars DataFrame with at least:
- Text column (default: `text`, configurable via `--text-column`)
- ID column (required: `line_id` or `text_block_id`)

Optional columns:
- `date`: Publication date
- `year`: Publication year
- `newspaper_title`: Newspaper name
- `page_id`: Page identifier
- Any other metadata

### Output Schema

Original columns + emotion predictions:

```python
{
    # Original columns (preserved)
    "line_id": str,
    "text": str,
    "date": datetime,
    "year": int,
    # ... other original columns ...
    
    # Binary predictions (0 or 1)
    "Sadness": int,
    "Love": int,
    "Joy": int,
    "Fear": int,
    "Anger": int,
    "Agitation": int,
    
    # Probability scores (0.0 to 1.0)
    "Sadness_prob": float,
    "Love_prob": float,
    "Joy_prob": float,
    "Fear_prob": float,
    "Anger_prob": float,
    "Agitation_prob": float,
}
```

### Example Output

```python
import polars as pl

df = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
print(df.head())

# Output:
# ┌───────────┬────────────────────┬────────────┬──────┬─────────┬──────┬─────┬───────────────┐
# │ line_id   │ text               │ date       │ year │ Sadness │ Love │ Joy │ Sadness_prob  │
# ├───────────┼────────────────────┼────────────┼──────┼─────────┼──────┼─────┼───────────────┤
# │ 1902_...  │ "Die Nachricht..." │ 1902-01-01 │ 1902 │ 1       │ 0    │ 0   │ 0.87          │
# │ 1902_...  │ "Eine frohe..."    │ 1902-01-01 │ 1902 │ 0       │ 0    │ 1   │ 0.12          │
# └───────────┴────────────────────┴────────────┴──────┴─────────┴──────┴─────┴───────────────┘
```

---

## Complete Workflows

### Workflow 1: Quick Emotion Analysis

```bash
# 1. Check model availability
newspaper-explorer analyze emotions models

# 2. Predict emotions (auto-detect input)
newspaper-explorer analyze emotions predict --source der_tag

# 3. Load and analyze results
python3 << 'EOF'
import polars as pl

df = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")

# Emotion counts
emotions = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
print("Emotion counts:")
print(df[emotions].sum())

# Emotion percentages
print("\nEmotion percentages:")
print((df[emotions].sum() / len(df) * 100).round(2))
EOF
```

### Workflow 2: High-Performance Processing

```bash
# 1. Check GPU availability
newspaper-explorer analyze emotions models

# 2. Process with optimized settings (4x L40S)
newspaper-explorer analyze emotions predict \
    --source der_tag \
    --batch-size 192 \
    --chunk-size 200000 \
    --fp16

# 3. Monitor GPU usage (in separate terminal)
watch -n 1 nvidia-smi
```

### Workflow 3: Normalized Text Processing

```bash
# 1. Preprocess text
newspaper-explorer data preprocess \
    --source der_tag \
    --steps normalization lemmatization

# 2. Predict on normalized text
newspaper-explorer analyze emotions predict \
    --source der_tag \
    --input-file data/processed/der_tag/der_tag_textblocks_preprocessed.parquet \
    --text-column text_normalized \
    --output-name emotions_normalized

# 3. Compare results
python3 << 'EOF'
import polars as pl

df_raw = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_norm = pl.read_parquet("results/der_tag/emotions/emotions_normalized.parquet")

print("Raw text emotion rates:")
print(df_raw[["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]].mean())

print("\nNormalized text emotion rates:")
print(df_norm[["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]].mean())
EOF
```

### Workflow 4: Temporal Analysis

```bash
# 1. Predict emotions
newspaper-explorer analyze emotions predict --source der_tag --fp16

# 2. Analyze temporal trends
python3 << 'EOF'
import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")

# Monthly emotion trends
df_monthly = df.group_by(pl.col("date").dt.to_string("%Y-%m")).agg([
    pl.col("Sadness").mean().alias("Sadness"),
    pl.col("Joy").mean().alias("Joy"),
    pl.col("Fear").mean().alias("Fear"),
    pl.col("Anger").mean().alias("Anger"),
]).sort("date")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_monthly["date"], df_monthly["Sadness"], label="Sadness", linewidth=2)
plt.plot(df_monthly["date"], df_monthly["Joy"], label="Joy", linewidth=2)
plt.plot(df_monthly["date"], df_monthly["Fear"], label="Fear", linewidth=2)
plt.plot(df_monthly["date"], df_monthly["Anger"], label="Anger", linewidth=2)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Emotion Rate", fontsize=12)
plt.title("Emotion Trends Over Time", fontsize=14)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/der_tag/emotions/emotion_trends.png", dpi=300)
print("Plot saved to: results/der_tag/emotions/emotion_trends.png")
EOF
```

---

## Analyzing Results

### Load Results

```python
import polars as pl

# Load predictions
df = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")

# Show schema
print(df.schema)

# Show sample
print(df.head())

# Basic statistics
print(df.describe())
```

### Aggregate Statistics

```python
# Count emotions
emotion_cols = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
emotion_counts = df.select(emotion_cols).sum()
print("Emotion counts:")
print(emotion_counts)

# Percentage of texts with each emotion
total_texts = len(df)
emotion_pct = (df.select(emotion_cols).sum() / total_texts * 100).round(2)
print("\nEmotion percentages:")
print(emotion_pct)

# Average probability scores
prob_cols = [f"{e}_prob" for e in emotion_cols]
avg_probs = df.select(prob_cols).mean()
print("\nAverage probability scores:")
print(avg_probs)

# Texts with multiple emotions
df_multi = df.with_columns(
    (pl.col("Sadness") + pl.col("Love") + pl.col("Joy") + 
     pl.col("Fear") + pl.col("Anger") + pl.col("Agitation")).alias("emotion_count")
)
print(f"\nTexts with multiple emotions: {len(df_multi.filter(pl.col('emotion_count') > 1)):,}")

# Most common emotion combinations
df_combo = df.select(emotion_cols).group_by(emotion_cols).count().sort("count", descending=True)
print("\nTop emotion combinations:")
print(df_combo.head(10))
```

### Temporal Analysis

```python
# Emotion trends over time
df_time = df.group_by("date").agg([
    pl.col("Sadness").mean().alias("sadness_rate"),
    pl.col("Joy").mean().alias("joy_rate"),
    pl.col("Fear").mean().alias("fear_rate"),
    pl.col("Anger").mean().alias("anger_rate"),
]).sort("date")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_time["date"], df_time["sadness_rate"], label="Sadness")
plt.plot(df_time["date"], df_time["joy_rate"], label="Joy")
plt.plot(df_time["date"], df_time["fear_rate"], label="Fear")
plt.plot(df_time["date"], df_time["anger_rate"], label="Anger")
plt.xlabel("Date")
plt.ylabel("Emotion Rate")
plt.title("Emotion Trends Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("emotion_trends.png")
```

### Filter by Emotion

```python
# Get all texts with sadness
df_sad = df.filter(pl.col("Sadness") == 1)
print(f"Texts with sadness: {len(df_sad):,}")

# Get texts with high sadness probability (> 0.8)
df_high_sad = df.filter(pl.col("Sadness_prob") > 0.8)
print(f"Texts with high sadness confidence: {len(df_high_sad):,}")

# Get texts with multiple negative emotions
df_negative = df.filter(
    (pl.col("Sadness") + pl.col("Fear") + pl.col("Anger") + pl.col("Agitation")) >= 2
)
print(f"Texts with multiple negative emotions: {len(df_negative):,}")

# Get texts with joy but no negative emotions
df_pure_joy = df.filter(
    (pl.col("Joy") == 1) & 
    (pl.col("Sadness") + pl.col("Fear") + pl.col("Anger") + pl.col("Agitation") == 0)
)
print(f"Texts with pure joy: {len(df_pure_joy):,}")
```

### Probability Analysis

```python
# Distribution of confidence scores
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
emotions = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]

for i, emotion in enumerate(emotions):
    ax = axes[i // 3, i % 3]
    probs = df[f"{emotion}_prob"].to_numpy()
    ax.hist(probs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f"{emotion} Probability Distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    ax.legend()

plt.tight_layout()
plt.savefig("emotion_probability_distributions.png")
```

---

## Integration with Other Analyses

### Combine with Entities

```python
# Load emotion predictions and entity extractions
df_emotions = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_entities = pl.read_parquet("results/der_tag/entities/entities.parquet")

# Join on text_block_id or line_id
df_combined = df_emotions.join(df_entities, on="text_block_id", how="inner")

# Find which entities are associated with fear
df_fear_entities = df_combined.filter(pl.col("Fear") == 1)
entity_fear_freq = (
    df_fear_entities
    .explode("entities")
    .group_by("entities")
    .count()
    .sort("count", descending=True)
)
print("Entities most associated with fear:")
print(entity_fear_freq.head(20))
```

### Combine with Layout Analysis

```python
# Load emotion predictions and layout detections
df_emotions = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_headlines = pl.read_parquet("results/der_tag/layout/der_tag_headlines.parquet")

# Aggregate emotions by page
df_page_emotions = df_emotions.group_by("page_id").agg([
    pl.col("Sadness").mean().alias("sadness_rate"),
    pl.col("Joy").mean().alias("joy_rate"),
    pl.col("Fear").mean().alias("fear_rate"),
])

# Join with headlines
df_combined = df_headlines.join(df_page_emotions, on="page_id")

# Find headlines associated with high fear
df_fear_headlines = df_combined.filter(pl.col("fear_rate") > 0.3)
print("Headlines from pages with high fear:")
print(df_fear_headlines.select(["headline_text", "fear_rate"]).head(20))
```

### Combine with Keywords

```python
# Load emotion predictions and keywords
df_emotions = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_keywords = pl.read_parquet("results/der_tag/keywords/tfidf_keywords.parquet")

# Join on page_id or date
df_combined = df_emotions.join(df_keywords, on="page_id", how="inner")

# Find keywords associated with anger
df_anger = df_combined.filter(pl.col("Anger") == 1)
anger_keywords = (
    df_anger
    .explode("keywords")
    .group_by("keywords")
    .agg(pl.count().alias("frequency"))
    .sort("frequency", descending=True)
)
print("Keywords most associated with anger:")
print(anger_keywords.head(20))
```

---

## Troubleshooting

### Out of GPU Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 32`
2. Disable multi-GPU: `--single-gpu`
3. Reduce chunk size: `--chunk-size 50000`
4. Use CPU: Remove GPU flags (much slower)
5. Process in smaller batches (multiple runs with `--limit`)

### Missing Models

**Error**: `FileNotFoundError: Model file not found`

**Solutions**:
1. Check model directory: `newspaper-explorer analyze emotions models`
2. Download models to `models/emotions/` directory
3. Verify file names match exactly: `Sadness.pt`, `Love.pt`, `Joy.pt`, `Fear.pt`, `Anger.pt`, `Agitation.pt`
4. Check file permissions

### Slow Processing

**Symptoms**: Very slow processing speed (<100 texts/second)

**Solutions**:
1. Enable FP16: `--fp16`
2. Increase batch size: `--batch-size 128` or higher
3. Use all GPUs: Remove `--single-gpu` flag
4. Check GPU utilization: `nvidia-smi`
5. Ensure PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
6. Use SSD/NVMe for data storage

### Low GPU Utilization

**Symptoms**: GPU usage < 50% during processing

**Solutions**:
1. Increase batch size (more work per batch)
2. Increase DataLoader workers (edit `predictor.py`: `num_workers=8`)
3. Increase prefetch factor (edit `predictor.py`: `prefetch_factor=8`)
4. Check if data loading is bottleneck (use `htop` to monitor CPU)
5. Use faster storage (SSD/NVMe)

### torch.compile Errors

**Error**: `AttributeError: module 'torch' has no attribute 'compile'`

**Solutions**:
1. Disable torch.compile: `--no-compile`
2. Upgrade PyTorch: `pip install --upgrade torch>=2.0.0`
3. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`

### Multiprocessing Errors

**Error**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

**Solutions**:
1. Use single GPU mode: `--single-gpu`
2. Check multiprocessing start method (should be 'spawn', not 'fork')
3. Restart Python kernel/process

### Resume Not Working

**Symptoms**: Processing restarts from beginning instead of resuming

**Solutions**:
1. Ensure `line_id` column exists in input data
2. Check for chunk files: `ls results/{source}/emotions/*_chunk_*.parquet`
3. Delete incomplete chunks to force restart
4. Verify output filename matches previous run

---

## Testing

### Quick Test (Small Sample)

```bash
# Test on 1000 rows
newspaper-explorer data parse --source der_tag --year 1902 --limit 1000

newspaper-explorer analyze emotions predict \
    --source der_tag \
    --input-file data/raw/der_tag/text/der_tag_lines.parquet \
    --batch-size 32 \
    --chunk-size 1000

# Check output
python3 -c "import polars as pl; df = pl.read_parquet('results/der_tag/emotions/emotion_predictions.parquet'); print(df.shape); print(df.head())"
```

### GPU Test

```bash
# Verify GPU availability
newspaper-explorer analyze emotions models

# Test GPU processing
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

### Performance Test

```bash
# Test processing speed
time newspaper-explorer analyze emotions predict \
    --source der_tag \
    --input-file test_data.parquet \
    --batch-size 64 \
    --fp16

# Monitor GPU usage
nvidia-smi dmon -s u
```

### Validation Test

```python
# Test prediction quality
import polars as pl

df = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")

# Check for NaN values
assert df.null_count().sum(axis=1)[0] == 0, "Found NaN values"

# Check probability ranges
prob_cols = ["Sadness_prob", "Love_prob", "Joy_prob", "Fear_prob", "Anger_prob", "Agitation_prob"]
for col in prob_cols:
    assert df[col].min() >= 0.0, f"{col} has values < 0"
    assert df[col].max() <= 1.0, f"{col} has values > 1"

# Check binary consistency
emotion_cols = ["Sadness", "Love", "Joy", "Fear", "Anger", "Agitation"]
for emotion in emotion_cols:
    binary_col = emotion
    prob_col = f"{emotion}_prob"
    
    # High prob should mostly be binary 1
    high_prob = df.filter(pl.col(prob_col) > 0.8)
    if len(high_prob) > 0:
        assert high_prob[binary_col].mean() > 0.8, f"{emotion}: High prob but low binary rate"
    
    # Low prob should mostly be binary 0
    low_prob = df.filter(pl.col(prob_col) < 0.2)
    if len(low_prob) > 0:
        assert low_prob[binary_col].mean() < 0.2, f"{emotion}: Low prob but high binary rate"

print("✓ All validation checks passed")
```

---

## Dependencies

Required packages (in `requirements.txt`):

```txt
# Core deep learning
torch>=2.0.0              # PyTorch with CUDA support
transformers>=4.30.0      # Hugging Face transformers

# Data processing
polars>=0.20.0           # Fast DataFrame operations
pyarrow>=12.0.0          # Parquet support

# CLI
click>=8.0.0             # Command-line interface

# Progress tracking
tqdm>=4.65.0             # Progress bars

# Optional (for visualization)
matplotlib>=3.7.0        # Plotting
seaborn>=0.12.0         # Statistical visualizations
```

### Installation

```bash
# Basic installation
pip install torch transformers polars pyarrow click tqdm

# With CUDA support (for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# With visualization
pip install matplotlib seaborn

# Or install all project dependencies
pip install -r requirements.txt
```

### System Requirements

**Minimum**:
- Python 3.8+
- 8 GB RAM
- 10 GB disk space (for models)

**Recommended**:
- Python 3.10+
- 32 GB RAM
- NVIDIA GPU with 24GB+ VRAM
- CUDA 11.8+
- SSD/NVMe storage
- 50 GB disk space (for models + data)

**Optimal**:
- Python 3.11+
- 64 GB RAM
- 4x NVIDIA L40S or similar (48GB each)
- CUDA 12.0+
- NVMe storage
- 500 GB disk space

---

## References

### Primary Sources

- **Kröncke, M., Konle, L., Winko, S., & Jannidis, F. (2023)**. *Gattungen und Emotionen in der Lyrik des Realismus und der frühen Moderne*. DHd 2023 Open Humanities Open Culture. DOI: [10.5281/zenodo.7715402](https://doi.org/10.5281/zenodo.7715402)

- **GitHub Repository**: [LeKonArD/Gattungen_und_Emotionen_dhd2023](https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023)

- **Model Download**: [OwnCloud GWDG](https://owncloud.gwdg.de/index.php/s/g2PjWWcknSRlMSd)

### Theory & Methodology

- **Shaver, P., Schwartz, J., Kirson, D., & O'Connor, C. (1987)**. Emotion knowledge: Further exploration of a prototype approach. *Journal of Personality and Social Psychology*, 52(6), 1061–1086. (Shaver's hierarchical emotion model)

- **German BERT**: [deepset/gbert-large](https://huggingface.co/deepset/gbert-large) - Pre-trained German language model

### Implementation

- **PyTorch**: [https://pytorch.org/](https://pytorch.org/) - Deep learning framework
- **Transformers**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers) - Hugging Face library for BERT models
- **Polars**: [https://pola.rs/](https://pola.rs/) - Fast DataFrame library

### Related Work

- **Emotion detection in text**: Surveys and state-of-the-art approaches
- **Historical text analysis**: Domain adaptation techniques
- **Large-scale NLP**: Best practices for processing millions of documents
