# Emotion Analysis Documentation

## Overview

The emotion analysis module provides BERT-based emotion classification for newspaper texts using the **Shaver emotion model** with 6 categories:

- **Sadness** (Traurigkeit)
- **Love** (Liebe)  
- **Joy** (Freude)
- **Fear** (Angst)
- **Anger** (Wut/Ärger)
- **Agitation** (Unruhe/Erregung)

Each emotion is predicted independently as a binary classification (0 or 1). A text can have multiple emotions or none.

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

## Quick Start

### 1. Check Model Availability

```bash
newspaper-explorer analyze emotions models
```

This shows:
- GPU availability and configuration
- Which emotion models are installed
- Model file sizes

### 2. Predict Emotions

```bash
# Basic usage (auto-detects input file)
newspaper-explorer analyze emotions predict --source der_tag

# With optimized settings for 4x L40S GPUs
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 192 --fp16

# Use custom input file
newspaper-explorer analyze emotions predict --source der_tag \
    --input-file data/processed/der_tag/textblocks_normalized.parquet \
    --text-column text_normalized
```

## CLI Commands

### `newspaper-explorer analyze emotions predict`

Predict emotions for newspaper texts.

**Options:**

- `--source` (required): Source name (e.g., 'der_tag')
- `--input-file`: Custom input parquet file (default: auto-detect textblocks or lines)
- `--text-column`: Name of text column (default: 'text')
- `--batch-size`: Batch size for inference (default: 64, increase for GPUs with more memory)
- `--chunk-size`: Number of rows per chunk (default: 100,000)
- `--model-dir`: Directory containing model files (default: 'models/emotions')
- `--fp16`: Use FP16 mixed precision (recommended for modern GPUs)
- `--single-gpu`: Force single GPU mode (default: use all available GPUs)
- `--output-name`: Base name for output file (default: 'emotion_predictions')

**Examples:**

```bash
# Standard prediction
newspaper-explorer analyze emotions predict --source der_tag

# High-performance mode (4x L40S)
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 192 --chunk-size 200000 --fp16

# Use normalized text
newspaper-explorer analyze emotions predict --source der_tag \
    --input-file data/processed/der_tag/textblocks_normalized.parquet \
    --text-column text_normalized

# Single GPU with smaller batches
newspaper-explorer analyze emotions predict --source der_tag \
    --single-gpu --batch-size 32
```

### `newspaper-explorer analyze emotions models`

Check emotion model availability and GPU configuration.

**Options:**

- `--model-dir`: Directory to check for model files (default: 'models/emotions')

**Example:**

```bash
newspaper-explorer analyze emotions models
```

## Output Format

The prediction results are saved as Parquet files in:
```
results/{source}/emotions/{output_name}.parquet
```

**Schema:**
- All original columns from input file
- `Sadness`: Binary emotion prediction (0/1)
- `Sadness_prob`: Probability score (0.0-1.0)
- `Love`: Binary emotion prediction (0/1)
- `Love_prob`: Probability score (0.0-1.0)
- `Joy`: Binary emotion prediction (0/1)
- `Joy_prob`: Probability score (0.0-1.0)
- `Fear`: Binary emotion prediction (0/1)
- `Fear_prob`: Probability score (0.0-1.0)
- `Anger`: Binary emotion prediction (0/1)
- `Anger_prob`: Probability score (0.0-1.0)
- `Agitation`: Binary emotion prediction (0/1)
- `Agitation_prob`: Probability score (0.0-1.0)

## Performance

### Expected Throughput

| Hardware | Texts/Second | Time for 61M texts |
|----------|--------------|-------------------|
| Single CPU | ~10-50 | 14-70 days |
| Single GPU (RTX 3090) | ~1,000-2,000 | 8-17 hours |
| 4x L40S (multi-GPU) | ~4,000-8,000 | 2-4 hours |

### Optimization Tips

1. **Use FP16** (`--fp16`): 30% faster with minimal accuracy loss
2. **Increase batch size**: For GPUs with more memory (e.g., 192 for L40S)
3. **Multi-GPU**: Automatic when multiple GPUs are available
4. **Larger chunks**: If you have plenty of RAM (e.g., 200,000)

### Optimal Settings for Common Hardware

**4x NVIDIA L40S (48GB each):**
```bash
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 192 --chunk-size 200000 --fp16
```

**Single NVIDIA RTX 3090 (24GB):**
```bash
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 64 --chunk-size 100000 --fp16 --single-gpu
```

**Single NVIDIA RTX 4090 (24GB):**
```bash
newspaper-explorer analyze emotions predict --source der_tag \
    --batch-size 96 --chunk-size 100000 --fp16 --single-gpu
```

## Model Requirements

### Required Files

The following model files must be present in `models/emotions/`:

- `Sadness.pt` (~1.3 GB)
- `Love.pt` (~1.3 GB)
- `Joy.pt` (~1.3 GB)
- `Fear.pt` (~1.3 GB)
- `Anger.pt` (~1.3 GB)
- `Agitation.pt` (~1.3 GB)

Total: ~8 GB disk space

### Model Download

Models can be downloaded from:
- [OwnCloud GWDG](https://owncloud.gwdg.de/index.php/s/g2PjWWcknSRlMSd) (original source)

Place the downloaded `.pt` files in the `models/emotions/` directory.

### Model Details

- **Base model**: `deepset/gbert-large` (German BERT)
- **Training data**: ~1,400 manually annotated German poems (1850-1910)
- **Task**: Binary emotion classification (6 independent classifiers)
- **Input**: German text (max 512 tokens, automatically truncated)
- **Output**: Binary prediction (0 or 1) + probability score (0.0-1.0)
- **Emotion framework**: Shaver's hierarchical emotion model

**Note**: While trained on poetry, the models generalize well to other German text genres including newspapers.

## Python API

You can also use the emotion predictor directly in Python:

```python
from pathlib import Path
from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    source_name="der_tag",
    batch_size=192,
    chunk_size=200000,
    use_fp16=True,
    multi_gpu=True,
)

# Predict from source (auto-detect input)
output_file = predictor.predict_from_source(
    text_column="text",
    output_name="emotions"
)

# Or predict from custom file
output_file = predictor.predict(
    input_file=Path("data/processed/der_tag/textblocks.parquet"),
    text_column="text",
    output_name="emotions"
)

print(f"Results saved to: {output_file}")
```

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
# Emotion trends over time (if date column exists)
df_time = df.group_by("date").agg([
    pl.col("Sadness").mean().alias("sadness_rate"),
    pl.col("Fear").mean().alias("fear_rate"),
    pl.col("Anger").mean().alias("anger_rate"),
]).sort("date")

# Plot (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_time["date"], df_time["sadness_rate"], label="Sadness")
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

## Integration with Other Analyses

### Combine with Entities

```python
# Load emotion predictions and entity extractions
df_emotions = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_entities = pl.read_parquet("results/der_tag/entities/entities.parquet")

# Join on text_block_id or line_id
df_combined = df_emotions.join(df_entities, on="text_block_id")

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

### Combine with Keywords

```python
# Load emotion predictions and keywords
df_emotions = pl.read_parquet("results/der_tag/emotions/emotion_predictions.parquet")
df_keywords = pl.read_parquet("results/der_tag/keywords/tfidf_keywords.parquet")

# Join on page or date
df_combined = df_emotions.join(df_keywords, on="newspaper_page_id")

# Find keywords associated with anger
df_anger = df_combined.filter(pl.col("Anger") == 1)
# Analyze keywords...
```

## Troubleshooting

### Out of GPU Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch-size 32`
2. Disable multi-GPU: `--single-gpu`
3. Use CPU: `--device cpu` (much slower)
4. Process smaller chunks: `--chunk-size 50000`

### Missing Models

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
1. Check model directory: `newspaper-explorer analyze emotions models`
2. Download models to `models/emotions/`
3. Verify file names match exactly: `Sadness.pt`, `Love.pt`, etc.

### Slow Processing

**Symptoms:** Very slow processing speed

**Solutions:**
1. Enable FP16: `--fp16`
2. Increase batch size: `--batch-size 128` or higher
3. Use all GPUs (remove `--single-gpu` if set)
4. Check GPU utilization: `nvidia-smi`

### Low GPU Utilization

**Symptoms:** GPU usage < 50% during processing

**Solutions:**
1. Increase batch size
2. Increase number of DataLoader workers (edit `predictor.py`)
3. Check if data loading is bottleneck (use SSD/NVMe)

## Architecture Notes

### Integration with Pipeline

The emotion analysis follows the same patterns as other analysis modules:

1. **Source-based**: Uses `--source` parameter instead of raw paths
2. **Configuration-driven**: Reads from `config.base.get_config()`
3. **Polars DataFrames**: Input and output use Parquet format
4. **Chunked processing**: Handles large datasets efficiently
5. **CLI + Python API**: Available both ways

### File Organization

```
src/newspaper_explorer/
├── analyze/
│   └── emotions/
│       ├── predictor.py           # Main predictor class
│       ├── predict_parquet.py     # Standalone script (legacy)
│       └── predict_parquet_multi_gpu.py  # Standalone script (legacy)
├── cli/
│   └── analyze/
│       ├── commands.py            # Register emotions group
│       └── emotions/
│           └── commands.py        # CLI commands
```

### Design Decisions

1. **No `__init__.py`**: Follows project pattern of explicit imports
2. **Direct imports**: `from newspaper_explorer.analyze.emotions.predictor import EmotionPredictor`
3. **Click for CLI**: Consistent with other commands
4. **Logging not print**: Uses `logging` module for library code, `click.echo()` for CLI
5. **Configuration integration**: Uses `get_config()` for paths

## Future Enhancements

Possible improvements:

1. **Probability thresholds**: Configurable thresholds for emotion detection (currently fixed at 0.5)
2. **Custom models**: Support for user-provided emotion models
3. **Emotion aggregation**: Pre-computed statistics at page/date/year level
4. **Visualization**: Built-in plotting for emotion trends
5. **Batch processing**: Process multiple sources in one command
6. **Model updates**: Fine-tune on newspaper-specific texts for better domain adaptation

## References

### Primary Sources

- **Kröncke, M., Konle, L., Winko, S., & Jannidis, F. (2023)**. *Gattungen und Emotionen in der Lyrik des Realismus und der frühen Moderne*. DHd 2023 Open Humanities Open Culture. DOI: [10.5281/zenodo.7715402](https://doi.org/10.5281/zenodo.7715402)

- **GitHub Repository**: [LeKonArD/Gattungen_und_Emotionen_dhd2023](https://github.com/LeKonArD/Gattungen_und_Emotionen_dhd2023)

### Theory & Methodology

- **Shaver, P., Schwartz, J., Kirson, D., & O'Connor, C. (1987)**. Emotion knowledge: Further exploration of a prototype approach. *Journal of Personality and Social Psychology*, 52(6), 1061–1086. (Shaver's hierarchical emotion model)

- **German BERT**: [deepset/gbert-large](https://huggingface.co/deepset/gbert-large) - Pre-trained German language model

### Implementation

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for BERT models
- **Multi-GPU**: Process-based parallelism for emotion classification
