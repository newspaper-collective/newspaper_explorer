# OCR Quality Validation

**Module:** `newspaper_explorer.data.preprocessing.validation`
**Status:** ✅ Complete
**Purpose:** Assess OCR text quality and filter low-quality content

> **📚 For comprehensive preprocessing documentation, see [PREPROCESSING.md](PREPROCESSING.md)**
> This document focuses specifically on quality validation methods.

---

## Overview

The quality validation module provides comprehensive quality assessment for historical German newspaper OCR text. It calculates multiple quality indicators to assess OCR accuracy and text cleanliness.

### File Structure

```
src/newspaper_explorer/data/preprocessing/
└── validation.py        # Quality metrics and filtering
```

### Available Functions

| Function | Purpose | Speed |
|----------|---------|-------|
| `calculate_quality_metrics()` | Calculate char/token ratio, OOV rate, proper noun density | ⚡⚡⚡⚡ Fast |
| `filter_by_quality_score()` | Filter DataFrame by quality score threshold | ⚡⚡⚡⚡⚡ Instant |
| `summarize_quality()` | Generate summary statistics for quality metrics | ⚡⚡⚡⚡⚡ Instant |

## Quality Metrics

### 1. Character-to-Token Ratio

**What it measures:** Average number of characters per token (word).

**Interpretation:**
- **Clean German:** 5-7 chars/token
- **Review needed:** 7-8 chars/token
- **Poor quality:** >8 chars/token

**Example:**
```python
"Die Zeitung berichtet"  # 6.1 chars/token (good)
"DieZeitungberichtet"    # 21 chars/token (poor - OCR removed spaces)
```

### 2. Out-of-Vocabulary (OOV) Rate

**What it measures:** Percentage of words not found in German dictionary.

**Interpretation:**
- **Good:** <5% OOV
- **Review:** 5-15% OOV
- **Poor:** >15% OOV

**Requirements:**
- Requires German wordlist file
- Falls back to char/token ratio if not provided

**Example:**
```python
"Die Zeitung berichtet"     # 0% OOV (all valid German)
"Die Zeeitung berichtet"    # 12.5% OOV ("Zeeitung" is misspelled)
"sssss uuuuu nnnnnn"        # 100% OOV (OCR garbage)
```

### 3. Proper Noun Density

**What it measures:** Ratio of proper nouns (capitalized words) to total tokens.

**Interpretation:**
- **Normal:** 10-30% (articles, names, places)
- **High:** >50% (may indicate OCR capitalization errors)
- **Very high:** >80% (likely poor OCR quality)

**Note:** Excludes common German articles (Der, Die, Das, Ein, Eine)

**Example:**
```python
"Die Zeitung berichtet über Berlin"  # 20% proper nouns (good)
"DIE ZEITUNG BERICHTET ÜBER BERLIN"  # 80% proper nouns (poor)
```

### 4. Overall Quality Score

**What it measures:** Combined assessment: good, review, or poor.

**Decision logic:**
1. **With OOV rate** (primary):
   - `good`: OOV < 5%
   - `review`: OOV 5-15%
   - `poor`: OOV > 15%

2. **Without OOV rate** (fallback):
   - `good`: char/token ≤ 7.0
   - `review`: char/token ≤ 8.0
   - `poor`: char/token > 8.0

## Functions

### calculate_quality_metrics()

Calculate comprehensive quality metrics for OCR text.

**Signature:**
```python
calculate_quality_metrics(
    df: pl.DataFrame,
    input_column: str = "text",
    german_wordlist_path: Optional[str] = None,
    output_column_prefix: str = "quality_",
) -> pl.DataFrame
```

**Parameters:**
- `df`: Input DataFrame
- `input_column`: Column to analyze (default: "text")
- `german_wordlist_path`: Path to German word list file (optional)
- `output_column_prefix`: Prefix for output metric columns (default: "quality_")

**Returns:**
DataFrame with added quality metric columns:
- `{prefix}char_token_ratio`: float
- `{prefix}oov_rate`: float
- `{prefix}proper_noun_density`: float
- `{prefix}score`: str (good/review/poor)

**Example:**
```python
from newspaper_explorer.data.preprocessing.validation import calculate_quality_metrics
import polars as pl

# Load data
df = pl.read_parquet("data.parquet")

# Calculate quality metrics (without wordlist)
df = calculate_quality_metrics(df, input_column="text")

# Or with German wordlist for OOV calculation
df = calculate_quality_metrics(
    df,
    input_column="text",
    german_wordlist_path="wordlist_de.txt",
)

# Access metrics
df.select([
    "text",
    "quality_char_token_ratio",
    "quality_oov_rate",
    "quality_score",
])
```

**Pipeline usage:**
```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor()
df = preprocessor.pipeline(
    df,
    steps=[
        "normalize_unicode",
        "normalize_long_s",
        "calculate_quality_metrics",  # Add quality metrics
    ],
)
```

### filter_by_quality_score()

Filter DataFrame by quality score threshold.

**Signature:**
```python
filter_by_quality_score(
    df: pl.DataFrame,
    quality_column: str = "quality_score",
    min_quality: str = "review",
) -> pl.DataFrame
```

**Parameters:**
- `df`: Input DataFrame with quality scores
- `quality_column`: Column containing quality scores (default: "quality_score")
- `min_quality`: Minimum quality to keep (default: "review")
  - `"good"`: Keep only good quality
  - `"review"`: Keep good + review quality
  - `"poor"`: Keep all (no filtering)

**Returns:**
DataFrame with low-quality rows filtered out

**Example:**
```python
from newspaper_explorer.data.preprocessing.validation import (
    calculate_quality_metrics,
    filter_by_quality_score,
)

# Calculate metrics first
df = calculate_quality_metrics(df)

# Keep only good and review quality
df = filter_by_quality_score(df, min_quality="review")

# Or keep only good quality
df = filter_by_quality_score(df, min_quality="good")
```

**Pipeline usage:**
```python
df = preprocessor.pipeline(
    df,
    steps=[
        "normalize_unicode",
        "calculate_quality_metrics",
        "filter_by_quality_score",  # Remove poor quality
    ],
)
```

### summarize_quality()

Generate summary statistics for quality metrics.

**Signature:**
```python
summarize_quality(
    df: pl.DataFrame,
    quality_column_prefix: str = "quality_",
) -> Dict[str, float]
```

**Parameters:**
- `df`: DataFrame with quality metrics
- `quality_column_prefix`: Prefix for quality columns (default: "quality_")

**Returns:**
Dictionary with summary statistics:
- `avg_char_token_ratio`: Average char/token ratio
- `median_char_token_ratio`: Median char/token ratio
- `avg_oov_rate`: Average OOV rate
- `median_oov_rate`: Median OOV rate
- `avg_proper_noun_density`: Average proper noun density
- `pct_good`: Percentage of good quality rows
- `pct_review`: Percentage of review quality rows
- `pct_poor`: Percentage of poor quality rows

**Example:**
```python
from newspaper_explorer.data.preprocessing.validation import (
    calculate_quality_metrics,
    summarize_quality,
)

# Calculate metrics
df = calculate_quality_metrics(df)

# Generate summary
summary = summarize_quality(df)

print(f"Average char/token ratio: {summary['avg_char_token_ratio']:.2f}")
print(f"Average OOV rate: {summary['avg_oov_rate']:.1%}")
print(f"Good quality: {summary['pct_good']:.1%}")
print(f"Review quality: {summary['pct_review']:.1%}")
print(f"Poor quality: {summary['pct_poor']:.1%}")
```

## German Wordlist

### Format

The German wordlist should be a plain text file with one word per line:

```
der
die
das
zeitung
berichtet
über
...
```

### Sources

**Recommended sources for German wordlists:**

1. **Hunspell German Dictionary**
   - Part of LibreOffice/Firefox spell checkers
   - Includes word stems and suffixes
   - ~60,000+ entries

2. **German Frequency Lists**
   - Leipzig Corpora Collection
   - ~100,000+ most common German words
   - Historical variants included

3. **Custom Historical German Wordlist**
   - Include Fraktur-specific variations
   - Historical spellings (e.g., "Thal" vs "Tal")
   - Period-appropriate vocabulary

### Usage

```python
# Download/create German wordlist
wordlist_path = "data/wordlist_de.txt"

# Use in quality calculation
df = calculate_quality_metrics(
    df,
    german_wordlist_path=wordlist_path,
)
```

## Use Cases

### 1. Quality Assessment

Assess overall OCR quality before further processing:

```python
# Calculate metrics
df = calculate_quality_metrics(df)

# Generate summary
summary = summarize_quality(df)

print(f"Dataset quality:")
print(f"  Good: {summary['pct_good']:.1%}")
print(f"  Review: {summary['pct_review']:.1%}")
print(f"  Poor: {summary['pct_poor']:.1%}")
```

### 2. Quality-Based Filtering

Filter out poor quality text before analysis:

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor()
df = preprocessor.pipeline(
    df,
    steps=[
        "normalize_unicode",
        "normalize_long_s",
        "calculate_quality_metrics",
        "filter_by_quality_score",  # Remove poor quality
    ],
)
```

### 3. Selective Processing

Apply different processing based on quality:

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor
from newspaper_explorer.data.preprocessing.normalization import normalize_unicode
from newspaper_explorer.data.preprocessing.validation import calculate_quality_metrics

# Calculate quality
df = calculate_quality_metrics(df)

# Process good quality minimally
df_good = df.filter(pl.col("quality_score") == "good")
df_good = normalize_unicode(df_good)

# Process poor quality more aggressively
preprocessor = TextPreprocessor()
df_poor = df.filter(pl.col("quality_score") == "poor")
df_poor = preprocessor.pipeline(
    df_poor,
    steps=[
        "normalize_unicode",
        "remove_garbage_words",
        "filter_number_only_lines",
        "normalize_long_s",
    ],
)

# Combine
df = pl.concat([df_good, df_poor])
```

### 4. Quality Reporting

Generate quality reports for documentation:

```python
# Group by year and calculate quality
quality_by_year = (
    df.group_by("year")
    .agg([
        pl.col("quality_score").value_counts(),
        pl.col("quality_char_token_ratio").mean().alias("avg_ratio"),
    ])
)

# Export report
quality_by_year.write_csv("quality_report_by_year.csv")
```

## Integration with Preprocessing Pipeline

### Basic Integration

```python
from newspaper_explorer.data.preprocessing.pipeline import TextPreprocessor

preprocessor = TextPreprocessor()

df = preprocessor.pipeline(
    df,
    steps=[
        # Cleaning
        "normalize_unicode",
        "normalize_long_s",
        "remove_garbage_words",
        "normalize_whitespace",

        # Quality validation
        "calculate_quality_metrics",
        "filter_by_quality_score",  # Remove poor quality
    ],
)
```

### Advanced Integration

```python
# Calculate quality with custom parameters
df = preprocessor.pipeline(
    df,
    steps=["calculate_quality_metrics"],
)

# Filter by quality with custom threshold
df = preprocessor.pipeline(
    df,
    steps=["filter_by_quality_score"],
)
```

## Testing

Run the test script to validate quality validation:

```bash
python test_quality.py
```

**Test coverage:**
- Good quality text (clean German)
- Review quality text (some OCR issues)
- Poor quality text (heavy fragmentation, repetition, merged words)
- Quality score calculation
- Summary statistics generation
- Quality-based filtering

## Performance

**Computational cost:** Low to moderate
- Character/token ratio: O(n) - very fast
- OOV calculation: O(n × m) where m = avg word lookups - moderate with hash set
- Proper noun detection: O(n) - very fast

**Typical processing time:**
- 10,000 rows: ~1-2 seconds (without wordlist)
- 10,000 rows: ~3-5 seconds (with wordlist)
- 100,000 rows: ~10-20 seconds (without wordlist)
- 100,000 rows: ~30-50 seconds (with wordlist)

**Memory usage:**
- Wordlist: ~5-10 MB for 100,000 words
- Quality metrics: 4 additional columns per row

## Research Basis

From `normalize.md` (Stage 6: Quality Validation):

> **Quality indicators:**
> - Character-to-token ratio (clean German: 5-7 chars/token)
> - Out-of-vocabulary rate (good: <5%, review: 5-15%, poor: >15%)
> - Overall quality score based on combined metrics
>
> **Use case:** Filter low-quality text before analysis, or apply
> different preprocessing strategies based on quality tier.

## Future Enhancements

Potential improvements for OCR quality validation:

1. **Character Error Rate (CER) Estimation**
   - Compare against ground truth samples
   - Statistical CER estimation without ground truth

2. **Line-Level Confidence Scores**
   - OCR engine confidence integration
   - Geometric features (spacing, alignment)

3. **Historical German Specific Metrics**
   - Fraktur character frequency analysis
   - Period-appropriate vocabulary coverage
   - Long s (ſ) pattern validation

4. **Machine Learning Quality Scoring**
   - Train classifier on labeled quality data
   - Feature engineering from metrics
   - Ensemble of quality indicators

5. **Interactive Quality Dashboard**
   - Visualize quality distribution
   - Sample low-quality examples
   - Quality trends over time/sources

## See Also

- **[PREPROCESSING.md](PREPROCESSING.md)** - Complete preprocessing guide
- **[WORDLISTS.md](WORDLISTS.md)** - German wordlist documentation
- **[NORMALIZATION.md](NORMALIZATION.md)** - Text normalization methods
- **[METADATA.md](METADATA.md)** - Preprocessing metadata system
