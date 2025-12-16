"""
Quality validation and metrics for OCR text.

Provides comprehensive quality assessment for historical German newspaper OCR text,
implementing the validation approach from normalize.md research.

Quality metrics include:
- Character error rate (CER) estimation
- Out-of-vocabulary (OOV) rate
- Character-to-token ratio
- Proper noun density
- Overall quality scoring
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

# Quality thresholds based on normalize.md research
OOV_RATE_GOOD_THRESHOLD = 0.05  # <5% OOV indicates good quality
OOV_RATE_REVIEW_THRESHOLD = 0.15  # 5-15% OOV needs review
CHAR_TOKEN_RATIO_GOOD_THRESHOLD = 7.0  # ≤7 chars/token is good
CHAR_TOKEN_RATIO_REVIEW_THRESHOLD = 8.0  # 7-8 chars/token needs review


def calculate_quality_metrics(
    df: pl.DataFrame,
    input_column: str = "text",
    german_wordlist_path: Optional[str] = None,
    output_column_prefix: str = "quality_",
) -> pl.DataFrame:
    """
    Calculate comprehensive quality metrics for OCR text.

    Based on normalize.md Stage 6 validation approach. Calculates multiple
    quality indicators to assess OCR accuracy and text cleanliness.

    Quality indicators:
    - char_token_ratio: Average characters per token (clean German: 5-7)
    - oov_rate: Out-of-vocabulary rate (good: <5%, review: 5-15%, poor: >15%)
    - proper_noun_density: Ratio of proper nouns to total tokens
    - quality_score: Overall assessment (good/review/poor)

    Args:
        df: Input DataFrame
        input_column: Column to analyze (default: "text")
        german_wordlist_path: Path to German word list file (optional)
        output_column_prefix: Prefix for output metric columns (default: "quality_")

    Returns:
        DataFrame with added quality metric columns:
        - {prefix}char_token_ratio: float
        - {prefix}oov_rate: float
        - {prefix}proper_noun_density: float
        - {prefix}score: str (good/review/poor)

    Example:
        >>> # Calculate quality metrics
        >>> df = calculate_quality_metrics(df)
        >>> # Access metrics
        >>> df.select(["text", "quality_char_token_ratio", "quality_score"])

    Note:
        From normalize.md research:
        - Clean German: 5-7 chars/token, <5% OOV
        - Review needed: 7-8 chars/token, 5-15% OOV
        - Poor quality: >8 chars/token, >15% OOV
    """
    logger.info(f"Calculating quality metrics: {input_column}")

    # Load German wordlist if provided
    vocab = None
    if german_wordlist_path:
        try:
            with Path(german_wordlist_path).open("r", encoding="utf-8") as f:
                vocab = {line.strip().lower() for line in f if line.strip()}
            logger.info(f"Loaded {len(vocab):,} words from German wordlist")
        except FileNotFoundError:
            logger.warning(f"German wordlist not found: {german_wordlist_path}")
            logger.warning("OOV rate will not be calculated")

    def calculate_metrics(text: str) -> dict[str, float]:
        """Calculate all quality metrics for a single text."""
        if not text or not text.strip():
            return {
                "char_token_ratio": 0.0,
                "oov_rate": 0.0,
                "proper_noun_density": 0.0,
                "quality": 0.0,
            }

        tokens = text.split()
        if not tokens:
            return {
                "char_token_ratio": 0.0,
                "oov_rate": 0.0,
                "proper_noun_density": 0.0,
                "quality": "poor",
            }

        # Metric 1: Character-to-token ratio
        char_count = len(text)
        token_count = len(tokens)
        char_token_ratio = char_count / token_count

        # Metric 2: Out-of-vocabulary rate
        oov_rate = 0.0
        if vocab:
            oov_count = sum(1 for t in tokens if t.lower() not in vocab)
            oov_rate = oov_count / token_count

        # Metric 3: Proper noun density
        # German nouns are capitalized, but exclude common articles
        common_capitalized = {"Der", "Die", "Das", "Ein", "Eine"}
        proper_nouns = sum(
            1 for t in tokens if t and t[0].isupper() and t not in common_capitalized
        )
        proper_noun_density = proper_nouns / token_count

        # Metric 4: Overall quality score
        # Based on normalize.md thresholds
        if vocab:
            # Use OOV rate as primary indicator when available
            if oov_rate < OOV_RATE_GOOD_THRESHOLD:
                quality = "good"
            elif oov_rate < OOV_RATE_REVIEW_THRESHOLD:
                quality = "review"
            else:
                quality = "poor"
        # Fall back to char/token ratio
        elif char_token_ratio <= CHAR_TOKEN_RATIO_GOOD_THRESHOLD:
            quality = "good"
        elif char_token_ratio <= CHAR_TOKEN_RATIO_REVIEW_THRESHOLD:
            quality = "review"
        else:
            quality = "poor"

        return {
            "char_token_ratio": char_token_ratio,
            "oov_rate": oov_rate,
            "proper_noun_density": proper_noun_density,
            "quality": quality,
        }

    # Apply metrics calculation to each row
    texts = df[input_column].to_list()
    metrics = [calculate_metrics(text) for text in texts]

    # Add metric columns
    df = df.with_columns(
        [
            pl.Series(
                f"{output_column_prefix}char_token_ratio",
                [m["char_token_ratio"] for m in metrics],
            ),
            pl.Series(f"{output_column_prefix}oov_rate", [m["oov_rate"] for m in metrics]),
            pl.Series(
                f"{output_column_prefix}proper_noun_density",
                [m["proper_noun_density"] for m in metrics],
            ),
            pl.Series(f"{output_column_prefix}score", [m["quality"] for m in metrics]),
        ]
    )

    # Log summary statistics
    avg_ratio = sum(m["char_token_ratio"] for m in metrics) / len(metrics)
    avg_oov = sum(m["oov_rate"] for m in metrics) / len(metrics)
    quality_counts = {"good": 0, "review": 0, "poor": 0}
    for m in metrics:
        quality_counts[m["quality"]] += 1

    logger.info(f"Quality metrics calculated for {len(df):,} rows")
    logger.info(f"  Average char/token ratio: {avg_ratio:.2f}")
    if vocab:
        logger.info(f"  Average OOV rate: {avg_oov:.1%}")
    logger.info(
        f"  Quality distribution: "
        f"{quality_counts['good']} good, "
        f"{quality_counts['review']} review, "
        f"{quality_counts['poor']} poor"
    )

    return df


def filter_by_quality_score(
    df: pl.DataFrame,
    quality_column: str = "quality_score",
    min_quality: str = "review",
) -> pl.DataFrame:
    """
    Filter DataFrame by quality score threshold.

    Removes rows below the specified quality threshold.

    Args:
        df: Input DataFrame with quality scores
        quality_column: Column containing quality scores (default: "quality_score")
        min_quality: Minimum quality to keep (default: "review")
                    Options: "good", "review", "poor"

    Returns:
        DataFrame with low-quality rows filtered out

    Example:
        >>> # Keep only good and review quality
        >>> df = filter_by_quality_score(df, min_quality="review")
        >>>
        >>> # Keep only good quality
        >>> df = filter_by_quality_score(df, min_quality="good")
    """
    logger.info(f"Filtering by quality score: keeping >= {min_quality}")

    original_count = len(df)

    if min_quality == "good":
        # Keep only good
        df = df.filter(pl.col(quality_column) == "good")
    elif min_quality == "review":
        # Keep good and review
        df = df.filter(pl.col(quality_column).is_in(["good", "review"]))
    elif min_quality == "poor":
        # Keep all (no filtering)
        pass
    else:
        raise ValueError(f"Invalid min_quality: {min_quality}. Use 'good', 'review', or 'poor'")

    filtered_count = original_count - len(df)
    logger.info(
        f"Filtered out {filtered_count:,} rows ({filtered_count / original_count * 100:.1f}%)"
    )
    logger.info(f"Remaining: {len(df):,} rows")

    return df


def summarize_quality(
    df: pl.DataFrame,
    quality_column_prefix: str = "quality_",
) -> dict[str, float]:
    """
    Generate summary statistics for quality metrics.

    Args:
        df: DataFrame with quality metrics
        quality_column_prefix: Prefix for quality columns (default: "quality_")

    Returns:
        Dictionary with summary statistics

    Example:
        >>> summary = summarize_quality(df)
        >>> print(f"Average OOV rate: {summary['avg_oov_rate']:.1%}")
        >>> print(f"Good quality: {summary['pct_good']:.1%}")
    """
    metrics = {}

    # Character-to-token ratio
    ratio_col = f"{quality_column_prefix}char_token_ratio"
    if ratio_col in df.columns:
        metrics["avg_char_token_ratio"] = df[ratio_col].mean()
        metrics["median_char_token_ratio"] = df[ratio_col].median()

    # OOV rate
    oov_col = f"{quality_column_prefix}oov_rate"
    if oov_col in df.columns:
        metrics["avg_oov_rate"] = df[oov_col].mean()
        metrics["median_oov_rate"] = df[oov_col].median()

    # Proper noun density
    noun_col = f"{quality_column_prefix}proper_noun_density"
    if noun_col in df.columns:
        metrics["avg_proper_noun_density"] = df[noun_col].mean()

    # Quality score distribution
    score_col = f"{quality_column_prefix}score"
    if score_col in df.columns:
        total = len(df)
        good_count = len(df.filter(pl.col(score_col) == "good"))
        review_count = len(df.filter(pl.col(score_col) == "review"))
        poor_count = len(df.filter(pl.col(score_col) == "poor"))

        metrics["pct_good"] = good_count / total if total > 0 else 0
        metrics["pct_review"] = review_count / total if total > 0 else 0
        metrics["pct_poor"] = poor_count / total if total > 0 else 0

    return metrics
