"""
Effect size measures for group comparisons.

Provides standardized effect size calculations:
- Cliff's Delta (non-parametric, -1 to +1)
- Rank-biserial correlation (related to Mann-Whitney U)
- Cohen's d (parametric, standardized mean difference)
- Hedges' g (bias-corrected Cohen's d for small samples)

Effect sizes help interpret practical significance beyond p-values.

Usage:
    from newspaper_explorer.analyze.statistics.effects import cliffs_delta, cohens_d

    # Non-parametric effect size
    delta = cliffs_delta(group1_values, group2_values)

    # Parametric effect size
    d = cohens_d(group1_values, group2_values)
"""

import logging
import math
from typing import Union

import polars as pl

logger = logging.getLogger(__name__)


# Effect size interpretation thresholds (conventional)
EFFECT_THRESHOLDS = {
    "cliffs_delta": {
        "negligible": 0.147,
        "small": 0.33,
        "medium": 0.474,
        # large: > 0.474
    },
    "cohens_d": {
        "negligible": 0.2,
        "small": 0.5,
        "medium": 0.8,
        # large: > 0.8
    },
}


def interpret_effect_size(
    value: float,
    measure: str = "cliffs_delta",
) -> str:
    """
    Interpret effect size magnitude using conventional thresholds.

    Args:
        value: Effect size value (absolute)
        measure: Effect size measure ('cliffs_delta' or 'cohens_d')

    Returns:
        Interpretation string: 'negligible', 'small', 'medium', or 'large'
    """
    thresholds = EFFECT_THRESHOLDS.get(measure, EFFECT_THRESHOLDS["cohens_d"])
    abs_val = abs(value)

    if abs_val < thresholds["negligible"]:
        return "negligible"
    elif abs_val < thresholds["small"]:
        return "small"
    elif abs_val < thresholds["medium"]:
        return "medium"
    else:
        return "large"


def cliffs_delta(
    group1: Union[list, pl.Series],
    group2: Union[list, pl.Series],
) -> float:
    """
    Calculate Cliff's Delta effect size.

    Non-parametric effect size measuring the probability that a randomly
    selected value from group1 is greater than a randomly selected value
    from group2, minus the reverse probability.

    Range: -1 to +1
    - 0: No effect (groups overlap completely)
    - +1: All values in group1 > all values in group2
    - -1: All values in group1 < all values in group2

    Interpretation (absolute value):
    - < 0.147: Negligible
    - < 0.33: Small
    - < 0.474: Medium
    - >= 0.474: Large

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cliff's Delta value (-1 to +1)

    Example:
        >>> delta = cliffs_delta(pre_war_joy, war_joy)
        >>> print(f"δ = {delta:.3f} ({interpret_effect_size(delta, 'cliffs_delta')})")
    """
    # Convert to lists if needed
    if isinstance(group1, pl.Series):
        group1 = group1.drop_nulls().to_list()
    if isinstance(group2, pl.Series):
        group2 = group2.drop_nulls().to_list()

    n1, n2 = len(group1), len(group2)

    if n1 == 0 or n2 == 0:
        return 0.0

    # Count dominance pairs
    # More efficient O(n log n) algorithm using sorting
    a_sorted = sorted(group1)
    b_sorted = sorted(group2)

    # Count how many times a > b (wins) and a == b (ties)
    wins = 0
    ties = 0
    j = 0  # Pointer for b_sorted

    for a_val in a_sorted:
        # Move j to first element in b >= a_val
        while j < n2 and b_sorted[j] < a_val:
            j += 1
        wins += j  # All elements before j are < a_val

        # Count ties (elements equal to a_val)
        k = j
        while k < n2 and b_sorted[k] == a_val:
            k += 1
        ties += k - j

    total_pairs = n1 * n2
    losses = total_pairs - wins - ties

    delta = (wins - losses) / total_pairs
    return delta


def rank_biserial_correlation(
    group1: Union[list, pl.Series],
    group2: Union[list, pl.Series],
    u_statistic: float = None,
) -> float:
    """
    Calculate rank-biserial correlation from Mann-Whitney U.

    Alternative formulation of effect size for Mann-Whitney U test.
    Mathematically equivalent to Cliff's Delta when computed directly.

    Range: -1 to +1 (same interpretation as Cliff's Delta)

    Args:
        group1: First group values
        group2: Second group values
        u_statistic: Pre-computed U statistic (optional, for efficiency)

    Returns:
        Rank-biserial correlation value
    """
    if isinstance(group1, pl.Series):
        group1 = group1.drop_nulls().to_list()
    if isinstance(group2, pl.Series):
        group2 = group2.drop_nulls().to_list()

    n1, n2 = len(group1), len(group2)

    if n1 == 0 or n2 == 0:
        return 0.0

    if u_statistic is None:
        from scipy.stats import mannwhitneyu

        u_statistic = mannwhitneyu(group1, group2, alternative="two-sided").statistic

    # r = 1 - (2U)/(n1*n2)
    r = 1 - (2 * u_statistic) / (n1 * n2)
    return r


def cohens_d(
    group1: Union[list, pl.Series],
    group2: Union[list, pl.Series],
    pooled: bool = True,
) -> float:
    """
    Calculate Cohen's d effect size.

    Parametric effect size measuring standardized mean difference.
    Assumes approximately normal distributions.

    Range: Unbounded, but typically -3 to +3
    - 0: No effect
    - Positive: group1 mean > group2 mean
    - Negative: group1 mean < group2 mean

    Interpretation (absolute value):
    - < 0.2: Negligible
    - < 0.5: Small
    - < 0.8: Medium
    - >= 0.8: Large

    Args:
        group1: First group values
        group2: Second group values
        pooled: If True, use pooled standard deviation (recommended).
                If False, use group2's std (original Cohen's d).

    Returns:
        Cohen's d value

    Example:
        >>> d = cohens_d(pre_war_anger, war_anger)
        >>> print(f"d = {d:.3f} ({interpret_effect_size(d, 'cohens_d')})")
    """
    if isinstance(group1, pl.Series):
        group1 = group1.drop_nulls().to_list()
    if isinstance(group2, pl.Series):
        group2 = group2.drop_nulls().to_list()

    n1, n2 = len(group1), len(group2)

    if n1 == 0 or n2 == 0:
        return 0.0

    s1 = pl.Series(group1)
    s2 = pl.Series(group2)

    mean1 = s1.mean()
    mean2 = s2.mean()
    std1 = s1.std()
    std2 = s2.std()

    if pooled:
        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        d = (mean1 - mean2) / pooled_std
    else:
        # Original Cohen's d (using control group std)
        if std2 == 0:
            return 0.0
        d = (mean1 - mean2) / std2

    return d


def hedges_g(
    group1: Union[list, pl.Series],
    group2: Union[list, pl.Series],
) -> float:
    """
    Calculate Hedges' g effect size.

    Bias-corrected version of Cohen's d for small samples (n < 20).
    Applies correction factor J to reduce upward bias.

    Same interpretation as Cohen's d.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Hedges' g value (bias-corrected Cohen's d)
    """
    if isinstance(group1, pl.Series):
        group1 = group1.drop_nulls().to_list()
    if isinstance(group2, pl.Series):
        group2 = group2.drop_nulls().to_list()

    n1, n2 = len(group1), len(group2)
    d = cohens_d(group1, group2, pooled=True)

    # Correction factor J (approximation)
    df = n1 + n2 - 2
    if df <= 0:
        return d

    # J ≈ 1 - 3/(4*df - 1)
    j = 1 - 3 / (4 * df - 1)

    return d * j


def compute_effect_size(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    groups: tuple[str, str],
    method: str = "cliffs_delta",
) -> dict:
    """
    Compute effect size between two groups in a DataFrame.

    Convenience function for computing effect sizes from DataFrames.

    Args:
        df: Input DataFrame
        value_col: Column containing values
        group_col: Column containing group labels
        groups: Tuple of (group1, group2) to compare
        method: Effect size method ('cliffs_delta', 'cohens_d', 'hedges_g')

    Returns:
        Dictionary with effect size value, interpretation, and group stats.

    Example:
        >>> result = compute_effect_size(df, "Joy_prob", "era", ("pre_war", "war"))
        >>> print(f"{result['method']}: {result['value']:.3f} ({result['interpretation']})")
    """
    group1_name, group2_name = groups

    vals1 = df.filter(pl.col(group_col) == group1_name).select(value_col).to_series().drop_nulls()
    vals2 = df.filter(pl.col(group_col) == group2_name).select(value_col).to_series().drop_nulls()

    # Compute effect size
    if method == "cliffs_delta":
        value = cliffs_delta(vals1, vals2)
        interpretation = interpret_effect_size(value, "cliffs_delta")
    elif method == "cohens_d":
        value = cohens_d(vals1, vals2)
        interpretation = interpret_effect_size(value, "cohens_d")
    elif method == "hedges_g":
        value = hedges_g(vals1, vals2)
        interpretation = interpret_effect_size(value, "cohens_d")  # Same thresholds as Cohen's d
    elif method == "rank_biserial":
        value = rank_biserial_correlation(vals1, vals2)
        interpretation = interpret_effect_size(value, "cliffs_delta")  # Same thresholds
    else:
        msg = f"Unknown method: {method}. Use 'cliffs_delta', 'cohens_d', 'hedges_g', or 'rank_biserial'"
        raise ValueError(msg)

    return {
        "method": method,
        "value": value,
        "interpretation": interpretation,
        "group1": group1_name,
        "group2": group2_name,
        "n1": len(vals1),
        "n2": len(vals2),
        "mean1": float(vals1.mean()) if len(vals1) > 0 else None,
        "mean2": float(vals2.mean()) if len(vals2) > 0 else None,
    }
